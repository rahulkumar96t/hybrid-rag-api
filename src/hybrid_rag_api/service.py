"""The RAG pipeline: ingest documents, retrieve evidence, generate cited answers."""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .chunking import chunk_text
from .config import Settings
from .embeddings import Embedder, make_embedder
from .generation import (
    SYSTEM_PROMPT,
    AnswerModel,
    ClaudeAnswerModel,
    StreamEnd,
    build_user_prompt,
    could_become_refusal,
    is_refusal_text,
    parse_citations,
)
from .loaders import load_bytes
from .retrieval import HybridIndex, Mode, RetrievedChunk
from .store import ChunkRecord, DocumentRecord, IndexStore, index_text

RefusalReason = Literal[
    "no_relevant_context",  # retrieval found nothing that matches the question
    "insufficient_context",  # the model said the passages do not answer it
    "uncited_answer",  # the model answered without citing any retrieved passage
    "model_refusal",  # the API declined the request (stop_reason == "refusal")
]

REFUSAL_MESSAGES: dict[str, str] = {
    "no_relevant_context": "I could not find anything in the indexed documents about this.",
    "insufficient_context": "The indexed documents do not contain enough information to answer.",
    "uncited_answer": "I could not produce an answer that is supported by the indexed documents.",
    "model_refusal": "The model declined to answer this request.",
}


class LLMUnavailableError(RuntimeError):
    """Generation was requested but no answer model is configured."""


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    source: str
    status: Literal["added", "updated", "unchanged"]
    chunks: int


@dataclass
class Answer:
    question: str
    answer: str
    refused: bool
    refusal_reason: RefusalReason | None = None
    citations: list[ChunkRecord] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    model: str | None = None
    raw_answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "citations": [
                {
                    "id": c.id,
                    "doc_id": c.doc_id,
                    "source": c.source,
                    "section": c.section,
                    "page": c.page,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                }
                for c in self.citations
            ],
            "invalid_citations": self.invalid_citations,
            "retrieved": [r.to_dict(snippet_chars=300) for r in self.retrieved],
            "model": self.model,
            "raw_answer": self.raw_answer,
        }


class RagService:
    def __init__(
        self,
        index: HybridIndex,
        llm: AnswerModel | None,
        settings: Settings | None = None,
    ) -> None:
        self.index = index
        self.llm = llm
        self.settings = settings or Settings()
        self._write_lock = threading.Lock()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
        llm: AnswerModel | None = None,
        with_llm: bool = True,
    ) -> RagService:
        embedder = embedder or make_embedder(settings.embedder)
        store = IndexStore(settings.index_path, embedder.name, embedder.dim)
        index = HybridIndex(store, embedder, rrf_k=settings.rrf_k)
        if llm is None and with_llm:
            llm = ClaudeAnswerModel(
                settings.model,
                effort=settings.effort,
                fallbacks=settings.fallbacks,
                max_tokens=settings.max_tokens,
            )
        return cls(index, llm, settings)

    # ------------------------------------------------------------------ ingest
    def ingest_bytes(self, filename: str, data: bytes) -> IngestResult:
        doc = load_bytes(filename, data)
        with self._write_lock:
            existing = self.index.store.get_document(doc.doc_id)
            if existing and existing.sha256 == doc.sha256:
                return IngestResult(doc.doc_id, doc.source, "unchanged", existing.n_chunks)
            chunks = chunk_text(
                doc.text, doc.kind, self.settings.chunk_words, self.settings.overlap_words
            )
            texts = [index_text(doc.title, c.section, c.text) for c in chunks]
            vectors = self.index.embedder.embed_documents(texts)
            self.index.store.upsert_document(doc, chunks, vectors)
            self.index.refresh()
        status = "updated" if existing else "added"
        return IngestResult(doc.doc_id, doc.source, status, len(chunks))

    def ingest_path(self, path: str | Path) -> IngestResult:
        doc_path = Path(path)
        return self.ingest_bytes(doc_path.name, doc_path.read_bytes())

    def delete_document(self, doc_id: str) -> bool:
        with self._write_lock:
            deleted = self.index.store.delete_document(doc_id)
            if deleted:
                self.index.refresh()
        return deleted

    def documents(self) -> list[DocumentRecord]:
        return self.index.store.list_documents()

    # --------------------------------------------------------------- retrieval
    def search(
        self, query: str, k: int | None = None, mode: Mode = "hybrid"
    ) -> list[RetrievedChunk]:
        return self.index.search(
            query, k=k or self.settings.top_k, candidates=self.settings.candidates, mode=mode
        )

    def _retrieve(self, question: str, k: int | None) -> tuple[list[RetrievedChunk], bool]:
        results = self.search(question, k)
        return results, self.index.has_support(results, self.settings.min_dense_score)

    def _require_llm(self) -> AnswerModel:
        if self.llm is None:
            raise LLMUnavailableError("no answer model configured (set ANTHROPIC_API_KEY)")
        return self.llm

    # -------------------------------------------------------------- answering
    def _refusal(
        self, question: str, reason: RefusalReason, retrieved: list[RetrievedChunk], **extra: Any
    ) -> Answer:
        return Answer(
            question=question,
            answer=REFUSAL_MESSAGES[reason],
            refused=True,
            refusal_reason=reason,
            retrieved=retrieved,
            **extra,
        )

    def _finalise(
        self,
        question: str,
        text: str,
        stop_reason: str | None,
        model: str,
        retrieved: list[RetrievedChunk],
    ) -> Answer:
        if stop_reason == "refusal":
            return self._refusal(question, "model_refusal", retrieved, model=model)
        if is_refusal_text(text):
            return self._refusal(
                question, "insufficient_context", retrieved, model=model, raw_answer=text
            )
        by_id = {r.chunk.id: r.chunk for r in retrieved}
        report = parse_citations(text, by_id)
        if self.settings.require_citations and not report.valid:
            return self._refusal(
                question,
                "uncited_answer",
                retrieved,
                model=model,
                raw_answer=text,
                invalid_citations=report.invalid,
            )
        return Answer(
            question=question,
            answer=text.strip(),
            refused=False,
            citations=[by_id[cid] for cid in report.valid],
            invalid_citations=report.invalid,
            retrieved=retrieved,
            model=model,
        )

    async def answer(self, question: str, k: int | None = None) -> Answer:
        llm = self._require_llm()
        retrieved, supported = self._retrieve(question, k)
        if not supported:
            return self._refusal(question, "no_relevant_context", retrieved)
        generation = await llm.generate(SYSTEM_PROMPT, build_user_prompt(question, retrieved))
        return self._finalise(
            question, generation.text, generation.stop_reason, generation.model, retrieved
        )

    async def answer_stream(
        self, question: str, k: int | None = None
    ) -> AsyncIterator[tuple[str, Any]]:
        """Yield ``("retrieval", hits)``, ``("token", text)`` events, then ``("done", answer)``.

        Text is held back only while it could still be the refusal marker, so a
        refusal never leaks ``INSUFFICIENT_CONTEXT`` to the client. Citation
        checks need the full answer, so the final verdict arrives in ``done``.
        """
        llm = self._require_llm()
        retrieved, supported = self._retrieve(question, k)
        yield "retrieval", [r.to_dict(snippet_chars=300) for r in retrieved]
        if not supported:
            yield "done", self._refusal(question, "no_relevant_context", retrieved)
            return

        parts: list[str] = []
        pending = ""
        state: Literal["probe", "emit", "suppress"] = "probe"
        end = StreamEnd(stop_reason=None, model=llm.name)
        async for event in llm.stream(SYSTEM_PROMPT, build_user_prompt(question, retrieved)):
            if isinstance(event, StreamEnd):
                end = event
                continue
            parts.append(event.text)
            if state == "emit":
                yield "token", event.text
            elif state == "probe":
                pending += event.text
                if could_become_refusal(pending):
                    continue
                if is_refusal_text(pending):
                    state = "suppress"  # stay silent for the rest of a refusal
                else:
                    state = "emit"
                    yield "token", pending
        if state == "probe" and pending:
            yield "token", pending  # the whole answer was shorter than the marker
        text = "".join(parts)
        yield "done", self._finalise(question, text, end.stop_reason, end.model, retrieved)
