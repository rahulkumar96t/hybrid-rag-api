"""Offline retrieval evaluation: recall@k, MRR and the refusal gate.

A dataset is JSONL. Each line has a ``question`` and either a list of
``relevant`` targets (``{"doc": "<doc id>", "section": "<heading substring>"}``)
or ``"answerable": false`` for questions the corpus does not cover. Relevance
is judged at the section level, so the labels stay valid when chunk sizes change.

* recall@k - share of a question's relevant targets that have at least one
  chunk in the top k, averaged over answerable questions. With one target per
  question this is the hit rate.
* MRR - mean of 1 / rank of the first relevant chunk (0 if none in the top
  ``max(ks)``).
* gate - how often the pre-generation refusal gate would fire, for unanswerable
  questions (want: high) and answerable ones (want: zero).

No LLM is involved, so the numbers are cheap to reproduce and deterministic for
the hashing embedder.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import chunk_text
from .embeddings import Embedder
from .loaders import discover, load_path
from .retrieval import HybridIndex, Mode
from .store import ChunkRecord, IndexStore, index_text


@dataclass(frozen=True)
class Target:
    doc: str
    section: str | None = None

    def matches(self, chunk: ChunkRecord) -> bool:
        if chunk.doc_id != self.doc:
            return False
        return self.section is None or self.section.lower() in chunk.section.lower()


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    targets: tuple[Target, ...]

    @property
    def answerable(self) -> bool:
        return bool(self.targets)


@dataclass
class EvalReport:
    mode: str
    embedder: str
    ks: tuple[int, ...]
    recall: dict[int, float]
    mrr: float
    answerable: int
    unanswerable: int
    gated_unanswerable: int
    gated_answerable: int
    misses: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "embedder": self.embedder,
            "recall": {f"@{k}": round(v, 4) for k, v in self.recall.items()},
            "mrr": round(self.mrr, 4),
            "answerable": self.answerable,
            "unanswerable": self.unanswerable,
            "gated_unanswerable": self.gated_unanswerable,
            "gated_answerable": self.gated_answerable,
            "misses": self.misses,
        }


def load_dataset(path: str | Path) -> list[EvalQuestion]:
    questions = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("answerable", True) and not row.get("relevant"):
            raise ValueError(f"{path}:{line_no}: answerable question without 'relevant' targets")
        targets = tuple(Target(t["doc"], t.get("section")) for t in row.get("relevant", []))
        questions.append(EvalQuestion(row.get("id", f"q{line_no}"), row["question"], targets))
    return questions


def evaluate(
    index: HybridIndex,
    questions: Sequence[EvalQuestion],
    ks: Sequence[int] = (1, 3, 5),
    mode: Mode = "hybrid",
    candidates: int = 30,
) -> EvalReport:
    ks = tuple(sorted(set(ks)))
    max_k = ks[-1]
    hits = dict.fromkeys(ks, 0.0)
    reciprocal_ranks = 0.0
    answerable = unanswerable = gated_unanswerable = gated_answerable = 0
    misses: list[str] = []
    started = time.perf_counter()
    for q in questions:
        results = index.search(q.question, k=max_k, candidates=candidates, mode=mode)
        gated = not index.has_support(results)
        if not q.answerable:
            unanswerable += 1
            gated_unanswerable += gated
            continue
        answerable += 1
        gated_answerable += gated
        first_rank = next(
            (
                rank
                for rank, r in enumerate(results, 1)
                if any(t.matches(r.chunk) for t in q.targets)
            ),
            None,
        )
        reciprocal_ranks += 1.0 / first_rank if first_rank else 0.0
        if first_rank is None:
            misses.append(q.id)
        for k in ks:
            top = results[:k]
            found = sum(any(t.matches(r.chunk) for r in top) for t in q.targets)
            hits[k] += found / len(q.targets)
    denominator = max(answerable, 1)
    return EvalReport(
        mode=mode,
        embedder=index.embedder.name,
        ks=ks,
        recall={k: hits[k] / denominator for k in ks},
        mrr=reciprocal_ranks / denominator,
        answerable=answerable,
        unanswerable=unanswerable,
        gated_unanswerable=gated_unanswerable,
        gated_answerable=gated_answerable,
        misses=misses,
        seconds=time.perf_counter() - started,
    )


def build_index(
    corpus: str | Path, embedder: Embedder, chunk_words: int = 200, overlap_words: int = 40
) -> HybridIndex:
    """Index a corpus directory into a throwaway in-memory store."""
    store = IndexStore(":memory:", embedder.name, embedder.dim)
    for path in discover([corpus]):
        doc = load_path(path)
        chunks = chunk_text(doc.text, doc.kind, chunk_words, overlap_words)
        vectors = embedder.embed_documents(
            [index_text(doc.title, c.section, c.text) for c in chunks]
        )
        store.upsert_document(doc, chunks, vectors)
    return HybridIndex(store, embedder)


def format_markdown(reports: Sequence[EvalReport]) -> str:
    if not reports:
        return ""
    ks = reports[0].ks
    header = [
        "mode",
        "embedder",
        *[f"recall@{k}" for k in ks],
        "MRR",
        "gate (unanswerable)",
        "gate (answerable)",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in reports:
        row = [
            r.mode,
            r.embedder,
            *[f"{r.recall[k]:.3f}" for k in ks],
            f"{r.mrr:.3f}",
            f"{r.gated_unanswerable}/{r.unanswerable}",
            f"{r.gated_answerable}/{r.answerable}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
