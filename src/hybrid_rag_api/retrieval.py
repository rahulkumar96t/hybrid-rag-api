"""Hybrid retrieval: BM25 and dense similarity fused with reciprocal rank fusion."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from .bm25 import BM25Index
from .embeddings import Embedder
from .fusion import reciprocal_rank_fusion
from .store import ChunkRecord, IndexStore
from .text import tokenize

Mode = Literal["hybrid", "bm25", "dense"]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: ChunkRecord
    score: float
    bm25_rank: int | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None

    def to_dict(self, snippet_chars: int | None = None) -> dict:
        data = asdict(self)
        chunk = data.pop("chunk")
        if snippet_chars is not None and len(chunk["text"]) > snippet_chars:
            chunk["text"] = chunk["text"][:snippet_chars].rstrip() + "..."
        return {**chunk, **data}


@dataclass(frozen=True)
class _Snapshot:
    chunks: tuple[ChunkRecord, ...]
    vectors: np.ndarray
    bm25: BM25Index


class HybridIndex:
    """In-memory search structures over the chunks held by an :class:`IndexStore`.

    The store is the source of truth. After every write the in-memory snapshot
    (chunk list, vector matrix, BM25 postings) is rebuilt and swapped in
    atomically, so concurrent searches always see a consistent view.
    """

    def __init__(self, store: IndexStore, embedder: Embedder, rrf_k: int = 60) -> None:
        self.store = store
        self.embedder = embedder
        self.rrf_k = rrf_k
        self._lock = threading.Lock()
        self._snapshot = _Snapshot((), np.zeros((0, embedder.dim), dtype=np.float32), BM25Index())
        self.refresh()

    def refresh(self) -> None:
        records, vectors = self.store.load_chunks()
        bm25 = BM25Index.build(tokenize(r.index_text) for r in records)
        with self._lock:
            self._snapshot = _Snapshot(tuple(records), vectors, bm25)

    def __len__(self) -> int:
        return len(self._snapshot.chunks)

    def search(
        self, query: str, k: int = 5, candidates: int = 30, mode: Mode = "hybrid"
    ) -> list[RetrievedChunk]:
        """Return the ``k`` best chunks for ``query``.

        Each retriever contributes its top ``candidates``; RRF fuses the two
        rankings. Per-retriever ranks and raw scores are kept on every result so
        callers (and the refusal gate) can see why a chunk was returned.
        """
        snap = self._snapshot
        if not snap.chunks or k <= 0:
            return []
        candidates = max(candidates, k)

        lexical: list[tuple[int, float]] = []
        if mode in ("hybrid", "bm25"):
            lexical = snap.bm25.top_n(tokenize(query), candidates)

        dense: list[tuple[int, float]] = []
        if mode in ("hybrid", "dense"):
            sims = snap.vectors @ self.embedder.embed_query(query)
            n = min(candidates, len(sims))
            top = np.argpartition(-sims, n - 1)[:n]
            top = top[np.lexsort((top, -sims[top]))]  # by similarity, then position
            dense = [(int(i), float(sims[i])) for i in top]

        lexical_info = {i: (rank, score) for rank, (i, score) in enumerate(lexical, start=1)}
        dense_info = {i: (rank, score) for rank, (i, score) in enumerate(dense, start=1)}
        fused = reciprocal_rank_fusion(
            {"bm25": [i for i, _ in lexical], "dense": [i for i, _ in dense]}, k=self.rrf_k
        )
        results = []
        for position, score in fused[:k]:
            b_rank, b_score = lexical_info.get(position, (None, None))
            d_rank, d_score = dense_info.get(position, (None, None))
            results.append(
                RetrievedChunk(
                    chunk=snap.chunks[position],
                    score=score,
                    bm25_rank=b_rank,
                    bm25_score=b_score,
                    dense_rank=d_rank,
                    dense_score=d_score,
                )
            )
        return results

    def has_support(self, results: list[RetrievedChunk], min_dense: float | None = None) -> bool:
        """Cheap pre-generation check that retrieval found *some* evidence.

        True if any result matched a query term (BM25 > 0) or reached the
        embedder's relevance threshold. This only filters out questions the
        corpus clearly does not cover; the model makes the finer call.
        """
        threshold = self.embedder.min_relevance if min_dense is None else min_dense
        return any(
            (r.bm25_score or 0.0) > 0.0 or (r.dense_score or -1.0) >= threshold for r in results
        )
