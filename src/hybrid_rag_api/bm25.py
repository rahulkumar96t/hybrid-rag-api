"""A small Okapi BM25 implementation over an in-memory inverted index."""

from __future__ import annotations

import heapq
import math
from collections import Counter
from collections.abc import Iterable, Sequence


class BM25Index:
    """Okapi BM25 with the non-negative ("Lucene") IDF variant.

    ``idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))`` stays positive even for
    terms that occur in most documents, so a matching term never lowers a score.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0 or not 0.0 <= b <= 1.0:
            raise ValueError("expected k1 >= 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._doc_lengths: list[int] = []
        self._total_length = 0

    @classmethod
    def build(cls, corpus: Iterable[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> BM25Index:
        index = cls(k1=k1, b=b)
        for tokens in corpus:
            index.add(tokens)
        return index

    def __len__(self) -> int:
        return len(self._doc_lengths)

    @property
    def avg_doc_length(self) -> float:
        return self._total_length / len(self._doc_lengths) if self._doc_lengths else 0.0

    def add(self, tokens: Sequence[str]) -> int:
        """Index one document and return its position."""
        doc_id = len(self._doc_lengths)
        for term, tf in Counter(tokens).items():
            self._postings.setdefault(term, []).append((doc_id, tf))
        self._doc_lengths.append(len(tokens))
        self._total_length += len(tokens)
        return doc_id

    def idf(self, term: str) -> float:
        df = len(self._postings.get(term, ()))
        n = len(self._doc_lengths)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query_tokens: Sequence[str]) -> dict[int, float]:
        """Score every document that shares at least one term with the query."""
        if not self._doc_lengths:
            return {}
        avgdl = self.avg_doc_length or 1.0
        result: dict[int, float] = {}
        for term in dict.fromkeys(query_tokens):  # unique terms, query order kept
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for doc_id, tf in postings:
                norm = self.k1 * (1.0 - self.b + self.b * self._doc_lengths[doc_id] / avgdl)
                result[doc_id] = result.get(doc_id, 0.0) + idf * tf * (self.k1 + 1.0) / (tf + norm)
        return result

    def top_n(self, query_tokens: Sequence[str], n: int) -> list[tuple[int, float]]:
        """Return up to ``n`` ``(doc_position, score)`` pairs, best first.

        Only documents with a positive score are returned; ties break on the lower
        position so results are deterministic.
        """
        if n <= 0:
            return []
        scored = self.scores(query_tokens)
        best = heapq.nsmallest(n, scored.items(), key=lambda item: (-item[1], item[0]))
        return [(doc_id, score) for doc_id, score in best if score > 0.0]
