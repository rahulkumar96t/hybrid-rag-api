"""Reciprocal rank fusion (Cormack, Clarke & Buettcher, SIGIR 2009)."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[T]],
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[T, float]]:
    """Fuse several ranked lists into one.

    Each item scores ``sum_i w_i / (k + rank_i)`` over the lists it appears in
    (ranks start at 1). RRF only looks at ranks, so BM25 scores and cosine
    similarities never need to be calibrated against each other. An item that
    appears twice in one list counts once, at its best rank. Ties keep the order
    in which items were first seen, which makes the output deterministic.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[T, float] = {}
    for name, ranking in rankings.items():
        weight = 1.0 if weights is None else weights.get(name, 1.0)
        seen: set[T] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda pair: -pair[1])
