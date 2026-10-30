"""Pluggable embedders.

``HashingEmbedder`` is deterministic and dependency-free: it hashes word
unigrams, word bigrams and character n-grams into a fixed-size signed vector
("feature hashing"). It captures lexical and sub-word overlap, not meaning, and
exists so the whole pipeline runs offline and in tests. ``FastEmbedEmbedder``
wraps a real sentence-embedding model through the optional ``fastembed`` extra.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache
from itertools import pairwise
from typing import Protocol, runtime_checkable

import numpy as np

from .text import tokenize


@runtime_checkable
class Embedder(Protocol):
    """Turns text into L2-normalised float32 vectors."""

    name: str
    dim: int
    # Cosine similarity at or above which a dense hit counts as evidence that the
    # corpus covers the question (used by the refusal gate).
    min_relevance: float

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


@lru_cache(maxsize=200_000)
def _bucket(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % dim, (1.0 if value >> 63 == 0 else -1.0)


class HashingEmbedder:
    """Signed feature hashing of words, word bigrams and character n-grams."""

    min_relevance = 0.2

    def __init__(self, dim: int = 1024, char_ngram: int = 4) -> None:
        if dim < 16:
            raise ValueError("dim must be at least 16")
        self.dim = dim
        self.char_ngram = char_ngram
        self.name = f"hashing-{dim}-c{char_ngram}"

    def _features(self, text: str) -> Counter[str]:
        tokens = tokenize(text)
        features: Counter[str] = Counter()
        for token in tokens:
            features[f"w:{token}"] += 1
            padded = f"<{token}>"
            if len(padded) > self.char_ngram:
                for i in range(len(padded) - self.char_ngram + 1):
                    features[f"c:{padded[i : i + self.char_ngram]}"] += 1
        for left, right in pairwise(tokens):
            features[f"b:{left} {right}"] += 1
        return features

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        for feature, count in self._features(text).items():
            # Character n-grams are numerous; down-weight them so whole-word
            # matches still dominate.
            weight = 0.5 if feature.startswith("c:") else 1.0
            idx, sign = _bucket(feature, self.dim)
            vec[idx] += sign * weight * (1.0 + math.log(count))
        return vec

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2_normalise(np.vstack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2_normalise(self._vector(text)[None, :])[0]


class FastEmbedEmbedder:
    """Local ONNX sentence embeddings via ``fastembed`` (install the extra)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", min_relevance: float = 0.6):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "the fastembed embedder needs the optional extra: uv sync --extra fastembed"
            ) from exc
        self._model = TextEmbedding(model_name=model_name)
        self.name = f"fastembed:{model_name}"
        self.min_relevance = min_relevance
        self.dim = int(self.embed_query("dimension probe").shape[0])

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2_normalise(np.vstack(list(self._model.passage_embed(list(texts)))))

    def embed_query(self, text: str) -> np.ndarray:
        vector = next(iter(self._model.query_embed(text)))
        return _l2_normalise(np.asarray(vector, dtype=np.float64)[None, :])[0]


def make_embedder(spec: str) -> Embedder:
    """Build an embedder from a spec string.

    ``hashing`` / ``hashing:2048`` or ``fastembed`` / ``fastembed:<model-name>``.
    """
    kind, _, arg = spec.partition(":")
    kind = kind.strip().lower()
    if kind == "hashing":
        return HashingEmbedder(dim=int(arg) if arg else 1024)
    if kind == "fastembed":
        return FastEmbedEmbedder(model_name=arg or "BAAI/bge-small-en-v1.5")
    raise ValueError(f"unknown embedder {spec!r}; expected 'hashing' or 'fastembed[:model]'")
