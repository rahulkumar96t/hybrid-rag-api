"""Runtime settings, read from environment variables with the ``RAG_`` prefix."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "claude-sonnet-5-5"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional(name: str, default: str | None) -> str | None:
    """Read a string setting where ``none``/``off`` disables the feature."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip()
    return None if value.lower() in {"none", "off", "false", "0"} else value


@dataclass(frozen=True)
class Settings:
    index_path: Path = Path("data/index.sqlite")
    embedder: str = "hashing"
    # Generation (Claude)
    model: str = DEFAULT_MODEL
    effort: str | None = "low"
    fallbacks: str | None = "default"
    max_tokens: int = 16000
    # Chunking, measured in whitespace-separated words
    chunk_words: int = 200
    overlap_words: int = 40
    # Retrieval
    top_k: int = 5
    candidates: int = 30
    rrf_k: int = 60
    min_dense_score: float | None = None
    require_citations: bool = True
    max_upload_bytes: int = 10 * 1024 * 1024

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            index_path=Path(os.environ.get("RAG_INDEX_PATH", "data/index.sqlite")),
            embedder=os.environ.get("RAG_EMBEDDER", "hashing"),
            model=os.environ.get("RAG_MODEL", DEFAULT_MODEL),
            effort=_env_optional("RAG_EFFORT", "low"),
            fallbacks=_env_optional("RAG_FALLBACKS", "default"),
            max_tokens=_env_int("RAG_MAX_TOKENS", 16000),
            chunk_words=_env_int("RAG_CHUNK_WORDS", 200),
            overlap_words=_env_int("RAG_OVERLAP_WORDS", 40),
            top_k=_env_int("RAG_TOP_K", 5),
            candidates=_env_int("RAG_CANDIDATES", 30),
            rrf_k=_env_int("RAG_RRF_K", 60),
            min_dense_score=_env_float("RAG_MIN_DENSE_SCORE", None),
            require_citations=_env_bool("RAG_REQUIRE_CITATIONS", True),
            max_upload_bytes=_env_int("RAG_MAX_UPLOAD_BYTES", 10 * 1024 * 1024),
        )
