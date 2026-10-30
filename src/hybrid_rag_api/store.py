"""Persistent index storage in a single SQLite file.

Chunk text, metadata and the float32 embedding (as a BLOB) live in one row, so
an ingest or delete is one transaction and the index can never hold vectors
for chunks that no longer exist. The embedder name and dimension are recorded
in ``meta``; opening an index with a different embedder is refused instead of
silently mixing vector spaces.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .loaders import LoadedDocument

SCHEMA_VERSION = "1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    kind TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    n_chunks INTEGER NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    section TEXT NOT NULL,
    text TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    page INTEGER,
    vector BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
"""


class IndexMismatchError(RuntimeError):
    """The index on disk was built with a different embedder."""


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    source: str
    title: str
    kind: str
    sha256: str
    n_chunks: int
    ingested_at: str


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    doc_id: str
    ord: int
    section: str
    text: str
    char_start: int
    char_end: int
    page: int | None
    source: str
    title: str

    @property
    def index_text(self) -> str:
        return index_text(self.title, self.section, self.text)


def index_text(title: str, section: str, text: str) -> str:
    """Text fed to BM25 and the embedder: a contextual header, then the chunk.

    The header repeats the document title and heading path, so a chunk deep in
    a section still matches queries that name the topic of the section.
    """
    header = f"{title} > {section}" if section else title
    return f"{header}\n{text}"


def chunk_id(doc_id: str, ordinal: int) -> str:
    return f"{doc_id}:{ordinal}"


class IndexStore:
    def __init__(self, path: str | Path, embedder_name: str, dim: int) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self.dim = dim
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)
            self._check_meta(embedder_name, dim)

    def _check_meta(self, embedder_name: str, dim: int) -> None:
        expected = {"schema_version": SCHEMA_VERSION, "embedder": embedder_name, "dim": str(dim)}
        stored = dict(self._conn.execute("SELECT key, value FROM meta").fetchall())
        if not stored:
            self._conn.executemany("INSERT INTO meta VALUES (?, ?)", expected.items())
            return
        for key, value in expected.items():
            if stored.get(key) != value:
                raise IndexMismatchError(
                    f"index {self.path} has {key}={stored.get(key)!r}, expected {value!r}; "
                    "rebuild it (delete the file and ingest again) or use the original embedder"
                )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return DocumentRecord(**dict(row)) if row else None

    def list_documents(self) -> list[DocumentRecord]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM documents ORDER BY id").fetchall()
        return [DocumentRecord(**dict(r)) for r in rows]

    def upsert_document(
        self, doc: LoadedDocument, chunks: Sequence[Chunk], vectors: np.ndarray
    ) -> DocumentRecord:
        """Replace a document and all of its chunks atomically."""
        if vectors.shape != (len(chunks), self.dim):
            raise ValueError(f"expected vectors of shape {(len(chunks), self.dim)}")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        record = DocumentRecord(
            id=doc.doc_id,
            source=doc.source,
            title=doc.title,
            kind=doc.kind,
            sha256=doc.sha256,
            n_chunks=len(chunks),
            ingested_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM documents WHERE id = ?", (doc.doc_id,))
            self._conn.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    record.source,
                    record.title,
                    record.kind,
                    record.sha256,
                    record.n_chunks,
                    record.ingested_at,
                ),
            )
            self._conn.executemany(
                "INSERT INTO chunks (id, doc_id, ord, section, text, char_start, char_end, page,"
                " vector) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        chunk_id(doc.doc_id, c.ord),
                        doc.doc_id,
                        c.ord,
                        c.section,
                        c.text,
                        c.char_start,
                        c.char_end,
                        c.page,
                        vectors[i].tobytes(),
                    )
                    for i, c in enumerate(chunks)
                ],
            )
        return record

    def delete_document(self, doc_id: str) -> bool:
        with self._lock, self._conn:
            cursor = self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        return cursor.rowcount > 0

    def load_chunks(self) -> tuple[list[ChunkRecord], np.ndarray]:
        """Load every chunk (in insertion order) and the matching vector matrix."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT c.id, c.doc_id, c.ord, c.section, c.text, c.char_start, c.char_end,"
                " c.page, c.vector, d.source, d.title"
                " FROM chunks c JOIN documents d ON d.id = c.doc_id ORDER BY c.seq"
            ).fetchall()
        records = [
            ChunkRecord(
                id=r["id"],
                doc_id=r["doc_id"],
                ord=r["ord"],
                section=r["section"],
                text=r["text"],
                char_start=r["char_start"],
                char_end=r["char_end"],
                page=r["page"],
                source=r["source"],
                title=r["title"],
            )
            for r in rows
        ]
        if not rows:
            return records, np.zeros((0, self.dim), dtype=np.float32)
        matrix = np.vstack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
        return records, matrix
