"""Load Markdown, plain-text and PDF files into a common document shape."""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

KINDS = {".md": "markdown", ".markdown": "markdown", ".txt": "text", ".pdf": "pdf"}
PAGE_BREAK = "\f"

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_H1_RE = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.MULTILINE)


class DocumentError(ValueError):
    """The file could not be turned into text."""


class UnsupportedFileType(DocumentError):
    """The file extension is not one of the supported formats."""


@dataclass(frozen=True)
class LoadedDocument:
    doc_id: str
    source: str
    title: str
    kind: str
    text: str
    sha256: str


def slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", name.lower()).strip("-")
    return slug[:64].strip("-") or "doc"


def kind_for(filename: str) -> str:
    kind = KINDS.get(Path(filename).suffix.lower())
    if kind is None:
        supported = ", ".join(sorted(KINDS))
        raise UnsupportedFileType(f"unsupported file type {filename!r} (supported: {supported})")
    return kind


def _pdf_text(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        locked = reader.is_encrypted and not reader.decrypt("")
        pages = [] if locked else [page.extract_text() or "" for page in reader.pages]
    except (PyPdfError, ValueError, KeyError) as exc:  # malformed files fail in many ways
        raise DocumentError(f"could not read PDF: {exc}") from exc
    if locked:
        raise DocumentError("password-protected PDFs are not supported")
    # Form feeds mark page boundaries; the chunker turns each page into a section.
    return PAGE_BREAK.join(page.replace(PAGE_BREAK, " ") for page in pages)


def load_bytes(filename: str, data: bytes) -> LoadedDocument:
    """Decode an uploaded or on-disk file. The document id is derived from its name."""
    name = Path(filename).name
    kind = kind_for(name)
    if kind == "pdf":
        text = _pdf_text(data)
    else:
        text = data.decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
    if not text.strip():
        raise DocumentError(f"{name} contains no extractable text")
    stem = Path(name).stem
    title = stem
    if kind == "markdown" and (match := _H1_RE.search(text)):
        title = match.group(1).strip()
    return LoadedDocument(
        doc_id=slugify(stem),
        source=name,
        title=title,
        kind=kind,
        text=text,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def load_path(path: str | Path) -> LoadedDocument:
    path = Path(path)
    return load_bytes(path.name, path.read_bytes())


def discover(paths: list[str | Path]) -> list[Path]:
    """Expand files and directories into a sorted list of supported files."""
    found: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found.update(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in KINDS)
        elif path.is_file():
            found.add(path)
        else:
            raise FileNotFoundError(path)
    return sorted(found)
