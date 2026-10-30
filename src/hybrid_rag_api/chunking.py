"""Structure-aware recursive chunking with overlap.

1. The document is cut into *sections*: Markdown headings (outside code
   fences), PDF pages, or the whole file for plain text. A chunk never spans
   two sections, and it remembers its heading path ("Guide > Install > Linux").
2. Each section is split recursively, trying the coarsest separator first:
   blank lines (paragraphs), then line breaks, then sentence ends, then any
   whitespace. Paragraph and sentence cuts are not allowed inside fenced code.
3. Adjacent pieces are packed greedily into chunks of at most ``max_words``
   words. The next chunk starts a few pieces back, so whole sentences/lines (up
   to ``overlap_words`` words) are repeated across the boundary.

Everything works on character offsets into the original text, so every chunk
maps back to an exact ``[char_start, char_end)`` range of its source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise

from .loaders import PAGE_BREAK
from .text import word_count

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
# Coarse to fine. The cut goes after the separator, so it stays with the left piece.
_SEPARATORS = (
    re.compile(r"\n[ \t]*\n\s*"),  # paragraphs
    re.compile(r"\n"),  # lines
    re.compile(r"(?<=[.!?])\s+"),  # sentences
    re.compile(r"\s+"),  # words
)
_LINE_LEVEL = 1  # cuts at line breaks are allowed inside code fences


@dataclass(frozen=True)
class Section:
    path: str
    start: int
    end: int
    page: int | None = None


@dataclass(frozen=True)
class Chunk:
    ord: int
    text: str
    section: str
    char_start: int
    char_end: int
    page: int | None = None


def markdown_sections(text: str) -> tuple[list[Section], list[tuple[int, int]]]:
    """Return heading-delimited sections and the spans of fenced code blocks."""
    sections: list[Section] = []
    fences: list[tuple[int, int]] = []
    stack: list[tuple[int, str]] = []
    current_start, current_path = 0, ""
    fence_marker: str | None = None
    fence_start = 0
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        fence = _FENCE_RE.match(stripped)
        if fence_marker is not None:
            if (
                fence
                and fence.group(1)[0] == fence_marker[0]
                and len(fence.group(1)) >= len(fence_marker)
            ):
                fences.append((fence_start, offset + len(line)))
                fence_marker = None
        elif fence:
            fence_marker, fence_start = fence.group(1), offset
        elif heading := _HEADING_RE.match(stripped):
            if offset > current_start:
                sections.append(Section(current_path, current_start, offset))
            level, title = len(heading.group(1)), heading.group(2).strip()
            stack = [(lvl, t) for lvl, t in stack if lvl < level]
            stack.append((level, title))
            current_start, current_path = offset, " > ".join(t for _, t in stack)
        offset += len(line)
    if fence_marker is not None:  # unterminated fence runs to the end
        fences.append((fence_start, len(text)))
    if len(text) > current_start:
        sections.append(Section(current_path, current_start, len(text)))
    return sections, fences


def page_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    start = 0
    for page_no, page in enumerate(text.split(PAGE_BREAK), start=1):
        end = start + len(page)
        sections.append(Section(f"page {page_no}", start, end, page=page_no))
        start = end + len(PAGE_BREAK)
    return sections


def _is_bare_heading(chunk: str) -> bool:
    return "\n" not in chunk and _HEADING_RE.match(chunk) is not None


def _inside(position: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < position < end for start, end in spans)


def split_spans(
    text: str,
    start: int,
    end: int,
    max_words: int,
    protected: list[tuple[int, int]] | None = None,
    level: int = 0,
) -> list[tuple[int, int]]:
    """Recursively split ``text[start:end]`` into contiguous spans of <= max_words."""
    if level >= len(_SEPARATORS) or word_count(text[start:end]) <= max_words:
        # The last separator is any whitespace, so a span that is still too long
        # here is a single "word" and cannot be cut further.
        return [(start, end)]
    guard = protected if protected and level != _LINE_LEVEL else []
    cuts = [
        m.end()
        for m in _SEPARATORS[level].finditer(text, start, end)
        if start < m.end() < end and not _inside(m.end(), guard)
    ]
    if not cuts:
        return split_spans(text, start, end, max_words, protected, level + 1)
    spans: list[tuple[int, int]] = []
    for left, right in pairwise([start, *cuts, end]):
        if word_count(text[left:right]) > max_words:
            spans.extend(split_spans(text, left, right, max_words, protected, level + 1))
        else:
            spans.append((left, right))
    return spans


def pack_spans(
    text: str, spans: list[tuple[int, int]], max_words: int, overlap_words: int
) -> list[tuple[int, int]]:
    """Greedily pack contiguous spans into windows, repeating trailing spans as overlap."""
    counts = [word_count(text[s:e]) for s, e in spans]
    windows: list[tuple[int, int]] = []
    i = 0
    while i < len(spans):
        j, total = i, 0
        while j < len(spans) and (j == i or total + counts[j] <= max_words):
            total += counts[j]
            j += 1
        windows.append((spans[i][0], spans[j - 1][1]))
        if j >= len(spans):
            break
        k, overlap = j, 0
        while k - 1 > i and overlap + counts[k - 1] <= overlap_words:
            k -= 1
            overlap += counts[k]
        i = k
    return windows


def chunk_text(text: str, kind: str, max_words: int = 200, overlap_words: int = 40) -> list[Chunk]:
    """Split a loaded document into chunks; see the module docstring."""
    if max_words < 1:
        raise ValueError("max_words must be >= 1")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be in [0, max_words)")
    fences: list[tuple[int, int]] = []
    if kind == "markdown":
        sections, fences = markdown_sections(text)
    elif kind == "pdf":
        sections = page_sections(text)
    else:
        sections = [Section("", 0, len(text))]

    chunks: list[Chunk] = []
    for section in sections:
        spans = split_spans(text, section.start, section.end, max_words, fences)
        for start, end in pack_spans(text, spans, max_words, overlap_words):
            raw = text[start:end]
            stripped = raw.strip()
            if not stripped or (kind == "markdown" and _is_bare_heading(stripped)):
                # A heading with no body adds nothing: its title is already part
                # of the section path of the chunks below it.
                continue
            lead = len(raw) - len(raw.lstrip())
            chunk_start = start + lead
            chunks.append(
                Chunk(
                    ord=len(chunks),
                    text=stripped,
                    section=section.path,
                    char_start=chunk_start,
                    char_end=chunk_start + len(stripped),
                    page=section.page,
                )
            )
    return chunks
