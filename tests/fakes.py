"""Test doubles: a scripted answer model and a tiny PDF writer."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from hybrid_rag_api.generation import Generation, StreamEnd, TextDelta


@dataclass
class FakeAnswerModel:
    """Scripted stand-in for Claude: returns canned replies and records prompts."""

    replies: list[str] = field(default_factory=list)
    stop_reason: str = "end_turn"
    chunk_size: int = 7
    name: str = "fake-model"
    calls: list[tuple[str, str]] = field(default_factory=list)

    def _next(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.replies:
            raise AssertionError("FakeAnswerModel called more times than scripted")
        return self.replies.pop(0)

    async def generate(self, system: str, user: str) -> Generation:
        return Generation(self._next(system, user), self.stop_reason, self.name)

    async def stream(self, system: str, user: str) -> AsyncIterator[TextDelta | StreamEnd]:
        text = self._next(system, user)
        for i in range(0, len(text), self.chunk_size):
            yield TextDelta(text[i : i + self.chunk_size])
        yield StreamEnd(self.stop_reason, self.name)


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal, valid single-font PDF with one text page per entry."""
    page_ids = [4 + 2 * i for i in range(len(pages))]
    objects: dict[int, str] = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] "
        f"/Count {len(pages)} >>",
        3: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for page_id, text in zip(page_ids, pages, strict=True):
        ops = ["BT", "/F1 12 Tf", "14 TL", "72 720 Td"]
        for line in text.split("\n"):
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"({escaped}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops)
        objects[page_id] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>"
        )
        objects[page_id + 1] = f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"
    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for object_id in sorted(objects):
        offsets[object_id] = len(out)
        out += f"{object_id} 0 obj\n{objects[object_id]}\nendobj\n".encode("latin-1")
    xref = len(out)
    size = max(objects) + 1
    out += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    for object_id in range(1, size):
        out += f"{offsets[object_id]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)
