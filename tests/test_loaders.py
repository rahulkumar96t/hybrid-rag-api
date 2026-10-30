import pytest

from hybrid_rag_api.loaders import PAGE_BREAK, DocumentError, discover, load_bytes, slugify
from tests.fakes import make_pdf


def test_markdown_title_from_first_h1_and_id_from_filename():
    doc = load_bytes("Getting Started.md", b"intro\n\n# Getting started\n\ntext")
    assert doc.doc_id == "getting-started"
    assert doc.title == "Getting started"
    assert doc.kind == "markdown"
    assert len(doc.sha256) == 64


def test_text_files_strip_bom_and_normalise_newlines():
    doc = load_bytes("notes.txt", "﻿line one\r\nline two".encode())
    assert doc.text == "line one\nline two"
    assert doc.title == "notes"


def test_pdf_pages_are_joined_with_page_breaks():
    data = make_pdf(["Hello from page one", "Second (page) text"])
    doc = load_bytes("manual.pdf", data)
    pages = doc.text.split(PAGE_BREAK)
    assert len(pages) == 2
    assert "Hello from page one" in pages[0]
    assert "Second (page) text" in pages[1]


def test_broken_pdf_raises_document_error():
    with pytest.raises(DocumentError):
        load_bytes("broken.pdf", b"%PDF-1.4 this is not really a pdf")


@pytest.mark.parametrize(
    ("name", "data", "message"),
    [("image.png", b"\x89PNG", "unsupported"), ("empty.md", b"  \n ", "no extractable text")],
)
def test_rejected_inputs(name, data, message):
    with pytest.raises(DocumentError, match=message):
        load_bytes(name, data)


def test_slugify():
    assert slugify("Release Notes (v2)") == "release-notes-v2"
    assert slugify("???") == "doc"
    assert len(slugify("x" * 200)) == 64


def test_discover_filters_supported_files(tmp_path):
    (tmp_path / "a.md").write_text("# A")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("b")
    (tmp_path / "c.png").write_bytes(b"x")
    found = discover([tmp_path])
    assert [p.name for p in found] == ["a.md", "b.txt"]
    with pytest.raises(FileNotFoundError):
        discover([tmp_path / "missing"])
