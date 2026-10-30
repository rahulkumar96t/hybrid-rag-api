from itertools import pairwise

import pytest

from hybrid_rag_api.chunking import chunk_text, markdown_sections, pack_spans, split_spans
from hybrid_rag_api.loaders import PAGE_BREAK
from hybrid_rag_api.text import word_count

MARKDOWN = """\
# Guide

Intro paragraph.

## Install

Run the installer.

### Linux

Use the tarball.

```bash
# not a heading, this is a shell comment

echo done
```

## Configure

Edit the file.
"""


def sentences(n: int, words_per_sentence: int = 10) -> str:
    return " ".join(
        " ".join(f"s{i}w{j}" for j in range(words_per_sentence - 1)) + f" end{i}." for i in range(n)
    )


def test_markdown_sections_track_heading_paths_and_ignore_fenced_hashes():
    sections, fences = markdown_sections(MARKDOWN)
    assert [s.path for s in sections] == [
        "Guide",
        "Guide > Install",
        "Guide > Install > Linux",
        "Guide > Configure",
    ]
    assert len(fences) == 1
    start, end = fences[0]
    assert MARKDOWN[start:end].startswith("```bash") and MARKDOWN[start:end].rstrip().endswith(
        "```"
    )


def test_chunks_carry_sections_and_exact_offsets():
    chunks = chunk_text(MARKDOWN, "markdown", max_words=50, overlap_words=5)
    assert [c.section for c in chunks] == [
        "Guide",
        "Guide > Install",
        "Guide > Install > Linux",
        "Guide > Configure",
    ]
    for c in chunks:
        assert MARKDOWN[c.char_start : c.char_end] == c.text
    assert [c.ord for c in chunks] == list(range(len(chunks)))


def test_bare_headings_produce_no_chunk():
    chunks = chunk_text("# Title\n\n## Section\n\nBody text here.\n", "markdown")
    assert [c.text for c in chunks] == ["## Section\n\nBody text here."]
    assert chunks[0].section == "Title > Section"


def test_long_sections_respect_max_words_and_overlap():
    text = sentences(30)  # 300 words, one paragraph
    chunks = chunk_text(text, "text", max_words=60, overlap_words=20)
    assert len(chunks) > 1
    assert all(word_count(c.text) <= 60 for c in chunks)
    for left, right in pairwise(chunks):
        # the next chunk starts inside the previous one (overlap) ...
        assert right.char_start < left.char_end
        # ... by whole sentences, never more than the overlap budget
        assert word_count(text[right.char_start : left.char_end]) <= 20
        assert text[right.char_start : right.char_start + 1] == "s"
    assert chunks[-1].char_end == len(text.rstrip())


def test_zero_overlap_tiles_the_text():
    text = sentences(12)
    chunks = chunk_text(text, "text", max_words=30, overlap_words=0)
    joined = " ".join(c.text for c in chunks)
    assert joined.split() == text.split()


def test_paragraph_boundaries_are_preferred():
    para_a, para_b = sentences(4), sentences(4)
    chunks = chunk_text(f"{para_a}\n\n{para_b}", "text", max_words=45, overlap_words=0)
    assert [c.text for c in chunks] == [para_a, para_b]


def test_code_fence_is_not_split_at_blank_lines():
    code = "```\n" + "\n\n".join(f"line_{i} = {i}" for i in range(10)) + "\n```"
    text = f"# Code\n\n{sentences(2)}\n\n{code}\n"  # 2 + 20 + 32 words
    chunks = chunk_text(text, "markdown", max_words=40, overlap_words=0)
    assert [c.text for c in chunks][-1] == code


def test_oversized_code_fence_falls_back_to_line_cuts():
    code = "```\n" + "\n".join(f"value_{i} = {i}" for i in range(40)) + "\n```"
    chunks = chunk_text(f"# Code\n\n{code}\n", "markdown", max_words=30, overlap_words=0)
    assert len(chunks) > 1
    assert all(word_count(c.text) <= 30 for c in chunks)


def test_run_on_text_falls_back_to_word_cuts():
    text = " ".join(f"w{i}" for i in range(25))
    assert [word_count(text[s:e]) for s, e in split_spans(text, 0, len(text), 10)] == [1] * 25
    chunks = chunk_text(text, "text", max_words=10, overlap_words=0)
    assert [word_count(c.text) for c in chunks] == [10, 10, 5]


def test_pack_spans_never_stalls_on_an_oversized_span():
    text = "a b c d e f g h"
    windows = pack_spans(text, [(0, 7), (8, 15)], max_words=2, overlap_words=1)
    assert windows == [(0, 7), (8, 15)]


def test_pdf_pages_become_sections():
    text = PAGE_BREAK.join(["First page text.", "Second page text."])
    chunks = chunk_text(text, "pdf")
    assert [(c.section, c.page, c.text) for c in chunks] == [
        ("page 1", 1, "First page text."),
        ("page 2", 2, "Second page text."),
    ]


def test_whitespace_only_input_yields_nothing():
    assert chunk_text("   \n\n  ", "text") == []


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_sizes(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("x", "text", max_words=size, overlap_words=overlap)
