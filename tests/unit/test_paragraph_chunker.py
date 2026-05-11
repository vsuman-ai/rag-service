from __future__ import annotations

import pytest
from core.chunking.paragraph_chunker import ParagraphChunker


def test_chunks_split_on_blank_lines():
    chunker = ParagraphChunker()
    chunks = chunker.chunk("doc-1", "Paragraph one.\n\nParagraph two.")
    assert [c.text for c in chunks] == ["Paragraph one.", "Paragraph two."]
    assert chunks[0].id == "doc-1::chunk-0"
    assert chunks[1].id == "doc-1::chunk-1"


def test_skips_empty_paragraphs():
    chunker = ParagraphChunker()
    chunks = chunker.chunk("doc-1", "First.\n\n\n\nSecond.\n\n   ")
    assert len(chunks) == 2


def test_windows_long_paragraphs_with_overlap():
    long_text = " ".join(f"word{i}" for i in range(500))
    chunker = ParagraphChunker(max_words=200, overlap_words=50)
    chunks = chunker.chunk("doc-1", long_text)
    assert len(chunks) >= 3
    # First window starts at word0
    assert chunks[0].text.startswith("word0 ")
    # Overlap: last 50 words of chunk 0 are the first 50 words of chunk 1.
    first_tail = chunks[0].text.split()[-50:]
    second_head = chunks[1].text.split()[:50]
    assert first_tail == second_head


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        ParagraphChunker(max_words=10, overlap_words=10)


def test_chunk_many_preserves_doc_ids():
    chunker = ParagraphChunker()
    chunks = chunker.chunk_many([("a", "first"), ("b", "second")])
    assert [c.doc_id for c in chunks] == ["a", "b"]
