"""Paragraph chunker with optional word-level overlap for long paragraphs.

The reference corpus is curated 5-10 paragraph technical text, so paragraph
splitting on blank lines is the right primitive. We add a guard: any
paragraph longer than ``max_words`` is split into overlapping windows so a
single oversized paragraph cannot dominate retrieval results.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A single chunk produced by a chunker.

    ``doc_id`` identifies the source document, ``chunk_index`` orders the
    chunks within that document.
    """

    id: str
    doc_id: str
    chunk_index: int
    text: str


class ParagraphChunker:
    """Splits documents into paragraph-level (optionally windowed) chunks."""

    def __init__(self, max_words: int = 220, overlap_words: int = 40) -> None:
        if max_words <= 0:
            raise ValueError("max_words must be > 0")
        if overlap_words < 0 or overlap_words >= max_words:
            raise ValueError("overlap_words must be in [0, max_words)")
        self._max_words = max_words
        self._overlap_words = overlap_words

    def chunk(self, doc_id: str, text: str) -> list[Chunk]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[Chunk] = []
        chunk_idx = 0
        for paragraph in paragraphs:
            windows = self._maybe_window(paragraph)
            for window in windows:
                chunks.append(
                    Chunk(
                        id=f"{doc_id}::chunk-{chunk_idx}",
                        doc_id=doc_id,
                        chunk_index=chunk_idx,
                        text=window,
                    )
                )
                chunk_idx += 1
        return chunks

    def chunk_many(self, documents: Sequence[tuple[str, str]]) -> list[Chunk]:
        """``documents`` is a sequence of ``(doc_id, text)`` tuples."""
        out: list[Chunk] = []
        for doc_id, text in documents:
            out.extend(self.chunk(doc_id, text))
        return out

    def _maybe_window(self, paragraph: str) -> list[str]:
        words = paragraph.split()
        if len(words) <= self._max_words:
            return [paragraph]
        step = self._max_words - self._overlap_words
        windows: list[str] = []
        for start in range(0, len(words), step):
            end = start + self._max_words
            windows.append(" ".join(words[start:end]))
            if end >= len(words):
                break
        return windows


__all__ = ["Chunk", "ParagraphChunker"]
