"""Document ingestion: chunk -> embed -> upsert.

The single-responsibility class used by ``POST /rag/ingest`` and the CLI
seeding step. It takes already-parsed ``(doc_id, text, metadata)`` tuples
so the corpus loader (JSON / file / API payload) can stay decoupled.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from core.chunking.paragraph_chunker import Chunk, ParagraphChunker
from core.embeddings.embedding_service import EmbeddingService
from core.utils.logger import logger
from core.vector_store.models import VectorRecord
from core.vector_store.vector_store import VectorStore


@dataclass(frozen=True)
class IngestDocument:
    """A single document handed to the ingestion service."""

    doc_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class IngestSummary:
    documents_ingested: int
    chunks_indexed: int
    collection_size: int


class IngestionService:
    def __init__(
        self,
        chunker: ParagraphChunker,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self._chunker = chunker
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def ingest(
        self,
        documents: Iterable[IngestDocument],
        replace: bool = False,
    ) -> IngestSummary:
        doc_list = list(documents)
        if not doc_list:
            return IngestSummary(0, 0, self._vector_store.count())

        if replace:
            self._vector_store.reset()

        chunks: list[Chunk] = []
        doc_metadata: dict[str, dict[str, Any]] = {}
        for document in doc_list:
            doc_chunks = self._chunker.chunk(document.doc_id, document.text)
            chunks.extend(doc_chunks)
            doc_metadata[document.doc_id] = document.metadata

        embeddings = self._embedding_service.embed([chunk.text for chunk in chunks])

        records = [
            VectorRecord(
                id=chunk.id,
                text=chunk.text,
                embedding=embedding,
                metadata={
                    **doc_metadata.get(chunk.doc_id, {}),
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                },
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self._vector_store.upsert(records)
        summary = IngestSummary(
            documents_ingested=len(doc_list),
            chunks_indexed=len(records),
            collection_size=self._vector_store.count(),
        )
        logger.info(
            "Ingest complete: {} docs -> {} chunks (collection size: {}).",
            summary.documents_ingested,
            summary.chunks_indexed,
            summary.collection_size,
        )
        return summary


__all__ = ["IngestDocument", "IngestSummary", "IngestionService"]
