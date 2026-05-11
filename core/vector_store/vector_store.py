"""Abstract vector store interface.

Production code depends on :class:`VectorStore`, never on a concrete backend.
Today the only implementation is :class:`ChromaVectorStore`; migrating to
Qdrant or Vertex AI Vector Search means writing a sibling class and changing
one wiring line in the DI container.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .models import QueryResult, VectorRecord


class VectorStore(ABC):
    """Minimal contract for an embedded or remote vector database."""

    @abstractmethod
    def upsert(self, records: Sequence[VectorRecord]) -> None:
        """Insert or update vectors. Implementations MUST be idempotent on ``id``."""

    @abstractmethod
    def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        where: dict[str, object] | None = None,
    ) -> list[QueryResult]:
        """Return the ``top_k`` nearest neighbours of ``embedding``.

        ``where`` is an optional metadata filter (backend-specific shape; for
        Chroma it follows the ``where`` clause schema).
        """

    @abstractmethod
    def count(self) -> int:
        """Number of vectors currently in the collection."""

    @abstractmethod
    def reset(self) -> None:
        """Drop all vectors. Used by tests and ``POST /rag/ingest?replace=true``."""

    def close(self) -> None:  # noqa: B027 - intentional no-op default; override only if needed
        """Release any underlying resources. Default no-op for in-process backends."""


__all__ = ["VectorStore"]
