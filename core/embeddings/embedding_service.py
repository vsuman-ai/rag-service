"""Abstract embedding service.

Both ingestion and retrieval go through this interface so that swapping a
local sentence-transformers model for Vertex AI ``textembedding-gecko`` is
purely a wiring change in the DI container.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable


class EmbeddingService(ABC):
    """Single method: text in, dense vector out."""

    @abstractmethod
    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        """Return one embedding per input text, in the same order."""

    def embed_one(self, text: str) -> list[float]:
        """Convenience: embed a single string."""
        return self.embed([text])[0]

    @abstractmethod
    def get_dimension(self) -> int:
        """Embedding vector size (e.g. 384 for ``all-MiniLM-L6-v2``)."""

    @property
    def model_name(self) -> str:
        return getattr(self, "_model_name", self.__class__.__name__)


__all__ = ["EmbeddingService"]
