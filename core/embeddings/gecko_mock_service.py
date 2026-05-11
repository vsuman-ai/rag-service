"""Embedding service that *simulates* Vertex AI ``textembedding-gecko``.

The assessment requires us to mock
``vertexai.language_models.TextEmbeddingModel``. We do that with a tiny shim
class :class:`MockTextEmbeddingModel` whose ``get_embeddings()`` signature
mirrors Vertex AI's, but the actual numbers come from a local
sentence-transformers model. This means:

  * Production code only ever sees the :class:`EmbeddingService` interface.
  * Tests can monkey-patch ``vertexai.language_models.TextEmbeddingModel``
    with :class:`MockTextEmbeddingModel` and assert the service is wired up
    correctly without making any network calls.
  * Migrating to real Vertex AI is one line:
    ``TextEmbeddingModel.from_pretrained("textembedding-gecko@003")``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

from core.utils.logger import logger

from .embedding_service import EmbeddingService
from .sentence_transformer_service import (
    DEFAULT_MODEL_NAME,
    SentenceTransformerEmbeddingService,
)


@dataclass(frozen=True)
class MockTextEmbedding:
    """Mirrors ``vertexai.language_models.TextEmbedding`` shape."""

    values: list[float]


class MockTextEmbeddingModel:
    """Mirrors ``vertexai.language_models.TextEmbeddingModel`` shape.

    The class method ``from_pretrained`` and instance method
    ``get_embeddings`` match the Vertex AI API so production code targeting
    that interface works unchanged.
    """

    _MODEL_ROUTING: ClassVar[dict[str, str]] = {
        "textembedding-gecko@003": DEFAULT_MODEL_NAME,
        "textembedding-gecko@002": DEFAULT_MODEL_NAME,
        "textembedding-gecko@001": DEFAULT_MODEL_NAME,
    }

    def __init__(self, backend: SentenceTransformerEmbeddingService, model_name: str) -> None:
        self._backend = backend
        self._model_name = model_name

    @classmethod
    def from_pretrained(cls, model_name: str) -> MockTextEmbeddingModel:
        local_model = cls._MODEL_ROUTING.get(model_name, DEFAULT_MODEL_NAME)
        logger.info(
            "Mock Vertex AI gecko routing: '{}' -> local '{}'.", model_name, local_model
        )
        return cls(SentenceTransformerEmbeddingService(model_name=local_model), model_name)

    def get_embeddings(self, texts: list[str]) -> list[MockTextEmbedding]:
        vectors = self._backend.embed(texts)
        return [MockTextEmbedding(values=v) for v in vectors]

    @property
    def name(self) -> str:
        return self._model_name


class GeckoMockEmbeddingService(EmbeddingService):
    """Embedding service that **uses** the mocked Vertex gecko model.

    This is what the application wires up by default. It satisfies the
    assessment's "mock the Vertex SDK" requirement while still producing real
    embeddings (via the underlying sentence-transformers model) so the
    benchmark numbers are meaningful.
    """

    def __init__(self, gecko_model_name: str = "textembedding-gecko@003") -> None:
        self._model_name = gecko_model_name
        self._mock_model = MockTextEmbeddingModel.from_pretrained(gecko_model_name)

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = list(texts)
        if not text_list:
            return []
        embeddings = self._mock_model.get_embeddings(text_list)
        return [embedding.values for embedding in embeddings]

    def get_dimension(self) -> int:
        return self._mock_model._backend.get_dimension()


__all__ = [
    "GeckoMockEmbeddingService",
    "MockTextEmbedding",
    "MockTextEmbeddingModel",
]
