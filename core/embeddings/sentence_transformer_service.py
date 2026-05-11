"""Concrete embedding service backed by ``sentence-transformers``.

Default model is ``all-MiniLM-L6-v2`` (384-dim, CPU-friendly, the de-facto
small-RAG baseline). Outputs are L2-normalised so they can be paired with a
cosine-similarity vector store without further preprocessing.
"""

from __future__ import annotations

from collections.abc import Iterable
from threading import Lock

import numpy as np
from sentence_transformers import SentenceTransformer

from core.utils.logger import logger

from .embedding_service import EmbeddingService

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbeddingService(EmbeddingService):
    """Lazily-loaded sentence-transformers model (loading is expensive)."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._normalize = normalize
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None
        self._lock = Lock()

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info("Loading sentence-transformer model '{}'.", self._model_name)
                    self._model = SentenceTransformer(self._model_name, device=self._device)
                    self._dimension = int(_resolve_dimension(self._model))
        return self._model

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = [t if isinstance(t, str) else str(t) for t in texts]
        if not text_list:
            return []
        model = self._ensure_model()
        raw = model.encode(
            text_list,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        vectors = np.asarray(raw, dtype=np.float32)
        return [list(map(float, row)) for row in vectors]

    def get_dimension(self) -> int:
        if self._dimension is None:
            self._ensure_model()
        assert self._dimension is not None
        return self._dimension


def _resolve_dimension(model: SentenceTransformer) -> int:
    """Pick whichever dimension accessor the installed library version exposes."""
    for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        accessor = getattr(model, attr, None)
        if callable(accessor):
            value = accessor()
            if value is not None:
                return int(value)
    raise RuntimeError("Unable to determine embedding dimension from sentence-transformers model.")


__all__ = ["DEFAULT_MODEL_NAME", "SentenceTransformerEmbeddingService"]
