"""Shared pytest fixtures.

The deterministic ``FakeEmbeddingService`` keeps unit tests fast (no model
downloads) while still exercising the real :class:`EmbeddingService`
interface. Tests that need the real sentence-transformer model are marked
``@pytest.mark.integration``.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

import pytest

# Make both packages importable even when running pytest without ``pip
# install -e .`` -- handy in CI containers and from a clean checkout. The
# project ships:
#   * ``core``  at the repo root  (framework-agnostic libs)
#   * ``rag``   under ``src/``    (FastAPI app + benchmarks + CLI)
_REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from core.containers.app_container import build_container  # noqa: E402
from core.embeddings.embedding_service import EmbeddingService  # noqa: E402
from core.llm.llm_client import LLMClient, LLMResponse  # noqa: E402
from core.vector_store.chroma_store import ChromaVectorStore  # noqa: E402

from rag.app.configurations.rag_configurations import RAGConfigurations  # noqa: E402


class FakeEmbeddingService(EmbeddingService):
    """Cheap deterministic embedding that depends only on text content.

    Generates an 8-dimensional bag-of-keywords vector. The 8 axes line up with
    SRE concepts our query expander injects, so the vector for "autoscaling
    horizontal scaling" lands near the autoscaling document but far from the
    caching document.
    """

    _KEYWORDS: ClassVar[list[tuple[str, ...]]] = [
        ("autoscal", "horizontal", "scale", "replica"),
        ("load balanc", "round-robin", "pool", "weighted"),
        ("cache", "lru", "redis", "stampede"),
        ("queue", "kafka", "back-pressure", "back pressure"),
        ("circuit breaker", "bulkhead", "half-open"),
        ("blue-green", "blue green", "rollout", "deploy"),
        ("metric", "trace", "log", "observab", "monitor"),
        ("rate limit", "throttle", "quota", "token bucket"),
    ]

    def __init__(self) -> None:
        self._dimension = len(self._KEYWORDS) + 1

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        return [self._vectorise(t) for t in texts]

    def get_dimension(self) -> int:
        return self._dimension

    def _vectorise(self, text: str) -> list[float]:
        lowered = text.lower()
        axes = [0.0] * len(self._KEYWORDS)
        for index, keywords in enumerate(self._KEYWORDS):
            axes[index] = float(sum(lowered.count(keyword) for keyword in keywords))
        # Length-stable bonus axis derived from a content hash so unrelated
        # text doesn't collapse to the same vector when no keywords hit.
        digest = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
        bonus = ((digest % 1000) / 1000.0) * 0.05  # very small magnitude
        vector = [*axes, bonus]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class FakeLLMClient(LLMClient):
    """LLMClient that always returns the same canned rewrite."""

    def __init__(
        self, rewrite: str = "REWRITE: autoscaling horizontal scaling load balancer"
    ) -> None:
        self._rewrite = rewrite

    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(text=self._rewrite, model=self.model_name)

    @property
    def model_name(self) -> str:
        return "fake-llm"


@pytest.fixture()
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def chroma_store(temp_dir: Path) -> ChromaVectorStore:
    return ChromaVectorStore(
        collection_name="test_collection",
        persist_directory=temp_dir / "chroma",
    )


@pytest.fixture()
def fake_embedding_service() -> FakeEmbeddingService:
    return FakeEmbeddingService()


@pytest.fixture()
def fake_llm_client() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture()
def test_config(temp_dir: Path) -> RAGConfigurations:
    """Config pointing at the temp dir so tests don't pollute repo state."""
    env_backup = {key: os.environ.get(key) for key in list(os.environ)}
    os.environ.update(
        {
            "RAG_CHROMA_PERSIST_DIRECTORY": str(temp_dir / "chroma"),
            "RAG_CHROMA_COLLECTION_NAME": "test_collection",
            "RAG_EMBEDDING_BACKEND": "gecko-mock",
            "RAG_LLM_BACKEND": "mock",
        }
    )
    config = RAGConfigurations()
    yield config
    # Restore env to avoid leaking between tests.
    for key, value in env_backup.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture()
def container_with_fakes(
    fake_embedding_service: FakeEmbeddingService,
    fake_llm_client: FakeLLMClient,
    temp_dir: Path,
) -> Any:
    """DI container with fake embedding + LLM but real ChromaDB store."""
    config = RAGConfigurations(
        chroma_persist_directory=temp_dir / "chroma",
        chroma_collection_name="test_collection",
    )
    container = build_container(config)
    container.embedding_service.override(fake_embedding_service)
    container.llm_client.override(fake_llm_client)
    return container


@pytest.fixture(autouse=True)
def _disable_chroma_telemetry() -> None:
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


__all__ = ["FakeEmbeddingService", "FakeLLMClient"]
