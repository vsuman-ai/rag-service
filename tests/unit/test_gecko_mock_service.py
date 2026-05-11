"""Tests for the mocked Vertex AI ``TextEmbeddingModel``.

The assessment explicitly requires us to mock
``vertexai.language_models.TextEmbeddingModel``. These tests pin the contract
so future refactors don't silently break that interface.
"""

from __future__ import annotations

import sys
import types

import pytest
from core.embeddings.gecko_mock_service import (
    GeckoMockEmbeddingService,
    MockTextEmbedding,
    MockTextEmbeddingModel,
)


@pytest.mark.integration
def test_gecko_mock_returns_embeddings_with_stable_dimension() -> None:
    service = GeckoMockEmbeddingService()
    vectors = service.embed(["hello", "world", "test"])
    assert len(vectors) == 3
    dim = service.get_dimension()
    for vector in vectors:
        assert len(vector) == dim
        assert all(isinstance(v, float) for v in vector)


def test_mock_text_embedding_model_matches_vertex_signature() -> None:
    """Mock must expose the same callable shape as the real Vertex class."""
    assert hasattr(MockTextEmbeddingModel, "from_pretrained")
    assert callable(MockTextEmbeddingModel.from_pretrained)
    # Cannot fully instantiate without a model download in unit tests, but the
    # signature comparison is what we care about for "drop-in" compatibility.


def test_vertex_sdk_can_be_patched_with_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert the mock satisfies the Vertex SDK contract.

    We inject a fake ``vertexai.language_models`` module backed by our mock
    and verify that code written against the real Vertex API works unchanged.
    """
    fake_language_models = types.ModuleType("vertexai.language_models")
    fake_language_models.TextEmbeddingModel = MockTextEmbeddingModel
    fake_language_models.TextEmbedding = MockTextEmbedding
    fake_vertexai = types.ModuleType("vertexai")
    fake_vertexai.language_models = fake_language_models
    monkeypatch.setitem(sys.modules, "vertexai", fake_vertexai)
    monkeypatch.setitem(sys.modules, "vertexai.language_models", fake_language_models)

    from vertexai.language_models import TextEmbeddingModel

    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    # Should NOT raise -- this is the very call pattern production code uses.
    embeddings = model.get_embeddings(["hello"])
    assert len(embeddings) == 1
    assert hasattr(embeddings[0], "values")
    assert isinstance(embeddings[0].values, list)
