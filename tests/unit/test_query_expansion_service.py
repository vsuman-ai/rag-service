"""Tests for QueryExpansionService and the mocked Vertex GenerativeModel."""

from __future__ import annotations

import sys
import types

import pytest
from core.llm.llm_client import LLMResponse
from core.llm.vertex_generative_mock import (
    GenerationConfig,
    MockGenerativeModel,
    VertexGenerativeMock,
)

from rag.app.services.query_expansion_service import QueryExpansionService


class _CapturingLLM:
    last_prompt: str | None = None

    def generate(self, prompt: str) -> LLMResponse:
        self.last_prompt = prompt
        return LLMResponse(text="REWRITE", model="fake")

    @property
    def model_name(self) -> str:
        return "fake"


def test_service_invokes_llm_and_returns_expansion() -> None:
    llm = _CapturingLLM()
    service = QueryExpansionService(llm_client=llm)
    expansion = service.expand("How does the system handle peak load?")
    assert llm.last_prompt is not None
    assert "Original query:" in llm.last_prompt
    assert expansion.expanded == "REWRITE"
    assert expansion.original == "How does the system handle peak load?"
    assert expansion.model == "fake"


def test_vertex_generative_mock_expands_with_sre_synonyms() -> None:
    mock = VertexGenerativeMock()
    response = mock.generate("Original query: How does the system handle peak load?\nRewrite...")
    rewritten = response.text.lower()
    # Mock should inject canonical SRE vocabulary so embeddings shift.
    assert "autoscaling" in rewritten
    assert "load balancer" in rewritten or "horizontal scaling" in rewritten


def test_mock_generative_model_matches_vertex_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mock must be drop-in compatible with vertexai.generative_models."""
    fake_module = types.ModuleType("vertexai.generative_models")
    fake_module.GenerativeModel = MockGenerativeModel
    fake_module.GenerationConfig = GenerationConfig
    monkeypatch.setitem(sys.modules, "vertexai", types.ModuleType("vertexai"))
    monkeypatch.setitem(sys.modules, "vertexai.generative_models", fake_module)

    from vertexai.generative_models import GenerationConfig as Gc
    from vertexai.generative_models import GenerativeModel

    model = GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        "Original query: How do we handle peak load?\nRewrite ...",
        generation_config=Gc(temperature=0.0),
    )
    assert hasattr(response, "text")
    assert "autoscaling" in response.text.lower()


def test_expansion_falls_back_to_original_when_llm_returns_blank() -> None:
    class _BlankLLM:
        def generate(self, prompt: str) -> LLMResponse:
            return LLMResponse(text="   ", model="fake")

        @property
        def model_name(self) -> str:
            return "fake"

    service = QueryExpansionService(llm_client=_BlankLLM())
    expansion = service.expand("anything")
    assert expansion.expanded == "anything"
