"""LLM client Protocol used by query-expansion.

The retrieval pipeline only sees this interface, never a concrete client. We
ship two implementations:

  * :class:`~core.llm.vertex_generative_mock.VertexGenerativeMock`
    -- the default; rule-based, deterministic, no network. Satisfies the
    assessment's "mock ``vertexai.generative_models.GenerativeModel``"
    requirement.
  * :class:`~core.llm.vertex_generative_client.VertexGenerativeClient`
    -- optional real client backed by Gemini 1.5 Flash on Vertex AI. Used
    only when ``RAG_LLM_BACKEND=vertex`` and the ``vertex`` extra is
    installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class LLMResponse:
    """Single completion returned by an LLM client."""

    text: str
    model: str


@runtime_checkable
class LLMClient(Protocol):
    """Anything that can answer ``generate(prompt)`` is an LLMClient."""

    def generate(self, prompt: str) -> LLMResponse: ...

    @property
    def model_name(self) -> str: ...


__all__ = ["LLMClient", "LLMResponse"]
