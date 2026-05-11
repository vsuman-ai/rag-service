"""Deterministic, network-free mock of ``vertexai.generative_models.GenerativeModel``.

For Strategy B ("AI-Enhanced Retrieval") the assessment requires a *mocked*
generative model that rewrites/expands the user query. We provide a
rule-based expander that:

  1. Inserts domain-specific synonyms (SRE / infra vocabulary) so that a
     vaguely-worded query like "How does the system handle peak load?"
     expands to mention "autoscaling", "horizontal scaling", "queue
     back-pressure", etc. -- terms the corpus actually uses.
  2. Adds a paraphrase template prefix ("In technical infrastructure
     terms, ...") so the embedding shifts towards the technical-paragraph
     manifold.

The expander is intentionally simple and deterministic so:
  * Benchmark output is reproducible run-to-run.
  * Tests don't have to deal with sampling noise.

The :class:`MockGenerativeModel` class exposes the same ``__init__`` and
``generate_content`` API as ``vertexai.generative_models.GenerativeModel``,
so swapping in the real SDK is just a single ``import`` change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .llm_client import LLMClient, LLMResponse

# Curated SRE / infra synonym map. Keys are lower-case terms a *user* would
# naturally type; values are the technical-vocabulary additions we want the
# rewritten query to include so its embedding lands closer to documents that
# use those terms.
_SYNONYM_MAP: dict[str, tuple[str, ...]] = {
    "peak load": ("autoscaling", "horizontal scaling", "load balancer"),
    "load": ("traffic", "throughput", "concurrent requests"),
    "handle": ("manage", "absorb", "process"),
    "slow": ("latency", "tail latency", "p99 response time"),
    "fast": ("low latency", "high throughput"),
    "crash": ("failure", "fault tolerance", "graceful degradation"),
    "outage": ("circuit breaker", "failover", "redundancy"),
    "scale": ("horizontal scaling", "autoscaling", "sharding"),
    "store": ("persist", "storage", "database"),
    "search": ("query", "retrieval", "indexing"),
    "user": ("client", "request", "consumer"),
    "data": ("dataset", "records", "payload"),
    "speed up": ("cache", "caching", "memoization"),
    "downtime": ("availability", "blue-green deployment", "rolling update"),
    "limit": ("rate limit", "throttle", "quota"),
    "secure": ("authentication", "authorization", "tls"),
    "monitor": ("observability", "metrics", "tracing"),
    "queue": ("message queue", "back-pressure", "buffering"),
}

_PREFIX = "In technical infrastructure terms, "


@dataclass(frozen=True)
class GenerationConfig:
    """Mirrors ``vertexai.generative_models.GenerationConfig``."""

    temperature: float = 0.0
    max_output_tokens: int = 256


@dataclass(frozen=True)
class MockGenerationResponse:
    """Mirrors ``vertexai.generative_models.GenerationResponse`` minimally."""

    text: str


class MockGenerativeModel:
    """Mirrors the public surface of ``vertexai.generative_models.GenerativeModel``."""

    def __init__(self, model_name: str = "gemini-1.5-flash") -> None:
        self._model_name = model_name

    def generate_content(
        self,
        prompt: str,
        generation_config: GenerationConfig | None = None,
    ) -> MockGenerationResponse:
        return MockGenerationResponse(text=_rewrite(prompt))

    @property
    def model_name(self) -> str:
        return self._model_name


class VertexGenerativeMock(LLMClient):
    """LLMClient implementation that wraps :class:`MockGenerativeModel`.

    This is what the application uses by default, satisfying the assessment's
    mocking requirement while still producing useful query rewrites.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash-mock") -> None:
        self._model_name = model_name
        self._model = MockGenerativeModel(model_name=model_name)

    def generate(self, prompt: str) -> LLMResponse:
        response = self._model.generate_content(prompt)
        return LLMResponse(text=response.text, model=self._model_name)

    @property
    def model_name(self) -> str:
        return self._model_name


def _rewrite(prompt: str) -> str:
    """Extract the user query out of the prompt and produce a richer rewrite.

    The prompt format from :mod:`QueryExpansionService` is::

        Original query: <text>
        Rewrite the query ...

    so we pull the line after ``Original query:`` if it exists, otherwise we
    treat the whole prompt as the query.
    """
    match = re.search(r"Original query:\s*(.+?)(?:\n|$)", prompt, re.IGNORECASE)
    user_query = match.group(1).strip() if match else prompt.strip()
    return _expand_query(user_query)


def _expand_query(query: str) -> str:
    lowered = query.lower()
    additions: list[str] = []
    seen: set[str] = set()
    for trigger, synonyms in _SYNONYM_MAP.items():
        if trigger in lowered:
            for synonym in synonyms:
                if synonym not in seen and synonym not in lowered:
                    seen.add(synonym)
                    additions.append(synonym)
    if not additions:
        return query
    addition_clause = ", ".join(additions)
    return f"{_PREFIX}{query.strip().rstrip('?.!')} (also consider: {addition_clause})."


__all__ = [
    "GenerationConfig",
    "MockGenerationResponse",
    "MockGenerativeModel",
    "VertexGenerativeMock",
]
