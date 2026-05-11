"""Wraps an :class:`LLMClient` and produces an expanded form of a user query.

Used exclusively by Strategy B ("AI-Enhanced Retrieval"). The service
captures both the original query and the expanded query so the benchmark
report can show reviewers exactly what the LLM did.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.llm.llm_client import LLMClient
from core.utils.logger import logger

_EXPANSION_PROMPT_TEMPLATE = (
    "You are a query rewriter for a technical-documentation retrieval system.\n"
    "Original query: {query}\n"
    "Rewrite the query to be embedding-friendly: keep the meaning, expand "
    "abbreviations, add closely-related technical synonyms that documents in "
    "this domain commonly use, and remove conversational fluff. Return only "
    "the rewritten query."
)


@dataclass(frozen=True)
class QueryExpansion:
    original: str
    expanded: str
    model: str


class QueryExpansionService:
    """Pure orchestration: format prompt, call LLMClient, return both forms."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def expand(self, query: str) -> QueryExpansion:
        prompt = _EXPANSION_PROMPT_TEMPLATE.format(query=query)
        response = self._llm_client.generate(prompt)
        expanded = (response.text or query).strip() or query
        logger.debug(
            "Query expansion via {}: '{}' -> '{}'",
            response.model,
            query,
            expanded,
        )
        return QueryExpansion(original=query, expanded=expanded, model=response.model)


__all__ = ["QueryExpansion", "QueryExpansionService"]
