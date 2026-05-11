"""Retrieval strategy identifiers used across services and DTOs."""

from __future__ import annotations

from enum import Enum


class RetrievalStrategyName(str, Enum):
    """Identifier for each retrieval strategy.

    ``RAW`` -- Strategy A: embed the user query verbatim and search.
    ``EXPANDED`` -- Strategy B: rewrite/expand the query with a (mocked)
    LLM and then embed+search.
    """

    RAW = "raw"
    EXPANDED = "expanded"


__all__ = ["RetrievalStrategyName"]
