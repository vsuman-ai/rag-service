"""Thin orchestrator: pick a strategy by name and execute it.

Mirrors the role of
``bigdata/onticai/semantic_search/src/app/services/semantic_search_service.py``
-- the service is a wiring layer, strategies own the logic.
"""

from __future__ import annotations

from rag.app.constants.strategy_constants import RetrievalStrategyName

from .retrieval_strategy import (
    ExpandedQueryStrategy,
    RawQueryStrategy,
    RetrievalStrategy,
    StrategyResult,
)


class RetrievalService:
    def __init__(
        self,
        raw_strategy: RawQueryStrategy,
        expanded_strategy: ExpandedQueryStrategy,
    ) -> None:
        self._strategies: dict[RetrievalStrategyName, RetrievalStrategy] = {
            RetrievalStrategyName.RAW: raw_strategy,
            RetrievalStrategyName.EXPANDED: expanded_strategy,
        }

    def search(
        self,
        query: str,
        top_k: int,
        strategy: RetrievalStrategyName,
        where: dict[str, object] | None = None,
    ) -> StrategyResult:
        if strategy not in self._strategies:
            raise ValueError(f"Unknown retrieval strategy: {strategy!r}")
        return self._strategies[strategy].search(query=query, top_k=top_k, where=where)

    def search_all(
        self,
        query: str,
        top_k: int,
        where: dict[str, object] | None = None,
    ) -> dict[RetrievalStrategyName, StrategyResult]:
        """Run every strategy. Used by the benchmark harness."""
        return {
            name: strategy.search(query=query, top_k=top_k, where=where)
            for name, strategy in self._strategies.items()
        }


__all__ = ["RetrievalService"]
