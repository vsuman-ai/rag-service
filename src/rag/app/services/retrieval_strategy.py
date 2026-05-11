"""Retrieval strategies.

Each strategy turns a user query into a vector-store query. The split
mirrors the assessment's "Strategy A vs Strategy B" framing and the
``SearchQueryPlanner`` pattern from
``bigdata/onticai/semantic_search/src/app/services/search_query_planner.py``:
the orchestrator stays thin, each strategy owns its own logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import perf_counter

from core.embeddings.embedding_service import EmbeddingService
from core.vector_store.models import QueryResult
from core.vector_store.vector_store import VectorStore

from rag.app.constants.strategy_constants import RetrievalStrategyName

from .query_expansion_service import QueryExpansion, QueryExpansionService


@dataclass(frozen=True)
class StrategyResult:
    """What every strategy returns: hits + provenance + latency."""

    strategy: RetrievalStrategyName
    query: str
    effective_query: str
    results: list[QueryResult]
    latency_ms: float
    expansion: QueryExpansion | None = None
    extras: dict[str, object] = field(default_factory=dict)


class RetrievalStrategy(ABC):
    """Common base: embed -> vector_store.query, wrapped with timing."""

    name: RetrievalStrategyName

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def search(
        self,
        query: str,
        top_k: int,
        where: dict[str, object] | None = None,
    ) -> StrategyResult:
        start = perf_counter()
        effective_query, expansion = self._effective_query(query)
        vector = self._embedding_service.embed_one(effective_query)
        results = self._vector_store.query(vector, top_k=top_k, where=where)
        elapsed_ms = (perf_counter() - start) * 1000.0
        return StrategyResult(
            strategy=self.name,
            query=query,
            effective_query=effective_query,
            results=results,
            latency_ms=elapsed_ms,
            expansion=expansion,
        )

    @abstractmethod
    def _effective_query(self, query: str) -> tuple[str, QueryExpansion | None]:
        """Return the query string that will actually be embedded."""


class RawQueryStrategy(RetrievalStrategy):
    """Strategy A: embed the user query verbatim and search."""

    name = RetrievalStrategyName.RAW

    def _effective_query(self, query: str) -> tuple[str, QueryExpansion | None]:
        return query, None


class ExpandedQueryStrategy(RetrievalStrategy):
    """Strategy B: rewrite/expand the user query, then embed and search."""

    name = RetrievalStrategyName.EXPANDED

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        query_expansion_service: QueryExpansionService,
    ) -> None:
        super().__init__(embedding_service, vector_store)
        self._expander = query_expansion_service

    def _effective_query(self, query: str) -> tuple[str, QueryExpansion | None]:
        expansion = self._expander.expand(query)
        return expansion.expanded, expansion


__all__ = [
    "ExpandedQueryStrategy",
    "RawQueryStrategy",
    "RetrievalStrategy",
    "StrategyResult",
]
