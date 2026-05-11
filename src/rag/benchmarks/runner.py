"""Benchmark runner: executes Strategy A & B over a query list, records metrics.

Produces a single JSON-serialisable report consumed by both the FastAPI
``/rag/benchmark`` endpoint and :mod:`rag.benchmarks.reporting`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rag.app.constants.strategy_constants import RetrievalStrategyName
from rag.app.services.retrieval_service import RetrievalService
from rag.app.services.retrieval_strategy import StrategyResult

from .metrics import (
    dedupe_preserve_order,
    jaccard_overlap,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class BenchmarkRunner:
    """Runs each query through every strategy and aggregates metrics."""

    def __init__(self, retrieval_service: RetrievalService) -> None:
        self._retrieval_service = retrieval_service

    def run(
        self,
        queries: Sequence[dict[str, Any]],
        top_k: int = 3,
    ) -> dict[str, Any]:
        comparisons: list[dict[str, Any]] = []
        for query_entry in queries:
            query_text = query_entry["query"]
            relevant = set(query_entry.get("relevant_doc_ids", []) or [])

            strategies = self._retrieval_service.search_all(query=query_text, top_k=top_k)
            raw = strategies[RetrievalStrategyName.RAW]
            expanded = strategies[RetrievalStrategyName.EXPANDED]

            raw_dto = _strategy_to_dict(raw, relevant, top_k)
            expanded_dto = _strategy_to_dict(expanded, relevant, top_k)

            overlap = jaccard_overlap(
                raw_dto["retrieved_doc_ids"], expanded_dto["retrieved_doc_ids"]
            )
            comparisons.append(
                {
                    "query": query_text,
                    "relevant_doc_ids": sorted(relevant),
                    "raw": raw_dto,
                    "expanded": expanded_dto,
                    "overlap_at_k": overlap,
                }
            )

        return {
            "top_k": top_k,
            "comparisons": comparisons,
            "summary": _summarise(comparisons, top_k),
        }


def _strategy_to_dict(
    result: StrategyResult,
    relevant: set[str],
    top_k: int,
) -> dict[str, Any]:
    retrieved_doc_ids = dedupe_preserve_order(
        [hit.metadata.get("doc_id") or hit.id for hit in result.results]
    )
    expansion = None
    if result.expansion is not None:
        expansion = {
            "original": result.expansion.original,
            "expanded": result.expansion.expanded,
            "model": result.expansion.model,
        }
    return {
        "query": result.query,
        "effective_query": result.effective_query,
        "latency_ms": result.latency_ms,
        "expansion": expansion,
        "hits": [
            {
                "id": hit.id,
                "doc_id": hit.metadata.get("doc_id") or hit.id,
                "text": hit.text,
                "score": hit.score,
                "metadata": hit.metadata,
            }
            for hit in result.results
        ],
        "retrieved_doc_ids": retrieved_doc_ids,
        "metrics": {
            "precision_at_k": precision_at_k(retrieved_doc_ids, relevant, top_k),
            "recall_at_k": recall_at_k(retrieved_doc_ids, relevant, top_k),
            "mrr": reciprocal_rank(retrieved_doc_ids, relevant),
        },
    }


def _summarise(comparisons: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    def mean(side: str, key: str) -> float:
        values = [entry[side]["metrics"][key] for entry in comparisons]
        return sum(values) / len(values) if values else 0.0

    def mean_latency(side: str) -> float:
        values = [entry[side]["latency_ms"] for entry in comparisons]
        return sum(values) / len(values) if values else 0.0

    return {
        "queries": len(comparisons),
        "top_k": top_k,
        "raw": {
            "precision_at_k": mean("raw", "precision_at_k"),
            "recall_at_k": mean("raw", "recall_at_k"),
            "mrr": mean("raw", "mrr"),
            "avg_latency_ms": mean_latency("raw"),
        },
        "expanded": {
            "precision_at_k": mean("expanded", "precision_at_k"),
            "recall_at_k": mean("expanded", "recall_at_k"),
            "mrr": mean("expanded", "mrr"),
            "avg_latency_ms": mean_latency("expanded"),
        },
        "average_overlap_at_k": (
            sum(entry["overlap_at_k"] for entry in comparisons) / len(comparisons)
            if comparisons
            else 0.0
        ),
    }


__all__ = ["BenchmarkRunner"]
