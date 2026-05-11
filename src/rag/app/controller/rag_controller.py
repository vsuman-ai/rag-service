"""FastAPI router exposing ingest / search / benchmark / health endpoints.

The controller is intentionally thin: every endpoint resolves its service
from the DI container and delegates immediately.
"""

from __future__ import annotations

from core.containers.app_container import AppContainer
from fastapi import APIRouter, Depends, HTTPException, status

from rag.app.constants.strategy_constants import RetrievalStrategyName
from rag.app.dependencies.container_dependency import get_container
from rag.app.dto.rag_dto import (
    BenchmarkComparisonDTO,
    BenchmarkRequestDTO,
    BenchmarkResponseDTO,
    IngestRequestDTO,
    IngestResponseDTO,
    QueryExpansionDTO,
    QueryHitDTO,
    SearchRequestDTO,
    SearchResponseDTO,
)
from rag.app.services.ingestion_service import IngestDocument
from rag.app.services.retrieval_strategy import StrategyResult

rag_router = APIRouter(prefix="/rag", tags=["rag"])


@rag_router.get("/health")
def health(container: AppContainer = Depends(get_container)) -> dict[str, object]:
    vector_store = container.vector_store()
    return {
        "status": "ok",
        "collection_size": vector_store.count(),
    }


@rag_router.post("/ingest", response_model=IngestResponseDTO)
def ingest(
    payload: IngestRequestDTO,
    container: AppContainer = Depends(get_container),
) -> IngestResponseDTO:
    if not payload.documents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one document is required.",
        )
    service = container.ingestion_service()
    summary = service.ingest(
        documents=[
            IngestDocument(doc_id=d.doc_id, text=d.text, metadata=d.metadata)
            for d in payload.documents
        ],
        replace=payload.replace,
    )
    return IngestResponseDTO(
        documentsIngested=summary.documents_ingested,
        chunksIndexed=summary.chunks_indexed,
        collectionSize=summary.collection_size,
    )


@rag_router.post("/search", response_model=SearchResponseDTO)
def search(
    payload: SearchRequestDTO,
    container: AppContainer = Depends(get_container),
) -> SearchResponseDTO:
    service = container.retrieval_service()
    result = service.search(
        query=payload.query,
        top_k=payload.top_k,
        strategy=payload.strategy,
        where=payload.where,
    )
    return _to_search_response(result)


@rag_router.post("/benchmark", response_model=BenchmarkResponseDTO)
def benchmark(
    payload: BenchmarkRequestDTO,
    container: AppContainer = Depends(get_container),
) -> BenchmarkResponseDTO:
    from rag.benchmarks.runner import BenchmarkRunner

    runner = BenchmarkRunner(retrieval_service=container.retrieval_service())
    queries = payload.queries or _default_benchmark_queries(container)
    if not queries:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No benchmark queries supplied or configured.",
        )
    report = runner.run(queries=[{"query": q} for q in queries], top_k=payload.top_k)
    return _to_benchmark_response(report)


# ---------- helpers ----------


def _to_search_response(result: StrategyResult) -> SearchResponseDTO:
    return SearchResponseDTO(
        strategy=result.strategy,
        query=result.query,
        effectiveQuery=result.effective_query,
        latencyMs=result.latency_ms,
        expansion=(
            QueryExpansionDTO(
                original=result.expansion.original,
                expanded=result.expansion.expanded,
                model=result.expansion.model,
            )
            if result.expansion
            else None
        ),
        hits=[
            QueryHitDTO(id=hit.id, text=hit.text, score=hit.score, metadata=hit.metadata)
            for hit in result.results
        ],
    )


def _default_benchmark_queries(container: AppContainer) -> list[str]:
    from rag.benchmarks.corpus_loader import load_benchmark_queries

    config = container.config()
    if not config.benchmark_queries_path.exists():
        return []
    return [entry["query"] for entry in load_benchmark_queries(config.benchmark_queries_path)]


def _to_benchmark_response(report: dict) -> BenchmarkResponseDTO:
    comparisons = [
        BenchmarkComparisonDTO(
            query=entry["query"],
            raw=_comparison_side(entry["raw"], RetrievalStrategyName.RAW),
            expanded=_comparison_side(entry["expanded"], RetrievalStrategyName.EXPANDED),
            overlapAtK=entry["overlap_at_k"],
        )
        for entry in report["comparisons"]
    ]
    return BenchmarkResponseDTO(
        topK=report["top_k"],
        comparisons=comparisons,
        summary=report["summary"],
    )


def _comparison_side(side: dict, strategy: RetrievalStrategyName) -> SearchResponseDTO:
    expansion_data = side.get("expansion") or None
    expansion = (
        QueryExpansionDTO(
            original=expansion_data["original"],
            expanded=expansion_data["expanded"],
            model=expansion_data["model"],
        )
        if expansion_data
        else None
    )
    return SearchResponseDTO(
        strategy=strategy,
        query=side["query"],
        effectiveQuery=side["effective_query"],
        latencyMs=side["latency_ms"],
        expansion=expansion,
        hits=[
            QueryHitDTO(
                id=hit["id"],
                text=hit["text"],
                score=hit["score"],
                metadata=hit["metadata"],
            )
            for hit in side["hits"]
        ],
    )


__all__ = ["rag_router"]
