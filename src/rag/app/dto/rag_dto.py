"""Request/response DTOs for the FastAPI surface.

Pydantic v2 models -- camelCase aliases on the wire, snake_case in Python,
matching the convention used in
``bigdata/onticai/semantic_search/src/app/dto/semantic_search_dto.py``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rag.app.constants.strategy_constants import RetrievalStrategyName


class _CamelModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


# ---------- Ingest ----------


class IngestDocumentDTO(_CamelModel):
    doc_id: str = Field(alias="docId")
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequestDTO(_CamelModel):
    documents: list[IngestDocumentDTO]
    replace: bool = Field(default=False)


class IngestResponseDTO(_CamelModel):
    documents_ingested: int = Field(alias="documentsIngested")
    chunks_indexed: int = Field(alias="chunksIndexed")
    collection_size: int = Field(alias="collectionSize")


# ---------- Search ----------


class SearchRequestDTO(_CamelModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=50, alias="topK")
    strategy: RetrievalStrategyName = Field(default=RetrievalStrategyName.RAW)
    where: dict[str, Any] | None = Field(default=None)


class QueryHitDTO(_CamelModel):
    id: str
    text: str
    score: float
    metadata: dict[str, Any]


class QueryExpansionDTO(_CamelModel):
    original: str
    expanded: str
    model: str


class SearchResponseDTO(_CamelModel):
    strategy: RetrievalStrategyName
    query: str
    effective_query: str = Field(alias="effectiveQuery")
    latency_ms: float = Field(alias="latencyMs")
    expansion: QueryExpansionDTO | None = None
    hits: list[QueryHitDTO]


# ---------- Benchmark ----------


class BenchmarkRequestDTO(_CamelModel):
    """Optional payload to override the benchmark queries baked into the repo."""

    queries: list[str] | None = None
    top_k: int = Field(default=3, ge=1, le=50, alias="topK")


class BenchmarkComparisonDTO(_CamelModel):
    query: str
    raw: SearchResponseDTO
    expanded: SearchResponseDTO
    overlap_at_k: float = Field(alias="overlapAtK")


class BenchmarkResponseDTO(_CamelModel):
    top_k: int = Field(alias="topK")
    comparisons: list[BenchmarkComparisonDTO]
    summary: dict[str, Any]


__all__ = [
    "BenchmarkComparisonDTO",
    "BenchmarkRequestDTO",
    "BenchmarkResponseDTO",
    "IngestDocumentDTO",
    "IngestRequestDTO",
    "IngestResponseDTO",
    "QueryExpansionDTO",
    "QueryHitDTO",
    "SearchRequestDTO",
    "SearchResponseDTO",
]
