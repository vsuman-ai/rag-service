"""Vector-store DTOs shared between the storage layer and services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VectorRecord:
    """A single document/chunk ready to be upserted into the vector store.

    ``embedding`` is plain ``list[float]`` (not numpy) so the dataclass remains
    hashable/serialisable; the vector store implementation is responsible for
    converting to the format its client expects.
    """

    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryResult:
    """One hit returned by the vector store."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any]


__all__ = ["QueryResult", "VectorRecord"]
