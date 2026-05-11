from __future__ import annotations

from core.vector_store.chroma_store import ChromaVectorStore
from core.vector_store.models import VectorRecord


def _record(idx: int, embedding: list[float]) -> VectorRecord:
    return VectorRecord(
        id=f"id-{idx}",
        text=f"text-{idx}",
        embedding=embedding,
        metadata={"doc_id": f"doc-{idx}"},
    )


def test_upsert_then_count(chroma_store: ChromaVectorStore) -> None:
    assert chroma_store.count() == 0
    chroma_store.upsert([_record(1, [1.0, 0.0, 0.0]), _record(2, [0.0, 1.0, 0.0])])
    assert chroma_store.count() == 2


def test_query_returns_results_sorted_by_score(chroma_store: ChromaVectorStore) -> None:
    chroma_store.upsert(
        [
            _record(1, [1.0, 0.0, 0.0]),
            _record(2, [0.0, 1.0, 0.0]),
            _record(3, [0.0, 0.0, 1.0]),
        ]
    )
    results = chroma_store.query([1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].id == "id-1"
    # Scores are non-increasing.
    assert results[0].score >= results[1].score
    # Top hit is near-perfect cosine.
    assert results[0].score > 0.99


def test_upsert_is_idempotent_on_id(chroma_store: ChromaVectorStore) -> None:
    chroma_store.upsert([_record(1, [1.0, 0.0])])
    chroma_store.upsert([_record(1, [1.0, 0.0])])
    assert chroma_store.count() == 1


def test_reset_clears_collection(chroma_store: ChromaVectorStore) -> None:
    chroma_store.upsert([_record(1, [1.0, 0.0])])
    chroma_store.reset()
    assert chroma_store.count() == 0


def test_query_top_k_zero_returns_empty(chroma_store: ChromaVectorStore) -> None:
    chroma_store.upsert([_record(1, [1.0, 0.0])])
    assert chroma_store.query([1.0, 0.0], top_k=0) == []
