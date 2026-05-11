"""Retrieval metrics used by the benchmark harness.

All metrics operate on document-id lists (not chunk-id lists) so that
multiple chunks from the same document do not over-credit a strategy.
"""

from __future__ import annotations

from collections.abc import Sequence


def precision_at_k(retrieved_doc_ids: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def recall_at_k(retrieved_doc_ids: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved_doc_ids: Sequence[str], relevant: set[str]) -> float:
    for index, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def jaccard_overlap(a: Sequence[str], b: Sequence[str]) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def dedupe_preserve_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


__all__ = [
    "dedupe_preserve_order",
    "jaccard_overlap",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
