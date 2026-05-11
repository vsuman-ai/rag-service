from __future__ import annotations

from rag.benchmarks.metrics import (
    dedupe_preserve_order,
    jaccard_overlap,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], {"a", "c"}, k=3) == pytest_approx(2 / 3)
    assert precision_at_k(["a", "b", "c"], {"x"}, k=3) == 0.0
    assert precision_at_k([], {"a"}, k=3) == 0.0


def test_recall_at_k():
    assert recall_at_k(["a", "b"], {"a", "c"}, k=2) == 0.5
    assert recall_at_k(["a"], set(), k=1) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a", "y"], {"a"}) == 0.5
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_jaccard_overlap():
    assert jaccard_overlap(["a", "b"], ["a", "b"]) == 1.0
    assert jaccard_overlap(["a"], ["b"]) == 0.0
    assert jaccard_overlap([], []) == 1.0


def test_dedupe_preserve_order():
    assert dedupe_preserve_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def pytest_approx(value):
    import pytest

    return pytest.approx(value)
