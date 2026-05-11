"""End-to-end pipeline test on the real curated corpus.

Uses fake embeddings (keyword bag-of-axes) and fake LLM rewrites so the test
runs in <1s without downloading any models, while still exercising every
production code path: chunker -> embedder -> ChromaDB -> strategy -> runner.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.chunking.paragraph_chunker import ParagraphChunker
from core.vector_store.chroma_store import ChromaVectorStore

from rag.app.services.ingestion_service import IngestionService
from rag.app.services.query_expansion_service import QueryExpansionService
from rag.app.services.retrieval_service import RetrievalService
from rag.app.services.retrieval_strategy import (
    ExpandedQueryStrategy,
    RawQueryStrategy,
)
from rag.benchmarks.corpus_loader import load_benchmark_queries, load_corpus
from rag.benchmarks.reporting import write_reports
from rag.benchmarks.runner import BenchmarkRunner
from tests.conftest import FakeEmbeddingService, FakeLLMClient


def _pipeline(temp_dir: Path):
    embedder = FakeEmbeddingService()
    store = ChromaVectorStore(
        collection_name="bench_test",
        persist_directory=temp_dir / "chroma",
    )
    chunker = ParagraphChunker()
    ingest = IngestionService(chunker, embedder, store)
    documents = load_corpus(Path("data/corpus/technical_paragraphs.json"))
    ingest.ingest(documents, replace=True)

    rewriter = QueryExpansionService(
        FakeLLMClient(rewrite="autoscaling horizontal scaling load balancer")
    )
    raw = RawQueryStrategy(embedder, store)
    expanded = ExpandedQueryStrategy(embedder, store, rewriter)
    service = RetrievalService(raw_strategy=raw, expanded_strategy=expanded)
    return service


def test_benchmark_pipeline_produces_complete_report(tmp_path: Path) -> None:
    service = _pipeline(tmp_path)
    queries = load_benchmark_queries(Path("data/corpus/benchmark_queries.json"))
    assert len(queries) >= 3
    runner = BenchmarkRunner(retrieval_service=service)
    report = runner.run(queries=queries, top_k=3)

    assert report["top_k"] == 3
    assert len(report["comparisons"]) == len(queries)
    for comparison in report["comparisons"]:
        for side in ("raw", "expanded"):
            assert "metrics" in comparison[side]
            assert "retrieved_doc_ids" in comparison[side]
            assert len(comparison[side]["hits"]) <= 3
            assert all("doc_id" in hit for hit in comparison[side]["hits"])

    summary = report["summary"]
    assert summary["queries"] == len(queries)
    assert "raw" in summary and "expanded" in summary


def test_reporting_writes_md_and_json(tmp_path: Path) -> None:
    service = _pipeline(tmp_path)
    queries = load_benchmark_queries(Path("data/corpus/benchmark_queries.json"))
    runner = BenchmarkRunner(retrieval_service=service)
    report = runner.run(queries=queries, top_k=3)

    md_path = tmp_path / "bench.md"
    json_path = tmp_path / "bench.json"
    write_reports(report, md_path, json_path)

    md_text = md_path.read_text(encoding="utf-8")
    assert "Strategy A" in md_text and "Strategy B" in md_text
    assert "Per-query comparison" in md_text

    json_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_payload["top_k"] == 3
    assert len(json_payload["comparisons"]) == len(queries)


def test_strategy_b_beats_or_matches_strategy_a_on_expanded_query(tmp_path: Path) -> None:
    """The canonical assessment query: Strategy B should help, not hurt."""
    service = _pipeline(tmp_path)
    runner = BenchmarkRunner(retrieval_service=service)
    report = runner.run(
        queries=[
            {
                "query": "How does the system handle peak load?",
                "relevant_doc_ids": [
                    "doc-autoscaling",
                    "doc-load-balancing",
                    "doc-rate-limiting",
                ],
            }
        ],
        top_k=3,
    )
    comparison = report["comparisons"][0]
    raw_recall = comparison["raw"]["metrics"]["recall_at_k"]
    expanded_recall = comparison["expanded"]["metrics"]["recall_at_k"]
    # The fake LLM rewrite explicitly injects "autoscaling horizontal scaling
    # load balancer" -> Strategy B must recall at least as much as A here.
    assert expanded_recall >= raw_recall
