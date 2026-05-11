from __future__ import annotations

from core.chunking.paragraph_chunker import ParagraphChunker

from rag.app.constants.strategy_constants import RetrievalStrategyName
from rag.app.services.ingestion_service import IngestDocument, IngestionService
from rag.app.services.query_expansion_service import QueryExpansionService
from rag.app.services.retrieval_strategy import (
    ExpandedQueryStrategy,
    RawQueryStrategy,
)
from tests.conftest import FakeLLMClient


def _seed(store, embedding_service):
    chunker = ParagraphChunker()
    ingest = IngestionService(chunker, embedding_service, store)
    ingest.ingest(
        [
            IngestDocument(
                doc_id="doc-autoscaling",
                text="Autoscaling adds replicas and horizontal scaling.",
                metadata={},
            ),
            IngestDocument(
                doc_id="doc-cache",
                text="Caching with LRU and redis avoids stampedes.",
                metadata={},
            ),
        ]
    )


def test_raw_strategy_returns_results_and_no_expansion(chroma_store, fake_embedding_service):
    _seed(chroma_store, fake_embedding_service)
    strategy = RawQueryStrategy(fake_embedding_service, chroma_store)
    result = strategy.search("autoscaling replicas", top_k=2)
    assert result.strategy is RetrievalStrategyName.RAW
    assert result.expansion is None
    assert result.results
    assert result.results[0].metadata["doc_id"] == "doc-autoscaling"
    assert result.latency_ms >= 0.0


def test_expanded_strategy_invokes_llm_and_uses_rewrite(
    chroma_store, fake_embedding_service, fake_llm_client
):
    _seed(chroma_store, fake_embedding_service)
    rewrite = "autoscaling horizontal scaling load balancer"
    fake_llm = FakeLLMClient(rewrite=rewrite)
    expander = QueryExpansionService(fake_llm)
    strategy = ExpandedQueryStrategy(fake_embedding_service, chroma_store, expander)
    result = strategy.search("how does the system handle peak load?", top_k=2)
    assert result.strategy is RetrievalStrategyName.EXPANDED
    assert result.expansion is not None
    assert result.expansion.expanded == rewrite
    assert result.effective_query == rewrite
    assert result.results[0].metadata["doc_id"] == "doc-autoscaling"
