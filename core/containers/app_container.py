"""Dependency-injector container.

Wires every concrete implementation behind the ABCs / Protocols used by the
service. Migrating to Vertex AI (gecko embeddings + Gemini + Matching Engine)
is purely a change of the providers in this file.

Pattern follows ``bigdata/onticai/embedding/lib/containers/app_container.py``.
"""

from __future__ import annotations

from dependency_injector import containers, providers

from core.chunking.paragraph_chunker import ParagraphChunker
from core.embeddings.embedding_service import EmbeddingService
from core.embeddings.gecko_mock_service import GeckoMockEmbeddingService
from core.embeddings.sentence_transformer_service import (
    SentenceTransformerEmbeddingService,
)
from core.llm.llm_client import LLMClient
from core.llm.vertex_generative_mock import VertexGenerativeMock
from core.vector_store.chroma_store import ChromaVectorStore
from core.vector_store.vector_store import VectorStore
from rag.app.configurations.rag_configurations import RAGConfigurations
from rag.app.services.ingestion_service import IngestionService
from rag.app.services.query_expansion_service import QueryExpansionService
from rag.app.services.retrieval_service import RetrievalService
from rag.app.services.retrieval_strategy import (
    ExpandedQueryStrategy,
    RawQueryStrategy,
)


def _build_embedding_service(config: RAGConfigurations) -> EmbeddingService:
    if config.embedding_backend == "sentence-transformer":
        return SentenceTransformerEmbeddingService(model_name=config.sentence_transformer_model)
    return GeckoMockEmbeddingService(gecko_model_name=config.embedding_model)


def _build_llm_client(config: RAGConfigurations) -> LLMClient:
    if config.llm_backend == "vertex":
        # Lazy import so unit tests don't need google-cloud-aiplatform installed.
        from core.llm.vertex_generative_client import VertexGenerativeClient

        return VertexGenerativeClient(
            model_name=config.llm_model,
            project=config.gcp_project,
            location=config.gcp_location,
        )
    return VertexGenerativeMock(model_name=f"{config.llm_model}-mock")


def _build_vector_store(config: RAGConfigurations) -> VectorStore:
    return ChromaVectorStore(
        collection_name=config.chroma_collection_name,
        persist_directory=config.chroma_persist_directory,
    )


def _build_chunker(config: RAGConfigurations) -> ParagraphChunker:
    return ParagraphChunker(
        max_words=config.chunk_max_words,
        overlap_words=config.chunk_overlap_words,
    )


class AppContainer(containers.DeclarativeContainer):
    """Wire-once, inject everywhere."""

    config = providers.Singleton(RAGConfigurations)

    embedding_service: providers.Provider[EmbeddingService] = providers.Singleton(
        _build_embedding_service, config=config
    )

    llm_client: providers.Provider[LLMClient] = providers.Singleton(
        _build_llm_client, config=config
    )

    vector_store: providers.Provider[VectorStore] = providers.Singleton(
        _build_vector_store, config=config
    )

    chunker = providers.Singleton(_build_chunker, config=config)

    query_expansion_service = providers.Singleton(
        QueryExpansionService, llm_client=llm_client
    )

    raw_strategy = providers.Singleton(
        RawQueryStrategy,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    expanded_strategy = providers.Singleton(
        ExpandedQueryStrategy,
        embedding_service=embedding_service,
        vector_store=vector_store,
        query_expansion_service=query_expansion_service,
    )

    retrieval_service = providers.Singleton(
        RetrievalService,
        raw_strategy=raw_strategy,
        expanded_strategy=expanded_strategy,
    )

    ingestion_service = providers.Singleton(
        IngestionService,
        chunker=chunker,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )


def build_container(config: RAGConfigurations | None = None) -> AppContainer:
    """Convenience factory used by the CLI / benchmark / FastAPI startup."""
    container = AppContainer()
    if config is not None:
        container.config.override(providers.Object(config))
    return container


__all__ = ["AppContainer", "build_container"]
