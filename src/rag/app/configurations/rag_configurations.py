"""Environment-driven configuration.

Pydantic-settings lets us read configuration from env vars (and a ``.env``
file) while still exposing strongly-typed accessors to the rest of the
service. All env-var names are prefixed with ``RAG_`` to avoid collisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMBackend = Literal["mock", "vertex"]
EmbeddingBackend = Literal["gecko-mock", "sentence-transformer"]


class RAGConfigurations(BaseSettings):
    """Service-wide config loaded once at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RAG_",
        extra="ignore",
    )

    # --- vector store ---
    chroma_persist_directory: Path = Field(
        default=Path("data/chroma"),
        description="Directory where ChromaDB persists collections.",
    )
    chroma_collection_name: str = Field(default="rag_chunks")

    # --- embeddings ---
    embedding_backend: EmbeddingBackend = Field(default="gecko-mock")
    embedding_model: str = Field(default="textembedding-gecko@003")
    sentence_transformer_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # --- LLM (Strategy B) ---
    llm_backend: LLMBackend = Field(default="mock")
    llm_model: str = Field(default="gemini-1.5-flash")
    gcp_project: str | None = Field(default=None, alias="GOOGLE_CLOUD_PROJECT")
    gcp_location: str = Field(default="us-central1", alias="GOOGLE_CLOUD_LOCATION")

    # --- chunking ---
    chunk_max_words: int = Field(default=220, ge=20, le=1024)
    chunk_overlap_words: int = Field(default=40, ge=0, le=512)

    # --- corpus paths ---
    corpus_path: Path = Field(default=Path("data/corpus/technical_paragraphs.json"))
    benchmark_queries_path: Path = Field(default=Path("data/corpus/benchmark_queries.json"))
    benchmark_output_md: Path = Field(default=Path("docs/retrieval_benchmark.md"))
    benchmark_output_json: Path = Field(default=Path("docs/retrieval_benchmark.json"))

    # --- API ---
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)


__all__ = ["EmbeddingBackend", "LLMBackend", "RAGConfigurations"]
