"""Lightweight CLI shortcuts beyond the benchmark harness."""

from __future__ import annotations

from pathlib import Path

import typer
from core.containers.app_container import build_container
from core.utils.logger import logger

from rag.app.configurations.rag_configurations import RAGConfigurations
from rag.benchmarks.corpus_loader import load_corpus

ingest_app = typer.Typer(add_completion=False, no_args_is_help=False)


@ingest_app.command()
def run(
    corpus_path: Path | None = typer.Option(None, "--corpus"),
    replace: bool = typer.Option(True, help="Reset the collection before ingesting."),
) -> None:
    config = RAGConfigurations()
    resolved = corpus_path or config.corpus_path
    container = build_container(config)
    documents = load_corpus(resolved)
    summary = container.ingestion_service().ingest(documents=documents, replace=replace)
    logger.info("Ingested {} docs / {} chunks.", summary.documents_ingested, summary.chunks_indexed)


def ingest() -> None:
    """Entry point for the ``rag-ingest`` console script."""
    ingest_app()


__all__ = ["ingest"]
