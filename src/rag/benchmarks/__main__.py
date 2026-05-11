"""CLI: ``python -m rag.benchmarks``.

End-to-end:

  1. Build the DI container from environment configuration.
  2. Ingest the curated corpus (replacing the existing collection).
  3. Run Strategy A & B across the benchmark queries.
  4. Write Markdown + JSON reports to the configured paths.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from core.containers.app_container import build_container
from core.utils.logger import logger

from rag.app.configurations.rag_configurations import RAGConfigurations
from rag.benchmarks.corpus_loader import load_benchmark_queries, load_corpus
from rag.benchmarks.reporting import write_reports
from rag.benchmarks.runner import BenchmarkRunner

app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def run(
    corpus_path: Path | None = typer.Option(None, "--corpus", help="Path to corpus JSON."),
    queries_path: Path | None = typer.Option(
        None, "--queries", help="Path to benchmark queries JSON."
    ),
    top_k: int = typer.Option(3, "--top-k", min=1, max=50),
    md_path: Path | None = typer.Option(
        None, "--md", help="Markdown output path (default: docs/retrieval_benchmark.md)."
    ),
    json_path: Path | None = typer.Option(
        None, "--json", help="JSON output path (default: docs/retrieval_benchmark.json)."
    ),
    skip_ingest: bool = typer.Option(False, help="Reuse the existing collection."),
) -> None:
    """Run the full Strategy A vs Strategy B benchmark."""
    config = RAGConfigurations()
    resolved_corpus = corpus_path or config.corpus_path
    resolved_queries = queries_path or config.benchmark_queries_path
    resolved_md = md_path or config.benchmark_output_md
    resolved_json = json_path or config.benchmark_output_json

    container = build_container(config)

    if not skip_ingest:
        logger.info("Ingesting corpus from {}.", resolved_corpus)
        documents = load_corpus(resolved_corpus)
        container.ingestion_service().ingest(documents=documents, replace=True)
    else:
        logger.info("Skipping ingest -- reusing existing collection.")

    logger.info("Loading benchmark queries from {}.", resolved_queries)
    queries = load_benchmark_queries(resolved_queries)
    if not queries:
        typer.echo("No benchmark queries found.", err=True)
        raise typer.Exit(code=2)

    runner = BenchmarkRunner(retrieval_service=container.retrieval_service())
    report = runner.run(queries=queries, top_k=top_k)

    write_reports(report, md_path=resolved_md, json_path=resolved_json)
    logger.info("Wrote Markdown report to {}.", resolved_md)
    logger.info("Wrote JSON report to {}.", resolved_json)

    summary = report["summary"]
    typer.echo(
        "\n=== Benchmark summary (mean across {} queries, top-{}): ===".format(
            summary["queries"], summary["top_k"]
        )
    )
    typer.echo(json.dumps(summary, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
    sys.exit(0)
