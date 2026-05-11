"""Loaders for the JSON corpus and benchmark-queries files.

Kept tiny on purpose -- the data files are the source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag.app.services.ingestion_service import IngestDocument


def load_corpus(path: str | Path) -> list[IngestDocument]:
    """Parse ``technical_paragraphs.json`` into ``IngestDocument`` list."""
    raw = _read_json(path)
    documents = []
    for entry in raw.get("documents", []):
        documents.append(
            IngestDocument(
                doc_id=entry["doc_id"],
                text=entry["text"],
                metadata={"topic": entry.get("topic", "")},
            )
        )
    return documents


def load_benchmark_queries(path: str | Path) -> list[dict[str, Any]]:
    """Parse ``benchmark_queries.json`` into a list of query dicts."""
    raw = _read_json(path)
    return list(raw.get("queries", []))


def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


__all__ = ["load_benchmark_queries", "load_corpus"]
