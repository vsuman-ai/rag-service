"""ChromaDB-backed implementation of :class:`VectorStore`.

Why ChromaDB:
  * Named in the assessment as an acceptable lightweight local store.
  * Built-in cosine similarity via ``hnsw:space=cosine`` collection metadata.
  * Built-in ``PersistentClient`` so vectors survive a service restart.
  * ``Collection`` maps 1:1 onto Vertex AI Matching Engine's ``Index`` /
    ``IndexEndpoint`` for the production migration story.

The class only depends on :mod:`chromadb` and the local :mod:`models` DTOs --
no leakage of chroma types past this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.config import Settings

from core.utils.logger import logger

from .models import QueryResult, VectorRecord
from .vector_store import VectorStore

_COSINE_SPACE = "cosine"


class ChromaVectorStore(VectorStore):
    """Embedded ChromaDB collection storing one ``VectorRecord`` per row."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str | Path | None = None,
    ) -> None:
        self._collection_name = collection_name
        self._persist_directory = (
            Path(persist_directory).resolve() if persist_directory else None
        )
        self._client = self._build_client()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": _COSINE_SPACE},
        )
        logger.debug(
            "ChromaVectorStore ready (collection={}, persist={})",
            collection_name,
            self._persist_directory or "in-memory",
        )

    def _build_client(self) -> chromadb.api.ClientAPI:
        settings = Settings(anonymized_telemetry=False, allow_reset=True)
        if self._persist_directory is None:
            return chromadb.EphemeralClient(settings=settings)
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(
            path=str(self._persist_directory), settings=settings
        )

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        ids = [record.id for record in records]
        embeddings = [list(record.embedding) for record in records]
        documents = [record.text for record in records]
        metadatas = [_sanitise_metadata(record.metadata) for record in records]
        self._collection.upsert(
            ids=ids,
            embeddings=cast(Any, embeddings),
            documents=documents,
            metadatas=cast(Any, metadatas),
        )
        logger.info("Upserted {} records into '{}'.", len(records), self._collection_name)

    def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        where: dict[str, object] | None = None,
    ) -> list[QueryResult]:
        if top_k <= 0:
            return []
        result = self._collection.query(
            query_embeddings=cast(Any, [list(embedding)]),
            n_results=top_k,
            where=cast(Any, where or None),
        )
        return _to_query_results(cast(dict[str, Any], result))

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        # Recreate the collection — cheaper and safer than ``client.reset()``
        # which wipes the entire persist directory.
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": _COSINE_SPACE},
        )
        logger.warning("Collection '{}' has been reset.", self._collection_name)

    def close(self) -> None:
        # chromadb manages resources internally; nothing explicit to release.
        return


def _sanitise_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Chroma only accepts scalar metadata values; coerce lists/dicts to str."""
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def _to_query_results(raw: dict[str, Any]) -> list[QueryResult]:
    ids = (raw.get("ids") or [[]])[0]
    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0] or [{} for _ in ids]
    distances = (raw.get("distances") or [[]])[0]
    results: list[QueryResult] = []
    for idx, doc_id in enumerate(ids):
        distance = float(distances[idx]) if idx < len(distances) else 0.0
        # Chroma returns ``1 - cosine_similarity``; convert back so higher == better.
        similarity = max(0.0, 1.0 - distance)
        results.append(
            QueryResult(
                id=str(doc_id),
                text=str(documents[idx]) if idx < len(documents) else "",
                score=similarity,
                metadata=dict(metadatas[idx] or {}),
            )
        )
    return results


__all__ = ["ChromaVectorStore"]
