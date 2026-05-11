"""FastAPI route tests using TestClient.

These tests build the application with the fake embedding + LLM overrides so
no model download or network call is made -- satisfying the assessment's
"mock the GCP SDK" requirement at the API layer too.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag.app.configurations.rag_configurations import RAGConfigurations
from rag.app.server import create_app
from tests.conftest import FakeEmbeddingService, FakeLLMClient


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    config = RAGConfigurations(
        chroma_persist_directory=tmp_path / "chroma",
        chroma_collection_name="api_test",
    )
    application = create_app(config)
    container = application.state.container
    container.embedding_service.override(FakeEmbeddingService())
    container.llm_client.override(FakeLLMClient(rewrite="autoscaling horizontal scaling"))
    return TestClient(application)


def _ingest(client: TestClient) -> None:
    payload = {
        "documents": [
            {
                "docId": "doc-autoscaling",
                "text": "Autoscaling adds replicas via horizontal scaling.",
                "metadata": {"topic": "scaling"},
            },
            {
                "docId": "doc-cache",
                "text": "Caching with LRU and redis prevents stampedes.",
                "metadata": {"topic": "caching"},
            },
        ],
        "replace": True,
    }
    response = client.post("/rag/ingest", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["documentsIngested"] == 2
    assert body["chunksIndexed"] == 2


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/rag/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ingest_then_search_raw(client: TestClient) -> None:
    _ingest(client)
    response = client.post(
        "/rag/search",
        json={"query": "autoscaling", "topK": 2, "strategy": "raw"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["strategy"] == "raw"
    assert body["expansion"] is None
    assert body["hits"][0]["metadata"]["doc_id"] == "doc-autoscaling"


def test_ingest_then_search_expanded(client: TestClient) -> None:
    _ingest(client)
    response = client.post(
        "/rag/search",
        json={
            "query": "how does the system handle peak load?",
            "topK": 2,
            "strategy": "expanded",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["strategy"] == "expanded"
    assert body["expansion"] is not None
    assert body["expansion"]["expanded"] == "autoscaling horizontal scaling"
    assert body["effectiveQuery"] == "autoscaling horizontal scaling"


def test_ingest_rejects_empty_payload(client: TestClient) -> None:
    response = client.post("/rag/ingest", json={"documents": []})
    assert response.status_code == 400


def test_benchmark_endpoint_returns_comparisons(client: TestClient) -> None:
    _ingest(client)
    response = client.post(
        "/rag/benchmark",
        json={"queries": ["autoscaling", "caching"], "topK": 2},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["topK"] == 2
    assert len(body["comparisons"]) == 2
    first = body["comparisons"][0]
    assert "raw" in first and "expanded" in first
    assert "overlapAtK" in first
