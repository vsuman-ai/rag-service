"""FastAPI entrypoint.

Wires the DI container once at startup and exposes the /rag/* router. Run
locally with::

    uv run uvicorn rag.app.server:app --reload

or via the console script::

    uv run rag-service
"""

from __future__ import annotations

import uvicorn
from core.containers.app_container import build_container
from core.utils.logger import logger
from fastapi import FastAPI

from rag.app.configurations.rag_configurations import RAGConfigurations
from rag.app.controller.rag_controller import rag_router
from rag.app.exception_handlers import register_exception_handlers


def create_app(config: RAGConfigurations | None = None) -> FastAPI:
    """Application factory -- preferred over a module-level ``app`` for tests."""
    settings = config or RAGConfigurations()
    application = FastAPI(
        title="RAG Service: Context-Aware Retrieval Engine",
        version="0.1.0",
        description=(
            "Local Retrieval-Augmented Generation pipeline that benchmarks "
            "Raw vector search (Strategy A) vs AI-Enhanced (query expansion) "
            "retrieval (Strategy B)."
        ),
    )
    application.state.config = settings
    application.state.container = build_container(settings)
    register_exception_handlers(application)
    application.include_router(rag_router)

    @application.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()


def main() -> None:
    settings = RAGConfigurations()
    logger.info("Starting RAG service on {}:{}.", settings.host, settings.port)
    uvicorn.run(
        "rag.app.server:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()


__all__ = ["app", "create_app", "main"]
