"""Centralised FastAPI exception handling.

Same pattern as
``bigdata/onticai/semantic_search/src/app/exception_handlers.py`` minus the
Slack alerting: any uncaught exception is logged with full traceback and
surfaced to the caller as HTTP 500 ``{"detail": "Internal Server Error"}``.
``HTTPException`` is left to FastAPI's default handler.
"""

from __future__ import annotations

from core.utils.logger import logger
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


async def unhandled_exception_handler(request: Request, error: Exception) -> JSONResponse:
    title = f"{request.method} {request.url.path} failed"
    logger.opt(exception=error).error("[rag] {}: {}", title, error)
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(Exception, unhandled_exception_handler)


__all__ = ["register_exception_handlers", "unhandled_exception_handler"]
