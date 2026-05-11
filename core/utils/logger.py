"""Loguru-based structured logger shared across the service.

Centralised so request/test code can ``from core.utils.logger import logger``
without each module repeating the loguru configuration. Mirrors the pattern used in
``bigdata/onticai/embedding/lib/utils/logger.py``.
"""

from __future__ import annotations

import os
import sys

from loguru import logger as _logger

_LOG_LEVEL = os.getenv("RAG_LOG_LEVEL", "INFO").upper()


def _configure_logger() -> None:
    _logger.remove()
    _logger.add(
        sys.stderr,
        level=_LOG_LEVEL,
        backtrace=False,
        diagnose=False,
        enqueue=False,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )


_configure_logger()

logger = _logger

__all__ = ["logger"]
