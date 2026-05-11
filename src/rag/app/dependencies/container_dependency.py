"""FastAPI dependency that exposes the shared :class:`AppContainer`.

The container is created once at app startup and stored on ``app.state`` --
each request reads it via this dependency. Mirrors
``bigdata/onticai/semantic_search/src/app/dependencies/container_dependency.py``
but without the per-request resource init since our resources are pure
singletons.
"""

from __future__ import annotations

from core.containers.app_container import AppContainer
from fastapi import Request


def get_container(request: Request) -> AppContainer:
    container: AppContainer | None = getattr(request.app.state, "container", None)
    if container is None:
        raise RuntimeError("AppContainer is not attached to FastAPI app.state.")
    return container


__all__ = ["get_container"]
