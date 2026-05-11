## Multi-stage build for a slim production image.
##
## Stage 1 installs the package + deps into a venv using uv (fast, deterministic).
## Stage 2 copies the venv + source into a minimal python:slim image.

FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY core ./core

RUN uv venv /opt/venv \
    && . /opt/venv/bin/activate \
    && uv pip install --no-cache .

FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    RAG_CHROMA_PERSIST_DIRECTORY=/data/chroma

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash app \
    && mkdir -p /data/chroma /app \
    && chown -R app:app /data /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app data ./data
COPY --chown=app:app docs ./docs

# Declared so a docker-compose named volume mounted here inherits ``app`` ownership.
VOLUME ["/data/chroma"]

USER app
WORKDIR /app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# The ``rag`` and ``core`` packages are installed into /opt/venv at build
# time, so the runtime image does not need the source tree on disk.
CMD ["uvicorn", "rag.app.server:app", "--host", "0.0.0.0", "--port", "8000"]
