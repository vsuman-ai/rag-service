## graphify

This project has a graphify knowledge graph at `graphify-out/`.

Rules:
- Before answering architecture or codebase questions, read
  `graphify-out/GRAPH_REPORT.md` for god nodes and community structure.
- If `graphify-out/wiki/index.md` exists, navigate it instead of reading raw
  files.
- For cross-module "how does X relate to Y" questions, prefer
  `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or
  `graphify explain "<concept>"` over `grep` — these traverse the graph's
  EXTRACTED + INFERRED edges instead of scanning files.
- After modifying code files in this session, run `graphify update .` to keep
  the graph current (AST-only, no API cost).

## Conventions

- **No pants.** This repo uses `uv` + `pyproject.toml` only.
- **Two packages**: `core/` (framework-agnostic libraries -- embeddings,
  vector store, LLM clients, chunking, DI container) and `src/rag/`
  (FastAPI app, controllers, DTOs, services, benchmarks, CLI). Don't import
  FastAPI / HTTP concerns into `core/`.
- **Every external SDK is behind a Protocol or ABC** (`EmbeddingService`,
  `VectorStore`, `LLMClient`). Production code never imports a concrete
  backend directly — only the DI container does.
- **Tests use pytest.** Mark tests that load real models with
  `@pytest.mark.integration` so the default `pytest -m "not integration"`
  stays fast.
