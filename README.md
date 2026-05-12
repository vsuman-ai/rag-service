# RAG Service: Context-Aware Retrieval Engine

A local Retrieval-Augmented Generation pipeline that ingests a small technical
corpus, embeds it, stores the vectors in ChromaDB, and benchmarks two
retrieval strategies:

- **Strategy A — Raw Vector Search**: embed the user query verbatim and run a
  cosine top-k.
- **Strategy B — AI-Enhanced Retrieval**: send the query through a (mocked)
  Vertex AI Gemini model to rewrite/expand it with domain-specific vocabulary,
  then embed and search.

#The full assessment brief lives at
#[ docs/GenAI_RAG_VectorSearch_Assessment.pdf`](docs/GenAI_RAG_VectorSearch_Assessment.pdf).

---

## TL;DR — run the benchmark in one command

```bash
# 1. Create the venv and install everything (CPU, no GPU required).
uv venv --python 3.10 .venv
uv pip install -e ".[dev]"

# 2. Generate docs/retrieval_benchmark.md + docs/retrieval_benchmark.json
uv run python -m rag.benchmarks
```

Open [`docs/retrieval_benchmark.md`](docs/retrieval_benchmark.md) to see the
Strategy A vs Strategy B comparison the assessment asks for.

---

## Tech stack (locked by the assessment brief)

| Concern | Choice | Why |
|---|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) routed via a **mocked** `vertexai.language_models.TextEmbeddingModel` (`textembedding-gecko@003`) | Simulates Vertex AI gecko behaviour locally; CPU-friendly. |
| Strategy B LLM | **`gemini-1.5-flash` (mocked)** via `vertexai.generative_models.GenerativeModel` shim. Optional real Gemini 1.5 Flash client behind `RAG_LLM_BACKEND=vertex`. | Gemini Flash is the cheapest/fastest Vertex generative model and the natural successor for gecko-era pipelines. |
| Vector DB | **ChromaDB** (`PersistentClient`, `hnsw:space=cosine`) | See [docs/architecture.md](docs/architecture.md) for the FAISS / ChromaDB / Qdrant / Pinecone comparison. |
| Similarity metric | **Cosine** (`hnsw:space=cosine`) | Sentence-transformer outputs are L2-normalised; cosine ignores magnitude. See architecture doc. |
| API | FastAPI + `dependency-injector` | Same pattern used in the reference `bigdata/onticai/semantic_search` service. |
| Build | `uv` + `pyproject.toml` (PEP 517 / hatchling) | No pants, no monorepo build system. |
| Tests | `pytest` | Required by the assessment. |

---

## What is Strategy B doing? (which LLM, mocked how)

Strategy B follows this flow:

```
user query
  -> QueryExpansionService.expand(query)
       -> LLMClient.generate(prompt)        # depends only on a Protocol
            -> default: VertexGenerativeMock (mirrors vertexai.GenerativeModel)
            -> optional: VertexGenerativeClient -> gemini-1.5-flash (real Vertex AI)
  -> rewritten query
  -> EmbeddingService.embed_one(rewritten)
  -> ChromaVectorStore.query(...)
```

The default, used in tests and the committed benchmark report, is
`VertexGenerativeMock`. It is:

- Rule-based & deterministic (no sampling, no network).
- Drop-in compatible with `vertexai.generative_models.GenerativeModel`:
  exposes `__init__(model_name)`, `generate_content(prompt, generation_config)`,
  and the same `response.text` attribute.
- Injects SRE/infra synonyms (`autoscaling`, `horizontal scaling`,
  `load balancer`, `circuit breaker`, etc.) so the rewritten query's embedding
  lands closer to documents that use that vocabulary.

To swap in real Gemini 1.5 Flash, install the optional extra and switch the
backend:

```bash
uv pip install -e ".[dev,vertex]"
export GOOGLE_CLOUD_PROJECT=your-project
export GOOGLE_CLOUD_LOCATION=us-central1
gcloud auth application-default login
export RAG_LLM_BACKEND=vertex   # default is "mock"
uv run python -m rag.benchmarks
```

The application code does not change — only the DI container wiring resolves
to `VertexGenerativeClient` instead of `VertexGenerativeMock`.

---

## Repo layout

The repo ships **two installable packages** kept deliberately small so any
concrete backend (Chroma -> Vertex AI Vector Search, gecko-mock -> real
Vertex gecko, etc.) is a wiring change only:

- [`core/`](core/) -- framework-agnostic libraries (embeddings, LLM clients,
  vector store, chunking, DI container, utils).
- [`src/rag/`](src/rag/) -- the application: FastAPI service, DTOs, request
  controller, retrieval orchestrator + strategies, benchmarks, CLI.

```
rag_service/                       <-- repo
├── pyproject.toml                 # uv-managed, no pants
├── README.md
├── AGENTS.md                      # graphify rules for agentic dev loops
├── Dockerfile                     # multi-stage production image
├── docker-compose.yml             # one-shot local stack
├── core/                          # package `core` -- framework-agnostic libs
│   ├── chunking/
│   ├── containers/                # dependency-injector wiring
│   ├── embeddings/                # EmbeddingService ABC + impls (gecko mock + s-t)
│   ├── llm/                       # LLMClient Protocol + VertexGenerativeMock/Client
│   ├── utils/                     # logger
│   └── vector_store/              # VectorStore ABC + ChromaVectorStore
├── src/
│   └── rag/                       # package `rag` -- the application
│       ├── app/
│       │   ├── configurations/    # RAGConfigurations (pydantic-settings)
│       │   ├── constants/
│       │   ├── controller/        # FastAPI router
│       │   ├── dependencies/      # FastAPI ``Depends`` container resolver
│       │   ├── dto/               # Pydantic v2 request/response models
│       │   ├── services/          # ingestion, retrieval strategies, query expansion
│       │   ├── exception_handlers.py
│       │   └── server.py
│       ├── benchmarks/            # runner + metrics + reporting + CLI entry
│       └── cli.py                 # `rag-ingest` console script
├── data/corpus/                   # curated 10-paragraph corpus + benchmark queries
├── docs/
│   ├── architecture.md            # cosine vs euclidean + Vertex AI migration
│   ├── retrieval_benchmark.md     # generated by the benchmark CLI
│   └── retrieval_benchmark.json
└── tests/                         # pytest: unit + integration + api
```

---

## Common commands

```bash
# Set up
uv venv --python 3.10 .venv
uv pip install -e ".[dev]"

# Run the FastAPI server (http://localhost:8000/docs for Swagger UI)
uv run rag-service

# Ingest the curated corpus into ChromaDB
uv run rag-ingest

# Run the full Strategy A vs Strategy B benchmark
uv run python -m rag.benchmarks
uv run python -m rag.benchmarks --top-k 5
uv run python -m rag.benchmarks --corpus path/to/other.json --queries path/to/q.json

# Reuse the existing ChromaDB collection (skip re-ingest)
uv run python -m rag.benchmarks --skip-ingest

# Tests
uv run pytest                          # everything (incl. real-model integration tests)
uv run pytest -m "not integration"     # fast-only
uv run pytest --cov=core --cov=src/rag # with coverage on both packages

# Lint / format / type-check
uv run ruff check core src tests
uv run ruff format core src tests
uv run mypy core src

# Pre-commit hooks
uv run pre-commit install
uv run pre-commit run --all-files
```

### REST API

After `uv run rag-service`, the OpenAPI docs are at `http://localhost:8000/docs`.

```bash
# Ingest
curl -X POST localhost:8000/rag/ingest -H 'content-type: application/json' -d '{
  "documents": [{"docId": "d1", "text": "Autoscaling handles peak load."}],
  "replace": true
}'

# Search with Strategy A (raw)
curl -X POST localhost:8000/rag/search -H 'content-type: application/json' -d '{
  "query": "How does the system handle peak load?",
  "topK": 3,
  "strategy": "raw"
}'

# Search with Strategy B (expanded)
curl -X POST localhost:8000/rag/search -H 'content-type: application/json' -d '{
  "query": "How does the system handle peak load?",
  "topK": 3,
  "strategy": "expanded"
}'

# Run the full benchmark over the committed query set
curl -X POST localhost:8000/rag/benchmark -H 'content-type: application/json' -d '{"topK": 3}'
```

### Docker

```bash
# Build the production image.
docker build -t rag-service:0.1.0 .

# Run standalone (in-memory Chroma volume).
docker run --rm -p 8000:8000 rag-service:0.1.0
curl http://localhost:8000/health

# Or use docker compose (named volume keeps the vector store across runs).
docker compose up -d
curl http://localhost:8000/rag/health      # -> {"status":"ok","collection_size":0}

# Pick a different host port (the in-container port stays 8000).
RAG_HOST_PORT=18002 docker compose up -d
curl http://localhost:18002/health

# Tear down (use ``-v`` to also drop the persisted vector store).
docker compose down -v
```

---

## Migrating to Vertex AI Vector Search (Matching Engine)

Production migration is a wiring change in the DI container — no business
logic moves. The vector path:

1. **Embeddings**: swap `GeckoMockEmbeddingService` for a thin
   `VertexEmbeddingService` that wraps
   `vertexai.language_models.TextEmbeddingModel.from_pretrained("textembedding-gecko@003")`.
2. **LLM (Strategy B)**: set `RAG_LLM_BACKEND=vertex` and install
   `pip install -e ".[vertex]"`. `VertexGenerativeClient` is already wired in
   the container — see [`core/llm/vertex_generative_client.py`](src/lib/llm/vertex_generative_client.py).
3. **Vector store**: implement a `MatchingEngineVectorStore` satisfying the
   same `VectorStore` ABC and switch the `vector_store` provider in
   [`core/containers/app_container.py`](src/lib/containers/app_container.py).

Full migration commands:

```bash
# One-off: set up Application Default Credentials.
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=<your-project>
export GOOGLE_CLOUD_LOCATION=us-central1
export VERTEX_AI_INDEX_ENDPOINT=<your-index-endpoint-id>
export VERTEX_AI_DEPLOYED_INDEX_ID=<your-deployed-index-id>

# 1. Create a Matching Engine index from the local ChromaDB vectors.
gcloud ai indexes create \
  --display-name="rag-service-chunks" \
  --region=$GOOGLE_CLOUD_LOCATION \
  --metadata-file=ops/matching_engine_index.json

# 2. Deploy the index to an endpoint.
gcloud ai index-endpoints create \
  --display-name="rag-service-endpoint" \
  --region=$GOOGLE_CLOUD_LOCATION

gcloud ai index-endpoints deploy-index $VERTEX_AI_INDEX_ENDPOINT \
  --deployed-index-id=$VERTEX_AI_DEPLOYED_INDEX_ID \
  --display-name="rag-service-deployed" \
  --region=$GOOGLE_CLOUD_LOCATION \
  --index=<INDEX_ID_FROM_STEP_1>

# 3. Re-ingest documents through the Vertex-backed pipeline.
export RAG_EMBEDDING_BACKEND=gecko-mock  # or swap to a real Vertex embedding backend
export RAG_LLM_BACKEND=vertex
uv run rag-ingest --corpus data/corpus/technical_paragraphs.json

# 4. Re-run the benchmark against the real Vertex stack.
uv run python -m rag.benchmarks
```

See [`docs/architecture.md`](docs/architecture.md) for the full migration
write-up including the cosine-vs-euclidean discussion the assessment asks
for.

---

## Assessment requirements — where to find each answer

| Requirement (from the PDF) | Location |
|---|---|
| Embedding model via `sentence-transformers` | [`core/embeddings/sentence_transformer_service.py`](src/lib/embeddings/sentence_transformer_service.py) |
| Mock of `vertexai.language_models.TextEmbeddingModel` | [`core/embeddings/gecko_mock_service.py`](src/lib/embeddings/gecko_mock_service.py) |
| Mock of `vertexai.generative_models.GenerativeModel` | [`core/llm/vertex_generative_mock.py`](src/lib/llm/vertex_generative_mock.py) |
| Lightweight local vector store | [`core/vector_store/chroma_store.py`](src/lib/vector_store/chroma_store.py) |
| Orchestrator class that ingests text dataset | [`src/rag/app/services/ingestion_service.py`](src/app/services/ingestion_service.py) |
| Strategy A — Raw Vector Search | [`RawQueryStrategy`](src/app/services/retrieval_strategy.py) |
| Strategy B — AI-Enhanced Retrieval | [`ExpandedQueryStrategy`](src/app/services/retrieval_strategy.py) + [`QueryExpansionService`](src/app/services/query_expansion_service.py) |
| Comparison report (JSON + Table) | [`docs/retrieval_benchmark.md`](docs/retrieval_benchmark.md) + [`docs/retrieval_benchmark.json`](docs/retrieval_benchmark.json) |
| ≥ 3 complex queries benchmarked | [`data/corpus/benchmark_queries.json`](data/corpus/benchmark_queries.json) (6 queries, first 3 are "complex") |
| Pytest suites verifying retrieval + mocking the GCP SDK | [`tests/`](tests/) |
| Cosine vs Euclidean explanation | [`docs/architecture.md`](docs/architecture.md) |
| Migration to Vertex AI Vector Search | [`docs/architecture.md`](docs/architecture.md) + this README |

---

## Sample benchmark output

After running `uv run python -m rag.benchmarks` against the committed
corpus, the headline numbers (top-3, averaged over 6 queries) are:

| Metric @ k=3 | Strategy A (Raw) | Strategy B (Expanded) | Delta |
|---|---:|---:|---:|
| precision@k | 0.389 | 0.444 | **+0.056** |
| recall@k | 0.861 | 0.944 | **+0.083** |
| MRR | 1.000 | 1.000 | 0.000 |
| avg latency (ms) | 12.5 | 4.9 | -7.5 |

Strategy B (query expansion) lifts both precision and recall on this corpus
because the curated benchmark queries are intentionally phrased the way a
*user* would phrase them ("How do we know when something is broken?") while
the documents use the canonical engineering vocabulary ("metrics, traces,
observability"). The mocked LLM bridges that vocabulary gap.

---

## License

MIT
