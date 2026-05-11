# Architecture & Design Decisions

This document answers two questions the assessment explicitly asks:

1. Choice of similarity metric — **cosine** vs **euclidean**.
2. How would we migrate this service to **Vertex AI Vector Search (Matching Engine)** in production.

It also captures the broader architectural decisions so a reviewer can read
the codebase top-down.

---

## 1. Components and how they fit together

```
            ┌──────────────────┐
 client ───▶│   FastAPI app    │
            │  /rag/ingest     │
            │  /rag/search     │     ┌─────────────────────────┐
            │  /rag/benchmark  │────▶│  AppContainer (DI)      │
            │  /rag/health     │     │  - embedding_service    │
            └──────────────────┘     │  - llm_client (Protocol)│
                                     │  - vector_store (ABC)   │
                                     │  - retrieval_service    │
                                     │  - ingestion_service    │
                                     └──────────┬──────────────┘
                                                │
                       ┌────────────────────────┼─────────────────────────┐
                       ▼                        ▼                         ▼
            ┌────────────────────┐  ┌────────────────────┐  ┌──────────────────────┐
            │ EmbeddingService   │  │  LLMClient (Proto) │  │   VectorStore (ABC)   │
            │ - GeckoMock (def.) │  │ - VertexGenMock(d) │  │ - ChromaVectorStore   │
            │ - SentenceTransfo. │  │ - VertexGenClient  │  │ - (Matching Engine)*  │
            └─────────┬──────────┘  └─────────┬──────────┘  └──────────┬───────────┘
                      ▼                       ▼                        ▼
       sentence-transformers      mocked rule-based               ChromaDB
       (all-MiniLM-L6-v2)         OR real Gemini 1.5 Flash        (PersistentClient)
```

`*` Matching Engine adapter is the migration target — see §4.

The split between `app/` (HTTP/DTO/DI wiring) and `lib/` (framework-agnostic
embedding / vector-store / LLM primitives) is borrowed from
`bigdata/onticai/semantic_search` and `bigdata/onticai/embedding/lib`.

---

## 2. Strategy A vs Strategy B — what they actually do

### Strategy A — Raw Vector Search

```
query ─▶ EmbeddingService.embed_one(query) ─▶ VectorStore.query(vector, top_k) ─▶ hits
```

Implemented by
[`RawQueryStrategy`](../src/app/services/retrieval_strategy.py).

### Strategy B — AI-Enhanced Retrieval

```
query
  ─▶ QueryExpansionService.expand(query)
       ─▶ LLMClient.generate(prompt)        # mocked Vertex Gemini
  ─▶ rewritten query
  ─▶ EmbeddingService.embed_one(rewritten)
  ─▶ VectorStore.query(vector, top_k)
  ─▶ hits
```

Implemented by
[`ExpandedQueryStrategy`](../src/app/services/retrieval_strategy.py).

The strategies share the same base class so the only difference is the
`_effective_query()` hook. This makes them trivially comparable — and lets
the benchmark runner treat them uniformly.

### Why query expansion helps

The curated benchmark queries are written in **conversational English**
("How do we know when something is broken in production?") while the corpus
uses **canonical engineering vocabulary** ("metrics, distributed traces,
observability"). Dense embeddings encode meaning, but the lexical mismatch
still pushes the cosine score down. Inserting domain synonyms via a
generative model closes that gap.

---

## 3. Cosine vs Euclidean similarity

We use **cosine similarity**, configured on the ChromaDB collection via
`metadata={"hnsw:space": "cosine"}`.

### Why cosine, not Euclidean

1. **Embeddings encode direction, not magnitude.** Models in the
   sentence-transformers family (and Vertex AI `textembedding-gecko`) are
   trained with contrastive objectives whose loss depends on the **angle**
   between vectors, not their length. The resulting vectors are deliberately
   length-uninformative — two sentences with the same meaning can land at
   different magnitudes depending on token count, while still pointing in the
   same direction.

2. **Euclidean distance is magnitude-sensitive.** `‖a − b‖` includes a
   contribution from `‖a‖ − ‖b‖`, which is noise for normalised text
   embeddings. Two paraphrases of the same idea can have a non-trivial L2
   distance simply because one is longer than the other.

3. **Cosine is bounded `[-1, 1]`**, which makes thresholds, fusion (RRF),
   and reranker scores comparable across queries. Euclidean is unbounded
   above and depends on the embedding model's scale.

4. **It's the de-facto industry choice for text retrieval.** OpenAI
   embeddings, Vertex AI `textembedding-gecko`, Cohere embed, and the BGE
   family all document cosine as the recommended metric. Matching this
   convention keeps the migration to Vertex AI seamless.

### Implementation note

`SentenceTransformerEmbeddingService` already L2-normalises its outputs (`normalize_embeddings=True`). Combined with `hnsw:space=cosine`, this means:

- The cosine score reported by ChromaDB is mathematically equivalent to the
  inner product of the normalised vectors.
- A FAISS migration would be a drop-in: use `IndexFlatIP` over the same
  normalised vectors and the rankings are bit-identical.

### When Euclidean **would** be right

- Vectors that encode position/geometry (computer-vision features pre-pool,
  embeddings of physical coordinates, etc.).
- Quantised embeddings where dequantisation reintroduces meaningful
  magnitude (e.g. some int8-PQ schemes). Even then, dot-product is usually
  preferred over L2.

For this assessment's corpus and embedding model, cosine is the correct
default and we keep it.

---

## 4. Vector DB comparison: why ChromaDB for this assessment

The assessment names FAISS, ChromaDB, and "a simple NumPy implementation" as
acceptable lightweight local stores. We scored four candidates against the
real constraints:

| Dimension | Weight | FAISS | **ChromaDB** | Qdrant (local) | Pinecone |
|---|---:|---:|---:|---:|---:|
| Matches "local + lightweight" rule | 3 | 5 | **5** | 3 | 0 (cloud-only) |
| Reviewer setup friction | 2 | 5 | **5** | 3 | 1 |
| Built-in cosine + metadata filter | 2 | 1 | **5** | 5 | 5 |
| Built-in persistence | 2 | 2 | **5** | 5 | 5 |
| Vertex AI Matching Engine migration story | 3 | 2 | **5** | 4 | 5 (SaaS↔SaaS) |
| Test determinism on tiny corpus | 1 | 5 | **4** | 4 | 1 |
| **Weighted score** | | 42 | **63** | 56 | 41 |

ChromaDB wins because:

1. The assessment names it explicitly.
2. `Collection` + `where`-clause metadata filtering + native cosine via
   `hnsw:space=cosine` ⇒ no hand-rolled normalisation/pickle plumbing.
3. The `Collection` model maps 1:1 onto Matching Engine's
   `Index` / `IndexEndpoint`, making the production migration write-up short
   and unambiguous.

Pinecone is disqualified by the "local" constraint. FAISS would have been a
fine second choice but loses ground on persistence and metadata. Qdrant is
what the [bigdata reference project](../../../code/bigdata/onticai/semantic_search)
uses, but it's a server-first DB and overkill for a 10-paragraph
assessment.

The `VectorStore` ABC means swapping any of these in later is a single
provider change in the DI container.

---

## 5. Migration to Vertex AI Vector Search (Matching Engine)

Production migration touches **three** wiring points; **no business logic
moves**.

### 5a. Embedding service

Replace
[`GeckoMockEmbeddingService`](../src/lib/embeddings/gecko_mock_service.py)
with a real Vertex client implementing the same `EmbeddingService` ABC:

```python
# Sketch: core/embeddings/vertex_embedding_service.py
from vertexai.language_models import TextEmbeddingModel

class VertexEmbeddingService(EmbeddingService):
    def __init__(self, model_name: str = "textembedding-gecko@003") -> None:
        self._model = TextEmbeddingModel.from_pretrained(model_name)

    def embed(self, texts):
        return [e.values for e in self._model.get_embeddings(list(texts))]

    def get_dimension(self) -> int:
        return 768  # textembedding-gecko
```

Wire it in by changing `_build_embedding_service` in
[`app_container.py`](../src/lib/containers/app_container.py).

### 5b. LLM client (Strategy B)

Already implemented:
[`VertexGenerativeClient`](../src/lib/llm/vertex_generative_client.py).
Toggle with `RAG_LLM_BACKEND=vertex` and install the optional extra:

```bash
uv pip install -e ".[dev,vertex]"
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=...
export GOOGLE_CLOUD_LOCATION=us-central1
export RAG_LLM_BACKEND=vertex
export RAG_LLM_MODEL=gemini-1.5-flash
```

### 5c. Vector store

Implement `MatchingEngineVectorStore(VectorStore)`. Pseudocode:

```python
# Sketch: core/vector_store/matching_engine_store.py
from google.cloud import aiplatform

class MatchingEngineVectorStore(VectorStore):
    def __init__(self, index_endpoint_id: str, deployed_index_id: str, ...):
        aiplatform.init(project=..., location=...)
        self._endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_id)
        self._deployed_index_id = deployed_index_id

    def upsert(self, records):
        # Use IndexUpsertDatapoints or batch upload via gcs.
        ...

    def query(self, embedding, top_k, where=None):
        matches = self._endpoint.find_neighbors(
            deployed_index_id=self._deployed_index_id,
            queries=[embedding],
            num_neighbors=top_k,
            filter=_to_namespace_filter(where),
        )
        return [_to_query_result(m) for m in matches[0]]

    def count(self): ...
    def reset(self): ...  # delete + recreate the index
```

Concept mapping ChromaDB ↔ Matching Engine:

| ChromaDB | Matching Engine |
|---|---|
| `Client` | `aiplatform.init(...)` |
| `Collection` | `Index` + `DeployedIndex` |
| `collection.upsert(ids, embeddings, metadatas)` | `IndexUpsertDatapoints` or GCS-based batch import |
| `collection.query(query_embeddings, n_results, where)` | `IndexEndpoint.find_neighbors(queries=, num_neighbors=, filter=)` |
| `metadata={"hnsw:space": "cosine"}` | `distance_measure_type="COSINE_DISTANCE"` on the index |
| `where={"doc_id": "x"}` | namespace filters / restricts on the datapoint |

### 5d. Ops checklist

1. **One-off**: create the index (specifying `dimensions=768` for gecko,
   `distanceMeasureType=COSINE_DISTANCE`) and deploy it to an endpoint.
2. Configure IAM: the service account running the FastAPI app needs
   `aiplatform.indexEndpoints.queryIndex` and
   `aiplatform.indexes.upsertDatapoints`.
3. Set `RAG_EMBEDDING_BACKEND` to the new Vertex backend, `RAG_LLM_BACKEND=vertex`,
   `VERTEX_AI_INDEX_ENDPOINT`, `VERTEX_AI_DEPLOYED_INDEX_ID`.
4. Re-run the benchmark — same harness, real backends — and commit the
   resulting `retrieval_benchmark.md` to compare against the local numbers.

### 5e. What stays the same

- All retrieval logic (`RetrievalService`, both strategies, the query
  expander, the benchmark runner, the metrics).
- All DTOs, the FastAPI surface, the OpenAPI schema.
- The test suite — every Vertex SDK call is behind a Protocol so the same
  pytest fixtures keep working with the real backend behind them.

That's the whole migration story: three small wrapper classes + container
wiring + a one-off `gcloud ai indexes create` command.
