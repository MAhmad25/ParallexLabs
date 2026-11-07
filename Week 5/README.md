# Week 5 — Evaluation & API (FastAPI)

**November 2–8, 2026 · Python 3.12 · uv · FastAPI · pytest · ChromaDB**

I wrapped the Week 4 RAG retrieval system in a FastAPI application with three
endpoints, structured JSON request logging, and proper HTTP error responses
(400, 422, 500). I wrote a retrieval evaluation script that computes
Precision@K and Recall@K on a six-query gold-standard test set, an automated
end-to-end evaluation script that exercises every endpoint and measures latency
percentiles (p50, p95, p99), and a pytest test suite covering all endpoints
including validation, filter correctness, and error handling.

The service reads the Week 4 ChromaDB index directly — no data is duplicated.
Query embedding uses the same Gemini model and dimensions as the indexed
vectors. Metadata filters (topic ID, topic label, sentiment) are combined with
AND and passed to Chroma before cosine ranking.

## What I built

| Task | Implementation |
| --- | --- |
| FastAPI service | Three endpoints: `/health`, `/query`, `/collection/stats` |
| Structured logging | JSON request log middleware writing to `logs/requests.jsonl` with `request_id`, timing, and status |
| HTTP errors | 400 for bad values, 422 for schema violations, 500 for internal failures |
| Retrieval evaluation | Precision@K and Recall@K over a gold JSONL test set |
| End-to-end evaluation | Schema validation, filter correctness, latency p50/p95/p99 |
| Unit tests | 22 pytest cases using mocked embedder and in-memory Chroma collection |

## Current run results

| Metric / Check | Value | Notes |
| --- | --- | --- |
| Unit test suite | **22 / 22 passed** (1.80s) | Mocks embedder & collection; runs completely offline |
| Corpus chunk count | **353 chunks** | Sourced directly from Week 4 ChromaDB index |
| Embedding model | `gemini-embedding-001` (768 dims) | Consistent with Week 4 index identity |
| Retrieval queries | 6 gold queries (3 relevant chunks each) | Anarchism, Albedo, Alabama, Lincoln, Oscars, Achilles |
| Retrieval **Mean Precision@5** | **53.3%** (0.5333) | Top-5 retrieval precision on gold relevant chunks |
| Retrieval **Mean Recall@5** | **88.9%** (0.8889) | 4 of 6 queries achieved 100% recall@5 |
| Retrieval latency | Mean: **580ms** per query | End-to-end embedding + Chroma cosine ranking |
| E2E test queries | **10 / 10 passed** (0 issues) | Schema valid, hit counts match, filters exact |
| E2E latency percentiles | Mean: **779.4ms** · p50: **717.5ms** · p95: **1504.2ms** · p99: **1504.2ms** | Exercised over HTTP via live FastAPI app |
| Structured request logging | Enabled | Written line-by-line to `logs/requests.jsonl` |

## Run the service

From PowerShell:

```powershell
Set-Location 'D:\ParallexLabs\Week 5'
uv sync --locked
```

If this is a fresh checkout, create the environment file:

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

The service reads the Week 4 index, so ensure Week 4 has been fully built
(enriched chunks, ChromaDB index, and index manifest exist).

Start the server:

```powershell
uv run --locked uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

The interactive API docs are at http://localhost:8000/docs.

### API endpoints

**GET /health** — Liveness and readiness check.

```json
{"status": "ok", "collection": "week4_...", "chunk_count": 353}
```

**POST /query** — Embed a question and retrieve top-K chunks.

```json
{
  "question": "What is anarchism?",
  "k": 5,
  "topic_id": 2,
  "sentiment": "negative"
}
```

Returns ranked hits with metadata, cosine similarity, and the applied filters.

**GET /collection/stats** — Index metadata.

```json
{
  "collection": "week4_...",
  "chunk_count": 353,
  "embedding_model": "gemini-embedding-001",
  "embedding_dimensions": 768,
  "distance_metric": "cosine"
}
```

### Error responses

| Code | When |
| --- | --- |
| 400 | Invalid filter value (e.g. unknown sentiment string) |
| 422 | Request body fails Pydantic validation (missing question, k out of range) |
| 500 | Internal server error (Gemini failure, ChromaDB issue) |

All errors return `{"detail": "..."}`.

## Run the tests

```powershell
uv run --locked pytest -v
```

Tests use a mocked Gemini embedder and an in-memory fake Chroma collection, so
they run offline without an API key and without the Week 4 database.

## Run retrieval evaluation

This computes Precision@K and Recall@K against the gold-standard queries in
`queries/retrieval_gold.jsonl`:

```powershell
uv run --locked python -m scripts.evaluate_retrieval
uv run --locked python -m scripts.evaluate_retrieval --k 10
```

The report is written to `reports/retrieval_evaluation.json`.

### Gold file format

Each line is a JSON object:

```json
{"query": "...", "relevant_ids": ["chunk_id_1", "chunk_id_2"]}
```

The six pre-built test queries cover Anarchism, Albedo, Alabama, Abraham
Lincoln, Academy Awards, and Achilles — using real chunk IDs from the Week 4
enriched corpus.

## Run end-to-end evaluation

Start the server first, then run the evaluation script:

```powershell
# Terminal 1
uv run --locked uvicorn src.app:app --port 8000

# Terminal 2
uv run --locked python -m scripts.evaluate_e2e
uv run --locked python -m scripts.evaluate_e2e --base-url http://localhost:8000
```

The script exercises 10 test queries (including filtered queries), validates
response structure, checks filter correctness, and reports latency percentiles.
The report is written to `reports/e2e_evaluation.json`.

## Layout

```text
Week 5/
├── .env.example
├── .python-version
├── pyproject.toml
├── README.md
├── src/
│   ├── app.py                 # FastAPI application and endpoints
│   ├── schemas.py             # Pydantic request/response models
│   ├── logging_config.py      # JSON structured request logging
│   ├── gemini.py              # Gemini embedding client (from Week 4)
│   ├── vector_store.py        # ChromaDB query operations (from Week 4)
│   ├── io_utils.py            # JSONL and JSON I/O utilities
│   └── paths.py               # Project paths (points to Week 4 data)
├── scripts/
│   ├── evaluate_retrieval.py  # Precision@K, Recall@K evaluation
│   └── evaluate_e2e.py        # End-to-end quality and latency evaluation
├── tests/
│   ├── conftest.py            # Mocked embedder, collection, and TestClient
│   └── test_api.py            # pytest unit tests (20 test cases)
├── queries/
│   └── retrieval_gold.jsonl   # Gold-standard test queries with relevant IDs
├── reports/                   # Generated evaluation reports
└── logs/                      # Structured request logs
```

## How it works

1. **Startup**: The FastAPI lifespan handler loads the Gemini API key, creates
   an embedder, and opens the Week 4 ChromaDB collection. If any of these fail,
   the server refuses to start.

2. **Query flow**: The `/query` endpoint validates the request with Pydantic,
   builds an optional Chroma metadata filter from the supplied topic and
   sentiment constraints, embeds the question with Gemini, queries ChromaDB for
   the top-K results by cosine similarity, and returns structured JSON.

3. **Logging**: Every request is logged as a JSON line with request ID, method,
   path, status code, duration, and client IP. Each response includes an
   `X-Request-ID` header.

4. **Testing**: The pytest suite substitutes a `FakeEmbedder` (deterministic
   random vectors, no API calls) and a `FakeCollection` (15 in-memory chunks
   with metadata filters) so all tests run offline.

5. **Retrieval evaluation**: The script embeds each gold query with Gemini,
   retrieves top-K from the real index, and compares the returned IDs against
   the annotated relevant IDs to compute Precision@K and Recall@K.

6. **E2E evaluation**: The script hits every endpoint over HTTP, checks
   response schemas, verifies that filters produce consistent results, and
   computes latency statistics.

## References

- [FastAPI Official Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [FastAPI TestClient](https://fastapi.tiangolo.com/tutorial/testing/)
- [Pydantic V2 Models](https://docs.pydantic.dev/latest/)
- [pytest Documentation](https://docs.pytest.org/en/stable/)
- [Chroma metadata filtering](https://docs.trychroma.com/docs/querying-collections/metadata-filtering)
- [Gemini embedding models](https://ai.google.dev/gemini-api/docs/embeddings)
