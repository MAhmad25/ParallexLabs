# Week 2 — Chunking, Embeddings & Vector DB

**October 12–18, 2026 · Python 3.12.12 · uv · Gemini · ChromaDB**

For Week 2, I extended my Week 1 preprocessing work into a semantic search pipeline.
I split the cleaned Wikipedia articles into overlapping chunks, generated embeddings
with Gemini, and stored the vectors and source metadata in ChromaDB. I then tested
the chunking and database handling and benchmarked retrieval across eight queries.
I also added an optional Gemini answer step that cites the retrieved passages.

I used Gemini for both embeddings and answer generation. The default embedding
model is `gemini-embedding-001` with 768 dimensions. I disabled Chroma's automatic
embedding function so stored vectors and query vectors use the same Gemini
configuration. The offline tests use separate temporary databases and test vectors.

## What I built

| Task | Implementation |
| --- | --- |
| Extend Week 1 | Reads `../Week 1/data/processed/clean_corpus.jsonl` without changing it |
| Tested chunking | Recursive boundary selection, overlap, exact source offsets, deterministic IDs |
| Embedding performance | Gemini batch wall time and throughput in ingestion reports |
| Populated vector database | Persistent cosine Chroma collection in local `chroma_db/` |
| Semantic search | Query embeddings use retrieval-query mode; document vectors use retrieval-document mode |
| Retrieval benchmark | Eight labeled queries, warmups, repeated measurements, p95, hit rate, recall, MRR |
| Edge cases | Resume after interruption, duplicate detection, incomplete collection checks, model isolation |
| LLM | Optional source-grounded Gemini answer with numbered citations |
| Reproducibility | Python pin, `uv.lock`, hashed requirements export, offline tests, provenance hashes |

## My results — October 18, 2026

| Check | Measured result |
| --- | --- |
| Input subset | First 10 real Week 1 articles |
| Chunks and persisted vectors | **353**, all 768-dimensional Gemini vectors |
| Initial ingestion | 96 stored chunks before HTTP 429; preserved without re-embedding |
| Resumed ingestion | 257 new chunks; complete collection with 353 total |
| Cumulative successful embedding wall time | **38.417 seconds** across both runs |
| Successful embedding throughput | **9.189 chunks/second**, excluding pacing and failed calls |
| Resumed run wall time | 193.190 seconds, including 12-second batch pacing |
| Independent database verification | All 353 texts, metadata records, dimensions, finite values, and vector norms passed |
| Offline unit/integration tests | **89 passed**, zero skipped; real local Chroma plus test-only embedding doubles |
| Lint, formatting, locked offline sync, dependency compatibility | Passed |
| Live benchmark | Eight queries, 24 measured runs, eight excluded warmups |
| Mean query embedding latency | **535.02 ms** |
| Mean local vector-search latency | **8.70 ms** |
| End-to-end retrieval mean / median / p95 | **544.07 / 512.05 / 840.85 ms** |
| Hit@1, document Recall@1, MRR@1 | **1.00** for this small labeled query set |
| Optional Gemini LLM | Live grounded-answer check; see verification summary |

I saved my results in these reports: [chunking](reports/chunking_report.json),
[initial interrupted ingestion](reports/ingestion_first_attempt.json),
[completed ingestion](reports/ingestion_report.json),
[active manifest](reports/index_manifest.json),
[independent database verification](reports/index_verification.json),
[retrieval benchmark](reports/retrieval_benchmark.json),
[test results](reports/pytest_results.xml), and
[combined verification](reports/verification_summary.json).

My first ingestion attempt stored 96 chunks before Gemini returned HTTP 429.
I resumed with a 12-second interval between batches and stored the remaining 257.
The embedding total sums successful batch timings from both attempts. It excludes
the failed API call and its retry waits; each report also records total runtime.
The retrieval timings come from queries against the Gemini-backed database.

## Layout

```text
ParallexLabs/                       # My weekly projects
├── Week 1/                         # Preprocessing
└── Week 2/                         # Chunking and retrieval
    ├── .env.example                # Configuration names; never a real key
    ├── .python-version
    ├── pyproject.toml
    ├── uv.lock
    ├── requirements.txt
    ├── src/
    │   ├── chunking.py             # Chunking and corpus preparation
    │   ├── gemini.py               # Gemini embeddings and grounded generation
    │   ├── vector_store.py         # Persistent explicit-vector Chroma collections
    │   ├── retrieval.py            # Resumable ingestion and timed semantic search
    │   ├── benchmark.py            # Latency and document relevance measurements
    │   ├── io_utils.py             # Atomic reports and hashes
    │   └── cli.py                  # Defaults and safe command errors
    ├── scripts/                    # Runnable Python modules
    ├── queries/benchmark.json      # Manually labeled query set
    ├── tests/                      # No network or credentials required
    ├── reports/                    # Tracked execution evidence
    ├── data/processed/             # Local chunk JSONL, Git-ignored
    ├── chroma_db/                  # Local populated database, Git-ignored
    └── .env                        # Local secret, Git-ignored
```

## Setup

Run PowerShell from **Week 2**, not the repository root:

```powershell
Set-Location 'D:\ParallexLabs\Week 2'
uv sync --locked
```

I used Python 3.12.12 and uv to keep the environment reproducible. uv reads the
interpreter pin from `.python-version` and installs locked packages into this
folder's `.venv`. The main dependencies are ChromaDB, the Google Gen AI SDK, httpx,
and python-dotenv. I used pytest for tests and Ruff for linting and formatting.
Embedding and LLM inference run through Gemini, so no local GPU is needed.

For a fresh checkout, create the local configuration once:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set these values; replace only the placeholder with your own
[Google AI Studio key](https://aistudio.google.com/api-keys):

```dotenv
GEMINI_API_KEY=replace_with_your_google_ai_studio_key
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_EMBEDDING_DIMENSIONS=768
GEMINI_LLM_MODEL=gemini-flash-latest
```

I keep the API key in the ignored `.env` file. An existing configuration can be
reused without copying the example again. Shell variables take precedence over
`.env`, and `GOOGLE_API_KEY` is accepted when `GEMINI_API_KEY` is absent. The
application loads only Week 2's `.env`.

## Run the workflow

### 1. Reuse the Week 1 corpus

The local input is Week 1's English JSONL output. If it is missing on a new checkout,
follow the [Week 1 instructions](../Week%201/README.md) to download and preprocess it:

```powershell
Set-Location '..\Week 1'
uv sync --locked
uv run --locked python scripts/download_data.py --limit 6000
uv run --locked python scripts/run_preprocessing.py
uv run --locked python scripts/validate_dataset.py
Set-Location '..\Week 2'
```

### 2. Create chunks

```powershell
uv run --locked python -m scripts.chunk_corpus
```

I used the **first 10 articles** to keep the embedding run within the available API
quota. All eight benchmark labels refer to this subset. The output is
`data/processed/chunks.jsonl`; its counts, source SHA-256, output SHA-256, settings,
article IDs, sizes, and measured chunking time are in `reports/chunking_report.json`.

To chunk all documents separately without spending API quota:

```powershell
uv run --locked python -m scripts.chunk_corpus --limit 0 --output data/processed/all_chunks.jsonl --report reports/all_chunking_report.json
```

`--chunk-size 1500`, `--overlap 200`, `--limit N`, and `--input PATH` are configurable.
`--limit 0` means all articles. Embedding the complete corpus requires a separate
ingestion run and more API quota.

### 3. Generate embeddings and populate ChromaDB

```powershell
uv run --locked python -m scripts.ingest
uv run --locked python -m scripts.verify_index
```

I configured ingestion to use 16 chunks per batch, 12 seconds between batch starts,
`gemini-embedding-001`, and 768 dimensions. This interval helped avoid further rate
limits during my run. Increase `--request-interval` up to 60 seconds or reduce
`--batch-size` if necessary. `--request-interval 0` disables pacing for a project
with sufficient quota.

Each completed batch logs embedding wall time and database-write time. Reports
record preexisting chunks, newly embedded chunks, batch times, cumulative embedding
time, throughput, total runtime, status, and final count. Embedding time includes
SDK/network work, normalization, and retries; it excludes intentional pacing.

Run **the same command** after a failure. Already stored IDs are checked before
embedding, so only missing chunks incur new requests. Completed batches persist
even when a later batch fails. The initial interrupted run is preserved in
`reports/ingestion_first_attempt.json`; the completed resumed run is recorded in
`reports/ingestion_report.json`.

After every expected chunk is stored, the collection is marked ready and
`reports/index_manifest.json` is atomically published. Search uses that manifest.
A rerun against a completed, identical snapshot performs no embedding calls.
`scripts.verify_index` independently checks every persisted text/metadata pair and
vector against the chunk file and manifest without using the Gemini API.

### 4. Search

```powershell
uv run --locked python -m scripts.search 'What measures reflected sunlight?' --k 3
```

The JSON output contains ranked source excerpts, article IDs, titles, URLs, chunk
offsets, cosine distances, cosine similarities, query embedding time, vector-search
time, and total search latency. Smaller cosine distance means closer vectors;
`cosine_similarity = 1 - distance` is a similarity score, not a probability.

Add optional Gemini answer generation:

```powershell
uv run --locked python -m scripts.search 'Who taught Alexander the Great?' --k 3 --answer --output data/processed/answer_example.json
```

The LLM sees only the retrieved excerpts, is instructed to cite `[1]`, `[2]`, etc.,
and to acknowledge insufficient evidence. Source text is treated as untrusted data.
Generation time is reported separately and is excluded from retrieval benchmarks.
This command is optional; semantic search itself does not call an LLM.

### 5. Benchmark retrieval

```powershell
uv run --locked python -m scripts.benchmark_retrieval
```

I wrote eight queries with manually labeled relevant article IDs. For each query,
I ran one warmup followed by three measured searches and evaluated results at
k = 1, 3, and 5. This produced **24 measured samples** plus eight excluded warmups.
Each search generated a fresh query embedding. I measured retrieval after opening
the database and client, so startup and database-opening time are excluded.

`reports/retrieval_benchmark.json` preserves every raw sample, query labels,
retrieved document IDs, per-query latency summaries, global summaries, failures,
and aggregate relevance metrics:

| Measurement | Meaning |
| --- | --- |
| `embedding_ms` | Gemini query embedding, network, retries, normalization |
| `vector_search_ms` | Local readiness/count checks, Chroma lookup, result assembly |
| `total_ms` | End-to-end search call; excludes LLM generation |
| Mean, median, min, max | Descriptive latency statistics in milliseconds |
| p95 | Nearest-rank 95th percentile; small sample sizes limit stability |
| Hit@k | Whether any retrieved chunk belongs to a labeled relevant article |
| Document Recall@k | Fraction of distinct labeled articles retrieved; repeated chunks count once |
| MRR@k | Reciprocal rank of the first relevant chunk, averaged over samples |

I calculated all three k metrics from the same top-5 lookup. Each query has one
labeled relevant article, so Hit@k and document Recall@k have the same value for
this query set. The labels check article relevance rather than individual chunks.
These results cover a small corpus and eight queries; broader evaluation would
need more varied data. The script rejects missing labels before making API requests.

Custom benchmark:

```powershell
uv run --locked python -m scripts.benchmark_retrieval --repeats 5 --warmups 1 --k 1 5 10 --output reports/custom_benchmark.json
```

A custom query JSON file is a nonempty list of objects with unique `id`, nonblank
`query`, and nonempty `relevant_document_ids` containing string article IDs.
Use `--queries PATH` and labels that exist in the indexed corpus.

## Chunking strategy

I chose a maximum **1,500-character window** with **200-character raw overlap**
to retain context across chunk boundaries. Within each window, the splitter
recursively searches for the last suitable paragraph break, newline, sentence
ending (`. `, `! `, `? `), and space, in that
priority order. If none is suitable, it cuts at the character boundary. Boundaries
are searched in the latter half of the window, beyond the requested overlap, to
avoid tiny chunks and guarantee forward progress.

Leading/trailing whitespace is trimmed after selecting a window. Every non-whitespace
source character remains covered, and stored offsets identify the exact substring
in the original article. The raw overlap is 200 characters; whitespace trimming
can reduce the visible overlap. Paragraph and sentence preferences are heuristics,
not a linguistic sentence parser. Week 1 collapses whitespace, so its articles
mostly use sentence/word boundaries. Character lengths are not token counts.

Chunk IDs are SHA-256 hashes of document ID, original document-content hash, chunk
index, and start/end offsets. Original title, URL, source, document hash, and article
ID are preserved for attribution. Empty/bad records, invalid hashes, duplicate
article IDs, and invalid chunk configurations fail explicitly. A failed chunking
run leaves an existing output file intact.

## Embedding design and Chroma edge cases

- **Models:** default `gemini-embedding-001` uses `RETRIEVAL_DOCUMENT` for indexed
  chunks and `RETRIEVAL_QUERY` for searches. Every vector is checked for dimension,
  finite values, and nonzero magnitude, then L2-normalized before storage/querying.
- **Embedding 2:** optionally set `GEMINI_EMBEDDING_MODEL=gemini-embedding-2` and
  ingest again. It uses Google's retrieval text prefixes and one independent
  request per text to avoid input aggregation. This is slower than the default
  batched Embedding 1 path. Its production API availability must be tested with
  your own key; offline coverage alone is not a live model check.
- **Snapshots:** collection names hash chunk-file SHA-256, model, dimensions,
  provider, and format version. Changed input/settings create a separate collection;
  old chunks cannot contaminate a new snapshot. Old snapshots are retained locally.
- **Mismatches:** collection metadata and cosine configuration are checked on open.
  Query settings must match the active manifest. Wrong dimension/model and corrupt
  metadata fail clearly, without substituting another provider.
- **Incomplete/empty data:** empty input is rejected. An unfinished collection is
  not searchable or advertised as ready. Resume ingestion to complete it.
- **Duplicates/retries:** duplicate IDs within input or a batch are rejected;
  persistent upserts are idempotent. A response lost after a completed database
  write is safe to retry because stable IDs overwrite the same records.
- **Batch limits:** database writes respect Chroma's advertised maximum batch size.
  Query k is capped at the collection count; zero/negative k is rejected.
- **API failures:** bounded retries handle 429, selected 5xx errors, and network
  transport failures. Rate-limit waits are bounded; persistent quota exhaustion
  exits nonzero and keeps completed work. Authentication/model errors fail without
  futile retries. Raw SDK errors and API keys are not written to reports.
- **Local database:** a persistent client needs write access to its directory.
  Corruption or SQLite locking errors fail without deleting data. Stop competing
  writers, back up the directory, and use a new `--db` directory for a clean rebuild.
  One writer per database is the supported workflow.

Embedding inputs have an 8,000-character application guard. This is not a precise
model token-limit guarantee; use smaller chunks for unusual token-dense text or
an API input-length error. The default English 1,500-character chunks were tested
through the actual API. This project does not silently truncate or drop failed chunks.

## Tests and dependency checks

```powershell
uv run --locked pytest -q --junitxml=reports/pytest_results.xml
uv run --locked ruff check .
uv run --locked ruff format --check .
uv pip check
```

I wrote the tests to run without credentials or internet. They cover Unicode,
whitespace, unbroken text, overlap boundaries, randomized non-whitespace coverage, source
offsets/hashes, stable IDs, invalid input, atomic output, Gemini task configuration,
Embedding 2 aggregation avoidance, normalization, malformed responses, retry
limits, safe errors, real persistent Chroma queries, batch duplicates, incompatible
metadata, interrupted ingestion/resume, idempotence, benchmark statistics,
document relevance, missing labels, and failure reports. Deterministic test-only
embeddings exercise local database behavior without claiming semantic model quality.

To intentionally update dependencies and regenerate the hashed export:

```powershell
uv lock
uv export --locked --format requirements.txt --all-groups --no-emit-project --output-file requirements.txt
```

Normal setup uses `uv sync --locked`; do not manually edit lock files.

## Data and limitations

Input comes from the Week 1 `wikimedia/wikipedia` dataset, configuration
`20231101.en`, pinned revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`.
My complete Week 1 corpus contains 5,936 accepted articles. I recorded the 10 indexed
articles in the chunking report. First-N sampling is
reproducible but not representative of Wikipedia. Article titles and URLs remain
attached to retrieved excerpts. Upstream CC-BY-SA/GFDL attribution and share-alike
terms still apply; see [Week 1 provenance](../Week%201/README.md).

I kept this project in my `MAhmad25/ParallexLabs` repository alongside Week 1.
The repository tracks the code, tests, dependency locks, query labels, and reports.
API keys, environments, generated text, and database files remain local and ignored.
A fresh clone can rebuild the database using the workflow above and a valid Gemini key.

I also packaged the populated database, chunk JSONL, manifest, and this README in
`artifacts/week2-populated-vector-db.zip`. I verified the database after extracting
the package. The file list and hash are in `reports/submission_artifact.json`, and
the extracted database check is in `reports/package_verification.json`.

API latency depends on quota, network conditions, Google's service, and retry
behavior. Tests and benchmarks do not establish production-scale retrieval quality,
concurrency guarantees, or model answer correctness. No CUDA requirement is added
by Week 2's hosted-model workflow.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Missing Week 1 input | Recreate the corpus using the Week 1 commands above |
| Missing key | Configure the ignored `.env`; do not put credentials in source or Git |
| HTTP 400/401/403 | Check the key, API access, input size, and model configuration |
| HTTP 404 | Select a Gemini model available to the key and reindex if the embedding model changes |
| HTTP 429 | Wait for the quota window or fix billing/quota, increase pacing, rerun ingestion |
| Missing/incomplete manifest | Finish ingestion; do not query a partially populated snapshot |
| Settings differ from active index | Restore model/dimensions or build the corresponding new snapshot |
| Benchmark label missing | Use the default first 10 articles or provide a matching labeled query file |
| Slow package download | Retry `uv sync --locked`; use `--system-certs` if Windows TLS trust requires it |
| Chroma deprecation warning | Current Chroma emits a legacy embedding-config warning for explicit `None`; API behavior is tested |

References: [Gemini embeddings and task types](https://ai.google.dev/gemini-api/docs/embeddings),
[Google Gen AI SDK](https://googleapis.github.io/python-genai/),
[Chroma collection configuration](https://docs.trychroma.com/docs/collections/configure),
[Chroma explicit-vector ingestion](https://docs.trychroma.com/docs/collections/add-data),
[Chroma query behavior](https://docs.trychroma.com/docs/querying-collections/query-and-get),
[uv projects](https://docs.astral.sh/uv/guides/projects/).
