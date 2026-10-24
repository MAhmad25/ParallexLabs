# Week 3: Gemini Grounded Generation

I extended my Week 2 Gemini and ChromaDB retrieval workflow into a small
retrieval-augmented generation (RAG) command-line app. It retrieves relevant
Wikipedia excerpts, asks Gemini to answer from those excerpts, checks the
answer's source labels, refuses weak matches, and records end-to-end latency.

## What I built

The request flow is:

1. I embed the question with Gemini and retrieve the top matching chunks from my
   persistent ChromaDB cosine index.
2. I compare the best cosine similarity with a configurable minimum. If there is
   no relevant match, the app returns a clear refusal without calling Gemini for
   answer generation.
3. I send the question and retrieved excerpts to Gemini through Google's
   `google-genai` SDK. A system instruction limits claims to the excerpts,
   treats excerpt text as untrusted data, and asks for a citation on every
   factual sentence.
4. I check that every citation refers to a retrieved source and that each answer
   sentence has a citation. If the check fails, the app withholds the generated
   answer and abstains.
5. I append request timing and outcome metadata to `reports/latency.jsonl`.
   The log stores a SHA-256 query fingerprint instead of the query or answer.

I use `gemini-embedding-001` for the query vector and Google's stable
`gemini-3.8-flash` model for text generation. I keep generation model selection
configurable so I can change it if Google's model availability changes.

## Project layout

```text
Week 3/
├── .env.example
├── .gitignore
├── .python-version
├── README.md
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── data/processed/       # Local chunks; ignored by Git
├── chroma_db/            # Local persistent index; ignored by Git
├── reports/              # Local reports and latency log
├── scripts/
│   ├── ask.py
│   ├── chunk_corpus.py
│   ├── ingest.py
│   └── verify_index.py
└── src/
    ├── __init__.py
    ├── chunking.py
    ├── cli.py
    ├── gemini.py
    ├── io_utils.py
    ├── latency.py
    ├── retrieval.py
    └── vector_store.py
```

Each week has its own `uv` project and database. Week 3 reads the cleaned
corpus from `Week 1` by default, then builds its own chunks and vector index.

## Setup and run

I use Python 3.12 and [uv](https://docs.astral.sh/uv/). From PowerShell:

```powershell
Set-Location 'D:\ParallexLabs\Week 3'
uv sync --locked
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

Put my Google AI Studio key in `Week 3/.env` as `GEMINI_API_KEY`. The app also
accepts `GOOGLE_API_KEY` from the shell and can fall back to the existing
`Week 2/.env` key when no Week 3 key or shell variable is set. I never commit
either `.env` file.

Build the default index from the first 10 cleaned Week 1 articles, matching the
Week 2 corpus scope:

```powershell
uv run --locked python -m scripts.chunk_corpus
uv run --locked python -m scripts.ingest --request-interval 12
uv run --locked python -m scripts.verify_index
```

Chunking is local. Ingestion sends each chunk to Gemini for embeddings and uses
the API quota. To index the full cleaned corpus instead, pass `--limit 0` while
chunking; the larger corpus requires more embedding requests and disk space.

Ask a question about the indexed articles:

```powershell
uv run --locked python -m scripts.ask "What does the indexed source say about the topic?"
```

The command prints JSON containing the answer, cited source metadata, retrieval
timing, generation timing, end-to-end timing, and the best similarity score.
Each valid request appends one JSON object to `reports/latency.jsonl`. To change the
retrieval count or the out-of-domain threshold:

```powershell
uv run --locked python -m scripts.ask "Your question" --k 5 --min-similarity 0.35
```

`--k` accepts 1 through 10. The default threshold is `0.35`; set
`RAG_MIN_SIMILARITY` in `.env` to change it for future runs. The app rejects
questions longer than 2,000 characters. The HTTP timeout defaults to 45 seconds
and can be changed with `GEMINI_TIMEOUT_SECONDS` (5 to 180 seconds).

If `Week 1/data/processed/clean_corpus.jsonl` is missing in a fresh checkout,
create it with the Week 1 preprocessing steps before running `chunk_corpus`.

## Prompt and reliability choices

- I separate system instructions from the user question using Gemini's
  `system_instruction` configuration.
- I pass retrieved excerpts as JSON strings with explicit source IDs. The system
  prompt tells the model that excerpts are untrusted content, not instructions.
- I set Gemini 3.8 Flash thinking to `low` and cap output at 2,048 tokens. I omit
  temperature because current Gemini 3.6 and later guidance says custom
  temperature is ignored and may become an error.
- I refuse to generate when the best retrieved cosine similarity is below the
  configured threshold. This is a practical heuristic, not a learned domain
  classifier. It should be tuned using representative in-domain and unrelated
  questions for a new corpus.
- I reject an answer if it cites an ID outside the retrieved source list, has no
  citations, or contains a sentence without a citation. I also reject a response
  stopped by a safety block or output-token limit.
- I retry transient Gemini 408, 429, selected 5xx, timeout, and transport errors
  at most twice after the initial request, with exponential backoff and jitter.
  A server retry delay is honored up to eight seconds. Invalid key/model and
  malformed-request errors are not retried. SDK retries are disabled so there is
  only one bounded retry layer.
- I log timings and outcome metadata, not raw queries, answers, API keys, or SDK
  exception bodies. A latency record includes query hash, status, retrieval and
  generation milliseconds, similarity, model, and source count.

## Latency fields

`reports/latency.jsonl` contains one record per command invocation:

| Field | Meaning |
| --- | --- |
| `embedding_ms` | Gemini question embedding time |
| `vector_search_ms` | Local ChromaDB lookup time |
| `retrieval_ms` | Embedding plus vector lookup |
| `generation_ms` | Gemini answer-generation time; zero when the app refuses before generation |
| `end_to_end_ms` | Retrieval and generation elapsed time, or total elapsed time for an early failure |
| `status` | `answered`, `refused`, or `error` |

I record measured values when I run the command rather than publishing a fixed
latency number, because API latency depends on network conditions, quota,
provider load, model settings, and retries.

## Limitations

Citation validation checks the citation format and that cited IDs exist in the
retrieved set. It cannot prove that a cited excerpt logically supports every
claim. The similarity threshold can also reject valid questions or admit a
misleading match; it must be calibrated for the selected corpus. The Week 2
baseline used only the first 10 articles, so retrieval quality from that small
sample is limited. This project does not guarantee that Gemini never
hallucinates.

## Troubleshooting

| Problem | What I check |
| --- | --- |
| Missing Gemini key | Set `GEMINI_API_KEY` in `.env` or `GOOGLE_API_KEY` in the shell |
| HTTP 401 or 403 | Check the key, project API access, quota, and billing status |
| HTTP 404 | Choose a generation model available to the Google AI Studio project |
| HTTP 429 | Check current quota, wait for its reset, or increase the ingestion request interval |
| HTTP 400 or 413 | Check model settings and reduce question or retrieved context size |
| No index manifest | Run `chunk_corpus` and `ingest` in that order |
| Missing cleaned corpus | Run the Week 1 preprocessing pipeline to create `Week 1/data/processed/clean_corpus.jsonl` |
| Low-relevance refusal | Ask about a topic covered in the indexed articles or calibrate the threshold |
| Citation-validation refusal | Treat it as an abstention; ask a narrower question with clearer source evidence |

I use the current [Gemini models guide](https://ai.google.dev/gemini-api/docs/models),
[Gemini text-generation guide](https://ai.google.dev/gemini-api/docs/generate-content),
[Gemini thinking guide](https://ai.google.dev/gemini-api/docs/thinking),
[Gemini troubleshooting guide](https://ai.google.dev/gemini-api/docs/troubleshooting),
and the [Google Gen AI Python SDK](https://googleapis.github.io/python-genai/).
