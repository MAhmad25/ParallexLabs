# Week 1 — Environment, Data Ingestion & Preprocessing

**Week:** October 5–11, 2026. **Package manager:** [uv](https://docs.astral.sh/uv/).

This project acquires real Wikipedia articles, cleans them with reusable Python functions,
filters for English, removes exact duplicates, and validates the resulting corpus. It includes
a local Git repository, a Python 3.12 virtual environment, dependency locks, unit tests, and
machine-readable reports.

**Hardware limitation:** the development PC has Intel UHD Graphics 620 graphics and no NVIDIA
CUDA GPU. CPU imports and processing can be verified here, but the assignment's CUDA requirement
requires a separate machine with compatible NVIDIA hardware. The default environment check
fails when CUDA is unavailable. `--allow-cpu` permits CPU work and explicitly records
`assignment_gpu_requirement_met: false`; it does not claim GPU verification.

## What was built

| Requirement | Implementation |
| --- | --- |
| Clean Git repository and isolated environment | Parent `ParallexLabs` Git repository contains this week; `.venv/` created by uv |
| Ignore datasets and secrets | `.gitignore` excludes data contents, `.env`, environments, and caches |
| Locked requirements | `uv.lock` plus an exported, version-pinned, hashed `requirements.txt` |
| Required imports and CUDA checks | `scripts/verify_env.py`; strict CUDA mode and explicit CPU override |
| At least 5,000 real documents | `scripts/download_data.py` downloads 6,000 Wikipedia articles |
| Modular cleaning and language filtering | `src/preprocessing.py` with independently callable functions |
| Unit tests | `tests/test_preprocessing.py` and `tests/test_pipeline.py` |
| Clean dataset | `data/processed/clean_corpus.jsonl`; optional Parquet output |
| Validated deliverable | `scripts/validate_dataset.py` checks every record and requires 5,000+ rows |

Actual execution results, including counts, hashes, package versions, and GPU status, are saved
under `reports/`. Data files exist locally and are intentionally excluded from Git.

## Verified results — October 11, 2026

| Check | Measured result |
| --- | --- |
| Python and packages | Python 3.12.12; 128 compatible packages installed with uv |
| Required imports | All 9 imports passed, including the 4 required libraries |
| Basic library operations | All 5 smoke tests passed |
| Raw acquisition | 6,000 real articles with 6,000 unique source IDs |
| Clean corpus | **5,936 English documents** in `data/processed/clean_corpus.jsonl` |
| Independent validation | Passed; 5,936 unique IDs and text hashes; zero errors |
| Clean text lengths | Minimum 108 characters; maximum 122,540 characters |
| Unit and integration tests | **50 passed**, zero failures, zero skipped |
| Ruff lint and formatting | Passed |
| CUDA verification | Hardware requirement remains unmet on Intel UHD Graphics 620 |

The pipeline rejected 64 articles: 41 for language confidence below 0.90, 17 for a different
detected language, and 6 for fewer than 100 characters. No accepted articles were exact duplicates.
The validator independently checked every cleaned record and confirmed the same output SHA-256
as the preprocessing report.

Evidence: [environment](reports/environment_report.json),
[strict CUDA check](reports/environment_strict.json),
[dataset provenance](reports/dataset_provenance.json),
[cleaning counts](reports/preprocessing_report.json),
[validation](reports/validation_report.json), [tests](reports/pytest_results.xml),
[installation](reports/installation_report.json), and [combined summary](reports/verification_summary.json).

Package downloads needed recovery from transient DNS failures and slow transfers. The exact
locked artifacts were downloaded into the ignored local `.cache/wheels` directory, including
a public PyPI mirror fallback, and all SHA-256 values were checked against `uv.lock`. Installation
then used `uv pip sync --require-hashes --no-index --find-links .cache/wheels --link-mode copy requirements.txt`.
The normal `uv sync --locked --offline` check and `uv pip check` subsequently passed. A fresh
machine can use the normal online setup instructions below.

## Project layout

```text
Week 1/
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
├── requirements.txt
├── README.md
├── src/
│   ├── preprocessing.py       # HTML, Unicode, whitespace, language, document cleaning
│   ├── pipeline.py            # File processing, exact deduplication, summary report
│   ├── validation.py          # Independent checks of the clean output
│   └── io_utils.py            # UTF-8 JSONL, atomic JSON reports, SHA-256 hashes
├── scripts/
│   ├── verify_env.py
│   ├── download_data.py
│   ├── run_preprocessing.py
│   └── validate_dataset.py
├── tests/
│   ├── test_preprocessing.py
│   └── test_pipeline.py
├── reports/                  # Tracked verification evidence; no article bodies
├── data/raw/                 # Git-ignored downloaded articles
├── data/processed/           # Git-ignored validated output
└── .venv/                    # Git-ignored Python environment
```

## Setup with uv

Run these PowerShell commands from the project folder:

```powershell
Set-Location 'D:\ParallexLabs\Week 1'
uv sync --locked
```

Python **3.12.12** is pinned in `.python-version`. uv can download that interpreter when it is
missing. `uv sync --locked` installs the versions already recorded in `uv.lock`, including
development tools. `uv run --locked` selects `.venv` automatically, so activation is optional.

Main dependencies: pandas, sentence-transformers, ChromaDB, spaCy, PyTorch (transitive through
sentence-transformers), Beautiful Soup, langdetect, Hugging Face Datasets, and PyArrow.
Development dependencies: pytest and Ruff. No API key, `.env`, paid service, or spaCy language
model is needed for the Week 1 workflow.

The project was initialized using `uv init --bare --name week-1 --python 3.12 --vcs git`,
followed by `uv python pin`, `uv add`, `uv add --dev`, and `uv sync`. Git ownership now
resides in the parent `ParallexLabs` folder so all weeks can be uploaded together.
Run Git commands from `D:\ParallexLabs`; run uv commands from this weekly project folder.

To regenerate the required exported lock after a deliberate dependency update:

```powershell
uv lock
uv export --locked --format requirements.txt --all-groups --no-emit-project --output-file requirements.txt
```

Alternatively, install from the hashed requirements export using uv:

```powershell
uv venv --python 3.12.12
uv pip sync --require-hashes requirements.txt
```

Use `uv sync --locked` for the normal project workflow. Do not manually edit `uv.lock` or
`requirements.txt`.

## Run the complete workflow

### 1. Verify the environment

Strict assignment check (requires working CUDA and returns exit code 1 when missing):

```powershell
uv run --locked python scripts/verify_env.py --output reports/environment_strict.json
```

On this Intel-only PC, run the explicit CPU check:

```powershell
uv run --locked python scripts/verify_env.py --allow-cpu
```

The script imports every required library and checks a pandas DataFrame, the
SentenceTransformer API, spaCy tokenization, an in-memory ChromaDB add/query operation with
provided embeddings, and a real PyTorch CPU matrix operation. On a CUDA machine it also runs
a CUDA matrix multiplication and synchronizes the GPU. It saves versions and outcomes to JSON.
It does not download an embedding model or claim model inference was tested.

For GPU completion, use a supported NVIDIA GPU and compatible driver, then install a CUDA
PyTorch build selected from the [official PyTorch instructions](https://pytorch.org/get-started/locally/).
Use `uv add torch --index pytorch=CUDA_INDEX_URL` with the actual PyTorch wheel index URL from
those instructions; uv records the selected source. Export requirements again after the
dependency change, and rerun the strict command until it exits successfully. Installing a
CUDA toolkit on Intel graphics cannot provide an NVIDIA GPU.

### 2. Acquire real documents

```powershell
uv run --locked python scripts/download_data.py --limit 6000
```

This streams the first 6,000 articles of Wikimedia's `20231101.en` snapshot into
`data/raw/wikipedia_en.jsonl`. It preserves the source article ID, URL, title, and text.
The fixed dataset revision is `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`.
Source metadata, acquisition time, unique IDs, and a raw-file SHA-256 are saved in
`reports/dataset_provenance.json`. The script fails on incomplete acquisition, unexpected field
types, or duplicate article IDs. Output is published only after the whole download succeeds.

Use `--force` to deliberately replace an existing raw download. Streaming avoids downloading
the full multi-million-article corpus, although the library can buffer Parquet row groups and
cache metadata under the ignored `.cache/` directory.

### 3. Clean and filter

```powershell
uv run --locked python scripts/run_preprocessing.py
```

Defaults: English, at least 100 characters, and language confidence of at least 0.90.
The output is `data/processed/clean_corpus.jsonl`. Counts, rejection reasons, cleaning settings,
and input/output hashes are saved in `reports/preprocessing_report.json`.

For Parquet instead:

```powershell
uv run --locked python scripts/run_preprocessing.py --output data/processed/clean_corpus.parquet --report reports/preprocessing_parquet_report.json
uv run --locked python scripts/validate_dataset.py --input data/processed/clean_corpus.parquet --report reports/validation_parquet_report.json
```

JSONL processing writes one article at a time; Parquet output collects accepted records before
writing with PyArrow. The latter is appropriate for this 6,000-article assignment.

### 4. Validate and test

```powershell
uv run --locked python scripts/validate_dataset.py
uv run --locked pytest -q --junitxml=reports/pytest_results.xml
uv run --locked ruff check .
uv run --locked ruff format --check .
```

Validation exits with code 1 for any error or fewer than 5,000 clean documents. It rechecks
language from the cleaned text, rather than trusting stored language labels. It also checks
required fields, unique IDs and normalized text hashes, text-length metadata, NFKC Unicode,
whitespace, invisible characters, remaining HTML, and content SHA-256 values. Results are saved
in `reports/validation_report.json`.

The tests cover malformed/nested HTML, entities, script/style/comment removal, mathematical
text, accented and non-Latin Unicode, zero-width/control characters, whitespace, bad input
types, deterministic language filtering, confidence thresholds, rejection accounting,
deduplication, both output formats, corrupt output detection, atomic failure behavior, and
CPU/GPU exit-code semantics. Tests use local fixtures and need no internet.

If you change filtering thresholds, pass the same `--language`, `--min-chars`, and
`--min-confidence` options to preprocessing and validation.

## Cleaning approach

1. Normalize compatibility characters with Unicode NFKC. Remove invisible formatting
   characters, replace zero-width spaces with word boundaries, and remove invalid surrogates.
2. Decode HTML entities, strip markup using Beautiful Soup, and remove scripts, styles,
   templates, noscript content, and comments. Preserve plain expressions such as `x < 5`.
3. Normalize Unicode again after HTML entity decoding, then collapse all whitespace into
   one ASCII space and trim the text.
4. Reject invalid records or articles shorter than 100 characters.
5. Detect language using a deterministic langdetect model with seed 0 and the first 2,000
   cleaned characters. Keep English predictions with confidence ≥ 0.90.
6. Deduplicate by article ID and SHA-256 of the normalized article text.
7. Retain source attribution and add language, confidence, length, and content-hash metadata.

Case, punctuation, numbers, accents, and wording are preserved. This week does not lowercase,
stem, remove stopwords, train models, or build a retrieval system.

Each cleaned JSONL record contains `id`, `title`, `url`, `text`, `source`, `language`,
`language_confidence`, `char_count`, `word_count`, and `content_sha256`.

## Dataset provenance and limitations

The [Wikimedia Wikipedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia) contains
articles extracted from real Wikipedia dumps. Its upstream processing already strips wiki
markup and some unwanted sections. Here, **raw** means the acquired upstream article text
before this project's own cleaning; it is not an untouched Wikimedia XML dump.

The dataset card lists CC-BY-SA 3.0 and GFDL. Article URLs and IDs are retained for attribution;
see the [Wikimedia licensing information](https://dumps.wikimedia.org/legal.html) and
[dataset card](https://huggingface.co/datasets/wikimedia/wikipedia) before redistributing data.
The source license still applies to cleaned article text.

The first-N sampling is reproducible but is not a random or representative sample of all
Wikipedia topics. Language detection is probabilistic; prefix-based detection can misclassify
mixed-language documents. Deduplication removes exact normalized matches, not near duplicates.
NFKC deliberately folds compatibility characters and can alter specialized typography.
The 6,000-document acquisition provides headroom for rejected articles, while validation
enforces the required final minimum instead of assuming every downloaded article survives.

## Submission and troubleshooting

Submit the source code, this README, both dependency lock files, tests, and `reports/`.
If the submission requires the clean data artifact, attach the local JSONL or Parquet file
separately; it is intentionally absent from Git. A new checkout can recreate it using the
commands above. Do not submit `.venv`, caches, or secrets.

- **CUDA unavailable:** expected on this Intel UHD 620 PC. Complete CPU work with `--allow-cpu`
  and use a compatible NVIDIA machine for the strict GPU deliverable.
- **Transient package download errors:** retry `uv sync --locked`. For slow networks,
  PowerShell can set `$env:UV_HTTP_TIMEOUT = '120'` and `$env:UV_CONCURRENT_DOWNLOADS = '2'`
  before retrying. `uv sync --locked --system-certs` uses the Windows certificate store.
- **Download already exists:** use the existing raw file, or add `--force` to redownload it.
- **Fewer than 5,000 clean records:** acquire a larger raw sample, rerun preprocessing, and
  validate again. Do not lower the validation minimum for the submitted deliverable.
- **Different filter settings:** keep preprocessing and validation flags in agreement.

References: [uv project workflow](https://docs.astral.sh/uv/guides/projects/),
[uv dependency export](https://docs.astral.sh/uv/concepts/projects/sync/),
[Hugging Face streaming](https://huggingface.co/docs/datasets/stream),
[PyTorch CUDA availability check](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available.html).
