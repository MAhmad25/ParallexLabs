# ParallexLabs

This repository contains my weekly coursework projects. I use uv for Python
package management and keep each week's code, tests, reports, and environment in
its own folder.

```text
ParallexLabs/
├── .gitignore
├── README.md
├── Week 1/
│   ├── README.md
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── requirements.txt
│   ├── src/
│   ├── scripts/
│   ├── tests/
│   └── reports/
└── Week 2/
    ├── README.md
    ├── pyproject.toml
    ├── uv.lock
    ├── requirements.txt
    ├── src/
    ├── scripts/
    ├── tests/
    ├── queries/
    └── reports/
```

In [Week 1](Week%201/README.md), I built a Wikipedia ingestion and preprocessing
pipeline, retained 5,936 validated English documents, and added 50 passing tests.
I verified the CPU workflow. CUDA testing remains pending because my machine has
Intel UHD Graphics 620 rather than an NVIDIA GPU.

In [Week 2](Week%202/README.md), I added text chunking, Gemini embeddings, persistent
ChromaDB storage, semantic search, and optional Gemini answers. I indexed the first
10 cleaned articles as 353 vectors and measured retrieval across eight queries.
The project has 89 passing tests. Each week's README explains my approach and
includes setup instructions, measured results, and limitations.

## Git commands

Run Git commands from the parent folder:

```powershell
Set-Location 'D:\ParallexLabs'
git status
git push origin main
```

After editing files, commit and upload the changes:

```powershell
git add .
git commit -m "Update weekly coursework"
git push origin main
```

I keep environments, caches, API keys, datasets, and databases out of Git. The data
is available locally and can be recreated using the weekly READMEs.

## Run Week 1

Run uv commands inside the weekly project:

```powershell
Set-Location 'D:\ParallexLabs\Week 1'
uv sync --locked
uv run --locked pytest -q
```

## Run Week 2

```powershell
Set-Location 'D:\ParallexLabs\Week 2'
uv sync --locked
uv run --locked pytest -q
uv run --locked python -m scripts.chunk_corpus
uv run --locked python -m scripts.ingest
uv run --locked python -m scripts.benchmark_retrieval
```

Configure Week 2's ignored `.env` with a Gemini API key before ingestion. Its default
live index uses the first 10 Week 1 articles. Database files remain local; reports
and reproducible source are tracked in this repository.
