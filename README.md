# ParallexLabs

Weekly coursework projects. This folder is the Git repository; each week's code
and Python environment live in a separate subfolder.

```text
ParallexLabs/
├── .gitignore
├── README.md
└── Week 1/
    ├── README.md
    ├── pyproject.toml
    ├── uv.lock
    ├── requirements.txt
    ├── src/
    ├── scripts/
    ├── tests/
    └── reports/
```

[Week 1 setup, results, and run instructions](Week%201/README.md): Wikipedia
ingestion and preprocessing with uv, 50 passing tests, and 5,936 validated
English documents. CUDA verification remains pending on compatible NVIDIA hardware.

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
git commit -m "Update Week 1"
git push origin main
```

Environments, caches, secrets, and dataset contents are ignored. Data remains
available locally and can be recreated using the Week 1 README.

## Run Week 1

Run uv commands inside the weekly project:

```powershell
Set-Location 'D:\ParallexLabs\Week 1'
uv sync --locked
uv run --locked pytest -q
```
