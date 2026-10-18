"""Shared paths and credential-safe CLI errors."""

import sys
from collections.abc import Callable

from src.gemini import ROOT

CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
DATABASE = ROOT / "chroma_db"
MANIFEST = ROOT / "reports" / "index_manifest.json"


def run(main: Callable):
    # PowerShell redirection can expose a cp1252 stdout; preserve source Unicode.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    except Exception as error:
        print(
            f"Error: {type(error).__name__}. Check file access, database health, and network.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
