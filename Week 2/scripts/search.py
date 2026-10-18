"""Semantic search, with optional Gemini-only grounded answer generation."""

import argparse
import json
from pathlib import Path

from src.cli import DATABASE, MANIFEST, run
from src.gemini import Gemini, Settings
from src.io_utils import distinct_paths, now, write_json
from src.retrieval import open_store, search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--answer", action="store_true", help="Generate a cited answer using Gemini"
    )
    parser.add_argument("--db", type=Path, default=DATABASE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--output", type=Path, help="Optional JSON result including source excerpts"
    )
    args = parser.parse_args()
    if args.output:
        distinct_paths(args.manifest, args.output)
    gemini = Gemini(Settings.from_env())
    try:
        store = open_store(args.db, args.manifest, gemini)
        result = {
            "created_at_utc": now(),
            "collection": store.name,
            **search(store, gemini, args.query, args.k),
        }
        if args.answer:
            result["answer"] = gemini.answer(args.query, result["hits"])
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        gemini.close()


if __name__ == "__main__":
    run(main)
