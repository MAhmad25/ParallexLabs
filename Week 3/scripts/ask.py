"""Answer a question from the Week 3 Gemini and ChromaDB index."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from src.cli import DATABASE, MANIFEST, ROOT, run
from src.gemini import Gemini, Settings
from src.latency import append_latency
from src.retrieval import ask, open_store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question about the indexed Week 1 articles")
    parser.add_argument("--k", type=int, default=5, help="Number of excerpts to retrieve (1-10)")
    parser.add_argument(
        "--min-similarity", type=float, help="Override the configured refusal threshold"
    )
    parser.add_argument("--db", type=Path, default=DATABASE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--latency-log",
        type=Path,
        default=ROOT / "reports" / "latency.jsonl",
        help="Append one privacy-conscious timing record per request",
    )
    args = parser.parse_args()
    if not 1 <= args.k <= 10:
        raise ValueError("k must be between 1 and 10")
    if not args.question.strip():
        raise ValueError("Question must not be empty")
    if len(args.question) > 2000:
        raise ValueError("Question exceeds the 2,000-character limit")

    started = perf_counter()
    log_result = {"status": "error"}
    gemini = None
    try:
        gemini = Gemini(Settings.from_env())
        store = open_store(args.db, args.manifest, gemini)
        result = ask(store, gemini, args.question, args.k, args.min_similarity)
        log_result = result
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if result["status"] == "error":
            raise RuntimeError(result["error"])
    except Exception as error:
        log_result["error_type"] = type(error).__name__
        raise
    finally:
        if "end_to_end_ms" not in log_result:
            log_result["end_to_end_ms"] = (perf_counter() - started) * 1000
        try:
            append_latency(args.latency_log, args.question, log_result)
        except OSError as error:
            print(
                f"Warning: unable to write the latency log ({type(error).__name__}).",
                file=sys.stderr,
            )
        finally:
            if gemini is not None:
                gemini.close()


if __name__ == "__main__":
    run(main)
