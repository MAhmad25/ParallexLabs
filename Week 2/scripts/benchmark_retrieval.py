"""Benchmark query embedding, vector lookup, total latency, and labeled retrieval."""

import argparse
from pathlib import Path

from src.benchmark import benchmark, load_queries
from src.cli import DATABASE, MANIFEST, run
from src.gemini import ROOT, Gemini, Settings
from src.io_utils import distinct_paths
from src.retrieval import open_store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=ROOT / "queries/benchmark.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/retrieval_benchmark.json")
    parser.add_argument("--db", type=Path, default=DATABASE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()
    distinct_paths(args.queries, args.manifest, args.output)
    queries = load_queries(args.queries)
    gemini = Gemini(Settings.from_env())
    try:
        report = benchmark(
            open_store(args.db, args.manifest, gemini),
            gemini,
            queries,
            args.output,
            args.repeats,
            args.warmups,
            tuple(args.k),
        )
    finally:
        gemini.close()
    total = report["latency_ms"]["total_ms"]
    print(f"{total['count']} measurements; mean {total['mean']:.1f}ms; p95 {total['p95']:.1f}ms")


if __name__ == "__main__":
    run(main)
