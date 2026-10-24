"""Chunk Week 1's cleaned corpus without modifying it."""

import argparse
from pathlib import Path

from src.chunking import ChunkConfig, chunk_corpus
from src.cli import CHUNKS, run
from src.gemini import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT.parent / "Week 1/data/processed/clean_corpus.jsonl"
    )
    parser.add_argument("--output", type=Path, default=CHUNKS)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/chunking_report.json")
    parser.add_argument("--chunk-size", type=int, default=1500)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--limit", type=int, default=10, help="First N documents; 0 means all")
    args = parser.parse_args()
    result = chunk_corpus(
        args.input, args.output, args.report, ChunkConfig(args.chunk_size, args.overlap), args.limit
    )
    print(
        f"Chunked {result['document_count']} documents into {result['chunk_count']} chunks "
        f"in {result['chunking_seconds']:.3f}s"
    )


if __name__ == "__main__":
    run(main)
