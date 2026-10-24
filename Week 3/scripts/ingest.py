"""Embed chunks with Gemini and resume ingestion into persistent ChromaDB."""

import argparse
from pathlib import Path

from src.cli import CHUNKS, DATABASE, MANIFEST, run
from src.gemini import ROOT, Gemini, Settings
from src.retrieval import ingest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--db", type=Path, default=DATABASE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/ingestion_report.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--request-interval",
        type=float,
        default=12,
        help="Minimum seconds between batches; increase for lower API quotas",
    )
    args = parser.parse_args()
    gemini = Gemini(Settings.from_env())
    try:
        report = ingest(
            args.chunks,
            args.db,
            args.manifest,
            args.report,
            gemini,
            args.batch_size,
            args.request_interval,
        )
    finally:
        gemini.close()
    print(f"Ready: {report['collection']} ({report['stored_chunks']} chunks)")


if __name__ == "__main__":
    run(main)
