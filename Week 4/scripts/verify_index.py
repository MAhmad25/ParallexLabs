"""Check the saved Week 4 vectors, NLP fields, and metadata filter behavior."""

import argparse
from pathlib import Path

from src.paths import ENRICHED_CHUNKS, INDEX_DIR, INDEX_MANIFEST, INDEX_VERIFICATION
from src.vector_store import verify_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=ENRICHED_CHUNKS)
    parser.add_argument("--db", type=Path, default=INDEX_DIR)
    parser.add_argument("--manifest", type=Path, default=INDEX_MANIFEST)
    parser.add_argument("--output", type=Path, default=INDEX_VERIFICATION)
    args = parser.parse_args()
    report = verify_index(
        args.chunks,
        db_path=args.db,
        manifest_path=args.manifest,
        output_path=args.output,
    )
    print(f"Verified {report['checked_records']} records and metadata filters")


if __name__ == "__main__":
    main()
