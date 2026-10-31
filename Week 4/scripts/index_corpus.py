"""Store enriched Week 3 chunks in a local ChromaDB collection."""

import argparse
from pathlib import Path

from src.gemini import GeminiEmbedder, Settings
from src.paths import ENRICHED_CHUNKS, INDEX_DIR, INDEX_MANIFEST, INDEX_REPORT
from src.vector_store import build_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=ENRICHED_CHUNKS)
    parser.add_argument("--db", type=Path, default=INDEX_DIR)
    parser.add_argument("--manifest", type=Path, default=INDEX_MANIFEST)
    parser.add_argument("--report", type=Path, default=INDEX_REPORT)
    parser.add_argument("--request-interval", type=float, default=12)
    args = parser.parse_args()
    settings = Settings.from_env()
    # This client performs no request when the Week 2 vectors can be reused.
    embedder = GeminiEmbedder(settings)
    try:
        report = build_index(
            args.chunks,
            db_path=args.db,
            manifest_path=args.manifest,
            report_path=args.report,
            embedding_model=settings.embedding_model,
            dimensions=settings.dimensions,
            embedder=embedder,
            request_interval=args.request_interval,
        )
    finally:
        if embedder:
            embedder.close()
    print(f"Ready: {report['collection']} ({report['chunk_count']} chunks)")
    if report["vector_provenance"]["reused_embeddings"]:
        print(f"Reused {report['vector_provenance']['reused_embeddings']} local Gemini vectors")


if __name__ == "__main__":
    main()
