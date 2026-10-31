"""Search the Week 4 index with optional topic and sentiment metadata filters."""

import argparse
import json
import sys
from pathlib import Path

from src.gemini import GeminiEmbedder, Settings
from src.paths import INDEX_DIR, INDEX_MANIFEST
from src.vector_store import build_where, open_collection, query_collection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question or phrase to search for")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--topic-id", type=int, help="Keep chunks assigned to this LDA topic")
    parser.add_argument("--topic-label", help="Keep chunks with this reviewed topic label")
    parser.add_argument("--sentiment", choices=["positive", "neutral", "negative", "unknown"])
    parser.add_argument("--db", default=INDEX_DIR, type=Path)
    parser.add_argument("--manifest", default=INDEX_MANIFEST, type=Path)
    args = parser.parse_args()
    if not args.question.strip() or len(args.question) > 2000:
        raise ValueError("Question must contain 1 to 2000 characters")
    if not 1 <= args.k <= 50:
        raise ValueError("--k must be between 1 and 50")
    collection, manifest = open_collection(args.db, args.manifest)
    identity = manifest["identity"]
    settings = Settings.from_env()
    if (
        settings.embedding_model != identity["embedding_model"]
        or settings.dimensions != identity["dimensions"]
    ):
        raise ValueError("Gemini query settings differ from the indexed vectors")
    where = build_where(topic_id=args.topic_id, topic_label=args.topic_label, sentiment=args.sentiment)
    embedder = GeminiEmbedder(settings)
    try:
        query_vector = embedder.embed([args.question], query=True)[0]
        hits = query_collection(collection, query_vector, k=args.k, where=where)
    finally:
        embedder.close()
    result = {
        "question": args.question,
        "filters": where or {},
        "count": len(hits),
        "hits": [
            {
                "rank": rank,
                "id": hit["id"],
                "title": hit["metadata"].get("title", ""),
                "topic_id": hit["metadata"]["topic_id"],
                "topic_label": hit["metadata"]["topic_label"],
                "topic_confidence": hit["metadata"]["topic_confidence"],
                "sentiment": hit["metadata"]["sentiment"],
                "sentiment_compound": hit["metadata"]["sentiment_compound"],
                "cosine_similarity": hit["cosine_similarity"],
                "text": hit["text"],
            }
            for rank, hit in enumerate(hits, 1)
        ],
    }
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
