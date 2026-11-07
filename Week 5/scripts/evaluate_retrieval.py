"""Retrieval evaluation — Precision@K and Recall@K on a gold-standard test set.

Gold file format (``queries/retrieval_gold.jsonl``)::

    {"query": "...", "relevant_ids": ["chunk_id_1", "chunk_id_2", ...]}

Usage::

    uv run --locked python -m scripts.evaluate_retrieval
    uv run --locked python -m scripts.evaluate_retrieval --k 10
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from src.gemini import GeminiEmbedder, Settings
from src.io_utils import read_jsonl, write_json
from src.paths import INDEX_DIR, INDEX_MANIFEST, REPORTS_DIR, RETRIEVAL_GOLD
from src.vector_store import open_collection, query_collection


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Fraction of retrieved documents that are relevant."""
    if not retrieved_ids:
        return 0.0
    return sum(1 for rid in retrieved_ids if rid in relevant_ids) / len(retrieved_ids)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Fraction of relevant documents that were retrieved."""
    if not relevant_ids:
        return 0.0
    return sum(1 for rid in retrieved_ids if rid in relevant_ids) / len(relevant_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5, help="Number of top results to evaluate")
    parser.add_argument("--gold", type=Path, default=RETRIEVAL_GOLD, help="Gold-standard JSONL file")
    parser.add_argument("--db", type=Path, default=INDEX_DIR)
    parser.add_argument("--manifest", type=Path, default=INDEX_MANIFEST)
    args = parser.parse_args()

    if not args.gold.exists():
        print(f"ERROR: Gold file not found: {args.gold}", file=sys.stderr)
        print("Create queries/retrieval_gold.jsonl with {query, relevant_ids} rows.", file=sys.stderr)
        sys.exit(1)

    gold_rows = list(read_jsonl(args.gold))
    if not gold_rows:
        print("ERROR: Gold file is empty.", file=sys.stderr)
        sys.exit(1)

    for i, row in enumerate(gold_rows, 1):
        if "query" not in row or "relevant_ids" not in row:
            print(f"ERROR: Gold row {i} must have 'query' and 'relevant_ids' keys.", file=sys.stderr)
            sys.exit(1)
        if not isinstance(row["relevant_ids"], list) or not row["relevant_ids"]:
            print(f"ERROR: Gold row {i} 'relevant_ids' must be a non-empty list.", file=sys.stderr)
            sys.exit(1)

    collection, manifest = open_collection(args.db, args.manifest)
    settings = Settings.from_env()
    embedder = GeminiEmbedder(settings)

    per_query_results = []
    total_precision = 0.0
    total_recall = 0.0
    total_latency = 0.0

    try:
        for row in gold_rows:
            query_text = row["query"]
            relevant = set(row["relevant_ids"])

            start = perf_counter()
            query_vector = embedder.embed([query_text], query=True)[0]
            hits = query_collection(collection, query_vector, k=args.k)
            elapsed = perf_counter() - start

            retrieved_ids = [hit["id"] for hit in hits]
            p_at_k = precision_at_k(retrieved_ids, relevant)
            r_at_k = recall_at_k(retrieved_ids, relevant)

            per_query_results.append({
                "query": query_text,
                "k": args.k,
                "relevant_count": len(relevant),
                "retrieved_ids": retrieved_ids,
                "true_positives": [rid for rid in retrieved_ids if rid in relevant],
                "precision_at_k": round(p_at_k, 4),
                "recall_at_k": round(r_at_k, 4),
                "latency_seconds": round(elapsed, 4),
            })
            total_precision += p_at_k
            total_recall += r_at_k
            total_latency += elapsed
    finally:
        embedder.close()

    n = len(per_query_results)
    report = {
        "status": "complete",
        "k": args.k,
        "query_count": n,
        "mean_precision_at_k": round(total_precision / n, 4) if n else 0.0,
        "mean_recall_at_k": round(total_recall / n, 4) if n else 0.0,
        "total_latency_seconds": round(total_latency, 4),
        "mean_latency_seconds": round(total_latency / n, 4) if n else 0.0,
        "collection": manifest["collection"],
        "embedding_model": manifest["identity"]["embedding_model"],
        "per_query": per_query_results,
    }

    output = REPORTS_DIR / "retrieval_evaluation.json"
    write_json(output, report)

    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport written to {output}")


if __name__ == "__main__":
    main()
