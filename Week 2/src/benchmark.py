"""Repeated latency measurements and manually labeled document retrieval metrics."""

import json
import math
import statistics
from pathlib import Path

from src.gemini import Gemini
from src.io_utils import now, write_json
from src.retrieval import search
from src.vector_store import VectorStore


def latency_summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "max": max(values),
    }


def relevance(hits: list[dict], relevant_ids: list[str], k: int) -> dict:
    if not relevant_ids:
        raise ValueError("Each benchmark query needs at least one labeled document")
    relevant = set(relevant_ids)
    found = {hit["metadata"]["document_id"] for hit in hits[:k]} & relevant
    rank = next(
        (i for i, hit in enumerate(hits[:k], 1) if hit["metadata"]["document_id"] in relevant), None
    )
    return {
        "hit": float(bool(found)),
        "document_recall": len(found) / len(relevant),
        "reciprocal_rank": 1 / rank if rank else 0.0,
    }


def load_queries(path: Path) -> list[dict]:
    queries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(queries, list) or not queries:
        raise ValueError("Query file must be a nonempty JSON array")
    ids = set()
    for row in queries:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or not row["id"]
            or row["id"] in ids
            or not isinstance(row.get("query"), str)
            or not row["query"].strip()
            or not isinstance(row.get("relevant_document_ids"), list)
            or not row["relevant_document_ids"]
            or any(not isinstance(v, str) or not v for v in row["relevant_document_ids"])
        ):
            raise ValueError("Malformed, duplicate, or unlabeled benchmark query")
        ids.add(row["id"])
    return queries


def benchmark(
    store: VectorStore,
    gemini: Gemini,
    queries: list[dict],
    output: Path,
    repeats: int = 3,
    warmups: int = 1,
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict:
    if repeats < 1 or warmups < 0 or not ks or min(ks) < 1:
        raise ValueError("Require repeats >= 1, warmups >= 0, and positive k values")
    if not queries:
        raise ValueError("Benchmark needs queries")
    # Missing labels make quality metrics misleading: fail before spending API calls.
    required_ids = {v for query in queries for v in query["relevant_document_ids"]}
    indexed = store.collection.get(
        where={"document_id": {"$in": sorted(required_ids)}}, include=["metadatas"]
    )
    present = {metadata["document_id"] for metadata in indexed["metadatas"]}
    if required_ids - present:
        raise ValueError(
            "Benchmark labels refer to documents absent from this index; "
            "use the first 10 Week 1 documents or a matching query file"
        )
    report = {
        "created_at_utc": now(),
        "status": "running",
        "collection": store.name,
        "identity": store.identity,
        "indexed_chunks": store.count(),
        "repeats": repeats,
        "warmups_per_query": warmups,
        "ks": list(ks),
        "query_embedding_cache": False,
        "timing_unit": "milliseconds",
        "p95_method": "nearest-rank",
        "samples": [],
        "queries": [],
        "failures": [],
    }
    write_json(output, report)
    try:
        for query in queries:
            try:
                for _ in range(warmups):
                    search(store, gemini, query["query"], max(ks))
                samples = []
                for iteration in range(repeats):
                    result = search(store, gemini, query["query"], max(ks))
                    sample = {
                        "query_id": query["id"],
                        "iteration": iteration + 1,
                        **{
                            key: result[key]
                            for key in ("embedding_ms", "vector_search_ms", "total_ms")
                        },
                        "retrieved_document_ids": [
                            hit["metadata"]["document_id"] for hit in result["hits"]
                        ],
                        "metrics": {
                            str(k): relevance(result["hits"], query["relevant_document_ids"], k)
                            for k in ks
                        },
                    }
                    samples.append(sample)
                    report["samples"].append(sample)
                    write_json(output, report)
                report["queries"].append(
                    {
                        **query,
                        "latency_ms": {
                            key: latency_summary([s[key] for s in samples])
                            for key in ("embedding_ms", "vector_search_ms", "total_ms")
                        },
                    }
                )
                print(
                    f"{query['id']}: mean {statistics.mean(s['total_ms'] for s in samples):.1f}ms",
                    flush=True,
                )
            except Exception as error:
                report["failures"].append(
                    {"query_id": query["id"], "failure_type": type(error).__name__}
                )
                raise
        report["status"] = "complete"
    finally:
        if report["status"] != "complete":
            report["status"] = "failed"
        samples = report["samples"]
        report["latency_ms"] = {
            key: latency_summary([sample[key] for sample in samples])
            for key in ("embedding_ms", "vector_search_ms", "total_ms")
        }
        report["quality"] = {
            str(k): {
                metric: statistics.mean(s["metrics"][str(k)][metric] for s in samples)
                if samples
                else None
                for metric in ("hit", "document_recall", "reciprocal_rank")
            }
            for k in ks
        }
        report["finished_at_utc"] = now()
        write_json(output, report)
    return report
