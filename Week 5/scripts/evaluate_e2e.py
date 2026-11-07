"""End-to-end evaluation — generation quality and latency via the FastAPI service.

Exercises the live ``/query`` endpoint with a set of test queries, measuring:
- Response structure validity (correct schema, non-empty hits)
- Latency percentiles (p50, p95, p99)
- Filter correctness (filtered results match the requested filter)

Usage (start the server in another terminal first)::

    uv run --locked uvicorn src.app:app --port 8000
    uv run --locked python -m scripts.evaluate_e2e
    uv run --locked python -m scripts.evaluate_e2e --base-url http://localhost:8000 --k 5
"""

import argparse
import json
import statistics
import sys
from time import perf_counter

import httpx

from src.io_utils import write_json
from src.paths import REPORTS_DIR


TEST_QUERIES = [
    {"question": "What is anarchism and what are its origins?", "k": 5},
    {"question": "What does the corpus say about climate and albedo?", "k": 5},
    {"question": "Tell me about Alabama history and government", "k": 5},
    {"question": "Who was Abraham Lincoln?", "k": 5},
    {"question": "What is the Academy Awards ceremony?", "k": 5},
    {"question": "Tell me about Achilles in Greek mythology", "k": 3},
    {"question": "What is the philosophy of Aristotle?", "k": 5},
    # Filtered queries
    {
        "question": "What does the corpus say about political theory?",
        "k": 5,
        "sentiment": "negative",
    },
    {
        "question": "Tell me about radiation and climate science",
        "k": 5,
        "topic_id": 4,
    },
    {
        "question": "History of the United States",
        "k": 3,
        "sentiment": "positive",
    },
]


def check_response_structure(data: dict, query: dict) -> list[str]:
    """Validate that the response matches the expected schema."""
    issues = []
    for key in ("question", "filters", "count", "hits"):
        if key not in data:
            issues.append(f"Missing top-level key: {key}")
    if data.get("question") != query["question"]:
        issues.append("Returned question does not match request")
    if not isinstance(data.get("hits"), list):
        issues.append("hits is not a list")
        return issues
    if data.get("count") != len(data["hits"]):
        issues.append(f"count ({data.get('count')}) != len(hits) ({len(data['hits'])})")
    for i, hit in enumerate(data["hits"]):
        for field in ("rank", "id", "title", "topic_id", "topic_label", "sentiment", "cosine_similarity", "text"):
            if field not in hit:
                issues.append(f"Hit {i} missing field: {field}")
    return issues


def check_filter_correctness(data: dict, query: dict) -> list[str]:
    """Verify that filtered results actually match the requested filter."""
    issues = []
    for hit in data.get("hits", []):
        if "sentiment" in query and hit.get("sentiment") != query["sentiment"]:
            issues.append(
                f"Hit {hit.get('id', '?')} sentiment={hit.get('sentiment')} "
                f"but filter requested {query['sentiment']}"
            )
        if "topic_id" in query and hit.get("topic_id") != query["topic_id"]:
            issues.append(
                f"Hit {hit.get('id', '?')} topic_id={hit.get('topic_id')} "
                f"but filter requested {query['topic_id']}"
            )
    return issues


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--k", type=int, default=None, help="Override k for all queries")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.base_url, timeout=args.timeout)

    # Health check
    try:
        health = client.get("/health")
        health.raise_for_status()
        print(f"✓ Health check passed: {health.json()}")
    except Exception as exc:
        print(f"✗ Health check failed: {exc}", file=sys.stderr)
        print("  Is the server running? Start with: uv run uvicorn src.app:app --port 8000", file=sys.stderr)
        sys.exit(1)

    # Collection stats
    stats_resp = client.get("/collection/stats")
    stats_resp.raise_for_status()
    stats = stats_resp.json()
    print(f"✓ Collection: {stats['collection']}  chunks={stats['chunk_count']}  "
          f"model={stats['embedding_model']}  dims={stats['embedding_dimensions']}")

    per_query = []
    latencies = []
    total_issues = 0

    for i, query in enumerate(TEST_QUERIES, 1):
        payload = {"question": query["question"], "k": args.k or query.get("k", 5)}
        if "sentiment" in query:
            payload["sentiment"] = query["sentiment"]
        if "topic_id" in query:
            payload["topic_id"] = query["topic_id"]
        if "topic_label" in query:
            payload["topic_label"] = query["topic_label"]

        start = perf_counter()
        resp = client.post("/query", json=payload)
        elapsed = perf_counter() - start

        latencies.append(elapsed)

        if resp.status_code != 200:
            issues = [f"HTTP {resp.status_code}: {resp.text[:200]}"]
        else:
            data = resp.json()
            issues = check_response_structure(data, query)
            issues += check_filter_correctness(data, query)

        status = "pass" if not issues else "fail"
        total_issues += len(issues)
        hit_count = data.get("count", 0) if resp.status_code == 200 else 0
        symbol = "✓" if not issues else "✗"

        print(f"{symbol} Query {i:2d} [{elapsed*1000:7.1f}ms] hits={hit_count:2d}  "
              f"{query['question'][:50]}{'...' if len(query['question']) > 50 else ''}")
        if issues:
            for issue in issues:
                print(f"    ⚠ {issue}")

        per_query.append({
            "query_index": i,
            "question": query["question"],
            "filters": {k: v for k, v in query.items() if k not in ("question", "k")},
            "k": payload["k"],
            "status": status,
            "http_status": resp.status_code,
            "hit_count": hit_count,
            "latency_seconds": round(elapsed, 4),
            "issues": issues,
        })

    # Latency statistics
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    latency_stats = {
        "count": n,
        "min_ms": round(min(latencies) * 1000, 1),
        "max_ms": round(max(latencies) * 1000, 1),
        "mean_ms": round(statistics.mean(latencies) * 1000, 1),
        "median_ms": round(statistics.median(latencies) * 1000, 1),
        "p95_ms": round(latencies_sorted[int(n * 0.95)] * 1000, 1) if n >= 2 else round(latencies_sorted[-1] * 1000, 1),
        "p99_ms": round(latencies_sorted[int(n * 0.99)] * 1000, 1) if n >= 2 else round(latencies_sorted[-1] * 1000, 1),
    }

    passed = sum(1 for q in per_query if q["status"] == "pass")
    failed = sum(1 for q in per_query if q["status"] == "fail")

    report = {
        "status": "complete",
        "base_url": args.base_url,
        "query_count": n,
        "passed": passed,
        "failed": failed,
        "total_issues": total_issues,
        "latency": latency_stats,
        "collection": stats,
        "per_query": per_query,
    }

    output = REPORTS_DIR / "e2e_evaluation.json"
    write_json(output, report)

    print(f"\n{'='*60}")
    print(f"  Queries: {n}   Passed: {passed}   Failed: {failed}")
    print(f"  Latency — mean: {latency_stats['mean_ms']:.1f}ms  "
          f"p50: {latency_stats['median_ms']:.1f}ms  "
          f"p95: {latency_stats['p95_ms']:.1f}ms  "
          f"p99: {latency_stats['p99_ms']:.1f}ms")
    print(f"  Report: {output}")
    print(f"{'='*60}")

    client.close()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
