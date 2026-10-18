import json

import pytest

from src.benchmark import benchmark, latency_summary, load_queries, relevance
from src.retrieval import ingest, open_store, search


def test_latency_percentiles_and_empty_input():
    assert latency_summary(list(range(1, 101)))["p95"] == 95
    assert latency_summary([10])["median"] == 10
    assert latency_summary([])["count"] == 0
    assert latency_summary([])["mean"] is None


def test_relevance_counts_unique_documents():
    hits = [{"metadata": {"document_id": ident}} for ident in ["wrong", "a", "a", "b"]]
    metrics = relevance(hits, ["a", "b"], 3)
    assert metrics == {"hit": 1.0, "document_recall": 0.5, "reciprocal_rank": 0.5}
    assert relevance(hits, ["absent"], 5)["hit"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        [{"id": "x", "query": "?"}],
        [{"id": "x", "query": "", "relevant_document_ids": ["a"]}],
    ],
)
def test_invalid_queries(tmp_path, payload):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_queries(path)


def test_benchmark_real_chroma_and_test_only_embeddings(tmp_path, chunk_file, fake_gemini):
    db, manifest, report = (tmp_path / name for name in ("db", "manifest.json", "ingest.json"))
    ingest(chunk_file, db, manifest, report, fake_gemini)
    store = open_store(db, manifest, fake_gemini)
    queries = [
        {"id": "space", "query": "planet orbit", "relevant_document_ids": ["space"]},
        {"id": "food", "query": "bread flour", "relevant_document_ids": ["food"]},
    ]
    output = tmp_path / "benchmark.json"
    result = benchmark(store, fake_gemini, queries, output, repeats=2, warmups=1)
    assert result["status"] == "complete"
    assert len(result["samples"]) == 4
    assert result["latency_ms"]["total_ms"]["count"] == 4
    assert result["quality"]["1"]["hit"] == 1
    assert result["quality"]["3"]["document_recall"] == 1
    # One ingest request plus (one warmup + two measurements) for each query.
    assert fake_gemini.calls == 7
    assert json.loads(output.read_text())["status"] == "complete"


def test_absent_labels_fail_before_query_api(tmp_path, chunk_file, fake_gemini):
    db, manifest, report = (tmp_path / name for name in ("db", "manifest.json", "ingest.json"))
    ingest(chunk_file, db, manifest, report, fake_gemini)
    store = open_store(db, manifest, fake_gemini)
    calls = fake_gemini.calls
    with pytest.raises(ValueError, match="absent"):
        benchmark(
            store,
            fake_gemini,
            [{"id": "x", "query": "q", "relevant_document_ids": ["unknown"]}],
            tmp_path / "bench",
        )
    assert fake_gemini.calls == calls


def test_failed_benchmark_records_failure(tmp_path, chunk_file, fake_gemini):
    db, manifest, report = (tmp_path / name for name in ("db", "manifest.json", "ingest.json"))
    ingest(chunk_file, db, manifest, report, fake_gemini)
    store = open_store(db, manifest, fake_gemini)
    fake_gemini.fail_on = fake_gemini.calls + 1
    output = tmp_path / "bench.json"
    with pytest.raises(RuntimeError):
        benchmark(
            store,
            fake_gemini,
            [{"id": "x", "query": "q", "relevant_document_ids": ["space"]}],
            output,
        )
    result = json.loads(output.read_text())
    assert result["status"] == "failed"
    assert result["latency_ms"]["total_ms"]["count"] == 0
    assert result["quality"]["1"]["hit"] is None


@pytest.mark.parametrize("question,k", [(" ", 3), ("hello", 0)])
def test_invalid_search_does_not_spend_api_calls(fake_gemini, question, k):
    with pytest.raises(ValueError):
        search(None, fake_gemini, question, k)
    assert fake_gemini.calls == 0
