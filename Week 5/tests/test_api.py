"""Unit tests for FastAPI endpoints using pytest."""

import pytest


# ------------------------------------------------------------------ health

class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["collection"] == "week4_test_collection"
        assert body["chunk_count"] == 15

    def test_health_has_request_id(self, client):
        resp = client.get("/health")
        assert "x-request-id" in resp.headers


# --------------------------------------------------------- collection stats

class TestCollectionStats:
    def test_stats_returns_metadata(self, client):
        resp = client.get("/collection/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["collection"] == "week4_test_collection"
        assert body["chunk_count"] == 15
        assert body["embedding_model"] == "gemini-embedding-001"
        assert body["embedding_dimensions"] == 768
        assert body["distance_metric"] == "cosine"


# --------------------------------------------------------------- query

class TestQuery:
    def test_valid_query(self, client):
        resp = client.post("/query", json={"question": "What is anarchism?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["question"] == "What is anarchism?"
        assert body["count"] == len(body["hits"])
        assert body["count"] <= 5  # default k
        for i, hit in enumerate(body["hits"], 1):
            assert hit["rank"] == i
            assert "id" in hit
            assert "text" in hit
            assert "title" in hit
            assert "topic_id" in hit
            assert "sentiment" in hit
            assert "cosine_similarity" in hit

    def test_query_with_k(self, client):
        resp = client.post("/query", json={"question": "test query", "k": 3})
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] <= 3

    def test_query_with_sentiment_filter(self, client):
        resp = client.post("/query", json={
            "question": "test query",
            "sentiment": "positive",
        })
        assert resp.status_code == 200
        body = resp.json()
        for hit in body["hits"]:
            assert hit["sentiment"] == "positive"

    def test_query_with_topic_filter(self, client):
        resp = client.post("/query", json={
            "question": "test query",
            "topic_id": 1,
        })
        assert resp.status_code == 200
        body = resp.json()
        for hit in body["hits"]:
            assert hit["topic_id"] == 1

    def test_query_with_combined_filters(self, client):
        resp = client.post("/query", json={
            "question": "test query",
            "topic_id": 0,
            "sentiment": "positive",
        })
        assert resp.status_code == 200
        body = resp.json()
        for hit in body["hits"]:
            assert hit["topic_id"] == 0
            assert hit["sentiment"] == "positive"

    def test_query_returns_filters_in_response(self, client):
        resp = client.post("/query", json={
            "question": "test",
            "sentiment": "negative",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["filters"] == {"sentiment": {"$eq": "negative"}}

    def test_query_empty_filters(self, client):
        resp = client.post("/query", json={"question": "test"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["filters"] == {}

    def test_cosine_similarity_ordering(self, client):
        resp = client.post("/query", json={"question": "ordering test", "k": 5})
        assert resp.status_code == 200
        body = resp.json()
        similarities = [hit["cosine_similarity"] for hit in body["hits"]]
        # The fake collection returns increasing distances, so similarities decrease
        assert similarities == sorted(similarities, reverse=True)


# -------------------------------------------------------- validation errors (422)

class TestValidationErrors:
    def test_missing_question(self, client):
        resp = client.post("/query", json={})
        assert resp.status_code == 422
        assert "detail" in resp.json()

    def test_empty_question(self, client):
        resp = client.post("/query", json={"question": ""})
        assert resp.status_code == 422

    def test_question_too_long(self, client):
        resp = client.post("/query", json={"question": "x" * 2001})
        assert resp.status_code == 422

    def test_invalid_sentiment(self, client):
        resp = client.post("/query", json={
            "question": "test",
            "sentiment": "very_happy",
        })
        assert resp.status_code == 422

    def test_k_too_small(self, client):
        resp = client.post("/query", json={"question": "test", "k": 0})
        assert resp.status_code == 422

    def test_k_too_large(self, client):
        resp = client.post("/query", json={"question": "test", "k": 51})
        assert resp.status_code == 422

    def test_negative_topic_id(self, client):
        resp = client.post("/query", json={"question": "test", "topic_id": -1})
        assert resp.status_code == 422


# ----------------------------------------------------------- bad requests (400)

class TestBadRequests:
    def test_invalid_json_body(self, client):
        resp = client.post(
            "/query",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422


# -------------------------------------------------------- server errors (500)

class TestServerErrors:
    def test_internal_error_returns_500(self, client):
        """Simulate an unexpected error during query processing."""
        original_embed = client.app.state.embedder.embed

        def broken_embed(*args, **kwargs):
            raise RuntimeError("Simulated Gemini failure")

        client.app.state.embedder.embed = broken_embed
        try:
            resp = client.post("/query", json={"question": "trigger error"})
            assert resp.status_code == 500
            body = resp.json()
            assert "detail" in body
            assert "internal" in body["detail"].lower() or "error" in body["detail"].lower()
        finally:
            client.app.state.embedder.embed = original_embed


# ----------------------------------------------------- response headers

class TestResponseHeaders:
    def test_request_id_header_on_query(self, client):
        resp = client.post("/query", json={"question": "header test"})
        assert "x-request-id" in resp.headers

    def test_request_id_header_on_stats(self, client):
        resp = client.get("/collection/stats")
        assert "x-request-id" in resp.headers
