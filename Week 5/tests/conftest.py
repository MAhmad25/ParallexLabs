"""Shared pytest fixtures — mock embedder, patched collection, and TestClient."""

import math
import random
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fake embedder that returns deterministic unit vectors
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """Returns random but deterministic 768-d unit vectors without calling Gemini."""

    def __init__(self):
        self.settings = MagicMock()
        self.settings.embedding_model = "gemini-embedding-001"
        self.settings.dimensions = 768
        self.request_count = 0

    def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        vectors = []
        for text in texts:
            rng = random.Random(hash(text) & 0xFFFFFFFF)
            raw = [rng.gauss(0, 1) for _ in range(768)]
            norm = math.hypot(*raw)
            vectors.append([v / norm for v in raw])
        self.request_count += len(texts)
        return vectors

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Fake Chroma collection
# ---------------------------------------------------------------------------

FAKE_CHUNKS = [
    {
        "id": f"chunk_{i}",
        "text": f"This is the text content of chunk number {i} about topic {i % 3}.",
        "metadata": {
            "document_id": str(i // 5),
            "title": f"Article {i // 5}",
            "url": f"https://example.com/article/{i // 5}",
            "source": "test",
            "chunk_index": i % 5,
            "start_char": i * 100,
            "end_char": (i + 1) * 100,
            "document_sha256": "abcd1234",
            "topic_id": i % 3,
            "topic_label": f"Topic {i % 3}",
            "topic_confidence": 0.8,
            "topic_status": "assigned",
            "sentiment": ["positive", "negative", "neutral"][i % 3],
            "sentiment_compound": [0.5, -0.5, 0.0][i % 3],
            "sentiment_positive": 0.3,
            "sentiment_neutral": 0.5,
            "sentiment_negative": 0.2,
            "sentiment_status": "scored",
            "topic_distribution_json": "[0.33, 0.33, 0.34]",
        },
    }
    for i in range(15)
]

FAKE_MANIFEST = {
    "collection": "week4_test_collection",
    "identity": {
        "schema_version": 1,
        "input_sha256": "test_hash",
        "embedding_model": "gemini-embedding-001",
        "dimensions": 768,
        "topic_method": "sklearn-lda-count-v1",
        "sentiment_method": "vader-lexicon-rule-v1",
        "provider": "gemini",
        "source_chunks_sha256": "test_source_hash",
    },
    "chunk_count": 15,
    "distance_metric": "cosine",
}


class FakeCollection:
    """In-memory stand-in for the Chroma collection."""

    def __init__(self, chunks: list[dict] = None):
        self._chunks = chunks or FAKE_CHUNKS
        self.metadata = {
            **FAKE_MANIFEST["identity"],
            "status": "ready",
            "expected_chunks": len(self._chunks),
        }
        self.name = FAKE_MANIFEST["collection"]

    def count(self):
        return len(self._chunks)

    def query(self, *, query_embeddings, n_results, include, where=None):
        results = list(self._chunks)

        # Apply where filter (simplified)
        if where:
            results = self._apply_filter(results, where)

        results = results[:n_results]

        return {
            "ids": [[c["id"] for c in results]],
            "documents": [[c["text"] for c in results]],
            "metadatas": [[c["metadata"] for c in results]],
            "distances": [[0.1 * (i + 1) for i in range(len(results))]],
        }

    def _apply_filter(self, chunks, where):
        if "$and" in where:
            for clause in where["$and"]:
                chunks = self._apply_filter(chunks, clause)
            return chunks
        for key, condition in where.items():
            if isinstance(condition, dict) and "$eq" in condition:
                value = condition["$eq"]
                chunks = [c for c in chunks if c["metadata"].get(key) == value]
        return chunks


# ---------------------------------------------------------------------------
# TestClient fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """Create a FastAPI TestClient with mocked embedder and collection."""
    fake_embedder = FakeEmbedder()
    fake_collection = FakeCollection()

    with (
        patch("src.app.Settings") as mock_settings_cls,
        patch("src.app.GeminiEmbedder") as mock_embedder_cls,
        patch("src.app.open_collection") as mock_open,
    ):
        mock_settings_cls.from_env.return_value = fake_embedder.settings
        mock_embedder_cls.return_value = fake_embedder
        mock_open.return_value = (fake_collection, FAKE_MANIFEST)

        from src.app import app

        with TestClient(app, raise_server_exceptions=False) as test_client:
            # Override state directly to ensure fixtures are used
            app.state.embedder = fake_embedder
            app.state.collection = fake_collection
            app.state.manifest = FAKE_MANIFEST
            app.state.settings = fake_embedder.settings
            yield test_client
