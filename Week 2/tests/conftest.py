"""Offline doubles are test-only; production commands always use Gemini."""

import json
from types import SimpleNamespace

import pytest

from src.chunking import chunk_document


@pytest.fixture
def chunks():
    return [
        chunk_document({"id": ident, "text": text, "title": title})[0]
        for ident, title, text in [
            ("space", "Astronomy", "A planet follows an orbit around a star."),
            ("food", "Baking", "Bread is baked using flour and yeast."),
            ("music", "Music", "A musician plays melodies on a piano."),
        ]
    ]


@pytest.fixture
def chunk_file(tmp_path, chunks):
    path = tmp_path / "chunks.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in chunks), encoding="utf-8")
    return path


class FakeGemini:
    """Small deterministic vectors exercise real Chroma behavior without API requests."""

    def __init__(self, fail_on=None):
        self.settings = SimpleNamespace(embedding_model="gemini-embedding-001", dimensions=3)
        self.calls = 0
        self.fail_on = fail_on

    def embed(self, texts, *, query=False):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("Simulated interrupted embedding request")
        vectors = []
        for text in texts:
            text = text.lower()
            vectors.append(
                [
                    1.0 if any(w in text for w in ("orbit", "planet")) else 0.01,
                    1.0 if any(w in text for w in ("bread", "flour")) else 0.01,
                    1.0 if any(w in text for w in ("music", "piano")) else 0.01,
                ]
            )
        return vectors, 0.001


@pytest.fixture
def fake_gemini():
    return FakeGemini()
