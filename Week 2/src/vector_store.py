"""Persistent cosine collections with explicit vectors and immutable snapshot identity."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from src.gemini import normalize_vector


def snapshot_identity(input_sha256: str, model: str, dimensions: int) -> dict:
    return {
        "input_sha256": input_sha256,
        "embedding_model": model,
        "dimensions": dimensions,
        "provider": "gemini",
        "format_version": 1,
    }


def collection_name(identity: dict) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return "week2_" + hashlib.sha256(encoded).hexdigest()[:24]


def validate_chunk(chunk: dict) -> None:
    for field in ("id", "document_id", "text", "title", "url", "source", "document_sha256"):
        if not isinstance(chunk.get(field), str):
            raise ValueError(f"Chunk field {field} must be a string")
    if not chunk["id"] or not chunk["document_id"] or not chunk["text"].strip():
        raise ValueError("Chunk id, document id, and text must be nonempty")
    for field in ("chunk_index", "start_char", "end_char"):
        if type(chunk.get(field)) is not int or chunk[field] < 0:
            raise ValueError(f"Chunk field {field} must be a nonnegative integer")
    if chunk["end_char"] - chunk["start_char"] != len(chunk["text"]):
        raise ValueError("Chunk offsets and text length disagree")
    if len(chunk["text"]) > 8000:
        raise ValueError("Chunk exceeds the conservative 8000-character API guard")


class VectorStore:
    def __init__(self, path: Path, identity: dict, *, expected: int | None = None):
        self.identity = identity
        self.name = collection_name(identity)
        self.dimensions = identity["dimensions"]
        self.client = chromadb.PersistentClient(
            path=str(path.resolve()),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        try:
            self.collection = self.client.get_collection(self.name, embedding_function=None)
        except NotFoundError:
            if expected is None:
                raise ValueError("Collection is missing; run ingestion first") from None
            if expected <= 0:
                raise ValueError("Cannot create a collection for an empty corpus") from None
            self.collection = self.client.create_collection(
                self.name,
                embedding_function=None,
                configuration={"hnsw": {"space": "cosine"}},
                metadata={**identity, "status": "building", "expected_chunks": expected},
            )
        metadata = self.collection.metadata or {}
        if any(metadata.get(key) != value for key, value in identity.items()):
            raise ValueError("Collection metadata mismatch; use the matching snapshot")
        if expected is not None and metadata.get("expected_chunks") != expected:
            raise ValueError("Collection chunk count metadata mismatch")
        if self.collection.configuration.get("hnsw", {}).get("space") != "cosine":
            raise ValueError("Collection must use cosine distance")

    def count(self) -> int:
        return self.collection.count()

    def require_ready(self):
        metadata = self.collection.metadata or {}
        if metadata.get("status") != "ready" or self.count() != metadata.get("expected_chunks"):
            raise ValueError("Collection is incomplete; rerun ingestion to resume")

    def missing(self, chunks: list[dict]) -> list[dict]:
        ids = [chunk["id"] for chunk in chunks]
        if not ids:
            return []
        found = set(self.collection.get(ids=ids, include=[])["ids"])
        return [chunk for chunk in chunks if chunk["id"] not in found]

    def upsert(self, chunks: list[dict], vectors: list[list[float]]):
        if len(chunks) != len(vectors):
            raise ValueError("Chunk and embedding counts must match")
        if not chunks:
            return
        for chunk in chunks:
            validate_chunk(chunk)
        ids = [chunk["id"] for chunk in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate chunk IDs within batch")
        normalized = [normalize_vector(vector, self.dimensions) for vector in vectors]
        maximum = self.client.get_max_batch_size()
        for offset in range(0, len(chunks), maximum):
            batch = chunks[offset : offset + maximum]
            self.collection.upsert(
                ids=[chunk["id"] for chunk in batch],
                documents=[chunk["text"] for chunk in batch],
                embeddings=normalized[offset : offset + maximum],
                metadatas=[
                    {k: v for k, v in chunk.items() if k not in {"id", "text"}} for chunk in batch
                ],
            )

    def mark_ready(self):
        metadata = self.collection.metadata or {}
        if self.count() != metadata.get("expected_chunks"):
            raise ValueError("Collection is missing chunks; resume ingestion")
        self.collection.modify(metadata={**metadata, "status": "ready"})

    def query(self, vector: list[float], k: int = 5) -> tuple[list[dict], float]:
        if k < 1:
            raise ValueError("k must be >= 1")
        vector = normalize_vector(vector, self.dimensions)
        started = perf_counter()
        self.require_ready()
        count = self.count()
        if count == 0:
            return [], (perf_counter() - started) * 1000
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )
        hits = [
            {
                "id": ident,
                "text": text,
                "metadata": metadata,
                "distance": distance,
                "cosine_similarity": 1 - distance,
            }
            for ident, text, metadata, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]
        return hits, (perf_counter() - started) * 1000
