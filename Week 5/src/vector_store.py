"""Query-side ChromaDB operations for the Week 4 filtered retrieval index."""

import json
import math
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from src.io_utils import read_jsonl
from src.paths import INDEX_DIR, INDEX_MANIFEST


def connect(path: Path):
    return chromadb.PersistentClient(
        path=str(path.resolve()),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def open_collection(db_path: Path = INDEX_DIR, manifest_path: Path = INDEX_MANIFEST):
    """Open the completed Week 4 index for querying."""
    if not manifest_path.exists():
        raise ValueError("No completed Week 4 index manifest; run Week 4 scripts.index_corpus first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest["identity"]

    def _collection_name(ident: dict) -> str:
        import hashlib
        payload = json.dumps(ident, sort_keys=True, separators=(",", ":")).encode()
        return "week4_" + hashlib.sha256(payload).hexdigest()[:24]

    if manifest["collection"] != _collection_name(identity):
        raise ValueError("Week 4 manifest collection does not match its identity")
    client = connect(db_path)
    collection = client.get_collection(manifest["collection"], embedding_function=None)
    if any((collection.metadata or {}).get(key) != value for key, value in identity.items()):
        raise ValueError("Week 4 collection metadata differs from its manifest")
    if collection.count() != manifest["chunk_count"]:
        raise ValueError("Week 4 collection is incomplete; rerun Week 4 scripts.index_corpus")
    if (collection.metadata or {}).get("status") != "ready":
        raise ValueError("Week 4 collection is not marked ready")
    return collection, manifest


def build_where(
    *, topic_id: int | None = None, topic_label: str | None = None, sentiment: str | None = None
):
    """Build a Chroma metadata filter from optional field constraints."""
    clauses = []
    if topic_id is not None:
        if topic_id < 0:
            raise ValueError("Topic ID must be nonnegative")
        clauses.append({"topic_id": {"$eq": topic_id}})
    if topic_label is not None:
        if not topic_label.strip():
            raise ValueError("Topic label filter must not be empty")
        clauses.append({"topic_label": {"$eq": topic_label.strip()}})
    if sentiment is not None:
        if sentiment not in {"positive", "neutral", "negative", "unknown"}:
            raise ValueError("Sentiment must be positive, neutral, negative, or unknown")
        clauses.append({"sentiment": {"$eq": sentiment}})
    if len(clauses) == 1:
        return clauses[0]
    if clauses:
        return {"$and": clauses}
    return None


def query_collection(
    collection, vector: list[float], *, k: int, where: dict | None = None
) -> list[dict]:
    """Retrieve the top-k chunks by cosine similarity, optionally filtered."""
    if not 1 <= k <= 50:
        raise ValueError("k must be between 1 and 50")
    kwargs = {
        "query_embeddings": [vector],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    result = collection.query(**kwargs)
    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    return [
        {
            "id": ident,
            "text": document,
            "metadata": metadata,
            "cosine_similarity": 1.0 - float(distance),
        }
        for ident, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        )
    ]
