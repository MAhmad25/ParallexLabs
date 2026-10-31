"""Persistent local ChromaDB index with scalar NLP metadata filters."""

import hashlib
import json
import math
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from src.io_utils import read_jsonl, sha256_file, write_json
from src.paths import (
    INDEX_DIR,
    INDEX_MANIFEST,
    INDEX_REPORT,
    INDEX_VERIFICATION,
    WEEK2_CHUNKS,
    WEEK2_DB,
    WEEK2_MANIFEST,
)

COLLECTION_PREFIX = "week4_"
BATCH_SIZE = 64
EMBED_BATCH_SIZE = 16


def index_identity(enriched_path: Path, embedding_model: str, dimensions: int) -> dict:
    return {
        "schema_version": 1,
        "input_sha256": sha256_file(enriched_path),
        "embedding_model": embedding_model,
        "dimensions": dimensions,
        "topic_method": "sklearn-lda-count-v1",
        "sentiment_method": "vader-lexicon-rule-v1",
        "provider": "gemini",
    }


def collection_name(identity: dict) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return COLLECTION_PREFIX + hashlib.sha256(payload).hexdigest()[:24]


def connect(path: Path):
    return chromadb.PersistentClient(
        path=str(path.resolve()),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def metadata_for(row: dict) -> dict:
    required = (
        "document_id",
        "title",
        "url",
        "source",
        "chunk_index",
        "start_char",
        "end_char",
        "document_sha256",
        "topic_id",
        "topic_label",
        "topic_confidence",
        "topic_status",
        "sentiment",
        "sentiment_compound",
        "sentiment_positive",
        "sentiment_neutral",
        "sentiment_negative",
        "sentiment_status",
    )
    metadata = {field: row.get(field, "") for field in required}
    metadata["topic_distribution_json"] = json.dumps(row.get("topic_distribution", []))
    for field in ("chunk_index", "start_char", "end_char", "topic_id"):
        metadata[field] = int(metadata[field])
    for field in (
        "topic_confidence",
        "sentiment_compound",
        "sentiment_positive",
        "sentiment_neutral",
        "sentiment_negative",
    ):
        metadata[field] = float(metadata[field])
    return metadata


def metadata_matches(actual: dict, expected: dict) -> bool:
    if set(actual) != set(expected):
        return False
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, float):
            if not isinstance(actual_value, int | float) or not math.isclose(
                actual_value, expected_value, rel_tol=1e-12, abs_tol=1e-12
            ):
                return False
        elif actual_value != expected_value:
            return False
    return True


def read_rows(path: Path) -> list[dict]:
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError("Enriched chunk file is empty")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Enriched chunk IDs must be unique, nonempty strings")
    if any(not isinstance(row.get("text"), str) or not row["text"].strip() for row in rows):
        raise ValueError("Every enriched chunk must contain text")
    return rows


def _target_collection(client, identity: dict, expected: int):
    name = collection_name(identity)
    try:
        collection = client.get_collection(name, embedding_function=None)
    except NotFoundError:
        collection = client.create_collection(
            name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={**identity, "status": "building", "expected_chunks": expected},
        )
    metadata = collection.metadata or {}
    if any(metadata.get(key) != value for key, value in identity.items()):
        raise ValueError("Week 4 collection identity mismatch")
    if metadata.get("expected_chunks") != expected:
        raise ValueError("Week 4 collection expected chunk count mismatch")
    if collection.configuration.get("hnsw", {}).get("space") != "cosine":
        raise ValueError("Week 4 collection must use cosine distance")
    return collection


def _load_source_vectors(rows: list[dict], expected_identity: dict) -> tuple[list[list[float]], dict] | None:
    if not (WEEK2_DB.exists() and WEEK2_MANIFEST.exists() and WEEK2_CHUNKS.exists()):
        return None
    manifest = json.loads(WEEK2_MANIFEST.read_text(encoding="utf-8"))
    source_identity = manifest.get("identity", {})
    if sha256_file(WEEK2_CHUNKS) != source_identity.get("input_sha256"):
        return None
    if sha256_file(WEEK2_CHUNKS) != expected_identity["source_chunks_sha256"]:
        return None
    if (
        source_identity.get("provider") != "gemini"
        or source_identity.get("embedding_model") != expected_identity["embedding_model"]
        or source_identity.get("dimensions") != expected_identity["dimensions"]
    ):
        return None
    client = connect(WEEK2_DB)
    collection = client.get_collection(manifest["collection"], embedding_function=None)
    if collection.count() != len(rows) or manifest.get("chunk_count") != len(rows):
        return None
    stored: dict[str, tuple[str, list[float]]] = {}
    for offset in range(0, collection.count(), BATCH_SIZE):
        page = collection.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=["documents", "embeddings"],
        )
        for ident, document, vector in zip(
            page["ids"], page["documents"], page["embeddings"], strict=True
        ):
            stored[ident] = (document, vector.tolist() if hasattr(vector, "tolist") else list(vector))
    if len(stored) != len(rows):
        return None
    vectors = []
    for row in rows:
        item = stored.get(row["id"])
        if item is None or item[0] != row["text"]:
            return None
        vector = [float(value) for value in item[1]]
        if len(vector) != expected_identity["dimensions"] or not all(map(math.isfinite, vector)):
            return None
        if not math.isclose(math.hypot(*vector), 1.0, abs_tol=1e-5):
            return None
        vectors.append(vector)
    provenance = {
        "source_collection": manifest["collection"],
        "source_manifest": WEEK2_MANIFEST.name,
        "source_chunk_sha256": source_identity["input_sha256"],
        "reused_embeddings": len(vectors),
    }
    return vectors, provenance


def build_index(
    enriched_path: Path,
    *,
    db_path: Path = INDEX_DIR,
    manifest_path: Path = INDEX_MANIFEST,
    report_path: Path = INDEX_REPORT,
    embedding_model: str,
    dimensions: int,
    embedder=None,
    request_interval: float = 12,
) -> dict:
    if not 0 <= request_interval <= 60:
        raise ValueError("Request interval must be between 0 and 60 seconds")
    rows = read_rows(enriched_path)
    source_hash = sha256_file(WEEK2_CHUNKS) if WEEK2_CHUNKS.exists() else ""
    identity = index_identity(enriched_path, embedding_model, dimensions)
    identity["source_chunks_sha256"] = source_hash
    name = collection_name(identity)
    client = connect(db_path)
    collection = _target_collection(client, identity, len(rows))
    started = __import__("time").perf_counter()
    imported = _load_source_vectors(rows, identity)
    if imported:
        vectors, provenance = imported
        for offset in range(0, len(rows), EMBED_BATCH_SIZE):
            batch_rows = rows[offset : offset + EMBED_BATCH_SIZE]
            collection.upsert(
                ids=[row["id"] for row in batch_rows],
                documents=[row["text"] for row in batch_rows],
                metadatas=[metadata_for(row) for row in batch_rows],
                embeddings=vectors[offset : offset + EMBED_BATCH_SIZE],
            )
    else:
        if embedder is None:
            raise ValueError(
                "Matching Week 2 vectors were not available; set a Gemini key to generate "
                "Week 4 embeddings from the enriched chunks"
            )
        import time

        previous_request = None
        provenance = {"source_collection": None, "source_manifest": None, "reused_embeddings": 0}
        for offset in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[offset : offset + BATCH_SIZE]
            existing = set(collection.get(ids=[row["id"] for row in batch_rows], include=[])["ids"])
            pending = [row for row in batch_rows if row["id"] not in existing]
            if not pending:
                continue
            if previous_request is not None:
                delay = request_interval - (time.perf_counter() - previous_request)
                if delay > 0:
                    time.sleep(delay)
            previous_request = time.perf_counter()
            vectors = embedder.embed([row["text"] for row in pending])
            collection.upsert(
                ids=[row["id"] for row in pending],
                documents=[row["text"] for row in pending],
                metadatas=[metadata_for(row) for row in pending],
                embeddings=vectors,
            )
    if collection.count() != len(rows):
        raise ValueError("Stored vector count does not match the enriched chunk count")
    collection.modify(metadata={**identity, "status": "ready", "expected_chunks": len(rows)})
    manifest = {
        "collection": name,
        "identity": identity,
        "chunk_count": len(rows),
        "distance_metric": "cosine",
    }
    write_json(manifest_path, manifest)
    report = {
        "status": "complete",
        "collection": name,
        "chunk_count": len(rows),
        "embedding_model": embedding_model,
        "embedding_dimensions": dimensions,
        "distance_metric": "cosine",
        "api_calls_for_indexing": getattr(embedder, "request_count", 0),
        "vector_provenance": provenance,
        "target_db": str(db_path),
        "elapsed_seconds": __import__("time").perf_counter() - started,
        "metadata_fields": sorted(metadata_for(rows[0]).keys()),
        "manifest": manifest_path.name,
    }
    write_json(report_path, report)
    return report


def open_collection(db_path: Path = INDEX_DIR, manifest_path: Path = INDEX_MANIFEST):
    if not manifest_path.exists():
        raise ValueError("No completed Week 4 index manifest; run scripts.index_corpus first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest["identity"]
    if manifest["collection"] != collection_name(identity):
        raise ValueError("Week 4 manifest collection does not match its identity")
    client = connect(db_path)
    collection = client.get_collection(manifest["collection"], embedding_function=None)
    if any((collection.metadata or {}).get(key) != value for key, value in identity.items()):
        raise ValueError("Week 4 collection metadata differs from its manifest")
    if collection.count() != manifest["chunk_count"]:
        raise ValueError("Week 4 collection is incomplete; rerun scripts.index_corpus")
    if (collection.metadata or {}).get("status") != "ready":
        raise ValueError("Week 4 collection is not marked ready")
    return collection, manifest


def build_where(*, topic_id: int | None = None, topic_label: str | None = None, sentiment: str | None = None):
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


def query_collection(collection, vector: list[float], *, k: int, where: dict | None = None) -> list[dict]:
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


def verify_index(
    enriched_path: Path,
    *,
    db_path: Path = INDEX_DIR,
    manifest_path: Path = INDEX_MANIFEST,
    output_path: Path = INDEX_VERIFICATION,
) -> dict:
    rows = read_rows(enriched_path)
    expected = {row["id"]: row for row in rows}
    collection, manifest = open_collection(db_path, manifest_path)
    if sha256_file(enriched_path) != manifest["identity"]["input_sha256"]:
        raise ValueError("Enriched input hash differs from the active Week 4 snapshot")
    checked = 0
    failures = []
    for offset in range(0, collection.count(), BATCH_SIZE):
        page = collection.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        for ident, text, metadata, vector in zip(
            page["ids"], page["documents"], page["metadatas"], page["embeddings"], strict=True
        ):
            source = expected.get(ident)
            if source is None or text != source["text"]:
                failures.append(f"Unknown ID or text mismatch: {ident}")
                continue
            if not metadata_matches(metadata, metadata_for(source)):
                failures.append(f"Metadata mismatch: {ident}")
            values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
            if (
                len(values) != manifest["identity"]["dimensions"]
                or not all(math.isfinite(value) for value in values)
                or not math.isclose(math.hypot(*values), 1.0, abs_tol=1e-5)
            ):
                failures.append(f"Invalid vector: {ident}")
            checked += 1
    filter_checks = []
    for sentiment in ("positive", "neutral", "negative", "unknown"):
        selected = collection.get(
            where={"sentiment": {"$eq": sentiment}},
            include=["metadatas"],
        )
        values = selected["metadatas"]
        exact = all(item["sentiment"] == sentiment for item in values)
        filter_checks.append({"filter": {"sentiment": sentiment}, "count": len(values), "exact": exact})
        if not exact:
            failures.append(f"Sentiment filter returned a mismatched row: {sentiment}")
    topic_candidates = [row for row in rows if row["topic_id"] >= 0]
    if topic_candidates:
        topic_id = topic_candidates[0]["topic_id"]
        selected = collection.get(
            where={"topic_id": {"$eq": topic_id}},
            include=["metadatas"],
        )
        exact = all(item["topic_id"] == topic_id for item in selected["metadatas"])
        filter_checks.append(
            {"filter": {"topic_id": topic_id}, "count": len(selected["ids"]), "exact": exact}
        )
        if not exact:
            failures.append("Topic ID filter returned a mismatched row")
    combined_filter = None
    candidates = [row for row in rows if row["topic_id"] >= 0 and row["sentiment"] != "unknown"]
    if candidates:
        candidate = candidates[0]
        combined_filter = {
            "$and": [
                {"topic_id": {"$eq": candidate["topic_id"]}},
                {"sentiment": {"$eq": candidate["sentiment"]}},
            ]
        }
        selected = collection.get(where=combined_filter, include=["metadatas"])
        exact = all(
            item["topic_id"] == candidate["topic_id"]
            and item["sentiment"] == candidate["sentiment"]
            for item in selected["metadatas"]
        )
        filter_checks.append(
            {"filter": combined_filter, "count": len(selected["ids"]), "exact": exact}
        )
        if not exact:
            failures.append("Combined topic and sentiment filter returned a mismatched row")
    if checked != len(expected) or collection.count() != len(expected):
        failures.append("Stored record count differs from the enriched source")
    report = {
        "status": "passed" if not failures else "failed",
        "collection": collection.name,
        "expected_records": len(expected),
        "checked_records": checked,
        "stored_text_metadata_and_vector_checked": True,
        "vector_dimensions_finiteness_and_norm_checked": True,
        "metadata_filter_checks": filter_checks,
        "failures": failures,
    }
    write_json(output_path, report)
    if failures:
        raise ValueError(f"Index verification failed; see {output_path.name}")
    return report
