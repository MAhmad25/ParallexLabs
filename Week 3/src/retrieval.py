"""Resumable ingestion and Gemini semantic search."""

import json
import re
from itertools import batched
from pathlib import Path
from time import perf_counter, sleep

from src.gemini import Gemini
from src.io_utils import distinct_paths, file_sha256, now, read_jsonl, write_json
from src.vector_store import VectorStore, collection_name, snapshot_identity, validate_chunk


def ingest(
    chunks_path: Path,
    db_path: Path,
    manifest_path: Path,
    report_path: Path,
    gemini: Gemini,
    batch_size: int = 16,
    request_interval: float = 0,
) -> dict:
    distinct_paths(chunks_path, manifest_path, report_path)
    if not 1 <= batch_size <= 100:
        raise ValueError("batch_size must be between 1 and 100")
    if not 0 <= request_interval <= 60:
        raise ValueError("request_interval must be between 0 and 60 seconds")
    seen = set()
    for row in read_jsonl(chunks_path):
        validate_chunk(row)
        if row["id"] in seen:
            raise ValueError("Duplicate chunk IDs in input")
        seen.add(row["id"])
    if not seen:
        raise ValueError("Cannot ingest an empty chunk file")
    identity = snapshot_identity(
        file_sha256(chunks_path), gemini.settings.embedding_model, gemini.settings.dimensions
    )
    store = VectorStore(db_path, identity, expected=len(seen))
    report = {
        "created_at_utc": now(),
        "status": "running",
        "identity": identity,
        "collection": store.name,
        "expected_chunks": len(seen),
        "batch_size": batch_size,
        "request_interval_seconds": request_interval,
        "already_present": store.count(),
        "embedded_this_run": 0,
        "batches": [],
        "embedding_seconds": 0.0,
        "upsert_seconds": 0.0,
    }
    write_json(report_path, report)
    started = perf_counter()
    previous_request = None
    try:
        for number, rows in enumerate(batched(read_jsonl(chunks_path), batch_size), 1):
            pending = store.missing(list(rows))
            if not pending:
                continue
            if previous_request is not None:
                delay = request_interval - (perf_counter() - previous_request)
                if delay > 0:
                    sleep(delay)
            previous_request = perf_counter()
            vectors, elapsed = gemini.embed([row["text"] for row in pending])
            write_start = perf_counter()
            store.upsert(pending, vectors)
            upsert_seconds = perf_counter() - write_start
            report["embedded_this_run"] += len(pending)
            report["embedding_seconds"] += elapsed
            report["upsert_seconds"] += upsert_seconds
            report["batches"].append(
                {
                    "batch": number,
                    "chunks": len(pending),
                    "embedding_seconds": elapsed,
                    "upsert_seconds": upsert_seconds,
                }
            )
            report["stored_chunks"] = store.count()
            write_json(report_path, report)
            print(
                f"Batch {number}: {store.count()}/{len(seen)} chunks; Gemini {elapsed:.3f}s",
                flush=True,
            )
        if file_sha256(chunks_path) != identity["input_sha256"]:
            raise ValueError("Chunks changed during ingestion; rerun against a stable input")
        store.mark_ready()
        write_json(
            manifest_path,
            {
                "created_at_utc": now(),
                "identity": identity,
                "collection": store.name,
                "chunk_count": store.count(),
            },
        )
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["failure_type"] = type(error).__name__
        # Do not persist raw SDK exceptions, which may contain request details.
        raise
    finally:
        report["total_seconds"] = perf_counter() - started
        report["stored_chunks"] = store.count()
        report["finished_at_utc"] = now()
        count = report["embedded_this_run"]
        seconds = report["embedding_seconds"]
        report["embedding_chunks_per_second"] = count / seconds if seconds else None
        write_json(report_path, report)
    return report


def open_store(db_path: Path, manifest_path: Path, gemini: Gemini) -> VectorStore:
    if not manifest_path.exists():
        raise ValueError("No completed index manifest; run scripts.ingest first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest["identity"]
    if (
        identity["embedding_model"] != gemini.settings.embedding_model
        or identity["dimensions"] != gemini.settings.dimensions
    ):
        raise ValueError(
            "Gemini settings differ from the active index; restore them or ingest again"
        )
    if manifest["collection"] != collection_name(identity):
        raise ValueError("Index manifest collection does not match its identity")
    store = VectorStore(db_path, identity)
    store.require_ready()
    return store


def search(store: VectorStore, gemini: Gemini, question: str, k: int = 5) -> dict:
    if k < 1:
        raise ValueError("k must be >= 1")
    if not question.strip():
        raise ValueError("Query must not be empty")
    started = perf_counter()
    vectors, embedding_seconds = gemini.embed([question], query=True)
    hits, vector_ms = store.query(vectors[0], k)
    return {
        "query": question,
        "k": k,
        "hits": hits,
        "embedding_ms": embedding_seconds * 1000,
        "vector_search_ms": vector_ms,
        "total_ms": (perf_counter() - started) * 1000,
    }


def citations_are_valid(answer: str, source_count: int) -> tuple[bool, list[int]]:
    """Require real source labels and at least one citation on every answer sentence."""
    labels = [int(value) for value in re.findall(r"\[(\d+)\]", answer)]
    allowed = set(range(1, source_count + 1))
    used = sorted(set(labels))
    if not used or not set(used).issubset(allowed):
        return False, used

    normalized = re.sub(r"(?m)^(\s*)\d+[.)]\s+", r"\1", answer)
    normalized = re.sub(
        r"([.!?])\s*((?:\[\d+\]\s*)+)",
        lambda match: f" {match.group(2).strip()}{match.group(1)}",
        normalized,
    )
    pieces = re.split(r"(?<=[.!?])\s+|\n+", normalized)
    for piece in pieces:
        sentence = re.sub(r"^[\s#>*\-`]+", "", piece).strip()
        if sentence.startswith("#"):
            continue
        if sentence and re.search(r"\w", sentence) and not re.search(r"\[\d+\]", sentence):
            return False, used
    return True, used


def ask(
    store: VectorStore,
    gemini: Gemini,
    question: str,
    k: int = 5,
    min_similarity: float | None = None,
) -> dict:
    """Retrieve, refuse weak or uncited evidence, and measure full request latency."""
    if len(question) > 2000:
        raise ValueError("Question exceeds the 2,000-character limit")
    threshold = gemini.settings.min_similarity if min_similarity is None else min_similarity
    if not 0 <= threshold <= 1:
        raise ValueError("min_similarity must be between 0 and 1")

    started = perf_counter()
    try:
        retrieval = search(store, gemini, question, k)
    except Exception as error:
        result = {
            "query": question,
            "collection": store.name,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error)
            if isinstance(error, (RuntimeError, ValueError))
            else (
                f"Request failed ({type(error).__name__}); check the index, network, "
                "and local files."
            ),
            "embedding_ms": None,
            "vector_search_ms": None,
            "retrieval_ms": (perf_counter() - started) * 1000,
            "generation_ms": 0.0,
            "end_to_end_ms": (perf_counter() - started) * 1000,
            "best_similarity": None,
            "sources": [],
            "hits": [],
        }
        return result
    hits = retrieval["hits"]
    best_similarity = max((hit["cosine_similarity"] for hit in hits), default=None)
    result = {
        "query": question,
        "collection": store.name,
        "retrieval_ms": retrieval["total_ms"],
        "embedding_ms": retrieval["embedding_ms"],
        "vector_search_ms": retrieval["vector_search_ms"],
        "best_similarity": best_similarity,
        "threshold": threshold,
        "hits": hits,
    }

    if not hits or best_similarity is None or best_similarity < threshold:
        result["status"] = "refused"
        result["refusal_reason"] = "low_retrieval_relevance"
        result["answer"] = {
            "text": (
                "I could not find sufficiently relevant evidence in the indexed sources "
                "to answer that question."
            ),
            "citations": [],
            "generation_ms": 0.0,
        }
    else:
        try:
            generated = gemini.answer(question, hits)
        except Exception as error:
            generation_ms = getattr(error, "generation_ms", 0.0)
            result["status"] = "error"
            result["error_type"] = type(error).__name__
            result["error"] = (
                str(error)
                if isinstance(error, (RuntimeError, ValueError))
                else (
                    f"Generation failed ({type(error).__name__}); "
                    "check Gemini API access and retry."
                )
            )
            result["answer"] = {"text": "", "citations": [], "generation_ms": generation_ms}
            result["generation_ms"] = generation_ms
            result["end_to_end_ms"] = (perf_counter() - started) * 1000
            result["sources"] = [
                {
                    "label": index,
                    "title": hit["metadata"].get("title", ""),
                    "url": hit["metadata"].get("url", ""),
                    "chunk_id": hit["id"],
                    "cosine_similarity": hit["cosine_similarity"],
                }
                for index, hit in enumerate(hits, 1)
            ]
            return result
        citations_valid, citations = citations_are_valid(generated["text"], len(hits))
        if not citations_valid:
            result["status"] = "refused"
            result["refusal_reason"] = "citation_validation_failed"
            result["answer"] = {
                "text": (
                    "I could not verify source citations for a grounded answer, so I am abstaining."
                ),
                "citations": [],
                "model": generated["model"],
                "generation_ms": generated["generation_ms"],
            }
        else:
            result["status"] = "answered"
            result["refusal_reason"] = None
            result["answer"] = {
                "text": generated["text"],
                "citations": citations,
                "model": generated["model"],
                "generation_ms": generated["generation_ms"],
            }

    result["end_to_end_ms"] = (perf_counter() - started) * 1000
    result["generation_ms"] = result["answer"]["generation_ms"]
    result["sources"] = [
        {
            "label": index,
            "title": hit["metadata"].get("title", ""),
            "url": hit["metadata"].get("url", ""),
            "chunk_id": hit["id"],
            "cosine_similarity": hit["cosine_similarity"],
        }
        for index, hit in enumerate(hits, 1)
    ]
    return result
