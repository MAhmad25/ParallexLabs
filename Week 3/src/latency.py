"""Append privacy-conscious per-request RAG latency records."""

import hashlib
import json
from pathlib import Path

from src.io_utils import now


def append_latency(path: Path, question: str, result: dict) -> None:
    """Persist timing and outcome metadata without storing query or answer text."""
    answer = result.get("answer", {})
    record = {
        "created_at_utc": now(),
        "query_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "status": result.get("status", "error"),
        "refusal_reason": result.get("refusal_reason"),
        "embedding_ms": result.get("embedding_ms"),
        "vector_search_ms": result.get("vector_search_ms"),
        "retrieval_ms": result.get("retrieval_ms"),
        "generation_ms": answer.get("generation_ms", result.get("generation_ms", 0.0)),
        "end_to_end_ms": result.get("end_to_end_ms"),
        "best_similarity": result.get("best_similarity"),
        "source_count": len(result.get("sources", [])),
        "model": answer.get("model"),
        "error_type": result.get("error_type"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
