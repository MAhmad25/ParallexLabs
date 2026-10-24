"""Recursive boundary selection with bounded character windows and exact offsets."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from src.io_utils import distinct_paths, file_sha256, now, read_jsonl, write_json


@dataclass(frozen=True)
class ChunkConfig:
    size: int = 1500
    overlap: int = 200

    def __post_init__(self):
        if self.size < 2 or not 0 <= self.overlap < self.size:
            raise ValueError("Require size >= 2 and 0 <= overlap < size")


DEFAULT_CONFIG = ChunkConfig()


def _boundary(text: str, lower: int, upper: int, separators: tuple[str, ...]) -> int:
    """Recursively try paragraph, line, sentence, word, then character boundaries."""
    if not separators:
        return upper
    separator, *remaining = separators
    found = text.rfind(separator, lower, upper)
    if found >= lower:
        return found + len(separator)
    return _boundary(text, lower, upper, tuple(remaining))


def split_text(text: str, config: ChunkConfig = DEFAULT_CONFIG) -> list[tuple[int, int, str]]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    chunks = []
    start = 0
    while start < len(text):
        upper = min(start + config.size, len(text))
        if upper == len(text):
            end = upper
        else:
            lower = start + max(config.size // 2, config.overlap + 1)
            end = _boundary(text, lower, upper, ("\n\n", "\n", ". ", "! ", "? ", " "))
        left, right = start, end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            chunks.append((left, right, text[left:right]))
        if end == len(text):
            break
        start = max(start + 1, end - config.overlap)
    return chunks


def chunk_document(row: dict, config: ChunkConfig = DEFAULT_CONFIG) -> list[dict]:
    doc_id, text = row.get("id"), row.get("text")
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("Every document needs a nonempty string id")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Document {doc_id} has empty or invalid text")
    for field in ("title", "url", "source"):
        if not isinstance(row.get(field, ""), str):
            raise ValueError(f"Document {doc_id}: {field} must be a string")
    digest = hashlib.sha256(text.encode()).hexdigest()
    if row.get("content_sha256", digest) != digest:
        raise ValueError(f"Document {doc_id} content hash does not match")
    result = []
    for index, (start, end, content) in enumerate(split_text(text, config)):
        identity = json.dumps([doc_id, digest, index, start, end], separators=(",", ":"))
        result.append(
            {
                "id": hashlib.sha256(identity.encode()).hexdigest(),
                "document_id": doc_id,
                "title": row.get("title", ""),
                "url": row.get("url", ""),
                "source": row.get("source", ""),
                "chunk_index": index,
                "start_char": start,
                "end_char": end,
                "document_sha256": digest,
                "text": content,
            }
        )
    return result


def chunk_corpus(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    config: ChunkConfig = DEFAULT_CONFIG,
    limit: int = 10,
) -> dict:
    distinct_paths(input_path, output_path, report_path)
    if limit < 0:
        raise ValueError("limit must be >= 0; 0 means all documents")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    seen = set()
    lengths = []
    documents = []
    started = perf_counter()
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in read_jsonl(input_path):
                if limit and len(seen) >= limit:
                    break
                chunks = chunk_document(row, config)
                if row["id"] in seen:
                    raise ValueError(f"Duplicate document id: {row['id']}")
                seen.add(row["id"])
                documents.append({"id": row["id"], "title": row.get("title", "")})
                for chunk in chunks:
                    lengths.append(len(chunk["text"]))
                    handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        if not lengths:
            raise ValueError("Corpus contains no usable documents")
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    elapsed = perf_counter() - started
    report = {
        "created_at_utc": now(),
        "status": "complete",
        "strategy": "recursive-boundary-v1",
        "config": asdict(config),
        "document_limit": limit,
        "documents": documents,
        "document_count": len(seen),
        "chunk_count": len(lengths),
        "min_chunk_chars": min(lengths),
        "max_chunk_chars": max(lengths),
        "mean_chunk_chars": sum(lengths) / len(lengths),
        "chunking_seconds": elapsed,
        "chunks_per_second": len(lengths) / elapsed,
        "input_file": input_path.name,
        "input_sha256": file_sha256(input_path),
        "output_file": output_path.name,
        "output_sha256": file_sha256(output_path),
    }
    write_json(report_path, report)
    return report
