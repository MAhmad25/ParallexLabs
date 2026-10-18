"""Independently verify the persisted database against every source chunk; no API calls."""

import argparse
import json
import math
from pathlib import Path

from src.cli import CHUNKS, DATABASE, MANIFEST, run
from src.gemini import ROOT
from src.io_utils import distinct_paths, file_sha256, now, read_jsonl, write_json
from src.vector_store import VectorStore, collection_name


def verify(chunks_path: Path, db_path: Path, manifest_path: Path, output: Path) -> dict:
    distinct_paths(chunks_path, manifest_path, output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest["identity"]
    if file_sha256(chunks_path) != identity["input_sha256"]:
        raise ValueError("Chunk file hash differs from the completed index snapshot")
    if manifest["collection"] != collection_name(identity):
        raise ValueError("Manifest collection name differs from its identity")
    expected = {row["id"]: row for row in read_jsonl(chunks_path)}
    store = VectorStore(db_path, identity)
    store.require_ready()
    checked = 0
    failures = []
    seen = set()
    for offset in range(0, store.count(), 100):
        data = store.collection.get(
            limit=100, offset=offset, include=["documents", "metadatas", "embeddings"]
        )
        for ident, text, metadata, vector in zip(
            data["ids"],
            data["documents"],
            data["metadatas"],
            data["embeddings"],
            strict=True,
        ):
            source = expected.get(ident)
            if source is None or ident in seen:
                failures.append("Unknown or duplicate stored ID")
                continue
            seen.add(ident)
            if text != source["text"]:
                failures.append("Stored text differs from source chunk")
            if metadata != {k: v for k, v in source.items() if k not in {"id", "text"}}:
                failures.append("Stored metadata differs from source chunk")
            if (
                len(vector) != identity["dimensions"]
                or not all(math.isfinite(v) for v in vector)
                or not math.isclose(math.hypot(*vector), 1, abs_tol=1e-6)
            ):
                failures.append("Invalid stored vector dimension, values, or normalization")
            checked += 1
    if len(expected) != checked or manifest["chunk_count"] != checked:
        failures.append("Stored count differs from source or manifest")
    report = {
        "created_at_utc": now(),
        "status": "passed" if not failures else "failed",
        "collection": store.name,
        "identity": identity,
        "expected_chunks": len(expected),
        "checked_chunks": checked,
        "source_text_and_metadata_checked": True,
        "vector_dimensions_finiteness_and_norm_checked": True,
        "api_calls": 0,
        "failures": failures,
    }
    write_json(output, report)
    if failures:
        raise ValueError(f"Index verification failed with {len(failures)} errors; see report")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--db", type=Path, default=DATABASE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/index_verification.json")
    args = parser.parse_args()
    report = verify(args.chunks, args.db, args.manifest, args.output)
    print(f"Verified {report['checked_chunks']} persisted chunks; no API calls")


if __name__ == "__main__":
    run(main)
