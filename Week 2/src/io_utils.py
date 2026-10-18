"""UTF-8 input validation, fingerprints, and atomic reports."""

import hashlib
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path


def now() -> str:
    return datetime.now(UTC).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON at {path.name}:{number}") from None
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path.name}:{number}")
            yield row


def distinct_paths(*paths: Path) -> None:
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Input, output, and report must use separate paths")
