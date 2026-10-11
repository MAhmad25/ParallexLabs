"""Independent corpus checks, including a second language-detection pass."""

import hashlib
import unicodedata
from collections import Counter
from pathlib import Path

from src.io_utils import file_sha256, read_jsonl
from src.preprocessing import HTML_TAG, CleaningConfig, detect_language, normalize_whitespace


def validate_corpus(path: Path, config: CleaningConfig, min_documents: int = 5000) -> dict:
    if min_documents < 1:
        raise ValueError("min_documents must be positive")
    if path.suffix == ".parquet":
        import pandas as pd

        records = pd.read_parquet(path).to_dict("records")
    elif path.suffix == ".jsonl":
        records = read_jsonl(path)
    else:
        raise ValueError("Corpus must have a .jsonl or .parquet extension")
    errors: Counter = Counter()
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    lengths = []
    count = 0
    for count, record in enumerate(records, start=1):
        required_strings = ("id", "title", "url", "text", "source", "language", "content_sha256")
        if any(not isinstance(record.get(key), str) or not record[key] for key in required_strings):
            errors["missing_or_invalid_fields"] += 1
            continue
        text = record["text"]
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if record["id"] in seen_ids:
            errors["duplicate_id"] += 1
        if digest in seen_hashes:
            errors["duplicate_content"] += 1
        seen_ids.add(record["id"])
        seen_hashes.add(digest)
        lengths.append(len(text))
        if digest != record["content_sha256"]:
            errors["content_hash_mismatch"] += 1
        if record.get("char_count") != len(text) or record.get("word_count") != len(text.split()):
            errors["length_metadata_mismatch"] += 1
        if len(text) < config.min_chars:
            errors["too_short"] += 1
        if text != normalize_whitespace(text):
            errors["unclean_whitespace"] += 1
        if not unicodedata.is_normalized("NFKC", text):
            errors["unnormalized_unicode"] += 1
        if any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in text):
            errors["invisible_or_control_characters"] += 1
        if HTML_TAG.search(text):
            errors["remaining_html"] += 1
        language = detect_language(text)
        if language.code != config.target_language or record["language"] != config.target_language:
            errors["wrong_language"] += 1
        if language.confidence < config.min_language_confidence:
            errors["low_language_confidence"] += 1
        recorded_confidence = record.get("language_confidence")
        if not isinstance(recorded_confidence, (int, float)) or not (
            config.min_language_confidence <= recorded_confidence <= 1
        ):
            errors["invalid_language_confidence"] += 1
        if count % 1000 == 0:
            print(f"Validated {count:,} documents", flush=True)
    if count < min_documents:
        errors["below_minimum_document_count"] += 1
    return {
        "passed": not errors,
        "documents": count,
        "minimum_documents": min_documents,
        "unique_ids": len(seen_ids),
        "unique_text_hashes": len(seen_hashes),
        "min_chars": min(lengths) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
        "mean_chars": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        "sha256": file_sha256(path),
        "errors": dict(sorted(errors.items())),
    }
