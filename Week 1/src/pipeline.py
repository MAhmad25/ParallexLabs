"""File pipeline: clean, filter, deduplicate, and record an auditable summary."""

import json
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from src.io_utils import file_sha256, read_jsonl, write_json
from src.preprocessing import CleaningConfig, clean_document


def process_corpus(
    input_path: Path, output_path: Path, report_path: Path, config: CleaningConfig
) -> dict:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files")
    if output_path.suffix not in {".jsonl", ".parquet"}:
        raise ValueError("Output must have a .jsonl or .parquet extension")
    if report_path.resolve() in {input_path.resolve(), output_path.resolve()}:
        raise ValueError("Report must be separate from data files")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    rejections: Counter = Counter()
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    raw_count = 0
    accepted_count = 0
    parquet_records = []
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for raw_count, record in enumerate(read_jsonl(input_path), start=1):
                cleaned, reason = clean_document(record, config)
                if cleaned is None:
                    rejections[reason] += 1
                    continue
                if cleaned["id"] in seen_ids:
                    rejections["duplicate_id"] += 1
                    continue
                if cleaned["content_sha256"] in seen_hashes:
                    rejections["duplicate_content"] += 1
                    continue
                seen_ids.add(cleaned["id"])
                seen_hashes.add(cleaned["content_sha256"])
                accepted_count += 1
                if output_path.suffix == ".parquet":
                    parquet_records.append(cleaned)
                else:
                    handle.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                if raw_count % 500 == 0:
                    print(f"Processed {raw_count:,}; accepted {accepted_count:,}", flush=True)
        if accepted_count == 0:
            raise ValueError("No documents passed the cleaning filters")
        if output_path.suffix == ".parquet":
            import pandas as pd

            pd.DataFrame(parquet_records).to_parquet(temporary, index=False, engine="pyarrow")
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "input_file": input_path.name,
        "output_file": output_path.name,
        "input_sha256": file_sha256(input_path),
        "output_sha256": file_sha256(output_path),
        "config": asdict(config),
        "raw_documents": raw_count,
        "clean_documents": accepted_count,
        "rejected_documents": sum(rejections.values()),
        "rejections_by_reason": dict(sorted(rejections.items())),
        "deduplication": "exact article ID and SHA-256 of normalized text",
    }
    write_json(report_path, report)
    return report
