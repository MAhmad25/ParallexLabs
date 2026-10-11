"""Stream a pinned Wikimedia Wikipedia snapshot into local raw JSONL."""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from src.io_utils import file_sha256, write_json  # noqa: E402

DATASET = "wikimedia/wikipedia"
CONFIG = "20231101.en"
REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/wikipedia_en.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/dataset_provenance.json")
    parser.add_argument("--force", action="store_true", help="Replace an existing raw JSONL file")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.output.exists() and not args.force:
        parser.error("Output already exists; use --force to download again")
    if args.output.resolve() == args.report.resolve():
        parser.error("--report must be separate from --output")

    from datasets import load_dataset

    print(f"Streaming {args.limit:,} articles from {DATASET}/{CONFIG}", flush=True)
    dataset = load_dataset(
        DATASET,
        CONFIG,
        split="train",
        streaming=True,
        revision=args.revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    seen_ids: set[str] = set()
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for count, article in enumerate(dataset.take(args.limit), start=1):
                record = {key: article[key] for key in ("id", "url", "title", "text")}
                if any(not isinstance(value, str) for value in record.values()):
                    raise ValueError(f"Source contains a non-string field at article {count}")
                if record["id"] in seen_ids:
                    raise ValueError(f"Duplicate source article ID: {record['id']}")
                seen_ids.add(record["id"])
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if count % 500 == 0:
                    print(f"Downloaded {count:,}/{args.limit:,} articles", flush=True)
        if count != args.limit:
            raise ValueError(f"Expected {args.limit} articles; source provided {count}")
        temporary.replace(args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    report = {
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "dataset": DATASET,
        "configuration": CONFIG,
        "revision": args.revision,
        "split": "train",
        "sampling": "first N articles in snapshot order; no shuffle",
        "documents": count,
        "unique_article_ids": len(seen_ids),
        "raw_file": args.output.name,
        "raw_bytes": args.output.stat().st_size,
        "raw_sha256": file_sha256(args.output),
        "dataset_url": "https://huggingface.co/datasets/wikimedia/wikipedia",
        "snapshot_url": f"https://huggingface.co/datasets/{DATASET}/tree/{args.revision}",
        "source_type": "Wikipedia dump-derived, upstream-extracted article text",
        "licenses_listed_by_dataset_card": ["CC-BY-SA-3.0", "GFDL"],
        "license_information": "https://dumps.wikimedia.org/legal.html",
        "attribution": "Article IDs, titles, and source URLs retained in raw and cleaned data",
    }
    write_json(args.report, report)
    print(f"Saved {count:,} real articles to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
