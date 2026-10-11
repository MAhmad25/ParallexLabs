"""Check every output document and fail if the corpus has fewer than 5,000 rows."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import write_json  # noqa: E402
from src.preprocessing import CleaningConfig  # noqa: E402
from src.validation import validate_corpus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/clean_corpus.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/validation_report.json")
    parser.add_argument("--min-documents", type=int, default=5000)
    parser.add_argument("--language", default="en")
    parser.add_argument("--min-chars", type=int, default=100)
    parser.add_argument("--min-confidence", type=float, default=0.90)
    args = parser.parse_args()
    if args.input.resolve() == args.report.resolve():
        parser.error("--report must be separate from --input")
    config = CleaningConfig(args.language, args.min_chars, args.min_confidence)
    report = validate_corpus(args.input, config, args.min_documents)
    report["validated_at_utc"] = datetime.now(UTC).isoformat()
    write_json(args.report, report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
