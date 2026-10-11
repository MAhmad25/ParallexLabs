"""Clean and deduplicate the local raw Wikipedia corpus."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import process_corpus  # noqa: E402
from src.preprocessing import CleaningConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/wikipedia_en.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/clean_corpus.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/preprocessing_report.json")
    parser.add_argument("--language", default="en")
    parser.add_argument("--min-chars", type=int, default=100)
    parser.add_argument("--min-confidence", type=float, default=0.90)
    args = parser.parse_args()
    config = CleaningConfig(args.language, args.min_chars, args.min_confidence)
    result = process_corpus(args.input, args.output, args.report, config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
