"""Apply LDA topic modeling and VADER sentiment to Week 3's chunks."""

import argparse
import json
import os
from pathlib import Path

from dotenv import dotenv_values

from src.io_utils import read_jsonl, sha256_file, write_json, write_jsonl
from src.nlp import enrich_chunks
from src.paths import ENRICHED_CHUNKS, INPUT_CHUNKS, NLP_REPORT, ROOT, TOPIC_LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=INPUT_CHUNKS, type=Path)
    parser.add_argument("--output", default=ENRICHED_CHUNKS, type=Path)
    parser.add_argument("--report", default=NLP_REPORT, type=Path)
    topic_settings = dotenv_values(ROOT / ".env")
    topic_default = os.getenv("WEEK4_TOPIC_COUNT") or topic_settings.get("WEEK4_TOPIC_COUNT") or "8"
    parser.add_argument("--topics", default=int(topic_default), type=int)
    parser.add_argument("--labels", default=TOPIC_LABELS, type=Path)
    args = parser.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            "Week 3 chunks are missing. From Week 3, run: "
            "uv run --locked python -m scripts.chunk_corpus"
        )
    chunks = list(read_jsonl(args.input))
    enriched, report = enrich_chunks(chunks, requested_topics=args.topics, labels_path=args.labels)
    report.update(
        {
            "input_file": args.input.name,
            "input_sha256": sha256_file(args.input),
            "output_file": args.output.name,
            "output_sha256": None,
            "topic_label_file": str(args.labels.relative_to(ROOT)) if args.labels.is_relative_to(ROOT) else args.labels.name,
        }
    )
    write_jsonl(args.output, enriched)
    report["output_sha256"] = sha256_file(args.output)
    write_json(args.report, report)
    print(
        f"Enriched {report['chunk_count']} chunks with {report['topic_count']} topics; "
        f"sentiment: {json.dumps(report['sentiment_counts'], sort_keys=True)}"
    )


if __name__ == "__main__":
    main()
