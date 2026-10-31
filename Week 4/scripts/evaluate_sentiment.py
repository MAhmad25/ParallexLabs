"""Score VADER predictions against the reviewed, manually labeled sample."""

import argparse
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.io_utils import read_jsonl, write_json
from src.paths import ENRICHED_CHUNKS, GOLD_LABELS, SENTIMENT_REPORT

LABELS = ["negative", "neutral", "positive"]


def evaluate(enriched_path: Path, gold_path: Path) -> dict:
    predictions = {row["id"]: row for row in read_jsonl(enriched_path)}
    gold = list(read_jsonl(gold_path))
    if not gold:
        raise ValueError("The manual sentiment sample is empty")
    gold_ids = [row.get("chunk_id") for row in gold]
    if any(not isinstance(value, str) for value in gold_ids) or len(set(gold_ids)) != len(gold_ids):
        raise ValueError("Manual sentiment chunk IDs must be unique strings")
    unknown = [row for row in gold if row.get("manual_label") not in LABELS]
    if unknown:
        raise ValueError("Manual labels must be negative, neutral, or positive")
    missing = [value for value in gold_ids if value not in predictions]
    if missing:
        raise ValueError(f"Manual sentiment chunk is absent from enriched corpus: {missing[0]}")
    y_true = [row["manual_label"] for row in gold]
    y_pred = [predictions[row["chunk_id"]]["sentiment"] for row in gold]
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    per_class = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    return {
        "status": "complete",
        "evaluation_method": "held-out human-reviewed chunk labels; gold labels are not used by VADER",
        "sample_count": len(gold),
        "gold_label_counts": dict(Counter(y_true)),
        "predicted_label_counts": dict(Counter(y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "labels_order": LABELS,
        "confusion_matrix_rows_true_columns_predicted": matrix.tolist(),
        "per_class": {label: per_class[label] for label in LABELS},
        "item_results": [
            {
                "chunk_id": row["chunk_id"],
                "title": predictions[row["chunk_id"]].get("title", ""),
                "manual_label": row["manual_label"],
                "predicted_label": predictions[row["chunk_id"]]["sentiment"],
                "compound": predictions[row["chunk_id"]]["sentiment_compound"],
                "correct": row["manual_label"] == predictions[row["chunk_id"]]["sentiment"],
                "review_note": row.get("review_note", ""),
            }
            for row in gold
        ],
        "limitations": [
            "This small sample estimates performance only on reviewed chunks from the selected Wikipedia subset.",
            "VADER is tuned for lexical sentiment and can misread factual prose, negation, jargon, and historical violence.",
            "Accuracy is not a measure of topic quality or generalization to other domains.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ENRICHED_CHUNKS)
    parser.add_argument("--gold", type=Path, default=GOLD_LABELS)
    parser.add_argument("--report", type=Path, default=SENTIMENT_REPORT)
    args = parser.parse_args()
    report = evaluate(args.input, args.gold)
    write_json(args.report, report)
    print(
        f"Sentiment evaluation: {report['sample_count']} manually labeled chunks, "
        f"accuracy {report['accuracy']:.3f}, macro-F1 {report['macro_f1']:.3f}"
    )


if __name__ == "__main__":
    main()
