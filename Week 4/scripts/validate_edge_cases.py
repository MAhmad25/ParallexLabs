"""Record how short text and a domain-specific phrase are handled."""

import argparse
from pathlib import Path

from src.io_utils import write_json
from src.nlp import enrich_chunks
from src.paths import ROOT, TOPIC_LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/edge_case_review.json")
    args = parser.parse_args()
    examples = [
        {"id": "edge-short", "title": "Short sample", "text": "A."},
        {
            "id": "edge-jargon",
            "title": "Albedo example",
            "text": (
                "Positive feedback in a climate model amplifies changes in solar radiation "
                "and ice cover. Albedo is a measure of reflected sunlight."
            ),
        },
    ]
    result, _ = enrich_chunks(examples, requested_topics=2, labels_path=TOPIC_LABELS)
    short, jargon = result
    short_handled = (
        short["topic_id"] == -1
        and short["topic_status"] == "no_vocabulary_terms"
        and short["sentiment"] == "unknown"
        and short["sentiment_status"] == "insufficient_text"
    )
    report = {
        "status": "passed_with_known_jargon_limit" if short_handled else "failed",
        "short_text": {
            "text": "A.",
            "topic_id": short["topic_id"],
            "topic_status": short["topic_status"],
            "sentiment": short["sentiment"],
            "sentiment_status": short["sentiment_status"],
            "handled_without_false_neutral": short_handled,
        },
        "technical_jargon": {
            "text": examples[1]["text"],
            "manual_expected_sentiment": "neutral",
            "predicted_sentiment": jargon["sentiment"],
            "compound": jargon["sentiment_compound"],
            "status": "known_false_positive" if jargon["sentiment"] != "neutral" else "matched",
            "note": (
                "The phrase 'positive feedback' is a climate-science term. VADER can treat "
                "the word 'positive' as praise; the manually labeled Albedo chunk shows this "
                "domain mismatch in the corpus evaluation."
            ),
        },
        "review_note": (
            "Short text is excluded from topic assignment and marked unknown for sentiment. "
            "Domain-specific false positives remain visible and documented rather than being "
            "silently converted to neutral."
        ),
    }
    write_json(args.report, report)
    print(
        f"Short text: {short['sentiment']} / {short['topic_status']}; "
        f"jargon example: {jargon['sentiment']} ({jargon['sentiment_compound']:.4f})"
    )
    if not short_handled:
        raise ValueError("Short text edge-case behavior did not match the documented policy")


if __name__ == "__main__":
    main()
