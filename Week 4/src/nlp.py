"""Reproducible LDA topics and lexicon-based sentiment for Wikipedia chunks."""

import json
import re
from collections import Counter
from time import perf_counter

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

TOKEN_PATTERN = r"(?u)\b[a-z][a-z0-9-]{1,}\b"
MIN_TOPIC_CHARS = 80
MIN_SENTIMENT_WORDS = 4
DOMAIN_STOP_WORDS = {
    "also",
    "among",
    "became",
    "called",
    "century",
    "first",
    "following",
    "including",
    "known",
    "later",
    "many",
    "may",
    "new",
    "often",
    "one",
    "people",
    "several",
    "since",
    "states",
    "united",
    "used",
    "using",
    "well",
    "would",
    "wikipedia",
}


def _topic_labels(path) -> dict[int, str]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid topic label file: {error.msg}") from None
    labels = raw.get("labels", {}) if isinstance(raw, dict) else None
    if not isinstance(labels, dict):
        raise ValueError("Topic labels must be an object under the 'labels' key")
    result = {}
    for key, value in labels.items():
        try:
            topic_id = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"Topic label key must be an integer: {key!r}") from None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Topic label {key!r} must be a nonempty string")
        result[topic_id] = value.strip()
    return result


def analyze_sentiment(text: str, analyzer: SentimentIntensityAnalyzer | None = None) -> dict:
    if not isinstance(text, str):
        raise ValueError("Sentiment input must be text")
    words = re.findall(TOKEN_PATTERN, text)
    if len(words) < MIN_SENTIMENT_WORDS:
        return {
            "sentiment": "unknown",
            "sentiment_compound": 0.0,
            "sentiment_positive": 0.0,
            "sentiment_neutral": 0.0,
            "sentiment_negative": 0.0,
            "sentiment_status": "insufficient_text",
        }
    analyzer = analyzer or SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {
        "sentiment": label,
        "sentiment_compound": float(compound),
        "sentiment_positive": float(scores["pos"]),
        "sentiment_neutral": float(scores["neu"]),
        "sentiment_negative": float(scores["neg"]),
        "sentiment_status": "scored",
    }


def enrich_chunks(chunks: list[dict], *, requested_topics: int = 6, labels_path) -> tuple[list[dict], dict]:
    if not chunks:
        raise ValueError("No chunks were supplied for NLP analysis")
    if requested_topics < 1:
        raise ValueError("Topic count must be at least one")
    for number, row in enumerate(chunks, 1):
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError(f"Chunk {number} has no valid ID")
        if not isinstance(row.get("text"), str):
            raise ValueError(f"Chunk {row['id']} has invalid text")

    original_requested_topics = requested_topics
    eligible = [
        index
        for index, row in enumerate(chunks)
        if len(row["text"].strip()) >= MIN_TOPIC_CHARS
        and len(re.findall(TOKEN_PATTERN, row["text"])) >= MIN_SENTIMENT_WORDS
    ]
    if not eligible:
        raise ValueError(
            f"No chunks contain at least {MIN_TOPIC_CHARS} characters and "
            f"{MIN_SENTIMENT_WORDS} words for topic modeling"
        )

    training_texts = [chunks[index]["text"] for index in eligible]
    title_terms = {
        token.lower()
        for row in chunks
        for token in re.findall(TOKEN_PATTERN, row.get("title", "").lower())
    }
    stop_words = set(ENGLISH_STOP_WORDS) | DOMAIN_STOP_WORDS | title_terms
    vectorizer = CountVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words=sorted(stop_words),
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 2),
        min_df=2 if len(training_texts) > 2 else 1,
        max_df=0.90 if len(training_texts) > 2 else 1.0,
        max_features=25_000,
    )
    try:
        matrix = vectorizer.fit_transform(training_texts)
    except ValueError as error:
        # Tiny or highly repetitive corpora can make min_df/max_df remove every term.
        if "empty vocabulary" not in str(error).lower() and "max_df corresponds" not in str(error):
            raise
        vectorizer.set_params(min_df=1, max_df=1.0)
        try:
            matrix = vectorizer.fit_transform(training_texts)
        except ValueError as fallback_error:
            raise ValueError(
                "Topic model has no usable vocabulary after the small-corpus fallback; "
                "check that input chunks contain meaningful words"
            ) from fallback_error
    if matrix.shape[1] == 0:
        raise ValueError("Topic model vocabulary is empty")

    topic_count = min(requested_topics, matrix.shape[1], matrix.shape[0])
    if topic_count < requested_topics:
        # Small corpora cannot support more distinct topics than usable terms/documents.
        requested_topics = topic_count
    model = LatentDirichletAllocation(
        n_components=topic_count,
        doc_topic_prior=0.15,
        topic_word_prior=0.02,
        learning_method="batch",
        max_iter=20,
        random_state=42,
        evaluate_every=-1,
    )
    started = perf_counter()
    model.fit(matrix)
    all_matrix = vectorizer.transform([row["text"] for row in chunks])
    distributions = model.transform(all_matrix)
    elapsed = perf_counter() - started
    feature_names = vectorizer.get_feature_names_out()
    manual_labels = _topic_labels(labels_path)
    sentiment_analyzer = SentimentIntensityAnalyzer()
    topic_summaries = []
    for topic_id, weights in enumerate(model.components_):
        top_indices = weights.argsort()[-10:][::-1]
        terms = [str(feature_names[index]) for index in top_indices]
        topic_summaries.append(
            {
                "topic_id": topic_id,
                "label": manual_labels.get(topic_id, "topic " + str(topic_id)),
                "top_terms": terms,
                "training_chunk_count": int(
                    sum(int(distributions[index].argmax()) == topic_id for index in eligible)
                ),
                "representative_chunks": [],
            }
        )
    for row_index, row in enumerate(chunks):
        if all_matrix[row_index].nnz == 0:
            row_distribution = None
            topic_id = -1
        else:
            row_distribution = distributions[row_index]
            topic_id = int(row_distribution.argmax())
        if topic_id >= 0:
            confidence = float(row_distribution[topic_id])
            distribution = [float(value) for value in row_distribution]
            topic_label = manual_labels.get(topic_id, "topic " + str(topic_id))
            summary = topic_summaries[topic_id]
            summary["representative_chunks"].append(
                {
                    "id": row["id"],
                    "title": row.get("title", ""),
                    "chunk_index": row.get("chunk_index", -1),
                    "confidence": round(confidence, 6),
                    "preview": row["text"][:280],
                }
            )
            topic_status = "assigned"
        else:
            confidence = 0.0
            distribution = []
            topic_label = "unassigned"
            topic_status = "no_vocabulary_terms"
        sentiment = analyze_sentiment(row["text"], sentiment_analyzer)
        row.update(
            {
                "topic_id": topic_id,
                "topic_label": topic_label,
                "topic_confidence": confidence,
                "topic_distribution": distribution,
                "topic_status": topic_status,
                **sentiment,
            }
        )
    for summary in topic_summaries:
        summary["representative_chunks"] = sorted(
            summary["representative_chunks"],
            key=lambda item: item["confidence"],
            reverse=True,
        )[:3]
    labels = Counter(row["sentiment"] for row in chunks)
    topic_counts = Counter(row["topic_id"] for row in chunks)
    report = {
        "status": "complete",
        "method": "scikit-learn LatentDirichletAllocation with count-vectorized unigrams and bigrams",
        "sentiment_method": "VADER lexicon and rule-based compound score",
        "requested_topic_count": original_requested_topics,
        "topic_count": topic_count,
        "chunk_count": len(chunks),
        "topic_training_chunk_count": len(eligible),
        "topic_unassigned_count": topic_counts.get(-1, 0),
        "sentiment_counts": dict(labels),
        "topic_counts": {str(key): value for key, value in sorted(topic_counts.items())},
        "topic_model": {
            "random_state": 42,
            "max_iter": 20,
            "learning_method": "batch",
            "min_document_chars": MIN_TOPIC_CHARS,
            "token_pattern": TOKEN_PATTERN,
            "min_df": vectorizer.min_df,
            "max_df": vectorizer.max_df,
            "vocabulary_size": len(feature_names),
            "corpus_title_terms_excluded": sorted(title_terms),
            "perplexity_on_training_chunks": float(model.perplexity(matrix)),
            "fit_seconds": elapsed,
        },
        "sentiment_rules": {
            "positive": "compound >= 0.05",
            "neutral": "-0.05 < compound < 0.05",
            "negative": "compound <= -0.05",
            "minimum_words": MIN_SENTIMENT_WORDS,
            "short_text_label": "unknown",
        },
        "topics": topic_summaries,
        "interpretation_note": (
            "LDA terms and representative passages require human review. Topic IDs are "
            "model clusters, not verified subject labels. VADER is a lexical tone estimate, "
            "not a factual quality or stance classifier."
        ),
    }
    return chunks, report
