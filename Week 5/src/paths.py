"""Project paths — Week 5 reads from the Week 4 index and enriched data."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent

# Week 4 data (read-only references)
WEEK4_ROOT = REPO_ROOT / "Week 4"
INDEX_DIR = WEEK4_ROOT / "chroma_db"
INDEX_MANIFEST = WEEK4_ROOT / "reports" / "index_manifest.json"
ENRICHED_CHUNKS = WEEK4_ROOT / "data" / "processed" / "enriched_chunks.jsonl"
TOPIC_LABELS = WEEK4_ROOT / "config" / "topic_labels.json"

# Week 2 fallback paths (for vector reuse checks)
WEEK2_CHUNKS = REPO_ROOT / "Week 2" / "data" / "processed" / "chunks.jsonl"
WEEK2_DB = REPO_ROOT / "Week 2" / "chroma_db"
WEEK2_MANIFEST = REPO_ROOT / "Week 2" / "reports" / "index_manifest.json"

# Week 5 outputs
REPORTS_DIR = ROOT / "reports"
RETRIEVAL_GOLD = ROOT / "queries" / "retrieval_gold.jsonl"
LOG_DIR = ROOT / "logs"

# Aliases expected by vector_store module
INDEX_REPORT = WEEK4_ROOT / "reports" / "index_report.json"
INDEX_VERIFICATION = WEEK4_ROOT / "reports" / "index_verification.json"
INPUT_CHUNKS = REPO_ROOT / "Week 3" / "data" / "processed" / "chunks.jsonl"
GOLD_LABELS = WEEK4_ROOT / "queries" / "sentiment_gold.jsonl"
NLP_REPORT = WEEK4_ROOT / "reports" / "nlp_analysis.json"
SENTIMENT_REPORT = WEEK4_ROOT / "reports" / "sentiment_evaluation.json"
