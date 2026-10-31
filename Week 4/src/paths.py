"""Project paths and defaults."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
INPUT_CHUNKS = REPO_ROOT / "Week 3" / "data" / "processed" / "chunks.jsonl"
ENRICHED_CHUNKS = ROOT / "data" / "processed" / "enriched_chunks.jsonl"
TOPIC_LABELS = ROOT / "config" / "topic_labels.json"
GOLD_LABELS = ROOT / "queries" / "sentiment_gold.jsonl"
NLP_REPORT = ROOT / "reports" / "nlp_analysis.json"
SENTIMENT_REPORT = ROOT / "reports" / "sentiment_evaluation.json"
INDEX_DIR = ROOT / "chroma_db"
INDEX_MANIFEST = ROOT / "reports" / "index_manifest.json"
INDEX_REPORT = ROOT / "reports" / "index_report.json"
INDEX_VERIFICATION = ROOT / "reports" / "index_verification.json"
WEEK2_CHUNKS = REPO_ROOT / "Week 2" / "data" / "processed" / "chunks.jsonl"
WEEK2_DB = REPO_ROOT / "Week 2" / "chroma_db"
WEEK2_MANIFEST = REPO_ROOT / "Week 2" / "reports" / "index_manifest.json"
