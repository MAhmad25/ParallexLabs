"""File-level integration: rejection accounting, deduplication, and validation."""

import json

import pytest

from scripts.verify_env import verification_exit_code
from src.io_utils import read_jsonl
from src.pipeline import process_corpus
from src.preprocessing import CleaningConfig
from src.validation import validate_corpus

ENGLISH = (
    "The city has a large public library where students read books and learn about history. "
    "The building contains several quiet rooms for studying and a collection of scientific "
    "journals. The library welcomes visitors every day and provides information to residents."
)


def write_raw(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def article(identifier, text=ENGLISH):
    return {
        "id": identifier,
        "title": "Library",
        "url": "https://en.wikipedia.org/wiki/Library",
        "text": text,
    }


@pytest.mark.parametrize("extension", [".jsonl", ".parquet"])
def test_pipeline_deduplicates_and_accounts_for_rejections(tmp_path, extension):
    raw = tmp_path / "raw.jsonl"
    output = tmp_path / ("clean" + extension)
    records = [article("1"), article("1"), article("2"), article("3", "short"), {"text": None}]
    write_raw(raw, records)
    report = process_corpus(raw, output, tmp_path / "report.json", CleaningConfig())
    assert report["raw_documents"] == 5
    assert report["clean_documents"] == 1
    assert report["rejected_documents"] == 4
    assert report["rejections_by_reason"] == {
        "duplicate_id": 1,
        "duplicate_content": 1,
        "too_short": 1,
        "invalid_record": 1,
    }
    validation = validate_corpus(output, CleaningConfig(), min_documents=1)
    assert validation["passed"]
    assert validation["sha256"] == report["output_sha256"]


def test_validator_detects_corrupted_text_and_insufficient_count(tmp_path):
    raw, output = tmp_path / "raw.jsonl", tmp_path / "clean.jsonl"
    write_raw(raw, [article("1")])
    process_corpus(raw, output, tmp_path / "report.json", CleaningConfig())
    record = list(read_jsonl(output))[0]
    record["text"] = "  " + record["text"]
    write_raw(output, [record, record])
    report = validate_corpus(output, CleaningConfig(), min_documents=5000)
    assert not report["passed"]
    assert report["errors"]["duplicate_id"] == 1
    assert report["errors"]["content_hash_mismatch"] == 2
    assert report["errors"]["unclean_whitespace"] == 2
    assert report["errors"]["below_minimum_document_count"] == 1


def test_malformed_json_does_not_replace_existing_output(tmp_path):
    raw, output = tmp_path / "raw.jsonl", tmp_path / "clean.jsonl"
    raw.write_text('{"invalid":\n', encoding="utf-8")
    output.write_text("previous successful output", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        process_corpus(raw, output, tmp_path / "report.json", CleaningConfig())
    assert output.read_text(encoding="utf-8") == "previous successful output"
    assert not output.with_suffix(".jsonl.tmp").exists()


def test_all_rejected_input_fails_without_publishing_empty_dataset(tmp_path):
    raw, output = tmp_path / "raw.jsonl", tmp_path / "clean.jsonl"
    write_raw(raw, [article("1", "short")])
    with pytest.raises(ValueError, match="No documents passed"):
        process_corpus(raw, output, tmp_path / "report.json", CleaningConfig())
    assert not output.exists()


def test_pipeline_refuses_to_overwrite_input(tmp_path):
    raw = tmp_path / "raw.jsonl"
    with pytest.raises(ValueError, match="different files"):
        process_corpus(raw, raw, tmp_path / "report.json", CleaningConfig())


def test_cpu_override_never_masks_failed_imports():
    report = {"imports_ok": True, "smoke_tests_ok": True, "cuda_verified": False}
    assert verification_exit_code(report, allow_cpu=False) == 1
    assert verification_exit_code(report, allow_cpu=True) == 0
    report["imports_ok"] = False
    assert verification_exit_code(report, allow_cpu=True) == 1


def test_cuda_verified_satisfies_strict_mode():
    report = {"imports_ok": True, "smoke_tests_ok": True, "cuda_verified": True}
    assert verification_exit_code(report, allow_cpu=False) == 0
