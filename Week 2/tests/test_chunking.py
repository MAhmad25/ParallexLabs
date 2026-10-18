import hashlib
import json
import random

import pytest

from src.chunking import ChunkConfig, chunk_corpus, chunk_document, split_text


@pytest.mark.parametrize("size,overlap", [(0, 0), (1, 0), (10, -1), (10, 10), (10, 11)])
def test_invalid_config(size, overlap):
    with pytest.raises(ValueError):
        ChunkConfig(size, overlap)


@pytest.mark.parametrize("text", ["", " ", "\n\t  "])
def test_whitespace_empty(text):
    assert split_text(text) == []


@pytest.mark.parametrize("text", ["abc", "  abc  ", "é界🙂\nhello", "x" * 500])
@pytest.mark.parametrize("overlap", [0, 3, 19])
def test_exact_offsets_bounds_and_nonwhitespace_coverage(text, overlap):
    pieces = split_text(text, ChunkConfig(20, overlap))
    covered = set()
    starts = []
    for start, end, content in pieces:
        assert content == text[start:end]
        assert 0 < len(content) <= 20
        assert content == content.strip()
        covered.update(range(start, end))
        starts.append(start)
    assert starts == sorted(starts)
    assert all(i in covered for i, char in enumerate(text) if not char.isspace())


def test_boundary_prefers_paragraph_over_word():
    text = "a" * 14 + "\n\n" + "b word word word word word"
    assert split_text(text, ChunkConfig(25, 0))[0][2] == "a" * 14


def test_unbroken_text_has_configured_overlap():
    result = split_text("x" * 65, ChunkConfig(20, 5))
    assert [(start, end) for start, end, _ in result] == [(0, 20), (15, 35), (30, 50), (45, 65)]


def test_randomized_no_content_loss():
    generator = random.Random(22)
    for _ in range(100):
        text = "".join(generator.choices("ab é🙂\n.!?\t", k=300))
        size = generator.randint(2, 80)
        overlap = generator.randrange(size)
        covered = set()
        for start, end, value in split_text(text, ChunkConfig(size, overlap)):
            assert value == text[start:end]
            assert len(value) <= size
            covered.update(range(start, end))
        assert all(i in covered for i, char in enumerate(text) if not char.isspace())


def test_nonstring_input():
    with pytest.raises(TypeError):
        split_text(None)


def test_ids_are_stable_and_content_sensitive():
    record = {"id": "one", "text": "hello world"}
    assert chunk_document(record) == chunk_document(record)
    assert chunk_document(record)[0]["id"] != chunk_document({**record, "text": "changed"})[0]["id"]


@pytest.mark.parametrize(
    "record",
    [
        {"text": "hello"},
        {"id": 12, "text": "hello"},
        {"id": "x", "text": " "},
        {"id": "x", "text": None},
        {"id": "x", "text": "hello", "title": None},
        {"id": "x", "text": "hello", "content_sha256": "wrong"},
    ],
)
def test_reject_invalid_documents(record):
    with pytest.raises(ValueError):
        chunk_document(record)


def test_week1_hash_and_attribution():
    text = "Real source content."
    row = {
        "id": "12",
        "title": "Title",
        "url": "https://en.wikipedia.org/wiki/Title",
        "source": "wikimedia/wikipedia",
        "text": text,
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    chunk = chunk_document(row)[0]
    assert chunk["document_id"] == "12"
    assert chunk["url"] == row["url"]
    assert chunk["source"] == row["source"]


def test_pipeline_limits_and_atomic_failure(tmp_path):
    source, output, report = (tmp_path / name for name in ("input.jsonl", "out.jsonl", "r.json"))
    rows = [{"id": str(i), "text": "source " * 30} for i in range(3)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = chunk_corpus(source, output, report, limit=2)
    assert result["document_count"] == 2
    before = output.read_bytes()
    source.write_text(json.dumps(rows[0]) + "\ninvalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        chunk_corpus(source, output, report)
    assert output.read_bytes() == before
    assert not output.with_suffix(".jsonl.tmp").exists()


@pytest.mark.parametrize("mode", ["empty", "duplicate", "same-path", "negative-limit"])
def test_pipeline_rejects_invalid_input(tmp_path, mode):
    source, output, report = (tmp_path / name for name in ("in.jsonl", "out.jsonl", "r.json"))
    row = json.dumps({"id": "one", "text": "source"}) + "\n"
    source.write_text("" if mode == "empty" else row + row, encoding="utf-8")
    with pytest.raises(ValueError):
        chunk_corpus(
            source,
            source if mode == "same-path" else output,
            report,
            limit=-1 if mode == "negative-limit" else 0,
        )
