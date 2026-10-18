import json

import pytest

from scripts.verify_index import verify
from src.retrieval import ingest, open_store, search
from src.vector_store import VectorStore, collection_name, snapshot_identity
from tests.conftest import FakeGemini


def test_persistent_store_query_and_idempotence(tmp_path, chunks):
    identity = snapshot_identity("test", "gemini-embedding-001", 3)
    store = VectorStore(tmp_path / "db", identity, expected=3)
    store.upsert(chunks, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    store.upsert(chunks, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    store.mark_ready()
    reopened = VectorStore(tmp_path / "db", identity)
    hits, ms = reopened.query([1, 0, 0], k=100)
    assert reopened.count() == 3
    assert len(hits) == 3
    assert hits[0]["metadata"]["document_id"] == "space"
    assert hits[0]["cosine_similarity"] == pytest.approx(1)
    assert ms >= 0


def test_incomplete_collection_cannot_be_queried(tmp_path):
    store = VectorStore(
        tmp_path / "db", snapshot_identity("test", "gemini-embedding-001", 3), expected=2
    )
    with pytest.raises(ValueError, match="incomplete"):
        store.query([1, 0, 0])
    with pytest.raises(ValueError, match="missing chunks"):
        store.mark_ready()


def test_missing_collection_is_not_silently_created(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        VectorStore(tmp_path / "db", snapshot_identity("test", "gemini-embedding-001", 3))


def test_model_dimensions_and_input_make_separate_snapshots():
    identities = [
        snapshot_identity("one", "gemini-embedding-001", 768),
        snapshot_identity("one", "gemini-embedding-2", 768),
        snapshot_identity("one", "gemini-embedding-001", 1536),
        snapshot_identity("two", "gemini-embedding-001", 768),
    ]
    assert len({collection_name(i) for i in identities}) == 4


@pytest.mark.parametrize("vectors", [[[1, 0]], [[0, 0, 0]], [[1, 0, float("nan")]], []])
def test_reject_bad_vectors_before_writing(tmp_path, chunks, vectors):
    store = VectorStore(
        tmp_path / "db", snapshot_identity("test", "gemini-embedding-001", 3), expected=1
    )
    with pytest.raises(ValueError):
        store.upsert(chunks[:1], vectors)
    assert store.count() == 0


def test_duplicate_ids_in_batch(tmp_path, chunks):
    store = VectorStore(
        tmp_path / "db", snapshot_identity("test", "gemini-embedding-001", 3), expected=1
    )
    with pytest.raises(ValueError, match="Duplicate"):
        store.upsert([chunks[0], chunks[0]], [[1, 0, 0], [1, 0, 0]])


def test_resume_after_failure_and_no_reembedding_on_rerun(tmp_path, chunk_file):
    db, manifest, report = (tmp_path / name for name in ("db", "manifest.json", "ingest.json"))
    failing = FakeGemini(fail_on=2)
    with pytest.raises(RuntimeError, match="interrupted"):
        ingest(chunk_file, db, manifest, report, failing, batch_size=1)
    assert json.loads(report.read_text())["stored_chunks"] == 1
    assert json.loads(report.read_text())["status"] == "failed"
    assert not manifest.exists()
    healthy = FakeGemini()
    result = ingest(chunk_file, db, manifest, report, healthy, batch_size=1)
    assert result["already_present"] == 1
    assert result["embedded_this_run"] == 2
    assert result["stored_chunks"] == 3
    assert healthy.calls == 2
    again = ingest(chunk_file, db, manifest, report, healthy)
    assert again["embedded_this_run"] == 0
    assert healthy.calls == 2
    assert (
        search(open_store(db, manifest, healthy), healthy, "planet orbit")["hits"][0]["metadata"][
            "document_id"
        ]
        == "space"
    )


def test_incompatible_query_configuration(tmp_path, chunk_file, fake_gemini):
    db, manifest, report = (tmp_path / name for name in ("db", "manifest.json", "ingest.json"))
    ingest(chunk_file, db, manifest, report, fake_gemini)
    fake_gemini.settings.dimensions = 4
    with pytest.raises(ValueError, match="settings differ"):
        open_store(db, manifest, fake_gemini)


def test_corrupt_collection_metadata_is_rejected(tmp_path, chunks):
    identity = snapshot_identity("test", "gemini-embedding-001", 3)
    store = VectorStore(tmp_path / "db", identity, expected=3)
    store.collection.modify(metadata={**identity, "dimensions": 4})
    with pytest.raises(ValueError, match="metadata mismatch"):
        VectorStore(tmp_path / "db", identity)


def test_duplicate_input_is_rejected_before_api(tmp_path, chunk_file, chunks, fake_gemini):
    with chunk_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(chunks[0]) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        ingest(chunk_file, tmp_path / "db", tmp_path / "manifest", tmp_path / "report", fake_gemini)
    assert fake_gemini.calls == 0


def test_independent_verification_detects_changed_database_text(tmp_path, chunk_file, fake_gemini):
    db, manifest, report = (tmp_path / name for name in ("db", "manifest.json", "ingest.json"))
    ingest(chunk_file, db, manifest, report, fake_gemini)
    checked = verify(chunk_file, db, manifest, tmp_path / "verification.json")
    assert checked["checked_chunks"] == 3
    assert checked["status"] == "passed"
    store = open_store(db, manifest, fake_gemini)
    ident = store.collection.get(limit=1)["ids"][0]
    store.collection.update(ids=[ident], documents=["Tampered source"], embeddings=[[1, 0, 0]])
    with pytest.raises(ValueError, match="verification failed"):
        verify(chunk_file, db, manifest, tmp_path / "verification.json")
