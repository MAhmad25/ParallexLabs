import math
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from google.genai import errors

from src.gemini import Gemini, Settings, normalize_vector


def response(vector=None, count=1):
    return SimpleNamespace(
        embeddings=[SimpleNamespace(values=vector or [1.0] * 768) for _ in range(count)]
    )


def make_gemini(model="gemini-embedding-001"):
    client = Mock()
    client.models.embed_content.return_value = response()
    return Gemini(Settings("test-key", embedding_model=model), client=client, sleeper=Mock())


@pytest.mark.parametrize(
    "changes",
    [
        {"api_key": ""},
        {"embedding_model": "openai"},
        {"dimensions": 3},
        {"llm_model": "gpt-4"},
        {"llm_model": "gemini-embedding-001"},
    ],
)
def test_invalid_settings(changes):
    with pytest.raises(ValueError):
        Settings(**{"api_key": "test-key", **changes})


def test_secret_not_in_repr():
    assert "test-key" not in repr(Settings("test-key"))


@pytest.mark.parametrize(
    "vector,dims",
    [([], 3), ([1, 2], 3), ([0, 0, 0], 3), ([math.nan, 1, 1], 3), ([math.inf, 1, 1], 3)],
)
def test_invalid_vectors(vector, dims):
    with pytest.raises(ValueError):
        normalize_vector(vector, dims)


def test_normalization():
    assert normalize_vector([3, 4], 2) == [0.6, 0.8]


@pytest.mark.parametrize("query,task", [(True, "RETRIEVAL_QUERY"), (False, "RETRIEVAL_DOCUMENT")])
def test_embedding1_task_and_dimensions(query, task):
    gemini = make_gemini()
    vectors, seconds = gemini.embed(["hello"], query=query)
    kwargs = gemini.client.models.embed_content.call_args.kwargs
    assert kwargs["config"].task_type == task
    assert kwargs["config"].output_dimensionality == 768
    assert math.hypot(*vectors[0]) == pytest.approx(1)
    assert seconds >= 0


def test_embedding2_independent_vectors_and_prefix():
    gemini = make_gemini("gemini-embedding-2")
    vectors, _ = gemini.embed(["first", "second"], query=True)
    assert len(vectors) == 2
    calls = gemini.client.models.embed_content.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["contents"] == "task: search result | query: first"
    assert calls[1].kwargs["contents"] == "task: search result | query: second"
    assert calls[0].kwargs["config"].task_type is None


@pytest.mark.parametrize("texts", [[""], [" "], [None], ["a" * 8001]])
def test_bad_texts_fail_before_network(texts):
    gemini = make_gemini()
    with pytest.raises(ValueError):
        gemini.embed(texts)
    gemini.client.models.embed_content.assert_not_called()


def test_empty_batch_is_noop():
    gemini = make_gemini()
    assert gemini.embed([]) == ([], 0.0)
    gemini.client.models.embed_content.assert_not_called()


def test_wrong_response_count():
    gemini = make_gemini()
    gemini.client.models.embed_content.return_value = response(count=0)
    with pytest.raises(ValueError, match="count"):
        gemini.embed(["hello"])


def test_rate_limit_retries_and_redacts():
    gemini = make_gemini()
    gemini.client.models.embed_content.side_effect = errors.ClientError(
        429, {"error": {"message": "sensitive test-key"}}
    )
    with pytest.raises(RuntimeError, match="429") as raised:
        gemini.embed(["hello"])
    assert "test-key" not in str(raised.value)
    assert gemini.client.models.embed_content.call_count == 3
    assert [call.args[0] for call in gemini.sleeper.call_args_list] == [30, 30]


def test_auth_error_is_not_retried():
    gemini = make_gemini()
    gemini.client.models.embed_content.side_effect = errors.ClientError(403, {})
    with pytest.raises(RuntimeError, match="403"):
        gemini.embed(["hello"])
    assert gemini.client.models.embed_content.call_count == 1


def test_server_retry_delay_is_bounded():
    gemini = make_gemini()
    gemini.client.models.embed_content.side_effect = [
        errors.ClientError(429, {"error": {"details": [{"retryDelay": "75.5s"}]}}),
        response(),
    ]
    gemini.embed(["hello"])
    gemini.sleeper.assert_called_once_with(30)


def test_transient_network_failure_then_success():
    gemini = make_gemini()
    gemini.client.models.embed_content.side_effect = [httpx.ConnectError("test-key"), response()]
    vectors, _ = gemini.embed(["hello"])
    assert len(vectors) == 1
    assert gemini.client.models.embed_content.call_count == 2


def test_gemini_answer_uses_context_and_model():
    gemini = make_gemini()
    gemini.client.models.generate_content.return_value = SimpleNamespace(text="Evidence [1].")
    answer = gemini.answer("question", [{"metadata": {"title": "Source"}, "text": "Evidence"}])
    assert answer["model"] == "gemini-flash-latest"
    kwargs = gemini.client.models.generate_content.call_args.kwargs
    assert "[1] Source" in kwargs["contents"]
    assert "untrusted" in kwargs["config"].system_instruction


def test_answer_rejects_empty_context():
    with pytest.raises(ValueError):
        make_gemini().answer("question", [])
