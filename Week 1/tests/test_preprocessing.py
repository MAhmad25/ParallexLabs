"""Cleaning and language edge cases; no network or pretrained model required."""

import pytest

from src.preprocessing import (
    CleaningConfig,
    clean_document,
    clean_text,
    detect_language,
    is_target_language,
    normalize_unicode,
    normalize_whitespace,
    strip_html,
)

ENGLISH = (
    "The university library contains many books about science and history. "
    "Students visit the library every day to read books and study for their examinations. "
    "The librarian helps people find reliable information about the world around them."
)
FRENCH = (
    "La bibliothèque de cette université contient beaucoup de livres sur la science et "
    "l'histoire. Les étudiants viennent chaque jour pour lire et préparer leurs examens. "
    "Les bibliothécaires aident les personnes à trouver des informations fiables."
)
SPANISH = (
    "La biblioteca de la universidad tiene muchos libros sobre la ciencia y la historia. "
    "Los estudiantes visitan la biblioteca todos los días para estudiar y preparar sus "
    "exámenes. Los bibliotecarios ayudan a encontrar información sobre el mundo."
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("<p>Hello <b>world</b></p>", "Hello world"),
        ("<p>one</p><p>two</p>", "one two"),
        ("<script>alert('bad')</script><style>body{}</style><p>Safe</p>", "Safe"),
        ("<!--hidden--><p>Visible</p>", "Visible"),
        ("<div>Unclosed <b>markup", "Unclosed markup"),
        ("Tom &amp; Jerry &nbsp; &lt; 5", "Tom & Jerry < 5"),
        ("&lt;p&gt;Encoded markup&lt;/p&gt;", "Encoded markup"),
        ("if x < 5 and y > 3", "if x < 5 and y > 3"),
        ("<noscript>hidden</noscript><template>hidden</template>Keep", "Keep"),
        ("", ""),
    ],
)
def test_html_cleanup(value, expected):
    assert normalize_whitespace(strip_html(value)) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("cafe\u0301", "café"),
        ("ＡＢＣ ① ﬁ", "ABC 1 fi"),
        ("hello\u200bworld", "hello world"),
        ("\ufeffsoft\u00adhyphen", "softhyphen"),
        ("left\x00right", "left right"),
        ("hello\ud800world", "helloworld"),
        ("السلام علیکم", "السلام علیکم"),
    ],
)
def test_unicode_normalization(value, expected):
    assert normalize_unicode(value) == expected


@pytest.mark.parametrize("value", [" a\t b\n c ", "a\u00a0b\u2003c", "a   b\r\nc"])
def test_whitespace_cleanup(value):
    assert normalize_whitespace(value) == "a b c"


def test_complete_pipeline_preserves_case_punctuation_and_accents():
    assert clean_text(" <p>Ｃafé\u00a0 &amp; SCIENCE!</p>\n") == "Café & SCIENCE!"


def test_clean_text_is_idempotent_for_normal_article_text():
    cleaned = clean_text("<h1>History</h1><p>Cafe\u0301\t science.</p>")
    assert clean_text(cleaned) == cleaned


@pytest.mark.parametrize(
    "function", [strip_html, normalize_unicode, normalize_whitespace, clean_text]
)
@pytest.mark.parametrize("value", [None, 42])
def test_nonstring_input_rejected(function, value):
    with pytest.raises(TypeError, match="text must be a string"):
        function(value)


@pytest.mark.parametrize("text", ["", "  ", "12345!?"])
def test_unknown_language_for_empty_or_nonlinguistic_input(text):
    assert detect_language(text).code == "unknown"
    assert not is_target_language(text)


def test_language_filter_accepts_english_and_rejects_other_languages():
    assert is_target_language(ENGLISH)
    assert not is_target_language(FRENCH)
    assert not is_target_language(SPANISH)
    assert is_target_language(FRENCH, "fr")


def test_language_detection_repeatable():
    assert detect_language(ENGLISH) == detect_language(ENGLISH)


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_language_threshold(threshold):
    with pytest.raises(ValueError):
        is_target_language(ENGLISH, min_confidence=threshold)


@pytest.mark.parametrize(
    "kwargs", [{"min_chars": 0}, {"min_language_confidence": 2}, {"target_language": ""}]
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        CleaningConfig(**kwargs)


def test_document_metadata_and_rejection_reasons():
    record = {"id": "1", "title": "Library", "url": "https://en.wikipedia.org/wiki/Library"}
    config = CleaningConfig()
    cleaned, reason = clean_document({**record, "text": ENGLISH}, config)
    assert reason is None
    assert cleaned["id"] == "1"
    assert cleaned["language"] == "en"
    assert cleaned["char_count"] == len(cleaned["text"])
    assert len(cleaned["content_sha256"]) == 64
    assert clean_document(record, config) == (None, "invalid_record")
    assert clean_document({**record, "text": "short"}, config) == (None, "too_short")
    assert clean_document({**record, "text": FRENCH}, config) == (None, "wrong_language")


def test_low_confidence_document_is_rejected(monkeypatch):
    from src import preprocessing

    monkeypatch.setattr(
        preprocessing, "detect_language", lambda text: preprocessing.LanguageResult("en", 0.5)
    )
    record = {"id": "1", "title": "Library", "url": "https://example.org", "text": ENGLISH}
    assert clean_document(record, CleaningConfig()) == (None, "low_language_confidence")
