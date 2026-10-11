"""Deterministic, composable cleaning with explicit language filtering.

No lowercasing, stemming, or stopword removal: article meaning and names are retained.
The language model uses a fixed seed and a bounded prefix to limit CPU work.
"""

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from bs4 import BeautifulSoup, Comment
from langdetect import LangDetectException
from langdetect.detector_factory import PROFILES_DIRECTORY, DetectorFactory

HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>|<!--[\s\S]*?-->")
WHITESPACE = re.compile(r"\s+")


def _require_text(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError("text must be a string")


def strip_html(text: str) -> str:
    """Remove HTML, comments, and script/style content; decode entities once.

    Only parse tag-like input. Plain mathematical expressions such as `x < 5`
    must survive. Entity-encoded markup is decoded before tag removal.
    """
    import html

    _require_text(text)
    text = html.unescape(text)
    if not HTML_TAG.search(text):
        return text
    soup = BeautifulSoup(text, "html.parser")
    for element in soup(["script", "style", "noscript", "template"]):
        element.decompose()
    for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
        comment.extract()
    return soup.get_text(separator=" ")


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Normalize Unicode and remove invisible formatting/control characters.

    Whitespace controls are left for the whitespace stage. Zero-width spaces become
    word boundaries; other formatting controls (including BOM/soft hyphen) disappear.
    """
    _require_text(text)
    text = unicodedata.normalize(form, text)
    result = []
    for character in text:
        category = unicodedata.category(character)
        if character == "\u200b":
            result.append(" ")
        elif category in {"Cf", "Cs"}:
            continue
        elif category == "Cc" and not character.isspace():
            result.append(" ")
        else:
            result.append(character)
    return "".join(result)


def normalize_whitespace(text: str) -> str:
    """Collapse Unicode whitespace to one ASCII space and trim both ends."""
    _require_text(text)
    return WHITESPACE.sub(" ", text).strip()


def clean_text(text: str) -> str:
    """Apply Unicode normalization, HTML cleanup, and whitespace cleanup."""
    return normalize_whitespace(normalize_unicode(strip_html(normalize_unicode(text))))


@lru_cache(maxsize=1)
def _language_factory() -> DetectorFactory:
    factory = DetectorFactory()
    factory.load_profile(PROFILES_DIRECTORY)
    factory.seed = 0
    return factory


@dataclass(frozen=True)
class LanguageResult:
    code: str
    confidence: float


def detect_language(text: str, sample_chars: int = 2000) -> LanguageResult:
    """Return the most likely language, or `unknown` for unidentifiable input."""
    _require_text(text)
    if sample_chars <= 0:
        raise ValueError("sample_chars must be positive")
    if not text.strip() or not any(character.isalpha() for character in text):
        return LanguageResult("unknown", 0.0)
    detector = _language_factory().create()
    detector.append(text[:sample_chars])
    try:
        result = detector.get_probabilities()
    except LangDetectException:
        return LanguageResult("unknown", 0.0)
    if not result:
        return LanguageResult("unknown", 0.0)
    return LanguageResult(result[0].lang, float(result[0].prob))


def is_target_language(
    text: str, target_language: str = "en", min_confidence: float = 0.90
) -> bool:
    """Accept only the requested language above the confidence threshold."""
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")
    result = detect_language(text)
    return result.code == target_language and result.confidence >= min_confidence


@dataclass(frozen=True)
class CleaningConfig:
    target_language: str = "en"
    min_chars: int = 100
    min_language_confidence: float = 0.90

    def __post_init__(self) -> None:
        if self.min_chars < 1:
            raise ValueError("min_chars must be positive")
        if not 0 <= self.min_language_confidence <= 1:
            raise ValueError("min_language_confidence must be between 0 and 1")
        if not self.target_language:
            raise ValueError("target_language cannot be empty")


def clean_document(record: dict, config: CleaningConfig) -> tuple[dict | None, str | None]:
    """Clean one article, preserving attribution; report why a document is rejected."""
    if not isinstance(record, dict):
        return None, "invalid_record"
    if any(not isinstance(record.get(key), str) for key in ("id", "title", "url", "text")):
        return None, "invalid_record"
    if any(not record[key].strip() for key in ("id", "title", "url")):
        return None, "invalid_record"
    text = clean_text(record["text"])
    title = clean_text(record["title"])
    if not title:
        return None, "invalid_record"
    if len(text) < config.min_chars:
        return None, "too_short"
    language = detect_language(text)
    if language.code != config.target_language:
        return None, "wrong_language"
    if language.confidence < config.min_language_confidence:
        return None, "low_language_confidence"
    return {
        "id": record["id"],
        "title": title,
        "url": record["url"],
        "text": text,
        "source": "wikimedia/wikipedia",
        "language": language.code,
        "language_confidence": language.confidence,
        "char_count": len(text),
        "word_count": len(text.split()),
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }, None
