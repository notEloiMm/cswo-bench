"""Language identification for every text field (spec 2), using lingua (accurate on short text, runs offline)."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache

from lingua import Language, LanguageDetectorBuilder

# Corpus languages (spec) + neighbours that show up on the same portals; restricting the set improves accuracy.
LANGS = [Language.ENGLISH, Language.GERMAN, Language.FRENCH, Language.SWEDISH, Language.FINNISH,
         Language.ESTONIAN, Language.LATVIAN, Language.LITHUANIAN, Language.DUTCH, Language.DANISH,
         Language.BOKMAL, Language.NYNORSK, Language.SPANISH, Language.ITALIAN, Language.POLISH, Language.RUSSIAN,
         # CrowdWater is global: without these, Portuguese/Hungarian/Czech notes are forced into a wrong language
         Language.PORTUGUESE, Language.HUNGARIAN, Language.CZECH, Language.SLOVAK, Language.SLOVENE,
         Language.CROATIAN, Language.ROMANIAN, Language.CATALAN, Language.ICELANDIC]
MIN_REL_DISTANCE = 0.1     # lingua returns no language when the top two are closer than this (tuned on samples)
MIN_CHARS = 3
NEIGHBOURS = {"sv": {"no", "da", "is"}, "no": {"sv", "da", "is"}, "da": {"no", "sv"}, "fi": {"et"}, "et": {"fi"}}
HINT_MIN_CONF = 0.05


@lru_cache
def _detector():
    return (LanguageDetectorBuilder.from_languages(*LANGS)
            .with_minimum_relative_distance(MIN_REL_DISTANCE).with_preloaded_language_models().build())


def _code(lang: Language) -> str:
    code = lang.iso_code_639_1.name.lower()
    return "no" if code in {"nb", "nn"} else code


def detect(text: str | None, hint: str | None = None) -> tuple[str | None, float | None]:
    """Return (ISO 639-1 code, confidence); 'und' when the text is ambiguous, None when there is no text.

    ``hint`` is the portal country's main language. It only resolves an ambiguous note when it is
    already the top-ranked candidate, so English notes from Lithuania stay English.
    """
    if not text or len(text.strip()) < MIN_CHARS:
        return None, None
    det = _detector()
    confs = det.compute_language_confidence_values(text)
    lang = det.detect_language_of(text)
    if lang is None:
        if hint and confs and _code(confs[0].language) == hint:
            lang = confs[0].language
        else:
            return "und", round(confs[0].value, 3) if confs else None
    # Short notes are often confused with a close neighbour ("2 hannar + 1 hona" -> Icelandic); when the portal's
    # language is a neighbour of the guess and still a plausible candidate, trust the portal.
    if hint and _code(lang) in NEIGHBOURS.get(hint, ()):
        alt = next((c for c in confs if _code(c.language) == hint), None)
        if alt is not None and alt.value >= HINT_MIN_CONF:
            lang = alt.language
    conf = next((c.value for c in confs if c.language == lang), None)
    return _code(lang), round(conf, 3) if conf is not None else None


def detect_all(text: str | None, min_share: float = 0.2) -> list[str] | None:
    """Languages covering >= min_share of the characters, for notes written in several languages
    (common on Finnish iNat: the same note in fi/sv/en). None for single-language or short notes."""
    if not text or len(text) < 40:
        return None
    spans = _detector().detect_multiple_languages_of(text)
    chars = Counter()
    for s in spans:
        chars[_code(s.language)] += s.end_index - s.start_index
    langs = [lg for lg, n in chars.most_common() if n / len(text) >= min_share]
    return langs if len(langs) > 1 else None


COUNTRY_LANG = {"FI": "fi", "SE": "sv", "EE": "et", "LV": "lv", "LT": "lt", "NO": "no", "DK": "da",
                "DE": "de", "FR": "fr", "NL": "nl", "BE": "nl", "GB": "en", "US": "en", "CA": "en"}
