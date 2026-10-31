"""Decide whether a record concerns a lake, river, stream, coast or sea (spec 1).

A record is kept when its taxon is obligately aquatic (configs/aquatic_taxa.yaml, ``obligate: true``)
or when its locality / habitat / note mentions a water body in the record's language or English.
"""
from __future__ import annotations

import re
from functools import lru_cache

import yaml

from cswo import CONFIGS

# Compounding languages put the water word last: "Oulujoki", "Peipsijärv", "Haksjön", "Bodensee", "Zuidplas".
COMPOUND_LANGS = {"fi", "et", "sv", "de", "nl", "da", "no"}
MIN_PREFIX = 5                      # shorter terms must match a whole word
# Country names in localities ("Lammi, Hämeenlinna, Suomi") contain short water stems (suo-mi); remove them first.
_COUNTRY_NAMES = re.compile(
    r"(?<!\w)(suomi|finland|sverige|sweden|eesti|estonia|latvija|latvia|lietuva|lithuania|norge|noreg|norway|danmark|"
    r"denmark|deutschland|germany|france|nederland|netherlands|belgi[eë]|belgique|belgium|united kingdom|"
    r"canada|united states|usa)(?!\w)", re.I)


@lru_cache
def _terms() -> dict:
    return yaml.safe_load((CONFIGS / "water_terms.yaml").read_text(encoding="utf-8"))


@lru_cache
def _compile(section: str, lang: str) -> dict[str, re.Pattern]:
    langs = {lang, "en"} if lang else {"en"}
    out = {}
    for label, by_lang in _terms()[section].items():
        alts = []
        for lg in langs:
            for t in by_lang.get(lg, []):
                t_esc = re.escape(t.lower())
                if lg in COMPOUND_LANGS and len(t) >= 3:
                    alts.append(rf"{t_esc}\w{{0,4}}(?!\w)")          # compound head + short inflection
                elif len(t) >= MIN_PREFIX:
                    alts.append(rf"(?<!\w){t_esc}\w*")              # word prefix
                else:
                    alts.append(rf"(?<!\w){t_esc}(?!\w)")           # whole word
        if alts:
            out[label] = re.compile("|".join(alts), re.I)
    return out


def match_terms(text: str | None, lang: str | None, section: str = "water_body") -> list[str]:
    """Return the labels from ``section`` whose terms occur in ``text``."""
    if not text:
        return []
    return [label for label, pat in _compile(section, lang or "en").items() if pat.search(text)]


def water_body(fields: dict[str, str | None], lang: str | None,
               obligate_aquatic: bool) -> tuple[bool, str | None, str | None]:
    """Return (keep, water_body_type, evidence).

    ``fields`` maps a field name (locality, habitat, text) to its value; matches in locality/habitat win over text.
    """
    for name in ("habitat", "locality", "text"):
        value = fields.get(name)
        if value and name == "locality":
            value = _COUNTRY_NAMES.sub(" ", value)
        hits = match_terms(value, lang)
        if hits:
            return True, hits[0], f"keyword:{name}:{hits[0]}"
    if obligate_aquatic:
        return True, "unknown", "taxon:obligate_aquatic"
    return False, None, None
