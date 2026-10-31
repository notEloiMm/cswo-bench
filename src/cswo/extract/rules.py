"""Rule-based extraction (spec 5): water-condition terms, and the gold entity types NER models do not cover.

* water condition: record-level multi-label (the gold label set minus "none stated") with the matched spans,
  from configs/water_terms.yaml in the note's language + English;
* WATER_BODY and WEATHER spans from the same config;
* ORGANISM: Latin names whose genus occurs in the corpus' own species labels ("Esox lucius", "Lemna sp.");
* QUANTITY: numbers with an optional unit or count word; DATE_TIME: dates, clock times, months/weekdays are left
  to the NER models.
Spans are character offsets into the text passed in.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from cswo.collect.common import read_jsonl
from cswo.harmonize.water_filter import _compile

_TOKEN_CHAR = re.compile(r"[\w'-]")

_DATE = re.compile(
    r"(?<![\w.,])(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[./]\d{1,2}[./]\d{2,4}|\d{1,2}\.\s?\d{1,2}\.(?!\d)"
    r"|(?:kl\.?|klo|at|um|à)\s?\d{1,2}[:.]\d{2}|\d{1,2}:\d{2}(?::\d{2})?)(?![\w,])", re.I)
_UNIT = (r"(?:\s?(?:mm|cm|dm|m|km|m2|m²|ha|m/s|km/h|l/s|m3/s|°C|°|%|uS|µS|ml|l|kg|g|"
         r"ex\.?|exx|st\.?|kpl|ind\.?|individuals?|par|pairs?|pari[a]?|pull|juv|ad))?")
_QUANTITY = re.compile(r"(?<![\w.,])(?:ca\.?\s?|c:a\s?|~\s?)?\d+(?:[.,]\d+)?(?:\s?[-–]\s?\d+(?:[.,]\d+)?)?" + _UNIT +
                       r"(?![\w])", re.I)
_CAPWORD = re.compile(r"\b[A-Z][a-z]{3,}\b")
_EPITHET = re.compile(r"\s+(?:(?:sp|spp|cf|aff)\.?|[a-z]{3,}(?:\s+[a-z]{3,})?)(?![\w])")

WATER_CONDITION_LABELS = ["clear", "turbid", "coloured", "algae", "foam", "ice", "oil", "litter", "odour",
                          "high_water", "low_or_dry", "flooding"]


def _expand(text: str, start: int, end: int) -> tuple[int, int]:
    """Grow a match to whole-token boundaries ("joki" inside "Oulujoki" -> "Oulujoki")."""
    while start > 0 and _TOKEN_CHAR.match(text[start - 1]):
        start -= 1
    while end < len(text) and _TOKEN_CHAR.match(text[end]):
        end += 1
    return start, end


def term_spans(text: str, lang: str | None, section: str) -> list[dict]:
    spans = []
    for label, pat in _compile(section, lang or "en").items():
        for m in pat.finditer(text):
            s, e = _expand(text, m.start(), m.end())
            spans.append({"start": s, "end": e, "text": text[s:e], "label": label})
    return _dedupe(spans)


@lru_cache
def genera(corpus_path: str) -> frozenset[str]:
    """Genus names from the corpus' structured species labels (iNaturalist, GBIF portals)."""
    out = set()
    for r in read_jsonl(Path(corpus_path)):
        sp = (r.get("labels") or {}).get("species")
        if sp:
            g = sp.split()[0]
            if g[:1].isupper() and len(g) >= 4 and g.isalpha():
                out.add(g)
    return frozenset(out)


def organism_spans(text: str, known_genera: frozenset[str]) -> list[dict]:
    spans = []
    for m in _CAPWORD.finditer(text):
        if m.group() in known_genera:
            end = m.end()
            if ep := _EPITHET.match(text, end):
                end = ep.end()
            spans.append({"start": m.start(), "end": end, "text": text[m.start():end], "label": "ORGANISM"})
    return spans


def number_spans(text: str) -> list[dict]:
    dates = [{"start": m.start(), "end": m.end(), "text": m.group(), "label": "DATE_TIME"} for m in _DATE.finditer(text)]
    taken = [(d["start"], d["end"]) for d in dates]
    qty = [{"start": m.start(), "end": m.end(), "text": m.group(), "label": "QUANTITY"}
           for m in _QUANTITY.finditer(text)
           if not any(s < m.end() and m.start() < e for s, e in taken)]
    return dates + qty


def _dedupe(spans: list[dict]) -> list[dict]:
    """Drop exact duplicates and spans contained in a longer span with the same label."""
    spans = sorted({(s["start"], s["end"], s["label"]): s for s in spans}.values(),
                   key=lambda s: (s["start"], -(s["end"] - s["start"])))
    out = []
    for s in spans:
        if not any(o["label"] == s["label"] and o["start"] <= s["start"] and s["end"] <= o["end"] for o in out):
            out.append(s)
    return out


def extract(text: str, lang: str | None, known_genera: frozenset[str]) -> dict:
    """All rule outputs for one text: entity spans (gold label names) and water-condition labels."""
    cond = term_spans(text, lang, "water_condition")
    entities = (
        [{**s, "label": "WATER_BODY"} for s in term_spans(text, lang, "water_body")]
        + [{**s, "label": "WEATHER"} for s in term_spans(text, lang, "weather")]
        + organism_spans(text, known_genera) + number_spans(text)
    )
    return {"entities": _dedupe(entities), "water_condition": sorted({s["label"] for s in cond}),
            "water_condition_spans": cond}
