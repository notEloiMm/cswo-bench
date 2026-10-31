"""Personal-name masking for released text (spec 5; course note: usernames and names are personal data).

A span is masked as [PERSON] when any of these holds:
* an observer-credit rule fires ("Obs Lars-Åke Pettersson", "Medobservatör ...", "leg. ...", "havainnoija ...");
* the transformer says PER with score >= TRANSFORMER_MIN;
* a language-specific spaCy model says PERSON and the span has >= 2 words or overlaps a transformer PER
  (single capitalised words are where small spaCy models misfire, e.g. German nouns).
Never masked: spans that are known genera/species or contain a water-body word (place-like, not personal).
English translations are masked with the same name strings found in the original (MT copies names through)
plus spaCy English PERSON entities under the same rule.
"""
from __future__ import annotations

import re
import unicodedata

from cswo.harmonize.water_filter import match_terms

TOKEN = "[PERSON]"
TRANSFORMER_MIN = 0.8
TRANSFORMER_MIN_SINGLE = 0.95
_UP = "A-ZÅÄÖÜÕŠŽ"
_INITIAL = rf"[{_UP}](?:-[{_UP}])*-?\."          # "K.", "P-E-." (Artportalen writes Pär-Erik as P-E-.)
_WORD = rf"[{_UP}][\w'’-]+"                     # no full stop: "Pettersson. Fiskade" must not join
# a credit keyword never starts a name: "Lingdell, Utvärderad av Eva Engblom" is two credits, not three names
_NOT_TRIGGER = (r"(?!(?i:utvärderad|projektansvarig|provtagare|inrapporterat|medobservatör|observatör|leg|det|"
                r"foto|photo|kuvaaja|havainnoija)\b)")
# a person: optional initials, 1-3 name words, optional trailing initial ("K. Andersson", "Anna V.")
_PERSON = (rf"{_NOT_TRIGGER}(?:{_INITIAL}\s*)*{_NOT_TRIGGER}{_WORD}"
           rf"(?:\s+(?:[{_UP}]\.?\s+)?{_NOT_TRIGGER}{_WORD}){{0,2}}"   # middle initial: "Carlo F Jensen"
           rf"(?:\s+{_INITIAL})?")
_LIST = rf"{_PERSON}(?:\s*(?:,|&|\+|och|and|ja|und|og|en|y|e|et)\s*{_PERSON})*"
# Strong triggers introduce names; "obs" also means "note:" in Swedish, so after it only a name-shaped span
# (initial + surname, or two name words) counts. Only the trigger is case-insensitive.
_STRONG = (r"(?i:observat(?:ör|or)(?:er)?|medobs(?:ervatör(?:er)?)?|rapportör|leg|det|coll|observer(?:s)?|"
           r"observed by|seen by|photo(?:graph)? by|havainnoija(?:t)?|havaitsija|kuvaaja|beobachter(?:in)?|"
           r"vaatleja|fotograaf|inrapporterat av|provtagare|utvärderad av|projektansvarig|fundet af|foto von|"
           r"referens|källa|ref|kuva|löytäjä|observerad av|noterad av|funnen av|"
           # bare "Foto:" (Spanish, Swedish, German CrowdWater credits); "foto" alone is a common noun
           r"foto(?=\s*:)|foto de|foto av|foto by)")
_CREDIT = re.compile(rf"(?<!\w){_STRONG}\s*[.:]?\s*(?P<names>{_LIST})")
_OBS = re.compile(rf"(?<!\w)(?i:obs)\s*[.:]?\s*(?P<names>(?:{_INITIAL}\s*)+{_WORD}|{_WORD}\s+{_WORD}"
                  rf"(?:\s+{_WORD})?)(?:\s*(?:,|och|and|ja|&)\s*(?:{_PERSON}))*")
# Artportalen herbarium-style collector field: "Nuphar lutea/3D0j0148/Köpinge s:n/AEx/ /Anna Exempel,Lund";
# the name runs from the empty field after the collector's initials to the next comma or slash (the town is kept)
_COLLECTOR = re.compile(rf"/[{_UP}][\w-]{{1,4}}/ ?/ ?(?P<names>[^/,]+)")
# capitalised field words that follow a credit keyword but are not names
_NOT_NAMES = {"hane", "hona", "honor", "hanar", "ad", "juv", "par", "flock", "ex", "ca", "minst", "koiras",
              "naaras", "pari", "male", "female", "pair", "adult", "adults", "juvenile", "unga", "ungar",
              "sp", "cf", "möjligen", "troligen", "probably", "possibly"}


def credit_spans(text: str) -> list[tuple[int, int]]:
    """Names after an observer-credit keyword. The keyword itself is kept: "Obs [PERSON]"."""
    spans = []
    for pat in (_CREDIT, _OBS, _COLLECTOR):
        for m in pat.finditer(text):
            s = m.start("names")
            e = m.end()
            if text[s:e].split()[0].strip(".,").lower() in _NOT_NAMES:
                continue
            while e > s and text[e - 1] in " ,":
                e -= 1
            spans.append((s, e))
    return _merge(spans)


# words that make a capitalised span a place, organisation or record field rather than a person
_NOT_PERSON_WORDS = {"park", "lane", "close", "road", "street", "st", "avenue", "way", "drive", "county",
                     "provincial", "national", "reserve", "kommune", "kommun", "län", "lst", "länsstyrelsen",
                     "universitet", "university", "water", "metod", "record", "guid", "museum", "naturskydd",
                     "förening", "seura", "ry", "ab", "oy", "gmbh", "ltd", "inc", "co", "works", "polder",
                     # taxonomy breadcrumbs ("Insecta Order", "Dytiscoidea Family")
                     "kingdom", "phylum", "class", "order", "superfamily", "family", "subfamily", "tribe", "genus",
                     "species", "subspecies", "insecta", "coleoptera", "odonata", "aves",
                     # head nouns of English common names of organisms ("Herring Gull", "Common Loon")
                     "gull", "loon", "wigeon", "duck", "goose", "swan", "heron", "egret", "grebe", "tern", "teal",
                     "merganser", "cormorant", "kingfisher", "coot", "moorhen", "rail", "sandpiper", "plover",
                     "dipper", "wagtail", "warbler", "stonefly", "mayfly", "caddisfly", "dragonfly", "damselfly",
                     "darter", "skimmer", "hawker", "chaser", "bluet", "dancer", "frog", "toad", "newt",
                     "salamander", "trout", "salmon", "pike", "perch", "bass", "carp", "sunfish", "minnow",
                     "mussel", "clam", "snail", "crayfish", "beetle", "lily", "pondweed", "duckweed", "sedge",
                     "rush", "reed", "milfoil", "otter", "beaver", "seal", "turtle",
                     "lac", "lago", "nwr", "refuge", "storm", "info"}
# a following credit keyword is not part of the name ("Jan Åslund, Utvärderad av ...")
_TRAILING_TRIGGER = re.compile(r",?\s*(?i:utvärderad|projektansvarig|provtagare|inrapporterat|medobservatör(?:er)?|"
                               r"observatör(?:er)?|leg|det|foto|photo|kuvaaja|havainnoija)\s*$")
_BINOMIAL = re.compile(r"^[A-Z][a-z]+ [a-z]{3,}$")          # "Cicadella viridis": a species, not a person


def _protected(span_text: str, lang: str | None, genera: frozenset[str], credit: bool = False) -> bool:
    """True when a candidate span is not a person. Water words are matched as separate English words only
    ("Lake Ontario", "Sewage Works"): Finnish and Estonian surnames are built from nature words (Korpisuo,
    Järvinen, Rantanen), so compound matching would unmask them. Credit-rule spans ("Kuva Mika Korpisuo") skip
    the genus and water checks: the keyword already says the span is a person."""
    words = [re.sub(r"['’]s$", "", w.strip(".,;:()")).lower() for w in span_text.split()]
    caps = [w.strip(".,;:()") for w in span_text.split()]
    if any(w in _NOT_PERSON_WORDS or w.rstrip("s") in _NOT_PERSON_WORDS or w in _NOT_NAMES for w in words):
        return True
    if credit:
        return False
    return (any(w in genera for w in caps) or bool(_BINOMIAL.match(span_text.strip()))
            or bool(match_terms(span_text, "en", "water_body")))


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def person_spans(text: str, lang: str | None, spacy_ents: list[dict], transformer_ents: list[dict],
                 spacy_is_language_specific: bool, genera: frozenset[str]) -> list[tuple[int, int]]:
    credits = credit_spans(text)
    cands = list(credits)
    tr = [(e["start"], e["end"]) for e in transformer_ents
          if e["label"] == "PERSON" and e.get("score", 1) >= TRANSFORMER_MIN]
    sp = ([(e["start"], e["end"]) for e in spacy_ents if e["label"] == "PERSON"]
          if spacy_is_language_specific else [])
    # Multi-word names from either model are masked. A lone word ("Gråand", "Joku", but also "Karin") is masked
    # only when both models agree with high confidence: a lone first name identifies little, while masking an
    # ordinary capitalised word destroys content.
    for s, e in tr + sp:
        if len(text[s:e].split()) >= 2:
            cands.append((s, e))
    for e in transformer_ents:
        span = (e["start"], e["end"])
        if (e["label"] == "PERSON" and len(e["text"].split()) == 1 and e.get("score", 1) >= TRANSFORMER_MIN_SINGLE
                and any(_overlaps(span, x) for x in sp)):
            cands.append(span)
    out = []
    for s, e in cands:
        if m := _TRAILING_TRIGGER.search(text, s, e):
            e = m.start()
        if e > s and not _protected(text[s:e], lang, genera, credit=(s, e) in credits):
            out.append((s, e))
    return _merge(out)


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def apply_mask(text: str, spans: list[tuple[int, int]]):
    """Return the masked text and a function mapping an old offset to the new one (None inside a masked span)."""
    if not spans:
        return text, lambda i: i
    parts, cuts, pos = [], [], 0
    for s, e in spans:
        parts += [text[pos:s], TOKEN]
        cuts.append((s, e))
        pos = e
    parts.append(text[pos:])
    masked = "".join(parts)

    def shift(i: int) -> int | None:
        delta = 0
        for s, e in cuts:
            if i >= e:
                delta += len(TOKEN) - (e - s)
            elif i > s:
                return None
        return i + delta

    return masked, shift


def remap(entities: list[dict], shift, masked_text: str) -> list[dict]:
    """Move entity offsets into the masked text; entities overlapping a masked name are dropped."""
    out = []
    for e in entities:
        s, en = shift(e["start"]), shift(e["end"])
        if s is not None and en is not None and masked_text[s:en] == e["text"]:
            out.append({**e, "start": s, "end": en})
    return out


# letters MT often writes without their diacritic but NFD does not decompose
_FOLD_EXTRA = {"ø": "o", "Ø": "O", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D"}


def fold(text: str) -> str:
    """Strip diacritics one character at a time ("Pål" -> "Pal"), keeping the length, so offsets found in the folded
    text are offsets in the original."""
    return "".join(_FOLD_EXTRA.get(c) or unicodedata.normalize("NFD", c)[0] for c in text)


# an initial MT may have rewritten in front of a surname ("P-E-. Lingdell" -> "P-E. Lingdell")
_LEFT_INITIAL = re.compile(rf"[{_UP}](?:[-.]+[{_UP}])*[-.]*\s+$")
_LEFT_WORD = re.compile(rf"({_WORD})\s+$")


def mask_translation(text_en: str, names: list[str], spacy_en_ents: list[dict], genera: frozenset[str]):
    """Mask the original's names in the translation, and nothing else: spaCy on machine-translated text labels
    garbled words as persons, so the translation is masked exactly where the original was (``spacy_en_ents`` is
    kept for the entity output, not used here). MT copies names but may rewrite initials, so besides the exact
    string every surname-like word of a masked name is matched and extended left over initials and over words
    that belong to the same name. Matching ignores diacritics, because MT may drop them ("Pål" -> "Pal")."""
    spans, folded = [], fold(text_en)
    for name in sorted(set(names), key=len, reverse=True):
        for m in re.finditer(rf"(?<!\w){re.escape(fold(name))}(?!\w)", folded):
            spans.append(m.span())
        name_words = set(re.findall(rf"[{_UP}][\w'’-]+", name))
        folded_words = {fold(w) for w in name_words}
        for word in name_words:
            if len(word) < 3 or _protected(word, "en", genera) or word.lower() in _NOT_NAMES:
                continue
            for m in re.finditer(rf"(?<!\w){re.escape(fold(word))}(?!\w)", folded):
                start = m.start()
                while True:
                    if left := _LEFT_INITIAL.search(folded, 0, start):
                        start = left.start()
                    elif (left := _LEFT_WORD.search(folded, 0, start)) and left.group(1) in folded_words:
                        start = left.start()
                    else:
                        break
                spans.append((start, m.end()))
    spans = _merge(spans)
    masked, shift = apply_mask(text_en, spans)
    return masked, len(spans), shift
