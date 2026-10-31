"""Shared tokenization, corpus loading and lexical-resource lookups for specs 3-4."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterator

from cswo import INTERIM, PROCESSED
from cswo.collect.common import read_jsonl

# A word is a run of letters, optionally joined by an apostrophe or hyphen ("blue-green", "don't").
# Digits are counted separately: they are a large part of field notes ("2 ex", "0.65m") and a noise indicator.
_WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*")
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_MASK = re.compile(r"\[(?:EMAIL|COORD|USER_URL|PERSON)\]")


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(_MASK.sub(" ", text))]


def count_numbers(text: str) -> int:
    return len(_NUM.findall(text))


def corpus_path():
    """The merged corpus once translations exist, else the harmonized one (same original text)."""
    p = PROCESSED / "cswo_all.jsonl"
    return p if p.exists() else INTERIM / "cswo_harmonized.jsonl"


def notes() -> Iterator[dict]:
    """Records with original text: {platform, language, text}."""
    for r in read_jsonl(corpus_path()):
        if r.get("text_original"):
            yield {"platform": r["source_platform"], "language": r.get("language") or "-",
                   "text": r["text_original"]}


def reference_corpora() -> dict[str, list[str]]:
    """Brown and Reuters as sentence strings (the sentence is the unit closest to a short field note)."""
    import nltk
    for pkg in ("brown", "reuters", "punkt_tab"):
        nltk.download(pkg, quiet=True)
    from nltk.corpus import brown, reuters
    return {"Brown": [" ".join(s) for s in brown.sents()], "Reuters": [" ".join(s) for s in reuters.sents()]}


# --------------------------------------------------------------------------------- lexical coverage (spec 3)

@lru_cache
def _wordnet():
    import nltk
    for pkg in ("wordnet", "omw-1.4", "omw-2.0"):
        nltk.download(pkg, quiet=True)
    from nltk.corpus import wordnet
    return wordnet


@lru_cache
def _finnwordnet_lemmas() -> frozenset[str]:
    return frozenset(w.lower().replace("_", " ") for w in _wordnet().all_lemma_names(lang="fin"))


@lru_cache(maxsize=500_000)
def in_wordnet_en(token: str) -> bool:
    """English: WordNet entry after morphological reduction (dogs -> dog). Function words have no entry,
    so even edited text stays well below 100%."""
    return _wordnet().morphy(token) is not None


@lru_cache(maxsize=500_000)
def in_finnwordnet(token: str) -> bool:
    """Finnish: FinnWordNet (via OMW) entry for the token or its simplemma lemma (järvellä -> järvi).
    Stands in for OMorFi, which needs HFST and does not install on Windows."""
    import simplemma
    lemmas = _finnwordnet_lemmas()
    return token in lemmas or simplemma.lemmatize(token, lang="fi") in lemmas


COVERAGE = {"en": ("WordNet", in_wordnet_en), "fi": ("FinnWordNet", in_finnwordnet)}
