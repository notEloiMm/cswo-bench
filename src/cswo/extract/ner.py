"""NER systems for spec 5: spaCy (per-language small models) and a multilingual transformer.

Both are mapped to the gold entity scheme (LOCATION, DATE_TIME, QUANTITY) plus PERSON, which is used only for
masking. WATER_BODY, WEATHER and ORGANISM have no NER counterpart; a LOCATION that contains a water-body word
("Lake Geneva", "Oulujoki") is relabelled WATER_BODY.

The transformer (Davlan/xlm-roberta-base-ner-hrl, PER/ORG/LOC/DATE) was trained on ten high-resource languages,
not on Finnish, Swedish or Estonian: on those it runs zero-shot through XLM-R's cross-lingual transfer.
"""
from __future__ import annotations

from functools import lru_cache

from cswo.harmonize.water_filter import match_terms

SPACY_MODELS = {
    "en": "en_core_web_sm", "sv": "sv_core_news_sm", "fi": "fi_core_news_sm", "de": "de_core_news_sm",
    "da": "da_core_news_sm", "nl": "nl_core_news_sm", "fr": "fr_core_news_sm", "es": "es_core_news_sm",
    "pt": "pt_core_news_sm", "it": "it_core_news_sm", "lt": "lt_core_news_sm", "ru": "ru_core_news_sm",
    "pl": "pl_core_news_sm", "no": "nb_core_news_sm", "ro": "ro_core_news_sm", "hr": "hr_core_news_sm",
    "sl": "sl_core_news_sm", "ca": "ca_core_news_sm",
}
SPACY_FALLBACK = "xx_ent_wiki_sm"          # Estonian, Hungarian, ... and undetermined notes
TRANSFORMER = "Davlan/xlm-roberta-base-ner-hrl"

# Native labels of the spaCy models (OntoNotes, Swedish SUC, WikiNER) and the transformer -> gold scheme.
LABEL_MAP = {
    "PERSON": "PERSON", "PER": "PERSON", "PRS": "PERSON",
    "LOC": "LOCATION", "GPE": "LOCATION", "FAC": "LOCATION",
    "DATE": "DATE_TIME", "TIME": "DATE_TIME", "TME": "DATE_TIME",
    "QUANTITY": "QUANTITY", "CARDINAL": "QUANTITY", "PERCENT": "QUANTITY", "MSR": "QUANTITY",
}


def _to_gold(label: str, text: str, lang: str | None) -> str | None:
    gold = LABEL_MAP.get(label)
    if gold == "LOCATION" and match_terms(text, lang, "water_body"):
        return "WATER_BODY"
    return gold


@lru_cache(maxsize=None)
def _spacy(model: str):
    import spacy
    # only the components NER needs: faster, and the parser is irrelevant here
    nlp = spacy.load(model, exclude=["parser", "lemmatizer", "tagger", "morphologizer", "attribute_ruler",
                                     "senter", "trainable_lemmatizer"])
    return nlp


def spacy_model_for(lang: str | None) -> str:
    return SPACY_MODELS.get(lang or "", SPACY_FALLBACK)


def spacy_entities(texts: list[str], lang: str | None, batch_size: int = 256) -> list[list[dict]]:
    nlp = _spacy(spacy_model_for(lang))
    out = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        ents = []
        for e in doc.ents:
            if gold := _to_gold(e.label_, e.text, lang):
                ents.append({"start": e.start_char, "end": e.end_char, "text": e.text, "label": gold,
                             "raw_label": e.label_})
        out.append(ents)
    return out


@lru_cache(maxsize=1)
def _transformer():
    import torch
    from transformers import pipeline
    return pipeline("token-classification", model=TRANSFORMER, aggregation_strategy="first",   # word-level: no "Pe" + "eter Tamm" fragments
                    device=0 if torch.cuda.is_available() else -1)


def transformer_entities(texts: list[str], langs: list[str | None], batch_size: int = 32,
                         min_score: float = 0.5) -> list[list[dict]]:
    if not texts:
        return []
    pipe = _transformer()
    # length-sorted batches: short notes are not padded to the longest note in a random batch (~2-3x faster)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    preds_sorted = pipe([texts[i] for i in order], batch_size=batch_size)
    preds_all: list = [None] * len(texts)
    for i, p in zip(order, preds_sorted):
        preds_all[i] = p
    out = []
    for text, lang, preds in zip(texts, langs, preds_all):
        ents = []
        for p in preds:
            s, e = int(p["start"]), int(p["end"])
            # aggregation can leave a leading space inside the span; trim to the word
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1] in " .,;:!?)":
                e -= 1
            if p["score"] >= min_score and (gold := _to_gold(p["entity_group"], text[s:e], lang)):
                ents.append({"start": s, "end": e, "text": text[s:e], "label": gold,
                             "raw_label": p["entity_group"], "score": round(float(p["score"]), 3)})
        out.append(ents)
    return out
