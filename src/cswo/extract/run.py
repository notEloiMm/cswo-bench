"""Spec 5 over the corpus: mask personal names, then extract entities and water-condition terms.

Pipeline position: harmonize.build -> translate merge -> extract.run (this) -> annotate.sample/tasks.
Reads the unmasked corpus (moved once to data/interim/cswo_all_unmasked.jsonl, gitignored), writes
* data/processed/cswo_all.jsonl  — masked text_original / text_en, persons_masked per record;
* data/processed/entities.jsonl  — per record: spaCy, transformer and rule outputs on the masked original,
  and spaCy + rules on the masked English text (for the English-original vs MT comparison, spec 7).
Transformer outputs are cached in data/interim/ner_transformer_cache.jsonl, so an interrupted run resumes.
Run: uv run python -m cswo.extract.run [--sample N]
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict

from tqdm import tqdm

from cswo import INTERIM, PROCESSED
from cswo.collect.common import read_jsonl, write_jsonl
from cswo.extract import mask, rules
from cswo.extract.ner import SPACY_FALLBACK, spacy_entities, spacy_model_for, transformer_entities
from cswo.harmonize.translate import OUT as CORPUS

UNMASKED = INTERIM / "cswo_all_unmasked.jsonl"
CACHE = INTERIM / "ner_transformer_cache.jsonl"
ENTITIES = PROCESSED / "entities.jsonl"
CHUNK = 2000


def _source_corpus(move: bool = True):
    """The unmasked corpus. A full run moves translate-merge's output aside (a fresh merge replaces it);
    a dry run reads it in place."""
    if CORPUS.exists() and "persons_masked" not in next(read_jsonl(CORPUS)):   # freshly merged, unmasked
        if not move:
            return CORPUS
        shutil.move(CORPUS, UNMASKED)
    return UNMASKED


def _transformer_cached(rows: list[dict]) -> dict[str, list[dict]]:
    cache = {r["id"]: r["ents"] for r in read_jsonl(CACHE)} if CACHE.exists() else {}
    todo = [r for r in rows if r["id"] not in cache]
    for i in tqdm(range(0, len(todo), CHUNK), desc="transformer NER"):
        part = todo[i:i + CHUNK]
        ents = transformer_entities([r["text_original"] for r in part], [r.get("language") for r in part],
                                    batch_size=64)
        write_jsonl(CACHE, ({"id": r["id"], "ents": e} for r, e in zip(part, ents)), append=True)
        cache.update({r["id"]: e for r, e in zip(part, ents)})
    return cache


def _spacy_by_lang(rows: list[dict], field: str, lang_of) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for r in rows:
        groups[lang_of(r)].append(r)
    out = {}
    for lang, part in tqdm(groups.items(), desc=f"spaCy on {field}"):
        for r, ents in zip(part, spacy_entities([r[field] for r in part], lang)):
            out[r["id"]] = ents
    return out


def run(sample: int | None = None) -> list[dict]:
    source = _source_corpus(move=not sample)
    records = list(read_jsonl(source))
    texts = [r for r in records if r.get("text_original")]
    if sample:
        texts = random.Random(0).sample(texts, sample)
    genera = rules.genera(str(source))
    tr = _transformer_cached(texts) if not sample else dict(zip(
        [r["id"] for r in texts], transformer_entities([r["text_original"] for r in texts],
                                                       [r.get("language") for r in texts], batch_size=64)))
    sp = _spacy_by_lang(texts, "text_original", lambda r: r.get("language"))
    en_rows = [r for r in texts if r.get("text_en") and r.get("language") != "en"]
    sp_en = _spacy_by_lang(en_rows, "text_en", lambda r: "en")

    stats, entities, report = Counter(), [], []
    for r in tqdm(texts, desc="mask + rules"):
        lang, text = r.get("language"), r["text_original"]
        specific = spacy_model_for(lang) != SPACY_FALLBACK
        persons = mask.person_spans(text, lang, sp[r["id"]], tr[r["id"]], specific, genera)
        masked, shift = mask.apply_mask(text, persons)
        names = [text[s:e] for s, e in persons]
        drop_person = lambda ents: [e for e in ents if e["label"] != "PERSON"]
        out = {"id": r["id"], "language": lang,
               "original": {"spacy": mask.remap(drop_person(sp[r["id"]]), shift, masked),
                            "transformer": mask.remap(drop_person(tr[r["id"]]), shift, masked),
                            "rules": rules.extract(masked, lang, genera)}}
        n_en = 0
        if r["id"] in sp_en:
            masked_en, n_en, shift_en = mask.mask_translation(r["text_en"], names, sp_en[r["id"]], genera)
            out["en"] = {"spacy": mask.remap(drop_person(sp_en[r["id"]]), shift_en, masked_en),
                         "rules": rules.extract(masked_en, "en", genera)}
            r["text_en"] = masked_en
        elif lang == "en":
            r["text_en"] = masked
        if persons:
            report.append({"id": r["id"], "lang": lang, "names": names, "masked": masked})
        r["text_original"] = masked
        r["persons_masked"] = len(persons) + n_en
        stats["records_with_names"] += bool(persons)
        stats["names_masked"] += len(persons)
        stats["names_masked_en"] += n_en
        entities.append(out)
    print(dict(stats))
    if sample:
        return report
    for r in records:
        r.setdefault("persons_masked", 0)
    write_jsonl(CORPUS, records)
    write_jsonl(ENTITIES, entities)
    print(f"-> {CORPUS}\n-> {ENTITIES}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, help="dry run on N random notes; prints masked names, writes nothing")
    args = ap.parse_args()
    report = run(args.sample)
    if args.sample:
        for x in report:
            print(json.dumps({"lang": x["lang"], "names": x["names"], "masked": x["masked"][:140]},
                             ensure_ascii=False))


if __name__ == "__main__":
    main()
