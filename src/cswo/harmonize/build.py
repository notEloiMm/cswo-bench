"""Harmonize every raw source into data/interim/cswo_harmonized.jsonl and write the spec 2 summary table.

Steps: map -> water filter -> cap large platforms -> drop templated notes -> language ID -> fill country -> validate.
Translation (text_en) is added afterwards on a GPU (notebooks/colab_translate.ipynb); English notes are copied here.
Run: uv run python -m cswo.harmonize.build [--platforms eyeonwater inaturalist gbif crowdwater]
"""
from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter

import pandas as pd
import reverse_geocoder as rg
from tqdm import tqdm

from cswo import INTERIM, RESULTS
from cswo.collect.common import write_jsonl
from cswo.harmonize import mappers
from cswo.harmonize.langid import COUNTRY_LANG, detect, detect_all
from cswo.harmonize.schema import Record
from cswo.harmonize.water_filter import water_body

SOURCES = {"eyeonwater": mappers.eyeonwater, "inaturalist": mappers.inaturalist,
           "gbif": mappers.gbif, "crowdwater": mappers.crowdwater}
TEMPLATE_MIN_COUNT = 50     # the same note >= 50 times on one platform is a form template, not an observation
TEMPLATE_MIN_CHARS = 30     # ...unless it is short ("1 ex.", "pair"), which is genuine terse field language,
                            # or carries a masked record number ("Tiirahavainto id = #######"), which is an import stub
# Platforms that would swamp the corpus get a fixed-size sample (Artportalen: ~1.6M usable notes -> 30k, group
# decision 2026-10-25). Only notes with >= CAP_MIN_TOKENS words are eligible ("ca", "minst" carry no content).
CAPS = {"artportalen": 30_000}
CAP_MIN_TOKENS = 3
CAPPED_SOURCES = {"gbif"}   # sources that can yield a capped platform


def _template_key(t: str) -> str:
    return re.sub(r"\d", "#", t.lower())


def _is_template(key: str, count: int) -> bool:
    return count >= TEMPLATE_MIN_COUNT and (len(key) >= TEMPLATE_MIN_CHARS or "###" in key)


def _fill_countries(rows: list[dict]) -> None:
    todo = [r for r in rows if not r.get("country") and r.get("lat") is not None]
    if todo:
        for r, hit in zip(todo, rg.search([(r["lat"], r["lon"]) for r in todo], mode=1, verbose=False)):
            r["country"] = hit["cc"]


def _hash_unit(record_id: str) -> float:
    """Deterministic pseudo-random number in [0, 1) per record: the sample is the same on every rerun."""
    return int(hashlib.sha1(f"cswo-sample|{record_id}".encode()).hexdigest()[:12], 16) / 16 ** 12


def _filtered(name: str, dropped: Counter):
    """Map one source and apply the aquatic-taxon / water-body filter and the cap eligibility rule."""
    for r in SOURCES[name]():
        if "_aquatic" in r:
            if not r["_aquatic"]:
                dropped["not_aquatic_taxon"] += 1
                continue
            # the portal country's language (+ English, always included) is enough to find water words
            lang = COUNTRY_LANG.get(r.get("_country_lang") or "")
            keep, wbt, ev = water_body({"locality": r.get("_locality"), "habitat": r.get("_habitat"),
                                        "text": r.get("text_original")}, lang, r["_obligate"])
            if not keep:
                dropped["no_water_evidence"] += 1
                continue
            r["water_body_type"], r["water_evidence"] = wbt, ev
        if r["source_platform"] in CAPS and len((r.get("text_original") or "").split()) < CAP_MIN_TOKENS:
            dropped["capped_platform_short_note"] += 1
            continue
        yield r


def build(platforms: list[str]) -> list[Record]:
    rows, dropped = [], Counter()
    for name in platforms:
        # Pass 1 only counts what a capped platform could contribute, so 4M rows never sit in memory.
        eligible = Counter() if name not in CAPPED_SOURCES else Counter(
            r["source_platform"] for r in tqdm(_filtered(name, Counter()), desc=f"count {name}")
            if r["source_platform"] in CAPS)
        keep_p = {p: min(1.0, CAPS[p] / n) for p, n in eligible.items()}
        for r in tqdm(_filtered(name, dropped), desc=f"map {name}"):
            p = r["source_platform"]
            if p in keep_p and _hash_unit(r["id"]) >= keep_p[p]:
                dropped[f"{p}_not_sampled"] += 1
                continue
            rows.append(r)

    freq = Counter((r["source_platform"], _template_key(r["text_original"])) for r in rows if r.get("text_original"))
    for r in rows:
        t = r.get("text_original")
        if t and _is_template(k := _template_key(t), freq[(r["source_platform"], k)]):
            r["text_original"] = None
            dropped["template_text_removed"] += 1

    n = len(rows)
    rows = [r for r in rows if r.get("text_original") or r.get("media")]
    dropped["no_text_no_media"] = n - len(rows)
    _fill_countries(rows)
    records = []
    for r in tqdm(rows, desc="langid+validate"):
        lang, conf = detect(r.get("text_original"), COUNTRY_LANG.get(r.get("country") or ""))
        r["language"], r["language_confidence"] = lang, conf
        r["languages_all"] = detect_all(r.get("text_original"))
        if lang == "en":
            r["text_en"] = r["text_original"]
        records.append(Record(**{k: v for k, v in r.items() if not k.startswith("_")}))
    print("Dropped / changed:", dict(dropped))
    return records


def summary_table(records: list[Record]) -> pd.DataFrame:
    """Records per platform x language, and share with text only / image only / both (spec 2)."""
    df = pd.DataFrame({
        "platform": [r.source_platform for r in records],
        "language": [r.language or "-" for r in records],
        "text": [r.has_text for r in records], "image": [r.has_image for r in records],
    })
    df["modality"] = df.apply(lambda x: "both" if x.text and x.image else "text_only" if x.text
                              else "image_only" if x.image else "neither", axis=1)
    by_lang = df.pivot_table(index="platform", columns="language", aggfunc="size", fill_value=0)
    share = (df.groupby("platform")["modality"].value_counts(normalize=True).unstack(fill_value=0) * 100).round(1)
    share.columns = [f"% {c}" for c in share.columns]
    out = by_lang.join(share)
    out.insert(0, "total", df.groupby("platform").size())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platforms", nargs="*", default=list(SOURCES))
    args = ap.parse_args()
    records = build(args.platforms)
    out = INTERIM / "cswo_harmonized.jsonl"
    n = write_jsonl(out, (r.model_dump(exclude_none=True) for r in records))
    table = summary_table(records)
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "tables" / "spec2_platform_language.csv")
    (RESULTS / "tables" / "spec2_platform_language.md").write_text(table.to_markdown(), encoding="utf-8")
    print(f"{n} records -> {out}\n{table.to_string()}")


if __name__ == "__main__":
    main()
