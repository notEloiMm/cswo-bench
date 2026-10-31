"""Export note + photo pairs for the spec 9 image-text notebook (notebooks/colab_clip.ipynb).

Two sets:
* gold: every gold record with a photo (scored against the adjudicated image_text labels in clip_eval)
* sample: SAMPLE_PER_PLATFORM other records with a note of >= MIN_WORDS words and a photo, per platform that has
  both (iNaturalist, CrowdWater), chosen by a hash of the record id, for the per-platform distributions
Each record gets a `partner` in the same set and platform (the next record in hash order), whose photo gives the
mismatched pair. Only the record id, platform, language, the masked note, its English translation and the photo
URL leave the machine; gold labels stay local.
Run: uv run python -m cswo.multimodal.clip_input   -> data/interim/clip_input.jsonl
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

from cswo import INTERIM
from cswo.annotate.adjudicate import GOLD
from cswo.annotate.sample import GOLD_DIR
from cswo.collect.common import read_jsonl, write_jsonl
from cswo.harmonize.translate import OUT as CORPUS

OUT = INTERIM / "clip_input.jsonl"
SAMPLE_PLATFORMS = ("inaturalist", "crowdwater")
SAMPLE_PER_PLATFORM = 300
MIN_WORDS = 3


def _h(key: str) -> str:
    return hashlib.sha1(f"cswo-clip|{key}".encode()).hexdigest()


def text_en(r: dict) -> str | None:
    """English text for CLIP: the translation, or the note itself when it is English or unidentified (und notes
    are mostly species names, codes or short English the language ID could not place)."""
    if r.get("text_en"):
        return r["text_en"]
    return r.get("text_original") if r.get("language") in ("en", "und") else None


def row(r: dict, kind: str) -> dict:
    return {"record_id": r["id"], "set": kind, "platform": r["source_platform"], "language": r.get("language"),
            "text_original": r.get("text_original"), "text_en": text_en(r), "image": r["media"][0]["url"]}


def add_partners(rows: list[dict]) -> list[dict]:
    """Pair every record with the next one (hash order) of the same set and platform: a cyclic shift, so no record
    is its own partner and every photo is used once as a mismatch."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["set"], r["platform"])].append(r)
    for g in groups.values():
        g.sort(key=lambda r: _h(r["record_id"]))
        for i, r in enumerate(g):
            r["partner"] = g[(i + 1) % len(g)]["record_id"] if len(g) > 1 else None
    return rows


def build(records, gold_ids: set[str], exclude: set[str]) -> list[dict]:
    gold, pool = [], defaultdict(list)
    for r in records:
        if not r.get("media") or not (r.get("text_original") or "").strip():
            continue
        if r["id"] in gold_ids:
            gold.append(row(r, "gold"))
        elif (r["id"] not in exclude and r["source_platform"] in SAMPLE_PLATFORMS
              and len(r["text_original"].split()) >= MIN_WORDS):
            pool[r["source_platform"]].append(r)
    sample = [row(r, "sample") for p in SAMPLE_PLATFORMS
              for r in sorted(pool[p], key=lambda r: _h(r["id"]))[:SAMPLE_PER_PLATFORM]]
    return add_partners(gold + sample)


def main() -> None:
    gold_ids = {g["record_id"] for g in read_jsonl(GOLD) if g["set"] == "gold" and g["has_photo"]}
    manifest = {m["id"] for m in read_jsonl(GOLD_DIR / "gold_manifest.jsonl")}
    rows = build(read_jsonl(CORPUS), gold_ids, manifest)
    n = write_jsonl(OUT, rows)
    print(f"{n} records ({sum(r['set'] == 'gold' for r in rows)} gold with photo) -> {OUT}")


if __name__ == "__main__":
    main()
