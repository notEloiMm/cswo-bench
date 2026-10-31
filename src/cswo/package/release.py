"""Spec 12: build the release files of CSWO-Bench from the masked corpus and the fixed splits.

data/release/ (committed to the repository; described in DATACARD.md):
* cswo_bench_{train,validation,test}.jsonl.gz — every record with a note, its split from data/splits.csv;
* cswo_bench_media_only.jsonl.gz — records without a note (EyeOnWater, most of CrowdWater and iNaturalist): photo URL
  + structured labels, no split (no text task can use them);
* CrowdWater records go to separate files with `crowdwater_odbl` in the name: CrowdWater data is ODbL-1.0
  (share-alike), so its subset must stay separable and is released under ODbL with attribution; everything else is
  CC0-1.0, CC-BY-4.0 or the EyeOnWater open data policy, per record in `licence`;
* gold.jsonl — the adjudicated gold labels (spec 6–7), keyed by record_id;
* MANIFEST.json — counts per file, split, platform and licence, and a SHA-256 per file; a copy is committed as
  data/release_manifest.json.
Photos are never copied: `media[].url` points to the platform, `media[].licence` and `redistributable` say what the
photo's own licence allows. A record id seen twice is written once (4 iNaturalist observations near a border were
harvested for two countries; the copies differ only in `country`; MANIFEST.json lists them). Every record is re-validated against the schema and checked for leftover e-mail
addresses before it is written.
Run: uv run python -m cswo.package.release
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict

from cswo import DATA
from cswo.annotate.adjudicate import GOLD
from cswo.baselines.splits import load_splits
from cswo.collect.common import read_jsonl
from cswo.harmonize.schema import Record
from cswo.harmonize.translate import OUT as CORPUS

RELEASE = DATA / "release"
MANIFEST_COPY = DATA / "release_manifest.json"
VERSION = "0.1.0"
ODBL_PLATFORMS = {"crowdwater"}
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def file_for(r: dict, split: str | None) -> str:
    part = "crowdwater_odbl_" if r["source_platform"] in ODBL_PLATFORMS else ""
    return f"cswo_bench_{part}{split}.jsonl.gz" if split else f"cswo_bench_{part}media_only.jsonl.gz"


def check(r: dict) -> dict:
    """Schema validation plus privacy checks; returns the record as released."""
    rec = Record.model_validate(r).model_dump(mode="json")
    for field in ("text_original", "text_en"):
        if rec.get(field) and _EMAIL.search(rec[field]):
            raise ValueError(f"{rec['id']}: e-mail address left in {field}")
    for c in ("lat", "lon"):
        if rec.get(c) is not None and round(rec[c], 2) != rec[c]:
            raise ValueError(f"{rec['id']}: {c} not coarsened")
    return rec


def sha256(path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(records, splits: dict[str, str], out=RELEASE) -> dict:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    handles, counts, per_split, seen, duplicates = {}, defaultdict(Counter), Counter(), set(), []
    try:
        for r in records:
            if r["id"] in seen:
                duplicates.append(r["id"])
                continue
            seen.add(r["id"])
            has_text = bool((r.get("text_original") or "").strip())
            split = splits.get(r["id"]) if has_text else None
            if has_text and split is None:
                raise ValueError(f"{r['id']} has text but no split: rerun cswo.baselines.splits")
            rec = check(r)
            rec["split"] = split
            name = file_for(r, split)
            if name not in handles:
                handles[name] = gzip.open(out / name, "wt", encoding="utf-8")
            handles[name].write(json.dumps(rec, ensure_ascii=False) + "\n")
            c = counts[name]
            c["records"] += 1
            c[f"platform={r['source_platform']}"] += 1
            c[f"licence={rec['licence']}"] += 1
            c["with photo"] += bool(rec["media"])
            per_split[split or "media_only"] += 1
    finally:
        for h in handles.values():
            h.close()
    gold = out / "gold.jsonl"
    shutil.copyfile(GOLD, gold)
    files = {name: {**dict(sorted(c.items())), "sha256": sha256(out / name)} for name, c in sorted(counts.items())}
    files["gold.jsonl"] = {"records": sum(1 for _ in read_jsonl(gold)), "sha256": sha256(gold)}
    manifest = {"name": "CSWO-Bench", "version": VERSION, "records": sum(per_split.values()),
                "splits": dict(sorted(per_split.items())), "duplicates_dropped": duplicates, "files": files}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    manifest = build(read_jsonl(CORPUS), load_splits())
    MANIFEST_COPY.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    for name, f in manifest["files"].items():
        print(f"{name:48s} {f['records']:>7,}")
    print(f"total {manifest['records']:,} records, splits {manifest['splits']} -> {RELEASE}")


if __name__ == "__main__":
    main()
