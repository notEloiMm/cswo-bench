"""Spec 6: write one Label Studio task file per annotator from data/gold/gold_manifest.jsonl.

Label Studio Community has no task assignment, so each student gets their own project and imports only their
own file (20 calibration tasks first, then their 100 gold tasks). The LLM annotator does not use Label Studio.
Reserve calibration tasks (second round, guideline: if calibration kappa < 0.6) go to a separate file per student.
Run: uv run python -m cswo.annotate.tasks   -> data/gold/labelstudio/tasks_<A1..>.json, calibration2_<A1..>.json
"""
from __future__ import annotations

import json
from base64 import b64encode
from collections import defaultdict

from cswo.annotate.sample import ANNOTATORS, GOLD_DIR, GUIDELINE_VERSION
from cswo.collect.common import read_jsonl
from cswo.harmonize.translate import OUT as CORPUS

OUT_DIR = GOLD_DIR / "labelstudio"
_NO_PHOTO_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" width="320" height="60"><rect width="320" height="60" '
                 'fill="#f3f2ef"/><text x="160" y="36" font-family="sans-serif" font-size="15" fill="#52514e" '
                 'text-anchor="middle">No photo for this record</text></svg>')
NO_PHOTO = "data:image/svg+xml;base64," + b64encode(_NO_PHOTO_SVG.encode()).decode()
SET_LABEL = {"calibration": "Calibration round (all students label these 20 first)",
             "calibration_reserve": "Calibration round 2 (only if the group decides to run it)",
             "gold": "Gold set"}


def task(r: dict, entry: dict) -> dict:
    if r.get("language") == "en":
        text_en = "(original is English)"
    elif r.get("text_en"):
        text_en = r["text_en"]
    else:
        text_en = f"(no machine translation: {r.get('translation_qc') or r.get('translation_model') or 'n/a'})"
    photo = (r.get("media") or [{}])[0].get("url")
    return {"data": {
        "record_id": r["id"], "set": entry["set"], "set_label": SET_LABEL[entry["set"]],
        "guideline_version": GUIDELINE_VERSION,
        "meta": f"Guideline {GUIDELINE_VERSION} · Platform: {r['source_platform']} · Language (auto-detected): "
                f"{r.get('language') or '-'} · Record: {r['id']} · Source: {r['provenance_url']}",
        "platform": r["source_platform"], "language": r.get("language"),
        "text_original": r["text_original"], "text_en": text_en,
        "image": photo or NO_PHOTO, "has_photo": bool(photo),
    }}


def main() -> None:
    manifest = list(read_jsonl(GOLD_DIR / "gold_manifest.jsonl"))
    wanted = {m["id"] for m in manifest}
    records = {r["id"]: r for r in read_jsonl(CORPUS) if r["id"] in wanted}
    files = defaultdict(list)
    for m in sorted(manifest, key=lambda m: m["set"] != "calibration"):     # calibration first
        for a in m["annotators"]:
            if a in ANNOTATORS:                                                # students only
                kind = "calibration2" if m["set"] == "calibration_reserve" else "tasks"
                files[(kind, a)].append(task(records[m["id"]], m))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for (kind, a), tasks in sorted(files.items()):
        path = OUT_DIR / f"{kind}_{a}.json"
        path.write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")
        n_cal = sum(t["data"]["set"] == "calibration" for t in tasks)
        print(f"{a} {kind}: {len(tasks)} tasks ({n_cal} calibration) -> {path.name}")


if __name__ == "__main__":
    main()
