"""Spec 7: convert the Label Studio exports into the guideline's annotation JSONL.

Input, in data/gold/exports/ (gitignored):
* <annotator id>.json — Label Studio "JSON" export of one student's project. The file name is the anonymous id
  (A1.json), so the Label Studio e-mail inside the export is never read or copied;
* <annotator id>.jsonl — labels already in the output format (the LLM annotator, which does not use Label Studio).
Output: data/gold/annotations.jsonl, one row per (record, annotator): record_id, set, annotator_id, read_from,
relevance, conditions, entities, image_text, uncertain, comment, guideline (+ platform, language, has_photo).

Guideline rules applied here so every later step can rely on them: a note "not about water" has no conditions and
no entities; entity spans exist only when read_from = original; image_text only when a photo is attached.
Run: uv run python -m cswo.annotate.merge
"""
from __future__ import annotations

import json
from pathlib import Path

from cswo.annotate.sample import GOLD_DIR, GUIDELINE_VERSION
from cswo.collect.common import read_jsonl, write_jsonl

EXPORTS = GOLD_DIR / "exports"
ANNOTATIONS = GOLD_DIR / "annotations.jsonl"
TASKS_DIR = GOLD_DIR / "labelstudio"
SINGLE = ("read_from", "relevance", "image_text")


def task_data(tasks_dir: Path = TASKS_DIR) -> dict[str, dict]:
    """record_id -> the task data shown to annotators (texts, platform, language, has_photo)."""
    out = {}
    for path in sorted(tasks_dir.glob("*.json")):
        for t in json.loads(path.read_text(encoding="utf-8")):
            out[t["data"]["record_id"]] = t["data"]
    return out


def parse_result(result: list[dict]) -> dict:
    """One Label Studio annotation (its "result" list) -> guideline fields."""
    out = {"read_from": None, "relevance": None, "conditions": [], "entities": [], "image_text": None,
           "uncertain": False, "comment": ""}
    for item in result:
        name, v = item.get("from_name"), item.get("value") or {}
        if item.get("type") == "labels":
            out["entities"].append({"start": v["start"], "end": v["end"], "text": v.get("text", ""),
                                    "label": v["labels"][0]})
        elif name in SINGLE and v.get("choices"):
            out[name] = v["choices"][0]
        elif name == "conditions":
            out["conditions"] = sorted(v.get("choices", []))
        elif name == "uncertain":
            out["uncertain"] = bool(v.get("choices"))
        elif name == "comment":
            out["comment"] = " ".join(v.get("text", [])).strip()
    return out


def normalise(row: dict, data: dict) -> dict:
    """Apply the guideline's conditional rules (hard cases 6 and 7) and check spans against the shown text.
    * about water / uncertain with no condition ticked -> ["none stated"], written explicitly, so an empty set and a
      skipped question can be told apart (guideline, "none stated");
    * not about water -> no conditions, entities or image-text (guideline: set relevance and stop);
    * entities are null for translation-read records (guideline export format: spans point into the original)."""
    conds = sorted(c for c in row["conditions"] if c != "none stated")
    row["conditions"] = conds or (["none stated"] if row["relevance"] in ("about water", "uncertain") else [])
    if row["relevance"] == "not about water":
        row["conditions"], row["entities"], row["image_text"] = [], None, None
    if row["read_from"] != "original":
        row["entities"] = None
    if not data.get("has_photo"):
        row["image_text"] = None
    text = data.get("text_original") or ""
    for e in row["entities"] or []:
        if text[e["start"]:e["end"]] != e["text"]:
            raise ValueError(f"{row['record_id']} ({row['annotator_id']}): span {e} does not match the note")
    if row["entities"] is not None:
        row["entities"] = sorted(row["entities"], key=lambda e: (e["start"], e["end"], e["label"]))
    return row


def _latest(task: dict) -> dict | None:
    done = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    return max(done, key=lambda a: (a.get("updated_at") or "", a.get("id") or 0)) if done else None


def from_labelstudio(path: Path, data: dict[str, dict]) -> list[dict]:
    annotator, rows = path.stem, []
    for task in json.loads(path.read_text(encoding="utf-8")):
        ann = _latest(task)
        if ann is None:
            continue
        rid = task["data"]["record_id"]
        rows.append({"record_id": rid, "annotator_id": annotator, **parse_result(ann["result"]),
                     "lead_time": round(ann["lead_time"], 1) if ann.get("lead_time") else None})
    return rows


def _with_meta(row: dict, data: dict[str, dict]) -> dict:
    d = data[row["record_id"]]
    row = normalise({**row, "set": d["set"]}, d)
    row.setdefault("lead_time", None)
    return {"record_id": row["record_id"], "set": d["set"], "annotator_id": row["annotator_id"],
            "platform": d["platform"], "language": d["language"], "has_photo": d["has_photo"],
            **{k: row[k] for k in ("read_from", "relevance", "conditions", "entities", "image_text",
                                   "uncertain", "comment", "lead_time")},
            "guideline": row.get("guideline", GUIDELINE_VERSION)}


def merge(exports: Path = EXPORTS, tasks_dir: Path = TASKS_DIR) -> list[dict]:
    data = task_data(tasks_dir)
    rows = []
    for path in sorted(exports.glob("*.json")):
        rows += from_labelstudio(path, data)
    for path in sorted(exports.glob("*.jsonl")):
        rows += [{**r, "annotator_id": r.get("annotator_id") or path.stem} for r in read_jsonl(path)]
    rows = [_with_meta(r, data) for r in rows]
    seen = set()
    for r in rows:
        key = (r["record_id"], r["annotator_id"])
        if key in seen:
            raise ValueError(f"record {key[0]} labelled twice by {key[1]}")
        seen.add(key)
    return sorted(rows, key=lambda r: (r["set"], r["record_id"], r["annotator_id"]))


def main() -> None:
    if not EXPORTS.exists() or not any(EXPORTS.iterdir()):
        raise SystemExit(f"no exports in {EXPORTS}: save each Label Studio export as <your ID>.json there")
    rows = merge()
    write_jsonl(ANNOTATIONS, rows)
    by = {}
    for r in rows:
        by.setdefault(r["annotator_id"], {}).setdefault(r["set"], 0)
        by[r["annotator_id"]][r["set"]] += 1
    print("annotations per annotator and set:", by)
    print(f"-> {ANNOTATIONS}")


if __name__ == "__main__":
    main()
