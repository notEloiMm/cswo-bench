"""Spec 7: build the adjudicated gold labels from the annotations and the resolved disagreement sheets.

Per record and field: the value all eligible annotators agree on; otherwise the "resolved" cell of
data/gold/disagreements_<round>.csv, which is either an annotator id (take that annotator's value, e.g. "A2")
or a literal in the sheet's own format ("algae; turbid", "none stated", "WATER_BODY[12:20] Borssjön; (none)").
Unresolved fields stay null and are listed in "unresolved", so the evaluation can skip them. A resolved cell
"unresolvable" (guideline: the adjudicator may mark a record unresolvable) also leaves the field null and lists it in
"unresolvable"; both annotators' labels stay in annotations.jsonl, and such records are excluded from the test split.
Entities come only from annotators who read the original; a record nobody read in the original has entities null.
Output: data/gold/gold.jsonl
Run: uv run python -m cswo.annotate.adjudicate
"""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from cswo.annotate.agreement import FIELDS, ROUNDS, _eligible, fmt
from cswo.annotate.merge import ANNOTATIONS, task_data
from cswo.annotate.sample import GOLD_DIR, LLM_ANNOTATOR
from cswo.collect.common import read_jsonl, write_jsonl

GOLD = GOLD_DIR / "gold.jsonl"
_SPAN = re.compile(r"^(\w+)\[(\d+):(\d+)\]\s?(.*)$")


def parse(field: str, value: str, text: str):
    value = value.strip()
    if field == "conditions":
        conds = sorted(v.strip() for v in value.split(";") if v.strip() and v.strip() != "none stated")
        return conds or ["none stated"]           # same explicit form as the merged annotations
    if field == "entities":
        if value == "(none)":
            return []
        spans = []
        for part in (p.strip() for p in value.split(";") if p.strip()):
            m = _SPAN.match(part)
            if not m:
                raise ValueError(f"cannot parse entity '{part}' (expected LABEL[start:end] text)")
            s, e = int(m[2]), int(m[3])
            spans.append({"start": s, "end": e, "text": text[s:e], "label": m[1]})
        return spans
    return value or None


def resolutions(gold_dir: Path = GOLD_DIR) -> dict[tuple[str, str], str]:
    out = {}
    for name in ROUNDS:
        path = gold_dir / f"disagreements_{name}.csv"
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as f:
                out.update({(r["record_id"], r["field"]): r["resolved"] for r in csv.DictReader(f)
                            if r.get("resolved", "").strip()})
    return out


def adjudicate(rows: list[dict], resolved: dict[tuple[str, str], str], data: dict[str, dict]) -> list[dict]:
    by_record = defaultdict(dict)
    for r in rows:
        # calibration gold comes from the students (their round and sheet); the LLM is a gold-set annotator only
        if r["set"].startswith("calibration") and r["annotator_id"] == LLM_ANNOTATOR:
            continue
        by_record[r["record_id"]][r["annotator_id"]] = r
    out = []
    for rid, rec in sorted(by_record.items()):
        any_row = next(iter(rec.values()))
        text = data.get(rid, {}).get("text_original") or ""
        g = {"record_id": rid, "set": any_row["set"], "platform": any_row["platform"],
             "language": any_row["language"], "has_photo": any_row["has_photo"], "annotators": sorted(rec),
             "read_from": {a: r["read_from"] for a, r in rec.items()}}
        unresolved, unresolvable = [], []
        for field in FIELDS:
            elig = _eligible(field, rec)
            values = {fmt(field, r): r[field] for r in elig.values()}
            if field == "entities" and not elig:
                g[field] = None                    # nobody read it in the original (or not about water)
            elif len(values) == 1:
                g[field] = next(iter(values.values()))
            elif not values:
                g[field] = [] if field == "conditions" else None
            elif (rid, field) in resolved and resolved[(rid, field)].strip().lower() == "unresolvable":
                g[field] = None
                unresolvable.append(field)
            elif (rid, field) in resolved:
                choice = resolved[(rid, field)].strip()
                g[field] = rec[choice][field] if choice in rec else parse(field, choice, text)
            else:
                g[field] = None
                unresolved.append(field)
        if g["relevance"] == "not about water":
            g["conditions"], g["entities"] = [], None
        g["unresolved"] = unresolved
        g["unresolvable"] = unresolvable
        out.append(g)
    return out


def main() -> None:
    gold = adjudicate(list(read_jsonl(ANNOTATIONS)), resolutions(), task_data())
    write_jsonl(GOLD, gold)
    left = Counter(f for g in gold for f in g["unresolved"])
    print(f"{len(gold)} gold records ({dict(Counter(g['set'] for g in gold))}); "
          f"with entity gold: {sum(g['entities'] is not None for g in gold)}")
    print("unresolved fields:", dict(left) or "none")
    print("unresolvable fields:", dict(Counter(f for g in gold for f in g["unresolvable"])) or "none")
    print(f"-> {GOLD}")


if __name__ == "__main__":
    main()
