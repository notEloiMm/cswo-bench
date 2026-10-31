"""Spec 7: inter-annotator agreement and the disagreement sheet for the resolution meeting.

Pairs compared: in the calibration rounds every pair of students (3 students -> 3 pairs); in the gold set the
student and the LLM that labelled the same record (guideline v0.1).
* Cohen's kappa for the record-level labels: relevance (3 classes); water condition, as one kappa over the
  flattened record x condition binary matrix (pooled, the figure the guideline's 0.6 threshold applies to) and per
  condition; image-text consistency on records with a photo. Conditions are compared on records that neither
  annotator marked "not about water"; "none stated" is the all-zero row.
* Span F1 between annotators on records both read in the original: exact (same start, end, label) and partial
  (same label, overlapping). F1 = 2 * matches / (spans of A + spans of B), counts pooled over records.
Every record where annotators differ goes to data/gold/disagreements_<set>.csv with an empty "resolved" column,
filled in during the discussion (see cswo.annotate.adjudicate). Re-running keeps resolutions already written.
Run: uv run python -m cswo.annotate.agreement   (after cswo.annotate.merge)
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

from cswo import RESULTS
from cswo.annotate.merge import ANNOTATIONS, task_data
from cswo.annotate.sample import GOLD_DIR, LLM_ANNOTATOR
from cswo.collect.common import read_jsonl

CONDITIONS = ["clear", "turbid", "coloured", "algae", "foam", "ice", "oil", "litter", "odour", "high water",
              "low or dry", "flooding"]
ENTITY_LABELS = ["WATER_BODY", "LOCATION", "ORGANISM", "QUANTITY", "DATE_TIME", "WEATHER"]
FIELDS = ["relevance", "conditions", "image_text", "entities"]
KAPPA_THRESHOLD = 0.6               # guideline: below this on conditions, run calibration round 2


def cohen_kappa(a: list, b: list) -> float:
    """Cohen's kappa of two equally long label sequences; NaN when undefined (both constant and identical)."""
    n = len(a)
    if n == 0 or n != len(b):
        return math.nan
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    return math.nan if pe == 1 else (po - pe) / (1 - pe)


def _overlap(x: dict, y: dict) -> bool:
    return x["label"] == y["label"] and x["start"] < y["end"] and y["start"] < x["end"]


def span_matches(a: list[dict], b: list[dict], partial: bool = False, label: str | None = None) -> tuple[int, int, int]:
    """(matches, |a|, |b|) with one-to-one matching; partial = same label and overlapping."""
    if label:
        a, b = [x for x in a if x["label"] == label], [y for y in b if y["label"] == label]
    key = lambda e: (e["start"], e["end"], e["label"])
    if not partial:
        return len({key(x) for x in a} & {key(y) for y in b}), len(a), len(b)
    used, m = set(), 0
    for x in a:
        for j, y in enumerate(b):
            if j not in used and _overlap(x, y):
                used.add(j)
                m += 1
                break
    return m, len(a), len(b)


def f1(m: int, na: int, nb: int) -> float:
    return 2 * m / (na + nb) if na + nb else math.nan


def _cond_vector(row: dict) -> list[int]:
    return [int(c in row["conditions"]) for c in CONDITIONS]


def _has_water(row: dict) -> bool:
    return row["relevance"] != "not about water"


def pair_items(rows: list[dict], sets: tuple[str, ...]) -> dict[tuple[str, str], list[tuple[dict, dict]]]:
    """(annotator a, annotator b) -> list of (row a, row b) on the records both labelled. In the gold set the
    student is paired with the LLM; all students are pooled as "student" vs "LLM"."""
    by_record = defaultdict(dict)
    for r in rows:
        if r["set"] in sets:
            by_record[r["record_id"]][r["annotator_id"]] = r
    pairs = defaultdict(list)
    for rec in by_record.values():
        for a, b in combinations(sorted(rec, key=lambda x: (x == LLM_ANNOTATOR, x)), 2):
            pairs[(a, b)].append((rec[a], rec[b]))
    return dict(pairs)


def pair_scores(items: list[tuple[dict, dict]]) -> dict:
    rel = [(x["relevance"], y["relevance"]) for x, y in items]
    cond = [(x, y) for x, y in items if _has_water(x) and _has_water(y)]
    flat_a = [v for x, _ in cond for v in _cond_vector(x)]
    flat_b = [v for _, y in cond for v in _cond_vector(y)]
    img = [(x["image_text"], y["image_text"]) for x, y in items if x["image_text"] and y["image_text"]]
    orig = [(x, y) for x, y in items if x["read_from"] == y["read_from"] == "original" and _has_water(x)
            and _has_water(y)]
    exact = [sum(t) for t in zip(*[span_matches(x["entities"], y["entities"]) for x, y in orig])] or [0, 0, 0]
    part = [sum(t) for t in zip(*[span_matches(x["entities"], y["entities"], True) for x, y in orig])] or [0, 0, 0]
    return {
        "records": len(items),
        "relevance_kappa": cohen_kappa(*map(list, zip(*rel))) if rel else math.nan,
        "relevance_agree": sum(a == b for a, b in rel) / len(rel) if rel else math.nan,
        "conditions_records": len(cond),
        "conditions_kappa": cohen_kappa(flat_a, flat_b),
        "conditions_exact_agree": sum(x["conditions"] == y["conditions"] for x, y in cond) / len(cond)
        if cond else math.nan,
        "image_text_records": len(img),
        "image_text_kappa": cohen_kappa(*map(list, zip(*img))) if img else math.nan,
        "span_records": len(orig),
        "span_f1_exact": f1(*exact),
        "span_f1_partial": f1(*part),
    }


def per_label(pairs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-condition kappa and per-entity-type span F1, pooled over all pairs."""
    items = [it for v in pairs.values() for it in v]
    cond = [(x, y) for x, y in items if _has_water(x) and _has_water(y)]
    crow = []
    for i, c in enumerate(CONDITIONS):
        a = [_cond_vector(x)[i] for x, _ in cond]
        b = [_cond_vector(y)[i] for _, y in cond]
        crow.append({"condition": c, "positives_a": sum(a), "positives_b": sum(b), "kappa": cohen_kappa(a, b)})
    orig = [(x, y) for x, y in cond if x["read_from"] == y["read_from"] == "original"]
    erow = []
    for lab in ENTITY_LABELS:
        ex = [sum(t) for t in zip(*[span_matches(x["entities"], y["entities"], label=lab) for x, y in orig])] or [0] * 3
        pa = [sum(t) for t in zip(*[span_matches(x["entities"], y["entities"], True, lab) for x, y in orig])] or [0] * 3
        erow.append({"label": lab, "spans_a": ex[1], "spans_b": ex[2], "f1_exact": f1(*ex), "f1_partial": f1(*pa)})
    return pd.DataFrame(crow).set_index("condition"), pd.DataFrame(erow).set_index("label")


# ---- disagreement sheet -------------------------------------------------------------------------------------

def fmt(field: str, row: dict | None) -> str:
    if row is None:
        return ""
    v = row[field]
    if field == "conditions":
        return "; ".join(v) if v else "none stated"
    if field == "entities":
        if row["read_from"] != "original":
            return "(read translation)"
        return "; ".join(f"{e['label']}[{e['start']}:{e['end']}] {e['text']}" for e in v) if v else "(none)"
    return v or ""


def _eligible(field: str, rows: dict[str, dict]) -> dict[str, dict]:
    if field == "entities":
        return {a: r for a, r in rows.items() if r["read_from"] == "original" and _has_water(r)}
    if field == "conditions":
        return {a: r for a, r in rows.items() if _has_water(r)}
    if field == "image_text":
        return {a: r for a, r in rows.items() if r["image_text"]}
    return rows


def disagreements(rows: list[dict], sets: tuple[str, ...], data: dict[str, dict]) -> list[dict]:
    by_record = defaultdict(dict)
    for r in rows:
        if r["set"] in sets:
            by_record[r["record_id"]][r["annotator_id"]] = r
    annotators = sorted({a for rec in by_record.values() for a in rec}, key=lambda x: (x == LLM_ANNOTATOR, x))
    out = []
    for rid, rec in sorted(by_record.items()):
        for field in FIELDS:
            elig = _eligible(field, rec)
            if len({fmt(field, r) for r in elig.values()}) > 1:
                d = data.get(rid, {})
                out.append({"record_id": rid, "field": field, "platform": d.get("platform"),
                            "language": d.get("language"), "text_original": d.get("text_original"),
                            "text_en": d.get("text_en"),
                            **{a: fmt(field, rec.get(a)) for a in annotators},
                            "comments": " | ".join(f"{a}: {r['comment']}" for a, r in rec.items() if r["comment"]),
                            "resolved": "", "hard_case": ""})
    return out


def write_sheet(path: Path, rows: list[dict]) -> None:
    """Write the sheet, keeping "resolved"/"hard_case" already filled in for the same (record, field)."""
    old = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as f:
            old = {(r["record_id"], r["field"]): r for r in csv.DictReader(f)}
    for r in rows:
        if prev := old.get((r["record_id"], r["field"])):
            r["resolved"], r["hard_case"] = prev.get("resolved", ""), prev.get("hard_case", "")
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:          # BOM: opens cleanly in Excel
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


ROUNDS = {"calibration": ("calibration",), "calibration2": ("calibration_reserve",), "gold": ("gold",)}


def main() -> None:
    rows = list(read_jsonl(ANNOTATIONS))
    data = task_data()
    out = RESULTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    for name, sets in ROUNDS.items():
        # calibration rounds measure the students against each other; the LLM is compared on the gold set
        round_rows = [r for r in rows if not (name.startswith("calibration") and r["annotator_id"] == LLM_ANNOTATOR)]
        pairs = pair_items(round_rows, sets)
        if not pairs:
            continue
        table = pd.DataFrame({f"{a} vs {b}": pair_scores(v) for (a, b), v in pairs.items()}).T
        pooled = [it for v in pairs.values() for it in v]
        table.loc["pooled"] = pair_scores(pooled)
        # guideline: report agreement separately for both-original, both-translation and mixed pairs
        for label, keep in (("both original", lambda x, y: x["read_from"] == y["read_from"] == "original"),
                            ("both translation", lambda x, y: x["read_from"] == y["read_from"] == "translation"),
                            ("mixed", lambda x, y: x["read_from"] != y["read_from"])):
            subset = [(x, y) for x, y in pooled if keep(x, y)]
            if subset:
                table.loc[f"pooled: {label}"] = pair_scores(subset)
        if name == "gold":                                    # all students vs the LLM, and per student
            table.index = [i.replace(f" vs {LLM_ANNOTATOR}", " vs LLM") for i in table.index]
        conds, ents = per_label(pairs)
        for suffix, t in (("", table), ("_conditions", conds), ("_entities", ents)):
            t.round(3).to_csv(out / f"spec7_agreement_{name}{suffix}.csv")
            (out / f"spec7_agreement_{name}{suffix}.md").write_text(t.to_markdown(floatfmt=".2f"), encoding="utf-8")
        sheet = disagreements(round_rows, sets, data)
        write_sheet(GOLD_DIR / f"disagreements_{name}.csv", sheet)
        k = table.loc["pooled", "conditions_kappa"]
        print(f"[{name}] {len(pairs)} pair(s), {int(table.loc['pooled', 'records'])} paired items")
        print(table.round(2).to_string())
        print(f"  {len(sheet)} disagreements -> {GOLD_DIR / f'disagreements_{name}.csv'}")
        if name == "calibration":
            verdict = "below" if k < KAPPA_THRESHOLD else "at or above"
            print(f"  conditions kappa {k:.2f} is {verdict} {KAPPA_THRESHOLD}"
                  + (": run calibration round 2 (guideline)" if k < KAPPA_THRESHOLD else ": go on to the gold set"))


if __name__ == "__main__":
    main()
