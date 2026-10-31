"""Spec 7: precision, recall and F1 of the spec 5 extractors on the adjudicated gold set.

Inputs: data/gold/gold.jsonl (cswo.annotate.adjudicate) and the extractor outputs for the gold records, cut once
from data/processed/entities.jsonl into data/gold/system_outputs.jsonl (small and masked, so it is committed and
the evaluation runs without the full corpus).
* Entities, on the original text: spaCy, the transformer, the rules and their union, per gold type; exact match
  (same offsets and type) and partial match (same type, overlapping). Only records the gold has spans for (about
  water and read in the original): a note marked "not about water" was never span-annotated.
* Water condition: the record-level rule labels against the gold conditions ("none stated" = empty set), on every
  record with resolved conditions; a not-about-water note counts with no conditions.
* Relevance: share of gold "about water" per platform, i.e. the precision of the spec 1 water filter.
* English original vs machine translation: the same metrics on English-original notes, on non-English notes in
  their own language, and on the English MT of those same notes (spaCy and rules were run on text_en in spec 5).
  MT spans cannot be aligned to offsets in the original, so this table matches entities by type and count per
  record (a predicted QUANTITY counts if the gold has a QUANTITY left unmatched in that record), for all rows alike.
Run: uv run python -m cswo.extract.evaluate
"""
from __future__ import annotations

import math
from collections import Counter

import pandas as pd

from cswo import RESULTS
from cswo.annotate.adjudicate import GOLD
from cswo.annotate.agreement import CONDITIONS, ENTITY_LABELS, span_matches
from cswo.annotate.sample import GOLD_DIR
from cswo.collect.common import read_jsonl, write_jsonl
from cswo.extract.run import ENTITIES

SYSTEM_OUTPUTS = GOLD_DIR / "system_outputs.jsonl"
SYSTEMS = ["spacy", "transformer", "rules"]


def system_outputs(gold_ids: set[str]) -> dict[str, dict]:
    cached = {r["id"]: r for r in read_jsonl(SYSTEM_OUTPUTS)} if SYSTEM_OUTPUTS.exists() else {}
    if gold_ids - set(cached):
        if not ENTITIES.exists():
            raise SystemExit(f"{len(gold_ids - set(cached))} gold records have no extractor output: run "
                             f"cswo.extract.run first ({ENTITIES} is missing)")
        cached.update({r["id"]: r for r in read_jsonl(ENTITIES) if r["id"] in gold_ids})
        write_jsonl(SYSTEM_OUTPUTS, sorted(cached.values(), key=lambda r: r["id"]))
    return cached


def entity_lists(out: dict) -> dict[str, list[dict]]:
    """System name -> spans on the original text; "union" merges all three (one span per offsets + type)."""
    o = out["original"]
    lists = {"spacy": o["spacy"], "transformer": o["transformer"], "rules": o["rules"]["entities"]}
    union = {(e["start"], e["end"], e["label"]): e for s in SYSTEMS for e in lists[s]}
    lists["union"] = list(union.values())
    return lists


def conditions_of(rule_out: dict) -> list[str]:
    return sorted(c.replace("_", " ") for c in rule_out["water_condition"])


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else math.nan
    r = tp / (tp + fn) if tp + fn else math.nan
    f = 2 * p * r / (p + r) if tp else (0.0 if tp + fp + fn else math.nan)     # F1 = 0 when anything was missed
    return {"P": p, "R": r, "F1": f}


def entity_table(gold: list[dict], outs: dict[str, dict]) -> pd.DataFrame:
    recs = [g for g in gold if g["entities"] is not None]
    rows = []
    for system in [*SYSTEMS, "union"]:
        for label in [*ENTITY_LABELS, None]:
            row = {"system": system, "label": label or "all (micro)"}
            for mode, partial in (("exact", False), ("partial", True)):
                tp = ng = npred = 0
                for g in recs:
                    m, a, b = span_matches(entity_lists(outs[g["record_id"]])[system], g["entities"], partial, label)
                    tp, npred, ng = tp + m, npred + a, ng + b
                row.update({f"{k} {mode}": v for k, v in prf(tp, npred - tp, ng - tp).items()})
            row.update({"gold spans": ng, "predicted": npred})
            rows.append(row)
    return pd.DataFrame(rows).set_index(["system", "label"])


def condition_table(gold: list[dict], outs: dict[str, dict], text: str = "original") -> pd.DataFrame:
    recs = [g for g in gold if g["conditions"] is not None]
    pred = {g["record_id"]: conditions_of(outs[g["record_id"]][text]["rules"]) for g in recs}
    rows, tps, fps, fns = [], 0, 0, 0
    for c in CONDITIONS:
        tp = sum(c in pred[g["record_id"]] and c in g["conditions"] for g in recs)
        fp = sum(c in pred[g["record_id"]] and c not in g["conditions"] for g in recs)
        fn = sum(c not in pred[g["record_id"]] and c in g["conditions"] for g in recs)
        rows.append({"condition": c, "gold": tp + fn, "predicted": tp + fp, **prf(tp, fp, fn)})
        tps, fps, fns = tps + tp, fps + fp, fns + fn
    t = pd.DataFrame(rows).set_index("condition")
    t.loc["all (micro)"] = {"gold": tps + fns, "predicted": tps + fps, **prf(tps, fps, fns)}
    supported = t.loc[CONDITIONS][t.loc[CONDITIONS, "gold"] > 0]
    t.loc["all (macro, labels in gold)"] = {"gold": supported["gold"].sum(), "predicted": supported["predicted"].sum(),
                                            "P": supported["P"].mean(), "R": supported["R"].mean(),
                                            "F1": supported["F1"].mean()}
    return t


def relevance_table(gold: list[dict]) -> pd.DataFrame:
    recs = [g for g in gold if g["relevance"]]
    t = pd.crosstab(pd.Series([g["platform"] for g in recs], name="platform"),
                    pd.Series([g["relevance"] for g in recs], name="relevance"), margins=True, margins_name="all")
    t["water filter precision"] = t.get("about water", 0) / t["all"]
    return t


def _type_count(pred: list[dict], gold: list[dict]) -> tuple[int, int, int]:
    p, g = Counter(e["label"] for e in pred), Counter(e["label"] for e in gold)
    tp = sum(min(p[k], g[k]) for k in p)
    return tp, sum(p.values()) - tp, sum(g.values()) - tp


def mt_table(gold: list[dict], outs: dict[str, dict]) -> pd.DataFrame:
    en = [g for g in gold if g["language"] == "en"]
    other = [g for g in gold if g["language"] != "en" and "en" in outs[g["record_id"]]]
    groups = {"English original": (en, "original"), "non-English, original language": (other, "original"),
              "non-English, MT to English": (other, "en")}
    rows = []
    for group, (recs, text) in groups.items():
        for system in ("spacy", "rules"):
            row = {"text": group, "system": system}
            ent = [g for g in recs if g["entities"] is not None]
            tp = fp = fn = 0
            for g in ent:
                o = outs[g["record_id"]][text]
                pred = o["rules"]["entities"] if system == "rules" else o["spacy"]
                a, b, c = _type_count(pred, g["entities"])
                tp, fp, fn = tp + a, fp + b, fn + c
            row.update({"entity records": len(ent), **{f"entity {k}": v for k, v in prf(tp, fp, fn).items()}})
            if system == "rules":
                cond = condition_table(recs, outs, text).loc["all (micro)"] if recs else {}
                row.update({"condition records": sum(g["conditions"] is not None for g in recs),
                            **{f"condition {k}": cond.get(k, math.nan) for k in ("P", "R", "F1")}})
            rows.append(row)
    return pd.DataFrame(rows).set_index(["text", "system"])


def main() -> None:
    gold = list(read_jsonl(GOLD))
    outs = system_outputs({g["record_id"] for g in gold})
    gold = [g for g in gold if g["record_id"] in outs]
    out = RESULTS / "tables"
    tables = {"spec7_extractors_entities": entity_table(gold, outs),
              "spec7_extractors_conditions": condition_table(gold, outs),
              "spec7_relevance_by_platform": relevance_table(gold),
              "spec7_mt_comparison": mt_table(gold, outs)}
    for name, t in tables.items():
        t.round(3).to_csv(out / f"{name}.csv")
        (out / f"{name}.md").write_text(t.to_markdown(floatfmt=".2f"), encoding="utf-8")
        print(f"\n{name}\n{t.round(2).to_string()}")
    print(f"\n-> {out}/spec7_*")


if __name__ == "__main__":
    main()
