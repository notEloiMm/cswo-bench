"""Spec 13: GLiNER, a zero-shot NER model, against the spec 5 extractors on the gold entity spans.

GLiNER (Zaratiana et al., NAACL 2024) is a bidirectional encoder that matches text spans against entity type names
given at inference time, so it can be asked for exactly the guideline's six types — including WATER_BODY, ORGANISM
and WEATHER, which none of the spec 5 NER models know. We use the multilingual checkpoint
`urchade/gliner_multi-v2.1` (mDeBERTa-v3-base); it runs on a CPU (~0.3 s per note).

Fixed before scoring, so nothing is tuned on the gold set (the 6 calibration spans are too few to tune on):
* label set `guideline`: the guideline's type names in plain words — the primary run;
  label set `descriptive`: a longer wording of each type, reported as a sensitivity check;
* threshold 0.5 (GLiNER's default) is the headline; 0.3–0.7 are reported as sensitivity;
* spans containing `[PERSON]` are dropped (the guideline never marks masked names).
Scored like spec 7 (`cswo.extract.evaluate`): exact and partial span P/R/F1 on the 136 gold records with entity
labels (150 spans), next to spaCy, the XLM-R NER, the rules and their union, plus GLiNER joined with the rules.
Run: uv run --with gliner python -m cswo.sota.gliner_ner run   -> results/predictions/spec13_gliner.jsonl
     uv run python -m cswo.sota.gliner_ner                    -> results/tables/spec13_*
"""
from __future__ import annotations

import argparse

import pandas as pd

from cswo import RESULTS
from cswo.annotate.adjudicate import GOLD
from cswo.annotate.agreement import ENTITY_LABELS, span_matches
from cswo.annotate.llm_input import OUT as GOLD_TEXTS
from cswo.collect.common import read_jsonl, write_jsonl
from cswo.extract.evaluate import entity_lists, prf, system_outputs

MODEL = "urchade/gliner_multi-v2.1"
LABEL_SETS = {
    "guideline": {"water body": "WATER_BODY", "location": "LOCATION", "organism": "ORGANISM",
                  "quantity": "QUANTITY", "date or time": "DATE_TIME", "weather": "WEATHER"},
    "descriptive": {"named lake, river, stream, pond, sea or canal": "WATER_BODY",
                    "place, town or site that is not a water body": "LOCATION",
                    "plant, animal, alga or fungus name": "ORGANISM",
                    "number, count or measurement": "QUANTITY", "date, time or relative time": "DATE_TIME",
                    "weather condition": "WEATHER"},
}
HEADLINE = ("guideline", 0.5)
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
MIN_SCORE = min(THRESHOLDS)
PRED = RESULTS / "predictions" / "spec13_gliner.jsonl"


def to_spans(raw: list[dict], label_map: dict[str, str]) -> list[dict]:
    """GLiNER output -> guideline spans (offsets kept, masked names dropped)."""
    return [{"start": e["start"], "end": e["end"], "text": e["text"], "label": label_map[e["label"]],
             "score": round(float(e["score"]), 4)}
            for e in raw if e["label"] in label_map and "[PERSON]" not in e["text"]]


def run() -> None:
    from gliner import GLiNER
    model = GLiNER.from_pretrained(MODEL)
    gold_ids = {g["record_id"] for g in read_jsonl(GOLD) if g["entities"] is not None}
    rows = []
    for r in read_jsonl(GOLD_TEXTS):
        if r["record_id"] not in gold_ids:
            continue
        for name, label_map in LABEL_SETS.items():
            raw = model.predict_entities(r["text_original"], list(label_map), threshold=MIN_SCORE)
            rows.append({"record_id": r["record_id"], "label_set": name, "model": MODEL,
                         "entities": to_spans(raw, label_map)})
    n = write_jsonl(PRED, rows)
    print(f"{n} rows ({len(gold_ids)} records x {len(LABEL_SETS)} label sets) -> {PRED}")


def at(spans: list[dict], threshold: float) -> list[dict]:
    return [e for e in spans if e["score"] >= threshold]


def score(pred: dict[str, list[dict]], gold: dict[str, list[dict]]) -> list[dict]:
    """Rows per type and micro-averaged: exact and partial P/R/F1 (spec 7 convention)."""
    rows = []
    for label in [*ENTITY_LABELS, None]:
        row = {"label": label or "all (micro)"}
        for mode, partial in (("exact", False), ("partial", True)):
            tp = npred = ng = 0
            for rid, g in gold.items():
                m, a, b = span_matches(pred.get(rid, []), g, partial, label)
                tp, npred, ng = tp + m, npred + a, ng + b
            row.update({f"{k} {mode}": v for k, v in prf(tp, npred - tp, ng - tp).items()})
        row.update({"gold spans": ng, "predicted": npred})
        rows.append(row)
    return rows


def tables(preds: list[dict], gold: dict[str, list[dict]], outs: dict[str, dict]) -> dict[str, pd.DataFrame]:
    by = {(p["label_set"], p["record_id"]): p["entities"] for p in preds}
    base = {s: {rid: entity_lists(outs[rid])[s] for rid in gold} for s in ("spacy", "transformer", "rules", "union")}

    def system(name, lab, thr):
        return {rid: at(by.get((lab, rid), []), thr) for rid in gold}

    head = system("gliner", *HEADLINE)
    joined = {rid: head[rid] + [e for e in base["rules"][rid]
                                if not any(span_matches([e], [x], True)[0] for x in head[rid])] for rid in gold}
    systems = {"spaCy (spec 5)": base["spacy"], "XLM-R NER (spec 5)": base["transformer"],
               "rules (spec 5)": base["rules"], "union of spec 5 systems": base["union"],
               "GLiNER multi v2.1 (zero-shot)": head, "GLiNER + rules": joined}
    per = []
    for name, pred in systems.items():
        per += [{"system": name, **r} for r in score(pred, gold)]
    per = pd.DataFrame(per).set_index(["system", "label"])
    main = per.xs("all (micro)", level="label")

    sens = []
    for lab in LABEL_SETS:
        for thr in THRESHOLDS:
            micro = score(system("gliner", lab, thr), gold)[-1]
            sens.append({"label set": lab, "threshold": thr, **{k: v for k, v in micro.items() if k != "label"}})
    sens = pd.DataFrame(sens).set_index(["label set", "threshold"])
    return {"spec13_entities": main, "spec13_entities_per_type": per, "spec13_gliner_sensitivity": sens}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default="score", choices=["run", "score"])
    if ap.parse_args().command == "run":
        run()
        return
    if not PRED.exists():
        raise SystemExit(f"{PRED} not found: uv run --with gliner python -m cswo.sota.gliner_ner run")
    gold = {g["record_id"]: g["entities"] for g in read_jsonl(GOLD) if g["entities"] is not None}
    t = tables(list(read_jsonl(PRED)), gold, system_outputs(set(gold)))
    out = RESULTS / "tables"
    for name, df in t.items():
        df.round(3).to_csv(out / f"{name}.csv")
        (out / f"{name}.md").write_text(df.to_markdown(floatfmt=".2f"), encoding="utf-8")
        if name != "spec13_entities_per_type":
            print(f"\n## {name}\n{df.round(2).to_string()}")
    print(f"\n-> {out}/spec13_*.md")


if __name__ == "__main__":
    main()
