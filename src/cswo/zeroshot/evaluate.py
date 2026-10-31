"""Spec 10: score the zero-shot LLM and VLM against the supervised baselines, and measure hallucination.

Inputs: data/zeroshot/items.jsonl (cswo.zeroshot.inputs), results/predictions/spec10_zeroshot.jsonl (downloaded from
notebooks/colab_zeroshot.ipynb), the spec 8 predictions and data/gold/system_outputs.jsonl (rules, spec 7).

* spec10_vs_supervised: habitat and water level, zero-shot models next to the majority class, TF-IDF + SVM and
  DistilBERT re-scored on the same test records (macro F1 is the headline, as in spec 8). An answer outside the
  label set counts as wrong.
* spec10_gold: relevance (accuracy, macro F1), water conditions (micro and macro P/R/F1 as in spec 7, with the
  rules as the non-LLM reference) and, for the VLM, image-text consistency (accuracy, Cohen's kappa) on the gold set.
  Caveat for the report: the gold set was adjudicated against Qwen2.5-VL annotations (spec 7), so the VLM row
  is not independent of the gold labels.
* spec10_hallucination: automatic checks on every output (unparsable, label outside the set, quoted evidence that
  does not occur in the note) and, once the sheets are filled in, the manual hallucination rate on 100 outputs
  per model (data/zeroshot/hallucination_<model>.csv, see data/zeroshot/HALLUCINATION_CHECK.md).
Run: uv run python -m cswo.zeroshot.evaluate sheets   (once, writes the empty manual sheets; never overwrites)
     uv run python -m cswo.zeroshot.evaluate          (tables)
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
from collections import Counter

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from cswo import DATA, RESULTS
from cswo.annotate.agreement import CONDITIONS
from cswo.annotate.sample import STUDENT_LANGUAGES, assign
from cswo.baselines.report import PRED_DIR, scores
from cswo.collect.common import read_jsonl
from cswo.extract.evaluate import conditions_of, prf, system_outputs
from cswo.zeroshot.inputs import ITEMS

PRED = PRED_DIR / "spec10_zeroshot.jsonl"
SHEET_DIR = DATA / "zeroshot"
SHEET_SIZE = {"habitat": 30, "water_level": 30, "gold": 40}
RELEVANCE = ["about water", "not about water", "uncertain"]
IMAGE_TEXT = ["consistent", "partly consistent", "inconsistent", "cannot judge"]
INVALID = "<invalid>"


def short_name(pred: dict) -> str:
    return f"{pred['model'].split('/')[-1].lower().replace('-instruct', '')} (zero-shot {pred['kind']})"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.casefold()).strip(" \"'.,;:…")


def found(evidence: str, note: str) -> bool:
    """Is the quoted evidence in the note? Case, spacing and surrounding quotes/punctuation are ignored, and a quote
    shortened with "..." counts if every part occurs."""
    parts = [_norm(p) for p in re.split(r"\.\.\.|…", evidence) if _norm(p)]
    return all(p in _norm(note) for p in parts)


def label_of(task: str, value, labels: list[str]) -> str:
    v = str(value or "").strip()
    v = re.sub(r"\s*\([^()]*\)$", "", v)               # the prompt's gloss copied back: "bäck (brook)"
    if task == "water_level" and re.fullmatch(r"[+-]?\d+", v):
        v = f"{int(v):+d}"
    match = [c for c in labels if c.casefold() == v.casefold()]
    return match[0] if match else INVALID


def validate(pred: dict, item: dict) -> dict:
    """Parsed, checked answer: label(s), and every quoted evidence with whether it occurs in the note."""
    out = pred.get("parsed")
    res = {"unparsable": out is None, "invalid": [], "evidence": []}
    out = out or {}
    if item["task"] in ("habitat", "water_level"):
        res["label"] = label_of(item["task"], out.get("label"), item["labels"])
        if res["label"] == INVALID:
            res["invalid"].append(f"label {out.get('label')!r}")
        if str(out.get("evidence") or "").strip():
            res["evidence"].append(str(out["evidence"]))
        return res
    rel = out.get("relevance")
    res["relevance"] = rel if rel in RELEVANCE else INVALID
    if res["relevance"] == INVALID:
        res["invalid"].append(f"relevance {rel!r}")
    conds = set()
    for c in out.get("conditions") or []:
        c = c if isinstance(c, dict) else {"label": c}
        name = str(c.get("label") or "").strip().lower()
        if name in CONDITIONS:
            conds.add(name)
            if str(c.get("evidence") or "").strip():
                res["evidence"].append(str(c["evidence"]))
        elif name not in ("", "none", "none stated"):
            res["invalid"].append(f"condition {name!r}")
    res["conditions"] = sorted(conds) if res["relevance"] != "not about water" else []
    it = out.get("image_text")
    res["image_text"] = (it if it in IMAGE_TEXT else INVALID) if pred["image_used"] else None
    if res["image_text"] == INVALID:
        res["invalid"].append(f"image_text {it!r}")
    return res


def checked(preds: list[dict], items: dict[str, dict]) -> list[dict]:
    rows = []
    for p in preds:
        it = items[p["item_id"]]
        v = validate(p, it)
        v["fabricated"] = [e for e in v["evidence"] if not found(e, it["text"])]
        rows.append({**p, "name": short_name(p), "check": v})
    return rows


def supervised_table(rows: list[dict], items: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = []
    for r in rows:
        if r["task"] in ("habitat", "water_level"):
            preds.append({"task": r["task"], "model": r["name"], "stopwords_removed": False,
                          "gold": items[r["item_id"]]["reference"], "pred": r["check"]["label"]})
    ids = {(it["task"], it["record_id"]) for it in items.values() if it["task"] != "gold"}
    for f in sorted(PRED_DIR.glob("spec8_*.jsonl")):
        preds += [p for p in read_jsonl(f) if (p["task"], p["id"]) in ids]
    return scores(preds)


def gold_table(rows: list[dict], items: dict[str, dict]) -> pd.DataFrame:
    gold_items = {k: it for k, it in items.items() if it["task"] == "gold"}
    rules = system_outputs({it["record_id"] for it in gold_items.values()})
    systems = {"rules (spec 5)": {it["record_id"]: {"conditions": conditions_of(rules[it["record_id"]]["original"]
                                                                                  ["rules"])}
                                  for it in gold_items.values()}}
    for r in rows:
        if r["task"] == "gold":
            systems.setdefault(r["name"], {})[r["record_id"]] = r["check"]
    out = []
    for name, pred in systems.items():
        ref = {it["record_id"]: it["reference"] for it in gold_items.values() if it["record_id"] in pred}
        row = {"system": name, "records": len(ref)}
        if all("relevance" in p for p in pred.values()):
            g = [ref[k]["relevance"] for k in ref]
            y = [pred[k]["relevance"] for k in ref]
            row.update({"relevance acc": accuracy_score(g, y),
                        "relevance macro F1": f1_score(g, y, labels=RELEVANCE, average="macro", zero_division=0)})
        tp = fp = fn = 0
        per = []
        for c in CONDITIONS:
            a = sum(c in pred[k]["conditions"] and c in (ref[k]["conditions"] or []) for k in ref)
            b = sum(c in pred[k]["conditions"] and c not in (ref[k]["conditions"] or []) for k in ref)
            d = sum(c not in pred[k]["conditions"] and c in (ref[k]["conditions"] or []) for k in ref)
            tp, fp, fn = tp + a, fp + b, fn + d
            if a + d:
                per.append(prf(a, b, d)["F1"])
        row.update({f"conditions micro {k}": v for k, v in prf(tp, fp, fn).items()})
        row["conditions macro F1 (labels in gold)"] = sum(per) / len(per) if per else math.nan
        pairs = [(ref[k]["image_text"], pred[k].get("image_text")) for k in ref
                 if ref[k]["image_text"] and pred[k].get("image_text")]
        if pairs:
            g, y = zip(*pairs)
            row.update({"image-text records": len(pairs), "image-text acc": accuracy_score(g, y),
                        "image-text kappa": cohen_kappa_score(g, y)})
        out.append(row)
    return pd.DataFrame(out).set_index("system")


def sheet_path(name: str):
    return SHEET_DIR / f"hallucination_{re.sub(r'[^a-z0-9.]+', '_', name.split(' (')[0])}.csv"


def manual_rates(name: str) -> dict:
    path = sheet_path(name)
    if not path.exists():
        return {}
    # sep=None sniffs the delimiter: Excel with a Spanish/Catalan/German locale saves CSV with ";"
    s = pd.read_csv(path, dtype=str, sep=None, engine="python", encoding="utf-8-sig").fillna("")
    filled = s[s["hallucinated"].str.strip().str.lower().isin(["yes", "no"])]
    yes = filled["hallucinated"].str.strip().str.lower() == "yes"
    types = Counter(filled.loc[yes, "hallucination_type"].str.strip().replace("", "unspecified"))
    return {"manual checked": len(filled), "manual hallucination %": 100 * yes.mean() if len(filled) else math.nan,
            "manual types": ", ".join(f"{t} {n}" for t, n in types.most_common())}


def hallucination_table(rows: list[dict]) -> pd.DataFrame:
    out = []
    for name in sorted({r["name"] for r in rows}):
        rs = [r for r in rows if r["name"] == name]
        with_ev = [r for r in rs if r["check"]["evidence"]]
        quotes = sum(len(r["check"]["evidence"]) for r in rs)
        fab = sum(len(r["check"]["fabricated"]) for r in rs)
        out.append({"model": name, "outputs": len(rs),
                    "unparsable %": 100 * sum(r["check"]["unparsable"] for r in rs) / len(rs),
                    "invalid label %": 100 * sum(bool(r["check"]["invalid"]) for r in rs) / len(rs),
                    "outputs quoting evidence": len(with_ev), "quotes": quotes,
                    "quotes not in note %": 100 * fab / quotes if quotes else math.nan,
                    "outputs with a fabricated quote %": 100 * sum(bool(r["check"]["fabricated"]) for r in rs) / len(rs),
                    **manual_rates(name)})
    return pd.DataFrame(out).set_index("model")


def _summary(r: dict) -> str:
    c = r["check"]
    if r["task"] == "gold":
        s = f"relevance={c['relevance']}; conditions={', '.join(c['conditions']) or '-'}"
        return s + (f"; image_text={c['image_text']}" if c["image_text"] else "")
    return f"label={c['label']}"


def write_sheets(rows: list[dict], items: dict[str, dict]) -> None:
    """100 outputs per model (SHEET_SIZE per task, by hash), dealt to the students by the languages they read."""
    picked = []
    for name in sorted({r["name"] for r in rows}):
        for task, n in SHEET_SIZE.items():
            rs = [r for r in rows if r["name"] == name and r["task"] == task]
            picked += sorted(rs, key=lambda r: hashlib.sha1(f"{name}|{r['item_id']}".encode()).hexdigest())[:n]
    students = sorted(STUDENT_LANGUAGES)
    keyed = [{"id": f"{r['name']}|{r['item_id']}", "_stratum": r["task"], "language": items[r["item_id"]]["language"]}
             for r in picked]
    who = assign(keyed, students, STUDENT_LANGUAGES, capacity=math.ceil(len(keyed) / len(students)))
    for name in sorted({r["name"] for r in picked}):
        path = sheet_path(name)
        if path.exists():
            print(f"{path} exists, not overwritten")
            continue
        sheet = []
        for r in (r for r in picked if r["name"] == name):
            it = items[r["item_id"]]
            sheet.append({"row": len(sheet) + 1, "checker": who[f"{name}|{r['item_id']}"], "item_id": r["item_id"],
                          "task": r["task"], "language": it["language"], "note": it["text"],
                          "note_en": it["text_en"] or "", "photo": it["image"] if r["image_used"] else "",
                          "model_output": _summary(r), "evidence": " | ".join(r["check"]["evidence"]),
                          "auto_flags": "; ".join(r["check"]["invalid"] +
                                                  [f"quote not in note: {e!r}" for e in r["check"]["fabricated"]]),
                          "hallucinated": "", "hallucination_type": "", "comment": ""})
        pd.DataFrame(sheet).to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{len(sheet)} rows -> {path} ({dict(Counter(x['checker'] for x in sheet))})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default="score", choices=["score", "sheets"])
    args = ap.parse_args()
    if not PRED.exists():
        raise SystemExit(f"{PRED} not found: run notebooks/colab_zeroshot.ipynb and save its output there")
    items = {it["item_id"]: it for it in read_jsonl(ITEMS)}
    rows = checked(list(read_jsonl(PRED)), items)
    if args.command == "sheets":
        write_sheets(rows, items)
        return
    main_table, per_class = supervised_table(rows, items)
    out = RESULTS / "tables"
    for name, t in (("spec10_vs_supervised", main_table), ("spec10_vs_supervised_per_class", per_class),
                    ("spec10_gold", gold_table(rows, items)), ("spec10_hallucination", hallucination_table(rows))):
        t.round(3).to_csv(out / f"{name}.csv")
        (out / f"{name}.md").write_text(t.to_markdown(floatfmt=".3f"), encoding="utf-8")
        if not name.endswith("per_class"):
            print(t.round(3).to_string(), "\n")
    print(f"-> {out}/spec10_*.md")


if __name__ == "__main__":
    main()
