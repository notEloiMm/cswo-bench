"""Spec 8: score every baseline's test predictions the same way.

Reads results/predictions/spec8_*.jsonl (spec8_svm.jsonl from cswo.baselines.svm, spec8_bert.jsonl downloaded
from notebooks/colab_bert.ipynb) and writes per task x model x stopword setting: accuracy, macro precision,
recall and F1 (the headline: classes are imbalanced), and weighted F1; plus a per-class table.
Run: uv run python -m cswo.baselines.report
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from cswo import RESULTS
from cswo.collect.common import read_jsonl

PRED_DIR = RESULTS / "predictions"


def scores(preds: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = defaultdict(list)
    for p in preds:
        groups[(p["task"], p["model"], p["stopwords_removed"])].append(p)
    rows, per_class = [], []
    for (task, model, stop), ps in sorted(groups.items()):
        gold, pred = [p["gold"] for p in ps], [p["pred"] for p in ps]
        labels = sorted(set(gold))
        P, R, F, _ = precision_recall_fscore_support(gold, pred, labels=labels, average="macro", zero_division=0)
        *_, Fw, _ = precision_recall_fscore_support(gold, pred, labels=labels, average="weighted", zero_division=0)
        rows.append({"task": task, "model": model, "stopwords removed": stop, "test records": len(ps),
                     "accuracy": accuracy_score(gold, pred), "macro P": P, "macro R": R, "macro F1": F,
                     "weighted F1": Fw})
        cp, cr, cf, cn = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
        per_class += [{"task": task, "model": model, "stopwords removed": stop, "class": c, "support": int(n),
                       "P": a, "R": b, "F1": f} for c, a, b, f, n in zip(labels, cp, cr, cf, cn)]
    idx = ["task", "model", "stopwords removed"]
    return pd.DataFrame(rows).set_index(idx), pd.DataFrame(per_class).set_index([*idx, "class"])


def main() -> None:
    files = sorted(PRED_DIR.glob("spec8_*.jsonl"))
    if not files:
        raise SystemExit(f"no predictions in {PRED_DIR}: run cswo.baselines.svm (and the BERT notebook)")
    preds = [p for f in files for p in read_jsonl(f)]
    main_table, per_class = scores(preds)
    out = RESULTS / "tables"
    for name, t in (("spec8_baselines", main_table), ("spec8_baselines_per_class", per_class)):
        t.round(3).to_csv(out / f"{name}.csv")
        (out / f"{name}.md").write_text(t.to_markdown(floatfmt=".3f"), encoding="utf-8")
    print(main_table.round(3).to_string())
    print(f"-> {out}/spec8_baselines*.md (from {', '.join(f.name for f in files)})")


if __name__ == "__main__":
    main()
