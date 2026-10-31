"""Spec 9: compare CLIP/SigLIP image-text similarity with the gold image-text consistency labels.

Input: results/predictions/spec9_clip.jsonl (downloaded from notebooks/colab_clip.ipynb; masked, committed) and
data/gold/gold.jsonl, so the evaluation runs without the corpus.
Tables (results/tables/):
* spec9_matched_vs_partner: per model x set x platform, mean similarity of the note with its own photo and with
  another record's photo, and the AUC of telling them apart (0.5 = chance). A model that captures consistency
  must at least score the real pair above a random one.
* spec9_vs_gold: per model, mean similarity per gold image_text label; AUC of "consistent" vs "partly
  consistent"/"inconsistent", AUC of the judged records vs "cannot judge" (photo shows no water); Spearman's rho between
  the similarity and the ordinal label (consistent 2, partly 1, inconsistent 0); AUC of the note being "about
  water" (gold relevance) among photo records.
Figure: results/figures/spec9_similarity.png (own vs partner photo; similarity by gold label).
Run: uv run python -m cswo.multimodal.clip_eval
"""
from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from cswo import RESULTS
from cswo.annotate.adjudicate import GOLD
from cswo.collect.common import read_jsonl
from cswo.multimodal.clip import MODELS

PRED = RESULTS / "predictions" / "spec9_clip.jsonl"
ORDINAL = {"consistent": 2, "partly consistent": 1, "inconsistent": 0}


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return math.nan
    return roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg)


def matched_table(preds: list[dict]) -> pd.DataFrame:
    groups = defaultdict(list)
    for p in preds:
        if p["score"] is None or p["score_partner"] is None:
            continue
        groups[(p["model"], p["set"], p["platform"])].append(p)
        groups[(p["model"], p["set"], "all")].append(p)
    rows = []
    for (model, kind, platform), ps in sorted(groups.items()):
        own, other = [p["score"] for p in ps], [p["score_partner"] for p in ps]
        rows.append({"model": model, "set": kind, "platform": platform, "pairs": len(ps),
                     "own photo": sum(own) / len(own), "other photo": sum(other) / len(other),
                     "own > other %": 100 * sum(a > b for a, b in zip(own, other)) / len(ps),
                     "AUC own vs other": auc(own, other)})
    return pd.DataFrame(rows).set_index(["model", "set", "platform"])


def gold_table(preds: list[dict], gold: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        ps = [p for p in preds if p["model"] == model and p["record_id"] in gold and p["score"] is not None]
        by = defaultdict(list)
        for p in ps:
            by[gold[p["record_id"]]["image_text"]].append(p["score"])
        row = {"model": model, "records": len(ps)}
        for label in ["consistent", "partly consistent", "inconsistent", "cannot judge"]:
            s = by.get(label, [])
            row[f"mean {label} (n={len(s)})"] = sum(s) / len(s) if s else math.nan
        row["AUC consistent vs partly/inconsistent"] = auc(by["consistent"],
                                                           by["partly consistent"] + by["inconsistent"])
        # oriented so that > 0.5 means "cannot judge" photos (no water shown) score lower than the judged ones
        row["AUC judged vs cannot judge"] = auc([s for k, v in by.items() if k in ORDINAL for s in v],
                                                by["cannot judge"])
        ordinal = [(p["score"], ORDINAL[gold[p["record_id"]]["image_text"]]) for p in ps
                   if gold[p["record_id"]]["image_text"] in ORDINAL]
        rho = spearmanr(*zip(*ordinal)) if len(ordinal) > 2 else None
        row["Spearman rho (ordinal)"] = rho.statistic if rho else math.nan
        row["Spearman p"] = rho.pvalue if rho else math.nan
        rel = defaultdict(list)
        for p in ps:
            rel[gold[p["record_id"]]["relevance"]].append(p["score"])
        row["AUC about water vs not"] = auc(rel["about water"], rel["not about water"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def figure(preds: list[dict], gold: dict[str, dict], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cswo.analysis.zipf_heaps import FIT, GRID, INK, INK_2, OBSERVED, SURFACE

    models = [m for m in MODELS if any(p["model"] == m for p in preds)]
    fig, axes = plt.subplots(2, len(models), figsize=(3.8 * len(models), 6.4), facecolor=SURFACE, squeeze=False)
    labels = ["consistent", "partly consistent", "inconsistent", "cannot judge"]
    for j, model in enumerate(models):
        ps = [p for p in preds if p["model"] == model and p["score"] is not None and p["score_partner"] is not None]
        ax = axes[0][j]
        lo = min(min(p["score"], p["score_partner"]) for p in ps)
        hi = max(max(p["score"], p["score_partner"]) for p in ps)
        bins = [lo + (hi - lo) * k / 30 for k in range(31)]
        ax.hist([p["score_partner"] for p in ps], bins=bins, color=FIT, alpha=0.55, label="other record's photo")
        ax.hist([p["score"] for p in ps], bins=bins, color=OBSERVED, alpha=0.55, label="own photo")
        ax.set_title(model, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("cosine similarity", fontsize=8, color=INK_2)
        ax.set_ylabel("records", fontsize=8, color=INK_2)
        ax = axes[1][j]
        g = [(gold[p["record_id"]]["image_text"], p["score"]) for p in ps if p["record_id"] in gold]
        data = [[s for lab, s in g if lab == label] for label in labels]
        for k, d in enumerate(data):
            ax.scatter([k + 1 + 0.08 * ((i % 5) - 2) for i in range(len(d))], d, s=10, color=OBSERVED, alpha=0.6,
                       linewidths=0)
            if d:
                ax.hlines(sum(d) / len(d), k + 0.7, k + 1.3, color=FIT, linewidth=2)
        ax.set_xticks(range(1, len(labels) + 1),
                      [f"{lab}\n(n={len(d)})" for lab, d in zip(labels, data)], fontsize=7)
        ax.set_ylabel("cosine similarity", fontsize=8, color=INK_2)
        ax.set_title("by gold image-text label (line = mean)", fontsize=9, color=INK_2, loc="left")
        for a in axes[:, j]:
            a.set_facecolor(SURFACE)
            a.grid(True, color=GRID, linewidth=0.6)
            a.tick_params(colors=INK_2, labelsize=7)
            for s in a.spines.values():
                s.set_color(GRID)
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper right", frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    if not PRED.exists():
        raise SystemExit(f"{PRED} not found: run notebooks/colab_clip.ipynb and save its output there")
    preds = list(read_jsonl(PRED))
    gold = {g["record_id"]: g for g in read_jsonl(GOLD) if g["set"] == "gold"}
    out = RESULTS / "tables"
    for name, t in (("spec9_matched_vs_partner", matched_table(preds)), ("spec9_vs_gold", gold_table(preds, gold))):
        t.round(3).to_csv(out / f"{name}.csv")
        (out / f"{name}.md").write_text(t.to_markdown(floatfmt=".3f"), encoding="utf-8")
        print(t.round(3).to_string(), "\n")
    figure(preds, gold, RESULTS / "figures" / "spec9_similarity.png")
    print(f"-> {out}/spec9_*.md, results/figures/spec9_similarity.png")


if __name__ == "__main__":
    main()
