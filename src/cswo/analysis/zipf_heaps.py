"""Spec 4: Zipf's and Heaps' laws per platform and per language, with Brown and Reuters as references.

Zipf: OLS of log10(frequency) on log10(rank) over all ranks, with the 90% confidence band of the fitted line and
the 90% prediction band (with 10^4 points the confidence band is very thin; the prediction band shows the scatter).
R^2 and adjusted R^2 are reported, plus the slope over the top 1000 ranks, because the long tail of words seen
once (frequency ties) dominates a full-range fit.
Heaps: vocabulary size V against token count N at every 5% of the tokens (notes in a fixed random order),
fitted as log V = log K + beta log N; spec4_heaps_increments_* lists the new types added in each 5% step.
Run: uv run python -m cswo.analysis.zipf_heaps
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import statsmodels.api as sm

from cswo import RESULTS
from cswo.analysis.text import notes, reference_corpora, tokenize

MIN_TOKENS = 25_000            # groups smaller than this are too small for a stable rank-frequency curve
TOP_RANKS = 1000
ALPHA = 0.10                   # 90% bands (spec 4)
SEED = 0
EXCLUDE_LANGS = {"und", "-"}

# Reference palette (dataviz skill, light mode): one series per panel, so only two roles are needed.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OBSERVED, FIT = "#2a78d6", "#eb6834"


# ------------------------------------------------------------------------------------------------ fitting

def _ols(x: np.ndarray, y: np.ndarray):
    return sm.OLS(y, sm.add_constant(x)).fit()


def zipf(tokens: list[str]) -> dict:
    freq = np.array(sorted(Counter(tokens).values(), reverse=True), dtype=float)
    rank = np.arange(1, len(freq) + 1, dtype=float)
    x, y = np.log10(rank), np.log10(freq)
    fit = _ols(x, y)
    top = _ols(x[:TOP_RANKS], y[:TOP_RANKS])
    lo, hi = fit.conf_int(ALPHA)[1]
    return {"types": len(freq), "tokens": int(freq.sum()), "slope": fit.params[1], "slope 90% lo": lo,
            "slope 90% hi": hi, "intercept": fit.params[0], "R2": fit.rsquared, "adj R2": fit.rsquared_adj,
            f"slope top{TOP_RANKS}": top.params[1], f"R2 top{TOP_RANKS}": top.rsquared,
            "_x": x, "_y": y, "_fit": fit}


def heaps(units: list[list[str]]) -> dict:
    order = units[:]
    random.Random(SEED).shuffle(order)
    total = sum(map(len, order))
    marks = [int(total * k / 20) for k in range(1, 21)]          # 5%, 10%, ..., 100%
    seen, n, ns, vs, m = set(), 0, [], [], 0
    for u in order:
        for t in u:
            seen.add(t)
            n += 1
            while m < len(marks) and n >= marks[m]:
                ns.append(n)
                vs.append(len(seen))
                m += 1
    N, V = np.array(ns, float), np.array(vs, float)
    fit = _ols(np.log10(N), np.log10(V))
    return {"tokens": total, "types": len(seen), "K": 10 ** fit.params[0], "beta": fit.params[1],
            "beta 90% lo": fit.conf_int(ALPHA)[1][0], "beta 90% hi": fit.conf_int(ALPHA)[1][1],
            "R2": fit.rsquared, "adj R2": fit.rsquared_adj, "_N": N, "_V": V, "_fit": fit}


# ------------------------------------------------------------------------------------------------ groups

def groups() -> tuple[dict[str, list[list[str]]], dict[str, list[list[str]]]]:
    by_platform, by_lang = defaultdict(list), defaultdict(list)
    for n in notes():
        toks = tokenize(n["text"])
        by_platform[n["platform"]].append(toks)
        by_lang[n["language"]].append(toks)
    big = lambda g: {k: v for k, v in sorted(g.items(), key=lambda kv: -sum(map(len, kv[1])))
                     if sum(map(len, v)) >= MIN_TOKENS and k not in EXCLUDE_LANGS}
    refs = {name: [tokenize(s) for s in sents] for name, sents in reference_corpora().items()}
    return {**big(by_platform), **refs}, {**big(by_lang), **refs}


# ------------------------------------------------------------------------------------------------ plots

def _style(ax, title: str, is_ref: bool) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title + ("  (reference)" if is_ref else ""), fontsize=10, color=INK, loc="left")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=INK_2, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)


def _grid(n: int):
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.1 * rows), facecolor=SURFACE, squeeze=False)
    axes = list(axes.flat)            # a list, not the .flat iterator: the legend reuses the first panel
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes


def plot_zipf(results: dict[str, dict], refs: set[str], path) -> None:
    fig, axes = _grid(len(results))
    for ax, (name, r) in zip(axes, results.items()):
        x, y, fit = r["_x"], r["_y"], r["_fit"]
        # draw at most ~400 log-spaced points: the fit uses every rank, the plot only needs the shape
        idx = np.unique(np.geomspace(1, len(x), 400).astype(int) - 1)
        grid_x = np.linspace(x.min(), x.max(), 100)
        pred = fit.get_prediction(sm.add_constant(grid_x)).summary_frame(alpha=ALPHA)
        ax.fill_between(grid_x, pred.obs_ci_lower, pred.obs_ci_upper, color=FIT, alpha=0.12, linewidth=0,
                        label="90% prediction band")
        ax.fill_between(grid_x, pred.mean_ci_lower, pred.mean_ci_upper, color=FIT, alpha=0.45, linewidth=0,
                        label="90% confidence band")
        ax.scatter(x[idx], y[idx], s=6, color=OBSERVED, linewidths=0, label="observed")
        ax.plot(grid_x, pred["mean"], color=FIT, linewidth=2, label="OLS fit")
        _style(ax, name, name in refs)
        ax.text(0.97, 0.95, f"slope {r['slope']:.2f}\nR² {r['R2']:.3f}\nadj R² {r['adj R2']:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK_2)
        ax.set_xlabel("log10 rank", fontsize=8, color=INK_2)
        ax.set_ylabel("log10 frequency", fontsize=8, color=INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_heaps(results: dict[str, dict], refs: set[str], path) -> None:
    fig, axes = _grid(len(results))
    for ax, (name, r) in zip(axes, results.items()):
        N, V, fit = r["_N"], r["_V"], r["_fit"]
        grid_n = np.linspace(N.min(), N.max(), 100)
        ax.plot(grid_n, 10 ** fit.predict(sm.add_constant(np.log10(grid_n))), color=FIT, linewidth=2,
                label="fit V = K·N^β")
        ax.scatter(N, V, s=14, color=OBSERVED, linewidths=0, zorder=3, label="observed (every 5% of tokens)")
        _style(ax, name, name in refs)
        ax.text(0.03, 0.95, f"β {r['beta']:.3f}\nK {r['K']:.1f}\nR² {r['R2']:.4f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color=INK_2)
        ax.set_xlabel("tokens N", fontsize=8, color=INK_2)
        ax.set_ylabel("vocabulary V", fontsize=8, color=INK_2)
        thousands = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1000:,.0f}k" if v >= 1000 else f"{v:.0f}")
        ax.xaxis.set_major_formatter(thousands)
        ax.yaxis.set_major_formatter(thousands)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ main

def increments(results: dict[str, dict]) -> pd.DataFrame:
    """New word types added in each 5% step of tokens: does vocabulary growth slow down, and how fast?"""
    rows = {}
    for k, r in results.items():
        new = np.diff(np.concatenate([[0], r["_V"]]))
        rows[k] = {**{f"{5 * (i + 1)}%": v for i, v in enumerate(new)}, "last / first step": new[-1] / new[0],
                   "last / second step": new[-1] / new[1]}
    return pd.DataFrame(rows).T


def _table(results: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame({k: {c: v for c, v in r.items() if not c.startswith("_")} for k, r in results.items()}).T


def main() -> None:
    tables, figures = RESULTS / "tables", RESULTS / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    by_platform, by_lang = groups()
    refs = {"Brown", "Reuters"}
    for label, g in (("platform", by_platform), ("language", by_lang)):
        z = {k: zipf([t for u in v for t in u]) for k, v in g.items()}
        h = {k: heaps(v) for k, v in g.items()}
        plot_zipf(z, refs, figures / f"spec4_zipf_by_{label}.png")
        plot_heaps(h, refs, figures / f"spec4_heaps_by_{label}.png")
        for name, res in (("zipf", z), ("heaps", h)):
            t = _table(res).astype(float)
            t.round(4).to_csv(tables / f"spec4_{name}_by_{label}.csv")
            (tables / f"spec4_{name}_by_{label}.md").write_text(t.to_markdown(floatfmt=".3f"), encoding="utf-8")
            print(f"\n## {name} by {label}\n{t.round(3).to_string()}")
        inc = increments(h)
        inc.round(3).to_csv(tables / f"spec4_heaps_increments_by_{label}.csv")
        (tables / f"spec4_heaps_increments_by_{label}.md").write_text(inc.to_markdown(floatfmt=".2f"), encoding="utf-8")
        print(f"\n## heaps increments by {label}\n{inc.round(2).to_string()}")


if __name__ == "__main__":
    main()
