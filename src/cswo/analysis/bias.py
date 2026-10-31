"""Spec 11: bias audit of the corpus: where, in which language, when, and by whom the observations were made.

* Geography: records per country (top countries + the rest), the share of the top 3, and the Gini coefficient over
  countries; a density map of the 1 km coordinates per platform (results/figures/spec11_map.png). No basemap: the
  coordinates alone trace the coastlines where records are dense.
* Language: notes per detected language, the Gini over languages, and the language mix of the gold set next to the
  corpus (the gold sample is stratified, so it should not simply copy the corpus imbalance).
* Season: records per month and per meteorological season (Dec-Feb winter, ...), flipped for the southern
  hemisphere (latitude < 0) so "winter" means the local winter (results/figures/spec11_months.png).
* Contributors: per platform, users, records per user, the share of the most active 1% and 10% of users, and the
  Gini coefficient of records per user, with Lorenz curves (results/figures/spec11_lorenz.png). CrowdWater publishes
  no user id, and EyeOnWater only for some records, so those are left out or marked partial.
Gini = 0 when every unit contributes the same, 1 when one unit contributes everything.
Run: uv run python -m cswo.analysis.bias
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from cswo import RESULTS
from cswo.analysis.dataset_stats import PLATFORMS
from cswo.analysis.text import corpus_path
from cswo.annotate.adjudicate import GOLD
from cswo.collect.common import read_jsonl

TOP_COUNTRIES = 15
TOP_LANGUAGES = 12
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
MIN_USER_RECORDS = 1_000      # platforms with fewer records that carry a user id are not given a Gini


def gini(values) -> float:
    x = np.sort(np.asarray(list(values), dtype=float))
    if len(x) == 0 or x.sum() == 0:
        return float("nan")
    n = len(x)
    return float(2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n)


def top_share(values, frac: float) -> float:
    x = np.sort(np.asarray(list(values), dtype=float))[::-1]
    k = max(1, int(round(len(x) * frac)))
    return float(x[:k].sum() / x.sum())


def season(month: int, lat: float | None) -> str:
    """Meteorological season in the observer's hemisphere."""
    if lat is not None and lat < 0:
        month = (month + 5) % 12 + 1
    return SEASONS[month]


def month_of(ts: str | None) -> int | None:
    if ts and len(ts) >= 7 and ts[5:7].isdigit() and 1 <= int(ts[5:7]) <= 12:
        return int(ts[5:7])
    return None


def collect(records) -> dict:
    s = {"country": Counter(), "country_p": Counter(), "lang": Counter(), "month": Counter(), "season": Counter(),
         "users": defaultdict(Counter), "with_user": Counter(), "records": Counter(), "coords": defaultdict(list)}
    for r in records:
        p = r["source_platform"]
        s["records"][p] += 1
        s["country"][r.get("country") or "unknown"] += 1
        s["country_p"][(p, r.get("country") or "unknown")] += 1
        if r.get("text_original"):
            s["lang"][r.get("language") or "-"] += 1
        m = month_of(r.get("timestamp"))
        if m:
            s["month"][(p, m)] += 1
            s["season"][(p, season(m, r.get("lat")))] += 1
        if r.get("user_hash"):
            s["users"][p][r["user_hash"]] += 1
            s["with_user"][p] += 1
        if r.get("lat") is not None and r.get("lon") is not None:
            s["coords"][p].append((r["lon"], r["lat"]))
    return s


def tables(s: dict, gold_langs: Counter) -> dict[str, pd.DataFrame]:
    total = sum(s["country"].values())
    top = s["country"].most_common(TOP_COUNTRIES)
    rest = total - sum(n for _, n in top)
    countries = pd.DataFrame([{"country": c, "records": n, "share %": 100 * n / total} for c, n in top]
                             + [{"country": f"other ({len(s['country']) - TOP_COUNTRIES} countries)", "records": rest,
                                 "share %": 100 * rest / total}]).set_index("country")
    plats = [p for p in PLATFORMS if s["records"][p]]
    geo = pd.DataFrame([{"platform": p, "countries": sum(1 for (q, _), n in s["country_p"].items() if q == p and n),
                         "top country": max(((c, n) for (q, c), n in s["country_p"].items() if q == p),
                                            key=lambda x: x[1])[0],
                         "top-3 share %": 100 * top_share([n for (q, _), n in s["country_p"].items() if q == p],
                                                          3 / max(1, sum(1 for (q, _) in s["country_p"] if q == p))),
                         # one country: concentration is total by construction, not a finding
                         "Gini over countries": gini(n for (q, _), n in s["country_p"].items() if q == p and n)
                         if sum(1 for (q, _), n in s["country_p"].items() if q == p and n) > 1 else np.nan}
                        for p in plats])
    geo.loc[len(geo)] = {"platform": "all", "countries": len(s["country"]), "top country": top[0][0],
                         "top-3 share %": 100 * sum(n for _, n in top[:3]) / total,
                         "Gini over countries": gini(s["country"].values())}
    geo = geo.set_index("platform")

    notes = sum(s["lang"].values())
    gold_total = sum(gold_langs.values()) or 1
    langs = [l for l, _ in s["lang"].most_common(TOP_LANGUAGES)]
    language = pd.DataFrame([{"language": l, "notes": s["lang"][l], "corpus share %": 100 * s["lang"][l] / notes,
                              "gold records": gold_langs[l], "gold share %": 100 * gold_langs[l] / gold_total}
                             for l in langs]).set_index("language")
    rest_n = notes - language["notes"].sum()
    language.loc[f"other ({len(s['lang']) - len(langs)} languages)"] = {
        "notes": rest_n, "corpus share %": 100 * rest_n / notes,
        "gold records": gold_total - language["gold records"].sum(),
        "gold share %": 100 * (gold_total - language["gold records"].sum()) / gold_total}
    language.loc["Gini over languages"] = {"notes": np.nan, "corpus share %": 100 * gini(s["lang"].values()),
                                           "gold records": np.nan, "gold share %": 100 * gini(gold_langs.values())}

    months = pd.DataFrame([[s["month"][(p, m)] for m in range(1, 13)] for p in plats], index=plats,
                          columns=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    seasons = pd.DataFrame([[s["season"][(p, x)] for x in ("winter", "spring", "summer", "autumn")] for p in plats],
                           index=plats, columns=["winter", "spring", "summer", "autumn"])
    seasons.loc["all"] = seasons.sum()
    seasons = seasons.div(seasons.sum(axis=1), axis=0).mul(100).add_suffix(" %")

    rows = []
    for p in plats:
        counts = list(s["users"][p].values())
        coverage = s["with_user"][p] / s["records"][p]
        if s["with_user"][p] < MIN_USER_RECORDS:
            rows.append({"platform": p, "records with user id %": 100 * coverage, "users": len(counts)})
            continue
        rows.append({"platform": p, "records with user id %": 100 * coverage, "users": len(counts),
                     "records per user (mean)": np.mean(counts), "records per user (median)": np.median(counts),
                     "top 1% users share %": 100 * top_share(counts, 0.01),
                     "top 10% users share %": 100 * top_share(counts, 0.10), "Gini": gini(counts)})
    users = pd.DataFrame(rows).set_index("platform")
    return {"spec11_countries": countries, "spec11_geography_by_platform": geo, "spec11_languages": language,
            "spec11_months": months, "spec11_seasons": seasons, "spec11_users": users}


def _style(ax, title):
    from cswo.analysis.zipf_heaps import GRID, INK, INK_2, SURFACE
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.tick_params(colors=INK_2, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(GRID)


def figures(s: dict, months: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from cswo.analysis.zipf_heaps import GRID, INK_2, OBSERVED, SURFACE
    out = RESULTS / "figures"
    plats = list(months.index)

    cmap = LinearSegmentedColormap.from_list("seq", ["#cfe0f7", OBSERVED, "#0b2e5c"])
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), facecolor=SURFACE)
    for ax, p in zip(axes.flat, plats):
        xy = np.array(s["coords"][p])
        ax.hexbin(xy[:, 0], xy[:, 1], gridsize=(120, 50), extent=(-180, 180, -60, 80), mincnt=1,
                  cmap=cmap, linewidths=0, norm=LogNorm())
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 80)
        ax.grid(True, color=GRID, linewidth=0.5)
        _style(ax, f"{p} ({len(xy):,} records)")
    fig.suptitle("Where the records are: density of 1 km coordinates (log scale), longitude × latitude",
                 fontsize=10, color=INK_2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "spec11_map.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), facecolor=SURFACE, sharex=True)
    for ax, p in zip(axes.flat, plats):
        row = months.loc[p]
        ax.bar(range(12), 100 * row / row.sum(), color=OBSERVED, width=0.8)
        ax.set_xticks(range(12), [m[0] for m in months.columns])
        ax.set_ylabel("% of records", fontsize=7, color=INK_2)
        ax.grid(True, axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        _style(ax, p)
    fig.suptitle("Records per month (calendar month, both hemispheres)", fontsize=10, color=INK_2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "spec11_months.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)

    users = [p for p in plats if s["with_user"][p] >= MIN_USER_RECORDS]
    fig, axes = plt.subplots(1, len(users), figsize=(2.8 * len(users), 3.2), facecolor=SURFACE, squeeze=False)
    for ax, p in zip(axes[0], users):
        x = np.sort(np.array(list(s["users"][p].values()), float))
        cum = np.concatenate([[0], np.cumsum(x) / x.sum()])
        share = np.linspace(0, 1, len(cum))
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
        ax.plot(share, cum, color=OBSERVED, linewidth=2)
        ax.set_xlabel("share of users", fontsize=7, color=INK_2)
        ax.set_ylabel("share of records", fontsize=7, color=INK_2)
        _style(ax, f"{p}: Gini {gini(x):.2f}")
    fig.suptitle("Lorenz curves of records per user (diagonal = everyone contributes equally)", fontsize=10,
                 color=INK_2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "spec11_lorenz.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    s = collect(read_jsonl(corpus_path()))
    gold_langs = Counter(g["language"] or "-" for g in read_jsonl(GOLD) if g["set"] == "gold")
    t = tables(s, gold_langs)
    out = RESULTS / "tables"
    for name, df in t.items():
        df.round(2).to_csv(out / f"{name}.csv")
        (out / f"{name}.md").write_text(df.to_markdown(floatfmt=".1f"), encoding="utf-8")
        print(f"\n## {name}\n{df.round(2).to_string()}")
    figures(s, t["spec11_months"])
    print(f"\n-> {out}/spec11_*.md, results/figures/spec11_{{map,months,lorenz}}.png")


if __name__ == "__main__":
    main()
