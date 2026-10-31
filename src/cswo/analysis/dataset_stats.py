"""Benchmark statistics for the report appendix (asked for in the project review, 2026-10-28).

Tables (results/tables/stats_*):
* water_body_by_platform: records per water-body type (lake, pond, river, stream, wetland, coast, sea, unknown) per
  platform, counts and row shares. The type comes from the record's locality/habitat/note keywords or the platform
  (`water_evidence`); EyeOnWater publishes no locality type, so it is all "unknown".
* water_evidence_by_platform: how each record entered the corpus (platform = water-only platform, keyword = a water
  word in locality/habitat/note, taxon = obligately aquatic species).
* years_by_platform: records per year (older years pooled).
* labels: the structured labels per platform: CrowdWater water-level classes, EyeOnWater Forel-Ule scale and bins,
  habitat coverage and its most frequent values, species richness.
* translation_models: per note language, the translation model, notes translated, rejected by QC, untranslated.
Figure: results/figures/stats_water_body_by_platform.png (share of each type per platform).
Run: uv run python -m cswo.analysis.dataset_stats
"""
from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from cswo import RESULTS
from cswo.analysis.text import corpus_path
from cswo.baselines.tasks import normalise_label
from cswo.collect.common import read_jsonl

WATER_TYPES = ["lake", "pond", "river", "stream", "wetland", "coast", "sea", "unknown"]
PLATFORMS = ["inaturalist", "crowdwater", "eyeonwater", "artportalen", "laji_fi", "gbif_other"]
FIRST_YEAR = 2015          # earlier records are pooled
TOP_VALUES = 10


def year_bucket(ts: str | None) -> str:
    if not ts or not ts[:4].isdigit():
        return "unknown"
    y = int(ts[:4])
    return f"before {FIRST_YEAR}" if y < FIRST_YEAR else str(y)


def _shares(counts: pd.DataFrame) -> pd.DataFrame:
    """Counts table -> the same table with a row-share (%) table appended column-wise for each column."""
    pct = counts.div(counts.sum(axis=1), axis=0).mul(100)
    out = counts.copy()
    out["total"] = counts.sum(axis=1)
    for c in counts.columns:
        out[f"{c} %"] = pct[c]
    return out


def collect(records) -> dict:
    s = {"water": Counter(), "evidence": Counter(), "years": Counter(), "level": Counter(), "fu": Counter(),
         "fu_bin": Counter(), "habitat_n": Counter(), "habitat": defaultdict(Counter), "species": defaultdict(Counter),
         "records": Counter(), "secchi": 0, "temperature": 0, "mt": defaultdict(Counter),
         "mt_model": defaultdict(Counter)}
    for r in records:
        p, lab = r["source_platform"], r.get("labels") or {}
        s["records"][p] += 1
        s["water"][(p, r.get("water_body_type") or "unknown")] += 1
        s["evidence"][(p, (r.get("water_evidence") or "none").split(":")[0])] += 1
        s["years"][(p, year_bucket(r.get("timestamp")))] += 1
        if lab.get("water_level_class") is not None:
            s["level"][str(lab["water_level_class"])] += 1
        if lab.get("forel_ule") is not None:
            s["fu"][int(lab["forel_ule"])] += 1
            s["fu_bin"][lab.get("forel_ule_bin3")] += 1
        s["secchi"] += lab.get("secchi_depth_m") is not None
        s["temperature"] += lab.get("temperature_c") is not None
        if lab.get("habitat"):
            s["habitat_n"][p] += 1
            s["habitat"][p][normalise_label("habitat", lab["habitat"])] += 1
        if lab.get("species"):
            s["species"][p][lab["species"]] += 1
        if r.get("text_original"):
            lang, model, qc = r.get("language") or "-", r.get("translation_model") or "", r.get("translation_qc") or ""
            m = s["mt"][lang]
            m["notes"] += 1
            if lang == "en":
                m["original English"] += 1
            elif model.startswith("untranslated") or not model:
                m["untranslated"] += 1
            else:
                s["mt_model"][lang][model.split("/")[-1]] += 1
                m["rejected by QC" if str(qc).startswith("rejected") else "translated"] += 1
    return s


def tables(s: dict) -> dict[str, pd.DataFrame]:
    plats = [p for p in PLATFORMS if s["records"][p]]
    water = pd.DataFrame([[s["water"][(p, t)] for t in WATER_TYPES] for p in plats], index=plats, columns=WATER_TYPES)
    water.loc["all"] = water.sum()
    ev_cols = sorted({k for _, k in s["evidence"]})
    evidence = pd.DataFrame([[s["evidence"][(p, k)] for k in ev_cols] for p in plats], index=plats, columns=ev_cols)
    evidence.loc["all"] = evidence.sum()
    year_cols = sorted({y for _, y in s["years"]}, key=lambda y: (y == "unknown", not y.startswith("before"), y))
    years = pd.DataFrame([[s["years"][(p, y)] for y in year_cols] for p in plats], index=plats, columns=year_cols)
    years.loc["all"] = years.sum()

    rows = []
    for c, n in sorted(s["level"].items(), key=lambda kv: int(kv[0])):
        rows.append({"label": "CrowdWater water-level class", "value": c, "records": n})
    for c, n in sorted(s["fu"].items()):
        rows.append({"label": "EyeOnWater Forel-Ule scale", "value": str(c), "records": n})
    for c in ("blue", "green", "brown"):
        rows.append({"label": "EyeOnWater Forel-Ule bin", "value": c, "records": s["fu_bin"][c]})
    rows += [{"label": "EyeOnWater measurement", "value": "Secchi depth", "records": s["secchi"]},
             {"label": "EyeOnWater measurement", "value": "water temperature", "records": s["temperature"]}]
    for p in plats:
        if s["habitat_n"][p]:
            rows.append({"label": f"{p} habitat (records with the field)", "value": "any", "records": s["habitat_n"][p]})
            rows += [{"label": f"{p} habitat (top {TOP_VALUES})", "value": v, "records": n}
                     for v, n in s["habitat"][p].most_common(TOP_VALUES)]
    for p in plats:
        if s["species"][p]:
            rows.append({"label": f"{p} species", "value": "distinct species", "records": len(s["species"][p])})
    labels = pd.DataFrame(rows).set_index(["label", "value"])

    mt = pd.DataFrame(s["mt"]).T.fillna(0).astype(int).sort_values("notes", ascending=False)
    mt = mt[[c for c in ("notes", "original English", "translated", "rejected by QC", "untranslated") if c in mt]]
    mt["model"] = [", ".join(m for m, _ in s["mt_model"][lang].most_common()) or "-" for lang in mt.index]
    return {"stats_water_body_by_platform": _shares(water), "stats_water_evidence_by_platform": _shares(evidence),
            "stats_years_by_platform": years, "stats_labels": labels, "stats_translation_models": mt}


def figure(water: pd.DataFrame, path) -> None:
    """Heatmap of the share of each water-body type per platform: one sequential hue, values printed in the cells."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from cswo.analysis.zipf_heaps import GRID, INK, INK_2, OBSERVED, SURFACE

    pct = water[[f"{t} %" for t in WATER_TYPES]]
    cmap = LinearSegmentedColormap.from_list("seq", [SURFACE, OBSERVED])
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=SURFACE)
    ax.imshow(pct.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            v = pct.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color=SURFACE if v > 55 else INK)
    ax.set_xticks(range(len(WATER_TYPES)), WATER_TYPES, fontsize=8, color=INK_2)
    ax.set_yticks(range(len(pct)), [f"{p} ({int(n):,})" for p, n in zip(pct.index, water["total"])], fontsize=8,
                  color=INK_2)
    ax.set_title("Share of records per water-body type (%), by platform", fontsize=10, color=INK, loc="left")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    t = tables(collect(read_jsonl(corpus_path())))
    out = RESULTS / "tables"
    for name, df in t.items():
        df.round(1).to_csv(out / f"{name}.csv")
        (out / f"{name}.md").write_text(df.to_markdown(floatfmt=".1f"), encoding="utf-8")
        print(f"\n## {name}\n{df.round(1).to_string()}")
    figure(t["stats_water_body_by_platform"], RESULTS / "figures" / "stats_water_body_by_platform.png")
    print(f"\n-> {out}/stats_*.md, results/figures/stats_water_body_by_platform.png")


if __name__ == "__main__":
    main()
