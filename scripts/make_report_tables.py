"""Write the report's appendix tables (report/tables/*.tex, booktabs) from results/tables/*.csv, so the numbers in
the report cannot drift from the results. Run after the analyses: uv run python scripts/make_report_tables.py
"""
import pandas as pd

from cswo import RESULTS, ROOT

SRC, OUT = RESULTS / "tables", ROOT / "report" / "tables"


def esc(s) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#"))


def fmt(v, spec: str) -> str:
    if pd.isna(v):
        return "--"
    if spec == "int":
        return f"{int(round(float(v))):,}".replace(",", "{,}")
    return f"{float(v):{spec}}"


def write(name: str, df: pd.DataFrame, cols: dict[str, tuple[str, str]], caption: str, label: str,
          index_name: str = "", small: bool = True, wide: bool = False) -> None:
    """cols: source column -> (header, format); wide = scale the table to the text width (one-column article)."""
    head = [index_name] + [h for h, _ in cols.values()]
    env = "table"
    lines = [rf"\begin{{{env}}}[htbp]", r"\centering", r"\small" if small else "", rf"\caption{{{caption}}}",
             rf"\label{{{label}}}", r"\resizebox{\linewidth}{!}{%" if wide else "",
             r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             " & ".join(esc(h) for h in head) + r" \\", r"\midrule"]
    for idx, row in df.iterrows():
        cells = [esc(" / ".join(map(str, idx)) if isinstance(idx, tuple) else idx)]
        cells += [fmt(row[c], f) if f != "str" else esc(row[c]) for c, (_, f) in cols.items()]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "}" if wide else "", rf"\end{{{env}}}"]
    (OUT / f"{name}.tex").write_text("\n".join(l for l in lines if l) + "\n", encoding="utf-8")


def csv(name: str, index_col=0) -> pd.DataFrame:
    return pd.read_csv(SRC / f"{name}.csv", index_col=index_col)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s3 = csv("spec3_stats_by_platform")
    write("spec3_platform", s3, {"notes": ("notes", "int"), "tokens": ("tokens", "int"),
                                 "vocabulary": ("types", "int"), "vocab@25k": ("types@25k", "int"),
                                 "hapax %": ("hapax %", ".0f"), "mean len": ("mean len", ".1f"),
                                 "median len": ("median", ".0f"), "numbers %": ("numbers %", ".1f"),
                                 "WordNet %": ("WordNet %", ".0f")},
          "Corpus statistics per platform and reference corpora (spec 3). Lengths in tokens; Brown and Reuters "
          "measured per sentence.", "tab:spec3", "platform", wide=True)
    for by in ("platform", "language"):
        z = csv(f"spec4_zipf_by_{by}")
        write(f"zipf_{by}", z, {"types": ("types", "int"), "slope": ("slope", ".3f"),
                                "slope 90% lo": ("90% lo", ".3f"), "slope 90% hi": ("90% hi", ".3f"),
                                "R2": ("$R^2$", ".3f"), "adj R2": ("adj. $R^2$", ".3f"),
                                "slope top1000": ("slope top-1000", ".3f")},
              f"Zipf fits by {by} (spec 4): OLS of $\\log_{{10}}$ frequency on $\\log_{{10}}$ rank with 90\\% "
              "confidence interval of the slope.", f"tab:zipf-{by}", by, wide=True)
        h = csv(f"spec4_heaps_by_{by}")
        write(f"heaps_{by}", h, {"tokens": ("tokens", "int"), "types": ("types", "int"), "K": ("$K$", ".2f"),
                                 "beta": ("$\\beta$", ".3f"), "beta 90% lo": ("90% lo", ".3f"),
                                 "beta 90% hi": ("90% hi", ".3f"), "R2": ("$R^2$", ".3f")},
              f"Heaps fits by {by} (spec 4): $V = K N^\\beta$ on vocabulary sizes at every 5\\% of the tokens.",
              f"tab:heaps-{by}", by)
    inc = csv("spec4_heaps_increments_by_platform")
    write("heaps_increments", inc, {c: (c, "int") for c in ("5%", "10%", "25%", "50%", "75%", "100%")}
          | {"last / first step": ("last/first", ".2f")},
          "New word types added in selected 5\\% steps of tokens, and the ratio of the last to the first step: "
          "growth slows down, much less for field notes than for edited text.", "tab:heaps-inc", "")
    w = csv("stats_water_body_by_platform")
    write("water_body", w, {f"{t} %": (t, ".0f") for t in
                            ("lake", "pond", "river", "stream", "wetland", "coast", "sea", "unknown")}
          | {"total": ("records", "int")},
          "Share of records (\\%) per water-body type, by platform.", "tab:water-body", "platform", wide=True)
    y = csv("stats_years_by_platform")
    write("years", y, {c: (c.replace("before ", "<"), "int") for c in y.columns}, "Records per year.",
          "tab:years", "platform", wide=True)
    mt = csv("stats_translation_models")
    write("translation", mt, {"notes": ("notes", "int"), "translated": ("translated", "int"),
                              "rejected by QC": ("rejected", "int"), "untranslated": ("untranslated", "int"),
                              "model": ("model", "str")},
          "Translation per note language: model, notes translated, rejected by the quality check, untranslated "
          "(English originals are not translated).", "tab:translation", "language")
    u = csv("spec11_users")
    write("users", u, {"records with user id %": ("with id %", ".0f"), "users": ("users", "int"),
                       "records per user (median)": ("median", ".0f"), "top 1% users share %": ("top 1% %", ".0f"),
                       "top 10% users share %": ("top 10% %", ".0f"), "Gini": ("Gini", ".2f")},
          "Records per user (spec 11). CrowdWater publishes no user id.", "tab:users", "platform")
    s = csv("spec11_seasons")
    write("seasons", s, {c: (c, ".0f") for c in s.columns},
          "Records per meteorological season in the observer's hemisphere (\\%).", "tab:seasons", "platform")
    e = csv("spec13_entities_per_type", index_col=[0, 1])
    write("entities_per_type", e, {"F1 exact": ("F1 exact", ".2f"), "F1 partial": ("F1 partial", ".2f"),
                                   "P partial": ("P", ".2f"), "R partial": ("R", ".2f"),
                                   "predicted": ("pred.", "int"), "gold spans": ("gold", "int")},
          "Entity extraction per system and type on the gold set (specs 7 and 13).", "tab:entities-type",
          "system / type")
    print(f"tables -> {OUT}")


if __name__ == "__main__":
    main()
