"""Spec 3: corpus statistics per platform and per language, compared with Brown and Reuters.

Per group: notes, tokens, vocabulary, vocabulary in a fixed 25k-token sample (raw vocabulary grows with corpus
size, so groups are only comparable at equal size), hapax share, mean/SD/median note length in tokens, share of
number tokens, and the share of tokens with a WordNet (English) or FinnWordNet (Finnish) entry.
Brown and Reuters are measured per sentence, the unit closest to a short field note.
Run: uv run python -m cswo.analysis.corpus_stats
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from cswo import RESULTS
from cswo.analysis.text import COVERAGE, count_numbers, notes, reference_corpora, tokenize

MIN_TOKENS = 5_000        # languages smaller than this are reported together as "other"
SAMPLE_TOKENS = 25_000    # Finnish has ~48k tokens; 25k keeps en, sv, fi, de comparable
SEED = 0


def _vocab_at(units: list[list[str]], n: int, seed: int = SEED) -> float:
    """Vocabulary size of the first n tokens after shuffling the notes (NaN if the group is smaller)."""
    order = units[:]
    random.Random(seed).shuffle(order)
    seen, total = set(), 0
    for u in order:
        for t in u:
            if total == n:
                return len(seen)
            seen.add(t)
            total += 1
    return len(seen) if total == n else float("nan")


def group_stats(units: list[list[str]], numbers: int, langs_of_units: list[str] | None = None) -> dict:
    """``units``: one token list per note (or sentence). ``langs_of_units`` picks the coverage resource."""
    lengths = np.array([len(u) for u in units])
    freq = Counter(t for u in units for t in u)
    n_tok = int(lengths.sum())
    row = {
        "notes": len(units), "tokens": n_tok, "vocabulary": len(freq),
        f"vocab@{SAMPLE_TOKENS // 1000}k": _vocab_at(units, SAMPLE_TOKENS),
        "hapax %": 100 * sum(1 for c in freq.values() if c == 1) / max(len(freq), 1),
        "mean len": lengths.mean(), "sd len": lengths.std(ddof=1) if len(units) > 1 else float("nan"),
        "median len": float(np.median(lengths)),
        "numbers %": 100 * numbers / max(numbers + n_tok, 1),
    }
    for lang, (name, fn) in COVERAGE.items():
        toks = [t for u, lg in zip(units, langs_of_units or [None] * len(units)) if lg == lang for t in u]
        row[f"{name} %"] = 100 * sum(map(fn, toks)) / len(toks) if toks else float("nan")
    return row


def compute() -> tuple[pd.DataFrame, pd.DataFrame]:
    by_platform, by_lang = defaultdict(list), defaultdict(list)
    for n in notes():
        toks = tokenize(n["text"])
        item = (toks, count_numbers(n["text"]), n["language"])
        by_platform[n["platform"]].append(item)
        by_lang[n["language"]].append(item)

    def table(groups: dict, key_name: str) -> pd.DataFrame:
        rows = {}
        for key, items in groups.items():
            units, nums, langs = zip(*items)
            rows[key] = group_stats(list(units), sum(nums), list(langs))
        return pd.DataFrame.from_dict(rows, orient="index").rename_axis(key_name)

    # fold small languages into "other" so every reported row has enough text to mean something
    sizes = {lg: sum(len(t) for t, _, _ in items) for lg, items in by_lang.items()}
    folded = defaultdict(list)
    for lg, items in by_lang.items():
        folded[lg if sizes[lg] >= MIN_TOKENS and lg not in ("-",) else "other"].extend(items)

    platform_t = table(by_platform, "platform").sort_values("tokens", ascending=False)
    lang_t = table(folded, "language").sort_values("tokens", ascending=False)
    ref = {}
    for name, sents in reference_corpora().items():
        ref[f"{name} (sentences)"] = group_stats([tokenize(s) for s in sents], sum(map(count_numbers, sents)),
                                                 ["en"] * len(sents))
    ref_t = pd.DataFrame.from_dict(ref, orient="index")
    return pd.concat([platform_t, ref_t]), pd.concat([lang_t, ref_t])


def main() -> None:
    out = RESULTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    platform_t, lang_t = compute()
    for name, t in (("spec3_stats_by_platform", platform_t), ("spec3_stats_by_language", lang_t)):
        t.round(2).to_csv(out / f"{name}.csv")
        (out / f"{name}.md").write_text(t.round(1).to_markdown(floatfmt=",.1f"), encoding="utf-8")
        print(f"\n## {name}\n{t.round(1).to_string()}")


if __name__ == "__main__":
    main()
