"""Spec 8 weak-label tasks: predict a platform's structured field from the note text alone.

| Task | Records | Label |
|---|---|---|
| water_level | CrowdWater with a note | labels.water_level_class (virtual staff gauge class, "-2" .. "+2") |
| forel_ule | EyeOnWater with a note | labels.forel_ule_bin3 (blue / green / brown) |
| habitat | Artportalen with a note | labels.habitat, lower-cased; the most frequent values only |

EyeOnWater publishes no notes (spec 2 table: 100% image only), so forel_ule has no text records; it is kept in
the task list so the report states that, rather than silently leaving it out.
Classes with fewer than MIN_CLASS records are dropped (a class needs train, validation and test examples), and
habitat keeps at most MAX_HABITATS values: Artportalen habitat is free text with a long tail.
Input text is text_original: the multilingual baselines read the note in its own language (spec 2 findings).
Stopword removal uses the NLTK list of the note's detected language; languages without a list (Estonian, und, ...)
are left unchanged.
Run: uv run python -m cswo.baselines.tasks export   -> results/tables/spec8_tasks.*, and
     data/interim/spec8_tasks.jsonl for the Colab notebook (`summary` writes the table only)
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from functools import lru_cache

import pandas as pd

from cswo import INTERIM, RESULTS
from cswo.baselines.splits import load_splits
from cswo.collect.common import read_jsonl, write_jsonl
from cswo.harmonize.translate import OUT as CORPUS

TASKS = {"water_level": ("crowdwater", "water_level_class"),
         "forel_ule": ("eyeonwater", "forel_ule_bin3"),
         "habitat": ("artportalen", "habitat")}
MIN_CLASS = 30
MAX_HABITATS = 10
EXPORT = INTERIM / "spec8_tasks.jsonl"

NLTK_LANG = {"en": "english", "sv": "swedish", "fi": "finnish", "de": "german", "da": "danish", "nl": "dutch",
             "fr": "french", "es": "spanish", "pt": "portuguese", "it": "italian", "no": "norwegian",
             "ru": "russian", "hu": "hungarian", "ro": "romanian", "sl": "slovene", "ca": "catalan"}
_WORD = re.compile(r"\w+")


@lru_cache(maxsize=None)
def stopwords(lang: str | None) -> frozenset[str]:
    if lang not in NLTK_LANG:
        return frozenset()
    import nltk
    try:
        from nltk.corpus import stopwords as sw
        words = sw.words(NLTK_LANG[lang])
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords as sw
        words = sw.words(NLTK_LANG[lang])
    return frozenset(words)


def remove_stopwords(text: str, lang: str | None) -> str:
    sw = stopwords(lang)
    if not sw:
        return text
    out = _WORD.sub(lambda m: "" if m.group().lower() in sw else m.group(), text)
    return re.sub(r"\s+", " ", out).strip()


def normalise_label(task: str, value) -> str | None:
    if value is None:
        return None
    v = re.sub(r"\s+", " ", str(value)).strip()
    if task == "habitat":
        v = v.lower().rstrip(" .,;:")          # "Sjö." and "sjö" are the same habitat
    return v or None


def build(records, splits: dict[str, str]) -> dict[str, list[dict]]:
    """task -> rows {id, task, split, language, text, text_nostop, label}."""
    out = {}
    for task, (platform, field) in TASKS.items():
        rows = []
        for r in records:
            if r["source_platform"] != platform or r["id"] not in splits:
                continue
            label = normalise_label(task, (r.get("labels") or {}).get(field))
            text = (r.get("text_original") or "").strip()
            if label and text:
                rows.append({"id": r["id"], "task": task, "split": splits[r["id"]], "language": r.get("language"),
                             "text": text, "label": label})
        counts = Counter(x["label"] for x in rows)
        keep = [c for c, n in counts.most_common() if n >= MIN_CLASS]
        if task == "habitat":
            keep = keep[:MAX_HABITATS]
        rows = [x for x in rows if x["label"] in set(keep)]
        for x in rows:
            x["text_nostop"] = remove_stopwords(x["text"], x["language"])
        out[task] = rows
    return out


def load(records=None) -> dict[str, list[dict]]:
    # a list, not the read_jsonl generator: build() loops over the records once per task
    return build(list(read_jsonl(CORPUS)) if records is None else list(records), load_splits())


def summary(tasks: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for task, xs in tasks.items():
        split = Counter(x["split"] for x in xs)
        # weak-label leakage check: how often the label is written in the note itself
        literal = sum(x["label"].lower() in x["text"].lower() for x in xs) / len(xs) if xs else 0
        rows.append({"task": task, "platform": TASKS[task][0], "records": len(xs),
                     "classes": len({x["label"] for x in xs}), **{s: split[s] for s in ("train", "validation", "test")},
                     "label in note": literal,
                     "class counts": ", ".join(f"{c} {n}" for c, n in Counter(x["label"] for x in xs).most_common())})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["summary", "export"])
    args = ap.parse_args()
    tasks = load()
    table = pd.DataFrame(summary(tasks)).set_index("task")
    table.to_csv(RESULTS / "tables" / "spec8_tasks.csv")
    (RESULTS / "tables" / "spec8_tasks.md").write_text(table.to_markdown(floatfmt=".2f"), encoding="utf-8")
    print(table.drop(columns="class counts").to_string())
    if args.command == "export":
        n = write_jsonl(EXPORT, (x for xs in tasks.values() for x in xs))
        print(f"{n} rows -> {EXPORT} (ids, language, note text and label only)")


if __name__ == "__main__":
    main()
