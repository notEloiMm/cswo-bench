"""Translation hand-off to Colab (spec 2): export the non-English notes, then merge the translations back.

  uv run python -m cswo.harmonize.translate export   -> data/interim/to_translate.jsonl (id, lang, text only)
  ... run notebooks/colab_translate.ipynb on a GPU     -> translations.jsonl
  uv run python -m cswo.harmonize.translate merge <translations.jsonl>  -> data/processed/cswo_all.jsonl

The export holds no coordinates or user hashes, so only note text leaves the machine.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from cswo import INTERIM, PROCESSED
from cswo.collect.common import read_jsonl, write_jsonl
from cswo.harmonize.mt import qc_flag

HARMONIZED = INTERIM / "cswo_harmonized.jsonl"
EXPORT = INTERIM / "to_translate.jsonl"
OUT = PROCESSED / "cswo_all.jsonl"
UNTRANSLATED = "untranslated:und"     # language could not be identified (codes, numbers, species names)


def export() -> None:
    rows = ({"id": r["id"], "lang": r["language"], "text": r["text_original"]}
            for r in read_jsonl(HARMONIZED)
            if r.get("text_original") and r.get("language") not in (None, "en", "und"))
    n = write_jsonl(EXPORT, rows)
    langs = Counter(r["lang"] for r in read_jsonl(EXPORT))
    print(f"{n} notes -> {EXPORT}\n{dict(langs.most_common())}")


def merge(path: Path) -> None:
    tr = {r["id"]: r for r in read_jsonl(path)}
    stats = Counter()

    def rows():
        for r in read_jsonl(HARMONIZED):
            if r.get("text_original") and r.get("language") not in (None, "en"):
                t = tr.get(r["id"])
                if t and t.get("text_en"):
                    r["translation_model"] = t["model"]
                    flag = qc_flag(r["text_original"], t["text_en"])
                    # Translations still failing QC after the repair pass are mostly language-ID errors on very
                    # short inputs ("Lemna minor" tagged Romanian -> EU regulation text): drop them, keep the reason.
                    r["text_en"] = None if flag else t["text_en"]
                    r["translation_qc"] = f"rejected:{flag}" if flag else "ok"
                    stats["translated"] += 1
                    stats[f"qc_{flag or 'ok'}"] += 1
                    stats["repaired"] += bool(t.get("repaired"))
                elif r.get("language") == "und":
                    r["translation_model"] = UNTRANSLATED
                    stats["und"] += 1
                else:
                    stats["missing"] += 1
            yield r

    n = write_jsonl(OUT, rows())
    print(f"{n} records -> {OUT}; {dict(stats)}")
    if stats["missing"]:
        print(f"WARNING: {stats['missing']} non-English notes have no translation; rerun the notebook to resume.")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export")
    m = sub.add_parser("merge")
    m.add_argument("translations", type=Path)
    args = ap.parse_args()
    export() if args.cmd == "export" else merge(args.translations)


if __name__ == "__main__":
    main()
