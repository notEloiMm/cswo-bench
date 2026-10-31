"""Fixed train/validation/test splits of the corpus (spec 8, released with the corpus in spec 12).

80/10/10 of the records with text, stratified by platform. Two constraints:
* repeated observations of one site (CrowdWater ROOT_ID, labels.series_id) stay in one split, otherwise a model
  is tested on a gauge whose earlier notes it was trained on;
* every gold record (all 340 in data/gold/gold_manifest.jsonl, with its whole series) is in test, so the gold set
  stays held out for the zero-shot LLM/VLM comparison (spec 10); except gold records with a field the adjudicator
  marked unresolvable, which the guideline excludes from test (they go to train/validation).
Inside a platform, groups (a series, or a single record) are ordered by a hash of their id and dealt into the
splits until each reaches its share of the platform's records, so the split is reproducible and independent of
file order. Records without text are not split (no text task can use them).
Known deviation (kept by group decision 2026-10-26): series with gold records go to test whole, which makes
CrowdWater 64/8/28 instead of 80/10/10; breaking series would leak gauges between train and test.
Output: data/splits.csv (id, split). Commit it: the splits are part of the benchmark.
Run: uv run python -m cswo.baselines.splits
"""
from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict

from cswo import DATA
from cswo.annotate.sample import GOLD_DIR
from cswo.collect.common import read_jsonl
from cswo.harmonize.translate import OUT as CORPUS

SPLITS = DATA / "splits.csv"
FRACTIONS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def _h(key: str) -> str:
    return hashlib.sha1(f"cswo-split|{key}".encode()).hexdigest()


def group_of(r: dict) -> str:
    sid = (r.get("labels") or {}).get("series_id")
    return f"{r['source_platform']}|series|{sid}" if sid else r["id"]


def make_splits(records: list[dict], held_out: set[str], not_test: set[str] = frozenset()) -> dict[str, str]:
    """id -> train / validation / test. ``not_test``: records the guideline keeps out of test (gold records an
    adjudicator marked unresolvable); their groups are dealt to train/validation only."""
    groups = defaultdict(list)
    for r in records:
        groups[group_of(r)].append(r)
    out = {}
    by_platform = defaultdict(list)
    for g, rows in groups.items():
        if any(r["id"] in held_out for r in rows) and not any(r["id"] in not_test for r in rows):
            out.update({r["id"]: "test" for r in rows})
        else:
            by_platform[rows[0]["source_platform"]].append(g)
    total = Counter(r["source_platform"] for r in records)
    for platform, gs in by_platform.items():
        filled = Counter(s for i, s in out.items() if i.split(":")[0] == platform)
        for g in sorted(gs, key=_h):
            # the split furthest below its target share takes the next group
            allowed = [s for s in FRACTIONS if not (s == "test" and any(r["id"] in not_test for r in groups[g]))]
            split = min(allowed, key=lambda s: filled[s] / (FRACTIONS[s] * total[platform]))
            out.update({r["id"]: split for r in groups[g]})
            filled[split] += len(groups[g])
    return out


def load_splits() -> dict[str, str]:
    if not SPLITS.exists():
        raise SystemExit(f"{SPLITS} missing: run uv run python -m cswo.baselines.splits")
    with SPLITS.open(encoding="utf-8", newline="") as f:
        return {r["id"]: r["split"] for r in csv.DictReader(f)}


def main() -> None:
    records = [r for r in read_jsonl(CORPUS) if (r.get("text_original") or "").strip()]
    held_out = {r["id"] for r in read_jsonl(GOLD_DIR / "gold_manifest.jsonl")}
    gold = GOLD_DIR / "gold.jsonl"
    not_test = {g["record_id"] for g in read_jsonl(gold) if g.get("unresolvable")} if gold.exists() else set()
    splits = make_splits(records, held_out, not_test)
    with SPLITS.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "split"])
        w.writerows(sorted(splits.items()))
    table = Counter((i.split(":")[0], s) for i, s in splits.items())
    for p in sorted({p for p, _ in table}):
        n = sum(table[(p, s)] for s in FRACTIONS)
        print(f"{p:12s} " + "  ".join(f"{s} {table[(p, s)]:6d} ({table[(p, s)] / n:.1%})" for s in FRACTIONS))
    print(f"-> {SPLITS}")


if __name__ == "__main__":
    main()
