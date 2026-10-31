"""Spec 6: draw the shared stratified gold sample and assign it to annotators.

Design follows the course annotation guideline v0.1 (group decision 2026-10-25):
* 20 calibration records, labelled by every student first (guideline "Quality checks");
* 20 reserve calibration records, used only if kappa on the first round is below 0.6 (second round);
* 300 gold records (n_students x 100). Each record is labelled by one student and by one LLM: students get
  disjoint sets of 100, the LLM ("LLM") labels all 300 in the original language. The LLM reads every language,
  so the guideline's original-reader vs translation-reader pairing arises wherever a student reads the MT.

Pool: records with an original note of >= 3 words (EyeOnWater has no text). Strata: platform x language group x
has-photo. Platforms get fixed quotas (so 26k Swedish notes do not swamp the gold set); inside a platform the
quota is split proportionally over language group x has-photo. Selection inside a stratum follows a hash of the
record id, so the sample is reproducible and independent of file order.

Annotator ids are anonymous (A1..An); the name mapping lives in data/gold/annotators.local.yaml (gitignored).
Run: uv run python -m cswo.annotate.sample
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict

from cswo import DATA
from cswo.collect.common import read_jsonl, write_jsonl
from cswo.harmonize.translate import OUT as CORPUS

GOLD_DIR = DATA / "gold"
ANNOTATORS = ["A1", "A2", "A3"]           # students (anonymous ids)
LLM_ANNOTATOR = "LLM"                      # second annotator on every gold record (guideline v0.1)
PER_ANNOTATOR = 100                        # spec 6: n students -> n x 100 gold records
# Languages each student reads well enough (guideline v0.1, "Languages and translation"); used to route notes
# to a student who can read the original. Everyone reads English.
STUDENT_LANGUAGES = {"A1": {"en", "de"},
                     "A2": {"en", "es", "ca", "pt", "it", "fr"},
                     "A3": {"en", "es", "ca", "pt", "it", "fr"}}
GUIDELINE_VERSION = "v0.1"
MIN_WORDS = 3
GOLD_QUOTA = {"inaturalist": 80, "crowdwater": 80, "artportalen": 55, "laji_fi": 55, "gbif_other": 30}
CALIB_QUOTA = {p: 4 for p in GOLD_QUOTA}
TOP_LANGS = 4                             # per platform; rarer languages form one "other" group


def _h(record_id: str, salt: str) -> str:
    return hashlib.sha1(f"cswo-gold|{salt}|{record_id}".encode()).hexdigest()


def _allocate(sizes: dict, total: int) -> dict:
    """Largest-remainder proportional allocation of ``total`` over strata of the given sizes."""
    n = sum(sizes.values())
    exact = {k: total * v / n for k, v in sizes.items()}
    alloc = {k: min(int(x), sizes[k]) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - int(exact[k]), reverse=True):
        if sum(alloc.values()) >= total:
            break
        if alloc[k] < sizes[k]:
            alloc[k] += 1
    return alloc


def pool() -> list[dict]:
    rows = [r for r in read_jsonl(CORPUS)
            if r["source_platform"] in GOLD_QUOTA and len((r.get("text_original") or "").split()) >= MIN_WORDS]
    top = defaultdict(set)
    for p in GOLD_QUOTA:
        langs = Counter(r.get("language") for r in rows if r["source_platform"] == p)
        top[p] = {lg for lg, _ in langs.most_common(TOP_LANGS)}
    for r in rows:
        lang = r.get("language") if r.get("language") in top[r["source_platform"]] else "other"
        r["_stratum"] = f"{r['source_platform']}|{lang}|{'photo' if r.get('media') else 'nophoto'}"
    return rows


def draw(rows: list[dict], quota: dict, salt: str, exclude: set[str]) -> list[dict]:
    picked = []
    for platform, q in quota.items():
        strata = defaultdict(list)
        for r in rows:
            if r["source_platform"] == platform and r["id"] not in exclude:
                strata[r["_stratum"]].append(r)
        for stratum, k in _allocate({s: len(v) for s, v in strata.items()}, q).items():
            picked += sorted(strata[stratum], key=lambda r: _h(r["id"], salt))[:k]
    return picked


def assign(gold: list[dict], students: list[str], languages: dict[str, set[str]] | None = None,
           capacity: int = PER_ANNOTATOR) -> dict[str, str]:
    """Give every gold record to one student, ``capacity`` records each.

    Records in a language only some students read (German -> A1, Portuguese -> A2/A3) go to those students
    first, so more notes are read in the original and get entity spans. The rest are dealt in stratum order to
    the least-loaded student, which keeps the platform mix similar across students.
    """
    languages = languages or {}
    ordered = sorted(gold, key=lambda r: (r["_stratum"], _h(r["id"], "assign")))
    load = Counter({s: 0 for s in students})
    out: dict[str, str] = {}

    def give(r: dict, candidates: list[str]) -> bool:
        free = [s for s in candidates if load[s] < capacity]
        if not free:
            return False
        s = min(free, key=lambda s: (load[s], students.index(s)))
        out[r["id"]] = s
        load[s] += 1
        return True

    for r in ordered:
        readers = [s for s in students if r.get("language") in languages.get(s, set())]
        if readers and len(readers) < len(students):
            give(r, readers)
    for r in ordered:
        if r["id"] not in out:
            give(r, students)
    return out


def main() -> None:
    if len(ANNOTATORS) * PER_ANNOTATOR != sum(GOLD_QUOTA.values()):
        raise SystemExit("GOLD_QUOTA must sum to n_students x 100")
    rows = pool()
    calib = draw(rows, CALIB_QUOTA, "calibration", set())
    gold = draw(rows, GOLD_QUOTA, "gold", {r["id"] for r in calib})
    reserve = draw(rows, CALIB_QUOTA, "calibration_reserve", {r["id"] for r in calib + gold})
    student_of = assign(gold, ANNOTATORS, STUDENT_LANGUAGES)
    entry = lambda r, kind, who: {"id": r["id"], "set": kind, "stratum": r["_stratum"], "annotators": who,
                                  "guideline": GUIDELINE_VERSION}
    manifest = ([entry(r, "calibration", ANNOTATORS) for r in calib]
                + [entry(r, "calibration_reserve", ANNOTATORS) for r in reserve]
                + [entry(r, "gold", [student_of[r["id"]], LLM_ANNOTATOR]) for r in gold])
    write_jsonl(GOLD_DIR / "gold_manifest.jsonl", manifest)

    active = [m for m in manifest if m["set"] != "calibration_reserve"]
    load = Counter(a for m in active for a in m["annotators"])
    print(f"calibration {len(calib)} (+{len(reserve)} reserve), gold {len(gold)}; tasks per annotator: {dict(load)}")
    print("gold per platform:", dict(Counter(m["stratum"].split("|")[0] for m in manifest if m["set"] == "gold")))
    print("gold with photo:", sum("|photo" in m["stratum"] for m in manifest if m["set"] == "gold"))
    print("gold per student:", dict(Counter(m["annotators"][0] for m in manifest if m["set"] == "gold")))
    lang = {r["id"]: r.get("language") for r in gold}
    readable = sum(lang[m["id"]] in STUDENT_LANGUAGES[m["annotators"][0]] | {"und"}
                   for m in manifest if m["set"] == "gold")
    print(f"gold notes the assigned student can read in the original (incl. und): {readable}/300")
    print(f"-> {GOLD_DIR / 'gold_manifest.jsonl'}")


if __name__ == "__main__":
    main()
