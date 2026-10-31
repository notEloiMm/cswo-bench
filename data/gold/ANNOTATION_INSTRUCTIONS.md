# Gold-set annotation: how to do it

**The rules for labelling are in the course guideline: "CSWO-Bench: Text Annotation Guideline v0.1"
(Moinul Islam).** Read it fully before starting. This file only covers the setup in this repository.

## Design (guideline v0.1)

| | Records | Labelled by |
|---|---|---|
| Calibration round | 20 | all three students, **first** |
| Calibration round 2 (reserve) | 20 | all three students, **only if** calibration kappa on conditions is below 0.6 |
| Gold set | 300 | one student (100 each, disjoint) **and** one LLM (all 300, original language) |
| **Per student** | **120 tasks** (20 calibration + 100 gold), ~8 h at 4 min per record | |

Each of us has an anonymous ID (A1, A2, A3). Agree who is who and write it in `data/gold/annotators.local.yaml`
(copy `annotators.local.example.yaml`; the file is gitignored). Before starting, each of us lists the
languages we can read well enough (guideline, "Languages and translation"); put them in the same file.

Gold records are routed by language (`STUDENT_LANGUAGES` in `src/cswo/annotate/sample.py`): German notes go to A1,
Spanish/French/Portuguese/Italian notes to A2 and A3, so 162 of the 300 gold notes are read in the original.
Swedish, Finnish, Danish and Estonian notes are read through the translation by whoever gets them. Notes marked
`und` (counts, codes, Latin names) usually need no language: read them as `original`.

After the calibration round we meet, compare labels, and resolve every disagreement by adding a row to
`data/gold/hard_cases.md`. Only then do we start the gold tasks.

## 1. Start Label Studio (on your own computer)

```bash
uvx --python 3.11 label-studio start
```

(Tested with Python 3.11. Without uv: `pip install label-studio` in a Python 3.11 environment, then
`label-studio start`.) It opens at http://localhost:8080; create your own local login there. Stop it with
Ctrl+C; your work is saved locally.

## 2. Create your project

1. **Create Project** → name it `CSWO gold <your ID>`, e.g. `CSWO gold A2`.
2. **Labeling Setup → Custom template → Code**: paste `src/cswo/annotate/labelstudio_config.xml`, Save.
3. **Data Import**: upload **your own file only**: `data/gold/labelstudio/tasks_<your ID>.json` (create it with `uv run python -m cswo.annotate.tasks`)
   (the 20 calibration tasks come first). Import `calibration2_<your ID>.json` only if the group runs round 2.

## 3. Label each task (order from the guideline)

| Step | In Label Studio |
|---|---|
| 1. read_from | `original` or `translation` |
| 2. Relevance | `about water` / `not about water` / `uncertain`. If `not about water`, stop: the next questions disappear. |
| 3. Water condition | tick every asserted condition, or **`none stated`**: never leave it empty on an about-water note (hard case 7) |
| 4. Entity spans | only if read_from = `original`: select words in the original note and pick a type |
| 5. Image-text consistency | only when a photo is shown and the note is about water; leave empty for "No photo for this record" |
| 6. Confidence | tick `uncertain` when unsure about any label and write one line why in the comment |

Notes specific to our data: the auto-detected language in the header can be wrong on very short notes;
`[PERSON]` in a note is a masked personal name (treat it as text, it is never an entity); a translation shown as
"(no machine translation: rejected:…)" failed our quality check, so read the original or mark the record
uncertain.

## 4. Export

**Export → JSON** (not JSON-MIN) and save it as `data/gold/exports/<your ID>.json`, e.g. `A2.json`. Export once
after calibration and again at the end. The exports are committed to the repository: Label Studio stores annotators
as numeric ids, but an unsaved **draft** can carry your Label Studio e-mail, so submit or discard drafts before
exporting (or ask whoever commits to strip it). The merge step (spec 7) converts the exports to the guideline's
JSONL format (record_id, anonymised annotator_id, read_from, relevance, conditions, entities, image_text,
uncertain, comment, guideline version).

**After the calibration round (2026-10-26):** paste the updated `labelstudio_config.xml` into your project
(Settings → Labeling Interface → Code → Save) before continuing; it hides image-text on notes that are not about
water. Read the proposed rulings in `hard_cases.md` and apply them to the gold tasks you have already done.

## 5. After calibration: agreement and the resolution meeting (spec 7)

```bash
uv run python -m cswo.annotate.merge        # exports/*.json -> data/gold/annotations.jsonl
uv run python -m cswo.annotate.agreement    # kappa + span F1 tables, data/gold/disagreements_calibration.csv
```

The agreement step prints whether the conditions kappa is below 0.6 (then import `calibration2_<ID>.json` and
run round 2). Open `disagreements_calibration.csv` in the meeting: one row per record and field where we differ.
For each row write the ruling in `resolved`, either an annotator id (`A2` = take A2's labels) or a value in the
sheet's own format (`algae; turbid`, `none stated`, `WATER_BODY[12:20] Borssjön; QUANTITY[0:1] 2`, `(none)`).
Put `yes` in `hard_case` when the ruling is worth recording, and add those to `hard_cases.md`. Re-running the
agreement step keeps the rulings already written.

After the gold round (students + `exports/LLM.jsonl`), run the same two steps again, resolve
`disagreements_gold.csv`, then:

```bash
uv run python -m cswo.annotate.adjudicate   # -> data/gold/gold.jsonl (unresolved fields are listed)
uv run python -m cswo.extract.evaluate      # extractor P/R/F1, English original vs MT
```
