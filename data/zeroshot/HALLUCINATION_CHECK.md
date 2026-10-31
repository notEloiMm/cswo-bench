# Spec 10: manual hallucination check

`hallucination_<model>.csv` holds 100 outputs per model (30 habitat, 30 water level, 40 gold), written by
`uv run python -m cswo.zeroshot.evaluate sheets`. The `checker` column says who checks each row; rows were dealt by
the languages each of us reads, and `note_en` gives the machine translation for the rest (as in annotation).
Open the CSV in Excel or LibreOffice, fill in `hallucinated`, `hallucination_type` and `comment`, save as CSV
(UTF-8), and commit. Then `uv run python -m cswo.zeroshot.evaluate` adds the manual rate to
`results/tables/spec10_hallucination.md`.

## What counts as a hallucination

An output **hallucinates** if it states something the input does not support. Being wrong is not enough.

| `hallucination_type` | when | example |
|---|---|---|
| `fabricated_evidence` | the quoted evidence is not in the note, or is in the note but does not say what the label claims | evidence "the water was brown" for a note that never mentions colour |
| `unsupported_label` | a condition, relevance or image-text label with no basis in the note (or, for image-text, in the photo) | "algae" for "2 pike seen near the jetty" |
| `invalid_label` | a label outside the allowed list, or an answer that is not about the input at all | habitat "river bank" (not one of the 10), a made-up class "+6" |
| `other` | anything else invented (people, places, numbers that are not in the note); say what in `comment` | |

**Not hallucinations:**
- A wrong guess when the note gives no clue and the evidence is empty (water level for "Foto: [PERSON]"): an
  error, not a hallucination. Write `no`.
- A defensible reading of an ambiguous note (the guideline's hard cases): `no`.
- Different wording of a quote that clearly exists in the note (case, spacing, a small typo): `no`.

`auto_flags` lists what the script already caught (label outside the set, quote not found in the note). Check them
anyway: a "quote not found" flag can come from a translated quote that is otherwise faithful (then write `no` and
explain in `comment`).

Fill `hallucinated` with `yes` or `no` only. Leave the row empty if you cannot judge it, and say why in `comment`;
empty rows are left out of the rate.
