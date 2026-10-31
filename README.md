# CSWO-Bench

Data card: [DATACARD.md](DATACARD.md) (composition, collection, preprocessing, licences, biases, limitations).
Code: MIT licence ([LICENSE](LICENSE)); data: the licence of each source, per record (see the data card).

A multilingual, multimodal benchmark of citizen-science water observations (University of Oulu NLP course, project 16).
Records from EyeOnWater, iNaturalist, GBIF-published Nordic portals (Artportalen, Laji.fi, Arter.dk) and CrowdWater
are harmonized into one JSON schema with original text, English
translation, photo URLs, coarsened coordinates, structured labels, licence and provenance.

## Setup

```bash
uv sync                       # Python 3.11 + dependencies into .venv
cp .env.example .env          # then fill in GBIF credentials and a shared CSWO_SALT
```

All group members must use **the same `CSWO_SALT`** so user hashes agree; never commit `.env`.
Heavy models (translation, DistilBERT, SigLIP, LLM/VLM) run in the Colab notebooks under `notebooks/`.

## Pipeline

| Step | Command | Output |
|---|---|---|
| Aquatic taxa config | `uv run python scripts/resolve_taxa.py` | `configs/aquatic_taxa.yaml` |
| EyeOnWater | `uv run python -m cswo.collect.eyeonwater` | `data/raw/eyeonwater/` |
| iNaturalist (≈40 min, resumable) | `uv run python -m cswo.collect.inaturalist` | `data/raw/inaturalist/inat_<CC>.jsonl` |
| GBIF portals | `uv run python -m cswo.collect.gbif submit` then `... gbif fetch <key>` | `data/raw/gbif/<key>.zip` + DOI |
| CrowdWater (manual) | download CSV → `data/raw/crowdwater/`, set licence in `configs/manual_sources.yaml`, `uv run python -m cswo.collect.manual crowdwater` | licence log row |
| Harmonize | `uv run python -m cswo.harmonize.build` | `data/interim/cswo_harmonized.jsonl`, `results/tables/spec2_*` |
| Export for translation | `uv run python -m cswo.harmonize.translate export` | `data/interim/to_translate.jsonl` (id, language, text only) |
| Translate (Colab, T4 GPU) | open `notebooks/colab_translate.ipynb`, run all | `translations.jsonl` |
| Merge translations | `uv run python -m cswo.harmonize.translate merge translations.jsonl` | `data/processed/cswo_all.jsonl` |
| Mask names + extract entities (spec 5) | `uv run python -m cswo.extract.run` (`--sample N` for a dry run) | masked `cswo_all.jsonl`, `data/processed/entities.jsonl` |
| Gold sample + Label Studio tasks (spec 6) | `uv run python -m cswo.annotate.sample` then `... annotate.tasks` | `data/gold/` |
| LLM second annotator (guideline v0.1) | `uv run python -m cswo.annotate.llm_input`, then `notebooks/colab_llm_annotate.ipynb` on a T4 (Qwen2.5-VL-7B, 4-bit); save the output as `data/gold/exports/LLM.jsonl` | LLM labels for the 300 gold + 20 calibration records |
| Merge annotations (spec 7) | exports saved as `data/gold/exports/<ID>.json` (+ `LLM.jsonl`), then `uv run python -m cswo.annotate.merge` | `data/gold/annotations.jsonl` |
| Agreement + disagreement sheets | `uv run python -m cswo.annotate.agreement` | `results/tables/spec7_agreement_*`, `data/gold/disagreements_<round>.csv` |
| Adjudicated gold | fill the `resolved` column, then `uv run python -m cswo.annotate.adjudicate` | `data/gold/gold.jsonl` |
| Extractor evaluation | `uv run python -m cswo.extract.evaluate` | `results/tables/spec7_{extractors,relevance,mt}_*` |
| Fixed splits (spec 8, 12) | `uv run python -m cswo.baselines.splits` | `data/splits.csv` (committed) |
| Weak-label tasks | `uv run python -m cswo.baselines.tasks export` | `results/tables/spec8_tasks.*`, `data/interim/spec8_tasks.jsonl` |
| TF-IDF + SVM | `uv run python -m cswo.baselines.svm` | `results/predictions/spec8_svm.jsonl` |
| DistilBERT (Colab, T4 GPU) | open `notebooks/colab_bert.ipynb`, run all; save output to `results/predictions/` | `spec8_bert.jsonl` |
| Score baselines | `uv run python -m cswo.baselines.report` | `results/tables/spec8_baselines*` |
| Image-text input (spec 9) | `uv run python -m cswo.multimodal.clip_input` | `data/interim/clip_input.jsonl` |
| CLIP / SigLIP (Colab, T4 GPU) | open `notebooks/colab_clip.ipynb`, run all; save output to `results/predictions/` | `spec9_clip.jsonl` |
| Score image-text consistency | `uv run python -m cswo.multimodal.clip_eval` | `results/tables/spec9_*`, `results/figures/spec9_similarity.png` |
| Zero-shot items (spec 10) | `uv run python -m cswo.zeroshot.inputs` (after `baselines.tasks export`) | `data/zeroshot/items.jsonl`, `data/interim/zeroshot_input.jsonl` |
| LLM + VLM (Colab, T4 GPU) | open `notebooks/colab_zeroshot.ipynb`, run all; save output to `results/predictions/` | `spec10_zeroshot.jsonl` |
| Hallucination sheets | `uv run python -m cswo.zeroshot.evaluate sheets`, then fill them in (`data/zeroshot/HALLUCINATION_CHECK.md`) | `data/zeroshot/hallucination_<model>.csv` |
| Score zero-shot models | `uv run python -m cswo.zeroshot.evaluate` | `results/tables/spec10_*` |
| Benchmark statistics (appendix) | `uv run python -m cswo.analysis.dataset_stats` | `results/tables/stats_*`, `results/figures/stats_water_body_by_platform.png` |
| Bias audit (spec 11) | `uv run python -m cswo.analysis.bias` | `results/tables/spec11_*`, `results/figures/spec11_{map,months,lorenz}.png` |
| GLiNER zero-shot NER (spec 13, CPU, ~2 min) | `uv run --with gliner python -m cswo.sota.gliner_ner run` | `results/predictions/spec13_gliner.jsonl` |
| Score GLiNER vs spec 5 extractors | `uv run python -m cswo.sota.gliner_ner` | `results/tables/spec13_*` |
| Release files (spec 12) | `uv run python -m cswo.package.release` | `data/release/` (split JSONL.gz, ODbL CrowdWater files, gold, MANIFEST), `data/release_manifest.json` |

Order matters: masking changes the note text, so it runs after the translation merge and before the gold tasks
are generated (gold entity spans are offsets into the masked text). A new translation merge requires re-running
`extract.run`, which reads the unmasked copy kept in `data/interim/cswo_all_unmasked.jsonl` (gitignored).

Every download appends a row to `data/licence_log.csv` (source, date, licence, DOI, record count).

## Design decisions

- **Licences.** Only CC0, CC-BY, CC-BY-SA, ODC-By and ODbL records are kept; CC-BY-NC is excluded (group decision).
  This drops all of Observation.org (CC-BY-NC), the main GBIF route to Latvian data, so Latvian is very thin.
  CrowdWater is ODbL (share-alike): the CrowdWater subset must be released under ODbL with attribution to
  CrowdWater, University of Zurich. Photos are released as URLs unless their own licence permits redistribution
  (`media[].redistributable`).
- **GBIF datasets.** Only the citizen-science datasets in `configs/gbif_datasets.yaml` are harvested. GBIF also carries
  scientific surveys (e.g. ICES DATRAS) whose remarks are boilerplate. iNaturalist is taken from its own API, not GBIF.
  Of the whitelisted portals, only Artportalen (SE), Laji.fi Notebook (FI) and Arter.dk (DK) publish observer notes
  to GBIF; the Estonian, Norwegian, naturgucker and Faune-France datasets have none. Estonian text comes from CrowdWater.
- **GBIF taxonomy.** GBIF occurrences are interpreted against the Catalogue of Life XR checklist, whose keys differ
  from the old backbone, so the SQL download filters on rank names (`gbif_xr` in `configs/aquatic_taxa.yaml`).
  Download 0003577-… used old keys and selected slime moulds; it is kept in `data/raw/gbif/superseded/` for the audit.
- **Artportalen sample.** ~1.6M Artportalen notes pass the filters. A 30k sample is kept (group decision): notes with
  ≥ 3 words, each kept with probability 30k/N using a hash of the record ID, so the sample is reproducible and
  proportional across taxa and years in expectation. Short notes ("ca", "minst") are therefore under-represented
  for Swedish compared with other platforms — relevant for the corpus statistics in specs 3–4.
- **CrowdWater.** Soil-moisture observations are dropped (not a water body). The export has no user column, so
  per-user contribution skew (spec 11) cannot be computed for CrowdWater. `ROOT_ID` (repeat visits to one gauge) is
  kept as `labels.series_id`, which the splits must respect to avoid leakage.
- **Personal names in notes.** 8.3% of notes name a person, mostly survey credits ("Provtagare …, Utvärderad
  av …", "Photo by …"). Names are replaced with `[PERSON]` in `text_original` and `text_en` (`src/cswo/extract/mask.py`):
  observer-credit rules, multi-word PERSON spans from either NER system, and single words only when spaCy and
  XLM-R agree. Spans that are genera, Latin binomials, common organism names or place/organisation words are
  protected. On an 80-span random check about 82% of masked spans were real names; the errors are mostly
  place and species names, i.e. the masking errs on the side of privacy. CrowdWater photo credits
  ("Foto: Ana", "Foto de Ana", "Foto: Ana y Eva"; ~300 notes, mostly first names) were added to the
  credit rule later; still missed are a misspelt trigger ("Foro: Ana"), a name inside a sentence ("... de
  Ana de este día") and one lower-case "Foto: ana". When a credit is followed by a capitalised word without a
  full stop, that word is masked too ("Foto: [PERSON] dia de lluvia" for "Foto: Ana Primer dia de lluvia").
  Artportalen herbarium-style collector fields ("Typha latifolia/3C0e3441/… s:n/AEx/ /Anna Exempel,Lund", ~190
  notes) are masked by their own rule; the town after the comma is kept.
  In translations, the original's names are matched ignoring diacritics, because MT may drop them ("Pål" → "Pal").
- **NER.** spaCy small models per language (Estonian and other languages without a model use `xx_ent_wiki_sm`)
  and `Davlan/xlm-roberta-base-ner-hrl`, which was not trained on Finnish, Swedish or Estonian and runs zero-shot
  there. `preshed` is pinned to 3.0.10 because 3.0.14 breaks spaCy model loading on Windows.
- **Water filter.** A record is kept if its taxon is obligately aquatic (fish, aquatic plants, ...) or its
  locality/habitat/note mentions a water body in its language or English (`configs/water_terms.yaml`).
  The evidence is stored in `water_evidence`. Place names that contain water words (Finnish "Lammi", "Joensuu")
  are a known source of false positives.
- **Templates.** Notes that repeat ≥ 50 times on one platform after masking digits (e.g. "Tiirahavainto id = #######")
  are import stubs and are blanked. Short genuine notes ("2 ex.") are kept.
- **Anonymization.** Coordinates rounded to 0.01° (≈1 km); user IDs replaced with a salted SHA-256 prefix; e-mail
  addresses in notes replaced with `[EMAIL]`. EyeOnWater exposes e-mails as nicknames: they are never stored.
- **Translation.** OPUS-MT per language (tc-big models for Finnish and Estonian); NLLB-200 distilled 600M for
  languages without an OPUS-MT model (pt, lt, no, ro, hr, sl, ...). Notes are translated one sentence at a time
  because both NLLB and OPUS-MT drop the second sentence of short two-sentence notes; output length is capped
  relative to the input and repeated 4-grams are blocked. Every translation is checked by `mt.qc_flag` (empty,
  repetition loop, omission, hallucination); flagged notes were retranslated (`uv run --with torch --with
  transformers --with sentencepiece python -c "..."` calling `mt.repair_file`), and the 37 still failing are
  rejected (`text_en` empty, `translation_qc: rejected:<reason>`) — mostly language-ID errors on species names
  ("Lemna minor" tagged Romanian, which NLLB turns into EU regulation text). Notes whose language
  could not be identified (`und`: codes, numbers, species names) are left untranslated
  (`translation_model: untranslated:und`). Mixed-language notes are translated as their dominant language.
  The model used is stored per record. `notebooks/colab_translate.ipynb` is generated from
  `src/cswo/harmonize/mt.py` by `scripts/make_translate_notebook.py`.
- **Exports and coordinates in notes.** Tiira (Finnish bird database) export dumps are reduced to the observer's
  note; the trailing dump held exact coordinates and observer names. Decimal coordinate pairs typed into any note
  are replaced with `[COORD]`.
- **Agreement and evaluation (spec 7).** Cohen's kappa on relevance, image-text and water condition (one kappa over
  the flattened record × condition matrix, the figure the guideline's 0.6 threshold applies to); span F1 between
  annotators, exact and overlap, only on records both read in the original. Calibration pairs every two students;
  the gold set pairs each student with the LLM. Extractors are scored on the adjudicated gold; the English-original
  vs MT comparison matches entities by type and count per record, because MT spans have no offsets in the original.
  `data/gold/system_outputs.jsonl` holds the extractor outputs for the gold records only, so the evaluation runs
  without the full corpus.
- **Gold design (spec 6) follows the course guideline v0.1, not the project text.** The project text asks for two
  students per record; the guideline asks for one student and one LLM. Each student labelled 100 disjoint records
  of one shared stratified sample (3 × 100 = 300) and all three labelled the same 20 calibration records; the LLM
  (Qwen2.5-VL-7B) labelled all 300. Student–student agreement is therefore only measured on the 20 calibration
  records; the gold set reports student–LLM agreement.
- **Splits (spec 8).** 80/10/10 of the records with text, stratified by platform. A CrowdWater series (`ROOT_ID`)
  stays in one split; all 340 gold-manifest records and their series are in test, so the gold set stays held out
  for spec 10. Assignment follows a hash of the group id, so `data/splits.csv` is reproducible.
  **Deviation (group decision 2026-10-26):** the 88 gold CrowdWater records belong to 65 series with 4,096 records
  (one gauge has 1,613 visits), so CrowdWater is 64/8/28 instead of 80/10/10; the other platforms are exactly
  80/10/10. We keep the leakage-free split (no gauge in both train and test) and report the deviation in the data
  card and the report.
- **Weak-label baselines (spec 8).** Tasks: CrowdWater water level class, Artportalen habitat (lower-cased, the 10
  most frequent values), and EyeOnWater Forel-Ule bin. **EyeOnWater has no notes, so the Forel-Ule task has no text
  records** and is reported as empty. Classes with < 30 records are dropped. Input is `text_original`; stopword
  removal uses NLTK's list for the note's language (none for Estonian or `und`, which stay unchanged).
  TF-IDF word 1–2-grams and char 2–5-grams with a balanced LinearSVC (C tuned on validation), and fine-tuned
  `distilbert-base-multilingual-cased` (best validation epoch of 4), each with and without stopwords; a majority
  predictor is the floor. Headline metric is macro F1. `notebooks/colab_bert.ipynb` is generated from
  `src/cswo/baselines/bert.py` by `scripts/make_bert_notebook.py`.
- **Forel-Ule bins.** FU 1–5 blue, 6–13 green, 14–21 brown.
- **Image-text consistency (spec 9).** Cosine similarity of note and photo with CLIP ViT-B/32 (English text:
  the translation, or the note when it is English or `und`) and multilingual SigLIP base (original note, and the
  translation for comparison). Scored on the 148 gold records with a photo against the adjudicated image-text label,
  and on 300 further notes with photos per platform (iNaturalist, CrowdWater). Only 8 gold records are partly
  consistent or inconsistent, so every note is also scored against another record's photo (same platform): a model
  that captures consistency must rank the real pair above the mismatched one (AUC own vs other).
  `notebooks/colab_clip.ipynb` is generated from `src/cswo/multimodal/clip.py` by `scripts/make_clip_notebook.py`.
- **Zero-shot LLM and VLM (spec 10).** Qwen2.5-7B-Instruct (note only) and Qwen2.5-VL-7B-Instruct (note and photo),
  4-bit, greedy, no examples in the prompt, on the habitat test split (117), 200 CrowdWater water level test records
  with a photo (hash sample; the spec 8 models are re-scored on the same records) and the 300 gold records
  (relevance, conditions, image-text). Every answer quotes its evidence from the note; a quote that does not occur in
  the note is counted automatically as fabricated, and 100 outputs per model are checked by hand
  (`data/zeroshot/HALLUCINATION_CHECK.md`). The gold set was adjudicated against Qwen2.5-VL annotations (spec 7), so
  the VLM's gold scores are not independent of the labels; the text-only LLM is a different checkpoint.
  `notebooks/colab_zeroshot.ipynb` is generated from `src/cswo/zeroshot/llm.py` by `scripts/make_zeroshot_notebook.py`.
