# CSWO-Bench data card

A multilingual, multimodal benchmark of citizen-science water observations. This card follows *Datasheets for
Datasets* (Gebru et al., 2021). Version 0.1.0, 2026-10-30. Counts per file and SHA-256 checksums are in
[`data/release_manifest.json`](data/release_manifest.json); every number below comes from `results/tables/` and
the report (`report/report.pdf`) in this repository.

## At a glance

| | |
|---|---|
| Records | 225,145 unique, from 6 sources |
| Records with a note | 80,886 (train 62,334 · validation 7,812 · test 10,740) |
| Languages of the notes | 24 identified + `und` (Swedish 32%, English 25%, Finnish 16%); all translated to English |
| Records with a photo | 174,530 (photos released as URLs, never copied) |
| Gold set | 300 records + 20 calibration records, adjudicated (relevance, water conditions, entity spans, image–text) |
| Weak labels | CrowdWater water-level class (13,839), EyeOnWater Forel-Ule colour (26,758), habitat text (4,877), species (132,499) |
| Licences | CC0-1.0, CC-BY-4.0, ODbL-1.0 (CrowdWater, share-alike), EyeOnWater open data policy |

## 1. Motivation

**Why was the dataset created?** Volunteers photograph and describe lakes, rivers, streams and coasts on many
platforms, each with its own schema, licence and language. There was no shared dataset to test whether NLP and
vision models can read these short field notes, extract water conditions and entities from them, or check a photo
against its note. CSWO-Bench harmonises six sources into one schema with fixed splits, a hand-labelled gold set and
baselines.

**Who created it?** Eloi, Sam and Finn, students of the University of Oulu NLP course 2026 (Project 16, proposed by
Moinul Islam).

**Funding.** None; built as course work with free tools (Google Colab T4 for the heavy models).

## 2. Composition

**What does an instance represent?** One citizen-science observation: optional note (`text_original`) and its English
translation (`text_en`), photo URLs with their licences (`media`), date (`timestamp`), coordinates rounded to 0.01°
(about 1 km), country, water-body type, platform labels, licence, provenance URL and a salted user hash. The full
schema is `src/cswo/harmonize/schema.py`.

| Source | Records | Licence | What it contributes |
|---|---|---|---|
| iNaturalist (API, 14 countries) | 85,899 | CC0-1.0 / CC-BY-4.0 | photos of aquatic species; 23% with a note |
| CrowdWater (University of Zurich) | 65,892 | ODbL-1.0 | photo + water-level class against a virtual gauge; 22% with a note |
| EyeOnWater | 26,758 | EyeOnWater open data policy | photo + Forel-Ule water colour; no notes |
| Artportalen, Sweden (GBIF; 30k sample of ~1.6M) | 28,223 | CC0-1.0 | Swedish notes, habitat field |
| Laji.fi, Finland (GBIF) | 16,707 | CC-BY-4.0 | Finnish notes |
| Other GBIF portals (Arter.dk, …) | 1,670 | CC-BY-4.0 | Danish and other notes |

**Water-body type** (from locality/habitat/note keywords): river 18%, stream 18%, lake 13%, coast 13%, wetland 2%,
sea 2%, pond 1%, unknown 34% (all of EyeOnWater, half of iNaturalist). Per platform:
`results/tables/stats_water_body_by_platform.md`.

**Is it a sample?** Partly. iNaturalist was harvested for aquatic taxa in 14 countries (North America, Western
Europe, Nordic and Baltic states) with CC0/CC-BY licences; Artportalen is a reproducible hash sample of 30,000 notes
of at least 3 words; GBIF portals were limited to a whitelist of citizen-science datasets; records needed an aquatic
taxon or a water word to enter (`water_evidence`).

**Labels.**
- *Weak labels* from the platforms' structured fields: CrowdWater water-level class (−6 … +7), EyeOnWater Forel-Ule
  1–21 and 3 bins (blue 1–5, green 6–13, brown 14–21), Secchi depth and temperature, habitat (free text), species.
- *Gold labels* (`gold.jsonl`) for 300 records: relevance (about water / not / uncertain), water conditions (12
  classes or "none stated"), entity spans in six types (WATER_BODY, LOCATION, ORGANISM, QUANTITY, DATE_TIME,
  WEATHER; on notes read in the original), image–text consistency. Annotation guideline v0.1 of the course; hard cases
  in `data/gold/hard_cases.md`.

**Splits.** 80/10/10 of the records with a note, stratified by platform, assigned by a hash of a group id
(`data/splits.csv`). Repeat visits to one CrowdWater gauge stay in one split; all gold records and their series are in
test. Consequence: CrowdWater is 64/8/28 instead of 80/10/10 (kept on purpose, see Limitations). Records without a note
are not split (`*_media_only` files).

**Missing information.** 66% of records have no note; 11,323 notes have undetermined language (`und`: codes, species
names) and are not translated; 37 translations were rejected by the quality check (`translation_qc`); CrowdWater
publishes no user id; EyeOnWater has no locality type.

**Confidential or sensitive data.** None intended. Personal data was removed (see Preprocessing); a few masking misses
remain (see Limitations).

## 3. Collection process

Collected 2026-10-25 with the scripts in `src/cswo/collect/`: iNaturalist public API (id cursor), EyeOnWater public
API, GBIF SQL downloads (each with a DOI, filtered by country, dataset whitelist, aquatic taxa on the Catalogue of
Life XR checklist and a non-empty remarks field), and the CrowdWater CSV export. Every download is logged in
`data/licence_log.csv` (source, date, licence, DOI, record count). Observation dates range mainly 2015–2026 (85%);
Artportalen and Laji.fi add older records. No one was contacted; all data was already public under open licences.

## 4. Preprocessing, cleaning, labelling

- **Harmonisation** into one pydantic schema (`src/cswo/harmonize/`); import stubs repeated 50+ times on one platform
  are blanked; Tiira export dumps are cut to the observer's note.
- **Anonymisation:** coordinates rounded to 0.01°; user ids replaced by a salted SHA-256 hash (secret salt); e-mail
  addresses → `[EMAIL]`, typed coordinates → `[COORD]`; personal names → `[PERSON]` in the note and its translation
  (observer-credit rules incl. "Foto:" and Artportalen collector fields, multi-word PERSON spans from spaCy or XLM-R
  NER, single words only when both agree; matching in translations ignores diacritics). About 8% of notes named a
  person; on an 80-span check about 82% of masked spans were real names (errs towards privacy).
- **Language identification:** `lingua` with a platform prior and a mixed-language detector.
- **Translation:** OPUS-MT per language (tc-big for Finnish and Estonian), NLLB-200 distilled 600M for the rest,
  sentence by sentence, output length capped, repetition blocked; an automatic check flags empty, looping, truncated
  and hallucinated output (177 of 48,741 flagged, 37 rejected). Model per language:
  `results/tables/stats_translation_models.md`.
- **Gold annotation:** stratified sample (platform × language group × photo); 20 calibration records labelled by
  all three students (relevance κ 0.80, conditions κ 1.00); each gold record labelled by one student and one LLM
  (Qwen2.5-VL-7B, as the course guideline prescribes); 237 disagreements adjudicated by a third person (student label
  kept 178, LLM 53, rewritten by the group 6). Label Studio exports are kept in `data/gold/exports/`.
- **Raw data** is not released (it contains personal data); the collection scripts can rebuild it.

## 5. Uses

**Intended:** benchmarking NLP on short, noisy, multilingual field notes — relevance filtering, water-condition
extraction, entity extraction, weak-label prediction, translation robustness — and image–text consistency; studying
citizen-science data quality.

**Baselines in this repository** (`report/report.pdf`): extractors spaCy / XLM-R / rules / GLiNER (best partial entity
F1 0.52, zero-shot GLiNER); TF-IDF + SVM and multilingual DistilBERT (habitat macro F1 0.73, water level 0.22);
CLIP / SigLIP image–text similarity (AUC own vs other photo ≈ 0.70); zero-shot Qwen2.5-7B / Qwen2.5-VL-7B (68–69%
of checked outputs hallucinate).

**Not intended:** identifying or profiling observers; inferring exact locations (coordinates are deliberately
coarse); environmental monitoring decisions without expert validation (labels are volunteer reports, not
measurements); treating the corpus as representative of water bodies worldwide.

**Known biases a user should account for** (spec 11, section "Spec 11" of `report/report.pdf`): geography (Sweden, US,
Switzerland 37%; Africa, South Asia, South America nearly absent; Gini over countries 0.91), language (sv/en/fi 72% of
notes; Latvian almost absent), season (summer 37%, winter 15%), contributors (top 1% of users wrote 30–61% of a
platform's records). Report results per platform and language.

## 6. Distribution

Released as gzipped JSON Lines in this repository (`data/release/`, built by `uv run python -m cswo.package.release`):
`cswo_bench_{train,validation,test}.jsonl.gz`, `cswo_bench_media_only.jsonl.gz`, the CrowdWater subset in separate
`cswo_bench_crowdwater_odbl_*.jsonl.gz` files, `gold.jsonl` and `MANIFEST.json` (checksums).

**Licences.** Each record carries its licence:
- CC0-1.0: Artportalen, part of iNaturalist — no conditions.
- CC-BY-4.0: Laji.fi, other GBIF portals, part of iNaturalist — attribute the platform; the record's
  `provenance_url` identifies the source, GBIF downloads are cited by DOI (`data/licence_log.csv`).
- ODbL-1.0: CrowdWater (University of Zurich) — the `crowdwater_odbl` files are released under ODbL with attribution
  to CrowdWater; derivative databases of that subset must stay under ODbL.
- EyeOnWater open data policy — attribute EyeOnWater.
- Our annotations (gold labels, translations, masking) are released under CC-BY-4.0.
- Photos are **not** redistributed: `media[].url` links to the platform and `media[].licence` /
  `redistributable` give the photo's own licence (about 31,000 iNaturalist photos are non-commercial or no-derivatives
  and must not be copied).

**Code** (collection, preprocessing, baselines, evaluation): this repository, MIT licence (`LICENSE`).

## 7. Maintenance

Maintained by the authors for the course; issues and corrections through the repository. Records removed from a
source platform should be removed from the next release; the pipeline is fully scripted (`README.md`) so a new
release can be rebuilt from fresh downloads with the same salt. Known errors are listed below and in the
report (`report/report.pdf`).

## Limitations

- **Gold design:** one student and one LLM per record, not two students (course guideline v0.1); student–student
  agreement is measured only on the 20 calibration records. The gold set was adjudicated against Qwen2.5-VL labels,
  so that model's gold scores are not independent.
- **CrowdWater split 64/8/28:** whole gauge series with gold records are in test to avoid leakage.
- **Small gold positives:** e.g. 5 inconsistent image–text pairs, 4 turbid notes; per-class scores are noisy.
- **Two group rulings** infer turbid/clear from a substrate description, beyond "label what the note says".
- **Residual masking misses:** a misspelt credit ("Foro: …"), a name inside a sentence, a lower-case "Foto: …".
- **Language ID** fails on short notes and species names; such errors propagate into translation.
- **Translations** pass a plausibility check, not a correctness check (domain terms and abbreviations).
- **Water-body type** is keyword-based and unknown for a third of the records; place names containing water words
  ("Lammi") are false positives; only 51% of gold notes are actually about water.
- **Duplicates:** 4 iNaturalist observations near a border were harvested for two countries; the release keeps one
  copy (225,145 unique records; the analysis tables count 225,149 rows).
- **Photo links** may break; photos on the source platforms may carry EXIF metadata.
- **DistilBERT** baselines were trained before the last masking fixes (fewer than 500 notes changed; SVMs moved by at
  most 0.01).
