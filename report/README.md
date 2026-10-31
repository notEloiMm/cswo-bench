# Final report

Following the course brief: the tables and plots of specs 2–9, the discussion of spec 11 (bias audit) and the
literature and limitations of spec 14, plus the course's mandatory items (students and IDs, code link, Use of AI).
Plain LaTeX `article` style (one column), about 13 pages with figures.

`report.tex` is the main file; sections are in `sections/`:

| File | Content | Owner (as in the presentation) |
|---|---|---|
| `intro.tex` | abstract, index terms, introduction | Finn |
| `specs2_4.tex` | spec 2 corpus, spec 3 statistics, spec 4 Zipf and Heaps | Finn |
| `specs5_7.tex` | spec 5 extraction, spec 6 gold set, spec 7 agreement and extractors | Eloi |
| `specs8_9.tex` | spec 8 weak-label baselines, spec 9 image–text | Sam |
| `specs11_14.tex` | spec 11 bias audit, spec 14 literature and limitations | all |
| `code_ai.tex` | code availability, Use of AI tools | all |

Red text = still to fill in before submission (student IDs, surname, other AI tools).

## Compile

- Overleaf: zip the `report/` folder → New project → Upload project; main document `report.tex`, compiler pdfLaTeX.
- Locally (MiKTeX): `pdflatex report && bibtex report && pdflatex report && pdflatex report`.

## After changing results

```
uv run python scripts/make_report_tables.py      # regenerate tables/*.tex (Zipf, Heaps)
cp results/figures/<name>.png report/figures/     # refresh a figure
```

Numbers written in the text come from `results/tables/`; check them if a result changes.
