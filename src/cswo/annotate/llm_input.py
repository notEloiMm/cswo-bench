"""Export the gold records for the LLM second annotator (notebooks/colab_llm_annotate.ipynb).

Only what the annotators saw leaves the machine: record id, masked original note, detected language and the photo
URL. Calibration records are included too, so the LLM can be compared with the students on the shared 20.
Run: uv run python -m cswo.annotate.llm_input   -> data/gold/llm_input.jsonl
"""
from __future__ import annotations

from cswo.annotate.merge import task_data
from cswo.annotate.sample import GOLD_DIR
from cswo.collect.common import read_jsonl, write_jsonl

OUT = GOLD_DIR / "llm_input.jsonl"


def main() -> None:
    data = task_data()
    manifest = [m for m in read_jsonl(GOLD_DIR / "gold_manifest.jsonl") if m["set"] in ("calibration", "gold")]
    rows = [{"record_id": m["id"], "set": m["set"], "language": data[m["id"]]["language"],
             "text_original": data[m["id"]]["text_original"],
             "image": data[m["id"]]["image"] if data[m["id"]]["has_photo"] else None}
            for m in manifest]
    n = write_jsonl(OUT, rows)
    print(f"{n} records ({sum(r['set'] == 'gold' for r in rows)} gold, "
          f"{sum(r['image'] is not None for r in rows)} with photo) -> {OUT}")


if __name__ == "__main__":
    main()
