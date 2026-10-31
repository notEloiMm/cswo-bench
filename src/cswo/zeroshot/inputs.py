"""Build the spec 10 zero-shot items (notebooks/colab_zeroshot.ipynb).

* habitat: every Artportalen test record of the spec 8 habitat task (117), the 10 habitat classes
* water_level: WATER_LEVEL_N CrowdWater test records of the spec 8 water level task that have a photo, chosen by a
  hash of the record id (the VLM reads the photo; the supervised baselines are re-scored on the same records)
* gold: the 300 gold records (relevance, conditions, and image-text when there is a photo)
Writes data/zeroshot/items.jsonl (committed: item id, task, masked note, its English translation, photo URL and
the reference label, so the evaluation and the hallucination sheets run without the corpus) and the Colab input
data/interim/zeroshot_input.jsonl, which leaves out the reference labels and the translation.
Needs data/interim/spec8_tasks.jsonl (`uv run python -m cswo.baselines.tasks export`) and the corpus.
Run: uv run python -m cswo.zeroshot.inputs
"""
from __future__ import annotations

import hashlib

from cswo import DATA, INTERIM
from cswo.annotate.adjudicate import GOLD
from cswo.baselines.tasks import EXPORT as TASKS
from cswo.collect.common import read_jsonl, write_jsonl
from cswo.harmonize.translate import OUT as CORPUS
from cswo.multimodal.clip_input import text_en

ITEMS = DATA / "zeroshot" / "items.jsonl"
COLAB_INPUT = INTERIM / "zeroshot_input.jsonl"
WATER_LEVEL_N = 200
SENT = ("item_id", "task", "record_id", "language", "text", "image", "labels")


def _h(key: str) -> str:
    return hashlib.sha1(f"cswo-zeroshot|{key}".encode()).hexdigest()


def level_key(c: str) -> int:
    return int(c)


def build(task_rows: list[dict], gold: list[dict], corpus: dict[str, dict]) -> list[dict]:
    items = []

    def item(task, rid, labels, reference):
        r = corpus[rid]
        return {"item_id": f"{task}|{rid}", "task": task, "record_id": rid, "platform": r["source_platform"],
                "language": r.get("language"), "text": r.get("text_original") or "", "text_en": text_en(r),
                "image": r["media"][0]["url"] if r.get("media") else None, "labels": labels, "reference": reference}

    habitat = [x for x in task_rows if x["task"] == "habitat"]
    labels = sorted({x["label"] for x in habitat})
    items += [item("habitat", x["id"], labels, x["label"])
              for x in sorted(habitat, key=lambda x: x["id"]) if x["split"] == "test"]

    level = [x for x in task_rows if x["task"] == "water_level"]
    labels = sorted({x["label"] for x in level}, key=level_key)
    test = [x for x in level if x["split"] == "test" and corpus[x["id"]].get("media")]
    items += [item("water_level", x["id"], labels, x["label"])
              for x in sorted(test, key=lambda x: _h(x["id"]))[:WATER_LEVEL_N]]

    for g in sorted(gold, key=lambda g: g["record_id"]):
        it = item("gold", g["record_id"], None, {k: g[k] for k in ("relevance", "conditions", "image_text")})
        if not g["has_photo"]:
            it["image"] = None
        items.append(it)
    return items


def main() -> None:
    task_rows = list(read_jsonl(TASKS))
    gold = [g for g in read_jsonl(GOLD) if g["set"] == "gold"]
    needed = {x["id"] for x in task_rows} | {g["record_id"] for g in gold}
    corpus = {r["id"]: r for r in read_jsonl(CORPUS) if r["id"] in needed}
    items = build(task_rows, gold, corpus)
    write_jsonl(ITEMS, items)
    n = write_jsonl(COLAB_INPUT, ({k: it[k] for k in SENT} for it in items))
    by = {t: sum(it["task"] == t for it in items) for t in ("habitat", "water_level", "gold")}
    print(f"{n} items {by}, {sum(it['image'] is not None for it in items)} with photo -> {ITEMS}, {COLAB_INPUT}")


if __name__ == "__main__":
    main()
