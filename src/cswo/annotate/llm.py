"""LLM second annotator for the gold set (guideline v0.1: each record is labelled by one student and one LLM).

Runs on Colab: scripts/make_llm_notebook.py embeds this file into notebooks/colab_llm_annotate.ipynb, so it has
no cswo imports (transformers + torch + PIL only). The model reads every note in the original language and, when
a photo is attached, the photo too; it returns JSON that is validated and converted to the guideline export
format (read as data/gold/exports/LLM.jsonl by cswo.annotate.merge).

Entity spans: the model returns the entity text; offsets are found in the note here. Entities whose text does not
occur in the note are dropped and counted in "dropped_entities" (a first hallucination signal for spec 10).
Decoding is greedy, so a rerun gives the same labels.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
GUIDELINE_VERSION = "v0.1"
PROMPT_VERSION = "llm-annotator-1"

CONDITIONS = ["clear", "turbid", "coloured", "algae", "foam", "ice", "oil", "litter", "odour", "high water",
              "low or dry", "flooding"]
ENTITY_LABELS = ["WATER_BODY", "LOCATION", "ORGANISM", "QUANTITY", "DATE_TIME", "WEATHER"]
RELEVANCE = ["about water", "not about water", "uncertain"]
IMAGE_TEXT = ["consistent", "partly consistent", "inconsistent", "cannot judge"]

# Condensed from the course guideline v0.1 and the group's calibration rulings (data/gold/hard_cases.md).
INSTRUCTIONS = """You annotate short field notes written by volunteers about lakes, rivers, streams and coasts.
Label only what the note asserts, not what the photo shows and not what you infer. Hedged assertions count
("maybe algae" -> algae). Negations do not count ("no ice yet" is not ice). Questions do not count.

RELEVANCE (one): "about water" if the note describes a water body, its state, or something in or on it (an animal
on a lake, a level reading, a locality that is itself a water body). "not about water" if it is about something
else, even on a water platform (an activity next to a lake that only names the lake; behaviour with no water
mentioned). "uncertain" if the text cannot be judged (empty, one word, garbled). If "not about water", stop:
conditions = [], entities = [], image_text = null.

WATER CONDITIONS (every one the note asserts; if about water and none, use ["none stated"]):
clear (water clear/transparent, bottom visible; a depth number alone is not clear) | turbid (murky, muddy,
cloudy, silty, "looks dirty") | coloured (a named colour that is not normal: brown, green, yellow, milky; brown
plus mud/silt/rain-wash is turbid AND coloured; green water without algae named is coloured only) | algae (algae,
bloom, scum described as algal, blue-green, levä, sinilevä, alg) | foam (foam, froth not attributed to algae) |
ice (ice present, forming, melting, frozen; frost on the shore is not ice) | oil (oil, fuel, sheen, rainbow film on
the water) | litter (rubbish, plastic, bottles in the water or on the shore) | odour (any smell from the water or
shore) | high water (level higher than usual) | low or dry (level lower than usual, dry or intermittent; "shallow"
alone is not) | flooding (water covering normally dry land, or the word flood/tulva). A water temperature or a
level number without a judgement is no condition.

ENTITIES (exact text copied from the note; shortest span naming the thing; no overlaps):
WATER_BODY a named or identified lake, river, stream, pond, bay, sea, canal, spring ("Lake Vättern", "the small
pond behind the school"; a bare "the lake" or "a pond" is NOT marked) | LOCATION a place that is not a water body
(town, park, street, bridge, monitoring station id) | ORGANISM a plant, animal, alga or fungus by any name |
QUANTITY a number with its unit, a count, a measurement, a gauge reading ("3 m", "12 ducklings"; for "12 mallards"
mark "12" QUANTITY and "mallards" ORGANISM; a water temperature is QUANTITY) | DATE_TIME a date, time or
relative time ("today", "15.6.") | WEATHER a weather condition or event ("heavy rain", "sunny", "frost").
Never mark "[PERSON]" or hashtags.

IMAGE-TEXT (only when a photo is given and the note is about water; first matching rule):
1 the photo shows no water and not the thing described (selfie, map, blur) -> "cannot judge";
2 the note asserts no condition and the photo shows water or the thing described -> "consistent";
3 every asserted condition is visible or cannot be shown in a photo (smell) -> "consistent";
4 some asserted conditions are visible and some contradicted -> "partly consistent";
5 the photo contradicts the main assertion -> "inconsistent".

Set "uncertain": true if you are unsure about any label, and say why in "comment" (one line).
Answer with JSON only, no other text:
{"relevance": "...", "conditions": [...], "entities": [{"text": "...", "label": "..."}],
 "image_text": "..." or null, "uncertain": false, "comment": ""}"""


def build_messages(record: dict, has_image: bool) -> list[dict]:
    note = f"Note (language: {record.get('language') or 'unknown'}):\n{record['text_original']}"
    content = ([{"type": "image"}] if has_image else []) + [
        {"type": "text", "text": note + ("\n\nA photo is attached above." if has_image else "\n\nNo photo.")}]
    return [{"role": "system", "content": [{"type": "text", "text": INSTRUCTIONS}]},
            {"role": "user", "content": content}]


def extract_json(text: str) -> dict | None:
    """The first JSON object in the model output (tolerates ```json fences and text around it)."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break
        start = text.find("{", start + 1)
    return None


def locate(note: str, entities: list[dict]) -> tuple[list[dict], int]:
    """Offsets for entity texts in reading order; returns (spans, number dropped as not found / overlapping)."""
    spans, dropped, pos = [], 0, 0
    for e in entities:
        text, label = str(e.get("text", "")).strip(), e.get("label")
        if not text or label not in ENTITY_LABELS or text == "[PERSON]":
            dropped += 1
            continue
        i = note.find(text, pos)
        if i == -1:
            i = note.find(text)                       # out of order: search from the start
        if i == -1:
            m = re.search(re.escape(text), note, re.I)  # case differences only
            i, text = (m.start(), note[m.start():m.end()]) if m else (-1, text)
        if i == -1 or any(i < s["end"] and s["start"] < i + len(text) for s in spans):
            dropped += 1
            continue
        spans.append({"start": i, "end": i + len(text), "text": text, "label": label})
        pos = i + len(text)
    return sorted(spans, key=lambda s: (s["start"], s["end"])), dropped


def to_annotation(record: dict, raw_text: str, has_image: bool, model: str = MODEL) -> dict:
    """Validate the model output and convert it to the guideline export row (always read in the original)."""
    out = extract_json(raw_text) or {}
    problems = [] if out else ["unparsable output"]
    relevance = out.get("relevance") if out.get("relevance") in RELEVANCE else None
    if relevance is None:
        problems.append(f"relevance {out.get('relevance')!r}")
        relevance = "uncertain"
    conds = sorted({c for c in out.get("conditions") or [] if c in CONDITIONS})
    bad = [c for c in out.get("conditions") or [] if c not in CONDITIONS and c != "none stated"]
    if bad:
        problems.append(f"unknown conditions {bad}")
    spans, dropped = locate(record["text_original"], out.get("entities") or [])
    image = out.get("image_text") if out.get("image_text") in IMAGE_TEXT else None
    if relevance == "not about water":
        conds, spans, image = [], None, None
    else:
        conds = conds or ["none stated"]
    if not has_image:
        image = None
    comment = str(out.get("comment") or "").strip()
    if problems:
        comment = (comment + " | " if comment else "") + "parse: " + "; ".join(problems)
    return {"record_id": record["record_id"], "annotator_id": "LLM", "read_from": "original",
            "relevance": relevance, "conditions": conds, "entities": spans, "image_text": image,
            "uncertain": bool(out.get("uncertain")) or bool(problems), "comment": comment,
            "guideline": GUIDELINE_VERSION, "model": model, "prompt_version": PROMPT_VERSION,
            "dropped_entities": dropped, "raw_output": raw_text}


class Annotator:
    """Qwen2.5-VL in 4-bit (fits a free Colab T4), greedy decoding."""

    def __init__(self, model: str = MODEL, max_pixels: int = 640 * 28 * 28):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        self.torch, self.name = torch, model
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                   bnb_4bit_quant_type="nf4") if torch.cuda.is_available() else None
        # max_pixels caps the image tokens of Qwen2.5-VL (memory on a T4); other processors do not take it
        extra = {"max_pixels": max_pixels} if "qwen" in model.lower() else {}
        self.processor = AutoProcessor.from_pretrained(model, **extra)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model, quantization_config=quant, device_map="auto" if quant else None,
            dtype=torch.float16 if quant else torch.float32).eval()

    def _image(self, url: str | None):
        if not url:
            return None
        import requests
        from PIL import Image
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "CSWO-Bench annotation (research)"})
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            return None                                   # broken link: annotate as text-only, noted in comment

    def annotate(self, record: dict, max_new_tokens: int = 400) -> dict:
        image = self._image(record.get("image"))
        messages = build_messages(record, image is not None)
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image] if image else None, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        raw = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        row = to_annotation(record, raw, image is not None, self.name)
        if record.get("image") and image is None:
            row["comment"] = (row["comment"] + " | " if row["comment"] else "") + "photo could not be loaded"
        return row


def annotate_file(src: Path, dst: Path, annotator: Annotator, log=print) -> None:
    """Label every record in src (JSONL from cswo.annotate.llm_input) into dst; resumable."""
    done = set()
    if dst.exists():
        with dst.open(encoding="utf-8") as f:
            done = {json.loads(line)["record_id"] for line in f if line.strip()}
    with src.open(encoding="utf-8") as f:
        todo = [r for r in map(json.loads, f) if r["record_id"] not in done]
    log(f"{len(todo)} records to label ({len(done)} already done)")
    for i, record in enumerate(todo, 1):
        row = annotator.annotate(record)
        with dst.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if i % 10 == 0 or i == len(todo):
            log(f"  {i}/{len(todo)}")
