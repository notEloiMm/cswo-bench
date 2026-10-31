"""Spec 10: zero-shot LLM (text) and VLM (text + photo) on the benchmark tasks.

Runs on Colab: scripts/make_zeroshot_notebook.py embeds this file into notebooks/colab_zeroshot.ipynb, so it has
no cswo imports (torch + transformers + PIL only). Input: zeroshot_input.jsonl from `cswo.zeroshot.inputs`.

Tasks (one prompt per item, no examples, greedy decoding):
* habitat: the Artportalen habitat value, from the 10 classes of spec 8 (test split)
* water_level: the CrowdWater water level class, from the 12 classes of spec 8 (a sample of the test split)
* gold: relevance and water conditions of the 300 gold records, and image-text consistency when there is a photo
Every answer must quote its evidence from the note, so fabricated evidence can be counted automatically (part of
the hallucination check in cswo.zeroshot.evaluate). The VLM sees the photo where there is one; the LLM never does.
Output rows: {item_id, task, record_id, model, kind, image_used, raw_output} plus the parsed fields; validation
against the label set is done locally, so the raw output is always kept.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

LLM = "Qwen/Qwen2.5-7B-Instruct"
VLM = "Qwen/Qwen2.5-VL-7B-Instruct"
PROMPT_VERSION = "zeroshot-1"

SYSTEM = ("You label citizen-science observations of lakes, rivers, streams and coasts. Use only the information "
          "given. Never invent facts. Answer with one JSON object and nothing else.")

HABITAT_GLOSS = {"vattendrag": "watercourse", "sjö": "lake", "bäck": "brook", "å": "small river", "älv": "large river",
                 "damm": "pond", "sjöstrand": "lake shore", "fors i medelstor å": "rapids in a medium-sized river",
                 "brant bäck i barrskog": "steep brook in conifer forest",
                 "kall, liten bäck i barrskog": "cold small brook in conifer forest"}

CONDITIONS = """clear (water clear or transparent) | turbid (murky, muddy, cloudy, silty) | coloured (unusual colour:
brown, green, yellow, milky) | algae (algae, bloom, blue-green algae) | foam | ice (ice present, forming or melting) |
oil (oil, fuel, sheen) | litter (rubbish, plastic in the water or on the shore) | odour (a smell from the water) |
high water (level higher than usual) | low or dry (level lower than usual, dried up) | flooding (water over
normally dry land)"""


def prompt(item: dict, image: bool) -> str:
    note = f"Note (language: {item.get('language') or 'unknown'}):\n\"\"\"{item['text']}\"\"\""
    photo = "\nA photo from the same observation is attached." if image else ""
    task = item["task"]
    if task == "habitat":
        labels = "\n".join(f"- {c}" + (f" ({HABITAT_GLOSS[c]})" if c in HABITAT_GLOSS else "") for c in item["labels"])
        return (f"{note}{photo}\n\nThis note comes from the Swedish species portal Artportalen. The observer also "
                f"filled in a free-text habitat field. Which of these values did they write?\n{labels}\n\n"
                'Answer: {"label": "<one value from the list, exactly as written>", '
                '"evidence": "<the words of the note that support it, copied exactly, or empty>"}')
    if task == "water_level":
        return (f"{note}{photo}\n\nThis observation comes from CrowdWater. The observer compares the water level with "
                "a virtual staff gauge drawn on the first photo of this spot and reports the class where the water "
                "surface is now: positive classes are above the reference level (higher water), negative classes "
                f"below it (lower water). Possible classes: {', '.join(item['labels'])}.\n\n"
                'Answer: {"label": "<one class from the list>", '
                '"evidence": "<the words of the note that support it, copied exactly, or empty>"}')
    image_part = ('\nimage_text: does the photo agree with what the note asserts? one of "consistent", "partly '
                  'consistent", "inconsistent", or "cannot judge" if the photo shows no water and not the thing '
                  'described.' if image else "")
    image_key = ', "image_text": "<label>"' if image else ""
    return (f"{note}{photo}\n\nrelevance: is the note about water (a water body, its state, or something in or on "
            'it)? one of "about water", "not about water", "uncertain" (empty or unreadable note).\n'
            f"conditions: every water condition the note itself asserts (not inferred, not negated), from: "
            f"{CONDITIONS}. Use [] if none or if the note is not about water.{image_part}\n\n"
            'Answer: {"relevance": "<label>", "conditions": [{"label": "<condition>", "evidence": "<words of the '
            f'note, copied exactly>"}}]{image_key}}}')


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
                    out = json.loads(text[start:i + 1])
                    return out if isinstance(out, dict) else None
                except json.JSONDecodeError:
                    break
        start = text.find("{", start + 1)
    return None


class Model:
    """A 7B instruct model in 4-bit (fits a free Colab T4), greedy decoding. vision=True loads a VLM."""

    def __init__(self, name: str, vision: bool, max_pixels: int = 640 * 28 * 28):
        import torch
        from transformers import BitsAndBytesConfig
        self.torch, self.name, self.vision = torch, name, vision
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                   bnb_4bit_quant_type="nf4") if torch.cuda.is_available() else None
        kw = dict(quantization_config=quant, device_map="auto" if quant else None,
                  dtype=torch.float16 if quant else torch.float32)
        if vision:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            extra = {"max_pixels": max_pixels} if "qwen" in name.lower() else {}
            self.processor = AutoProcessor.from_pretrained(name, **extra)
            self.model = AutoModelForImageTextToText.from_pretrained(name, **kw).eval()
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.processor = AutoTokenizer.from_pretrained(name)
            self.model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()

    def _image(self, url: str | None):
        if not url or not self.vision:
            return None
        import requests
        from PIL import Image
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "CSWO-Bench research"})
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            return None                                   # broken link: the item is answered from the text

    def answer(self, item: dict, max_new_tokens: int = 300) -> dict:
        image = self._image(item.get("image"))
        text = prompt(item, image is not None)
        if self.vision:
            messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                        {"role": "user", "content": ([{"type": "image"}] if image else []) + [{"type": "text",
                                                                                              "text": text}]}]
            chat = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=[chat], images=[image] if image else None, return_tensors="pt")
        else:
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
            chat = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = self.processor([chat], return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        decode = self.processor.batch_decode
        raw = decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return {"item_id": item["item_id"], "task": item["task"], "record_id": item["record_id"], "model": self.name,
                "kind": "vlm" if self.vision else "llm", "image_used": image is not None,
                "prompt_version": PROMPT_VERSION, "parsed": extract_json(raw), "raw_output": raw}


def run_file(src: Path, dst: Path, model: Model, log=print) -> None:
    """Answer every item of src with model into dst; resumable (items this model already answered are skipped)."""
    done = set()
    if dst.exists():
        with dst.open(encoding="utf-8") as f:
            done = {(r["item_id"], r["model"]) for r in map(json.loads, f) if r}
    with src.open(encoding="utf-8") as f:
        todo = [r for r in map(json.loads, f) if (r["item_id"], model.name) not in done]
    log(f"{model.name}: {len(todo)} items to answer")
    for i, item in enumerate(todo, 1):
        row = model.answer(item)
        with dst.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if i % 25 == 0 or i == len(todo):
            log(f"  {i}/{len(todo)}")
