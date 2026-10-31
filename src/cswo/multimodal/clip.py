"""Spec 9: image-text similarity of every note and its photo with CLIP and multilingual SigLIP.

Runs on Colab: scripts/make_clip_notebook.py embeds this file into notebooks/colab_clip.ipynb, so it has no cswo
imports (torch + transformers + PIL only). Input: clip_input.jsonl from `cswo.multimodal.clip_input`.

For every record and model the score is the cosine similarity of the note embedding and the photo embedding.
Each record also carries a `partner` (another record of the same platform), and the note is scored against the
partner's photo too: a mismatched pair the model should score lower. That gives a consistency test that does not
depend on the few gold "inconsistent" labels.
* clip-en: openai/clip-vit-base-patch32 on text_en (CLIP's text encoder is English only; 77 tokens)
* siglip-multi: google/siglip-base-patch16-256-multilingual on text_original (64 tokens)
* siglip-multi-en: the same SigLIP on text_en, to separate the effect of language from the model
Output rows: {record_id, set, platform, model, text_field, score, score_partner}; score is None when the text or photo is missing.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

MODELS = {"clip-en": ("openai/clip-vit-base-patch32", "text_en", 77),
          "siglip-multi": ("google/siglip-base-patch16-256-multilingual", "text_original", 64),
          "siglip-multi-en": ("google/siglip-base-patch16-256-multilingual", "text_en", 64)}
BATCH = 32


def fetch_images(rows: list[dict], cache: Path, log=print) -> dict[str, Path]:
    """Download every photo once into cache/ (shared by all models); url -> local file, broken links left out."""
    import requests
    from PIL import Image
    cache.mkdir(parents=True, exist_ok=True)
    out, failed = {}, 0
    for url in sorted({r["image"] for r in rows if r.get("image")}):
        path = cache / (hashlib.sha1(url.encode()).hexdigest() + ".jpg")
        if not path.exists():
            try:
                resp = requests.get(url, timeout=60, headers={"User-Agent": "CSWO-Bench research"})
                resp.raise_for_status()
                Image.open(io.BytesIO(resp.content)).convert("RGB").save(path, "JPEG", quality=90)
            except Exception:
                failed += 1
                continue
        out[url] = path
    log(f"{len(out)} photos available, {failed} could not be downloaded")
    return out


def _features(x):
    # transformers 4.x returns a tensor, newer versions may return a model output
    return x if hasattr(x, "norm") else x.pooler_output


def score_model(rows: list[dict], images: dict[str, Path], key: str, log=print) -> list[dict]:
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    name, field, max_len = MODELS[key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(name).to(device).eval()
    proc = AutoProcessor.from_pretrained(name)

    urls = sorted({r["image"] for r in rows if r.get("image") in images})
    img_emb = {}
    with torch.inference_mode():
        for i in range(0, len(urls), BATCH):
            batch = urls[i:i + BATCH]
            pix = proc(images=[Image.open(images[u]).convert("RGB") for u in batch], return_tensors="pt").to(device)
            e = _features(model.get_image_features(**pix))
            for u, v in zip(batch, torch.nn.functional.normalize(e, dim=-1).cpu()):
                img_emb[u] = v
        texts = [r for r in rows if (r.get(field) or "").strip()]
        txt_emb = {}
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            tok = proc(text=[r[field] for r in batch], padding="max_length", truncation=True, max_length=max_len,
                       return_tensors="pt").to(device)
            tok.pop("pixel_values", None)
            e = _features(model.get_text_features(**tok))
            for r, v in zip(batch, torch.nn.functional.normalize(e, dim=-1).cpu()):
                txt_emb[r["record_id"]] = v
    image_of = {r["record_id"]: r.get("image") for r in rows}
    out = []
    for r in rows:
        t = txt_emb.get(r["record_id"])
        own, other = img_emb.get(r.get("image")), img_emb.get(image_of.get(r.get("partner")))
        out.append({"record_id": r["record_id"], "set": r["set"], "platform": r["platform"], "model": key,
                    "text_field": field,
                    "score": float(t @ own) if t is not None and own is not None else None,
                    "score_partner": float(t @ other) if t is not None and other is not None else None})
    log(f"{key}: {sum(o['score'] is not None for o in out)}/{len(out)} records scored")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def run_file(src: Path, dst: Path, cache: Path, models=tuple(MODELS), log=print) -> None:
    """Score every record of src with each model into dst; a model already in dst is skipped (resumable)."""
    with src.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    done = set()
    if dst.exists():
        with dst.open(encoding="utf-8") as f:
            done = {json.loads(line)["model"] for line in f if line.strip()}
    images = fetch_images(rows, cache, log)
    for key in models:
        if key in done:
            log(f"{key}: already in {dst.name}")
            continue
        out = score_model(rows, images, key, log)
        with dst.open("a", encoding="utf-8") as f:
            f.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in out)
