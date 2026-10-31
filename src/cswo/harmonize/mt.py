"""Machine translation of notes to English (spec 2). Runs on Colab: scripts/make_translate_notebook.py embeds
this file into notebooks/colab_translate.ipynb, so keep it free of cswo imports (transformers + torch only).

Model choice per language: a Helsinki-NLP OPUS-MT model where one exists (tc-big for Finnish and Estonian, the
course's focus languages); NLLB-200 (distilled 600M) for every other language.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

OPUS = {
    "sv": "Helsinki-NLP/opus-mt-sv-en", "fi": "Helsinki-NLP/opus-mt-tc-big-fi-en",
    "et": "Helsinki-NLP/opus-mt-tc-big-et-en", "de": "Helsinki-NLP/opus-mt-de-en",
    "da": "Helsinki-NLP/opus-mt-da-en", "nl": "Helsinki-NLP/opus-mt-nl-en", "fr": "Helsinki-NLP/opus-mt-fr-en",
    "es": "Helsinki-NLP/opus-mt-es-en", "it": "Helsinki-NLP/opus-mt-it-en", "ru": "Helsinki-NLP/opus-mt-ru-en",
    "hu": "Helsinki-NLP/opus-mt-hu-en", "pl": "Helsinki-NLP/opus-mt-pl-en", "lv": "Helsinki-NLP/opus-mt-lv-en",
}
NLLB = "facebook/nllb-200-distilled-600M"
NLLB_CODES = {   # FLORES-200 codes for languages without an OPUS-MT model (and fallbacks)
    "pt": "por_Latn", "lt": "lit_Latn", "no": "nob_Latn", "ro": "ron_Latn", "hr": "hrv_Latn", "sl": "slv_Latn",
    "is": "isl_Latn", "ca": "cat_Latn", "sk": "slk_Latn", "cs": "ces_Latn", "sv": "swe_Latn", "fi": "fin_Latn",
    "et": "est_Latn", "de": "deu_Latn", "da": "dan_Latn", "nl": "nld_Latn", "fr": "fra_Latn", "es": "spa_Latn",
    "it": "ita_Latn", "ru": "rus_Cyrl", "hu": "hun_Latn", "pl": "pol_Latn", "lv": "lvs_Latn",
}
MAX_CHARS = 400          # longer notes are split into sentences so nothing is truncated by the model
_SENT = re.compile(r"(?<=[.!?])\s+")


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Split at word boundaries when a single 'sentence' is too long (exports without punctuation)."""
    out, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > max_chars:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return out + ([cur] if cur else [])


def split_long(text: str, max_chars: int = MAX_CHARS, per_sentence: bool = False) -> list[str]:
    """Split a note into chunks of whole sentences, each <= max_chars. ``per_sentence`` gives one chunk per
    sentence: both NLLB and OPUS-MT drop the second sentence of short two-sentence notes
    ("Vapautettu. Rännistä syöksytorven ritilälle" -> "Released.")."""
    if len(text) <= max_chars and not per_sentence:
        return [text]
    sentences = [p for s in _SENT.split(text) for p in (_hard_split(s, max_chars) if len(s) > max_chars else [s])]
    if per_sentence:
        return sentences
    chunks, cur = [], ""
    for sent in sentences:
        if cur and len(cur) + 1 + len(sent) > max_chars:
            chunks.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def model_for(lang: str) -> str:
    return OPUS.get(lang, NLLB)


_LOOP = re.compile(r"(\b\w+\b)(?:\W+\1\b){3,}|(\b\w+-\w+\b)(?:\W*\2){3,}")


def qc_flag(source: str, text_en: str | None) -> str | None:
    """Heuristic check of one translation; returns the problem or None.
    empty; loop (a word repeated 4+ times that is not repeated in the source); omission (a multi-sentence note
    whose translation is < 60% of its length); hallucination (translation > 2.5x the source and > 25 chars)."""
    if not text_en or not text_en.strip():
        return "empty"
    if _LOOP.search(text_en) and not _LOOP.search(source):
        return "loop"
    ratio = len(text_en) / max(len(source), 1)
    if len(_SENT.split(source)) > 1 and ratio < 0.6:
        return "omission"
    if ratio > 2.5 and len(text_en) > 25:
        return "hallucination"
    return None


class Translator:
    """Loads one model at a time and translates a list of strings in length-sorted batches."""

    def __init__(self, device: str | None = None, batch_size: int = 32):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.name = self.tok = self.model = None

    def load(self, name: str) -> None:
        if name == self.name:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.model = None
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
        self.tok = AutoTokenizer.from_pretrained(name)
        dtype = self.torch.float16 if self.device == "cuda" else self.torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name, dtype=dtype).to(self.device).eval()
        self.name = name

    def translate(self, texts: list[str], lang: str) -> list[str]:
        name = model_for(lang)
        self.load(name)
        # no_repeat_ngram_size stops loops ("Hey, hey, hey, ..."); the length cap below stops run-on hallucination
        gen_kwargs = {"num_beams": 4, "no_repeat_ngram_size": 4}
        if name == NLLB:
            self.tok.src_lang = NLLB_CODES[lang]
            gen_kwargs["forced_bos_token_id"] = self.tok.convert_tokens_to_ids("eng_Latn")
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out: list[str] = [""] * len(texts)
        for b in range(0, len(order), self.batch_size):
            idx = order[b:b + self.batch_size]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                           max_length=512).to(self.device)
            cap = min(512, int(2.5 * enc["input_ids"].shape[1]) + 10)
            with self.torch.inference_mode():
                gen = self.model.generate(**enc, max_new_tokens=cap, **gen_kwargs)
            for i, t in zip(idx, self.tok.batch_decode(gen, skip_special_tokens=True)):
                out[i] = t.strip()
        return out


def translate_file(src: Path, dst: Path, translator: Translator, checkpoint_every: int = 2000,
                   log=print) -> None:
    """Translate every note in src (JSONL: id, lang, text) into dst (JSONL: id, text_en, model).
    Resumable: ids already in dst are skipped, and results are appended every ``checkpoint_every`` notes."""
    done = set()
    if dst.exists():
        with dst.open(encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    by_lang: dict[str, list[dict]] = defaultdict(list)
    with src.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] not in done and (r["lang"] in OPUS or r["lang"] in NLLB_CODES):
                by_lang[r["lang"]].append(r)
    log(f"{sum(map(len, by_lang.values()))} notes to translate ({len(done)} already done)")
    for lang, rows in sorted(by_lang.items(), key=lambda kv: -len(kv[1])):
        log(f"{lang}: {len(rows)} notes with {model_for(lang)}")
        for start in range(0, len(rows), checkpoint_every):
            part = rows[start:start + checkpoint_every]
            pieces, owner = [], []
            for k, r in enumerate(part):
                for chunk in split_long(r["text"], per_sentence=True):
                    pieces.append(chunk)
                    owner.append(k)
            translated = translator.translate(pieces, lang)
            joined: list[list[str]] = [[] for _ in part]
            for k, t in zip(owner, translated):
                joined[k].append(t)
            with dst.open("a", encoding="utf-8") as f:
                for r, parts in zip(part, joined):
                    f.write(json.dumps({"id": r["id"], "text_en": " ".join(parts), "model": translator.name},
                                       ensure_ascii=False) + "\n")
            log(f"  {lang}: {min(start + checkpoint_every, len(rows))}/{len(rows)}")


def repair_file(src: Path, dst: Path, translator: Translator, log=print) -> dict[str, int]:
    """Retranslate every note whose current translation fails qc_flag, appending the new translations to dst
    (the last line per id wins when merging). Returns the count of flags before the repair."""
    with src.open(encoding="utf-8") as f:
        source = {r["id"]: r for r in map(json.loads, f)}
    current: dict[str, dict] = {}
    with dst.open(encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            current[t["id"]] = t
    flagged = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for i, t in current.items():
        if i in source and (flag := qc_flag(source[i]["text"], t.get("text_en"))):
            flagged[source[i]["lang"]].append(source[i])
            counts[flag] += 1
    log(f"{sum(counts.values())} translations flagged: {dict(counts)}")
    for lang, rows in flagged.items():
        pieces, owner = [], []
        for k, r in enumerate(rows):
            for chunk in split_long(r["text"], per_sentence=True):
                pieces.append(chunk)
                owner.append(k)
        translated = translator.translate(pieces, lang)
        joined: list[list[str]] = [[] for _ in rows]
        for k, t in zip(owner, translated):
            joined[k].append(t)
        with dst.open("a", encoding="utf-8") as f:
            for r, parts in zip(rows, joined):
                f.write(json.dumps({"id": r["id"], "text_en": " ".join(parts), "model": translator.name,
                                    "repaired": True}, ensure_ascii=False) + "\n")
        log(f"  {lang}: repaired {len(rows)}")
    return dict(counts)
