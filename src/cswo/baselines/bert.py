"""Spec 8: fine-tune multilingual DistilBERT on the weak-label tasks, with and without stopword removal.
Runs on Colab: scripts/make_bert_notebook.py embeds this file into notebooks/colab_bert.ipynb, so keep it free of
cswo imports (torch + transformers only).

distilbert-base-multilingual-cased, a linear head on the [CLS] vector, class-weighted cross-entropy (balanced, as
the SVM), AdamW with linear warm-up/decay. Trained for EPOCHS epochs; the epoch with the best validation macro-F1
is the one scored on test. Input: spec8_tasks.jsonl from `cswo.baselines.tasks export`.
Output rows match the SVM's: {id, task, model, stopwords_removed, epoch, gold, pred} for the test split.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

MODEL = "distilbert-base-multilingual-cased"
MODEL_NAME = "distilbert-multilingual"
EPOCHS, LR, BATCH, MAX_LEN, SEED = 4, 3e-5, 32, 128, 0


def macro_f1(gold: list[str], pred: list[str]) -> float:
    labels = sorted(set(gold) | set(pred))
    f = []
    for c in labels:
        tp = sum(g == p == c for g, p in zip(gold, pred))
        fp = sum(p == c != g for g, p in zip(gold, pred))
        fn = sum(g == c != p for g, p in zip(gold, pred))
        f.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return sum(f) / len(f) if f else 0.0


def train_eval(rows: list[dict], stop: bool, epochs: int = EPOCHS, lr: float = LR, batch: int = BATCH,
               max_len: int = MAX_LEN, seed: int = SEED, model_name: str = MODEL) -> tuple[list[str], int]:
    """Test predictions of the best validation epoch, and that epoch (1-based)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    field = "text_nostop" if stop else "text"
    part = {s: [x for x in rows if x["split"] == s] for s in ("train", "validation", "test")}
    classes = sorted({x["label"] for x in part["train"]})
    idx = {c: i for i, c in enumerate(classes)}
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(classes)).to(device)

    def encode(xs):
        # an empty note after stopword removal still needs one token
        return tok([x[field] or x["text"] for x in xs], truncation=True, max_length=max_len)

    def batches(xs, enc, shuffle):
        order = list(range(len(xs)))
        if shuffle:
            random.shuffle(order)
        for i in range(0, len(order), batch):
            ids = order[i:i + batch]
            pad = tok.pad({k: [enc[k][j] for j in ids] for k in enc}, return_tensors="pt")
            yield {k: v.to(device) for k, v in pad.items()}, ids

    enc = {s: encode(xs) for s, xs in part.items()}
    y_train = torch.tensor([idx[x["label"]] for x in part["train"]], device=device)
    counts = Counter(x["label"] for x in part["train"])
    weight = torch.tensor([len(part["train"]) / (len(classes) * counts[c]) for c in classes], device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(part["train"]) + batch - 1) // batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    @torch.no_grad()
    def predict(s):
        model.eval()
        out = [None] * len(part[s])
        for b, ids in batches(part[s], enc[s], False):
            for j, k in zip(ids, model(**b).logits.argmax(-1).tolist()):
                out[j] = classes[k]
        return out

    best = (-1.0, 0, None)
    for epoch in range(1, epochs + 1):
        model.train()
        for b, ids in batches(part["train"], enc["train"], True):
            loss = loss_fn(model(**b).logits, y_train[ids])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        f = macro_f1([x["label"] for x in part["validation"]], predict("validation"))
        print(f"  epoch {epoch}: validation macro-F1 {f:.3f}")
        if f > best[0]:
            best = (f, epoch, predict("test"))
    return best[2], best[1]


def run_file(src: Path, dst: Path, **kw) -> None:
    """Every task x stopword setting in src; settings already in dst are skipped, so a rerun resumes."""
    tasks = defaultdict(list)
    for line in src.open(encoding="utf-8"):
        x = json.loads(line)
        tasks[x["task"]].append(x)
    done = set()
    if dst.exists():
        done = {(r["task"], r["stopwords_removed"]) for r in map(json.loads, dst.open(encoding="utf-8"))}
    for task, rows in tasks.items():
        for stop in (False, True):
            if (task, stop) in done:
                continue
            print(f"{task}, stopwords removed={stop}: {Counter(x['split'] for x in rows)}")
            preds, epoch = train_eval(rows, stop, **kw)
            test = [x for x in rows if x["split"] == "test"]
            with dst.open("a", encoding="utf-8") as f:
                for x, p in zip(test, preds):
                    f.write(json.dumps({"id": x["id"], "task": task, "model": MODEL_NAME, "stopwords_removed": stop,
                                        "epoch": epoch, "gold": x["label"], "pred": p}, ensure_ascii=False) + "\n")
