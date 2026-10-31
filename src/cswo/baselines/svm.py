"""Spec 8: TF-IDF + linear SVM on the weak-label tasks, with and without stopword removal.

Two feature sets: word uni+bigrams (the spec's baseline) and character 2-5-grams inside word boundaries, which
spec 4 suggests suit this open-vocabulary micro-text (names, codes, inflection, compounds). LinearSVC with
balanced class weights; C is chosen on the validation macro-F1, the model is trained on train only and scored
once on test. A majority-class predictor gives the floor.
Output: results/predictions/spec8_svm.jsonl (test predictions; scored by cswo.baselines.report).
Run: uv run python -m cswo.baselines.svm
"""
from __future__ import annotations

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

from cswo import RESULTS
from cswo.baselines.tasks import load
from cswo.collect.common import write_jsonl

PREDICTIONS = RESULTS / "predictions" / "spec8_svm.jsonl"
C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)
MAX_ITER = 20_000       # the default 1000 does not converge for large C on char n-grams
FEATURES = {
    "tfidf-word+svm": dict(analyzer="word", token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), min_df=2),
    "tfidf-char+svm": dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2),
}


def fit_predict(rows: list[dict], features: str, stop: bool) -> tuple[list[str], float]:
    """Test predictions and the chosen C."""
    field = "text_nostop" if stop else "text"
    part = {s: [x for x in rows if x["split"] == s] for s in ("train", "validation", "test")}
    vec = TfidfVectorizer(lowercase=True, sublinear_tf=True, **FEATURES[features])
    X = {s: (vec.fit_transform if s == "train" else vec.transform)([x[field] for x in xs]) for s, xs in part.items()}
    y = {s: [x["label"] for x in xs] for s, xs in part.items()}
    best = max(C_GRID, key=lambda c: f1_score(
        y["validation"], LinearSVC(C=c, class_weight="balanced", max_iter=MAX_ITER).fit(X["train"], y["train"]).predict(X["validation"]),
        average="macro", zero_division=0))
    clf = LinearSVC(C=best, class_weight="balanced", max_iter=MAX_ITER).fit(X["train"], y["train"])
    return list(clf.predict(X["test"])), best


def run(tasks: dict[str, list[dict]]) -> list[dict]:
    preds = []
    for task, rows in tasks.items():
        test = [x for x in rows if x["split"] == "test"]
        if not test or not any(x["split"] == "train" for x in rows):
            print(f"{task}: no text records, skipped")
            continue
        majority = Counter(x["label"] for x in rows if x["split"] == "train").most_common(1)[0][0]
        runs = [("majority", False, [majority] * len(test), None)]
        for features in FEATURES:
            for stop in (False, True):
                p, c = fit_predict(rows, features, stop)
                runs.append((features, stop, p, c))
                print(f"{task:12s} {features:15s} stopwords removed={stop!s:5s} C={c}")
        for model, stop, p, c in runs:
            preds += [{"id": x["id"], "task": task, "model": model, "stopwords_removed": stop, "C": c,
                       "gold": x["label"], "pred": str(y)} for x, y in zip(test, p)]
    return preds


def main() -> None:
    n = write_jsonl(PREDICTIONS, run(load()))
    print(f"{n} test predictions -> {PREDICTIONS}")


if __name__ == "__main__":
    main()
