"""
Score the trained model on the frozen scorecard (scorecard_data.py).

    python prototype/scorecard.py                     # the shipped trained_lstm.npz
    python prototype/scorecard.py other_weights.npz   # any other weights file

Prints AGGREGATE scores only (accuracy, macro-F1, per-label precision and
recall, next to keyword rules), never which turns were wrong, so nobody
can tune the model against the scorecard turn by turn. Refuses to run if
scorecard_data.py has been edited since it was frozen on 13 Oct 2026.
"""

import hashlib
import os
import sys

import numpy as np

from lstm_model import TRAINED_WEIGHTS, label_probabilities, load_trained_model

FROZEN_SHA256_PREFIX = "f1ae61878026c3fb"   # sha256 of scorecard_data.py, 13 Oct 2026
LABELS = ["discussion", "decision", "action_item"]
_DECISION_KEYWORDS = ["confirmed", "agreed", "approved", "decided", "noted"]
_ACTION_KEYWORDS = ["can you", "could you", "please", "follow up"]


def keyword_label(text: str) -> str:
    lowered = text.lower()
    if any(k in lowered for k in _ACTION_KEYWORDS):
        return "action_item"
    if any(k in lowered for k in _DECISION_KEYWORDS):
        return "decision"
    return "discussion"


def score(gold: list[str], pred: list[str]) -> dict:
    gold, pred = np.array(gold), np.array(pred)
    out = {"accuracy": float((gold == pred).mean())}
    f1s = []
    for label in LABELS:
        tp = np.sum((pred == label) & (gold == label))
        fp = np.sum((pred == label) & (gold != label))
        fn = np.sum((pred != label) & (gold == label))
        out[label] = (tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0)
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    out["macro_f1"] = float(np.mean(f1s))
    return out


def predict_labels(meetings: list[dict], trained: dict) -> list[str]:
    pred = []
    for m in meetings:
        probs = label_probabilities([(s, t) for s, t, _l in m["turns"]], trained)
        pred += [trained["labels"][k] for k in probs.argmax(axis=1)]
    return pred


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "scorecard_data.py"), "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if not digest.startswith(FROZEN_SHA256_PREFIX):
        sys.exit("scorecard_data.py has changed since it was frozen, so its scores would "
                 "no longer be comparable. Restore it from git.")
    from scorecard_data import SCORECARD

    path = sys.argv[1] if len(sys.argv) > 1 else TRAINED_WEIGHTS
    trained = load_trained_model(path)
    if trained is None:
        sys.exit(f"No weights at {path}: run python prototype/train_lstm.py first.")

    gold = [l for m in SCORECARD for _s, _t, l in m["turns"]]
    rows = {"Trained LSTM": score(gold, predict_labels(SCORECARD, trained)),
            "Keyword rules": score(gold, [keyword_label(t) for m in SCORECARD for _s, t, _l in m["turns"]])}
    n_models = len(trained["members"])
    print(f"Scorecard: {len(SCORECARD)} frozen meetings, {len(gold)} turns "
          f"({os.path.basename(path)}, {n_models} model{'s' if n_models > 1 else ''})\n")
    print(f"  {'':16s}{'accuracy':>9s}{'macro-F1':>10s}   precision / recall per label")
    for name, r in rows.items():
        per = "  ".join(f"{lab[:3]} {r[lab][0]:.2f}/{r[lab][1]:.2f}" for lab in LABELS)
        print(f"  {name:16s}{r['accuracy']:>9.2f}{r['macro_f1']:>10.2f}   {per}")
