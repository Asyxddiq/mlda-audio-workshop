"""
Train the hourglass LSTM to label meeting turns (Asy).

    python prototype/train_lstm.py

Takes a few minutes on a laptop CPU and only needs numpy. Writes
prototype/trained_lstm.npz, which lstm_model.run_pipeline() and
meeting_minutes.label_turns_model() pick up automatically.

What gets trained
-----------------
The SAME network lstm_model.py runs, plus two small pieces:

    words -> embedding table -> HourglassLSTM (5 layers) -> output at each
    turn's last token (8 numbers) -> head (8 -> 3 scores) -> softmax ->
    discussion / decision / action_item

All of it is trained together: the embeddings (32 numbers per word), every
gate of all 5 LSTM layers, and the head. Because the weights are stored in
exactly the shapes lstm_model.py already uses, there's nothing to port:
lstm_model.py just loads them.

How it learns (plain NumPy, no PyTorch)
---------------------------------------
1. Forward: run one meeting through the network, keeping every gate value.
2. Loss: cross-entropy between the predicted label scores and the gold
   labels from training_data.py.
3. Backward: backpropagation through time -- walk every LSTM layer
   backwards step by step, working out how much each weight contributed
   to the loss. Fold/unfold are reshapes, so their gradients are just the
   reverse reshape; a skip connection (a + b) passes its gradient to both.
4. Update: Adam nudges every weight a little in the direction that lowers
   the loss. One update per meeting; one "epoch" = every training meeting
   once.
check_gradients() confirms step 3 against a slow numerical estimate
before any training runs, so a maths slip can't silently ruin training.

Training choices, and why
-------------------------
- Split by MEETING (train / dev / test, see training_data.py), never by
  turn, so the test meetings are conversations the model has never seen.
- Training also uses 60 short generated practice meetings
  (training_data.generate_meetings). The first 146 hand-written training
  turns alone were memorised within ~10 epochs.
- Words seen only once in training become <unk>, and 15% of training
  words are randomly swapped for <unk> each epoch. Otherwise the model
  memorises rare words and has no idea what to do with a new one.
- Early stopping: after each epoch, check the loss on the dev meeting and
  keep the weights from the best epoch.
- 5 random seeds are trained. Since 13 Oct all 5 are kept as an
  ENSEMBLE: each turn's label probabilities are averaged over the 5
  models, which is steadier than trusting one lucky seed. (ENSEMBLE =
  False keeps only the seed with the lowest dev loss, as before.) The test
  meetings and the probe set are only looked at after training, and every
  seed's test score is printed so a lucky seed can't hide.
- Settings tuned 13 Oct on the dev meetings only (12 one-at-a-time
  changes, then the best combination): learning rate
  0.01 (was 0.005) and 32-number embeddings (was 16). Test, probes and the
  scorecard were scored once, after the choice.

What this does NOT claim
------------------------
A few hundred hand-written turns is tiny. The model learns which words (and which
previous turns) go with which label in THESE meetings; it will make
mistakes on phrasing it hasn't seen. The scores below are on 3 test
meetings (45 turns) and 12 probes, so one turn either way moves them a
lot. Treat them as "does it work at all", not as a benchmark.
"""

import time

import numpy as np

from lstm_model import (HourglassLSTM, SimpleLSTM, TURN_BOUNDARY_TOKEN,
                        fold_time, pad_to_multiple_of_4, tokenize_transcript,
                        unfold_time)
from training_data import GENERATED, MEETINGS, PROBES

LABELS = ["discussion", "decision", "action_item"]
UNK_TOKEN = "<unk>"
EMBEDDING_DIM = 32     # 13 Oct: was 16 (tuned on dev)
BASE_UNITS = 8
WEIGHTS_FILE = "trained_lstm.npz"

EPOCHS = 40
PATIENCE = 15          # stop if dev loss hasn't improved for this many epochs
LEARNING_RATE = 0.01   # 13 Oct: was 0.005 (tuned on dev)
LR_DECAY = 0.95       # learning rate is multiplied by this after every epoch
WEIGHT_DECAY = 1e-4
UNK_DROPOUT = 0.15     # 13 Oct: was 0.25
SEEDS = [0, 1, 2, 3, 4]
ENSEMBLE = True        # 13 Oct: average all 5 seeds instead of keeping one

# Same keyword lists as meeting_minutes.py.
_DECISION_KEYWORDS = ["confirmed", "agreed", "approved", "decided", "noted"]
_ACTION_KEYWORDS = ["can you", "could you", "please", "follow up"]


# ---------------------------------------------------------------------------
# 1. Data: words -> ids, with <unk> for anything rare or unseen
# ---------------------------------------------------------------------------

def build_vocab(train_meetings: list[dict], min_count: int = 2) -> list[str]:
    counts: dict[str, int] = {}
    for meeting in train_meetings:
        tokens, _ends = tokenize_transcript([(s, t) for s, t, _l in meeting["turns"]])
        for word in tokens:
            counts[word] = counts.get(word, 0) + 1
    kept = sorted(w for w, c in counts.items() if c >= min_count)
    return [UNK_TOKEN] + [w for w in kept if w != UNK_TOKEN]


def encode(turns: list[tuple], word_to_id: dict[str, int]) -> dict:
    """turns: (speaker, text) or (speaker, text, label) tuples."""
    tokens, ends = tokenize_transcript([(t[0], t[1]) for t in turns])
    ids = np.array([word_to_id.get(w, 0) for w in tokens])
    labels = np.array([LABELS.index(t[2]) for t in turns]) if len(turns[0]) == 3 else None
    return {"ids": ids, "ends": ends, "labels": labels,
            "protect": np.array([w == TURN_BOUNDARY_TOKEN for w in tokens])}


# ---------------------------------------------------------------------------
# 2. One LSTM layer, forward (keeping every gate) and backward
# ---------------------------------------------------------------------------

GATES = ("f", "i", "c", "o")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def layer_forward(layer: SimpleLSTM, x_seq: np.ndarray):
    """Same maths as SimpleLSTM.step(), but remembers what backward needs."""
    H = layer.hidden_size
    h, c = np.zeros(H), np.zeros(H)
    outs, cache = [], []
    for x_t in x_seq:
        z = np.concatenate([h, x_t])
        f = _sigmoid(layer.W_f @ z + layer.b_f)
        i = _sigmoid(layer.W_i @ z + layer.b_i)
        g = np.tanh(layer.W_c @ z + layer.b_c)
        o = _sigmoid(layer.W_o @ z + layer.b_o)
        c_prev = c
        c = f * c_prev + i * g
        tc = np.tanh(c)
        h = o * tc
        outs.append(h)
        cache.append((z, f, i, g, o, c_prev, tc))
    return np.stack(outs), cache


def layer_backward(layer: SimpleLSTM, cache, d_out: np.ndarray):
    """Backpropagation through time for one layer.

    d_out: gradient of the loss w.r.t. this layer's outputs, (T, H).
    Returns gradient w.r.t. the inputs (T, I) and a dict of weight grads.
    """
    H = layer.hidden_size
    grads = {f"W_{k}": np.zeros_like(getattr(layer, f"W_{k}")) for k in GATES}
    grads.update({f"b_{k}": np.zeros(H) for k in GATES})
    d_x = np.zeros((len(cache), layer.input_size))
    dh_next, dc_next = np.zeros(H), np.zeros(H)

    for t in reversed(range(len(cache))):
        z, f, i, g, o, c_prev, tc = cache[t]
        dh = d_out[t] + dh_next
        dc = dc_next + dh * o * (1 - tc ** 2)
        pre = {                                   # grads before each gate's activation
            "f": dc * c_prev * f * (1 - f),
            "i": dc * g * i * (1 - i),
            "c": dc * i * (1 - g ** 2),
            "o": dh * tc * o * (1 - o),
        }
        dz = np.zeros_like(z)
        for k in GATES:
            grads[f"W_{k}"] += np.outer(pre[k], z)
            grads[f"b_{k}"] += pre[k]
            dz += getattr(layer, f"W_{k}").T @ pre[k]
        dh_next, d_x[t] = dz[:H], dz[H:]
        dc_next = dc * f
    return d_x, grads


# ---------------------------------------------------------------------------
# 3. The full model: embeddings -> hourglass -> head
# ---------------------------------------------------------------------------

class TrainableModel:
    def __init__(self, vocab: list[str], seed: int):
        rng = np.random.default_rng(seed)
        self.vocab = vocab
        self.embeddings = rng.uniform(-0.5, 0.5, (len(vocab), EMBEDDING_DIM))
        self.net = HourglassLSTM(EMBEDDING_DIM, BASE_UNITS, seed=seed * 10)
        self.head_W = rng.uniform(-0.3, 0.3, (len(LABELS), BASE_UNITS))
        self.head_b = np.zeros(len(LABELS))
        self.layers = [self.net.layer1, self.net.layer2, self.net.layer3,
                       self.net.layer4, self.net.layer5]

    # -- every trainable array, by name (used by Adam, saving, grad check) --
    def params(self) -> dict[str, np.ndarray]:
        p = {"embeddings": self.embeddings, "head_W": self.head_W, "head_b": self.head_b}
        for n, layer in enumerate(self.layers, start=1):
            for k in GATES:
                p[f"layer{n}_W_{k}"] = getattr(layer, f"W_{k}")
                p[f"layer{n}_b_{k}"] = getattr(layer, f"b_{k}")
        return p

    def forward(self, ids: np.ndarray, ends: list[int]):
        x = self.embeddings[ids]
        T = len(x)
        xp = pad_to_multiple_of_4(x)
        L = self.layers
        out1, c1 = layer_forward(L[0], xp)
        out2, c2 = layer_forward(L[1], fold_time(out1))
        out3, c3 = layer_forward(L[2], fold_time(out2))
        out4, c4 = layer_forward(L[3], unfold_time(out3) + out2)
        out5, c5 = layer_forward(L[4], unfold_time(out4) + out1)
        turn_vecs = out5[ends]
        logits = turn_vecs @ self.head_W.T + self.head_b
        self._cache = (ids, ends, T, xp.shape[0], turn_vecs, (c1, c2, c3, c4, c5), out5.shape)
        return logits

    def backward(self, d_logits: np.ndarray) -> dict[str, np.ndarray]:
        ids, ends, T, Tp, turn_vecs, (c1, c2, c3, c4, c5), out5_shape = self._cache
        L = self.layers
        grads = {"head_W": d_logits.T @ turn_vecs, "head_b": d_logits.sum(axis=0)}

        d_out5 = np.zeros(out5_shape)
        np.add.at(d_out5, ends, d_logits @ self.head_W)
        d_in5, g5 = layer_backward(L[4], c5, d_out5)
        d_out1 = d_in5.copy()                       # skip 1 -> 5
        d_out4 = fold_time(d_in5)                   # reverse of unfold
        d_in4, g4 = layer_backward(L[3], c4, d_out4)
        d_out2 = d_in4.copy()                       # skip 2 -> 4
        d_out3 = fold_time(d_in4)
        d_f2, g3 = layer_backward(L[2], c3, d_out3)
        d_out2 += unfold_time(d_f2)                 # reverse of fold
        d_f1, g2 = layer_backward(L[1], c2, d_out2)
        d_out1 += unfold_time(d_f1)
        d_x, g1 = layer_backward(L[0], c1, d_out1)

        d_emb = np.zeros_like(self.embeddings)
        np.add.at(d_emb, ids, d_x[:T])
        grads["embeddings"] = d_emb
        for n, g in enumerate((g1, g2, g3, g4, g5), start=1):
            for name, value in g.items():
                grads[f"layer{n}_{name}"] = value
        return grads


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def loss_and_grad(model: TrainableModel, ids, ends, labels):
    """Mean cross-entropy over the meeting's turns, and its gradients."""
    probs = softmax(model.forward(ids, ends))
    n = len(labels)
    loss = -np.log(probs[np.arange(n), labels] + 1e-12).mean()
    d_logits = probs.copy()
    d_logits[np.arange(n), labels] -= 1
    return loss, model.backward(d_logits / n)


def predict(model: TrainableModel, example: dict) -> np.ndarray:
    return model.forward(example["ids"], example["ends"]).argmax(axis=1)


class Ensemble:
    """Several trained models voting together: average their label
    probabilities, then take the most likely label. forward() returns log
    probabilities, so predict() and evaluate() work on it unchanged."""

    def __init__(self, models: list):
        self.models = models
        self.vocab = models[0].vocab

    def forward(self, ids, ends):
        probs = np.mean([softmax(m.forward(ids, ends)) for m in self.models], axis=0)
        return np.log(probs + 1e-12)


# ---------------------------------------------------------------------------
# 4. Gradient check: analytic backward vs. numerical estimate
# ---------------------------------------------------------------------------

def check_gradients(seed: int = 0, model_cls=None) -> float:
    rng = np.random.default_rng(seed)
    model = (model_cls or TrainableModel)([UNK_TOKEN] + [f"w{i}" for i in range(9)], seed=seed)
    ids = rng.integers(0, 10, size=13)
    ends, labels = [3, 8, 12], np.array([0, 2, 1])
    _loss, grads = loss_and_grad(model, ids, ends, labels)

    worst = 0.0
    eps = 1e-5
    for name, array in model.params().items():
        for _ in range(3):
            idx = tuple(rng.integers(0, s) for s in array.shape)
            if name == "embeddings":
                idx = (int(rng.choice(ids)),) + idx[1:]
            old = array[idx]
            array[idx] = old + eps
            plus, _ = loss_and_grad(model, ids, ends, labels)
            array[idx] = old - eps
            minus, _ = loss_and_grad(model, ids, ends, labels)
            array[idx] = old
            numeric = (plus - minus) / (2 * eps)
            analytic = grads[name][idx]
            # Floor of 1e-6: below that both numbers are rounding noise.
            rel = abs(numeric - analytic) / max(1e-6, abs(numeric) + abs(analytic))
            worst = max(worst, rel)
    return worst


# ---------------------------------------------------------------------------
# 5. Training loop: Adam, <unk> dropout, early stopping on the dev meeting
# ---------------------------------------------------------------------------

def mean_loss(model, examples) -> float:
    losses = []
    for ex in examples:
        probs = softmax(model.forward(ex["ids"], ex["ends"]))
        losses.append(-np.log(probs[np.arange(len(ex["labels"])), ex["labels"]] + 1e-12).mean())
    return float(np.mean(losses))


def train_one_seed(vocab, train, dev, seed, verbose=False, model_cls=None):
    model = (model_cls or TrainableModel)(vocab, seed)   # compare_rnn.py passes an RNN here
    rng = np.random.default_rng(1000 + seed)
    params = model.params()
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    beta1, beta2, step = 0.9, 0.999, 0

    best = (np.inf, None, -1)
    for epoch in range(EPOCHS):
        start = time.perf_counter()
        lr = LEARNING_RATE * LR_DECAY ** epoch
        for ex in rng.permutation(len(train)):
            ex = train[ex]
            ids = ex["ids"].copy()
            drop = (rng.random(len(ids)) < UNK_DROPOUT) & ~ex["protect"]
            ids[drop] = 0
            _loss, grads = loss_and_grad(model, ids, ex["ends"], ex["labels"])

            norm = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
            scale = min(1.0, 5.0 / (norm + 1e-12))       # clip exploding gradients
            step += 1
            for k, p in params.items():
                g = grads[k] * scale + WEIGHT_DECAY * p
                m[k] = beta1 * m[k] + (1 - beta1) * g
                v[k] = beta2 * v[k] + (1 - beta2) * g ** 2
                m_hat = m[k] / (1 - beta1 ** step)
                v_hat = v[k] / (1 - beta2 ** step)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)   # in place

        train_loss, dev_loss = mean_loss(model, train), mean_loss(model, dev)
        if verbose:
            print(f"  epoch {epoch:2d}  train loss {train_loss:.3f}  dev loss {dev_loss:.3f}"
                  f"  ({time.perf_counter() - start:.1f}s)")
        if dev_loss < best[0]:
            best = (dev_loss, {k: p.copy() for k, p in params.items()}, epoch)
        elif epoch - best[2] >= PATIENCE:
            break

    for k, p in params.items():           # restore the best epoch's weights
        p[...] = best[1][k]
    return model, best[0], best[2]


# ---------------------------------------------------------------------------
# 6. Scoring and baselines
# ---------------------------------------------------------------------------

def keyword_label(text: str) -> int:
    lowered = text.lower()
    if any(k in lowered for k in _ACTION_KEYWORDS):
        return LABELS.index("action_item")
    if any(k in lowered for k in _DECISION_KEYWORDS):
        return LABELS.index("decision")
    return LABELS.index("discussion")


def scores(gold: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Accuracy and macro-F1 (the average F1 of the three labels, so the
    rarer decision/action_item labels count as much as discussion)."""
    acc = float((gold == pred).mean())
    f1s = []
    for k in range(len(LABELS)):
        tp = np.sum((pred == k) & (gold == k))
        fp = np.sum((pred == k) & (gold != k))
        fn = np.sum((pred != k) & (gold == k))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return acc, float(np.mean(f1s))


def evaluate(model, meetings_raw, examples):
    gold = np.concatenate([ex["labels"] for ex in examples])
    model_pred = np.concatenate([predict(model, ex) for ex in examples])
    kw_pred = np.array([keyword_label(t) for m in meetings_raw for _s, t, _l in m["turns"]])
    maj_pred = np.zeros_like(gold)                       # discussion
    return {"model": scores(gold, model_pred), "keyword": scores(gold, kw_pred),
            "majority": scores(gold, maj_pred)}, gold, model_pred


def probe_results(model, word_to_id):
    gold, model_pred, kw_pred = [], [], []
    for probe in PROBES:
        ex = encode(probe, word_to_id)
        gold.append(ex["labels"][-1])
        model_pred.append(predict(model, ex)[-1])
        kw_pred.append(keyword_label(probe[-1][1]))
    return np.array(gold), np.array(model_pred), np.array(kw_pred)


def save(models: list, path: str):
    """Save one model (same keys as before) or several (an ensemble: each
    member's arrays get an m0_, m1_, ... prefix, plus a 'members' count)."""
    # float16 halves the file; checked to give identical labels on every
    # meeting and probe. lstm_model.py converts back to float64 on load.
    if len(models) == 1:
        arrays = {k: v.astype(np.float16) for k, v in models[0].params().items()}
    else:
        arrays = {f"m{n}_{k}": v.astype(np.float16)
                  for n, model in enumerate(models) for k, v in model.params().items()}
        arrays["members"] = np.array(len(models))
    np.savez_compressed(path, vocab=np.array(models[0].vocab), labels=np.array(LABELS),
                        embedding_dim=EMBEDDING_DIM, base_units=BASE_UNITS, **arrays)


# ---------------------------------------------------------------------------
# 7. Run it
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    worst = check_gradients()
    print(f"Gradient check: worst relative error {worst:.1e} "
          f"({'OK' if worst < 1e-4 else 'FAILED'})")
    assert worst < 1e-4, "backward() doesn't match the numerical gradient"

    split = {s: [m for m in MEETINGS if m["split"] == s] for s in ("train", "dev", "test")}
    split["train"] = split["train"] + GENERATED
    vocab = build_vocab(split["train"])
    word_to_id = {w: i for i, w in enumerate(vocab)}
    data = {s: [encode(m["turns"], word_to_id) for m in ms] for s, ms in split.items()}
    print(f"Vocab: {len(vocab)} words (seen 2+ times in training, plus {UNK_TOKEN})")
    print(f"Turns: train {sum(len(e['labels']) for e in data['train'])} "
          f"({len(GENERATED)} of the {len(data['train'])} meetings generated), "
          f"dev {sum(len(e['labels']) for e in data['dev'])}, "
          f"test {sum(len(e['labels']) for e in data['test'])}, probes {len(PROBES)}\n")

    runs = []
    start = time.perf_counter()
    for seed in SEEDS:
        model, dev_loss, epoch = train_one_seed(vocab, data["train"], data["dev"], seed)
        test_acc, test_f1 = evaluate(model, split["test"], data["test"])[0]["model"]
        runs.append((dev_loss, seed, model))
        print(f"seed {seed}: best epoch {epoch:2d}, dev loss {dev_loss:.3f} | "
              f"test accuracy {test_acc:.2f}, macro-F1 {test_f1:.2f}")
    print(f"({time.perf_counter() - start:.0f}s for {len(SEEDS)} seeds)\n")

    dev_loss, seed, best = min(runs, key=lambda r: r[0])
    members = [best] + [m for _d, s, m in runs if s != seed]
    if ENSEMBLE:
        model = Ensemble(members)
        print(f"Ensemble of all {len(members)} seeds (label probabilities averaged)")
    else:
        model = best
        print(f"Kept seed {seed} (lowest dev loss; chosen before looking at test)")
    dev_results, _g, _p = evaluate(model, split["dev"], data["dev"])
    print(f"Dev ({sum(len(e['labels']) for e in data['dev'])} turns): accuracy "
          f"{dev_results['model'][0]:.2f}, macro-F1 {dev_results['model'][1]:.2f}\n")

    results, gold, pred = evaluate(model, split["test"], data["test"])
    p_gold, p_model, p_kw = probe_results(model, word_to_id)
    print(f"{'Test meetings (45 turns)':28s}{'accuracy':>10s}{'macro-F1':>10s}{'probes':>9s}")
    probe_scores = {"model": (p_model == p_gold).mean(), "keyword": (p_kw == p_gold).mean(),
                    "majority": (p_gold == 0).mean()}
    names = {"model": "Trained hourglass LSTM", "keyword": "Keyword rules",
             "majority": "Always 'discussion'"}
    for key in ("model", "keyword", "majority"):
        acc, f1 = results[key]
        print(f"  {names[key]:26s}{acc:>10.2f}{f1:>10.2f}{probe_scores[key]:>9.2f}")

    # The keyword lists were written while looking at the sample meeting
    # ("noted", "confirmed", "follow up" each match one of its turns), so
    # also score the two test meetings nobody looked at while writing rules.
    unseen = [m for m in split["test"] if m["name"] != "sample_meeting"]
    unseen_results, _g, _p = evaluate(model, unseen, [encode(m["turns"], word_to_id) for m in unseen])
    n_unseen = sum(len(m["turns"]) for m in unseen)
    print(f"\nWithout the sample meeting ({len(unseen)} meetings, {n_unseen} turns):")
    for key in ("model", "keyword", "majority"):
        acc, f1 = unseen_results[key]
        print(f"  {names[key]:26s}{acc:>10.2f}{f1:>10.2f}")

    print("\nTest confusion (rows = gold, columns = predicted):")
    print(f"  {'':12s}" + "".join(f"{l[:11]:>12s}" for l in LABELS))
    for g, gl in enumerate(LABELS):
        row = [np.sum((gold == g) & (pred == p)) for p in range(len(LABELS))]
        print(f"  {gl:12s}" + "".join(f"{n:>12d}" for n in row))

    print("\nProbes (reply only):")
    for probe, g, p in zip(PROBES, p_gold, p_model):
        mark = "ok " if g == p else "MISS"
        print(f"  {mark} {probe[0][1][:45]:45s} -> {probe[1][1]:22s} "
              f"gold {LABELS[g]:11s} model {LABELS[p]}")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), WEIGHTS_FILE)
    save(members if ENSEMBLE else [best], path)
    print(f"\nSaved {WEIGHTS_FILE} ({os.path.getsize(path) / 1024:.0f} KB). "
          "pipeline.py now labels turns with the trained model.")
