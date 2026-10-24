"""
Does the LSTM actually beat a plain RNN here? (Pasan and Rahini's question)

    python prototype/compare_rnn.py        # ~15-20 min on a laptop CPU, numpy only

Builds the SAME 5-layer hourglass (same sizes, folds, skips, embeddings,
head, data, training settings and 5 seeds as train_lstm.py) but with plain
RNN layers,  h = tanh(W . [h_prev, x] + b),  instead of LSTM layers.
Both sides are trained here from scratch on the current training_data.py
(5 seeds each), so the comparison doesn't depend on which weights file is
shipped.

Rule, written down BEFORE the first run: "LSTM better" needs BOTH
  +1 probe or more (mean over the 5 seeds, out of 12), AND
  +0.05 macro-F1 or more on the 3 test meetings (mean over the 5 seeds).
Otherwise the verdict is "no clear difference". A clear win for the RNN
would be reported as such.

It also measures memory reach: blank out the turn d turns back (its words
become <unk>) and see how much the current turn's label probabilities move
(total variation, 0 = that earlier turn had no effect).

Result: no clear difference between the two on this task.
"""

import time

import numpy as np

import train_lstm as T
from lstm_model import fold_time, pad_to_multiple_of_4, unfold_time
from training_data import GENERATED, MEETINGS


# ---------------------------------------------------------------------------
# 1. A plain RNN layer: forward and backward (backpropagation through time)
# ---------------------------------------------------------------------------

class SimpleRNNLayer:
    def __init__(self, input_size: int, hidden_size: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.input_size, self.hidden_size = input_size, hidden_size
        scale = 1.0 / np.sqrt(input_size + hidden_size)
        self.W = rng.uniform(-scale, scale, (hidden_size, hidden_size + input_size))
        self.b = np.zeros(hidden_size)


def rnn_layer_forward(layer: SimpleRNNLayer, x_seq: np.ndarray):
    h = np.zeros(layer.hidden_size)
    outs, cache = [], []
    for x_t in x_seq:
        z = np.concatenate([h, x_t])
        h = np.tanh(layer.W @ z + layer.b)
        outs.append(h)
        cache.append((z, h))
    return np.stack(outs), cache


def rnn_layer_backward(layer: SimpleRNNLayer, cache, d_out: np.ndarray):
    H = layer.hidden_size
    grads = {"W": np.zeros_like(layer.W), "b": np.zeros(H)}
    d_x = np.zeros((len(cache), layer.input_size))
    dh_next = np.zeros(H)
    for t in reversed(range(len(cache))):
        z, h = cache[t]
        pre = (d_out[t] + dh_next) * (1 - h ** 2)
        grads["W"] += np.outer(pre, z)
        grads["b"] += pre
        dz = layer.W.T @ pre
        dh_next, d_x[t] = dz[:H], dz[H:]
    return d_x, grads


# ---------------------------------------------------------------------------
# 2. The same hourglass and training interface as train_lstm.TrainableModel
# ---------------------------------------------------------------------------

class TrainableRNNModel(T.TrainableModel):
    def __init__(self, vocab: list[str], seed: int):
        rng = np.random.default_rng(seed)
        u1, u2, u3 = T.BASE_UNITS, T.BASE_UNITS * 2, T.BASE_UNITS * 4
        self.vocab = vocab
        self.embeddings = rng.uniform(-0.5, 0.5, (len(vocab), T.EMBEDDING_DIM))
        self.head_W = rng.uniform(-0.3, 0.3, (len(T.LABELS), T.BASE_UNITS))
        self.head_b = np.zeros(len(T.LABELS))
        s = seed * 10
        self.layers = [SimpleRNNLayer(T.EMBEDDING_DIM, u1, s), SimpleRNNLayer(u1 * 2, u2, s + 1),
                       SimpleRNNLayer(u2 * 2, u3, s + 2), SimpleRNNLayer(u3 // 2, u2, s + 3),
                       SimpleRNNLayer(u2 // 2, u1, s + 4)]

    def params(self):
        p = {"embeddings": self.embeddings, "head_W": self.head_W, "head_b": self.head_b}
        for n, layer in enumerate(self.layers, start=1):
            p[f"layer{n}_W"], p[f"layer{n}_b"] = layer.W, layer.b
        return p

    def forward(self, ids, ends):
        x = self.embeddings[ids]
        T_real = len(x)
        xp = pad_to_multiple_of_4(x)
        L = self.layers
        out1, c1 = rnn_layer_forward(L[0], xp)
        out2, c2 = rnn_layer_forward(L[1], fold_time(out1))
        out3, c3 = rnn_layer_forward(L[2], fold_time(out2))
        out4, c4 = rnn_layer_forward(L[3], unfold_time(out3) + out2)
        out5, c5 = rnn_layer_forward(L[4], unfold_time(out4) + out1)
        turn_vecs = out5[ends]
        self._cache = (ids, ends, T_real, turn_vecs, (c1, c2, c3, c4, c5), out5.shape)
        return turn_vecs @ self.head_W.T + self.head_b

    def backward(self, d_logits):
        ids, ends, T_real, turn_vecs, (c1, c2, c3, c4, c5), out5_shape = self._cache
        L = self.layers
        grads = {"head_W": d_logits.T @ turn_vecs, "head_b": d_logits.sum(axis=0)}
        d_out5 = np.zeros(out5_shape)
        np.add.at(d_out5, ends, d_logits @ self.head_W)
        d_in5, g5 = rnn_layer_backward(L[4], c5, d_out5)
        d_out1 = d_in5.copy()
        d_in4, g4 = rnn_layer_backward(L[3], c4, fold_time(d_in5))
        d_out2 = d_in4.copy()
        d_f2, g3 = rnn_layer_backward(L[2], c3, fold_time(d_in4))
        d_out2 += unfold_time(d_f2)
        d_f1, g2 = rnn_layer_backward(L[1], c2, d_out2)
        d_out1 += unfold_time(d_f1)
        d_x, g1 = rnn_layer_backward(L[0], c1, d_out1)
        d_emb = np.zeros_like(self.embeddings)
        np.add.at(d_emb, ids, d_x[:T_real])
        grads["embeddings"] = d_emb
        for n, g in enumerate((g1, g2, g3, g4, g5), start=1):
            grads[f"layer{n}_W"], grads[f"layer{n}_b"] = g["W"], g["b"]
        return grads


# ---------------------------------------------------------------------------
# 3. Scoring: test scores, probes, and memory reach
# ---------------------------------------------------------------------------

def memory_reach(model, examples, max_back: int = 4) -> list[float]:
    reach = []
    for d in range(1, max_back + 1):
        moves = []
        for ex in examples:
            base = T.softmax(model.forward(ex["ids"], ex["ends"]))
            starts = [0] + [e + 1 for e in ex["ends"][:-1]]     # first token of each turn
            for k in range(d, len(ex["ends"])):
                ids = ex["ids"].copy()
                turn = np.arange(starts[k - d], ex["ends"][k - d] + 1)   # tokens of turn k-d
                ids[turn[~ex["protect"][turn]]] = 0                       # words -> <unk>, <eot> kept
                moved = T.softmax(model.forward(ids, ex["ends"]))[k]
                moves.append(0.5 * np.abs(moved - base[k]).sum())
        reach.append(float(np.mean(moves)))
    return reach


def score_model(model, word_to_id, test_meetings, test_examples):
    acc, f1 = T.evaluate(model, test_meetings, test_examples)[0]["model"]
    p_gold, p_pred, _kw = T.probe_results(model, word_to_id)
    return acc, f1, int((p_gold == p_pred).sum())


def run():
    worst = T.check_gradients(model_cls=TrainableRNNModel)
    print(f"RNN gradient check: worst relative error {worst:.1e} ({'OK' if worst < 1e-4 else 'FAILED'})")
    assert worst < 1e-4

    split = {s: [m for m in MEETINGS if m["split"] == s] for s in ("train", "dev", "test")}
    split["train"] = split["train"] + GENERATED
    vocab = T.build_vocab(split["train"])
    word_to_id = {w: i for i, w in enumerate(vocab)}
    data = {s: [T.encode(m["turns"], word_to_id) for m in ms] for s, ms in split.items()}

    models = {"LSTM": [], "RNN": []}
    runs = {"LSTM": [], "RNN": []}
    start = time.perf_counter()
    for kind, cls in (("RNN", TrainableRNNModel), ("LSTM", T.TrainableModel)):
        for seed in T.SEEDS:
            model, dev_loss, epoch = T.train_one_seed(vocab, data["train"], data["dev"], seed, model_cls=cls)
            runs[kind].append(score_model(model, word_to_id, split["test"], data["test"]))
            models[kind].append(model)
            acc, f1, probes = runs[kind][-1]
            print(f"{kind} seed {seed}: best epoch {epoch:2d}, dev loss {dev_loss:.3f} | test accuracy "
                  f"{acc:.2f}, macro-F1 {f1:.2f}, probes {probes}/12  ({time.perf_counter() - start:.0f}s)")
    lstm_runs, rnn_runs = runs["LSTM"], runs["RNN"]

    print(f"\n{'5-seed means (test, 45 turns)':32s}{'accuracy':>9s}{'macro-F1':>10s}{'probes':>8s}")
    means = {}
    for name, kind_runs in (("LSTM hourglass", lstm_runs), ("RNN hourglass", rnn_runs)):
        means[name] = np.mean(np.array(kind_runs, dtype=float), axis=0)
        acc, f1, probes = means[name]
        lo, hi = min(r[1] for r in kind_runs), max(r[1] for r in kind_runs)
        print(f"  {name:30s}{acc:>9.2f}{f1:>10.2f}{probes:>8.1f}   (macro-F1 range {lo:.2f}-{hi:.2f})")
    d_f1 = means["LSTM hourglass"][1] - means["RNN hourglass"][1]
    d_probes = means["LSTM hourglass"][2] - means["RNN hourglass"][2]
    if d_probes >= 1 and d_f1 >= 0.05:
        verdict = "LSTM better"
    elif -d_probes >= 1 and -d_f1 >= 0.05:
        verdict = "RNN better"
    else:
        verdict = "no clear difference"
    print(f"\n  LSTM minus RNN: macro-F1 {d_f1:+.2f}, probes {d_probes:+.1f}  ->  {verdict}")

    print("\nMemory reach on the test meetings (mean over seeds): how much the current")
    print("turn's prediction moves when the turn d back is blanked out")
    lstm_reach = np.mean([memory_reach(m, data["test"]) for m in models["LSTM"]], axis=0)
    rnn_reach = np.mean([memory_reach(m, data["test"]) for m in models["RNN"]], axis=0)
    print("  turns back     " + "".join(f"{d:>7d}" for d in range(1, len(lstm_reach) + 1)))
    print("  LSTM           " + "".join(f"{r:>7.3f}" for r in lstm_reach))
    print("  RNN            " + "".join(f"{r:>7.3f}" for r in rnn_reach))


if __name__ == "__main__":
    run()
