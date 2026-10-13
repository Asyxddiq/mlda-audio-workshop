"""
Hands-on: the participants' part of the workshop (agenda blocks 4 and 8).

    python prototype/hands_on.py              # run your version
    python prototype/hands_on.py --solution   # run the reference answers

Two small pieces of the model are left for you to write. Everything else
(training, the hourglass, the minutes) is already done, so each piece is
one line of maths, and a check tells you straight away if it's right.

PART A  (block 4)  The RNN step: build the memory, then break it
    1. Fill in rnn_step() below. Run the file: the check should say OK.
    2. Set MEMORY_ON = False ("amnesia after every word") and run again.
       Watch the model stop noticing the first word.

PART B  (block 8)  The LSTM notebook update, then play with the eraser
    1. Fill in lstm_cell_update() below. Run: the check should say OK.
       Your line now runs INSIDE the trained model, so if it's wrong the
       model's scores collapse.
    2. Turn FORGET_KNOB. It is added to the forget gate: negative = "erase
       a lot", positive = "keep a lot". Watch whether the notebook still
       holds the first word at the end, and what the same knob does to the
       trained model's test score.

Runs on free Google Colab or any laptop; numpy only.
"""

import sys

import numpy as np

SOLUTION = "--solution" in sys.argv

# ===========================================================================
# PART A: one RNN step
# ===========================================================================

MEMORY_ON = True   # set to False for "amnesia after every word"


def rnn_step(x, h_prev, W_x, W_h, b):
    """One step of a simple RNN.

    x       this word, as numbers          (input_size,)
    h_prev  the memory so far              (hidden_size,)
    returns the new memory                 (hidden_size,)

    new memory = mix of (this word) + (old memory):
        h = tanh(W_x . x + W_h . h_prev + b)
    In numpy, "." is @ and tanh is np.tanh.
    If MEMORY_ON is False, leave out the old-memory part (W_h . h_prev).
    """
    # TODO: write the line above in numpy and return it.
    return None


# ===========================================================================
# PART B: the LSTM notebook (cell state) update
# ===========================================================================

FORGET_KNOB = 0.0  # try -6, -3, 0, +3, +6


def lstm_cell_update(f, i, g, c_prev):
    """The notebook update inside every LSTM step.

    f       eraser (forget gate), between 0 and 1: how much old notebook to keep
    i       pencil (input gate), between 0 and 1: how much new note to write
    g       the new note itself (candidate), between -1 and 1
    c_prev  the old notebook

    notebook = (what you keep) x (old notebook) + (what you add) x (new note)
        c = f * c_prev + i * g
    """
    # TODO: write the line above in numpy and return it.
    return None


# ===========================================================================
# Everything below runs your two functions. No need to edit it.
# ===========================================================================

if SOLUTION:
    import hands_on_reference as _ref

    def rnn_step(x, h_prev, W_x, W_h, b):  # noqa: F811
        return _ref.rnn_step(x, h_prev, W_x, W_h, b, memory_on=MEMORY_ON)

    lstm_cell_update = _ref.lstm_cell_update  # noqa: F811


def _check(name, got, expected) -> bool:
    if got is None:
        print(f"  {name}: not written yet (it returned None)")
        return False
    ok = np.allclose(np.asarray(got, dtype=float), expected, atol=1e-6)
    print(f"  {name}: {'OK' if ok else 'not quite'}  got {np.round(got, 4)}, expected {np.round(expected, 4)}")
    return ok


def part_a():
    print("PART A  the RNN step")
    rng = np.random.default_rng(0)
    W_x, W_h, b = rng.normal(size=(2, 3)), rng.normal(size=(2, 2)), rng.normal(size=2)
    x, h_prev = rng.normal(size=3), rng.normal(size=2)
    expected = (np.tanh(W_x @ x + W_h @ h_prev + b) if MEMORY_ON else np.tanh(W_x @ x + b))
    if not _check(f"check (MEMORY_ON = {MEMORY_ON})", rnn_step(x, h_prev, W_x, W_h, b), expected):
        print("  Fix rnn_step() first, then run again.\n")
        return

    # Two sentences that differ ONLY in the first word. Does the memory at
    # the end still know which first word it saw?
    rng = np.random.default_rng(1)
    hidden = 8
    W_x, W_h, b = rng.normal(0, 0.5, (hidden, 4)), rng.normal(0, 0.25, (hidden, hidden)), np.zeros(hidden)
    words_a = rng.normal(size=(12, 4))
    words_b = words_a.copy()
    words_b[0] = rng.normal(size=4)                     # only the first word is different
    h_a, h_b = np.zeros(hidden), np.zeros(hidden)
    gaps = []
    for x_a, x_b in zip(words_a, words_b):
        h_a, h_b = rnn_step(x_a, h_a, W_x, W_h, b), rnn_step(x_b, h_b, W_x, W_h, b)
        gaps.append(np.abs(h_a - h_b).sum())
    print(f"\n  Two 12-word sentences, different first word only. How different is the")
    print(f"  memory after each word? (0 = the first word is forgotten)")
    print("  word  " + " ".join(f"{n:>5d}" for n in range(1, 13)))
    print("  gap   " + " ".join(f"{g:5.2f}" for g in gaps))
    if MEMORY_ON:
        kept = np.mean([gaps[k + 1] / gaps[k] for k in range(len(gaps) - 1) if gaps[k] > 1e-12])
        print(f"  The gap shrinks each step (about {kept:.0%} kept per step): the memory fades.")
        print("  Now set MEMORY_ON = False and run again.\n")
    else:
        print("  With no old-memory term, the gap is 0 from word 2 on: the first word is gone.")
        print("  The memory really lives in that one term, W_h . h_prev.\n")


def _patch_lstm():
    """Make every LSTM step in lstm_model.py use YOUR lstm_cell_update()."""
    import lstm_model

    def step(self, x_t, h_prev, c_prev):
        combined = np.concatenate([h_prev, x_t])
        f_t = self._sigmoid(self.W_f @ combined + self.b_f)
        i_t = self._sigmoid(self.W_i @ combined + self.b_i)
        c_tilde = np.tanh(self.W_c @ combined + self.b_c)
        c_t = lstm_cell_update(f_t, i_t, c_tilde, c_prev)
        o_t = self._sigmoid(self.W_o @ combined + self.b_o)
        return o_t * np.tanh(c_t), c_t

    lstm_model.SimpleLSTM.step = step


def _member_predict(member, vocab, turns):
    """Label probabilities for each turn from ONE trained model."""
    from lstm_model import pad_to_multiple_of_4, tokenize_transcript
    tokens, ends = tokenize_transcript([(t[0], t[1]) for t in turns])
    x = member["embeddings"][[vocab.get(w, 0) for w in tokens]]
    out, _ = member["model"].forward(pad_to_multiple_of_4(x))
    scores = out[:len(x)][ends] @ member["head_W"].T + member["head_b"]
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def part_b():
    print("PART B  the LSTM notebook update")
    rng = np.random.default_rng(0)
    f, i = rng.uniform(size=4), rng.uniform(size=4)
    g, c_prev = rng.uniform(-1, 1, size=4), rng.uniform(-1, 1, size=4)
    if not _check("check", lstm_cell_update(f, i, g, c_prev), f * c_prev + i * g):
        print("  Fix lstm_cell_update() first, then run again.")
        return
    _patch_lstm()                                        # every LSTM step now uses your line
    from lstm_model import SimpleLSTM

    # 1. The same two sentences trick as Part A, now with one small LSTM
    #    layer and your notebook update: does the notebook at the end still
    #    know which first word it saw?
    layer = SimpleLSTM(4, 8, seed=3)
    layer.b_f = layer.b_f + FORGET_KNOB                  # turn the eraser knob
    rng = np.random.default_rng(1)
    words_a = rng.normal(size=(12, 4))
    words_b = words_a.copy()
    words_b[0] = rng.normal(size=4)                      # only the first word is different
    h_a = h_b = c_a = c_b = np.zeros(8)
    gaps = []
    for x_a, x_b in zip(words_a, words_b):
        h_a, c_a = layer.step(x_a, h_a, c_a)
        h_b, c_b = layer.step(x_b, h_b, c_b)
        gaps.append(np.abs(c_a - c_b).sum())
    print(f"\n  FORGET_KNOB = {FORGET_KNOB:+.0f}  (added to the forget gate)")
    print("  Two 12-word sentences, different first word only. How different is the")
    print("  NOTEBOOK after each word? (0 = the first word has been erased)")
    print("  word  " + " ".join(f"{n:>5d}" for n in range(1, 13)))
    print("  gap   " + " ".join(f"{x:5.2f}" for x in gaps))

    # 2. The same knob on the TRAINED model (one of the ensemble's members),
    #    scored on the 3 test meetings.
    from lstm_model import load_trained_model
    from training_data import MEETINGS
    trained = load_trained_model()
    if trained is None:
        print("  No trained_lstm.npz yet: run python prototype/train_lstm.py first.")
        return
    member = trained["members"][0]
    for n in range(1, 6):
        trained_layer = getattr(member["model"], f"layer{n}")
        trained_layer.b_f = trained_layer.b_f + FORGET_KNOB
    vocab = {w: k for k, w in enumerate(trained["vocab"])}
    right = total = 0
    for m in [m for m in MEETINGS if m["split"] == "test"]:
        pred = _member_predict(member, vocab, m["turns"]).argmax(axis=1)
        right += sum(trained["labels"][p] == t[2] for p, t in zip(pred, m["turns"]))
        total += len(m["turns"])
    print(f"\n  Trained model with the same knob, on the test meetings: "
          f"{right}/{total} turns right ({right / total:.2f})")
    print("  Try -6 (erase a lot: the gap dies at once) and +6 (keep a lot: the first")
    print("  word is still there at the end). The trained model learned its own eraser")
    print("  settings at 0, so pushing the knob either way makes its labels worse.")


if __name__ == "__main__":
    part_a()
    part_b()
