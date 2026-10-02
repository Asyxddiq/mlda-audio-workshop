"""
Part 5 teaching demo: how an RNN turns audio frames into text.

Plain NumPy, forward pass only (untrained, random weights), same style as
lstm_model.py. The point is the SHAPES and the CTC decoding idea, not accuracy.

Pipeline:  audio frames (T x F)  ->  RNN  ->  per-frame letter probabilities
           ->  CTC greedy decode  ->  text

Participants complete ONE block: ctc_greedy_decode (marked TODO(participant)).
Everything else is given. A reference solution is at the bottom.
"""
import numpy as np

CHARS = ["<blank>"] + list("abcdefghijklmnopqrstuvwxyz '")  # index 0 = CTC blank
F = 13         # features per frame (MFCC-sized)
H = 32         # hidden units
rng = np.random.default_rng(42)


class SimpleRNN:
    """h_t = tanh(W_x x_t + W_h h_(t-1) + b), then a softmax over characters."""

    def __init__(self, n_in, n_hidden, n_out):
        self.W_x = rng.normal(0, 0.3, (n_hidden, n_in))
        self.W_h = rng.normal(0, 0.3, (n_hidden, n_hidden))
        self.b = np.zeros(n_hidden)
        self.W_o = rng.normal(0, 0.3, (n_out, n_hidden))
        self.b_o = np.zeros(n_out)

    def forward(self, frames):
        """frames: (T, F) -> probs: (T, len(CHARS)). One row per audio frame."""
        h = np.zeros(self.W_h.shape[0])
        probs = []
        for x_t in frames:                       # one frame at a time
            h = np.tanh(self.W_x @ x_t + self.W_h @ h + self.b)
            logits = self.W_o @ h + self.b_o
            e = np.exp(logits - logits.max())    # stable softmax
            probs.append(e / e.sum())
        return np.array(probs)


def ctc_greedy_decode(probs):
    """Turn per-frame probabilities into a string.

    CTC rules:
      1. Take the most likely symbol at every frame.
      2. Merge consecutive repeats  (h h e e l l l o -> h e l o).
      3. Remove blanks (index 0).
    The blank is what lets 'll' in 'hello' survive: l <blank> l.
    """
    # ---- TODO(participant): implement the 3 rules above -----------------
    best = probs.argmax(axis=1)
    out, prev = [], None
    for idx in best:
        if idx != prev and idx != 0:
            out.append(CHARS[idx])
        prev = idx
    return "".join(out)
    # ---- end TODO ---------------------------------------------------------


if __name__ == "__main__":
    frames = rng.normal(size=(50, F))            # fake "audio": 50 frames
    model = SimpleRNN(F, H, len(CHARS))
    probs = model.forward(frames)
    print("frames in:", frames.shape, "-> probs out:", probs.shape)
    print("each row sums to 1:", np.allclose(probs.sum(axis=1), 1))
    print("decoded (random weights, so gibberish):", repr(ctc_greedy_decode(probs)))

    # Sanity check the decoder on a hand-made example: h h e <b> l l <b> l o -> "hello"
    def onehot(seq):
        m = np.zeros((len(seq), len(CHARS)))
        for t, c in enumerate(seq):
            m[t, c] = 1
        return m
    ix = {c: i for i, c in enumerate(CHARS)}
    demo = onehot([ix["h"], ix["h"], ix["e"], 0, ix["l"], ix["l"], 0, ix["l"], ix["o"]])
    assert ctc_greedy_decode(demo) == "hello"
    print("decoder check passed: 'hello'")