"""
LSTM prototype (Asy + Nomt) -- hourglass version.

What this file does
-------------------
Runs the WHOLE placeholder transcript through a 5-layer "hourglass" LSTM
in one continuous pass (state never reset between speakers) and prints a
hidden state at the end of each turn. Turning those hidden states into
actual meeting minutes is Nomt's next piece, built on top of this file.

Where the hourglass idea comes from
-----------------------------------
Li, Basit, Daraz & Jan (2024), "Deep causal speech enhancement and
recognition using efficient long-short term memory Recurrent Neural
Network", PLOS ONE 19(1): e0291240. https://doi.org/10.1371/journal.pone.0291240

Their model stacks 5 LSTM layers shaped like an hourglass (their Table 1):

    layer   units   time steps
      1      256       512
      2      512       256
      3     1024*      128      <- the narrow "neck" in time, widest in units
      4      512       256
      5      256       512
    (* printed as 1014 in the paper, almost certainly a typo for 1024,
       since every other layer is exactly double/half its neighbour)

Going down, every layer HALVES the number of time steps and DOUBLES the
units. Coming back up, it does the reverse. The trick that makes this
work without losing information: instead of throwing time steps away,
two neighbouring steps are glued side by side into one longer vector
("fold"), and later split back apart ("unfold"). Nothing is deleted, the
same numbers are just arranged differently. The paper also adds skip
connections from layer 1 to layer 5 and from layer 2 to layer 4, so the
detailed early features aren't forgotten by the time they reach the end.

What we simplified for the workshop
-----------------------------------
- Sizes are scaled down: 8 / 16 / 32 / 16 / 8 units instead of
  256 / 512 / 1024 / 512 / 256. Same doubling pattern, just small enough
  to read and to run instantly on a laptop or free Google Colab.
- The paper puts an "attention gate" on each skip connection to decide
  which parts matter most. We leave that out: our skips simply ADD the
  earlier layer's output back in. Easier to follow, same core idea.
- The paper cleans noisy speech audio. We apply the same shape of network
  to our transcript tokens instead. The architecture carries over; the
  task is ours.
- This file is the forward pass only. Training lives in train_lstm.py
  (plain NumPy backpropagation), which saves trained_lstm.npz;
  run_pipeline() loads it when it exists. The __main__ walk-through below
  still uses random weights, to show the shapes, folding and skips.

Why continuous across turns: the reason this workshop teaches LSTM is
long-range memory across MANY turns (e.g. remembering in turn 9 that
Priya flagged catering as pending in turn 4). That only works if state is
threaded through the whole conversation instead of wiped at each line.

Written in plain NumPy, no ML framework, so the gates stay visible as
ordinary maths. Runs as-is on Google Colab (numpy is preinstalled).
"""

import os
import re

import numpy as np

from transcriber import get_transcript  # Track 1; falls back to the placeholder when no audio is given


# ---------------------------------------------------------------------------
# 1. One LSTM layer (unchanged from before -- this is the teaching core)
# ---------------------------------------------------------------------------

class SimpleLSTM:
    """A minimal, single-layer LSTM, run manually over a sequence.

    Gates, standard LSTM formulation (the same equations as the paper's
    Eqs. 1-5):
        forget_gate  f_t = sigmoid(W_f . [h_{t-1}, x_t] + b_f)
        input_gate   i_t = sigmoid(W_i . [h_{t-1}, x_t] + b_i)
        candidate    c~_t = tanh(W_c . [h_{t-1}, x_t] + b_c)
        cell state   c_t = f_t * c_{t-1} + i_t * c~_t
        output_gate  o_t = sigmoid(W_o . [h_{t-1}, x_t] + b_o)
        hidden state h_t = o_t * tanh(c_t)
    """

    def __init__(self, input_size: int, hidden_size: int, seed: int = 0):
        self.input_size = input_size
        self.hidden_size = hidden_size

        rng = np.random.default_rng(seed)
        concat_size = input_size + hidden_size

        # One weight matrix + bias per gate. Small random init is enough
        # for a forward-pass demo; a trained version would learn these.
        scale = 1.0 / np.sqrt(concat_size)
        self.W_f = rng.uniform(-scale, scale, (hidden_size, concat_size))
        self.W_i = rng.uniform(-scale, scale, (hidden_size, concat_size))
        self.W_c = rng.uniform(-scale, scale, (hidden_size, concat_size))
        self.W_o = rng.uniform(-scale, scale, (hidden_size, concat_size))
        self.b_f = np.zeros(hidden_size)
        self.b_i = np.zeros(hidden_size)
        self.b_c = np.zeros(hidden_size)
        self.b_o = np.zeros(hidden_size)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def step(self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray):
        """One time step: one input vector in, (hidden, cell) state out."""
        combined = np.concatenate([h_prev, x_t])

        f_t = self._sigmoid(self.W_f @ combined + self.b_f)
        i_t = self._sigmoid(self.W_i @ combined + self.b_i)
        c_tilde = np.tanh(self.W_c @ combined + self.b_c)
        c_t = f_t * c_prev + i_t * c_tilde
        o_t = self._sigmoid(self.W_o @ combined + self.b_o)
        h_t = o_t * np.tanh(c_t)

        return h_t, c_t

    def forward(self, x_seq: np.ndarray) -> np.ndarray:
        """Run this layer over a whole sequence, carrying state throughout.

        x_seq:   shape (time_steps, input_size)
        returns: shape (time_steps, hidden_size), one hidden state per step
        """
        h_t = np.zeros(self.hidden_size)
        c_t = np.zeros(self.hidden_size)
        hidden_states = []

        for x_t in x_seq:
            h_t, c_t = self.step(x_t, h_t, c_t)
            hidden_states.append(h_t)

        return np.stack(hidden_states)


# ---------------------------------------------------------------------------
# 2. Fold / unfold: changing the number of time steps WITHOUT losing data
# ---------------------------------------------------------------------------

def fold_time(x: np.ndarray) -> np.ndarray:
    """Halve the time steps by gluing each pair of neighbouring steps together.

    (time_steps, features) -> (time_steps / 2, features * 2)

    Example with 4 steps of 2 numbers each:
        [[a, b],            [[a, b, c, d],
         [c, d],     ->      [e, f, g, h]]
         [e, f],
         [g, h]]
    Every number is still there, just in fewer, longer rows.

    Side effect worth knowing: once [a, b] and [c, d] share a row, the
    layer sees both at the same moment, so token 1 gets a peek at token 2.
    After two folds, a token can see up to 3 tokens ahead. The paper
    describes its model as "causal" (uses no future information) because
    it is meant for live audio. For meeting minutes we always have the
    full transcript anyway, so a small look-ahead is harmless here.
    """
    steps, features = x.shape
    return x.reshape(steps // 2, features * 2)


def unfold_time(x: np.ndarray) -> np.ndarray:
    """Exact reverse of fold_time: double the time steps, halve the features."""
    steps, features = x.shape
    return x.reshape(steps * 2, features // 2)


# ---------------------------------------------------------------------------
# 3. The hourglass: 5 stacked LSTM layers + 2 skip connections
# ---------------------------------------------------------------------------

class HourglassLSTM:
    """Five SimpleLSTM layers arranged like the paper's hourglass.

        input (T steps)
          |
        layer 1   T steps,    8 units  ----------------------+ skip
          | fold                                             |
        layer 2   T/2 steps, 16 units  -----------+ skip     |
          | fold                                  |          |
        layer 3   T/4 steps, 32 units  (neck)     |          |
          | unfold                                |          |
        layer 4   T/2 steps, 16 units  <----------+ (added)  |
          | unfold                                           |
        layer 5   T steps,    8 units  <---------------------+ (added)
          |
        output (T steps, 8 numbers per token)

    T must be divisible by 4 (two folds). pad_to_multiple_of_4() handles that.
    """

    def __init__(self, input_size: int, base_units: int = 8, seed: int = 0):
        u1, u2, u3 = base_units, base_units * 2, base_units * 4  # 8, 16, 32

        # Each layer's input size is whatever shape arrives after fold/unfold.
        self.layer1 = SimpleLSTM(input_size, u1, seed=seed)
        self.layer2 = SimpleLSTM(u1 * 2, u2, seed=seed + 1)   # folded layer-1 output
        self.layer3 = SimpleLSTM(u2 * 2, u3, seed=seed + 2)   # folded layer-2 output
        self.layer4 = SimpleLSTM(u3 // 2, u2, seed=seed + 3)  # unfolded layer-3 output
        self.layer5 = SimpleLSTM(u2 // 2, u1, seed=seed + 4)  # unfolded layer-4 output

    def forward(self, x_seq: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Run the full hourglass. Returns (final output, every layer's output)."""
        out1 = self.layer1.forward(x_seq)                  # (T,   8)
        out2 = self.layer2.forward(fold_time(out1))        # (T/2, 16)
        out3 = self.layer3.forward(fold_time(out2))        # (T/4, 32)  the neck

        # Skip connection 2 -> 4: add layer 2's output back in before layer 4.
        # (The paper puts an attention gate here; we just add.)
        in4 = unfold_time(out3) + out2                     # (T/2, 16)
        out4 = self.layer4.forward(in4)                    # (T/2, 16)

        # Skip connection 1 -> 5: add layer 1's output back in before layer 5.
        in5 = unfold_time(out4) + out1                     # (T,   8)
        out5 = self.layer5.forward(in5)                    # (T,   8)

        layers = {"layer 1": out1, "layer 2": out2, "layer 3": out3,
                  "layer 4": out4, "layer 5": out5}
        return out5, layers


# ---------------------------------------------------------------------------
# 4. Turning the transcript into numbers (tokenize + embed)
# ---------------------------------------------------------------------------

TURN_BOUNDARY_TOKEN = "<eot>"  # marks "end of turn" so the model can see speaker changes


def split_words(text: str) -> list[str]:
    """Lowercase words, with ? and ! kept as their own tokens.

    "Sure." and "sure" become the same word, and "Can you do it?" ends in a
    separate "?" token -- a useful signal that a turn is a request.
        "Can you check it?" -> ["can", "you", "check", "it", "?"]
    """
    return re.findall(r"[a-z0-9']+|[?!]", text.lower())


# Whisper writes "I will", "let us", "do not", "OK"; people (and our training meetings)
# mostly say "I'll", "let's", "don't", "okay". Same meaning, different words, and a model
# trained on one spelling is half-lost on the other (16 Oct: "I will bring..." was read as
# plain discussion, "I'll bring..." as an action item). So before the model reads any text,
# both spellings are turned into the same one. Applied in training AND live.
_CLEAN_UP = [
    (r"\b(?:um+|uh+|uhm+|erm+|ehm+)\b", " "),                       # filler sounds
    (r"\bo\.?k\.?\b", "okay"), (r"\ball right\b", "alright"),
    (r"\bgonna\b", "going to"), (r"\bwanna\b", "want to"), (r"\blemme\b", "let me"),
    (r"\bcannot\b", "can't"), (r"\bcan not\b", "can't"), (r"\bwill not\b", "won't"),
    (r"\b(do|does|did|is|are|was|were|has|have|had|would|could|should)\s+not\b", r"\1n't"),
    (r"\blet us\b", "let's"),
    (r"\b(i|you|he|she|it|we|they|that|there|who)\s+will\b", r"\1'll"),
    (r"\b(i|you|he|she|it|we|they|that|there)\s+would\b", r"\1'd"),
    (r"\bi\s+am\b", "i'm"),
    (r"\b(you|we|they)\s+are\b", r"\1're"),
    (r"\b(it|that|there|here|what|who|he|she|where|how)\s+is\b", r"\1's"),
]


def normalise(text: str) -> str:
    """Lowercase and unify spelling, so the model sees one form of each phrase.

        "OK, I will send it at 4."      ->  "okay, i'll send it at 4."
        "Let us not.  We do not know."  ->  "let's not.  we don't know."
    """
    text = text.lower().replace("’", "'").replace("‘", "'").replace("`", "'")
    for pattern, replacement in _CLEAN_UP:
        text = re.sub(pattern, replacement, text)
    return text


def model_words(text: str) -> list[str]:
    """What the model reads: split_words() after normalise()."""
    return re.findall(r"[a-z0-9']+|[?!]", normalise(text))


def tokenize_transcript(turns: list[tuple[str, str]]) -> tuple[list[str], list[int]]:
    """Flatten (speaker, text) turns into one long list of words, plus the
    position of the LAST token of each turn (used to read off the hidden
    state at each turn boundary).

    An <eot> token goes between turns so the sequence still carries a
    signal for "a new person is speaking now".
    """
    all_tokens: list[str] = []
    turn_end_indices: list[int] = []

    for i, (_speaker, text) in enumerate(turns):
        all_tokens.extend(model_words(text))
        if i < len(turns) - 1:
            all_tokens.append(TURN_BOUNDARY_TOKEN)
        turn_end_indices.append(len(all_tokens) - 1)

    return all_tokens, turn_end_indices


def build_vocab(tokens: list[str]) -> dict[str, int]:
    """Give every unique word its own number: {"let's": 0, "kick": 1, ...}."""
    vocab: dict[str, int] = {}
    for word in tokens:
        vocab.setdefault(word, len(vocab))
    return vocab


def embed_tokens(tokens: list[str], vocab: dict[str, int], embeddings: np.ndarray) -> np.ndarray:
    """Swap each word for its row of numbers in the embedding table."""
    return embeddings[[vocab[word] for word in tokens]]


def pad_to_multiple_of_4(x_seq: np.ndarray) -> np.ndarray:
    """Add zero rows at the END so the length divides by 4 (needed for two
    folds). Padding goes after the real tokens, and the LSTM only reads
    forwards, so it cannot change any real token's result in layer 1."""
    remainder = len(x_seq) % 4
    if remainder == 0:
        return x_seq
    padding = np.zeros((4 - remainder, x_seq.shape[1]))
    return np.vstack([x_seq, padding])


# ---------------------------------------------------------------------------
# 5. Importable hand-off for meeting_minutes.py (Nomt)
# ---------------------------------------------------------------------------

def run_pipeline(audio_path: str | None = None,
                 transcript: str | None = None,
                 use_trained: bool = True,
                 ) -> tuple[list[tuple[str, str]], np.ndarray, list[int]]:
    """Run the pipeline once and return the hand-off contract from progress.md:
    turns, the (num_tokens, 8) hourglass output, and turn_end_indices.

    Mirrors the __main__ block below as an importable function instead of a
    script, so meeting_minutes.py can call this instead of re-running main.

    Input, in order of preference:
      - transcript: an already-made "Speaker: text" transcript (skips Track 1)
      - audio_path: an audio file, transcribed by transcriber.py (Whisper)
      - neither:    the placeholder transcript (transcriber.py falls back to it)

    Weights: the trained ones from trained_lstm.npz if train_lstm.py has
    been run (and use_trained is True), otherwise random ones as before.
    """
    if transcript is None:
        transcript = get_transcript(audio_path)
    turns = [
        (line.split(":", 1)[0].strip(), line.split(":", 1)[1].strip())
        for line in transcript.strip().splitlines()
        if ":" in line and line.split(":", 1)[1].strip()
    ]
    if not turns:
        raise ValueError(
            "No speaker turns found in the transcript (expected lines like "
            "'Speaker1: text'). Was the audio silent or too short?"
        )

    EMBEDDING_DIM = 16
    BASE_UNITS = 8

    all_tokens, turn_end_indices = tokenize_transcript(turns)

    trained = load_trained_model() if use_trained else None
    if trained is not None:
        # Trained: fixed vocab from training; words it never saw -> <unk> (id 0)
        vocab = {word: i for i, word in enumerate(trained["vocab"])}
        x_seq = trained["embeddings"][[vocab.get(word, 0) for word in all_tokens]]
        model = trained["model"]
    else:
        # Untrained: random embeddings and random LSTM weights (the original demo)
        vocab = build_vocab(all_tokens)
        rng = np.random.default_rng(42)
        embeddings = rng.uniform(-0.5, 0.5, (len(vocab), EMBEDDING_DIM))
        x_seq = embed_tokens(all_tokens, vocab, embeddings)
        model = HourglassLSTM(input_size=EMBEDDING_DIM, base_units=BASE_UNITS)

    real_length = len(x_seq)
    x_padded = pad_to_multiple_of_4(x_seq)
    output, _layers = model.forward(x_padded)
    output = output[:real_length]

    return turns, output, turn_end_indices


# ---------------------------------------------------------------------------
# 5b. Trained weights (made by train_lstm.py)
# ---------------------------------------------------------------------------

TRAINED_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_lstm.npz")


def _load_member(saved, prefix: str) -> dict:
    """One trained model's arrays (prefix "" for a single model, "m0_" etc.
    for an ensemble member) copied into a fresh HourglassLSTM."""
    model = HourglassLSTM(int(saved["embedding_dim"]), int(saved["base_units"]))
    layers = [model.layer1, model.layer2, model.layer3, model.layer4, model.layer5]
    for n, layer in enumerate(layers, start=1):
        for gate in ("f", "i", "c", "o"):
            setattr(layer, f"W_{gate}", saved[f"{prefix}layer{n}_W_{gate}"].astype(np.float64))
            setattr(layer, f"b_{gate}", saved[f"{prefix}layer{n}_b_{gate}"].astype(np.float64))
    return {
        "model": model,
        "embeddings": saved[f"{prefix}embeddings"].astype(np.float64),
        "head_W": saved[f"{prefix}head_W"].astype(np.float64),
        "head_b": saved[f"{prefix}head_b"].astype(np.float64),
    }


def load_trained_model(path: str = TRAINED_WEIGHTS) -> dict | None:
    """Load train_lstm.py's output, or return None if it hasn't been run.

    The file holds the vocab, the embedding table, every gate of the 5
    layers, and the classifier head, in exactly the shapes this file uses,
    so loading is just copying arrays into a fresh HourglassLSTM.

    Since 13 Oct the file can hold an ENSEMBLE of several trained models
    (train_lstm.py, ENSEMBLE = True). "members" lists all of them; the
    top-level "model", "embeddings", "head_W", "head_b" are member 0, so
    code that only wants one model keeps working.
    """
    if not os.path.isfile(path):
        return None
    saved = np.load(path)
    if "members" in saved.files:
        members = [_load_member(saved, f"m{n}_") for n in range(int(saved["members"]))]
    else:
        members = [_load_member(saved, "")]
    return {
        **members[0],
        "members": members,
        "vocab": [str(word) for word in saved["vocab"]],
        "labels": [str(label) for label in saved["labels"]],
    }


def label_probabilities(turns: list[tuple[str, str]], trained: dict | None = None) -> np.ndarray | None:
    """Probability of each label for each turn, averaged over every model in
    trained_lstm.npz (one row per turn, columns in trained["labels"] order).
    Returns None if there are no trained weights."""
    trained = trained or load_trained_model()
    if trained is None:
        return None
    vocab = {word: i for i, word in enumerate(trained["vocab"])}
    tokens, ends = tokenize_transcript(turns)
    ids = [vocab.get(word, 0) for word in tokens]
    all_probs = []
    for member in trained["members"]:
        x = member["embeddings"][ids]
        output, _layers = member["model"].forward(pad_to_multiple_of_4(x))
        scores = output[:len(x)][ends] @ member["head_W"].T + member["head_b"]
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        all_probs.append(e / e.sum(axis=1, keepdims=True))
    return np.mean(all_probs, axis=0)


# ---------------------------------------------------------------------------
# 6. Run it
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    transcript = get_transcript()
    turns = [
        (line.split(":", 1)[0].strip(), line.split(":", 1)[1].strip())
        for line in transcript.strip().splitlines()
        if ":" in line
    ]

    EMBEDDING_DIM = 16
    BASE_UNITS = 8

    all_tokens, turn_end_indices = tokenize_transcript(turns)
    vocab = build_vocab(all_tokens)

    rng = np.random.default_rng(42)
    embeddings = rng.uniform(-0.5, 0.5, (len(vocab), EMBEDDING_DIM))

    x_seq = embed_tokens(all_tokens, vocab, embeddings)
    real_length = len(x_seq)
    x_padded = pad_to_multiple_of_4(x_seq)

    # Quick proof that folding really loses nothing:
    demo = rng.uniform(size=(8, 4))
    assert np.array_equal(unfold_time(fold_time(demo)), demo)

    model = HourglassLSTM(input_size=EMBEDDING_DIM, base_units=BASE_UNITS)
    output, layers = model.forward(x_padded)
    output = output[:real_length]  # drop the padding rows again

    print(f"Vocab size: {len(vocab)} words (incl. {TURN_BOUNDARY_TOKEN})")
    print(f"Tokens in conversation: {real_length} "
          f"(padded to {len(x_padded)} so it divides by 4)")
    print("Fold -> unfold round trip is lossless: True\n")

    print("Hourglass shape (compare with the paper's Table 1):")
    print(f"  {'layer':<9}{'units':>7}{'time steps':>13}")
    for name, out in layers.items():
        steps, units = out.shape
        print(f"  {name:<9}{units:>7}{steps:>13}")
    print()

    for (speaker, text), end_idx in zip(turns, turn_end_indices):
        h = output[end_idx]
        print(f"[{speaker}] {text}")
        print(f"  -> output after this turn (first 4 of 8 dims): {np.round(h[:4], 3)}")

    print(
        "\nThis walk-through uses RANDOM starting weights, so these values are "
        "not meaningful content-wise. What it shows is the full hourglass running "
        "end to end over all 9 turns: shapes fold and unfold correctly, skips "
        "line up, and state carries through the whole conversation. "
        "train_lstm.py trains these weights; run_pipeline() and pipeline.py "
        "use the trained ones."
    )
