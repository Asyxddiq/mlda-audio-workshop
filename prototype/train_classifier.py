"""
Training scaffold for the meeting-minutes classifier (Asy + Nomt).

STATUS: scaffold, not the real trained model. One meeting (the same 9-turn
sample transcript everything else in this repo uses), hand-labelled, just
to prove the training loop runs, the loss drops, and to time one epoch
before Weekend 2. Weekend 2 replaces GOLD_LABELS with ~100 turns across
several meetings, split by meeting into train/held-out.

Why this is a separate file in PyTorch instead of extending lstm_model.py
--------------------------------------------------------------------------
lstm_model.py stays NumPy, forward-pass only, and untouched -- it's the
readable teaching artifact. This file trains a classifier with the exact
same 4-gate cell (W_f, W_i, W_c, W_o), just re-expressed with
torch.nn.Parameter so autograd can differentiate through it. Deliberately
NOT using nn.LSTM: it packs all four gates into one matrix with two bias
vectors, which makes porting trained weights back into the NumPy version
fiddly. Mirroring the cell exactly means porting is a straight copy.

torch.set_default_dtype(torch.float64) matches the NumPy side's float64,
so a later weight port can be checked for exact parity instead of just
"close enough".

What this does NOT claim
-------------------------
With ~100 training turns and random-until-trained embeddings, this learns
token-ID-to-label associations for words it has SEEN in training, not
general sentence meaning. Words never seen in training map to <unk> and
get no useful vector -- that's the honest limitation, not a bug to hide.
Real generalisation to unseen phrasing would need pretrained embeddings
or much more data; out of scope here.

How to run
----------
Needs `pip install torch` (Colab already has it). Run:
    python prototype/train_classifier.py
"""

import time

import torch
import torch.nn as nn

from lstm_model import run_pipeline, tokenize_transcript

LABELS = ["discussion", "decision", "action_item"]
UNK_TOKEN = "<unk>"

# Hand-labelled for the current 9-turn sample transcript. Matches the
# categories label_turns_stub() uses in meeting_minutes.py, so both files
# agree on what "this meeting" actually looks like.
GOLD_LABELS = [
    "discussion", "discussion", "discussion", "decision",
    "action_item", "discussion", "discussion", "discussion", "decision",
]

# Kept only as a comparison baseline for this training script -- NOT used
# in meeting_minutes.py, which still runs on label_turns_stub().
_DECISION_KEYWORDS = ["confirmed", "agreed", "approved", "decided", "noted"]
_ACTION_KEYWORDS = ["can you", "could you", "please", "follow up"]


# ---------------------------------------------------------------------------
# 1. Vocab with <unk> -- Weekend 2's held-out meetings will have words this
#    scaffold's single-meeting vocab has never seen. Without <unk>, an unseen
#    word is an out-of-range embedding index, i.e. a crash, not bad output.
# ---------------------------------------------------------------------------

def build_vocab_with_unk(tokens: list[str]) -> dict[str, int]:
    vocab = {UNK_TOKEN: 0}
    for word in tokens:
        vocab.setdefault(word, len(vocab))
    return vocab


def tokens_to_ids(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    unk_id = vocab[UNK_TOKEN]
    return [vocab.get(word, unk_id) for word in tokens]


# ---------------------------------------------------------------------------
# 2. The same 4-gate cell as SimpleLSTM.step() in lstm_model.py, re-expressed
#    with nn.Parameter so autograd can train it. Four SEPARATE bias
#    parameters -- chained assignment (a = b = c = nn.Parameter(...)) would
#    silently tie all four gates to one shared bias.
# ---------------------------------------------------------------------------

class LSTMCellTorch(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size
        scale = 1.0 / concat_size ** 0.5

        def gate_weights() -> nn.Parameter:
            return nn.Parameter(torch.empty(hidden_size, concat_size).uniform_(-scale, scale))

        self.W_f = gate_weights()
        self.W_i = gate_weights()
        self.W_c = gate_weights()
        self.W_o = gate_weights()
        self.b_f = nn.Parameter(torch.zeros(hidden_size))
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        self.b_c = nn.Parameter(torch.zeros(hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))

    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        combined = torch.cat([h_prev, x_t])
        f_t = torch.sigmoid(self.W_f @ combined + self.b_f)
        i_t = torch.sigmoid(self.W_i @ combined + self.b_i)
        c_tilde = torch.tanh(self.W_c @ combined + self.b_c)
        c_t = f_t * c_prev + i_t * c_tilde
        o_t = torch.sigmoid(self.W_o @ combined + self.b_o)
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Fresh (h, c) = 0 every call -- this is the state reset between
        meetings that training relies on. Within one call, state carries
        across every step, same as SimpleLSTM.forward()."""
        h_t = torch.zeros(self.hidden_size, dtype=x_seq.dtype)
        c_t = torch.zeros(self.hidden_size, dtype=x_seq.dtype)
        hidden_states = []
        for x_t in x_seq:
            h_t, c_t = self.step(x_t, h_t, c_t)
            hidden_states.append(h_t)
        return torch.stack(hidden_states)


def fold_time(x: torch.Tensor) -> torch.Tensor:
    steps, features = x.shape
    return x.reshape(steps // 2, features * 2)


def unfold_time(x: torch.Tensor) -> torch.Tensor:
    steps, features = x.shape
    return x.reshape(steps * 2, features // 2)


def pad_to_multiple_of_4(x: torch.Tensor) -> torch.Tensor:
    steps, features = x.shape
    remainder = steps % 4
    if remainder == 0:
        return x
    padding = torch.zeros(4 - remainder, features, dtype=x.dtype)
    return torch.cat([x, padding], dim=0)


class HourglassTorch(nn.Module):
    """Same 5-layer shape as HourglassLSTM in lstm_model.py: 8/16/32/16/8
    units, two folds down, two unfolds up, skips 1->5 and 2->4."""

    def __init__(self, input_size: int, base_units: int = 8):
        super().__init__()
        u1, u2, u3 = base_units, base_units * 2, base_units * 4

        self.layer1 = LSTMCellTorch(input_size, u1)
        self.layer2 = LSTMCellTorch(u1 * 2, u2)
        self.layer3 = LSTMCellTorch(u2 * 2, u3)
        self.layer4 = LSTMCellTorch(u3 // 2, u2)
        self.layer5 = LSTMCellTorch(u2 // 2, u1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        out1 = self.layer1(x_seq)
        out2 = self.layer2(fold_time(out1))
        out3 = self.layer3(fold_time(out2))

        in4 = unfold_time(out3) + out2
        out4 = self.layer4(in4)

        in5 = unfold_time(out4) + out1
        out5 = self.layer5(in5)

        return out5


# ---------------------------------------------------------------------------
# 3. Embed (trainable) -> hourglass -> read off turn ends -> classify
# ---------------------------------------------------------------------------

class MinutesClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 16, base_units: int = 8,
                 n_labels: int = len(LABELS)):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.hourglass = HourglassTorch(embed_dim, base_units)
        self.head = nn.Linear(base_units, n_labels)

    def forward(self, token_ids: torch.Tensor, turn_end_indices: list[int]) -> torch.Tensor:
        """token_ids: shape (T,) for ONE meeting, unpadded."""
        x = self.embed(token_ids)                       # (T, embed_dim)
        real_length = x.shape[0]
        x_padded = pad_to_multiple_of_4(x)
        per_token = self.hourglass(x_padded)[:real_length]
        turn_vecs = per_token[turn_end_indices]          # (n_turns, base_units)
        return self.head(turn_vecs)                      # (n_turns, n_labels)


# ---------------------------------------------------------------------------
# 4. Baselines -- what the trained model has to beat
# ---------------------------------------------------------------------------

def majority_baseline_accuracy(labels: torch.Tensor) -> float:
    counts = torch.bincount(labels, minlength=len(LABELS))
    majority = counts.argmax()
    return (labels == majority).float().mean().item()


def keyword_baseline_labels(turns: list[tuple[str, str]]) -> list[str]:
    predictions = []
    for _speaker, text in turns:
        lowered = text.lower()
        if any(keyword in lowered for keyword in _ACTION_KEYWORDS):
            predictions.append("action_item")
        elif any(keyword in lowered for keyword in _DECISION_KEYWORDS):
            predictions.append("decision")
        else:
            predictions.append("discussion")
    return predictions


def keyword_baseline_accuracy(turns: list[tuple[str, str]], gold_labels: list[str]) -> float:
    predictions = keyword_baseline_labels(turns)
    correct = sum(pred == gold for pred, gold in zip(predictions, gold_labels))
    return correct / len(gold_labels)


# ---------------------------------------------------------------------------
# 5. Build one meeting's training example
# ---------------------------------------------------------------------------

def prepare_meeting(turns: list[tuple[str, str]], vocab: dict[str, int],
                     gold_labels: list[str]) -> dict:
    tokens, turn_end_indices = tokenize_transcript(turns)
    token_ids = tokens_to_ids(tokens, vocab)
    label_ids = [LABELS.index(label) for label in gold_labels]
    return {
        "token_ids": torch.tensor(token_ids, dtype=torch.long),
        "turn_end_indices": turn_end_indices,
        "labels": torch.tensor(label_ids, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# 6. Train -- one forward call per meeting resets state, matching "reset
#    between meetings, carry state within one".
# ---------------------------------------------------------------------------

def train(model: MinutesClassifier, meetings: list[dict], epochs: int = 50, lr: float = 0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        start = time.perf_counter()
        optimizer.zero_grad()

        all_logits, all_labels = [], []
        for meeting in meetings:
            logits = model(meeting["token_ids"], meeting["turn_end_indices"])
            all_logits.append(logits)
            all_labels.append(meeting["labels"])
        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)

        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        elapsed = time.perf_counter() - start
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
        print(f"epoch {epoch:2d}  loss {loss.item():.3f}  acc {accuracy:.2f}  ({elapsed:.3f}s)")


if __name__ == "__main__":
    # Match lstm_model.py's float64 so a later weight port can be checked
    # for exact parity instead of a tolerance. Must happen before any
    # parameters are created.
    torch.set_default_dtype(torch.float64)

    # Reuse lstm_model.py's own turn-parsing (via run_pipeline) so this
    # script's idea of "the transcript" can never drift from the NumPy
    # side's. The NumPy output itself is unused here -- we're training a
    # separate, differentiable copy of the same architecture.
    turns, _np_output, _np_turn_end_indices = run_pipeline()

    tokens, _turn_end_indices = tokenize_transcript(turns)
    vocab = build_vocab_with_unk(tokens)

    # SCAFFOLD: one meeting. Weekend 2: build a list of meetings here, each
    # from its own transcript + gold labels, then split the list BY MEETING
    # into train / held-out (never split individual turns across the two).
    meeting = prepare_meeting(turns, vocab, GOLD_LABELS)
    meetings = [meeting]

    torch.manual_seed(0)
    model = MinutesClassifier(vocab_size=len(vocab))

    print(f"Vocab size (incl. {UNK_TOKEN}): {len(vocab)}")
    print(f"Turns: {len(turns)}, labels: {GOLD_LABELS}\n")

    print(f"Majority-class baseline accuracy: {majority_baseline_accuracy(meeting['labels']):.2f}")
    print(f"Keyword-rule baseline accuracy:   {keyword_baseline_accuracy(turns, GOLD_LABELS):.2f}")
    print(
        "(Both baselines are measured on the TRAINING meeting here, since "
        "there's no held-out meeting yet -- that comparison only means "
        "something once Weekend 2 adds more meetings to split.)\n"
    )

    train(model, meetings, epochs=50, lr=0.01)

    print(
        "\nThis is a scaffold on one hand-labelled meeting: it proves the "
        "loop runs, the loss drops, and times an epoch. It is not evidence "
        "the trained classifier beats keyword rules -- that comparison "
        "needs held-out meetings the model wasn't trained on."
    )
