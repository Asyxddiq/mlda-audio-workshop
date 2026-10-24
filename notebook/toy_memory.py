"""A toy memory with a STATED, fixed keep rate -- shared by the live blocks
3-5 (and reused by WP8's block 6 toy LSTM knob demo) so the short-vs-long
"how much of the first word survives" aha is visible and exactly predictable,
instead of an emergent property of random weights.

This is deliberately NOT how a real RNN computes its hidden state (no
learned W_x/W_h, no tanh squashing) -- it is a leaky integrator with one
number, KEEP_RATE, stated on screen every time it is used:

    h_t = KEEP_RATE * h_{t-1} + (1 - KEEP_RATE) * x_t

Because every word after the first is IDENTICAL between the two sequences
being compared (only the first word differs), the input terms cancel
exactly from word 2 on, so the gap between the two sequences' memories
decays as a PURE geometric series:

    gap(word n) = gap(word 1) * KEEP_RATE ** (n - 1)

This is a mathematical fact of the formula, not a regularity that happens
to hold most of the time -- it is identical for every word, every run,
checked over 500 random trials in this file's own self-test (no variance
at all, to floating-point precision). `hands_on.py`'s own toy RNN (Part A,
the amnesia cell) is untouched -- this is a separate, additional
illustration for the live cells only.
"""
import numpy as np

KEEP_RATE = 0.8  # a STATED toy setting, not learned -- see the module docstring

# The headline (block 5c) compares whole CONVERSATIONS, 10-100+ tokens long,
# not single sentences (block 3-5b, 5-16 words). At KEEP_RATE=0.8 an ordinary
# 2-3 turn exchange already decays to ~10-15%, reading as "mostly gone" next
# to a long conversation's ~0% -- not the "short keeps it, long loses it"
# contrast the headline is making. 0.9 keeps a short exchange clearly above
# the noise floor while a long conversation (80+ tokens) still decays to
# ~0%. Stated on screen wherever it's used, same as KEEP_RATE above.
HEADLINE_KEEP_RATE = 0.9


def step(word_vec: np.ndarray, h_prev: np.ndarray, keep_rate: float = KEEP_RATE) -> np.ndarray:
    """One step of the toy memory: h_t = keep_rate * h_prev + (1 - keep_rate) * word_vec."""
    return keep_rate * h_prev + (1 - keep_rate) * word_vec


def gap_curve(words_a: list, words_b: list, word_vector_fn, keep_rate: float = KEEP_RATE) -> list:
    """The memory gap between two word sequences, one step at a time.

    `words_a`/`words_b` should differ in at most the first word (see
    `first_word_pair` below for how the live cells build `words_b` so
    this holds even if Whisper misheard something later in either
    recording). `word_vector_fn` is normally `live_mic.word_vector`.
    """
    h_a = h_b = np.zeros(8 if not callable(word_vector_fn) else len(word_vector_fn(words_a[0])))
    gaps = []
    for wa, wb in zip(words_a, words_b):
        h_a = step(word_vector_fn(wa), h_a, keep_rate)
        h_b = step(word_vector_fn(wb), h_b, keep_rate)
        gaps.append(float(np.abs(h_a - h_b).mean()))
    return gaps


def first_word_pair(words_a: list, words_b_spoken: list) -> list:
    """Build a `words_b` that differs from `words_a` in ONLY the first word:
    `words_b_spoken`'s own first word, plus `words_a`'s own remaining words.
    `words_b_spoken` is only ever used for its first word -- whatever Whisper
    heard for the rest of that second recording is discarded, so a mis-heard
    tail can never show up as a false "difference" in the gap (see
    )."""
    return [words_b_spoken[0]] + list(words_a[1:])


def first_line_pair(tokens_a: list, alt_line_tokens: list, turn_token: str = "<eot>") -> list:
    """Line-level version of `first_word_pair`, for a whole conversation (WP9's
    F2(a)): swap only the FIRST turn's tokens for `alt_line_tokens`, keeping
    every later turn (including its own `<eot>` markers) exactly as spoken.
    If there is no `<eot>` in `tokens_a` (a one-turn recording), the whole
    thing counts as "line 1" and is replaced.

    `alt_line_tokens` is cycled or trimmed to line 1's own length first, so
    the two sequences always have the SAME length and every later word
    stays lined up word-for-word -- a fixed-length alt line of the wrong
    length would shift everything after it out of step, and `gap_curve`'s
    `zip` would silently drop the extra words on the longer side."""
    idx = tokens_a.index(turn_token) if turn_token in tokens_a else len(tokens_a)
    alt = list(alt_line_tokens) or ["<unk>"]
    alt = (alt * (idx // len(alt) + 1))[:idx]
    if turn_token in tokens_a:
        return alt + list(tokens_a[idx:])
    return alt


def line_boundary_index(tokens: list, turn_token: str = "<eot>") -> int:
    """Index of the first `<eot>` in `tokens` (end of line 1) -- the right
    reference point for a LINE-level swap (`first_line_pair`): line 1 can
    share its first word or two with the alternate line ("The meeting is on
    Friday..." vs "The meeting is on Monday..."), so the gap can be exactly 0
    at word 1 -- use the gap once the whole differing line has been read
    instead. Falls back to the last index if there's no `<eot>` (a one-line
    recording)."""
    return tokens.index(turn_token) if turn_token in tokens else len(tokens) - 1


def words_below(gaps: list, threshold_pct: float = 10.0) -> int | None:
    """The first word position (1-indexed) at which the gap, as a percentage
    of word 1's gap, drops below `threshold_pct`. None if it never does."""
    if not gaps or gaps[0] <= 0:
        return None
    for i, g in enumerate(gaps, start=1):
        if g / gaps[0] * 100 < threshold_pct:
            return i
    return None


if __name__ == "__main__":
    # Self-test: the geometric decay is exact, not just usually true -- run it
    # over many random word-vector sequences and confirm zero variance.
    import random
    import statistics
    import zlib

    def word_vector(word, dim=8):
        seed = zlib.crc32(word.encode("utf-8")) & 0xFFFFFFFF
        return np.random.default_rng(seed).normal(size=dim)

    rng = random.Random(0)
    vocab = ["the", "team", "decided", "to", "postpone", "launch", "until", "march", "red", "blue",
             "cars", "drive", "fast", "cats", "dogs", "window", "evening", "monday", "tuesday"]
    short_pcts, long_pcts = [], []
    for _ in range(500):
        words = [rng.choice(vocab) for _ in range(16)]
        words_b = first_word_pair(words, ["ALT"])
        g4 = gap_curve(words[:4], words_b[:4], word_vector)
        g16 = gap_curve(words, words_b, word_vector)
        short_pcts.append(g4[-1] / g4[0] * 100)
        long_pcts.append(g16[-1] / g16[0] * 100)
    print(f"KEEP_RATE = {KEEP_RATE}")
    print(f"short (4 words):  median {statistics.median(short_pcts):.1f}%, "
          f"range {min(short_pcts):.4f}%-{max(short_pcts):.4f}% over 500 trials")
    print(f"long (16 words):  median {statistics.median(long_pcts):.4f}%, "
          f"range {min(long_pcts):.6f}%-{max(long_pcts):.6f}% over 500 trials")
    assert max(short_pcts) - min(short_pcts) < 1e-6, "expected zero variance (exact geometric decay)"
    assert max(long_pcts) - min(long_pcts) < 1e-6, "expected zero variance (exact geometric decay)"
    print("self-test passed: the decay is exact, not just usually true")
