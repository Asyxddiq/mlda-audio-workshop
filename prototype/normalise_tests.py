"""
Tests for the text clean-up step (lstm_model.normalise / model_words), added 16 Oct.

    python prototype/normalise_tests.py

Needs only numpy (through lstm_model). Each line below is: what Whisper or a person
might write -> the words the model reads. The point of the clean-up is that the long
and short spellings of the same thing give the SAME words. Also checks the three
things the clean-up must NOT do: change a name + "will", change possession ("I have
the keys"), or change split_words (audio_check.py compares raw words on purpose, so
"I will" vs "I'll" still counts as a transcription difference there).
"""

import sys

from lstm_model import model_words, split_words

CASES = [
    # contractions: long form -> short form (what the training meetings use)
    ("I will bring the cash box.",           "i'll bring the cash box"),
    ("I'll bring the cash box.",             "i'll bring the cash box"),
    ("We will meet at noon.",                "we'll meet at noon"),
    ("They will not come.",                  "they won't come"),
    ("I do not know.",                       "i don't know"),
    ("It is not ready.",                     "it isn't ready"),
    ("We cannot start yet.",                 "we can't start yet"),
    ("That is settled.",                     "that's settled"),
    ("It is decided.",                       "it's decided"),
    ("We are going with option B.",          "we're going with option b"),
    ("I am happy with that.",                "i'm happy with that"),
    ("Let us begin.",                        "let's begin"),
    ("I would rather not.",                  "i'd rather not"),
    # spelling and filler sounds
    ("OK, sounds good.",                     "okay sounds good"),
    ("All right, let's go.",                 "alright let's go"),
    ("We’ll do it.",                    "we'll do it"),
    ("I'm gonna send it.",                   "i'm going to send it"),
    ("Um, so, uh, the budget.",              "so the budget"),
    # numbers and times are left alone (the shipped weights were trained on them as written)
    ("We have 600 dollars.",                 "we have 600 dollars"),
    ("The 21st at 5pm.",                     "the 21st at 5pm"),
    # things that must stay as they are
    ("Kai will handle it.",                  "kai will handle it"),
    ("Mei is going to help.",                "mei is going to help"),
    ("I have the keys.",                     "i have the keys"),
    ("Can you do it?",                       "can you do it ?"),
    ("One thing today.",                     "one thing today"),
]


def main() -> int:
    wrong = 0
    for said, expected in CASES:
        got = " ".join(model_words(said))
        if got != expected:
            wrong += 1
            print(f"  FAIL  {said!r}\n        model reads: {got!r}\n        expected:    {expected!r}")
    # the same sentence in both spellings must give identical words
    pairs = [("I will send it", "I'll send it"), ("Let us go", "Let's go"), ("OK then", "Okay then"),
             ("We do not agree", "We don't agree"), ("It is final", "It's final")]
    for long_form, short_form in pairs:
        if model_words(long_form) != model_words(short_form):
            wrong += 1
            print(f"  FAIL  {long_form!r} and {short_form!r} should read the same")
    # split_words (used by audio_check.py to compare transcripts) is untouched
    if split_words("I will go, OK?") != ["i", "will", "go", "ok", "?"]:
        wrong += 1
        print("  FAIL  split_words changed: audio_check.py needs the raw words")
    total = len(CASES) + len(pairs) + 1
    print(f"{total - wrong}/{total} clean-up checks pass")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
