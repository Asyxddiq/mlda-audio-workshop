"""
Audio test: does what you SAID come out the same as what you TYPED? (Track 1 check)

    python prototype/audio_check.py prototype/audio_tests/1_sample_meeting.txt meeting.m4a
    python prototype/audio_check.py prototype/audio_tests/1_sample_meeting.txt --heard heard.txt

Read one of the scripts in prototype/audio_tests/ aloud (two people taking
turns), record it, and give this script the recording. It runs the audio
through the real chain (Whisper -> pause-based speakers -> LSTM -> minutes)
and the typed script through the same LSTM and minutes, then compares them:

  1. Turns     same number of turns as the script? (merged / split if not)
  2. Speakers  same back-and-forth pattern as the script?
  3. Words     how many of the script's words were heard (case and
               punctuation ignored, apart from ? and !), turn by turn
  4. Labels    same label per turn from the audio as from the typed script
  5. Minutes   the same decisions and action items

PASS = same turns, same speaker pattern, at least 90% of the words, the same
labels and the same minutes. --heard FILE checks a saved "Speaker1: ..."
transcript instead of audio (no Whisper needed), e.g. one pasted from an
earlier run.

Needs Whisper and ffmpeg for audio (see pipeline.py, "Real audio").
"""

import os
import sys

from lstm_model import run_pipeline, split_words
from meeting_minutes import generate_minutes, label_turns_keywords, label_turns_model
from pipeline import load_text

WORDS_NEEDED = 0.90


def _turns(transcript: str) -> list[tuple[str, str]]:
    return [(line.split(":", 1)[0].strip(), line.split(":", 1)[1].strip())
            for line in transcript.strip().splitlines()
            if ":" in line and line.split(":", 1)[1].strip()]


def _as_speaker_numbers(turns):
    """'Alex', 'Priya', 'Alex' -> 'Speaker1', 'Speaker2', 'Speaker1' (order of first appearance)."""
    names = {}
    return [(names.setdefault(s, f"Speaker{len(names) + 1}"), t) for s, t in turns]


def _edit_distance(a: list[str], b: list[str]) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def word_accuracy(said: str, heard: str) -> float:
    """1 - word error rate, floored at 0: 1.00 = every word heard as said."""
    a, b = split_words(said), split_words(heard)
    if not a:
        return 1.0 if not b else 0.0
    return max(0.0, 1 - _edit_distance(a, b) / len(a))


def _labels_and_minutes(turns):
    transcript = "\n".join(f"{s}: {t}" for s, t in turns)
    turns, output, ends = run_pipeline(transcript=transcript)
    labels = label_turns_model(turns, output, ends) or label_turns_keywords(turns)
    return [l["category"] for l in labels], generate_minutes(turns, labels)


def _minutes_items(minutes: str) -> list[str]:
    return [line for line in minutes.splitlines() if line.startswith("- ")]


def check(script_path: str, heard_transcript: str) -> bool:
    script_text, gold = load_text(script_path)
    said = _as_speaker_numbers(_turns(script_text))
    heard = _turns(heard_transcript)
    if not heard:
        print("Heard no speech at all (no 'Speaker: text' lines). FAIL")
        return False

    said_labels, said_minutes = _labels_and_minutes(said)
    heard_labels, heard_minutes = _labels_and_minutes(heard)

    same_count = len(said) == len(heard)
    same_speakers = same_count and [s for s, _ in said] == [s for s, _ in heard]
    overall = word_accuracy(" ".join(t for _, t in said), " ".join(t for _, t in heard))
    same_labels = same_count and said_labels == heard_labels
    same_minutes = _minutes_items(said_minutes) == _minutes_items(heard_minutes)

    print(f"Script: {os.path.basename(script_path)}\n")
    if same_count:
        print(f"  {'#':>2}  {'said':<44} {'heard':<44} {'words':>5}  {'label said / heard'}")
        for n, ((s1, t1), (s2, t2), l1, l2) in enumerate(zip(said, heard, said_labels, heard_labels), 1):
            mark = "" if (s1 == s2 and l1 == l2) else "  <-"
            short = lambda s, t: (f"{s[-1]}: {t}")[:44]
            print(f"  {n:>2}  {short(s1, t1):<44} {short(s2, t2):<44} {word_accuracy(t1, t2):>5.0%}"
                  f"  {l1} / {l2}{mark}")
    else:
        print("  Said:")
        for s, t in said:
            print(f"    {s}: {t}")
        print("  Heard:")
        for s, t in heard:
            print(f"    {s}: {t}")
    print()

    kind = "same" if same_count else ("merged (fewer turns)" if len(heard) < len(said) else "split (more turns)")
    rows = [
        ("Turns", same_count, f"said {len(said)}, heard {len(heard)}: {kind}"),
        ("Speakers", same_speakers, "same back-and-forth" if same_speakers else "pattern differs"),
        ("Words", overall >= WORDS_NEEDED, f"{overall:.0%} of the script's words heard (need {WORDS_NEEDED:.0%})"),
        ("Labels", same_labels, "same label on every turn" if same_labels else
         ("differ on " + ", ".join(str(n) for n, (a, b) in enumerate(zip(said_labels, heard_labels), 1) if a != b)
          if same_count else "can't compare turn by turn")),
        ("Minutes", same_minutes, "same decisions and action items" if same_minutes else "differ (below)"),
    ]
    for name, ok, note in rows:
        print(f"  {name:<9} {'OK  ' if ok else 'FAIL'}  {note}")
    if gold and same_count:
        right = sum(g == h for g, h in zip(gold, heard_labels))
        print(f"  (hand labels in the script: the audio run matches {right}/{len(gold)})")
    if not same_minutes:
        show = lambda m: "    " + "\n    ".join(l for l in m.splitlines() if l.startswith(("## ", "- ")))
        print("\n  Minutes from the typed script:\n" + show(said_minutes))
        print("  Minutes from the audio:\n" + show(heard_minutes))
    passed = all(ok for _, ok, _ in rows)
    print(f"\n{'PASS' if passed else 'FAIL'}: audio {'matches' if passed else 'does not match'} the typed script")
    return passed


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 3 and args[1] == "--heard":
        if not os.path.isfile(args[2]):
            sys.exit(f"No transcript file at {args[2]}")
        with open(args[2], encoding="utf-8") as f:
            heard = f.read()
    elif len(args) == 2:
        if not os.path.isfile(args[1]):
            sys.exit(f"No audio file at {args[1]}")
        from transcriber import get_transcript
        try:
            heard = get_transcript(args[1])
        except ModuleNotFoundError as error:
            if error.name == "whisper":
                sys.exit("Whisper isn't installed. Run: pip install openai-whisper")
            raise
    else:
        sys.exit("Usage: python prototype/audio_check.py SCRIPT.txt AUDIO_FILE\n"
                 "   or: python prototype/audio_check.py SCRIPT.txt --heard TRANSCRIPT.txt")
    if not os.path.isfile(args[0]):
        sys.exit(f"No script file at {args[0]}")
    sys.exit(0 if check(args[0], heard) else 1)
