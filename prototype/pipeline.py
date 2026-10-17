"""
The whole pipeline in one command: audio -> transcript -> LSTM -> minutes.

    python prototype/pipeline.py                # sample meeting, no audio needed
    python prototype/pipeline.py meeting.wav    # real audio (see "Real audio" below)
    python prototype/pipeline.py --compare      # also show keyword-rule vs trained-LSTM label per turn
    python prototype/pipeline.py --compare meeting.wav
    python prototype/pipeline.py --compare --meeting hackathon_1
                                                # a labelled meeting from training_data.py,
                                                # with its hand labels as a third column
    python prototype/pipeline.py --compare --text my_meeting.txt
                                                # your own lines, one "Speaker: text" per
                                                # line; add "act " / "dec " / "dis " in
                                                # front of every line to see hand labels too
                                                # (example: prototype/my_meeting_example.txt)

One stage per owner, in order:
  1. Pasan + Rahini  transcriber.get_transcript(audio_path)
                     -> one "Speaker: text" line per turn
  2. Asy             lstm_model.run_pipeline(transcript=...)
                     -> turns, hourglass output (tokens x 8), turn_end_indices
  3. Nomt            meeting_minutes.label_turns_model(...)  (trained LSTM)
                     or label_turns_keywords(turns)          (no weights yet)
                     -> one label per turn; generate_minutes() -> Markdown

Trained weights come from `python prototype/train_lstm.py`, which writes
prototype/trained_lstm.npz. Without that file the pipeline still runs,
with keyword rules for stage 3.

What this does NOT claim
------------------------
- The trained model learned from a few hundred hand-written turns plus
  generated practice meetings (see README for its scores). It catches
  replies like "Sure." that keyword rules can't, but it still misses
  some decisions, e.g. reported confirmations like "Venue is confirmed".
- Speakers come from transcriber.py's pause rule (a gap of 0.8 s or more
  means the next person). Works on synthetic speech since the 8 Oct fix;
  a human recording is still to test. A reply that follows in under 0.8 s
  can still be merged into the previous speaker's turn, and more than two
  speakers are not told apart.

Real audio
----------
Needs Whisper and ffmpeg on top of requirements.txt:
    pip install openai-whisper
    (ffmpeg: preinstalled on Colab; on Windows, `winget install --id Gyan.FFmpeg -e`
     or `conda install -c conda-forge ffmpeg`)
"""

import os
import sys

from lstm_model import run_pipeline
from meeting_minutes import generate_minutes, label_turns_keywords, label_turns_model
from transcriber import get_transcript


def print_comparison(turns: list[tuple[str, str]], keyword_labels: list[dict],
                     trained_labels: list[dict] | None, gold: list[str] | None = None) -> None:
    """One row per turn: keyword-rule label vs trained-LSTM label, * where they differ.

    With `gold` (hand labels, from --meeting) there is a third column, and
    each labeller's guess is marked x where it disagrees with the hand label.
    """
    if trained_labels is None:
        print("Compare         no trained weights (run train_lstm.py), so nothing to compare\n")
        return
    print("Compare         keyword rules vs trained LSTM (* = they differ"
          + ("; x = disagrees with the hand label)" if gold else ")"))
    print(f"      {'#':>2}  {'speaker':<9} {'text':<60}  {'keywords':<13} {'trained LSTM':<13}"
          + (f" {'hand label':<12}" if gold else ""))
    differ = kw_right = lstm_right = 0
    for n, ((speaker, text), kw, lstm) in enumerate(zip(turns, keyword_labels, trained_labels), 1):
        shown = text if len(text) <= 60 else text[:57] + "..."
        k, m = kw["category"], lstm["category"]
        mark = "*" if k != m else ""
        differ += bool(mark)
        if gold:
            g = gold[n - 1]
            kw_right += k == g
            lstm_right += m == g
            k += "" if k == g else " x"
            m += "" if m == g else " x"
        print(f"      {n:>2}  {speaker:<9} {shown:<60}  {k:<13} {m:<13}"
              + (f" {gold[n - 1]:<12}" if gold else "") + f" {mark}")
    print(f"      {differ} of {len(turns)} turns differ")
    if gold:
        print(f"      matches the hand label: keywords {kw_right}/{len(turns)}, "
              f"trained LSTM {lstm_right}/{len(turns)}")
    print()


def load_meeting(name: str) -> tuple[str, list[str]]:
    """A labelled meeting from training_data.py as (transcript, hand labels)."""
    from training_data import MEETINGS
    for meeting in MEETINGS:
        if meeting["name"] == name:
            transcript = "\n".join(f"{s}: {t}" for s, t, _ in meeting["turns"])
            return transcript, [label for _s, _t, label in meeting["turns"]]
    names = ", ".join(m["name"] for m in MEETINGS)
    raise ValueError(f"No meeting called {name!r}. Choose one of: {names}")


LABEL_PREFIXES = {"act": "action_item", "dec": "decision", "dis": "discussion",
                  "action_item": "action_item", "decision": "decision", "discussion": "discussion"}


def load_text(path: str) -> tuple[str, list[str] | None]:
    """Your own meeting from a text file, as (transcript, hand labels or None).

    One turn per line, "Speaker: text". Optionally start EVERY line with a
    label (act / dec / dis) to get a hand-label column in --compare:
        act  Mei: Can you send the slides tonight?
    Blank lines and lines starting with # are skipped.
    """
    lines, labels = [], []
    with open(path, encoding="utf-8") as f:
        for number, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            first = line.split(maxsplit=1)[0].lower()
            if first in LABEL_PREFIXES and len(line.split(maxsplit=1)) > 1:
                labels.append(LABEL_PREFIXES[first])
                line = line.split(maxsplit=1)[1].strip()
            if ":" not in line or not line.split(":", 1)[1].strip():
                raise ValueError(f"{path}, line {number}: expected 'Speaker: text', got {raw.strip()!r}")
            lines.append(line)
    if not lines:
        raise ValueError(f"{path} has no 'Speaker: text' lines")
    if labels and len(labels) != len(lines):
        raise ValueError(f"{path}: label every line (act / dec / dis) or none of them; "
                         f"{len(labels)} of {len(lines)} lines have one")
    return "\n".join(lines), (labels or None)


def unknown_words(turns: list[tuple[str, str]]) -> list[str]:
    """Words the trained model never saw in training: it reads them as <unk>."""
    from lstm_model import load_trained_model, model_words
    trained = load_trained_model()
    if trained is None:
        return []
    known = set(trained["vocab"])
    return sorted({w for _s, t in turns for w in model_words(t) if w not in known})


def run(audio_path: str | None = None, verbose: bool = True, compare: bool = False,
        meeting: str | None = None, text: str | None = None) -> str:
    """Run all three stages and return the minutes as Markdown."""
    # 1. Transcriber (Pasan + Rahini), a labelled meeting (--meeting) or your own text (--text)
    gold = None
    if meeting:
        transcript, gold = load_meeting(meeting)
    elif text:
        transcript, gold = load_text(text)
    else:
        transcript = get_transcript(audio_path)

    # 2. Hourglass LSTM (Asy)
    turns, output, turn_end_indices = run_pipeline(transcript=transcript)

    # 3. Labels + minutes (Nomt): trained model, or keyword rules if untrained
    labels = label_turns_model(turns, output, turn_end_indices)
    trained_labels = labels
    labeller = "trained hourglass LSTM"
    if labels is None:
        labels = label_turns_keywords(turns)
        labeller = "keyword rules; run train_lstm.py to use the trained model"
    minutes = generate_minutes(turns, labels)

    if verbose:
        source = (f"labelled meeting '{meeting}' in training_data.py" if meeting
                  else f"text file {text}" if text
                  else audio_path if audio_path else "placeholder sample (no audio given)")
        counts = {c: sum(label["category"] == c for label in labels)
                  for c in ("discussion", "decision", "action_item")}
        print(f"1/3 Transcript  from {source}")
        for line in transcript.strip().splitlines():
            print(f"      {line}")
        print(f"\n2/3 LSTM        {len(turns)} turns, {output.shape[0]} tokens "
              f"-> output {output.shape[0]} x {output.shape[1]}")
        print(f"\n3/3 Labels      {counts['discussion']} discussion, "
              f"{counts['decision']} decision, {counts['action_item']} action item "
              f"({labeller})\n")
        if compare:
            print_comparison(turns, label_turns_keywords(turns), trained_labels, gold)
        unknown = unknown_words(turns) if text else []
        if unknown:
            print(f"Words the model never saw in training (read as <unk>): {', '.join(unknown)}\n")
        print(minutes)

    return minutes


if __name__ == "__main__":
    args = sys.argv[1:]
    compare = "--compare" in args
    options = {}
    for option, example in (("--meeting", "--meeting hackathon_1"), ("--text", "--text my_meeting.txt")):
        if option in args:
            at = args.index(option)
            if at + 1 >= len(args) or args[at + 1].startswith("--"):
                sys.exit(f"{option} needs a value, e.g. {example}")
            options[option] = args[at + 1]
            del args[at:at + 2]
    meeting, text = options.get("--meeting"), options.get("--text")
    unknown = [a for a in args if a.startswith("--") and a != "--compare"]
    if unknown:
        sys.exit(f"Unknown option {unknown[0]}. Options: --compare, --meeting NAME, --text FILE")
    paths = [a for a in args if a != "--compare"]
    audio = paths[0] if paths else None
    if sum(bool(x) for x in (audio, meeting, text)) > 1:
        sys.exit("Give only one input: an audio file, --meeting NAME or --text FILE")
    if audio and not os.path.isfile(audio):
        sys.exit(f"No audio file at {audio}")
    if text and not os.path.isfile(text):
        sys.exit(f"No text file at {text}")
    try:
        run(audio, compare=compare, meeting=meeting, text=text)
    except ModuleNotFoundError as error:
        if error.name == "whisper":
            sys.exit("Whisper isn't installed. Run: pip install openai-whisper")
        raise
    except FileNotFoundError as error:
        # Whisper calls the ffmpeg program to read audio.
        sys.exit(f"Couldn't start ffmpeg ({error}). Install it: "
                 "winget install --id Gyan.FFmpeg -e  (or: conda install -c conda-forge ffmpeg)")
    except RuntimeError as error:
        if "Failed to load audio" in str(error):
            sys.exit(f"ffmpeg couldn't read {audio}. Try a .wav or .mp3 file.")
        raise
    except ValueError as error:
        sys.exit(str(error))
