"""
The whole pipeline in one command: audio -> transcript -> LSTM -> minutes.

    python prototype/pipeline.py                # sample meeting, no audio needed
    python prototype/pipeline.py meeting.wav    # real audio (see "Real audio" below)
    python prototype/pipeline.py --compare      # also show keyword-rule vs trained-LSTM label per turn
    python prototype/pipeline.py --compare meeting.wav

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
- The trained model learned from 211 hand-written turns plus generated
  practice meetings (see train_lstm.py for its scores). It catches
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
                     trained_labels: list[dict] | None) -> None:
    """One row per turn: keyword-rule label vs trained-LSTM label, * where they differ."""
    if trained_labels is None:
        print("Compare         no trained weights (run train_lstm.py), so nothing to compare\n")
        return
    print("Compare         keyword rules vs trained LSTM (* = they differ)")
    print(f"      {'#':>2}  {'speaker':<9} {'text':<60}  {'keywords':<12} {'trained LSTM':<12}")
    differ = 0
    for n, ((speaker, text), kw, lstm) in enumerate(zip(turns, keyword_labels, trained_labels), 1):
        shown = text if len(text) <= 60 else text[:57] + "..."
        mark = "*" if kw["category"] != lstm["category"] else ""
        differ += bool(mark)
        print(f"      {n:>2}  {speaker:<9} {shown:<60}  {kw['category']:<12} {lstm['category']:<12} {mark}")
    print(f"      {differ} of {len(turns)} turns differ\n")


def run(audio_path: str | None = None, verbose: bool = True, compare: bool = False) -> str:
    """Run all three stages and return the minutes as Markdown."""
    # 1. Transcriber (Pasan + Rahini)
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
        source = audio_path if audio_path else "placeholder sample (no audio given)"
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
            print_comparison(turns, label_turns_keywords(turns), trained_labels)
        print(minutes)

    return minutes


if __name__ == "__main__":
    args = sys.argv[1:]
    compare = "--compare" in args
    paths = [a for a in args if a != "--compare"]
    audio = paths[0] if paths else None
    if audio and not os.path.isfile(audio):
        sys.exit(f"No audio file at {audio}")
    try:
        run(audio, compare=compare)
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
