"""
Talk like a meeting. Your own audio goes through every phase, and each phase says PASS or FAIL.

    python testing/try_it.py --name Asy          # records you, then runs the four phases on your audio
    python testing/try_it.py --file talk.wav     # the same, for a recording you already have

When you talk, pretend to be two people: say one turn, STOP FOR 2 SECONDS, say the next turn.
No script. Phase 2 checks itself; for phases 1, 3 and 4 you read the output and answer y or n.

  1  Audio -> text   Whisper writes down what you said                         (you judge)
  2  RNN             your audio -> 13 numbers per 10 ms -> the RNN -> CTC       (checks itself)
  3  LSTM            the trained hourglass labels each turn you spoke           (you judge)
  4  Minutes         decisions and action items from those labels               (you judge)

The result is saved in testing/results/manual/.
"""

import argparse
import datetime
import os
import sys
import wave

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Anaconda on Windows (see solo_test.py)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "prototype"))
from audio_features import mfcc          # audio -> 13 numbers per 10 ms (shared with the notebook)
GAP = 1.5          # seconds of silence that count as the next person (same as the solo test)
RATE = 16000


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def record(path: str) -> None:
    import numpy as np
    import sounddevice as sd
    chunks = []
    ask("Press Enter, then talk. Say one turn, stop for 2 seconds, say the next turn.\n"
        "Press Enter again when you are done... ")
    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16",
                        callback=lambda indata, *_: chunks.append(indata.copy())):
        ask("...recording (Enter to stop) ")
    audio = np.concatenate(chunks) if chunks else np.zeros((0, 1), np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(audio.tobytes())
    print(f"Recorded {len(audio) / RATE:.0f} s -> {path}")


class Results:
    def __init__(self):
        self.rows: list[tuple[int, str, bool]] = []

    def add(self, n: int, name: str, ok: bool, why: str) -> None:
        self.rows.append((n, name, ok))
        print(f"\n  {'PASS' if ok else 'FAIL'}  phase {n}: {why}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Talk like a meeting; every phase runs on your audio.")
    ap.add_argument("--name", default="live", help="your name, for the saved result file")
    ap.add_argument("--file", help="a recording to use instead of the microphone")
    args = ap.parse_args()

    import numpy as np
    out_dir = os.path.join(HERE, "results", "manual")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    result_path = os.path.join(out_dir, f"{args.name}_live_{stamp}.txt")
    from phases import Tee
    screen = sys.stdout
    sys.stdout = Tee(screen, open(result_path, "w", encoding="utf-8"))
    print(f"Live phase-by-phase test -- {args.name}, {datetime.datetime.now():%d %b %Y %H:%M}")

    audio_path = args.file
    if not audio_path:
        audio_path = os.path.join(out_dir, f"{args.name}_live_{stamp}.wav")
        record(audio_path)
    res = Results()

    # ---- Phase 1: audio -> text ------------------------------------------------------------
    print("\n=== Phase 1: Audio -> text (Whisper listens to your audio)")
    print("  (the first time, this downloads the Whisper model)")
    from transcriber import WHISPER_MODEL, get_transcript
    if WHISPER_MODEL == "medium.en":
        print("  (medium.en hears the most but is slow: about two minutes per minute of speech on a laptop CPU.\n"
              "   For a quick try: set WHISPER_MODEL=small.en first.)")
    transcript = get_transcript(audio_path, pause_gap=GAP)
    if not transcript.strip():
        res.add(1, "Audio -> text", False, "nothing was heard; check the microphone isn't muted")
        transcript = ""
    else:
        print(f"  model {WHISPER_MODEL}; every line below is one turn:")
        for line in transcript.strip().splitlines():
            print(f"    {line}")
        ok = ask("\n  Is that what you said, with your pauses as the turns? [y/n] ").lower().startswith("y")
        res.add(1, "Audio -> text", ok, "you said it matches what you spoke" if ok else "you said it does not match what you spoke")

    # ---- Phase 2: the RNN on your audio ----------------------------------------------------
    print("\n=== Phase 2: RNN (your audio -> features -> RNN -> CTC)")
    from whisper.audio import load_audio
    import rnn_ctc_demo as demo
    samples = load_audio(audio_path)
    seconds = len(samples) / RATE
    feats = mfcc(samples)
    feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)
    print(f"  your audio: {seconds:.1f} s -> {feats.shape[0]} frames x {feats.shape[1]} numbers (one frame every 10 ms)")
    model = demo.SimpleRNN(demo.F, demo.H, len(demo.CHARS))
    probs = model.forward(feats)
    best = probs.argmax(axis=1)
    letters = "".join("_" if i == 0 else demo.CHARS[i] for i in best[:60])
    print(f"  the RNN's letter guess for the first 60 frames: {letters}")
    print(f"  after the CTC decoder (merge repeats, drop blanks): '{demo.ctc_greedy_decode(probs)[:60]}'")
    print("  (gibberish is expected: this RNN has random weights. It shows the data flow on YOUR audio;\n"
          "   in the pipeline, Whisper does the real recognition in phase 1.)")
    shapes_ok = (feats.shape[1] == demo.F and abs(feats.shape[0] - seconds * 100) <= 3
                 and probs.shape == (feats.shape[0], len(demo.CHARS)) and np.allclose(probs.sum(axis=1), 1)
                 and bool(np.isfinite(feats).all()))
    res.add(2, "RNN", shapes_ok,
            f"{feats.shape[0]} frames in, {probs.shape} probabilities out, every row sums to 1" if shapes_ok
            else "the frame count, shapes or probabilities are wrong")

    # ---- Phase 3: the LSTM -----------------------------------------------------------------
    print("\n=== Phase 3: LSTM (the trained hourglass labels each turn)")
    if not transcript:
        res.add(3, "LSTM", False, "no transcript to label (phase 1 heard nothing)")
        res.add(4, "Minutes", False, "no transcript to write minutes from")
    else:
        from lstm_model import run_pipeline
        from meeting_minutes import generate_minutes, label_turns_keywords, label_turns_model
        turns, output, ends = run_pipeline(transcript=transcript)
        labels = label_turns_model(turns, output, ends)
        keywords = [l["category"] for l in label_turns_keywords(turns)]
        print(f"  {'turn':<58} {'LSTM':<12} keyword rules")
        for (speaker, text), l, k in zip(turns, labels, keywords):
            print(f"  {(speaker + ': ' + text)[:56]:<58} {l['category']:<12} {k}")
        print("  (action_item = someone must do something; decision = something was settled; discussion = the rest)")
        ok = ask("\n  Is each turn labelled the way you meant it? [y/n] ").lower().startswith("y")
        res.add(3, "LSTM", ok, "you said every label is right" if ok else "you said at least one label is wrong")

        # ---- Phase 4: the minutes ----------------------------------------------------------
        print("\n=== Phase 4: Meeting minutes")
        for line in generate_minutes(turns, labels).strip().splitlines():
            print(f"    {line}")
        ok = ask("\n  Do these minutes capture the decisions and action items you spoke? [y/n] ").lower().startswith("y")
        res.add(4, "Minutes", ok, "you said the minutes are right" if ok else "you said the minutes are wrong or incomplete")

    print("\n=== Summary")
    names = {1: "Audio -> text", 2: "RNN", 3: "LSTM", 4: "Meeting minutes"}
    for n, name, ok in res.rows:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}. {names[n]}")
    print(f"\nSaved to {os.path.relpath(result_path, os.path.dirname(HERE))}")
    print("Push it, or send it to Claude, so it goes into testing/testing.md.")
    sys.stdout.streams[1].close()
    sys.stdout = screen
    return 0 if all(ok for _, _, ok in res.rows) else 1


if __name__ == "__main__":
    sys.exit(main())
