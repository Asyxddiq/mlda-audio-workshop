"""
Try the model: talk like a meeting, see what it makes of it.

    python testing/try_it.py                 # records you, then shows the transcript, the label of every turn, and the minutes
    python testing/try_it.py --file talk.wav # the same for a recording you already have
    python testing/try_it.py --text          # no microphone: type the turns instead

When you talk, pretend to be two people: say one turn, STOP FOR 2 SECONDS, say the next turn.
There is no script and nothing is scored: you read the result and decide if it is right.
"""

import argparse
import os
import sys
import wave

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Anaconda on Windows (see solo_test.py)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "prototype"))
GAP = 1.5          # seconds of silence that count as the next person (same as the solo test)
RATE = 16000


def record(path: str) -> None:
    import numpy as np
    import sounddevice as sd
    chunks = []
    input("Press Enter, then talk. Say one turn, stop for 2 seconds, say the next turn.\n"
          "Press Enter again when you are done... ")
    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16",
                        callback=lambda indata, *_: chunks.append(indata.copy())):
        input("...recording (Enter to stop) ")
    audio = np.concatenate(chunks) if chunks else np.zeros((0, 1), np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(audio.tobytes())
    print(f"Recorded {len(audio) / RATE:.0f} s -> {path}")


def typed() -> str:
    print("Type one turn per line as 'Name: text'. Empty line to finish.")
    lines = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        lines.append(line if ":" in line else f"{'Ann' if len(lines) % 2 == 0 else 'Ben'}: {line}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Talk (or type) like a meeting and see what the model makes of it.")
    ap.add_argument("--file", help="a recording to use instead of the microphone")
    ap.add_argument("--text", action="store_true", help="type the turns instead of speaking")
    args = ap.parse_args()

    from lstm_model import run_pipeline
    from meeting_minutes import generate_minutes, label_turns_keywords, label_turns_model

    if args.text:
        transcript = typed()
    else:
        path = args.file
        if not path:
            import datetime
            path = os.path.join(HERE, "results", "manual", f"try_it_{datetime.datetime.now():%Y-%m-%d_%H%M%S}.wav")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            record(path)
        print("\nWhisper is listening to it (the first time it downloads the model)...")
        from transcriber import get_transcript
        transcript = get_transcript(path, pause_gap=GAP)
    if not transcript.strip():
        print("\nNothing heard. Check the microphone isn't muted, and try again.")
        return 1

    print("\n1. WHAT WHISPER HEARD (each line is one turn):")
    for line in transcript.strip().splitlines():
        print(f"   {line}")

    turns, output, ends = run_pipeline(transcript=transcript)
    model = [l["category"] for l in label_turns_model(turns, output, ends)]
    keywords = [l["category"] for l in label_turns_keywords(turns)]
    print(f"\n2. WHAT THE LSTM MAKES OF EACH TURN (action_item / decision / discussion):")
    print(f"   {'turn':<60} {'LSTM':<12} keyword rules")
    for (speaker, text), m, k in zip(turns, model, keywords):
        print(f"   {(speaker + ': ' + text)[:58]:<60} {m:<12} {k}")

    print("\n3. THE MEETING MINUTES:")
    for line in generate_minutes(turns, label_turns_model(turns, output, ends)).strip().splitlines():
        print(f"   {line}")
    print("\nDoes it look right? If not, note which turn it got wrong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
