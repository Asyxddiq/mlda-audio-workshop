"""
Individual (solo) audio test: one person, one command, a saved result.

    python testing/solo_test.py --name Asy
        records you from the microphone while it shows one line at a time,
        then checks the recording and saves the result
    python testing/solo_test.py --name Asy --file my_recording.m4a
        you recorded on your phone instead (read the lines with a 2-second
        stop between them): checks that file and saves the result
    python testing/solo_test.py --name Asy --script solo_2
        the second script (default: solo_1, the sample meeting)

What it checks (the same as prototype/audio_check.py): the recording goes
through the real chain (Whisper -> pause-based speakers -> LSTM -> minutes)
and is compared with the typed script. Two verdicts:
  transcriber  same turns, same back-and-forth, at least 90% of the words
  end to end   also the same labels and the same minutes

The result is saved as testing/results/manual/<name>_<script>_<date-time>.txt
(and the microphone recording as a .wav next to it), so it can be pushed or
sent on, and listed in testing/testing.md.

Needs: pip install openai-whisper (and ffmpeg). For the microphone mode also
pip install sounddevice; without it, record on your phone and use --file.
"""

import argparse
import contextlib
import datetime
import importlib.metadata
import importlib.util
import io
import os
import platform
import shutil
import sys
import time
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOTYPE = os.path.join(HERE, "..", "prototype")
sys.path.insert(0, PROTOTYPE)

SCRIPTS = os.path.join(HERE, "scripts")
RESULTS = os.path.join(HERE, "results", "manual")
SAMPLE_RATE = 16000
GAP_SECONDS = 2.0          # silence the script leaves between lines (the pause rule needs > 0.8 s)


def script_path(name: str) -> str:
    if os.path.isfile(name):
        return name
    matches = sorted(f for f in os.listdir(SCRIPTS) if f.startswith(name) and f.endswith(".txt"))
    if not matches:
        sys.exit(f"No script called {name!r} in {SCRIPTS}. Try solo_1 or solo_2.")
    return os.path.join(SCRIPTS, matches[0])


def check_tools(need_mic: bool) -> None:
    missing = []
    if importlib.util.find_spec("whisper") is None:
        missing.append("Whisper:     pip install openai-whisper")
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg:      Windows: winget install --id Gyan.FFmpeg -e   "
                       "(Mac: brew install ffmpeg; Colab: already there)")
    if need_mic and importlib.util.find_spec("sounddevice") is None:
        missing.append("microphone:  pip install sounddevice   "
                       "(or record on your phone and run again with --file my_recording.m4a)")
    if missing:
        sys.exit("Install these first, then run again:\n  " + "\n  ".join(missing))


def record_guided(lines: list[str], out_path: str) -> None:
    """Show one line at a time while recording; Enter = done with this line."""
    import numpy as np
    import sounddevice as sd

    chunks = []

    def keep(indata, _frames, _time, status):
        chunks.append(indata.copy())

    print("\nRead each line out loud when it appears, then press Enter.")
    print("After Enter, stay quiet: the next line appears after a 2-second pause.")
    print("Skip the names; just read the words. Don't stop in the middle of a line.\n")
    input("Press Enter to start recording... ")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=keep):
        time.sleep(1.0)
        for n, line in enumerate(lines, 1):
            words = line.split(":", 1)[1].strip()
            input(f"\n  [{n}/{len(lines)}]  {words}\n        (Enter when done) ")
            print("        ...quiet...", end="", flush=True)
            time.sleep(GAP_SECONDS)
            print()
    audio = np.concatenate(chunks) if chunks else np.zeros((0, 1), np.int16)
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.tobytes())
    seconds = len(audio) / SAMPLE_RATE
    peak = int(abs(audio).max()) if len(audio) else 0
    print(f"\nRecorded {seconds:.0f} s -> {out_path}")
    if peak < 1000:
        print("Warning: the recording is very quiet. Check the microphone isn't muted "
              "or set to the wrong input.")


def whisper_description(model: str) -> str:
    import whisper
    try:
        return f"openai-whisper {importlib.metadata.version('openai-whisper')}, model '{model}'"
    except importlib.metadata.PackageNotFoundError:
        first_line = (whisper.__doc__ or "unknown").strip().splitlines()[0]
        return f"model '{model}', NOT the openai-whisper package: {first_line}"


class _Tee(io.StringIO):
    """Print as usual and keep a copy for the result file."""

    def write(self, text):
        sys.__stdout__.write(text)
        return super().write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Solo audio test: record, check, save the result.")
    parser.add_argument("--name", required=True, help="your name, for the result file (e.g. Asy)")
    parser.add_argument("--script", default="solo_1", help="solo_1 (default), solo_2, or a path")
    parser.add_argument("--file", help="a recording you already made (m4a / wav / mp3)")
    parser.add_argument("--out", default=RESULTS, help="folder for the result (default testing/results/manual)")
    args = parser.parse_args()

    path = script_path(args.script)
    check_tools(need_mic=args.file is None)

    from audio_check import check
    from pipeline import load_text
    from transcriber import WHISPER_MODEL, get_transcript

    text, _gold = load_text(path)
    lines = text.splitlines()
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    base = f"{args.name}_{os.path.splitext(os.path.basename(path))[0]}_{stamp}"
    os.makedirs(args.out, exist_ok=True)

    if args.file:
        if not os.path.isfile(args.file):
            sys.exit(f"No recording at {args.file}")
        audio_path = args.file
    else:
        audio_path = os.path.join(args.out, base + ".wav")
        record_guided(lines, audio_path)

    print(f"\nTranscribing with Whisper '{WHISPER_MODEL}' (the first run downloads the model)...\n")
    heard = get_transcript(audio_path)

    report = _Tee()
    with contextlib.redirect_stdout(report):
        print(f"Solo audio test -- {args.name}, {datetime.datetime.now():%d %b %Y %H:%M}")
        print(f"Script:    {os.path.relpath(path, os.path.join(HERE, '..'))}")
        print(f"Recording: {os.path.basename(audio_path)} ({'microphone, guided' if not args.file else 'own file'})")
        print(f"Whisper:   {whisper_description(WHISPER_MODEL)}; Python {platform.python_version()} on {platform.system()}\n")
        print("Heard (what Whisper and the pause rule produced):")
        print("  " + heard.strip().replace("\n", "\n  ") + "\n")
        passed = check(path, heard)
    result_path = os.path.join(args.out, base + ".txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(report.getvalue())
    print(f"\nSaved the result to {os.path.relpath(result_path, os.path.join(HERE, '..'))}")
    print("Push it, or send it to Claude, so it goes into testing/testing.md.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
