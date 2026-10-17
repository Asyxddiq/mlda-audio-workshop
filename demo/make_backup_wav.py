"""
Build the backup WAV for the demo (D1): demo/script.txt, in two synthetic
voices, for when nobody can act the meeting out live.

    python demo/make_backup_wav.py

How it is made
---------------
Windows comes with two built-in SAPI voices on this machine (check with
`Get-InstalledVoices` below): Microsoft Zira and Microsoft Hazel, both
female -- there is no built-in male voice on this laptop, so the two
speakers are told apart by voice timbre and pacing, not by gender. Mei
always speaks as Zira, Ravi always speaks as Hazel.

Each turn in demo/script.txt is synthesised to its own WAV with PowerShell's
System.Speech.Synthesis.SpeechSynthesizer (one subprocess call per turn),
then the turns are concatenated here with 2.0 s of silence between them --
comfortably over the pause_gap=1.5 the demo uses for real two-person speech
(prototype/demo.py; see RULES.md / the 17 Oct PM direction -- 1.5s is what
was actually tested on real speech, not transcriber.py's 0.8s default,
which has only been tested on solo readings), so the pause-based speaker
splitter in transcriber.py sees the same alternating pattern as the two
voices.

Needs: Windows, PowerShell (no extra install -- System.Speech ships with
.NET). Not run as part of run_tests.py (no audio I/O there); run it by hand
if demo/backup_meeting.wav needs regenerating (e.g. after editing
demo/script.txt).
"""
import os
import subprocess
import sys
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "script.txt")
OUT = os.path.join(HERE, "backup_meeting.wav")
SILENCE_S = 2.0   # comfortably over the 1.5s pause_gap the demo uses for real speech (PM direction, 17 Oct)
RATE = 16000

# Any two distinct installed SAPI voices work; these are the two present on
# the machine this was built on (see the docstring above).
VOICES = {"Mei": "Microsoft Zira Desktop", "Ravi": "Microsoft Hazel Desktop"}

_PS_ONE_LINE = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$s.SelectVoice('{voice}'); "
    "$s.SetOutputToWaveFile('{path}'); "
    "$s.Speak([IO.File]::ReadAllText('{text_path}')); "
    "$s.Dispose()"
)


def _read_turns(path: str) -> list[tuple[str, str]]:
    turns = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # "dec  Mei: text" or "Mei: text" -- drop the act/dec/dis label if present.
            parts = line.split(maxsplit=1)
            if parts[0].lower() in ("act", "dec", "dis") and len(parts) > 1:
                line = parts[1]
            speaker, text = line.split(":", 1)
            turns.append((speaker.strip(), text.strip()))
    return turns


def _speak_to_wav(voice: str, text: str, out_path: str, tmp_dir: str) -> None:
    text_path = os.path.join(tmp_dir, "line.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)
    ps = _PS_ONE_LINE.format(voice=voice, path=out_path.replace("\\", "\\\\"),
                             text_path=text_path.replace("\\", "\\\\"))
    result = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"PowerShell TTS failed for voice {voice!r}: {result.stderr}")


def build(script_path: str = SCRIPT, out_path: str = OUT) -> str:
    import tempfile
    turns = _read_turns(script_path)
    missing = {s for s, _ in turns if s not in VOICES}
    if missing:
        raise ValueError(f"No voice assigned for speaker(s) {missing}; edit VOICES in this file.")

    frames = []
    params = None
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, (speaker, text) in enumerate(turns):
            clip_path = os.path.join(tmp_dir, f"turn{i:02d}.wav")
            _speak_to_wav(VOICES[speaker], text, clip_path, tmp_dir)
            with wave.open(clip_path, "rb") as w:
                if params is None:
                    params = w.getparams()
                frames.append(w.readframes(w.getnframes()))
            if i < len(turns) - 1:
                silence_frames = int(SILENCE_S * params.framerate)
                frames.append(b"\x00" * silence_frames * params.sampwidth * params.nchannels)

    with wave.open(out_path, "wb") as out:
        out.setparams(params)
        for chunk in frames:
            out.writeframes(chunk)
    return out_path


if __name__ == "__main__":
    if sys.platform != "win32":
        sys.exit("This uses Windows SAPI (System.Speech); only runs on Windows.")
    path = build()
    with wave.open(path, "rb") as w:
        duration = w.getnframes() / w.getframerate()
    print(f"Wrote {path} ({duration:.1f}s)")
