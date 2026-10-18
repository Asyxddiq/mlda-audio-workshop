"""
Build the no-mic fallback clips for the live-mic cells (WP7 L2):

    python notebook/make_live_clips.py

Writes one WAV per prompt_id into `notebook/audio/live/`, bundled into
`workshop_files.zip` by `make_bundle.py` so every live cell still produces
its aha with no mic at all.

Same technique as `demo/make_backup_wav.py`: Windows SAPI via PowerShell's
System.Speech.Synthesis.SpeechSynthesizer, two voices already installed on
this machine (Microsoft Zira, Microsoft Hazel) so the shipped fallback set
itself uses more than one voice, same as the verification runs do.

`block2_noisy` is synthesised clean, then has the same synthetic hiss cells
15-16 of the notebook use mixed in afterwards here, so the fallback still
shows the accuracy drop the live "say it with noise" prompt is for -- a
no-mic room can't produce a real noisy recording, so this fakes the one
part (noise) that needs a live mic, honestly, as a pre-built clip.
"""
import os
import subprocess
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "audio", "live")
RATE = 16000

VOICES = {"a": "Microsoft Zira Desktop", "b": "Microsoft Hazel Desktop"}

# prompt_id -> (voice, text). Word counts (via lstm_model.model_words) checked
# to match the live-cell pairs exactly except the first word, where that
# matters (block4, block5).
PROMPTS = {
    "block2_clear": ("a", "The budget review finished early and everyone left happy."),
    "block2_noisy": ("a", "The budget review finished early and everyone left happy."),
    "block2_backward": ("b", "Friday."),
    "block3_sentence": ("a", "The team decided to postpone the launch until March."),
    "block4_cats": ("b", "Cats are sitting near the window this evening."),
    "block4_dogs": ("b", "Dogs are sitting near the window this evening."),
    "block5_short_a": ("a", "Red cars drive fast."),
    "block5_short_b": ("a", "Blue cars drive fast."),
    "block5_long_a": ("b", "Monday mornings always feel quieter than every other weekday "
                           "afternoon session we have had this term."),
    "block5_long_b": ("b", "Tuesday mornings always feel quieter than every other weekday "
                           "afternoon session we have had this term."),
}

_PS_ONE_LINE = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$s.SelectVoice('{voice}'); "
    "$s.SetOutputToWaveFile('{path}'); "
    "$s.Speak([IO.File]::ReadAllText('{text_path}')); "
    "$s.Dispose()"
)


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


def _add_noise(path: str, amount: float = 0.1, seed: int = 0) -> None:
    with wave.open(path, "rb") as w:
        params = w.getparams()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    noisy = samples + amount * np.random.default_rng(seed).normal(size=samples.shape)
    noisy = np.clip(noisy, -1.0, 1.0)
    pcm = (noisy * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setparams(params)
        w.writeframes(pcm.tobytes())


def build() -> None:
    import tempfile
    os.makedirs(OUT_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        for prompt_id, (voice_key, text) in PROMPTS.items():
            out_path = os.path.join(OUT_DIR, f"{prompt_id}.wav")
            _speak_to_wav(VOICES[voice_key], text, out_path, tmp_dir)
            if prompt_id == "block2_noisy":
                _add_noise(out_path, amount=0.1)
            with wave.open(out_path, "rb") as w:
                duration = w.getnframes() / w.getframerate()
            print(f"wrote {out_path} ({duration:.1f}s, {VOICES[voice_key]})")


if __name__ == "__main__":
    if sys.platform != "win32":
        sys.exit("This uses Windows SAPI (System.Speech); only runs on Windows.")
    build()
