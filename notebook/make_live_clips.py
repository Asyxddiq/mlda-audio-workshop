"""
Build the no-mic fallback clips for the live-mic cells (WP7/WP8 L2):

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

WP8 adds multi-turn prompts (two people, or one person playing both) --
each turn is synthesised separately, then concatenated with SILENCE_GAP of
silence between turns, same technique `demo/make_backup_wav.py` uses for the
hook's backup WAV, comfortably over the `pause_gap` the live cells use to
split speakers.
"""
import os
import subprocess
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "audio", "live")
RATE = 16000
SILENCE_GAP = 2.0  # seconds between turns in a multi-turn prompt

VOICES = {"a": "Microsoft Zira Desktop", "b": "Microsoft Hazel Desktop"}

# prompt_id -> (voice, text) for a single-turn prompt, OR
# prompt_id -> [(voice, text), (voice, text), ...] for a multi-turn prompt
# (turns joined with SILENCE_GAP of silence). Word counts (via
# lstm_model.model_words) checked to match the live-cell pairs exactly
# except the first word, where that matters (block4, block5).
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
    # WP8
    "block6_gates": [("a", "Can you check the numbers before Friday?"), ("b", "Sure.")],
    "block7_request": ("a", "Can you send me the report by tomorrow?"),
    "block7_proposal": ("a", "Let's move the deadline to next week."),
    "block7_fact": ("a", "The budget was approved yesterday."),
    "block8_exchange": [("a", "Can you confirm the venue?"), ("b", "Sure.")],
    "sure_request": [("a", "Can you book the room?"), ("b", "Sure.")],
    "sure_proposal": [("a", "Shall we move it to Friday?"), ("b", "Sure.")],
    "block9_meeting": [
        ("a", "Let's finalise the catering order this week."),
        ("b", "Sounds good, I will call the vendor tomorrow."),
        ("a", "Can you also confirm the guest count by Friday?"),
        ("b", "Sure, I will check with the committee."),
    ],
    # WP9 F2(a): short vs long conversation, same first-line key fact
    "headline_short_conv": [
        ("a", "The meeting is on Friday at three."),
        ("b", "Got it, thanks."),
        ("a", "See you then."),
    ],
    "headline_long_conv": [
        ("a", "The meeting is on Friday at three."),
        ("b", "Sounds good."),
        ("a", "We also need to finalise the catering order this week."),
        ("b", "I will call the vendor tomorrow."),
        ("a", "Can you confirm the guest count by Friday?"),
        ("b", "Sure, I will check with the committee."),
        ("a", "Let's also review the budget numbers before we close out."),
        ("b", "Agreed, I will send those over tonight."),
    ],
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


def _concat_with_silence(clip_paths: list, out_path: str, gap_s: float) -> None:
    frames = []
    params = None
    for i, clip_path in enumerate(clip_paths):
        with wave.open(clip_path, "rb") as w:
            if params is None:
                params = w.getparams()
            frames.append(w.readframes(w.getnframes()))
        if i < len(clip_paths) - 1:
            silence_frames = int(gap_s * params.framerate)
            frames.append(b"\x00" * silence_frames * params.sampwidth * params.nchannels)
    with wave.open(out_path, "wb") as out:
        out.setparams(params)
        for chunk in frames:
            out.writeframes(chunk)


def build() -> None:
    import tempfile
    os.makedirs(OUT_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        for prompt_id, spec in PROMPTS.items():
            out_path = os.path.join(OUT_DIR, f"{prompt_id}.wav")
            if isinstance(spec, tuple):
                voice_key, text = spec
                _speak_to_wav(VOICES[voice_key], text, out_path, tmp_dir)
                if prompt_id == "block2_noisy":
                    _add_noise(out_path, amount=0.1)
                voices_used = VOICES[voice_key]
            else:
                turn_paths = []
                for i, (voice_key, text) in enumerate(spec):
                    turn_path = os.path.join(tmp_dir, f"{prompt_id}_turn{i}.wav")
                    _speak_to_wav(VOICES[voice_key], text, turn_path, tmp_dir)
                    turn_paths.append(turn_path)
                _concat_with_silence(turn_paths, out_path, SILENCE_GAP)
                voices_used = " / ".join(VOICES[v] for v, _ in spec)
            with wave.open(out_path, "rb") as w:
                duration = w.getnframes() / w.getframerate()
            print(f"wrote {out_path} ({duration:.1f}s, {voices_used})")


if __name__ == "__main__":
    if sys.platform != "win32":
        sys.exit("This uses Windows SAPI (System.Speech); only runs on Windows.")
    build()
