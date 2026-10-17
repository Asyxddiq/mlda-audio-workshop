"""
Make the four Kahoot clips for "where transcription struggles" (N2,
Agenda item 2: "3 noisy scenarios, 1 ideal -> which gives the most
accurate transcript?"), all the same sentence (`notebook/audio/sample_short.wav`,
the same 3-sentence reference text used in the notebook's section 2):

    python notebook/make_noisy_clips.py

Writes four small WAV files to `notebook/audio/`:
  - `noisy_1_ideal.wav`      -- the clean clip, unmodified (the control)
  - `noisy_2_hiss.wav`       -- background hiss (same as the notebook's own
                                 NOISE=0.1 cell)
  - `noisy_3_poor_mic.wav`   -- a poor microphone: every 4th sample kept,
                                 the rest dropped (same as the notebook's
                                 own "poor" cell)
  - `noisy_4_second_voice.wav` -- a second voice talking underneath, quietly
                                 (a snippet of `sample_meeting.wav` mixed in
                                 at low volume) -- the "more realistic" noise
                                 the orchestrator asked for

The notebook's own N2 cell plays all four, measures `word_accuracy` for
each with Whisper, and shows the result as an answer key: these numbers
are measured live when the notebook runs, not pre-computed and read out --
and they move with the Whisper model size (see the notebook's own note on
this).

Re-run this after changing anything above; rebuild `workshop_files.zip`
with `make_bundle.py` afterwards.
"""
import os
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(HERE, "audio")
RATE = 16000


def _read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype("float32") / 32768.0


def _write_wav(path: str, samples: np.ndarray) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


def build() -> list[str]:
    clean = _read_wav(os.path.join(AUDIO, "sample_short.wav"))

    ideal = clean

    hiss = (clean + 0.1 * np.random.default_rng(0).normal(size=clean.shape)).astype("float32")

    poor_mic = np.repeat(clean[::4], 4).astype("float32")[: len(clean)]
    if len(poor_mic) < len(clean):     # pad back to the same length if truncated by the repeat
        poor_mic = np.pad(poor_mic, (0, len(clean) - len(poor_mic)))

    # A second voice talking underneath: a same-length snippet of the
    # (different) sample meeting, mixed in -- a more realistic "noisy room"
    # than synthetic hiss, since it's real speech competing with real
    # speech, not an easily-filtered hum.
    #
    # Mix factor 0.45: chosen by sweeping 0.25-1.0 and measuring word_accuracy
    # on base.en, small.en and medium.en (the three sizes the notebook can
    # pick -- see plan/handoffs/HANDOFF-WP2.md Round 2). Below ~0.4 the clip ties the ideal
    # one at 100% on small.en (the model Colab picks on a GPU); above ~0.5 the
    # ranking itself changes (hiss overtakes it). 0.45 is the only factor
    # tried where "ideal" is the unique top AND the order (ideal > second
    # voice > hiss > poor mic) holds on all three sizes.
    meeting = _read_wav(os.path.join(AUDIO, "sample_meeting.wav"))
    snippet = meeting[: len(clean)] if len(meeting) >= len(clean) else np.pad(meeting, (0, len(clean) - len(meeting)))
    second_voice = (clean + 0.45 * snippet).astype("float32")

    out = [
        ("noisy_1_ideal.wav", ideal),
        ("noisy_2_hiss.wav", hiss),
        ("noisy_3_poor_mic.wav", poor_mic),
        ("noisy_4_second_voice.wav", second_voice),
    ]
    paths = []
    for name, samples in out:
        path = os.path.join(AUDIO, name)
        _write_wav(path, samples)
        paths.append(path)
    return paths


if __name__ == "__main__":
    paths = build()
    for path in paths:
        print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")
