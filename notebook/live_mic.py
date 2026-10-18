"""Shared live-mic helpers for the participant notebook (WP7/WP8).

Three calls, reused by every live cell -- no block copy-pastes recorder code:

    live_mic.record(prompt_id, seconds=5)        -> WAV path (browser mic, Colab only)
    live_mic.hear(path, model)                   -> transcribed text (reuses the Whisper
                                                     model already loaded in section 1 --
                                                     this module never loads its own)
    live_mic.say(prompt_id, model, seconds=5)    -> (text, path, used_fallback)

No-mic fallback (L2): if this isn't Colab, there's no mic, or permission is denied,
`record()` raises a plain `RuntimeError` -- never an uncaught JS or ffmpeg traceback --
and `say()` catches it and falls back to the matching pre-recorded clip in
`audio/live/<prompt_id>.wav`, so every live cell still produces its aha with no mic at
all. Pass `typed=` to `say()` to skip audio entirely (useful for testing, or a
participant who can't speak in the room).

Recording (L6): a fixed ~5s window with a visible countdown printed to the cell's
output, not press-Enter-to-stop -- smoother for 30 people in one room. Reuses
`demo/hook_demo.ipynb`'s eval_js + MediaRecorder pattern (mic released via
`stream.getTracks().forEach(t => t.stop())`, ffmpeg's stderr shown on failure).
"""
import os
import subprocess
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE_AUDIO_DIR = os.path.join(HERE, "audio", "live")

RECORD_JS = """
const b2text = blob => new Promise(resolve => {
  const reader = new FileReader();
  reader.onloadend = () => resolve(reader.result);
  reader.readAsDataURL(blob);
});
var recorder, chunks = [], micStream;
async function startRecording() {
  micStream = await navigator.mediaDevices.getUserMedia({audio: true});
  recorder = new MediaRecorder(micStream);
  chunks = [];
  recorder.ondataavailable = e => chunks.push(e.data);
  recorder.start();
}
async function stopRecording() {
  recorder.stop();
  await new Promise(resolve => recorder.onstop = resolve);
  micStream.getTracks().forEach(t => t.stop());  // release the mic / turn off the browser's recording indicator
  const blob = new Blob(chunks);
  return await b2text(blob);
}
"""


def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def record(prompt_id: str, seconds: float = 5.0, out_dir: str = ".") -> str:
    """Record `seconds` of browser mic audio to `<prompt_id>.wav`, with a visible
    countdown printed here (not a press-Enter prompt -- smoother for a full room).

    Raises `RuntimeError` with a plain sentence -- never an uncaught traceback --
    if this isn't Colab, there's no mic, or permission was denied; callers (`say`,
    below) catch it and fall back to the shipped clip.
    """
    if not _is_colab():
        raise RuntimeError("No live mic here (not running in Colab) -- using the fallback clip instead.")
    from base64 import b64decode

    from IPython.display import Javascript, display
    from google.colab.output import eval_js

    display(Javascript(RECORD_JS))
    try:
        eval_js("startRecording()")
    except Exception as e:
        raise RuntimeError(f"Couldn't start the mic (permission denied, or no mic here) -- using the fallback clip instead. ({e})")
    print(f"Recording for {int(seconds)}s -- speak now.")
    for left in range(int(seconds), 0, -1):
        print(f"  {left}...", end="\r", flush=True)
        time.sleep(1)
    print("  done.          ")
    data_url = eval_js("stopRecording()")
    _, encoded = data_url.split(",", 1)
    webm_path = os.path.join(out_dir, f"{prompt_id}.webm")
    wav_path = os.path.join(out_dir, f"{prompt_id}.wav")
    with open(webm_path, "wb") as f:
        f.write(b64decode(encoded))
    # ffmpeg converts the browser's webm/opus to WAV -- Whisper reads WAV directly.
    result = subprocess.run(["ffmpeg", "-y", "-i", webm_path, wav_path], capture_output=True, text=True)
    if result.returncode != 0:
        last_line = result.stderr.strip().splitlines()[-1] if result.stderr else ""
        raise RuntimeError(f"Couldn't convert the recording -- using the fallback clip instead. (ffmpeg: {last_line})")
    return wav_path


def hear(path: str, model) -> str:
    """Transcribe `path` with the Whisper `model` already loaded in section 1
    (this module never loads a second one)."""
    return model.transcribe(path, language="en")["text"].strip()


def say(prompt_id: str, model, seconds: float = 5.0, typed: str = None):
    """Record (or use `typed`, or fall back to the shipped clip), then transcribe.

    Returns `(text, wav_path, used_fallback)`.
    """
    if typed is not None:
        print(f"(typed override, no audio) heard: {typed}")
        return typed, None, False
    try:
        path = record(prompt_id, seconds=seconds)
        return hear(path, model), path, False
    except RuntimeError as e:
        print(e)
        fallback = os.path.join(LIVE_AUDIO_DIR, f"{prompt_id}.wav")
        if not os.path.isfile(fallback):
            raise RuntimeError(f"No fallback clip for {prompt_id!r} either (expected {fallback}) -- can't continue this cell.")
        print(f"Using the fallback clip instead: audio/live/{prompt_id}.wav")
        return hear(fallback, model), fallback, True


def word_vector(word: str, dim: int = 8) -> np.ndarray:
    """A fixed, hash-seeded random vector for `word` -- the same word always gets
    the same vector, every run, standing in for a LEARNED embedding. This is
    NOT trained on anything and has no relationship to the trained model's own
    embeddings (see `lstm_model.load_trained_model` for those)."""
    seed = zlib.crc32(word.encode("utf-8")) & 0xFFFFFFFF
    return np.random.default_rng(seed).normal(size=dim)
