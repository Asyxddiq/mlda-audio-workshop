"""
Real audio -> transcript (Track 1: Pasan + Rahini).

Same contract as transcript_placeholder.py:
    get_transcript(audio_path: str | None = None) -> str
Returns plain text, one line per speaker turn, formatted "Speaker1: text".

- audio_path=None  -> falls back to the placeholder, so nothing downstream breaks.
- audio_path=file  -> Whisper transcribes, then a crude pause-based step
                      assigns speakers and merges same-speaker segments.

Whisper is NOT in requirements.txt (numpy only). It is imported lazily, so
this file imports fine without it. On Colab: pip install openai-whisper

Speaker labels are \\w+ only ("Speaker1", not "Speaker 1") so a simple
regex like ^(\\w+): (.*) parses every line.
"""
from __future__ import annotations

from transcript_placeholder import get_transcript as _placeholder

PAUSE_GAP = 0.8      # seconds of silence that counts as a speaker change
NUM_SPEAKERS = 2     # speakers alternate: 1, 2, 1, 2 ...
WHISPER_MODEL = "base"

_model = None


def _transcribe_segments(audio_path: str) -> list[dict]:
    """Run Whisper, return [{'start': s, 'end': s, 'text': str}, ...]."""
    global _model
    import whisper  # lazy: only needed when a real file is given

    if _model is None:
        _model = whisper.load_model(WHISPER_MODEL)
    result = _model.transcribe(audio_path)
    return result["segments"]


def segments_to_transcript(segments: list[dict],
                           pause_gap: float = PAUSE_GAP,
                           num_speakers: int = NUM_SPEAKERS) -> str:
    """Pause-based speaker labelling. Pure function, testable without audio.

    Rule: a gap longer than pause_gap between segments means the next
    person is talking. Consecutive segments from the same speaker are
    merged into ONE line, because the downstream classifier labels
    whole turns.
    """
    turns: list[list] = []  # [speaker_index, text]
    speaker = 0
    prev_end = None
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        if prev_end is not None and seg["start"] - prev_end > pause_gap:
            speaker = (speaker + 1) % num_speakers
        if turns and turns[-1][0] == speaker:
            turns[-1][1] += " " + text
        else:
            turns.append([speaker, text])
        prev_end = seg["end"]
    return "\n".join(f"Speaker{s + 1}: {t}" for s, t in turns) + "\n"


def get_transcript(audio_path: str | None = None) -> str:
    if audio_path is None:
        return _placeholder()
    return segments_to_transcript(_transcribe_segments(audio_path))


if __name__ == "__main__":
    # Self-test with fake segments, no audio or Whisper needed.
    fake = [
        {"start": 0.0, "end": 2.0, "text": " Let's kick off."},
        {"start": 2.2, "end": 4.0, "text": " First item is the budget."},   # same speaker, merged
        {"start": 5.5, "end": 7.0, "text": " We're 8% under."},            # pause -> Speaker2
        {"start": 8.5, "end": 9.5, "text": " Good."},                      # pause -> Speaker1
    ]
    out = segments_to_transcript(fake)
    print(out)
    assert out.splitlines()[0] == "Speaker1: Let's kick off. First item is the budget."
    assert len(out.splitlines()) == 3
    print("self-test passed")