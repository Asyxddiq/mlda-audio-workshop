"""
Placeholder for the RNN / Transcriber team's output (Pasan + Rahini).

This stands in for the real audio -> transcript pipeline while that track
is still being built. It exists so the LSTM + meeting-minutes pipeline
(Asy + Nomt) can be developed and tested end-to-end right now, without
waiting on the transcriber.

Contract for the real pipeline to match later:
    get_transcript(audio_path: str | None = None) -> str
    Given an audio file path (or nothing, for now), return a plain-text,
    speaker-labelled transcript. Downstream code should only ever call
    get_transcript() and never care whether the transcript came from here
    or from the real RNN model.

When Pasan/Rahini's transcriber is ready, swap the body of get_transcript()
for a call into their code. Nothing downstream should need to change.
"""

SAMPLE_TRANSCRIPT = """\
Alex: Let's kick off. First item is the Q3 budget review.
Priya: I've sent the numbers over, we're currently 8% under budget.
Alex: Good, let's keep monitoring that. Next, the workshop venue.
Priya: Venue is confirmed for the 14th, catering still pending.
Alex: Can you follow up with catering by Friday?
Priya: Yes, I'll email them today.
Alex: Great, last item, feedback from the last session.
Priya: Mostly positive, a few people wanted more hands-on time.
Alex: Noted, let's add that to the next agenda. Meeting adjourned.
"""


def get_transcript(audio_path: str | None = None) -> str:
    """Return a stand-in transcript.

    Ignores `audio_path` for now -- it's accepted so the function signature
    already matches what the real transcriber will need to provide.
    """
    return SAMPLE_TRANSCRIPT


if __name__ == "__main__":
    print(get_transcript())
