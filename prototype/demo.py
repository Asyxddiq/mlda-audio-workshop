"""
The one-button demo for the workshop hook (Agenda item 1, D3): press one
button, see the app produce the same list as minutes.

    python prototype/demo.py                     # default: demo/script.txt
    python prototype/demo.py demo/backup_meeting.wav
                                                  # the backup WAV (two synthetic voices)
    python prototype/demo.py some_other.wav       # any other recording / text file

No flags needed for the default case. Just the pipeline with the demo
script as the default input, and the minutes printed large and readable (a
boxed banner) instead of pipeline.py's three-stage log, which is for
debugging, not for an audience.

Runs on a laptop (this file, from the command line) or in Colab
(`demo/hook_demo.ipynb`, the facilitator-only notebook -- the participants'
notebook does not call this file at all; see RULES.md's "where the builder
may write").

Two audio-specific choices, both from the 17 Oct PM direction (see
demo/ATTEMPTS.md):
  - pause_gap=1.5, not transcriber.py's 0.8 default: 1.5s is what was
    actually tested on two people talking (see testing/); 0.8s has only
    been tested on a solo reading.
  - Speaker1/Speaker2 are renamed to the real actor names (SPEAKER_NAMES)
    before labelling. The model itself never reads speaker names
    (tokenize_transcript drops them -- lstm_model.py), so this does not
    change any label; it only changes whose name shows up as the owner in
    the printed minutes.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Anaconda on Windows (see RULES.md)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEFAULT_SCRIPT = os.path.join(os.path.dirname(HERE), "demo", "script.txt")
AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
PAUSE_GAP = 1.5
# The two actors in demo/script.txt, in speaking order (Mei speaks turn 1).
# Edit this if a different pair of facilitators reads the skit.
SPEAKER_NAMES = ["Mei", "Ravi"]


def _banner(minutes: str) -> str:
    """Minutes text, boxed and upper-cased on the headings, for the room to read."""
    width = max((len(line) for line in minutes.splitlines()), default=40) + 4
    width = max(width, 50)
    top = "+" + "=" * width + "+"
    lines = [top]
    for line in minutes.splitlines():
        lines.append("|  " + line.ljust(width - 2) + "|")
    lines.append(top)
    return "\n".join(lines)


def _rename_speakers(turns: list[tuple[str, str]], names: list[str]) -> list[tuple[str, str]]:
    """'Speaker1'/'Speaker2' -> the real actor names, by order of first
    appearance. Safe to do after transcription: the model only ever sees
    tokens, never the speaker label (tokenize_transcript, lstm_model.py)."""
    mapping: dict[str, str] = {}
    renamed = []
    for speaker, text in turns:
        if speaker not in mapping:
            mapping[speaker] = names[len(mapping)] if len(mapping) < len(names) else speaker
        renamed.append((mapping[speaker], text))
    return renamed


def run_from_audio(audio_path: str, pause_gap: float = PAUSE_GAP,
                   speaker_names: list[str] = SPEAKER_NAMES) -> str:
    """Audio -> minutes, with pause_gap and speaker-name mapping (see module
    docstring). Used for both the live skit recording and the backup WAV."""
    from lstm_model import run_pipeline
    from meeting_minutes import generate_minutes, label_turns_keywords, label_turns_model
    from transcriber import get_transcript

    transcript = get_transcript(audio_path, pause_gap=pause_gap)
    turns, output, turn_end_indices = run_pipeline(transcript=transcript)
    # Rename BEFORE labelling: add_owners_and_deadlines() (inside label_turns_model /
    # label_turns_keywords) reads turns[i][0] as the owner's name, so renaming has to
    # happen here, not after, or the owners in the minutes stay "Speaker1"/"Speaker2"
    # even though the Attendees line already shows the real names (found while
    # testing this file -- see demo/ATTEMPTS.md). output/turn_end_indices don't
    # depend on the speaker strings, only on token position, so this is safe.
    turns = _rename_speakers(turns, speaker_names)
    labels = label_turns_model(turns, output, turn_end_indices) or label_turns_keywords(turns)
    return generate_minutes(turns, labels)


def run_demo(source: str | None = None, banner: bool = True) -> str:
    """Run the full pipeline on `source` (a WAV, or a text script) and print
    the minutes large and readable. Returns the minutes Markdown.

    `source=None` (the default, and what a bare `python prototype/demo.py`
    does) uses demo/script.txt -- the committed, answer-keyed mock meeting.

    `banner=False` (used by demo/hook_demo.ipynb) skips the boxed-text
    printout: the notebook shows the minutes once, large, with
    `IPython.display.Markdown` instead -- printing both was finding 8 in
    the WP1 review. The laptop command keeps the banner (no notebook
    rendering there).
    """
    from pipeline import run

    if source is None:
        source = DEFAULT_SCRIPT
    is_audio = source.lower().endswith(AUDIO_EXTENSIONS)

    print("Press. One. Button.\n")
    print(f"(input: {source})\n")

    if is_audio:
        minutes = run_from_audio(source)
    else:
        minutes = run(text=source, verbose=False)

    if banner:
        print(_banner(minutes))
        print("\nCompare this with what the room called out.")
    return minutes


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg and not os.path.isfile(arg):
        sys.exit(f"No file at {arg}")
    try:
        run_demo(arg)
    except ModuleNotFoundError as error:
        if error.name == "whisper":
            sys.exit("Whisper isn't installed. Run: pip install openai-whisper")
        raise
