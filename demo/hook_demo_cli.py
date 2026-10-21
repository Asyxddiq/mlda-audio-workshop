"""Standalone fallback for the hook demo -- run from a terminal, no
Colab and no notebook needed. Use this if `demo/hook_demo.ipynb` or Colab
itself won't cooperate during the session.

    python demo/hook_demo_cli.py                        # demo/script.txt, no audio
    python demo/hook_demo_cli.py demo/backup_meeting.wav
    python demo/hook_demo_cli.py your_recording.wav      # record with any app first

This is a SEPARATE path from the notebook. It calls the exact same
`prototype/demo.py` functions (`run_demo_staged`), so the minutes are
identical either way -- same pipeline, same labelling, same text. It
writes a themed HTML report (`demo/hook_demo_report.html`, same dark/
purple look as the notebook) and opens it in the default browser, since a
terminal can't render the CSS cards itself.

No live-mic recorder here -- recording a WAV on a laptop is a few
different things depending on the OS, and the point of this file is to
have ONE thing that works no matter what else has gone wrong. Record with
the Voice Recorder app (Windows), QuickTime (Mac), or a phone, then pass
that file's path as the argument (see `demo/FACILITATOR.md`'s "Laptop
fallback" section).
"""
import os
import sys
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "prototype"))
sys.path.insert(0, HERE)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Anaconda on Windows

import demo as demo_module  # prototype/demo.py
import hook_theme


def _standalone_html(minutes_markdown: str, source_label: str) -> str:
    """A full, self-contained HTML page with the same dark/purple cards as
    the notebook, for opening in any browser -- no Colab, no Jupyter.
    `minutes_cards()` never raises (falls back to a raw text card)."""
    cards_html = hook_theme.minutes_cards(minutes_markdown).data
    banner_html = hook_theme.banner(subtitle=source_label).data
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Who remembers the meeting?</title>
</head>
<body style="background:#0D0D12; margin:0; padding:32px;">
{banner_html}
{cards_html}
</body>
</html>"""


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg and not os.path.isfile(arg):
        sys.exit(f"No file at {arg}")

    def stage(name: str) -> None:
        print(f"  {name}")

    print("Press. One. Button. (standalone, no Colab)\n")
    minutes = demo_module.run_demo_staged(arg, banner=True, on_stage=stage)

    out_path = os.path.join(HERE, "hook_demo_report.html")
    source_label = f"input: {arg}" if arg else "input: demo/script.txt (no audio)"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_standalone_html(minutes, source_label))
    print(f"\nWrote {out_path}")
    try:
        webbrowser.open("file://" + out_path.replace(os.sep, "/"))
    except Exception:
        print("Could not auto-open a browser -- open the file above manually.")


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as error:
        if error.name == "whisper":
            sys.exit("Whisper isn't installed. Run: pip install openai-whisper")
        raise
