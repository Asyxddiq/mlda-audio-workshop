# Facilitator page: the hook (Agenda item 1, 00:00-00:08)

Script, key and this whole plan were revised on 17 Oct 2026 (see
`ATTEMPTS.md` for the full history, including why the first version was
replaced). The script was chosen after seeing how the model reads it
(Rule H, `plan/RULES.md`): `demo/script.txt` is the orchestrator session's
script, independently re-run and re-verified here, not reworded further.

## What to say

1. "You've just sat through an hour-long CCA meeting. Who actually
   remembers what was decided, and who's doing what?"
2. Run the skit live (two presenters read `demo/script.txt` as Mei and
   Ravi), recorded from the browser mic in `demo/hook_demo.ipynb`
   (facilitator notebook, cells 2-3), or play `demo/backup_meeting.wav` if
   nobody can act it out.
3. Let the room call out the decisions and to-dos. Don't correct them yet.
4. **Press one button** (`demo/hook_demo.ipynb` cell 3, or on a laptop:
   `python prototype/demo.py skit.wav`).
5. Read the printed minutes against what the room called out
   (`demo/ANSWER_KEY.md` has the reference list).
6. Close with the two "Sure." exchanges, read aloud (not run through the
   app on stage -- see "The two Sure. exchanges" below):
   - "Can you book the room?" "Sure."
   - "Shall we move it to Friday?" "Sure."
   Ask the room what "Sure." means each time. They get it instantly because
   they remember the line before. Then: "Today we teach a computer to do
   that."

## Running it: Colab (primary) or laptop (fallback)

**Colab, `demo/hook_demo.ipynb`** (facilitator-only -- not the participants'
notebook, which only has a one-line pointer to this file):
- Cell 1, **before the session starts**: installs Whisper, unzips
  `workshop_files.zip`, and preloads the `medium.en` model (slow the first
  time -- do this ahead of time, not live).
- Cell 2: records the skit from the browser microphone to `skit.wav`
  (needs mic permission in the browser tab; Colab, so a T4 GPU speeds up
  Whisper).
- Cell 3: the button, on the live recording.
- Cell 4: the same button on the backup WAV, if cell 2/3 didn't work out.

**Laptop fallback (no internet / Colab unavailable):** record the skit any
way you like (phone, laptop mic) and run
`python prototype/demo.py your_recording.wav`. If Whisper's default
`medium.en` is too slow or can't download on the day, set
`WHISPER_MODEL=small.en` first -- this was tested on the backup WAV and
gets the same result (see `ATTEMPTS.md` and `HANDOFF-WP1.md`).

## If the minutes differ from the room's list

The default text run (`python prototype/demo.py`) and the audio runs (live
skit's backup WAV, both `medium.en` and `small.en`, `pause_gap=1.5`) all
match the answer key exactly (10/10 turns, both decisions, both action
items, correct owners and deadlines -- see `HANDOFF-WP1.md`). If something
still looks off on the day:

- **An item the room called out is missing, or an extra item appears:**
  say so plainly -- "the model isn't perfect, here's a case where a human
  still beats it" -- and move on.
- **The live skit doesn't come out right:** the model needs the two
  speakers to strictly alternate turns, in one sentence each (no internal
  full stop) -- see `demo/script.txt`'s header and `ATTEMPTS.md` for why.
  If a turn gets skipped, merged or split while acting it out, re-run with
  the backup WAV instead (cell 4, or
  `python prototype/demo.py demo/backup_meeting.wav`).
- **Nobody can act it out at all:** use the backup WAV (cell 4) instead of
  skipping the skit.

## The backup WAV

`demo/backup_meeting.wav` is the script spoken in two synthetic Windows
voices (Microsoft Zira as Mei, Microsoft Hazel as Ravi -- both built into
Windows, no install needed), 2.0 s of silence between turns -- comfortably
over the `pause_gap=1.5` the demo uses for real two-person speech (not
`transcriber.py`'s untested-on-two-people 0.8 s default). Made with
`demo/make_backup_wav.py`; see that file's docstring for exactly how, and
`HANDOFF-WP1.md` for the full, honest result of running it through Whisper.

## The two "Sure." exchanges

These are a **human exercise** (D5 in `plan/SPEC.md`) -- read aloud to the
room, not run through the app on stage. We did run them through the model
once, honestly, to see what it does with them (`demo/sure_exchanges.txt`,
reported in `HANDOFF-WP1.md`): it is not perfect on the second exchange,
which is exactly the point of asking the room instead of the app.
