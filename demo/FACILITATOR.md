# Facilitator page: the hook

Agenda item 1, 00:00-00:08. Notebook: `demo/hook_demo.ipynb`.
Cell numbers count from 0, markdown and code cells together.
Blocks 2-10 are in `demo/RUNSHEET.md`.

## Running order

1. Before the session, run cell 2. It installs Whisper and loads the model. The first run is slow.
2. Rehearse once with the checklist below. The live skit is untested until you do.
3. Ask the room:
   > "You've just sat through an hour-long CCA meeting. Who actually remembers what was decided, and who's doing what?"
4. Run the skit. Live: run cell 4 to record. Backup WAV: run cell 8 to play it.
5. Let the room call out the decisions and to-dos. Don't correct them. Then press the button: cell 6 (live) or cell 10 (backup).
6. Read the two "Sure." exchanges aloud. Don't run them on stage.
   > "Can you book the room?" "Sure."
   > "Shall we move it to Friday?" "Sure."

   Ask what "Sure." means each time. Then say:
   > "Today we teach a computer to do that."

## If something goes wrong

| What happens | Do this |
|---|---|
| Cell 2 fails, or Whisper won't load | Run cell 12. It reads `demo/script.txt` as text. No audio, instant. |
| Skit not rehearsed, or acting it feels risky | Run cell 8 to play the backup WAV. Let the room call out. Then run cell 10. This is the one path verified end to end. |
| A turn is skipped, merged or split (live skit) | Switch to the backup WAV: cell 8, then cell 10. |
| Minutes differ from the room's list | Say: "The model isn't perfect. Here a human still beats it." Move on. Don't re-run for a different answer. |
| No GPU (CPU-only Colab) | Cell 2 prints `GPU: no` and switches to `small.en`. The press takes about 1 minute, not 3. |
| Mic permission denied, or no browser mic | Use the backup WAV (cells 8, 10) or the laptop fallback below. |

## Pacing for the live skit

1. Say each turn in one go. No long pause inside it.
2. Two turns in the script have two sentences. Say both without stopping.
3. Leave a clear 2-second pause before the other person speaks.
4. Don't overlap.

Don't use a natural pace. The transcriber starts a new speaker at a gap of 1.5 seconds or more. A quick back-and-forth gets heard as one person. A long pause mid-turn gets heard as a speaker change. Either way the owners in the minutes come out wrong.

## Rehearsal checklist

- [ ] Rehearse the live skit with the two people who will act it out. Follow the pacing above. Only the backup WAV has been tested end to end.
- [ ] Record the rehearsal on a phone. Run `python prototype/demo.py your_recording.wav`. Check it against `demo/ANSWER_KEY.md`.
- [ ] If it doesn't match the key, use the backup WAV on the day.
- [ ] Run cell 2 of `demo/hook_demo.ipynb` before the session.
- [ ] `python prototype/run_tests.py` says `0 failed`.
- [ ] Both backup voices (Zira as Mei, Hazel as Ravi) are female. Say the names aloud so the room can tell them apart.

## What to expect in the printed minutes

Read them against the room's list. `demo/ANSWER_KEY.md` has the reference. Two harmless differences:

- Whisper may write "T-shirts" as "t -shirts" in audio runs. The typed run says "T-shirts". The speech-to-text stage does this, not the minutes model. Explain only if asked.
- Action items keep the full spoken sentence ("I'll send the order form to everyone tonight."). The key has the shorter "Send the order form to everyone". Same item.

## Cells in `demo/hook_demo.ipynb`

Facilitator only. It uses `facilitator_files.zip`, which you keep. The participants' `workshop_files.zip` holds no demo answers (`demo/script.txt` and the answer key are in neither participant zip). It is safe to send early. See `demo/RUNSHEET.md`, onboarding.

| Cell | What it does |
|---|---|
| 2 | Run before the session. Installs Whisper, unzips `facilitator_files.zip`, prints whether a GPU is present, preloads the model (`medium.en` with a GPU, `small.en` without). |
| 4 | Records the skit from the browser mic to `skit.wav`. Needs mic permission in the tab. |
| 6 | The button, on the live recording. |
| 8 | Plays the backup WAV to the room. No button yet. |
| 10 | The button, on the backup WAV. Run only after the room has called out. |
| 12 | Text-only fallback. No Whisper at all. |

## Laptop fallback (no internet, or no Colab)

1. Record the skit with any app (phone, laptop mic).
2. Run `WHISPER_MODEL=small.en python prototype/demo.py your_recording.wav`.

Use `small.en`. On the backup WAV (53.5 s of audio, nothing else running), `medium.en` took 2 min 56 s on a laptop CPU. `small.en` took 1 min 7 s. The result is the same.

## Timing

- The script is 68 spoken words. Read straight through, it takes 25-30 s.
- Nine speaker changes, each with a 2 s pause, make the skit about 50 s. The Agenda says "~1 minute".
- The backup WAV runs 53.5 s. It has 2.0 s of silence between turns.

## The backup WAV

`demo/backup_meeting.wav` is the script in two synthetic Windows voices. Microsoft Zira is Mei. Microsoft Hazel is Ravi. Both are built into Windows. There is 2.0 s of silence between turns, over the `pause_gap=1.5` the demo uses. Made with `demo/make_backup_wav.py` (see its docstring). It is the only path run and verified end to end.

## The two "Sure." exchanges

These are a human exercise. Read them aloud. Don't run them in the app on stage.

We did run them through the model once (`demo/sure_exchanges.txt`). It gets the second exchange wrong. That is why you ask the room instead of the app. The notebook's "Sure. test" cell (block 7) shows the model's confidence.

