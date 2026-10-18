# Facilitator page: the hook (Agenda item 1, 00:00-00:08)

Cell numbers below are `demo/hook_demo.ipynb`'s actual cell indexes (0-indexed,
counting markdown and code cells together), not section numbers -- cell 2 is
Setup, cell 4 is the recorder, cell 6 is the live-recording button, cell 8 is
the backup WAV, cell 10 is the text-only fallback. The detailed workshop-wide
run order (blocks 2-10, what to say, expected output, timing) is in
`demo/RUNSHEET.md`; this page covers the hook only.

## Running order, in six steps

1. **Before the session:** run cell 2 (installs Whisper, preloads the model --
   slow the first time, do this ahead of time).
2. **Rehearse** with the checklist below -- the live-skit path is untested
   until you do.
3. Ask the room: "Who actually remembers what was decided, and who's doing
   what?"
4. Run the skit (cell 4 + 6 live, or play the backup WAV from cell 8 if
   nobody can act it out, or hasn't rehearsed).
5. Let the room call out decisions and to-dos first -- **then** press the
   button.
6. Close with the two "Sure." exchanges, read aloud (not run on stage).

## If something goes wrong

| What happens | What to do |
|---|---|
| Cell 2 fails, or Whisper won't load | Use cell 10, the text-only fallback (`demo/script.txt`, no audio, instant) -- keeps the hook going with no model at all. |
| The live skit wasn't rehearsed, or acting it out live feels risky | Play the backup WAV from cell 8 instead (`display(Audio(...))` plays it to the room first, then the button). This is the one path verified end to end. |
| A turn gets skipped, merged or split (live skit) | Re-run with the backup WAV (cell 8). |
| The minutes differ from what the room called out | Say so plainly -- "the model isn't perfect, here's a case where a human still beats it" -- and move on. Don't re-run hoping for a different answer. |
| No GPU / a CPU-only Colab | Cell 2 prints `GPU: no` and switches to `small.en` automatically (about 1 minute instead of `medium.en`'s 3). |
| Mic permission denied, or no browser mic | Skip to the backup WAV (cell 8), or the laptop fallback below. |

## Pre-session rehearsal checklist

- [ ] Rehearse the live skit with the two people who will act it out: say
  each turn in one go, a clear ~2-second pause before the other person
  speaks, don't overlap (see "Pacing" below -- this has not been tested
  end to end, only the backup WAV has).
- [ ] Record that rehearsal on a phone, then run
  `python prototype/demo.py your_recording.wav` and check it against
  `demo/ANSWER_KEY.md`.
- [ ] If the rehearsal doesn't match the key, use the backup WAV on the
  day instead -- it is the one path that has actually been verified
  end to end.
- [ ] Run cell 2 of `demo/hook_demo.ipynb` before the session (preloads
  Whisper -- slow the first time).
- [ ] `python prototype/run_tests.py` says `0 failed`.
- [ ] Note: both synthetic voices in the backup WAV (Zira as Mei, Hazel as
  Ravi) are female -- the room may need the names said aloud to tell them
  apart, since the voices alone don't signal it.

## What to say

1. "You've just sat through an hour-long CCA meeting. Who actually
   remembers what was decided, and who's doing what?"
2. Run the skit live (two presenters read `demo/script.txt` as Mei and
   Ravi), recorded from the browser mic (cells 4 then 6 in
   `demo/hook_demo.ipynb`), or play `demo/backup_meeting.wav` (cell 8) if
   nobody can act it out or the live path hasn't been rehearsed.
   **Pacing:** say each turn in one go, with no long pause inside it (two
   turns in the script have two sentences -- say both without stopping);
   leave a clear ~2-second pause before the other person speaks; don't
   overlap. (Not "normal pace" -- the transcriber splits speakers on a gap
   of 1.5 seconds or more, so a natural, quick back-and-forth gets heard as
   one person, and a long pause mid-turn gets heard as a speaker change
   too, and either way the minutes' owners come out wrong. See
   `ATTEMPTS.md`'s backup-WAV attempts for why.)

   **The live-skit path is untested until the team rehearses it** (see the
   checklist above: two people, a phone recording, then
   `python prototype/demo.py <file>`). Until that rehearsal happens and
   passes, **the backup WAV is the safe route** for the actual event.
3. Let the room call out the decisions and to-dos. Don't correct them yet.
4. **Press one button** (`demo/hook_demo.ipynb` cell 6, or on a laptop:
   `python prototype/demo.py skit.wav`).
5. Read the printed minutes against what the room called out
   (`demo/ANSWER_KEY.md` has the reference list). Two things to expect,
   both harmless:
   - Whisper may write "T-shirts" as "t -shirts" in the audio runs (the
     typed-text run says "T-shirts"). That is the speech-to-text stage
     rendering a word oddly, not the minutes model -- no need to explain
     it unless someone asks.
   - The action items keep the full spoken sentence ("I'll send the order
     form to everyone tonight.") rather than the key's shorter phrasing
     ("Send the order form to everyone") -- same item, just not trimmed.
6. Close with the two "Sure." exchanges, read aloud (not run through the
   app on stage -- see "The two Sure. exchanges" below):
   - "Can you book the room?" "Sure."
   - "Shall we move it to Friday?" "Sure."
   Ask the room what "Sure." means each time. They get it instantly because
   they remember the line before. Then: "Today we teach a computer to do
   that."

## Running it: Colab (primary) or laptop (fallback)

**Colab, `demo/hook_demo.ipynb`** (facilitator-only -- uses its own
`facilitator_files.zip`, which the facilitator keeps; hand out
`workshop_files.zip`, the participants' separate zip, only *after* the
hook, so nobody sees `demo/script.txt` -- or the answer key, which is kept
out of both zips entirely -- before being asked to call out the decisions
and to-dos):
- Cell 2, **before the session starts**: installs Whisper, unzips
  `facilitator_files.zip`, prints whether a GPU is present, and preloads
  the model (`medium.en` with a GPU, `small.en` without -- slow the first
  time, do this ahead of time, not live).
- Cell 4: records the skit from the browser microphone to `skit.wav`
  (needs mic permission in the browser tab; a T4 GPU speeds up Whisper).
- Cell 6: the button, on the live recording.
- Cell 8: plays the backup WAV to the room, then the same button on it.
- Cell 10: text-only fallback, no Whisper at all, if the model won't load.

**Laptop fallback (no internet / Colab unavailable):** record the skit any
way you like (phone, laptop mic) and run
`python prototype/demo.py your_recording.wav`. `medium.en` on a laptop CPU
is slow -- timed on the backup WAV (53.5s of audio) with nothing else
running: `medium.en` 2 min 56 s, `small.en` 1 min 7 s. **`medium.en` is
well over 60 seconds on a laptop CPU, so the laptop fallback should use
`small.en`** (`WHISPER_MODEL=small.en python prototype/demo.py
your_recording.wav`) -- this gets the exact same result as `medium.en`,
just faster.

## Timing

The script is 68 spoken words -- about 25-30 s read straight through. With
the required ~2 s pauses between all nine speaker changes, the full skit
runs about 50 s, close to the Agenda's "~1 minute". The backup WAV itself
runs 53.5 s for the same reason (2.0 s silences between turns, not just
pauses in speech).

## The backup WAV

`demo/backup_meeting.wav` is the script spoken in two synthetic Windows
voices (Microsoft Zira as Mei, Microsoft Hazel as Ravi -- both built into
Windows, no install needed; both voices are female, see the checklist
above), 2.0 s of silence between turns -- comfortably over the
`pause_gap=1.5` the demo uses for real two-person speech. Made with
`demo/make_backup_wav.py`; see that file's docstring for exactly how. This
is currently the only path that has been run and verified end to end.

## The two "Sure." exchanges

These are a **human exercise** (D5 in `plan/SPEC.md`) -- read aloud to the
room, not run through the app on stage. We did run them through the model
once, honestly, to see what it does with them (`demo/sure_exchanges.txt`):
it is not perfect on the second exchange, which is exactly the point of
asking the room instead of the app. (The notebook's own "Sure. test" cell,
block 6-7, revisits this honestly with the model's confidence printed.)

---

## Provenance

The first script and key (`6b7424c`) were committed before any model run.
The final script and key (this version) were drafted by the PM with model
feedback during drafting, and the key was fixed from a plain reading of
each draft, never bent to match the model -- see `ATTEMPTS.md` for the full
history, both this session's own rewording of the superseded first script
and the PM's attempts on the final one.

Both decisions in the demo are phrased as agreement to a proposal
("Sounds good") -- this scores 0.77 and 0.60 for `decision` in the
model's own confidence, independently re-derived in
`plan/handoffs/HANDOFF-WP1.md`. Plain-fact decisions ("Venue is confirmed
for the 14th") are the model's known weak spot and are deliberately
**not** in this demo -- see `labelling_guide.md` and `plan/SPEC.md`'s
tensions list.

The default text run (`python prototype/demo.py`) and the backup-WAV runs
(both `medium.en` and `small.en`, `pause_gap=1.5`) all match the answer
key (10/10 turns, both decisions, both action items, correct owners and
deadlines -- see `plan/handoffs/HANDOFF-WP1.md`; "match" there means the
standard check's 80%-word-similar comparison, not byte-identical text, so
small wording differences are expected and fine).
