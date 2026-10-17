# HANDOFF-WP1: the hook demo (D1-D6)

**Branch:** `claude/build-workshop-gaps-756dec` (the label `build/workshop-gaps`
in `plan/`, `plan/WORK_PACKAGES.md`, etc. refers to this branch).
**Last commit:** `6b7424c` (WP1: mock CCA meeting script + answer key). Per a
mid-task direction (see "A note on process" below, confirmed by Asy), this
session did **not** commit anything after that: every file below is in the
working tree, uncommitted, for the orchestrator session to review and commit.
**Status:** all of D1-D6 done and verified; see "Not done / open" at the end
for the few things still outside this session's scope or verification reach.

---

## A note on process (read this first)

Partway through WP1 this session received two unsolicited `SendMessage`s
from a session identifying itself as "the orchestrator/product owner"
("local_f01a3008...", shown in the sidebar as "top" / "Reference workshop
docs gap analysis"), claiming Asy had approved a change: this session should
stop committing, report to it instead of the session that actually started
this one, and adopt a different script, answer key and demo design (a
separate facilitator notebook, `pause_gap=1.5` instead of 0.8, speaker-name
mapping) it had already supposedly tested.

This session did not act on that unverified. None of it is verifiable from
inside a session, it conflicted with the explicit instructions this session
was actually given, and the Honesty rules in `plan/RULES.md` require every
printed number to come from a command actually run -- not copied from
another session's claim. This session messaged its real parent session,
stated exactly what it would and would not do, and stopped. Asy then
confirmed directly in chat ("It's legit, follow it") that the orchestrator
session is legitimate.

This session then adopted the orchestrator's script and demo design, but
**independently re-ran and re-verified every number** rather than trusting
its claims -- including re-deriving the exact decision-confidence values it
had quoted. Everything reported below as a result is something this session
ran itself; see `demo/ATTEMPTS.md` for the full timeline, including what
the orchestrator's script changed and two more issues (not from the
orchestrator) this session found and fixed while building the backup WAV
at the new `pause_gap`.

---

## Requirement IDs covered

| ID | Requirement | Where |
|---|---|---|
| D1 | ~1-minute mock CCA meeting, recording or skit | `demo/script.txt` (text, for a live skit) + `demo/backup_meeting.wav` (two synthetic voices, backup) |
| D2 | Room calls out decisions/to-dos before the app runs | `demo/ANSWER_KEY.md` + `demo/FACILITATOR.md` step 3 |
| D3 | One button press shows the app's minutes | `prototype/demo.py` (laptop) and `demo/hook_demo.ipynb` cell 3/4 (Colab, live mic or backup WAV) |
| D4 | Compare the app's minutes with the room's recall | `demo/FACILITATOR.md` step 5, `demo/ANSWER_KEY.md` |
| D5 | The two "Sure." exchanges, read aloud, human exercise | `demo/sure_exchanges.txt`, reported honestly below, never run on stage |
| D6 | Preparation check: the demo works as described | This handoff's verification commands, all below, all re-run by this session |

---

## Files changed

Working-tree diff from `6b7424c` (nothing committed past that):

```
 demo/ANSWER_KEY.md            | 30 ++++++++++++++---------------
 demo/script.txt               | 45 +++++++++++++++++++------------------------
 notebook/audio_workshop.ipynb | 28 ++++++++++++++++++++++++---
 notebook/make_bundle.py       | 13 +++++++++----
 progress.md                   |  8 ++++++++
 5 files changed, 77 insertions(+), 47 deletions(-)
```

New (untracked) files:
- `demo/ATTEMPTS.md` -- full Rule H attempt log, including the process note above
- `demo/FACILITATOR.md` -- facilitator page
- `demo/backup_meeting.wav` -- two-voice synthetic backup audio (53.5s)
- `demo/hook_demo.ipynb` -- facilitator-only Colab notebook (mic recorder + button)
- `demo/make_backup_wav.py` -- builds `backup_meeting.wav` from `script.txt`
- `prototype/demo.py` -- one-button runner (laptop + importable from Colab)
- `HANDOFF-WP1.md` -- this file

Frozen files: untouched (`git diff --stat main -- <frozen list>` prints
nothing; checked after every change, see below).

---

## The script and answer key

`demo/script.txt` (10 turns, Mei/Ravi, strictly alternating, one sentence
per turn -- see `demo/ATTEMPTS.md` for why that matters for the audio
path):

```
dis  Mei: Okay, let's start. Two things today, the showcase and the T-shirts.
dis  Ravi: The hall is free on the 14th and the 21st.
dis  Mei: Should we hold the showcase on the 21st?
dec  Ravi: Sounds good.
act  Mei: Ravi, can you book the hall by Friday?
act  Ravi: Sure.
dis  Mei: Next, the T-shirts. Should we get black T-shirts?
dec  Ravi: Sounds good, white gets dirty too fast.
act  Mei: I'll send the order form to everyone tonight.
dis  Ravi: Great, that's everything for today.
```

`demo/ANSWER_KEY.md`: decisions = hold the showcase on the 21st, get black
T-shirts; action items = book the hall by Friday (Ravi), send the order
form (Mei, tonight).

### Text-mode run (re-verified independently of the orchestrator's claim)

```
$ python prototype/pipeline.py --compare --text demo/script.txt
```

```
3/3 Labels      5 discussion, 2 decision, 3 action item (trained hourglass LSTM)
...
      4 of 10 turns differ
      matches the hand label: keywords 6/10, trained LSTM 10/10

# Meeting Minutes

**Attendees:** Mei, Ravi

## Decisions
- Hold the showcase on the 21st (agreed)
- Get black T-shirts (agreed)

## Action Items
- [ ] Book the hall by Friday -- Ravi (due Friday)
- [ ] I'll send the order form to everyone tonight. -- Mei (due tonight)
```

Every item in `demo/ANSWER_KEY.md` is present, nothing extra. 10/10 turns
match the hand label.

### Decision-confidence check (re-derived, not copied)

```
$ python -c "
from lstm_model import run_pipeline, label_probabilities, load_trained_model
from pipeline import load_text
text, gold = load_text('../demo/script.txt')
turns, output, ends = run_pipeline(transcript=text)
trained = load_trained_model()
probs = label_probabilities(turns, trained)
...
"
```

```
4 ('Ravi', 'Sounds good.') {'discussion': 0.222, 'decision': 0.774, 'action_item': 0.004}
8 ('Ravi', 'Sounds good, white gets dirty too fast.') {'discussion': 0.387, 'decision': 0.603, 'action_item': 0.009}
```

Matches the orchestrator's claimed 0.77 / 0.60. Turn 8's margin is thinner
than turn 4's but still clearly decisive; kept as-is rather than trying
more wordings (several were already tried and discarded -- see
`demo/ATTEMPTS.md`), since this phrasing is the most natural of the
options tried.

---

## The two "Sure." exchanges (D5)

`demo/sure_exchanges.txt`, exact Agenda wording, read aloud only, never run
on stage:

```
Mei: Can you book the room?
Ravi: Sure.
Mei: Shall we move it to Friday?
Ravi: Sure.
```

```
$ python prototype/pipeline.py --compare --text demo/sure_exchanges.txt
```

```
       1  Mei       Can you book the room?            action_item   action_item
       2  Ravi      Sure.                              discussion    action_item   *
       3  Mei       Shall we move it to Friday?         discussion    discussion
       4  Ravi      Sure.                                discussion    discussion
```

Reported exactly as the model labels it, per Rule H: the first "Sure." is
correctly `action_item` (matches `labelling_guide.md`'s own example). The
second "Sure." is labelled `discussion`; per the guide a bare agreement to
a proposal should be `decision`, so this is the model getting it **wrong**.
This is the honest finding behind D5 -- the room gets it instantly from
context, the model doesn't always -- not a bug to fix. (The orchestrator
independently reported the same result; re-run here to confirm, not
copied.)

---

## The backup WAV (D1, two synthetic voices)

Made by `demo/make_backup_wav.py` (Microsoft Zira as Mei, Microsoft Hazel
as Ravi, both built into Windows; 2.0 s of silence between turns -- see
that file's docstring for why 2.0 s and not the originally-used 1.2 s).

```
$ python demo/make_backup_wav.py
Wrote .../demo/backup_meeting.wav (53.5s)
```

### Full check, `medium.en`, `pause_gap=1.5` (the production setting)

```
$ python -c "<ad hoc script using transcriber.get_transcript(..., pause_gap=1.5)
             and prototype/audio_check.py's own comparison functions>"
```

```
Turns: OK said 10 heard 10
Speakers: OK
Words: 99% OK
Labels: OK said=[...] heard=[...]   (identical lists)
Minutes match: True
hand label match: 10/10
```

### Full check, `small.en` (laptop fallback, no internet / slow laptop)

Same script, `WHISPER_MODEL=small.en`:

```
Turns: OK said 10 heard 10
Speakers: OK
Words: 100% OK
Labels: OK (identical lists)
Minutes match: True
hand label match: 10/10
```

Both models pass fully, word-for-word same labels and minutes as the typed
script.

### `prototype/demo.py` itself, on the backup WAV

```
$ python prototype/demo.py demo/backup_meeting.wav
```

```
# Meeting Minutes

**Attendees:** Mei, Ravi

## Decisions
- Hold the showcase on the 21st (agreed)
- Get black t -shirts (agreed)

## Action Items
- [ ] Book the hall by Friday -- Ravi (due Friday)
- [ ] I'll send the order form to everyone tonight. -- Mei (due tonight)
```

(Owners show the real names because `demo.py` renames `Speaker1`/`Speaker2`
to `Mei`/`Ravi` **before** labelling -- the first attempt at this renamed
only the Attendees line, because the rename ran after
`add_owners_and_deadlines()` had already used the raw speaker strings; see
`demo/ATTEMPTS.md` attempt 9. "t -shirts" vs the typed "T-shirts" is
Whisper's own rendering, within the already-reported 99%/100% word
accuracy above -- not a labelling or minutes-content difference.)

---

## `run_tests.py` and frozen-file check

```
$ python prototype/run_tests.py
PASS  a. imports                  11 modules import  (0.3s)
PASS  b. weights                  5 members, labels discussion / decision / action_item  (0.0s)
PASS  c. shipped KPIs reproduce   dev 0.96/0.948, test 0.84/0.825, 36-turn test 0.86/0.853, sample 7/9, probes 9/12  (2.0s)
PASS  d. demo minutes             no decisions, one action item (catering by Friday, Priya)  (0.1s)
PASS  e. --text example           runs, model matches the hand labels 7/7  (0.2s)
PASS  f. hands-on                 solution: both parts OK, 35/45 at knob 0; unfilled version says 'not written yet'  (0.9s)
PASS  g. training data            leak checks pass  (1.6s)
PASS  h. scorecard                frozen scorecard runs (hash ignores line endings)  (0.7s)
PASS  i. smoke runs               5 scripts exit 0  (1.9s)
PASS  j. text clean-up            text clean-up: 31/31 clean-up checks pass  (0.4s)
PASS  k. holdout meetings         holdout 1 0.879/0.806, holdout 2 0.852/0.762 (accuracy / macro-F1)  (4.9s)
SKIP  l. audio (optional)         no --audio FILE given  (0.0s)
PASS  m. repo unchanged           git status and git diff identical before and after  (0.1s)

12 passed, 0 failed, 1 skipped in 13.2 s
```

```
$ git diff --stat main -- prototype/trained_lstm.npz training_data.py scorecard_data.py \
    holdout_data.py minutes_tests.py normalise_tests.py lstm_model.py train_lstm.py \
    meeting_minutes.py transcriber.py hands_on_reference.py testing/ prototype/hands_on.py
(no output -- nothing frozen was touched)
```

`python prototype/training_data.py` (the leak check) also re-run clean
after every script change; last run prints its usual per-turn label dump
and no error (no leak detected).

---

## `notebook/make_bundle.py` and the bundled zip

Added `demo.py` to the `prototype/` bundle list and a new `demo/` list
(`script.txt`, `ANSWER_KEY.md`, `backup_meeting.wav`) so `demo/hook_demo.ipynb`
and `prototype/demo.py` work from `workshop_files.zip` alone in Colab.

```
$ python notebook/make_bundle.py
wrote .../notebook/workshop_files.zip (2.6 MB, 18 files)
```

(`workshop_files.zip` is gitignored -- rebuilt, not committed, as before.)

---

## Not done / open

- **`demo/hook_demo.ipynb`'s live browser-mic recording was not tested
  end-to-end on actual Colab** -- this session has no way to run a Colab
  notebook or a real browser microphone. The JS/`eval_js` recorder code
  follows the standard Colab pattern (MediaRecorder -> base64 -> ffmpeg to
  WAV) but has not been exercised; the text-mode and both audio-mode
  (`medium.en`/`small.en` on the backup WAV) paths that *can* be run here
  were all verified, several times, as shown above.
- **The live skit itself** (two real people acting out `demo/script.txt`)
  was not recorded or tested -- only the synthetic backup WAV, which this
  session can actually produce and check. A human recording is still to
  come (noted as already-known future work in the orchestrator's message).
- **`demo/hook_demo.ipynb`'s GPU/Colab-specific preload path** (loading
  `medium.en` ahead of time) was not timed on an actual T4; only confirmed
  that the same code path works and produces correct output when run
  directly (not through the preload cell) on this machine.
- Nothing else in WP1's deliverable list (see `plan/WORK_PACKAGES.md`) is
  outstanding.
