# HANDOFF-WP3: LSTM gates, training and Hands-on B (N6-N8)

**Branch:** `claude/build-workshop-gaps-756dec`.
**Last commit:** `86d4b8e` (WP2 round 2, passed review). Nothing committed
since -- same protocol as WP1/WP2: the orchestrator alone commits.
**Status:** N6, N7, N8 all done in `notebook/audio_workshop.ipynb`, plus
the WP2 carry-over fix below. Not touched: notebook section 5 (meeting
minutes, "your own meeting" -- WP4) and no frozen file.

---

## WP2 carry-over (nit, fixed first)

Cell 23's answer-key table quoted `small.en` as "this notebook's GPU
default" result, implying it was measured on the actual Colab GPU path,
when it was a CPU run like the others -- and on `small.en` the second-voice
clip and hiss are only 4 points apart (87% vs 83%), not a wide enough gap
to call "slightly" either way. Reworded: all three rows now say plainly
they're CPU runs, and the order sentence now says "held on the three sizes
we tried (CPU runs); check it on the day's Colab before the vote" instead
of implying a measured GPU result.

---

## Requirement IDs covered

| ID | Requirement | Where |
|---|---|---|
| N6 | LSTM: eraser/pencil/voice gates, `c = f·c_prev + i·g`, 5-layer hourglass picture, `<eot>` bookmark | Section 4 intro rewritten; two new cells (hourglass shape, `<eot>` demo) |
| N7 | Training in one picture; worked `-log p` example; keyword vs LSTM on unseen lines; the "Venue is confirmed" failure; honesty on long-range memory | New "### Teaching it, and catching it out" subsection, before Hands-on B |
| N8 | Hands-on B: the check, then the knob, showing the toy result AND the trained model's score; "keep a lot" ≠ "better" | New cell after the existing Part B cells, sweeping -6/0/+6 |

---

## What changed, cell by cell

(Final notebook's cell numbers, 0-indexed, after all WP3 edits.)

**Section 4 header (cell 32), rewritten:** now names all three gates
(eraser = forget, pencil = input, **voice = output**, missing before),
gives the cell-state line in both words and symbols
(`c = f·c_prev + i·g`), and introduces `<eot>` as the "speaker changed"
bookmark.

**New cells 33, 35 (N6):**
- 33: computes the real hourglass shape live, on the trained model and
  this notebook's own loaded meeting (`transcript`, from section 1) --
  not hardcoded from the README. Also defines `turns`, `output`,
  `turn_ends`, reused by N7's loss-example cell and (again, redundantly
  but harmlessly, as before) by section 5.
- 35 (with markdown 34): tokenizes a 2-turn example with
  `lstm_model.tokenize_transcript`, printing the token list with `<eot>`
  visible between the turns.

Cell 36 (fold/unfold): unchanged.

**New subsection, cells 37-41 (N7), "### Teaching it, and catching it
out", inserted before "### YOUR TURN (Part B)":**
- 37: training-in-one-picture, in words, and the loss formula.
- 38: the worked `-log p` example -- real numbers from
  `lstm_model.label_probabilities` on this notebook's own meeting, for
  the turn the model gets right ("Yes, I'll email them today.") and the
  one it misses ("Venue is confirmed for the 14th...").
- 39: bridges to testing on unseen lines.
- 40: `label_turns_keywords` vs `label_turns_model` on
  `prototype/demo_student_council.txt` (18 turns, three speakers, never
  in `training_data.py` -- see that file's own header), with the hand
  labels and both labellers' counts, plus the one line the trained LSTM
  gets wrong.
- 41: the honesty line (SPEC tension 3) -- the task needs about one
  previous turn, not a long memory; the trained LSTM is never claimed to
  remember further back than a plain RNN. Also states plainly, from the
  actual numbers above, that the keyword checklist happens to catch
  "Venue is confirmed" (it contains the word "confirmed") while the
  trained model misses it -- neither approach is perfect, reported as
  found, not smoothed over.

**Hands-on B (cells 42-44):** unchanged, code untouched.

**New cells 45-46 (N8), after the knob question, before section 5:**
sweeps `hands_on.FORGET_KNOB` over -6, 0, +6 and calls `hands_on.part_b()`
for each, printing both the toy 12-word gap and the trained model's test
score at each setting in one go. `hands_on.py` itself untouched -- the
knob is set and `part_b()` called exactly as the existing cell 43 already
does, just three times.

Section 5 (meeting minutes, "your own meeting"): untouched.

---

## Verified numbers (every one from a command run here)

**Hourglass shape, live on this notebook's own meeting (cell 33):**
```
layer 1    8 units   88 steps
layer 2   16 units   44 steps
layer 3   32 units   22 steps
layer 4   16 units   44 steps
layer 5    8 units   88 steps
```
Matches `README.md`'s own worked example exactly (same token count, 88,
for this particular meeting) -- confirmed by running it, not copied from
the README.

**`<eot>` demo (cell 35):**
```
['can', 'you', 'follow', 'up', 'with', 'catering', 'by', 'friday', '?', '<eot>', 'yes', "i'll", 'email', 'them', 'today']
'<eot>' marks "speaker changed" between the two turns (turn_end_indices = [9, 14]).
```

**Loss example (cell 38):**
```
Venue is confirmed for the 14th, catering still pending.
  gold = decision, p(correct) = 0.004, loss = 5.61

Yes, I'll email them today.
  gold = action_item, p(correct) = 0.996, loss = 0.00
```

**Keyword vs trained LSTM, held-out meeting (cell 40):**
```
keyword rules: 12/18 turns right
trained LSTM:  17/18 turns right

LSTM miss: 'No refreshments this time, we are saving the money for the trip.'  (gold decision, got discussion)
```
A finding worth flagging explicitly (and flagged in cell 41): on the
"Venue is confirmed" turn specifically (from the loss example above, not
this held-out file, which doesn't contain that exact line), the keyword
checklist actually gets it right (`"confirmed"` is literally in
`meeting_minutes._DECISION_KEYWORDS`) while the trained LSTM gets it
wrong. The notebook says this plainly rather than picking the framing
that makes the trained model look better.

**N8 knob sweep, solved (verified standalone with `hands_on_reference.py`'s
answer, monkeypatched in -- never written into `hands_on.py`):**
```
FORGET_KNOB = -6   toy gap dies by word 3   trained model: 10/45 (0.22)
FORGET_KNOB = +0   toy gap fades gradually  trained model: 35/45 (0.78)
FORGET_KNOB = +6   toy gap GROWS each word  trained model: 22/45 (0.49)
```
`35/45` at 0 matches the reviewer's own recorded WP2-era number exactly.
Both extremes score worse than 0 -- confirms "keep a lot is not better"
honestly, from a real run, not asserted. (With the notebook's own
unfilled `lstm_cell_update` stub, this cell correctly prints "not written
yet" three times instead, same as every other YOUR-TURN cell before it's
solved -- shown in the full run below.)

---

## Per-cell CPU timings against blocks 6-8 (20, 10, 20 minutes)

All from the full zip-only run below (`base.en`, CPU, no GPU available):

| Block | Agenda minutes | New/changed cells | CPU time |
|---|---|---|---|
| 6 (LSTM) | 20 | hourglass shape (33): 0.2s; `<eot>` demo (35): 0.0s; fold/unfold (36, unchanged): 0.0s | ~0.2s |
| 7 (Teaching it, catching it out) | 10 | loss example (38): 0.1s; keyword-vs-LSTM (40): 0.3s | ~0.4s |
| 8 (Hands-on B) | 20 | YOUR TURN stub (43): 0.0s; knob sweep (46): 0.0s (unfilled; the solved path above ran in well under a second standalone, 45-turn scoring three times on a tiny trained model) | ~0.0s |

All three blocks' compute is a fraction of a second against 20, 10 and 20
minutes of talk/hands-on time -- nothing here is close to overrunning.

---

## Full end-to-end run, empty folder with only the (rebuilt) zip

```
$ python notebook/make_bundle.py
wrote .../notebook/workshop_files.zip (3.8 MB, 22 files)   # +1: demo_student_council.txt (N7)

$ <nbclient run from a folder holding only the fresh zip and the notebook>
[ 2]    0.2s  [ 4]   13.9s  [ 6]    2.5s  [ 8]    0.0s  [10]    0.2s  [12]   14.3s
[14]    4.1s  [16]    1.9s  [18]    2.0s  [20]    0.0s  [22]    0.3s  [24]    8.0s
[26]    0.0s  [28]    0.0s  [31]    0.3s  [33]    0.2s  [35]    0.0s  [36]    0.0s
[38]    0.1s  [40]    0.3s  [43]    0.0s  [46]    0.0s  [48]    0.2s  [50]    0.2s

=== OK in 57.2s ===
```
All 52 cells ran without error. Cell 46 (knob sweep) printed "not written
yet" three times, as expected for the unfilled stub -- checked the
executed notebook's output directly, not just "OK, no exception" (shown
above, under "Verified numbers").

**Compile check, every code cell, full joined source** (per WP2's lesson
about the embedded-newline bug -- found and fixed two more instances of
exactly that mistake while writing WP3's new cells, before this check was
run; both are gone from the final notebook):
```
$ python -c "
import json
nb = json.load(open('notebook/audio_workshop.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        compile(''.join(c['source']), f'cell{i}', 'exec')
print('all code cells compile cleanly')
"
all code cells compile cleanly
```

### `run_tests.py` and frozen-file check

```
$ python prototype/run_tests.py
PASS  a. imports                  11 modules import  (0.9s)
PASS  b. weights                  5 members, labels discussion / decision / action_item  (0.1s)
PASS  c. shipped KPIs reproduce   dev 0.96/0.948, test 0.84/0.825, 36-turn test 0.86/0.853, sample 7/9, probes 9/12  (6.4s)
PASS  d. demo minutes             no decisions, one action item (catering by Friday, Priya)  (0.3s)
PASS  e. --text example           runs, model matches the hand labels 7/7  (0.5s)
PASS  f. hands-on                 solution: both parts OK, 35/45 at knob 0; unfilled version says 'not written yet'  (2.5s)
PASS  g. training data            leak checks pass  (4.4s)
PASS  h. scorecard                frozen scorecard runs (hash ignores line endings)  (2.1s)
PASS  i. smoke runs               5 scripts exit 0  (5.2s)
PASS  j. text clean-up            text clean-up: 31/31 clean-up checks pass  (1.0s)
PASS  k. holdout meetings         holdout 1 0.879/0.806, holdout 2 0.852/0.762 (accuracy / macro-F1)  (6.6s)
SKIP  l. audio (optional)         no --audio FILE given  (0.0s)
PASS  m. repo unchanged           git status and git diff identical before and after  (0.2s)

12 passed, 0 failed, 1 skipped in 30.5 s
```

```
$ git diff --stat main -- prototype/trained_lstm.npz training_data.py scorecard_data.py \
    holdout_data.py minutes_tests.py normalise_tests.py lstm_model.py train_lstm.py \
    meeting_minutes.py transcriber.py hands_on_reference.py testing/ prototype/hands_on.py
(no output -- nothing frozen was touched)
```

---

## Files changed

```
 notebook/audio_workshop.ipynb | cells 32-46 rewritten/extended, 9 new cells (52 cells total, was 43)
 notebook/make_bundle.py       | +demo_student_council.txt to PROTOTYPE (N7's held-out meeting)
```
No new standalone files this round (N6-N8 reuse existing `prototype/`
modules: `lstm_model.py`, `meeting_minutes.py`, `pipeline.py`,
`hands_on.py` -- all read-only, none edited).

---

## Not done / open

- **GPU/Colab timing**: same limitation as WP1/WP2, no GPU available here.
  The CPU timings above are all well under a second per cell regardless,
  so this is unlikely to matter, but it is still unverified on an actual
  T4.
- Nothing else in WP3's deliverable list (N6-N8, `plan/WORK_PACKAGES.md`)
  is outstanding. Section 5 deliberately untouched (WP4).
