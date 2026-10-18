# HANDOFF-WP4: your own meeting, integration (N9) -- the last package

**Branch:** `claude/build-workshop-gaps-756dec`.
**Last commit:** `198fee6` (WP3 round 2, passed review). Nothing committed
since -- same protocol as every round: the orchestrator alone commits.
**Status:** N9 done, both zips split and tested, contents table and
section headers now carry Agenda block numbers and minutes, `progress.md`
and `debug_cheatsheet.md` updated, handoffs tidied into `plan/handoffs/`.
This is the last work package (`plan/WORK_PACKAGES.md`).

---

## WP3 carry-over (nit, fixed first)

Cell 39 said the held-out figures (131/149, 104/122 for the LSTM; 111/149,
93/122 for keyword rules) both came from "`run_tests.py` check k" -- but
check k only prints the LSTM's accuracy, not the keyword baseline's.
Reworded: "LSTM: `run_tests.py` check k; keyword rules: a per-meeting
count over `holdout_data.py`" -- the keyword count was computed by this
session, not printed by any existing script (see WP3's Round 2 for how
it was independently reproduced).

---

## Requirement IDs covered

| ID | Requirement | Where |
|---|---|---|
| N9 | Own-meeting cell: 5 lines, labels optional-but-prompted, model labels out, mismatches flagged, unknown words listed, a "which earlier line" prompt | New "### Your own meeting" subsection, section 5, after the minutes |
| Integration | `make_bundle.py` rebuilt; every cell run from an empty folder holding only the relevant zip; minutes-per-section table | Both zips split (WP3) and re-tested here; see the table below |

---

## N9: the own-meeting cell

Reuses `pipeline.load_text` exactly as written (frozen-adjacent,
untouched) by writing the typed text to a temp file and reading it back,
rather than re-implementing its parsing -- the same all-or-nothing label
rule `load_text` already enforces ("label every line or none") is the
rule this cell's own instructions state, not a new one invented for the
notebook.

Default example (kept similar in spirit to the old "Your own meeting"
cell, now with labels, and deliberately including a "plain fact" decision
so the exercise has something real to find on the first run, not just a
clean 5/5):
```
dis  Sam: Let's go over what's left before the trip.
dec  Lee: The venue is confirmed for Friday.
dis  Sam: Should we also book the shuttle?
act  Sam: Lee, can you call the shuttle company?
act  Lee: Sure.
```

Output (from the real run, below):
```
discussion   (you)   discussion   (model)   Sam: Let's go over what's left before the trip.
decision     (you)   discussion   (model)   Lee: The venue is confirmed for Friday.  <- X
discussion   (you)   discussion   (model)   Sam: Should we also book the shuttle?
action_item  (you)   action_item  (model)   Sam: Lee, can you call the shuttle company?
action_item  (you)   action_item  (model)   Lee: Sure.

# Meeting Minutes

**Attendees:** Lee, Sam

## Decisions
- (none)

## Action Items
- [ ] Call the shuttle company -- Lee

Words the model never saw in training (read as <unk>): lee, shuttle, what's
The model knows 553 words; anything else is read as <unk>.

Pick one line it got wrong. Which EARLIER line should it have remembered?
```
`553` is the real, re-verified vocab size (`len(load_trained_model()["vocab"])`),
not a guessed number. The deliberate miss ("Venue is confirmed...", the
same known weak spot as N7) means a participant has something genuine to
find on the very first run, without needing to write their own tricky
line first.

If no labels are given, the cell shows only the model's labels and prints
a line asking the participant to add `act`/`dec`/`dis` to the start of
every line and run again.

The sentence answering the Agenda's "subject to the training" note is in
the intro markdown, before the code: "the model was tested on 17 held-out
club and CCA meetings and gets about 0.85-0.88 of turns right, but its
vocabulary is small (553 words), so unusual words for your topic are read
as `<unk>` and ignored" -- matching N7/WP3's independently-verified 0.879
and 0.852 holdout accuracies, not new numbers.

---

## Contents table and section headers (Agenda blocks 2-9)

The notebook's own 5 top-level sections map onto 8 Agenda blocks (3 of
them -- Hands-on A, "why memory fades", and "teaching it" -- are
subsections inside sections 3 and 4, not separate top-level sections, so
their headers now carry the block number instead of renumbering
everything). New contents table (cell 0):

| section | content | agenda block(s) | minutes |
|---|---|---|---|
| 1 | Audio to text with Python | 2 (part) | ~6 |
| 2 | Where transcription struggles | 2 (part) | ~4 |
| 3 | RNNs: reading words as a sequence (+ Hands-on A, + why memory fades) | 3, 4, 5 | 15 + 10 + 7 |
| 4 | LSTMs: fixing the short memory (+ teaching it, + Hands-on B) | 6, 7, 8 | 20 + 10 + 20 |
| 5 | Meeting minutes (+ your own meeting) | 9 | 15 |

Every section/subsection header now says its block and minutes inline
(`## 3. RNNs: reading words as a sequence (Agenda block 3, 15 min)`,
`### YOUR TURN (Part A): one step of an RNN (Agenda block 4, Hands-on A,
10 min)`, etc.) -- re-read after editing, matches the table.

---

## Minutes-per-section table (section, block, minutes allowed, CPU measured)

From the full zip-only run below (`base.en`, CPU, no GPU available):

| Section | Agenda block(s) | Minutes allowed | CPU compute measured |
|---|---|---|---|
| 1 + 2 | 2 | 10 | ~58.3 s (Whisper load + two transcriptions + 4 Kahoot clips dominate; see per-cell breakdown in `plan/handoffs/HANDOFF-WP2.md`/`HANDOFF-WP3.md`) |
| 3 (RNN intro + tokenizer) | 3 | 15 | 0.0 s |
| 3 (Hands-on A) | 4 | 10 | 0.0 s (unsolved stub) / 0.0 s (solved) |
| 3 (why memory fades) | 5 | 7 | 0.3 s |
| 4 (LSTM intro + hourglass + `<eot>` + fold/unfold) | 6 | 20 | 0.6 s |
| 4 (teaching it, catching it out) | 7 | 10 | 0.4 s |
| 4 (Hands-on B + knob sweep) | 8 | 20 | 0.2 s (unsolved) / 0.5 s (solved) |
| 5 (minutes + own meeting) | 9 | 15 | 0.4 s |

Every block's actual compute is a small fraction of its allotted minutes
-- sections 1-2's ~58 s is the only block that's more than a couple of
seconds, and it's still under 10% of its 10-minute budget.

---

## Both zips, rebuilt and tested, unsolved AND solved

```
$ python notebook/make_bundle.py
wrote .../notebook/workshop_files.zip (2.8 MB, 17 files)
wrote .../notebook/facilitator_files.zip (3.7 MB, 20 files)
```

**Participants' notebook, folder with only `workshop_files.zip`, Parts A/B
unsolved (the shipped state):**
```
$ <nbclient run from a folder holding only the zip and the notebook>
... (54 cells)
=== OK in 66.4s ===
```
All 54 cells ran without error. Cell 51 (N9) printed exactly the output
quoted above.

**Same folder, Parts A and B solved** (a throwaway copy of the notebook
with the reference lines filled in -- `np.tanh(W_x @ x + W_h @ h_prev +
b)` and `f * c_prev + i * g` -- never written into the shipped notebook or
`hands_on.py`; used only to prove the solved path also runs clean):
```
=== OK in 65.6s ===
```
All 54 cells ran without error. Part A: `check (MEMORY_ON = True): OK`.
Part B: `check: OK`, knob sweep prints `35/45 (0.78)` at knob 0, matching
every earlier round's number.

**Facilitator notebook, folder with only `facilitator_files.zip`** (cells
2 and 8 only -- cells 4/6 need a real browser/Colab microphone, same
limitation as every earlier round):
```
--- running cell 2 ---
ffmpeg: found
Whisper (medium.en) preloaded and ready.
[2] 20.2s OK
--- running cell 8 ---
Press. One. Button.
(input: demo/backup_meeting.wav)
[8] 113.7s OK
```
Both ran without error.

### `compile()` check, every code cell, full joined source

```
$ python -c "
import json
nb = json.load(open('notebook/audio_workshop.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        compile(''.join(c['source']), f'cell{i}', 'exec')
print('all code cells compile cleanly,', len(nb['cells']), 'total cells')
"
all code cells compile cleanly, 54 total cells
```
No embedded-newline mistakes found this round on the first pass -- the
`compile()` habit from WP2/WP3 is now just how every new cell gets
checked before it's trusted.

### `run_tests.py` and frozen-file check

```
$ python prototype/run_tests.py
PASS  a. imports                  11 modules import  (0.3s)
PASS  b. weights                  5 members, labels discussion / decision / action_item  (0.0s)
PASS  c. shipped KPIs reproduce   dev 0.96/0.948, test 0.84/0.825, 36-turn test 0.86/0.853, sample 7/9, probes 9/12  (3.1s)
PASS  d. demo minutes             no decisions, one action item (catering by Friday, Priya)  (0.1s)
PASS  e. --text example           runs, model matches the hand labels 7/7  (0.2s)
PASS  f. hands-on                 solution: both parts OK, 35/45 at knob 0; unfilled version says 'not written yet'  (1.0s)
PASS  g. training data            leak checks pass  (2.1s)
PASS  h. scorecard                frozen scorecard runs (hash ignores line endings)  (1.0s)
PASS  i. smoke runs               5 scripts exit 0  (2.2s)
PASS  j. text clean-up            text clean-up: 31/31 clean-up checks pass  (0.5s)
PASS  k. holdout meetings         holdout 1 0.879/0.806, holdout 2 0.852/0.762 (accuracy / macro-F1)  (3.2s)
SKIP  l. audio (optional)         no --audio FILE given  (0.0s)
PASS  m. repo unchanged           git status and git diff identical before and after  (0.1s)

12 passed, 0 failed, 1 skipped in 14.0 s
```

```
$ git diff --stat main -- prototype/trained_lstm.npz training_data.py scorecard_data.py \
    holdout_data.py minutes_tests.py normalise_tests.py lstm_model.py train_lstm.py \
    meeting_minutes.py transcriber.py hands_on_reference.py testing/ prototype/hands_on.py
(no output -- nothing frozen was touched)
```

---

## `progress.md` and `debug_cheatsheet.md`

- `progress.md`: one new dated log entry, "Sun 18 Oct 2026 -- WP2-WP4: the
  participants' notebook, N1-N10", summarising all three packages in a
  few lines (additive only -- checked `git diff --stat main -- progress.md`
  shows only insertions, same pattern as every earlier round).
- `debug_cheatsheet.md`: three new rows (additive only, "new rows" rule) --
  a `NameError` when sections are skipped, the wrong zip uploaded into the
  wrong notebook, and a "Building this notebook" row logging the
  embedded-newline mistake and the `compile()`-check fix for it, since
  that genuinely happened twice while building this and is worth keeping
  on record.

---

## Handoffs tidied into `plan/handoffs/`

```
$ git mv HANDOFF-WP1.md plan/handoffs/HANDOFF-WP1.md
$ git mv HANDOFF-WP2.md plan/handoffs/HANDOFF-WP2.md
$ git mv HANDOFF-WP3.md plan/handoffs/HANDOFF-WP3.md
```
(`HANDOFF-WP4.md`, this file, is written directly to `plan/handoffs/`.)
Every reference to `HANDOFF-WP1.md`/`2`/`3` across the repo
(`demo/ANSWER_KEY.md`, `demo/ATTEMPTS.md`, `demo/FACILITATOR.md`,
`demo/sure_exchanges.txt`, `notebook/make_noisy_clips.py`, `progress.md`,
`plan/PROMPTS.md`, `plan/WORK_PACKAGES.md`) updated to the new path.
Checked with `grep -rn "HANDOFF-WP[0-9]\.md"` across the repo: every match
now starts with `plan/handoffs/`.

---

## Files changed

```
 debug_cheatsheet.md            | +3 rows (additive)
 demo/ANSWER_KEY.md             | path fix only
 demo/ATTEMPTS.md               | path fixes only
 demo/FACILITATOR.md            | path fixes only
 demo/sure_exchanges.txt        | path fix only
 notebook/audio_workshop.ipynb  | N9 cells, contents table, section/subsection headers, WP3 carry-over
 notebook/make_noisy_clips.py   | path fix only (a comment)
 plan/PROMPTS.md                | path fix only
 plan/WORK_PACKAGES.md          | path fix only
 progress.md                    | +1 log entry (additive)
 HANDOFF-WP1.md -> plan/handoffs/HANDOFF-WP1.md   (git mv)
 HANDOFF-WP2.md -> plan/handoffs/HANDOFF-WP2.md   (git mv)
 HANDOFF-WP3.md -> plan/handoffs/HANDOFF-WP3.md   (git mv)
 plan/handoffs/HANDOFF-WP4.md   | new (this file)
```

---

## Not done / open

- Same GPU/Colab/live-mic limitations as every earlier round -- no GPU or
  Colab available from this session, so the T4 preload timing and the
  actual browser-mic recorder remain unverified here (noted the same way
  in every prior handoff).
- Nothing else in WP4's deliverable list, or in
  `plan/WORK_PACKAGES.md` as a whole, is outstanding. This was the last
  work package.

---

## Round 2 (final review verdict: CHANGES on `cecf3da`, no blockers, whole-branch pass otherwise clean)

Reviewer's findings: `git show claude/review-workshop-gaps-53e548:reviews/WP4.md`.
Both findings and all nits applied below, independently re-verified.
Nothing committed -- same protocol as every round.

| # | Finding | Fix applied | Verified |
|---|---|---|---|
| 1 (should-fix) | A leftover, unlabelled own-meeting cell (the pre-WP4 `MY_MEETING = '''Sam: Should we move...'''` cell) was still sitting right after the new N9 cell, same variable name, no heading -- a participant could easily edit the wrong one | Deleted the leftover cell entirely (53 cells now, was 54) | Confirmed by index: the only `MY_MEETING` cell left is the N9 one |
| 2 (should-fix) | Labelling only some lines, a line with no `Name:`, or empty text all raised an uncaught `ValueError` with a stack trace and the temp-file's path | Wrapped `load_text(...)` in `try/except ValueError`, printing the message with the temp path stripped plus a hint, then skipping the rest of the cell cleanly (an `if my_text is not None:` block, no `SystemExit`) | All three of the reviewer's exact cases re-tested (below): clean one-line messages, no traceback |
| 3 (nit) | "553 words" was a typed constant | `vocab_size = len(load_trained_model()["vocab"])`, printed from that variable | Re-ran: still prints `553`, now computed live |
| 4 (nit) | "17 held-out club and CCA meetings" is loose -- several aren't clubs or CCAs | Reworded to "17 held-out meetings (clubs, projects, group work)" | Re-read |
| 5 (nit) | Unknown-word list includes names, which could read as if names matter to the model | Added: "Names show up as unseen words too, for the same reason -- the model never actually reads them (they are dropped before it sees the text), so this is expected and harmless." | Re-read |

**A real bug found while applying finding 2, not in the review:** the
first attempt stripped the temp-file path with `str(e).split(": ", 1)[-1]`,
which works for two of the three error messages but garbles the
"has no 'Speaker: text' lines" one -- that message's first `": "` is
inside the quoted phrase `'Speaker: text'`, not after the path, so the
split cut it at the wrong point (`"Couldn't read that meeting: text'
lines"`). Fixed by replacing the known `my_meeting_path` substring
directly instead of splitting on a delimiter that can appear elsewhere in
the message, then stripping leftover punctuation.

### The three error cases, re-tested after the fix

```
$ <exec cell 51's real source with MY_MEETING swapped for each case, prerequisites imported>

=== g_partly_labelled (2 of 3 lines labelled) ===
Couldn't read that meeting: label every line (act / dec / dis) or none of them; 2 of 3 lines have one
Hint: label every line or none of them, and write each line as 'Name: text'
(something after the colon).
(no exception)

=== h_no_speaker_name (a line with no "Name:") ===
Couldn't read that meeting: line 2: expected 'Speaker: text', got "dis  Let's start the meeting."
Hint: label every line or none of them, and write each line as 'Name: text'
(something after the colon).
(no exception)

=== i_empty (blank text) ===
Couldn't read that meeting: has no 'Speaker: text' lines
Hint: label every line or none of them, and write each line as 'Name: text'
(something after the colon).
(no exception)
```
All three now print one clean, readable line each -- no traceback, no
temp-file path, no mangled message.

### Full notebook, empty folder with only the rebuilt zip -- unsolved and solved

```
$ python notebook/make_bundle.py
wrote .../notebook/workshop_files.zip (2.8 MB, 17 files)
wrote .../notebook/facilitator_files.zip (3.7 MB, 20 files)

$ <nbclient run from a folder holding only workshop_files.zip + the notebook, Parts A/B unsolved>
... (53 cells)
=== OK in 70.9s ===
```
Cell 51's output, unsolved notebook, default (labelled) example:
```
discussion   (you)   discussion   (model)   Sam: Let's go over what's left before the trip.
decision     (you)   discussion   (model)   Lee: The venue is confirmed for Friday.  <- X
discussion   (you)   discussion   (model)   Sam: Should we also book the shuttle?
action_item  (you)   action_item  (model)   Sam: Lee, can you call the shuttle company?
action_item  (you)   action_item  (model)   Lee: Sure.

# Meeting Minutes
...
Words the model never saw in training (read as <unk>): lee, shuttle, what's
The model knows 553 words; anything else is read as <unk>.

Pick one line it got wrong. Which EARLIER line should it have remembered?
```

```
$ <same folder, Parts A and B solved inline (throwaway copy, hands_on.py untouched)>
... (53 cells)
=== OK in 75.2s ===
```
Part A: `check (MEMORY_ON = True): OK`. Part B: `check: OK`, knob sweep
`10/45 (0.22)` at -6, `35/45 (0.78)` at 0, `22/45 (0.49)` at +6 -- same
numbers as every earlier round.

### `compile()` check, every code cell, full joined source

```
$ python -c "
import json
nb = json.load(open('notebook/audio_workshop.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        compile(''.join(c['source']), f'cell{i}', 'exec')
print('all code cells compile cleanly,', len(nb['cells']), 'total cells')
"
all code cells compile cleanly, 53 total cells
```

### `run_tests.py` and frozen-file check

```
$ python prototype/run_tests.py
PASS  a. imports                  11 modules import  (0.9s)
PASS  b. weights                  5 members, labels discussion / decision / action_item  (0.1s)
PASS  c. shipped KPIs reproduce   dev 0.96/0.948, test 0.84/0.825, 36-turn test 0.86/0.853, sample 7/9, probes 9/12  (20.4s)
PASS  d. demo minutes             no decisions, one action item (catering by Friday, Priya)  (0.4s)
PASS  e. --text example           runs, model matches the hand labels 7/7  (0.7s)
PASS  f. hands-on                 solution: both parts OK, 35/45 at knob 0; unfilled version says 'not written yet'  (2.4s)
PASS  g. training data            leak checks pass  (3.4s)
PASS  h. scorecard                frozen scorecard runs (hash ignores line endings)  (1.3s)
PASS  i. smoke runs               5 scripts exit 0  (2.9s)
PASS  j. text clean-up            text clean-up: 31/31 clean-up checks pass  (0.6s)
PASS  k. holdout meetings         holdout 1 0.879/0.806, holdout 2 0.852/0.762 (accuracy / macro-F1)  (3.5s)
SKIP  l. audio (optional)         no --audio FILE given  (0.0s)
PASS  m. repo unchanged           git status and git diff identical before and after  (0.1s)

12 passed, 0 failed, 1 skipped in 36.8 s
```

```
$ git diff --stat main -- prototype/trained_lstm.npz training_data.py scorecard_data.py \
    holdout_data.py minutes_tests.py normalise_tests.py lstm_model.py train_lstm.py \
    meeting_minutes.py transcriber.py hands_on_reference.py testing/ prototype/hands_on.py
(no output -- nothing frozen was touched)
```

### Files changed this round

```
 notebook/audio_workshop.ipynb | deleted the leftover cell (54 -> 53 cells); cell 50/51 reworded and hardened
```
No other files changed this round.
