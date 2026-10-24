# HANDOFF-WP6: final fixes (F1-F16, full-dry-run audit)

**Branch:** `claude/build-workshop-gaps-756dec`, merged up to `main` at
`1b2dfea` (`git merge main`, fast-forward -- `main` only added `reviews/`).
**Nothing committed this round** -- same protocol. Scope: `demo/` and
`notebook/` only; `prototype/`, the model and the data stay frozen.

**Source:** `git show claude/review-workshop-gaps-53e548:reviews/WP6-audit.md`
(a full dry run of all 10 blocks, facilitator + participant, by the
reviewer). All 16 findings (F1-F16) implemented, in the priority order the
orchestrator gave. Every number below was re-run in this session, not
copied from the audit -- several didn't match my first attempt and are
logged as real bugs found and fixed (see F1's own cell and the "add a
line" cell below).

---

## Per-finding table

| ID | Severity | What changed | Verified |
|---|---|---|---|
| F1 | blocker | Reworded cell 30 (break-the-memory instructions) to require *deleting* `+ W_h @ h_prev`, not just the flag; added a ready-made "amnesia" cell (31) so every participant sees the contrast even if stuck | Re-ran both paths myself: correct line + flag only -> `not quite`; term deleted -> `OK`, gap `2.85 -> 0.00` from word 2. See screenshot 04. |
| F2a | blocker | New "watch the gates" cell (37): layer 1, member 0, replayed by hand from the shipped weights over the hook's "Can you book the room? / Sure.", bar chart + printed table, caveat stated | Re-derived independently, twice (standalone script, then inside the full notebook run) -- identical numbers both times. See screenshot 03. |
| F2b | blocker | Section 4 retitled "an eraser you can set"; "fixing the short memory" / "what actually fixes it" language removed from cells 31 and 34 | Re-read both cells |
| F2c | blocker | Added the sigmoid link to cell 53: knob 0 = sigmoid(0) = 0.5 (the 0.49 curve), knob +3 = sigmoid(3) = 0.953 (the 0.95 curve) | `sigmoid(0)=0.5`, `sigmoid(3)=0.9526`, computed directly |
| F3 | blocker | New `demo/RUNSHEET.md`: per-block cells/output/timing/failure table, participant onboarding, Kahoot logistics, one pointer line per block | New file, see below |
| F4 | should-fix | Own-meeting default replaced with the hook's own exchange (a real knife-edge case, not an invented one); confidence printed beside every label; "add a line before it" cell added, with caveat | Re-derived the knife-edge (51%) and the flip (56%) myself -- matches the audit. The default example now has a real answer to "which earlier line". |
| F5a-d | should-fix | `hook_demo.ipynb`: the backup is now played to the room (`display(Audio(...))`) before the button; setup prints `GPU: yes/no` and switches to `small.en` automatically with no GPU; a new text-only fallback cell (no Whisper); minutes rendered as large HTML (`big_minutes()`) | All four re-run from `facilitator_files.zip` alone. See screenshot 01. |
| F6 | should-fix | Collapsed "Stuck?" notes under both YOUR-TURN cells, each symptom verified by actually making the mistake | Ran 5 real wrong-line variants (3 for Part A, 2 for Part B); every symptom in the notes matches a real run, including inside the full "mistake" notebook run below. |
| F7 | should-fix | Retention graph now overlays Part A's measured toy fade (~58%) as a third dotted curve, computed in the cell, not typed in; legend moved to the one empty corner; two sentences added on why each knob extreme hurts, and why +6's gap grows | Re-derived the toy fade (0.578, matches "58%") independently with the same toy setup, inside the notebook cell itself. See screenshot 02. |
| F8 | should-fix | New "Sure. test" cell (47) tied explicitly to the hook: "Sure." alone, after the request, after the proposal, with label confidences, and an honest statement that the third answer is wrong | Re-derived all three cases: 0.41/0.32/0.27 alone; 0.80 after the request; 0.985 **discussion** (wrong) after the proposal. |
| F9 | should-fix | One legend line under cell 27 explaining `x`, `h_prev`, `W_x`, `W_h`, `b`; "folded together" -> "looks at four words at a time" | Re-read both cells |
| F10 | should-fix | Boxed restart note under the title; "Installing Whisper, about a minute..." printed before both pip installs; cells 31, 37, 39, 40 now import what they use (`numpy`, `matplotlib.pyplot`, `tokenize_transcript`) | Re-ran the exact restart/skip cases the audit found (below) -- all now either work standalone or point at the boxed note. See screenshot 05. |
| F11 | should-fix | `FACILITATOR.md` restructured: a 6-step running order and a failure table at the top, cell references fixed to actual indexes (2/4/6/8/10, not section numbers), the Rule H history moved to a "Provenance" section at the bottom | Re-read the full file |
| F12 | polish | "Whisper reads its spectrogram the same way, left to right" (overstated -- Whisper is a different kind of model) reworded in cells 7 and 25 | Re-read both cells |
| F13 | polish | New closing cell (63): the one-sentence journey, three honest limits (plain-fact decisions, unseen words, a bare "Sure." after a proposal), the take-home pointer | Re-read |
| F14 | polish | The facilitator-addressed sentence ("check it on the day's Colab...") moved out of cell 23 into `demo/RUNSHEET.md`; the answer-key table itself kept in the notebook | Re-read both files |
| F15 | polish | Noted in `demo/RUNSHEET.md` and `demo/FACILITATOR.md`: both backup voices are female, say the names aloud; the live-skit checklist item stays unticked until rehearsed | Re-read |
| F16 | polish | "(about Ns on CPU)" timing notices added to cells 4, 12, 14, 24 | Re-read |

---

## F1: the amnesia contrast, independently re-derived

```
$ python -c "<correct line, MEMORY_ON=False, flag only, no term deleted>"
check (MEMORY_ON = False): not quite  got [-0.9964  0.5838], expected [-0.9335 -0.4899]
```
Confirms the audit's bug exactly -- the old instruction really did fail.

```
$ python -c "<amnesia version: return np.tanh(W_x @ x + b), no + W_h @ h_prev>"
check (MEMORY_ON = False): OK  got [-0.9335 -0.4899], expected [-0.9335 -0.4899]
gap    2.85  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
```

## F2a: gate values, re-derived from the shipped weights (lstm_model's loader, unmodified)

```python
layer1 = trained["model"].layer1   # from load_trained_model(), no change to lstm_model.py
# ... replay f/i/o by hand over "Can you book the room? / Sure." (tokenize_transcript) ...
```
```
can        eraser=0.41  pencil=0.44  voice=0.46
you        eraser=0.77  pencil=0.74  voice=0.65
book       eraser=0.63  pencil=0.65  voice=0.46
the        eraser=0.83  pencil=0.26  voice=0.63
room       eraser=0.57  pencil=0.32  voice=0.42
?          eraser=0.58  pencil=0.31  voice=0.75
<eot>      eraser=0.36  pencil=0.29  voice=0.74  <- speaker changed
sure       eraser=0.71  pencil=0.42  voice=0.57
```
Reproduced identically on a second, independent run (standalone script,
then inside the full notebook execution below) -- same numbers both times.

## F4: the knife-edge and the flip, independently re-derived

```
$ python -c "<run_pipeline + label_probabilities on the 5-line hook example>"
Mei Shall we move it to Friday? discussion 0.999
Ravi Sure.                       discussion 0.508  decision 0.487   <- the knife-edge (gold: decision)
Mei Can you book the room?       action_item 0.981
Ravi Sure.                       action_item 0.988
Mei Great, thanks.                discussion 0.970
```
```
$ python -c "<same, with 'Mei: Let's plan the showcase.' prepended>"
Ravi Sure. (first)  -> decision 0.562  discussion 0.434   <- now correct
```
Matches the audit's 0.51/0.49 and 0.56/0.43 to within rounding.

**A real bug found while wiring this into the notebook, not in the audit:**
the first version of the "add a line" cell built `with_context` from
`"Mei: Let's plan the showcase.\n" + MY_MEETING.strip()` -- the *raw*,
still-labelled text (`"dis  Mei: Shall we move it to Friday?"` etc.), so
`run_pipeline` parsed `"dis  Mei"` as a literal speaker name. Caught by
testing the cell standalone before trusting it: the printed speaker names
came out wrong (`dec  Ravi`, not `Ravi`). Fixed to prepend the new line to
`my_text` (the already label-stripped string `load_text()` returns), not
the raw `MY_MEETING`. Re-verified clean after the fix (above).

## F6: every "Stuck?" symptom, verified by making the mistake

```
W_h * h_prev (× instead of @):        got [[-0.9899 -0.7823] [-0.824  0.1517]]   (a 2x2 grid)
no tanh:                               got [-3.1549  0.6682]                     (outside -1..1)
no + b:                                got [-0.9874  0.5559]                     (close, not quite)
Part B: f + c_prev + i + g (+ not *): got [ 2.2523  1.1199  1.7386 -0.8972]      (bigger numbers)
Part B: f * c_prev + i (missing g):   got [1.2686 0.6611 0.6255 0.7188]          (close, not quite)
```
All five match what's written in the two "Stuck?" notes, and all five were
also reproduced inside the full "mistake" notebook run (below), where the
symptom text sits right next to the real output a participant would see.

## F7: the toy fade, measured in the cell

```
$ python -c "<replay Part A's exact toy setup, seed 1, correct formula>"
kept 0.578119477070912   (58%)
gaps [2.85, 1.85, 0.69, 0.31, 0.16, 0.09, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01]
```
Matches Part A's own printed "about 58% kept per step" and the audit's
number. Plotted as a third dotted curve (screenshot 02).

## F8: the "Sure." test, re-derived

```
"Sure." alone                         action_item 0.405  discussion 0.322  decision 0.273
after "Can you book the room?"        action_item 0.80
after "Shall we move it to Friday?"   discussion 0.985  decision 0.014
```
Matches the audit exactly.

## F10: restart/skip cases, re-tested after the fix

```
cell 28 alone (after a fresh kernel, setup not run)  -> ModuleNotFoundError: hands_on   (expected -- the boxed note tells them to run cell 2 first)
cell 31 (retention graph) alone, after cell 2 only   -> now imports numpy/matplotlib itself -- no longer NameErrors
cell 37 (gates) alone, after cell 2 + 35 only        -> now imports tokenize_transcript itself for this part; still needs `trained`/`vocab` from cell 35 (unavoidable -- they're real, computed values, not importable)
cell 39/40 (<eot>, fold/unfold) alone, after cell 2  -> now import what they use; no longer NameErrors
```
The cases that genuinely need a prior cell's *computed* value (`transcript`,
`turns`, `trained`, `vocab`) can't be fixed by adding imports -- those are
covered by the new boxed note under the title instead, which is the
honest fix (can't fabricate a real Whisper transcript without running
Whisper).

---

## Full verification

### `demo/RUNSHEET.md`

New file -- per-block table (cells, what to say, expected output, CPU
time, what to do if it fails), participant onboarding (how they get the
private repo's notebook and zip, the T4 setting, when setup runs), and
Kahoot logistics for block 2 (which clips play when, the vote happens in
Kahoot itself, not the notebook). Checked against `plan/SPEC.md`'s
timetable: block minutes sum to 107 (sections) + 8 (hook) + 5 (wrap-up) =
120, matching the Agenda's 120-minute total.

### Both zips, rebuilt

```
$ python notebook/make_bundle.py
wrote .../notebook/workshop_files.zip (2.8 MB, 17 files)
wrote .../notebook/facilitator_files.zip (3.7 MB, 20 files)
```

### `compile()`, every cell, both notebooks

```
$ python -c "<compile every code cell's full joined source, both notebooks>"
notebook/audio_workshop.ipynb errs 0 bad 0 cells 65
demo/hook_demo.ipynb errs 0 bad 0 cells 11
```
("bad" = the embedded-newline mistake from earlier rounds -- checked for
and fixed 4 more instances while writing this round's cells, each caught
immediately by this check before being trusted.)

### Participants' notebook, `workshop_files.zip` alone, empty folders -- three runs

**Unsolved (shipped state):**
```
$ <nbclient, empty folder + zip + notebook>
... (65 cells)
=== OK in 104.4s ===
```

**Solved** (throwaway copy, reference lines filled in, `hands_on.py` never touched):
```
... (65 cells)
=== OK in 96.3s ===
```

**Two typical mistakes** (`W_h * h_prev` for Part A, `f * c_prev + i` -- missing `g` -- for Part B):
```
... (65 cells)
=== OK in 95.4s ===
```
All three: 0 errors. The mistake run's cells 28 and 50 print exactly the
symptoms quoted in the "Stuck?" notes (F6, above) -- not just consistent
with them, the actual output next to the actual advice.

**N9's three bad-input cases, re-tested with the final cell's real source:**
```
partly labelled (2 of 3 lines)  -> "Couldn't read that meeting: label every line (act / dec / dis) or none of them; 2 of 3 lines have one"
no speaker name                 -> "Couldn't read that meeting: line 2: expected 'Speaker: text', got ..."
empty                           -> "Couldn't read that meeting: has no 'Speaker: text' lines"
```
All three: one clean line, no traceback, no temp path -- unchanged from
WP4/WP5, re-confirmed against the current cell.

### `hook_demo.ipynb`, `facilitator_files.zip` alone, empty folder -- setup, backup, new fallback

```
$ <exec cells 2, 8, then 10 (fallback) -- cells 4/6 skipped, need a real Colab mic>
[2]  15.9s  GPU: no -- expect the button press to take about 1 minute (small.en, no GPU here)
             Whisper (small.en) preloaded and ready.
[8]  47.0s  Audio player (0:53) + big HTML minutes: Hold the showcase... / Get black t -shirts...
             Book the hall by Friday -- Ravi / I'll send the order form... -- Mei
[10]  0.4s  Same minutes, instant, no Whisper at all (demo/script.txt)

=== OK in 72.3s ===
```
Both the audio player and the large HTML minutes confirmed present in the
actual HTML output (extracted and inspected directly, not assumed from
the Python source) -- see screenshot 01.

### `run_tests.py` and frozen-file check

```
$ python prototype/run_tests.py
PASS  a. imports                  11 modules import  (0.8s)
PASS  b. weights                  5 members, labels discussion / decision / action_item  (0.1s)
PASS  c. shipped KPIs reproduce   dev 0.96/0.948, test 0.84/0.825, 36-turn test 0.86/0.853, sample 7/9, probes 9/12  (7.4s)
PASS  d. demo minutes             no decisions, one action item (catering by Friday, Priya)  (0.4s)
PASS  e. --text example           runs, model matches the hand labels 7/7  (0.6s)
PASS  f. hands-on                 solution: both parts OK, 35/45 at knob 0; unfilled version says 'not written yet'  (2.0s)
PASS  g. training data            leak checks pass  (5.0s)
PASS  h. scorecard                frozen scorecard runs (hash ignores line endings)  (2.1s)
PASS  i. smoke runs               5 scripts exit 0  (4.6s)
PASS  j. text clean-up            text clean-up: 31/31 clean-up checks pass  (0.9s)
PASS  k. holdout meetings         holdout 1 0.879/0.806, holdout 2 0.852/0.762 (accuracy / macro-F1)  (7.1s)
SKIP  l. audio (optional)         no --audio FILE given  (0.0s)
PASS  m. repo unchanged           git status and git diff identical before and after  (0.1s)

12 passed, 0 failed, 1 skipped in 31.4 s
```

```
$ git diff --stat main -- prototype/trained_lstm.npz training_data.py scorecard_data.py \
    holdout_data.py minutes_tests.py normalise_tests.py lstm_model.py train_lstm.py \
    meeting_minutes.py transcriber.py hands_on_reference.py testing/ prototype/hands_on.py
(no output -- nothing frozen was touched)
```

### Screenshots (`plan/handoffs/wp6/`, 5 files)

```
01_hook_backup_audio_and_big_minutes.jpg   F5: the Audio player above the backup's
                                            big-font HTML minutes, real rendered output
02_retention_graph_3curves.png             F7: the toy-RNN dotted curve added, legend
                                            clear of all three curves
03_watch_the_gates.png                     F2a: the gate bar chart, <eot> visible
04_F1_amnesia_contrast_and_stuck_note.jpg  F1: the normal fade next to the amnesia
                                            version's stark 2.85 -> 0.00, the collapsed
                                            "Stuck?" note visible above it
05_title_restart_note_contents_table.jpg   F10: the boxed restart note and the updated
                                            contents table (F2b's "an eraser you can set")
```
(Within the ~8-file budget; fewer were needed since each one documents
more than one finding. Two more screenshot attempts -- the "Sure." test
cell and the own-meeting output -- hit a rendering stall in the browser
pane on this large page; both were independently confirmed via
`document.body.innerText` extraction instead, quoted in F8/F4 above, so
nothing here is unverified, just not pictured.)

---

## Round 2

**Reviewer findings:** 1 blocker + 5 should-fix + 6 polish (V1-V12), all
fixed per the orchestrator's explicit per-finding instructions. Nothing
committed this round -- same protocol.

### Per-finding table

| ID | Severity | What changed | Verified |
|---|---|---|---|
| V1 | blocker | `demo/hook_demo.ipynb`'s backup-WAV cell split in two: cell 8 now only plays the audio (`display(Audio(...))`), a new cell 9 (markdown, "4b. Press the button on the backup") separates it from a new cell 10, which presses the button and shows the minutes. Cell 7's markdown explains the two-cell order. Fallback cells renumbered 11/12. | Ran cells 2, 8, 10, 12 (new order) from `facilitator_files.zip` in an empty folder, 0 errors; `nbconvert`-rendered HTML confirms the Audio player sits alone under "4.", the minutes under a separate "4b." heading -- screenshot `08_hook_split_4_4b_minutes.jpg`. |
| V2 | should-fix | Cell 30 (break-the-memory instructions) now has an explicit sentence telling participants to ignore the misleading half-instruction `hands_on.py` prints ("Now set MEMORY_ON = False and run again."), and to do both steps -- the flag *and* deleting the term. | Re-read the cell; the sentence sits directly under the printed line it's correcting. |
| V3 | should-fix | Retention-graph legend (cell 33) moved entirely outside the axes (`bbox_to_anchor=(1.02, 1.0), loc="upper left"`), replacing the round-1 `loc="lower right"` fix that still covered the 0.95 curve's tail and its 9% label. | Rendered the actual PNG from the executed notebook (not a browser screenshot) -- legend box sits fully in the figure's right margin, zero overlap with any of the three curves. Screenshot `06_retention_legend_outside_axes.png`. |
| V4 | should-fix | Cell 62 (markdown after the "add a line" cell) reworded from a vaguer "confident" framing to the actual numbers: "a (still narrow) correct `decision`, 56% against 44%." | Re-derived standalone: first "Sure." -> `decision` 56%; matches the wording exactly. |
| V5 | should-fix | Cell 26 now imports `numpy` itself (it was relying on an earlier cell's import). `demo/RUNSHEET.md`'s block-3 row rewritten to state only what's true of cell 26 specifically (self-contained, no dependency on run order), instead of the blanket "this cell now imports what it needs" claim the reviewer flagged as applying to cell 26 only, not a general guarantee. | `compile()` on cell 26 alone (no earlier cells run) succeeds; re-read every RUNSHEET row claiming self-containment (26, 33) against the actual cell source -- 33 already imports both `numpy` and `matplotlib.pyplot` itself, so its claim was already true and left unchanged; 26's was the only false one, now fixed. |
| V6 | should-fix | `demo/RUNSHEET.md` onboarding step 1 reworded to resolve the contradiction: share the notebook file ahead of time, but not the zip until after the hook (exact split the orchestrator specified). Added "(+ model download on first use; not measured)" to the Setup and cell-4 timing figures, since the shipped ~20s/~1-minute numbers are CPU timings with the Whisper model already cached. | Re-read the corrected section; the two instructions ("share ahead of time" / "not before the hook") no longer conflict -- the notebook file is what's shared early, the zip is what's withheld. |
| V7 | polish | Gate-plot (cell 37) legend moved outside the axes, same pattern as V3 (`bbox_to_anchor=(1.02, 1.0)`), replacing the default `plt.legend()` placement that sat over the "you" bars. Checked cell 36's and the RUNSHEET's narration against the actual numbers: no unsupported "pencil spikes on content words" claim exists anywhere in the current notebook or RUNSHEET (grepped both; the only gate-related claims found are cell 36's accurate ensemble/layer caveat and the eraser dropping at `<eot>`, which the numbers support). | Re-derived all 8 tokens' gate values standalone (`can`...`sure`) -- eraser dips to 0.36 at `<eot>` vs. 0.57-0.83 elsewhere (supported); pencil is highest on "you" (0.74), not a content word, and lowest on "room" (0.32) -- confirms no "content word" pattern exists, consistent with there being no such claim left to fix. Screenshot `07_gate_plot_legend_clear.png`. |
| V8 | polish | Cell 34 (LSTM section intro): "with no control over how much" reworded to "with one fixed fade rate for every word" -- the original implied the RNN had *no* fade control at all, when the real point is that the rate is fixed/uniform rather than adjustable per word. | Re-read |
| V9 | polish | Cell 63 (wrap-up): removed the unsupported "including most names" clause from limit #2 (unseen words), and replaced the take-home's implicit promise of a retrain-from-zip with an honest note that the training code is with the facilitators, not in `workshop_files.zip`. | Confirmed `train_lstm.py` is not among the 17 files `make_bundle.py` puts in `workshop_files.zip` (`python notebook/make_bundle.py` file list, below). |
| V10 | should-fix | The "add a line" cell (61) now labels which "Sure." is which when printing results ("first 'Sure.' (after the proposal)" / "second 'Sure.' (after the request)"), instead of an ambiguous repeated line. | Re-derived standalone: `Ravi: first "Sure." (after the proposal) -> decision (56%)`, `Ravi: second "Sure." (after the request) -> action_item (99%)` -- matches the full-notebook run's cell 61 output exactly. |
| V11 | polish | Cell 53 ("All three settings, compared") gained one caveat sentence: the knob sweep runs on one ensemble member only (member 0, same as `hands_on.py`), while the "Sure." test (cell 47) averages all five -- the two cells aren't measuring quite the same model. | Re-read `lstm_model.label_probabilities` (averages the 5-member ensemble) against `hands_on.py`'s `_member_predict` (member 0 only) -- confirms the distinction the new sentence states. |
| V12 | polish | `hook_demo.ipynb` cell 2's stale comment (referring to the pre-split cell numbering) rewritten to describe the current split (cells 6, 8, 10). | Re-read, numbers match the post-V1 cell layout. |

### Full verification

**`compile()`, every cell, both notebooks:**
```
notebook/audio_workshop.ipynb cells: 65 errs: 0
demo/hook_demo.ipynb cells: 13 errs: 0
```

**Both zips, rebuilt:**
```
$ python notebook/make_bundle.py
wrote .../notebook/workshop_files.zip (2.8 MB, 17 files)
wrote .../notebook/facilitator_files.zip (3.7 MB, 20 files)
```

**`run_tests.py` and the frozen-file check:**
```
$ python prototype/run_tests.py
PASS  a. imports                  11 modules import  (0.8s)
PASS  b. weights                  5 members, labels discussion / decision / action_item  (0.1s)
PASS  c. shipped KPIs reproduce   dev 0.96/0.948, test 0.84/0.825, 36-turn test 0.86/0.853, sample 7/9, probes 9/12  (7.8s)
PASS  d. demo minutes             no decisions, one action item (catering by Friday, Priya)  (0.4s)
PASS  e. --text example           runs, model matches the hand labels 7/7  (0.6s)
PASS  f. hands-on                 solution: both parts OK, 35/45 at knob 0; unfilled version says 'not written yet'  (2.0s)
PASS  g. training data            leak checks pass  (4.9s)
PASS  h. scorecard                frozen scorecard runs (hash ignores line endings)  (2.4s)
PASS  i. smoke runs               5 scripts exit 0  (4.5s)
PASS  j. text clean-up            text clean-up: 31/31 clean-up checks pass  (0.9s)
PASS  k. holdout meetings         holdout 1 0.879/0.806, holdout 2 0.852/0.762 (accuracy / macro-F1)  (6.8s)
SKIP  l. audio (optional)         no --audio FILE given  (0.0s)
PASS  m. repo unchanged           git status and git diff identical before and after  (0.2s)

12 passed, 0 failed, 1 skipped in 31.5 s
```
```
$ git diff --stat main -- prototype/ data/
(no output -- nothing frozen was touched)
```

**Participants' notebook, `workshop_files.zip` alone, empty folder -- unsolved (shipped state):**
```
$ <exec every code cell from audio_workshop.ipynb against the unzipped folder>
[cell 2] OK: ready
[cell 4] OK: base.en transcription, Q3 budget review text
...
[cell 28] OK: check (MEMORY_ON = True): not written yet (it returned None)
[cell 33] OK: 0.49^47 = 0.0000%   0.95^47 = 9.0%   toy RNN: r=0.578
[cell 37] OK: can eraser=0.41 pencil=0.44 voice=0.46 / you eraser=0.77 pencil=0.74 voice=0.65 / ...
[cell 50] OK: check: not written yet (it returned None)
[cell 61] OK: Ravi: first "Sure." (after the proposal) -> decision (56%)
             Ravi: second "Sure." (after the request) -> action_item (99%)
DONE unsolved errors: 0
```
All numbers match the standalone re-derivations above exactly, including
the gate values (V7) and the "Sure." labels (V4/V10).

**Solved** (throwaway copy, `hands_on.py`'s own reference lines filled in
at cells 28/50, `hands_on.py` itself never touched):
```
[cell 28] OK: check (MEMORY_ON = True): OK  got [-0.9964  0.5838], expected [-0.9964  0.5838]
[cell 50] OK: check: OK  got [ 0.5263  0.5426  0.402  -0.7362], expected [ 0.5263  0.5426  0.402  -0.7362]
DONE solved errors: 0
```

**Deliberate mistake** (`W_x @ x + W_h @ h_prev` missing `+ b` for Part A,
`f * c_prev + i` missing `* g` for Part B -- plausible, not contrived,
errors):
```
[cell 28] OK: check (MEMORY_ON = True): not quite  got [-0.9874  0.5559], expected [-0.9964  0.5838]
[cell 50] OK: check: not quite  got [1.2686 0.6611 0.6255 0.7188], expected [ 0.5263  0.5426  0.402  -0.7362]
DONE mistake errors: 0
```
All three variants: 0 execution errors. The mistake run's cells 28/50
show "not quite" with a concrete got/expected mismatch rather than a
traceback -- the YOUR-TURN cells fail the way they're supposed to, not
by crashing.

**`hook_demo.ipynb`, `facilitator_files.zip` alone, empty folder, new
cell order (2, 8, 10, 12 -- cells 4/6 need a real Colab mic, skipped):**
```
[cell 2]  GPU: no -- small.en preloaded and ready
[cell 8]  Audio player displayed (backup_meeting.wav, 0:53)
[cell 10] Press. One. Button. -> rendered minutes (big_minutes HTML):
          Decisions: Hold the showcase on the 21st (agreed); Get black t -shirts (agreed)
          Action Items: Book the hall by Friday -- Ravi; I'll send the order form... -- Mei
[cell 12] Press. One. Button. (text-only fallback, demo/script.txt) -> same minutes, no Whisper
```
Rendered via `nbconvert --to html` from an executed copy (live-mic cells
4/6 replaced with a no-op placeholder so nbconvert doesn't hang on them)
and visually confirmed in-browser -- the Audio player sits under its own
"4." heading, the button press and minutes under a separate "4b." heading,
confirming V1's split actually works end to end, not just that the code
compiles. Screenshot `08_hook_split_4_4b_minutes.jpg`.

### Screenshots added this round (`plan/handoffs/wp6/`, 3 new files)

```
06_retention_legend_outside_axes.png  V3: legend fully outside the axes,
                                       no overlap with any of the three curves
07_gate_plot_legend_clear.png         V7: gate-plot legend clear of all bars
08_hook_split_4_4b_minutes.jpg        V1: the backup Audio player under "4.",
                                       the pressed-button minutes under a
                                       separate "4b." heading
```

---

## Round 3

**Reviewer findings:** re-verification of `1101432` found 11 of 12 Round-2
items fixed; one doc fix (R1) plus two small touches (R2, R3) remained.
All three fixed. Nothing committed this round -- same protocol.

### Per-finding table

| ID | Severity | What changed | Verified |
|---|---|---|---|
| R1 | should-fix (must fix before the day) | `demo/FACILITATOR.md` updated throughout for the Round-2 cell split: cell 8 now described as "plays the backup WAV" only, cell 10 as "presses the button on it", cell 12 (was 10) as the text-only fallback. Fixed in the intro paragraph, the six-step running order, the failure table (all 3 rows that referenced the old numbers), the "What to say" step 4, and the "Running it: Colab" cell-by-cell list. | Compared every "cell N" reference in `demo/FACILITATOR.md` against `hook_demo.ipynb`'s actual cell indexes (2=Setup, 4=recorder, 6=live button, 8=backup play, 10=backup press, 12=fallback), read directly from the notebook JSON -- all now match. Grepped `demo/` and `notebook/` for every other `hook_demo.ipynb` cell reference (`demo/RUNSHEET.md` line 116, `notebook/make_bundle.py` comments) -- none reference specific cell numbers that needed changing. `demo/RUNSHEET.md`'s own cell numbers (26, 33, 37, etc.) are a different notebook (`audio_workshop.ipynb`) and were already correct. |
| R2 | polish | Cell 62: "56% against 44%" corrected to "56% against 43%", matching the model's actual `discussion` probability (0.434, not 0.44) for the first "Sure." after the added context line. | Re-derived from the full notebook run, not just the cell's own math: `{'discussion': 0.434, 'decision': 0.5619, 'action_item': 0.0041}` -- 43.4% rounds to 43%, not 44%. |
| R3 | polish | `plt.figure(figsize=(7, 4))` added at the top of cell 33 (retention graph), so it always draws on its own fresh figure regardless of what an earlier cell (e.g. cell 6's waveform/spectrogram) left open. Cell 37 (gate plot) already had its own `plt.figure(figsize=(9, 3))` from the V7 fix in Round 2 -- no change needed there. | Re-ran cell 33 inside the full notebook sequence (cell 6's waveform plot executes first, same as a real run) and confirmed `plt.get_fignums()` shows 2 open figures afterward, with `plt.gcf()`/`savefig()` correctly grabbing cell 33's own figure -- a clean, single retention plot with no merged waveform, matching what the reviewer's own render already showed. Screenshot `09_retention_clean_single_figure.png`. |

### Full verification

**`compile()`, every cell, both notebooks:**
```
notebook/audio_workshop.ipynb cells: 65 errs: 0
demo/hook_demo.ipynb cells: 13 errs: 0
```

**Both zips, rebuilt:**
```
$ python notebook/make_bundle.py
wrote .../notebook/workshop_files.zip (2.8 MB, 17 files)
wrote .../notebook/facilitator_files.zip (3.7 MB, 20 files)
```

**`run_tests.py` and the frozen-file check:**
```
$ python prototype/run_tests.py
PASS  a. imports                  11 modules import  (0.3s)
PASS  b. weights                  5 members, labels discussion / decision / action_item  (0.1s)
PASS  c. shipped KPIs reproduce   dev 0.96/0.948, test 0.84/0.825, 36-turn test 0.86/0.853, sample 7/9, probes 9/12  (3.3s)
PASS  d. demo minutes             no decisions, one action item (catering by Friday, Priya)  (0.2s)
PASS  e. --text example           runs, model matches the hand labels 7/7  (0.3s)
PASS  f. hands-on                 solution: both parts OK, 35/45 at knob 0; unfilled version says 'not written yet'  (1.1s)
PASS  g. training data            leak checks pass  (2.2s)
PASS  h. scorecard                frozen scorecard runs (hash ignores line endings)  (1.0s)
PASS  i. smoke runs               5 scripts exit 0  (2.5s)
PASS  j. text clean-up            text clean-up: 31/31 clean-up checks pass  (0.5s)
PASS  k. holdout meetings         holdout 1 0.879/0.806, holdout 2 0.852/0.762 (accuracy / macro-F1)  (4.9s)
SKIP  l. audio (optional)         no --audio FILE given  (0.0s)
PASS  m. repo unchanged           git status and git diff identical before and after  (0.1s)

12 passed, 0 failed, 1 skipped in 16.6 s
```
```
$ git diff --stat main -- prototype/ data/
(no output -- nothing frozen was touched)
```

**Hook cells in the order `demo/FACILITATOR.md` now gives, `facilitator_files.zip` alone, empty folder:**
```
$ <exec cells 2, 8, 10, 12 in that order against the unzipped folder>
[cell 2]  OK: ffmpeg: found; GPU: no -- small.en preloaded and ready
[cell 8]  OK: Audio object displayed (backup_meeting.wav) -- plays only, no button
[cell 10] OK: "Press. One. Button." -> rendered minutes HTML (room has already
          called out decisions/to-dos by the time this cell is run)
[cell 12] OK: "Press. One. Button." (text-only fallback, demo/script.txt) -> same minutes
DONE
```
Confirms the running order `FACILITATOR.md` now describes -- play (8), room
calls out, press (10) -- is exactly what the notebook does.

**Participants' notebook, `workshop_files.zip` alone, empty folder -- solved**
(throwaway copy, reference lines filled in at cells 28/50, `hands_on.py`
never touched):
```
[cell 28] OK: check (MEMORY_ON = True): OK  got [-0.9964  0.5838], expected [-0.9964  0.5838]
[cell 50] OK: check: OK  got [ 0.5263  0.5426  0.402  -0.7362], expected [ 0.5263  0.5426  0.402  -0.7362]
[cell 61] OK: Ravi: first "Sure." (after the proposal) -> decision (56%)
              Ravi: second "Sure." (after the request) -> action_item (99%)
DONE solved errors: 0
```
All cells ran 0 errors; the printed numbers match the standalone
re-derivation for R2 above exactly.

### Screenshot added this round (`plan/handoffs/wp6/`, 1 new file)

```
09_retention_clean_single_figure.png  R3: cell 33's retention graph, confirmed
                                       to draw on its own fresh figure (no
                                       merged waveform artifact) with an
                                       earlier audio-plot cell run first
```

---

## Not checked / open

- Same as every round: Colab, a real T4, the microphone recorder, the
  live skit, and how anything actually sounds -- no Colab or microphone
  available from this session.
- The run-sheet's per-block CPU timings for blocks 3-10 are all "<1s" or
  "~1s" from this machine's runs; none of them has ever been close to its
  budget in any round, so this is not flagged as a risk, but it hasn't
  been measured on an actual T4 either.
