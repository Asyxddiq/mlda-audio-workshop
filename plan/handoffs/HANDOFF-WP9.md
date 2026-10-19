# HANDOFF-WP9: learner flow, the headline aha, cheat sheet, timing (+ WP10 personalisation)

Branch `claude/build-workshop-gaps-756dec`, on top of WP7 round 2 + WP8 (`aab980f`) + `main`'s `plan/LEARNER_FLOW_SPEC.md`. Nothing committed. Scope: `notebook/` and `demo/`; `prototype/` untouched.

## WP10 candidate table (Step 1, as requested -- builder's proposal)

| Candidate | Minutes | Difficulty (no-ML) | Reinforces | Risk if broken |
|---|---|---|---|---|
| **Set the dial so the long conversation is remembered** *(implemented)* | 2-3 | Low (one integer, one check) | The whole headline: what a "keep" setting controls | None -- doesn't gate anything downstream |
| Write the "Why" for one experiment in their own words | 3-5 | Low (reflective, no code) | Forces articulating the concept, not just reading it | None -- pure markdown |
| Choose their own prompts/sentences for an experiment | 2-3 per experiment | Low-Medium (must avoid breaking word-count assumptions in 4b/5b) | Ownership; "does it generalise to MY words" | Low-Medium -- a badly chosen pair (e.g. identical sentences) makes the aha boring, not broken |
| Write the labels for their own 9b meeting | 3-5 | Medium (needs to understand act/dec/dis) | The labelling task itself | Low -- only affects their own comparison column |
| Write one line of the minutes-formatting (e.g. the checkbox line) | 5-8 | Medium (string formatting, needs a clear spec) | Minutes generation, concretely | Medium -- touches rendering; needs a solid fallback to the frozen `generate_minutes` |
| Turn a built-in experiment into one they design end to end | 10-15 | High | Everything at once | High -- hardest to give a safe fallback for |

**Implemented now (this package):** the dial-setting YOUR TURN (cells 110-112, after the toy LSTM contrast) -- lowest risk, reuses the headline's own infrastructure, directly reinforces the just-taught concept, and has no downstream dependents if skipped. That's roughly 2-3 minutes of hands-on added to a ~117-cell notebook -- a deliberately small first slice (closer to 5% of the workshop's hands-on time than 20%); the PM can add more from the table above if there's room.

## What's built (WP9)

- **F2, the headline** (new cells 103-114, after 9b): participants record a SHORT 2-3 turn exchange and a LONG 8+ turn conversation, both with the same key fact in line 1. `toy_memory.first_line_pair` (new) swaps line 1 for a fixed alternate fact while keeping every later turn exactly as spoken. Two-bar chart: toy RNN (keep 80%) retention, short vs. long. A second chart contrasts the toy LSTM at knob +3. Then the YOUR TURN dial cell. Then the **trained-model memory reach** test: on the participant's own 9b meeting, changing line 1 barely moves the last line's label; changing the line just before last moves it a lot -- real `label_probabilities`, not a toy, worded so it never claims the trained LSTM "remembers like a toy."
- **F3**: `notebook/CHEATSHEET.md` (setup, live-mic mechanics, per-block prompt table, failure recovery, YOUR TURN hints without answers, take-home) -- added to both zips.
- **F4**: RUNSHEET updated with the headline's rows and an honest timing note (adds ~2 min to block 9, tight; cut cells 93-101 first if short on time, not the headline itself).
- **F1**: light touch, not a full re-architecture (time-boxed per the PM's 3-day deadline). Blocks 2-9 already follow Predict/Try it/See it/Why from WP7-WP8's live cells; this package didn't additionally fold existing theory into collapsed "More detail" cells -- flagged as open below if the PM wants it.

## A bug found and fixed (line-level swap, not in the original WP7 pattern)

`toy_memory.first_line_pair` can share its first word with the alternate line ("The meeting is on **Friday**..." vs "...on **Monday**...") -- both start "the meeting is on", so the word-1 gap can be exactly 0 even though the lines differ, which would divide-by-zero or silently misreport "0% difference from the start." Fixed with `toy_memory.line_boundary_index()`: normalise from the gap at the **end of line 1** (the first `<eot>`), not word 1. Re-verified: no crash, sensible percentages on the shipped conversation.

## L4 proof

- Toy RNN headline: exact `KEEP_RATE^(n-1)` decay still holds past the line boundary (same guarantee as WP7's block 4/5, just referenced from end-of-line-1 instead of word 1).
- Toy LSTM contrast (knob +3): on the shipped 8-turn conversation, retention climbs sharply with the knob (0% at knob 0, up to ~190% plateau by knob ~12) -- confirms the dial has real, visible effect; exact numbers depend on the random toy init, so the notebook doesn't promise a fixed percentage, only the direction.
- Trained-model reach: uses the same `label_probabilities` ensemble as every other trained-model cell in the notebook -- no new claim about the model itself, only a new way of probing it.

*(Superseded by Round 2 below -- cell numbers in this section are from the first pass, before the PM's placement changes.)*

## Round 2

PM decisions applied: moved the headline's pieces to the blocks the slides actually teach them in, fixed the "190%" problem, raised WP10 to its full 5-piece set, folded long theory into `<details>`, and folded in the WP7-8 reviewer's 6 blockers + 2 nits. One verification pass at the end, as asked.

### Placement (PM decision 1)

- **Block 5** (after the existing short/long sentence cells, 5b): the toy-RNN short-vs-long **conversation** headline now lives here, where the slide teaches "why simple memory fades" -- cells 57-61 (`5c`).
- **Block 8** (next to the existing knob, after 8b): the toy-LSTM keep-dial contrast and the YOUR TURN dial now live here -- cells 100-105 (`8c` + the dial).
- **Block 9**: only the trained-model memory-reach test stays, right after 9b's own meeting -- cells 121-123.
- RUNSHEET, CHEATSHEET and the notebook's own cell cross-references all updated to the new indexes.

### The "190%" problem (PM decision 2)

Replaced the raw percentage (which could read "190%", confusing) with a plain **KEPT / NOT KEPT** verdict against a 50% threshold -- a bar chart with two rows, not a number that can exceed 100. When the underlying percentage is implausibly high (the knob adds without ever erasing), the verdict says so in one line instead of showing a number nobody can make sense of. Screenshot `wp9/02_dial_kept_not_kept.png`.

### WP10, raised to its full set (PM decision 3)

| Piece | Where | What it is |
|---|---|---|
| (a) the dial YOUR TURN | block 8 (cells 103-105) | *(carried over from round 1, now correctly placed)* |
| (b) your own first-word-swap pair | block 4 (cells 46-48) | Type two sentences; the check confirms only the first word differs; doesn't gate anything below. |
| (c) label your own meeting first | block 9 (cells 116-119) | Split the old 9b cell in two so participants write `act`/`dec`/`dis` guesses BEFORE the model's labels print, then see both side by side. |
| (d) Why, in your own words | after the wrap-up (cells 126-127) | Free-text markdown + a tiny print-back cell. |
| (e) What I built today | last cell (129) | Reads back the dial value, the typed pair, the meeting labels and the explanation from whichever pieces the participant actually did -- nothing required. Screenshot `wp9/03_what_i_built_today.txt`. |

Not done (per the PM's explicit exclusion): minutes-formatting, end-to-end experiment design.

### F1: folding (PM decision 4)

8 of the longest theory-dump markdown cells across blocks 2-9 (the notebook's title cell, the answer-key table, the RNN/LSTM section intros, the loss-and-training explanation, the held-out-meeting stats, the knob-sweep deep-dive, and the own-meeting instructions) now show a short visible idea, with the rest in a collapsed `<details><summary>More detail</summary>`. Nothing deleted, same content, just folded. Each block's visible flow now reads idea -> Predict -> Try it -> See it -> Why -> hands-on without the long asides in the way.

### WP7-8 fixes (reviewer's 6 blockers + 2 nits, folded in per the PM's instruction)

| # | Fix | Verified |
|---|---|---|
| B1 | 8b's knob sweep now applies the knob to **all 5** ensemble members and averages (same ensembling `label_probabilities` uses everywhere else), instead of member 0 only. | Re-ran 3 of my own "Can you ...? / Sure." exchanges: knob -6 now correctly flips 2 of 3 to the wrong label (`discussion` instead of `action_item`), knob 0/+6 keep the right one -- matches the reviewer's own direction (erasing hurts, not helps). |
| B2 | 6b's gate plot now uses the **trained** layer 1 (member 0), the same technique the static "watch the gates" cell uses, instead of an untrained toy LSTM. Why text says "usually dips at the speaker change" (not "always"). | Reviewer's own number (82% of 1000 random exchanges) is quoted directly in the Why; this is the real trained model now, not a toy. |
| B3 | Cells 44 and 55 (the 4b and 5b percentage calculations) guard against both recordings starting with the same word -- prints a plain message and substitutes a placeholder word instead of `ZeroDivisionError`. | Re-ran with an intentionally identical pair: prints the message, plot still renders. |
| B4 | Cell 44's printed line fixed: "keeps exactly 80% every word" (was "shrinks by exactly 80%", backwards). | Re-read against the actual math: 0.8 is the keep rate, confirmed. |
| B5 | 6b's knob-sweep cell (72) no longer depends on 4b's variables -- if 4b (bonus) was skipped, it records its own Cats/Dogs pair on the spot. | Ran the whole notebook with every "(bonus)" cell skipped: 0 errors (see Full verification below). |
| B6 | The Sure. test and 8b both detect a merged turn (no pause heard) and print "I heard it as one turn -- leave a 2s pause..." instead of going silent. | Re-read the shared `label_sure_exchange` helper; both call sites use it. |
| N1 | Block 9b's recording window: 20s -> 35s. RUNSHEET and CHEATSHEET say "up to 35s". | -- |
| N2 | Block 7b now shows the loss of the **intended** label (request=action_item, proposal=decision, fact=decision), not just the model's own top guess -- "catching it out" means showing how wrong it is against what it *should* say. | Re-read the cell's new `INTENDED` mapping. |

**Two more bugs found while re-verifying (not in the reviewer's list, caught by the "skip every bonus cell" run):**
- `import live_mic` lived only inside block 2b, marked **optional** -- skipping 2b broke every later live-mic cell in the notebook (20 cascading `NameError`s). Moved the import to the always-run Whisper cell (4).
- `SimpleLSTM` lost its import when B2's rewrite removed it from 6b's old toy-LSTM code, but 6b's knob cell (72) and the headline's toy-LSTM cells (101, 104) still needed it. Re-added `from lstm_model import SimpleLSTM` directly in cell 72, the first cell that uses it.

Both were caught **only** by the bonus-skip run the PM asked for -- a plain unsolved/solved run never exercises a skipped "optional" cell, so this is exactly what that extra run is for.

### Full verification (once, at the end, as asked)

```
compile(): 130/130 cells, 0 errors
python notebook/make_bundle.py: workshop_files.zip 5.8 MB (40 files), facilitator_files.zip 6.8 MB (43 files)
git diff --stat main -- prototype/ data/: empty
python prototype/run_tests.py: 12 passed, 0 failed, 1 skipped

Whole notebook from workshop_files.zip, empty folder, no-mic path forced:
  unsolved:              0 errors
  solved:                0 errors
  unsolved + bonus skipped (cells 26,28,30,43,44,54,55): 0 errors
```

Screenshots in `plan/handoffs/wp9/`:
```
01_headline_short_vs_long.png   The headline chart: short exchange 21.0%, long conversation 0.0%
02_dial_kept_not_kept.png       The dial result: KEPT / NOT KEPT, not a raw percentage
03_what_i_built_today.txt       The final summary cell's actual printed output
```

## Not checked / open

- No Colab/real mic/T4 from this build session, same as WP7/WP8.
- The honest finding on the trained-model memory-reach test (block 9): checked against 6 varied replacement lines, changing line 1 and changing the line before last move the last line's answer by a **similar, modest** amount on average -- not the clean "barely vs. a lot" asymmetry the original spec assumed. The notebook's Why text was rewritten to say this plainly (not a clean recency law) rather than ship a claim that doesn't hold up; worth the PM's awareness since it affects the strength of that specific "aha."
- The headline's ~2-minute addition to block 9 is now gone (moved to blocks 5 and 8, where there was more room); block 9's own time budget is back to roughly what it was before WP9.
- `plan/SPEC.md`'s Agenda minutes are outside this package's scope to change.
