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

*(Full verification -- unsolved/solved run, compile(), run_tests.py, frozen check, screenshots -- in progress; see "Not checked / open" below for anything still outstanding when this was sent.)*

## Not checked / open

- No Colab/real mic/T4 from this build session, same as WP7/WP8.
- F1's full re-architecture (folding existing long theory blocks into collapsed "More detail" cells across blocks 2-9) was not done in this pass -- time-boxed given the 3-day deadline; happy to do a follow-up pass if the PM wants it prioritised over further WP10 candidates.
- The headline's ~2-minute addition to block 9 (15 min budget) is tight; `plan/SPEC.md`'s Agenda minutes are outside this package's scope to change.
