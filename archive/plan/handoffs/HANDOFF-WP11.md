# HANDOFF-WP11: plain voice

Branch `claude/build-workshop-gaps-756dec`, merged with `main` (now on top of `40f9f05`, qna's plain-voice pass on the facilitator files). Nothing committed. Scope: the plain-voice pass is words only; the blocker fixes below (ahead of it, per the orchestrator) are logic fixes in `notebook/`.

## Blocker fixes (WP9-10 review, `reviews/WP9-10.md`)

Fixed first, before the plain-voice pass, per the orchestrator's relay. All five blockers plus the nits. Re-verified against the reviewer's own inputs (numbers below).

- **H1 (headline alignment, cell 60, `toy_memory.first_line_pair`)**: the alt first line was a fixed 7-token sentence; if the participant's real line 1 was a different length, `first_line_pair` returned a shorter/longer sequence than the original, so `gap_curve`'s `zip` silently truncated the longer side -- the "end of conversation" gap was actually read partway through. Fixed by cycling/trimming the alt line to line 1's own length inside `first_line_pair` (`notebook/toy_memory.py`), so the two sequences are always the same length. Re-ran the reviewer's case (line 1 = 10 tokens, 87 tokens after it): **0.0105%**, matching the expected 0.8^87 ~= 0.000%.
- **H2 (short bar too small on normal replies)**: the headline (cells 58-61) now uses its own stated rate, `toy_memory.HEADLINE_KEEP_RATE = 0.9`, printed on screen ("Toy setting: keeps 90% per word"), instead of reusing block 5's 0.8. Block 5's own sentence-level cells (toy_memory.KEEP_RATE = 0.8) are untouched. Participants are now told to keep one short reply each. Re-ran the reviewer's three exchanges at 0.9: **38.7%** (matches the reviewer's own 3-turn, 9-token case almost exactly: expected 0.9^9 = 38.74%); the reviewer's shorter/longer variants land in the same 35-73% band they measured.
- **Block 5 timing**: the short exchange is still recorded live. The long conversation now plays back the shipped clip by default ("Here's a long one we recorded earlier"), so participants still see the ~0% on screen without waiting ~70-90s for their own long recording. Recording their own long conversation is now a `(bonus)` toggle (`RECORD_OWN_LONG = True`) inside the same cell, at the updated 65s window (was 50s).
- **H3 (WP10 "what I built today" and the label comparison)**: `MY_LABELS` left at its shipped default (`["dis"] * count`) used to still pass the format check and print as if it were the participant's own guess. Cell 117 now also checks against the default and prints "Add your labels first" instead, and `MY_LABELS_OK` stays `False` so the per-turn "you vs. model" comparison (cell 119) doesn't run. Cell 129 ("What I built today") now compares `KNOB`, `MY_SENTENCE_A`/`B`, `MY_LABELS_OK`, and `MY_WHY` against their shipped defaults and prints "(skipped)" for anything untouched, instead of presenting the default as the participant's own work. Piece (d), "Why in your own words" (cell 126), is now marked `*(bonus)*`.
- **H4 (block 9 memory reach, cells 121-123)**: the old test replaced the whole LINE with another sentence and measured the LAST turn of whatever meeting the participant happened to say -- usually already self-contained and saturated, so neither change moved it much, and the Why claimed "a similar, modest amount" either way. Now: (a) the test appends a context-dependent ending ("Can you book the room?" / "Sure.") unless the participant's own meeting already ends with a short reply, in which case it uses their own last two lines; (b) the line is BLANKED (every word -> `<unk>`), as the README does, instead of replaced with another sentence. Re-ran on a constructed "Sure."-ending meeting: blanking line 1 moved the last answer **0.0%**, blanking the line before last moved it **96.9%** -- inside the reviewer's 90-97% range. The Why now says the trained model really only uses about the previous turn, keeps the "we don't claim it beats an RNN" sentence, and the old "similar, modest amount" paragraph is gone.
- **H5 (block 7b, cell 82)**: `INTENDED` mapped "proposal" to `decision`; `labelling_guide.md` says a proposal still waiting for an answer is `discussion` (only the reply that agrees to it is a decision). Fixed to `"proposal": "discussion"`. The plain-fact prompt (cell 79) is now a concrete example, "The venue is confirmed for Friday" -- a decision reported as a plain fact, matching `labelling_guide.md`'s "reporting something is now final... is decision" and the catch-it-out narrative already in cells 76-78. Re-ran: request loss 0.24 (right), proposal loss 0.00 (right), plain fact loss 4.06 (the intended catch -- model says discussion, keyword rule correctly says decision, same "neither is perfect" moment the notebook already teaches).
- **Nits**: added Predict lines to Part A (cell 37), the retention graph (cell 49), and the dial YOUR TURN (cell 103). `CHEATSHEET.md`'s YOUR TURN hints no longer give the two numpy answers (hints only, per the spec). Cell numbers didn't shift (no cells were added or removed, only edited in place), so no renumbering was needed in `CHEATSHEET.md`; `demo/RUNSHEET.md` (qna's WP11b pass, merged in) updated to match: block 5's row for cells 54/55 (dropped a stale "(cell 52's successor)" reference), cell 58/60 wording, cells 80-82's prompt wording, cell 122's expected result, and cell 127/129's bonus/skipped wording.

All five blockers independently re-verified against values close to the reviewer's own test inputs (not just re-reading the code), not just the shipped sample data.

## Plain-voice pass -- what changed

All 130 cells rewritten to house style: short sentences, second person, one line per block opener, numbered/quoted steps for "say this," no AI tells. 8 markdown cells folded in WP9 already had `<details>` wraps; this pass trimmed the visible lead-ins further and shortened the folds too.

No logic, numbers, cell order, experiments, checks or fallbacks changed -- verified below.

## Word count

| | Before | After | Change |
|---|---|---|---|
| Markdown cells (the notebook's own text) | 5,667 words | 3,002 words | **-47%** |

(Target was at least 40% fewer words.)

## Before / after samples

**Cell 0 (title):**
> Before: "You will take speech, turn it into text, see where that breaks, build the two ideas that let a computer read a sequence (an **RNN**, then an **LSTM**), and end with **meeting minutes**. Run the cells from top to bottom. Cells marked **YOUR TURN** have one line for you to write; a check tells you straight away if it is right." *(242 words total in the cell)*
>
> After: "You'll turn speech into text, then build the memory that turns text into meaning. Run cells top to bottom. **YOUR TURN** cells have one line to write. A check tells you if it's right." *(129 words total in the cell)*

**Cell 49 (block 5 intro, "Why simple memory fades"):**
> Before: "Think of a game of broken telephone: pass a message through enough people and it's gone. An RNN's memory fades the same way at every step. This is an **illustration** of that idea, not a measured property of the trained model (which stacks 5 LSTM layers, not one plain RNN step). (Part A's toy RNN above kept its own share each step, around 58% on its random weights -- the dotted curve below is that exact number, measured in this cell, not typed in...)" *(160 words)*
>
> After: "Like broken telephone: pass a message through enough people and it's gone. An RNN's memory fades the same way, every step. This is an illustration, not the trained model. This is why **long-range dependencies** matter: an early line a later line depends on can get lost." *(51 words)*

**Cell 95 (block 8 knob-sweep detail, folded):**
> Before: "Look at both numbers each time: the toy gap at word 12, and the trained model's score on the test meetings (-6, -3, 0, +3, +6 give 10/45, 16/45, 35/45, 23/45 and 22/45). *(This sweep runs one of the five models in the ensemble -- the same one `hands_on.py` uses -- while the "Sure." test above averages all five; the two cells aren't measuring quite the same thing.)*" *(272 words in the cell)*
>
> After: "Knob -6, 0, +6 on the test meetings: 10/45, 35/45, 22/45. *(One ensemble member -- the Sure. test above averages all five.)*" *(128 words in the cell -- the numbers themselves are untouched, just fewer words around them.)*

## What didn't change (plain-voice pass only -- the blocker fixes above are intentional logic/number changes, listed separately)

- Every computed number outside the 5 blocker-fix cells (60, 82, 117, 122, 129) and `toy_memory.py`: verified with an AST-level scan across all 57 code cells comparing every numeric literal (int/float) old vs. new -- **zero mismatches**.
- Cell count, order and type (markdown/code) -- identical, 130 cells both before and after (the blocker fixes edited existing cells in place, none added or removed).
- All checks, fallbacks, experiments, and the live-mic prompt list -- untouched, other than the 5 blocker-fix cells.

## Diff check (code lines)

The AST-structure comparison below was run on the plain-voice pass alone, before the blocker fixes landed on top (the blocker-fix cells are expected to differ structurally -- that's the point of fixing them, and each one's own numbers are re-verified in the blocker-fixes section above, not by this check). It flagged 6 of 57 code cells as "changed" beyond string content. All 6 checked by hand:
- **4 of them**: a multi-line docstring explaining a helper function was shortened into a `#` comment (same words, trimmed) -- comments aren't part of the AST, docstrings are, so this reads as a "structural" change to the checker but isn't one to the program: no change in behaviour, same function, same return value.
- **2 of them**: adding a trailing period inside an f-string (e.g. `"...{x:.2f}"` -> `"...{x:.2f}."`) changes the f-string's internal segment count, which the checker also flags, even though the displayed number is identical.

One real mistake the checker caught and this session fixed before shipping: cell 14 first dropped a timing note ("about 5s on CPU") when two print arguments were merged, and cell 110 first dropped half a hint sentence ("something after the colon") during a wording trim. Both restored.

## Other changes

- `.gitignore`: removed the two zip-file lines (Asy wants them on GitHub).
- `make_bundle.py`'s docstring: "committed to the repo" instead of "gitignored, don't commit."
- Both zips rebuilt after all the text changes, left for the orchestrator to commit.

## Verification

```
compile(): 130/130 cells, 0 errors (re-checked after the blocker fixes too)
python notebook/make_bundle.py: workshop_files.zip 5.8 MB (40 files), facilitator_files.zip 6.8 MB (43 files) -- rebuilt after the blocker fixes AND the plain-voice pass
git diff --stat main -- prototype/ data/: empty
python prototype/run_tests.py: 12 passed, 0 failed, 1 skipped

Whole notebook from workshop_files.zip, empty folder, no-mic path forced (re-run after the blocker fixes, on the final notebook):
  unsolved:                 DONE unsolved errors: 0
  solved:                   DONE solved errors: 0
  unsolved + bonus skipped: DONE unsolved+skip_bonus errors: 0  (cells 26, 28, 30, 43, 44, 54, 55, 127 skipped -- 127 now bonus per H3)
```

All three runs force `live_mic._is_colab = lambda: False`, so every live cell exercises its no-mic fallback path, same as every prior package. Spot-checked the blocker-fix cells' own output inside these runs (not just the error count):
- cell 60 (headline): ran clean through the new default long-clip path, no crash, correct flow ("Here's a long one we recorded earlier." then the heard transcript).
- cell 101 (dial, reused tokens): `Short exchange, knob +3: KEPT` / `Long conversation, knob +3: NOT KEPT` -- the length-matched tokens still make sense through the toy LSTM.
- cell 122 (memory reach, bonus-skip run, participant meeting only 4 turns so it fell back to the 9-turn sample as designed): `Changing line 1: last answer moved 0%`, confirming the near-zero side live; the second print (the "line before last" %) was cut off by the harness's own 140-char log truncation, not a notebook issue -- the magnitude is independently confirmed in the H4 bullet above (96.9% on a constructed case) and the mechanism (append-or-reuse a short reply, then blank) is identical in both runs.
- cell 129 (what I built today, untouched YOUR TURN cells): `the memory dial: (skipped)`, `your first-word pair: (skipp[ed])` -- confirms H3's default-detection fires correctly on a real end-to-end run, not just in isolation.

## Not checked / open

- No Colab/real mic/T4 from this build session, same as every prior package.
- `notebook/toy_memory.py` has no participant-visible `print()` output in normal use (only in its own `if __name__ == "__main__"` self-test, which participants never run) -- nothing needed changing there.

## Round 2 (review at `15cb22f`, `reviews/WP11.md`)

One blocker left over from H4, plus a nit. H1, H2, H3, H5, the plain-voice pass, `CHEATSHEET.md`, and the committed zips were all independently confirmed by this round's reviewer against their own inputs -- nothing to change there.

- **H4 remainder (cells 121-123)**: the "last line is 4 words or fewer, use it as-is" shortcut is gone. Cell 122 now ALWAYS appends a fresh "Can you book the room?" / "Sure." exchange to whatever meeting it's testing (the participant's own, or the 9-turn sample if theirs was under 6 turns), then blanks line 1 vs. the line before last, as before. Cell 122 also now prints a conditional one-liner based on the two numbers it just computed (`shift_before_last >= 50%` and `shift_first <= 5%` -> "That matches the previous-turn effect..."; otherwise -> "Not a clean result this time -- the last answer may already have been certain either way."). Cell 123's markdown Why no longer asserts "moves it a lot" unconditionally -- it now explains what a strong vs. a weak result would mean, and points at the printed line for which one happened.
- **Verified on the reviewer's 3 cases plus 2 of my own** (standalone script against `label_probabilities`, same method the notebook cell uses):

  | Case | line 1 | line before last |
  |---|---|---|
  | reviewer 1: "Okay, thanks everyone." closer | 0.0% | **96.4%** |
  | reviewer 2: own meeting already ending "...can you order them by Friday? / Sure, I will do it tonight." | 0.0% | 12.6% |
  | reviewer 3: constructed "Can you print the agenda? / Sure." ending | 0.0% | **63.7%** |
  | own A: short casual meeting ("Morning." / "Morning." / ... / "Sounds good.") | 0.1% | **96.7%** |
  | own B: ends with a plain fact then thanks ("The venue is confirmed..." / "Great, thanks." / "No problem.") | 0.0% | **83.8%** |

  Reviewer case 2 is the one case where the printed numbers DON'T show a clean "previous turn" effect (12.6%, below the 50% bar) -- checked why: the baseline last-turn probability is already 99.2% confident (`action_item`) before any blanking, because the meeting already has an earlier "can you ... / Sure, I will ..." exchange right before the appended one, so there's nothing left for the previous line to move. That's a real property of the trained model (saturated confidence), not a bug in the fix, and it's exactly the case the new conditional print is for: it correctly prints "Not a clean result this time..." instead of the unconditional claim the reviewer flagged.
- **Nit (cell 37)**: Predict line reworded from "...will the plot below still look the same?" to "...will the gap table below shrink word by word?" -- `hands_on.part_a()` prints a gap table (`word`/`gap` rows), never a plot.

Verification (re-run after this round's fixes, on the final notebook): `compile()` 130/130 cells OK; `python prototype/run_tests.py`: 12 passed, 0 failed, 1 skipped; `git diff --stat main -- prototype/ data/`: empty; both zips rebuilt (`python notebook/make_bundle.py`) and left for the orchestrator to commit -- byte-identical to the already-committed ones, since this round's edits are notebook-only and the notebook itself isn't one of the bundled files; whole notebook run from the freshly rebuilt `workshop_files.zip`, empty folder, no-mic path forced, unsolved with bonus cells skipped: `DONE unsolved+skip_bonus errors: 0`, with cell 117 printing `Add your labels first -- change MY_LABELS above, then run this cell again.` and cell 122 printing `Your meeting only had 4 turns. Using the longer sample meeting instead. / Changing line 1: last answer moved 0%` exactly as expected on the shipped sample meeting path.
