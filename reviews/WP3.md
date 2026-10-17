VERDICT: CHANGES

Reviewed commit `2f2d470` (parent `86d4b8e`), in a detached worktree. IDs claimed: N6, N7, N8, plus the WP2 carry-over (cell 23 CPU wording).
Everything below was re-run by me, on CPU only (no GPU, no Colab).

Short version: the technical content is right and every number you named reproduces. There are no blockers. I found three should-fix items: the held-out meeting was the model's best case and the cell does not say so; the loss example prints a probability that does not reproduce the loss shown next to it; and the notebook invites participants to skip sections 1-2, which breaks sections 3-5. All three are a few lines of text. Verdict is CHANGES for those, and I answer your `hands_on_reference.py` question at the end: drop it from the zip.

## Findings

| # | Severity | ID | File:line | What is wrong | How to fix |
|---|---|---|---|---|---|
| 1 | should-fix | N7, honesty | `notebook/audio_workshop.ipynb` cells 39-41 | The keyword-vs-LSTM test uses `student_council` and presents 12/18 vs 17/18 as the test on unseen lines. Both numbers are real and the meeting is truly held out. But `progress.md:218` records that this meeting was chosen as the typed demo because it is "the model at its best (17 of 18 turns, no wrong items)". I scored all 17 held-out meetings with the same two labellers: keyword rules 111/149 (0.74) and 131/149 (0.88) for the LSTM on the first set, 93/122 (0.76) against 104/122 (0.85) on the second. `student_council`'s 5-turn gap is the largest in either set, and in three meetings (`waiting_room`, `allotment_group`, `culture_night`) the keyword rules score better than the LSTM. The cell text never says it picked the best meeting, which reads as the cherry-picking Rule H warns about, here applied to the test. | Say in cell 39 or 41 that this is the team's best-case held-out meeting, and quote the overall figure (about 0.85-0.88 for the LSTM, about 0.74-0.76 for the keyword rules across 17 held-out meetings; `python prototype/run_tests.py` check k prints 0.879 and 0.852 for the two sets). Optionally, if you want it computed in the notebook, add `holdout_data.py` to the zip and score `HOLDOUT` and `HOLDOUT2` in a cell; otherwise quoting the figures with their source is enough. |
| 2 | should-fix | N7 | cell 38 (output) and cell 39 | The worked loss example prints `p(correct) = 0.004, loss = 5.61`. A participant who checks it with a calculator gets -ln(0.004) = 5.52, not 5.61. The cell prints the probability to 3 decimals but computes the loss from the unrounded probability (0.0037; I confirmed `p = 0.0037`, `-ln p = 5.61`). N7 is the one place the Programme wants participants to see the formula work, so the mismatch will be noticed. The cell also never says the log is natural. | Print `p(correct) = {p_correct:.4f}` (0.0037 and 0.9957) and say "natural log" once in cell 37. |
| 3 | should-fix | N6-N8 (robustness) | cell 3 (the sections 1-2 note) | Cell 3 says participants "can follow along or skip ahead" through sections 1-2. If a participant does skip them (runs only setup and then section 3), sections 3-5 fail. I ran setup plus cells 25 onward: `NameError: np` (cell 26), `plt` (cell 31), `transcript` (cell 33) and `turns` (cell 38). Section 5 already needed `transcript` before WP3, but the new section-4 cells make the dependency larger. | Change the note to: "Participants: run these cells too (about a minute on a T4); the later sections reuse their results." Or make sections 3-5 self-contained by importing `numpy` and defining `transcript` from the placeholder when it is missing. The one-line wording change is enough. |
| 4 | nit | N6 | cell 32 | "memory has far fewer steps to survive there" presents the hourglass as a memory improvement; cell 41 then correctly disclaims any claim about the trained classifier. They do not contradict each other, but cell 32 reads stronger than the evidence. | Say "fewer steps" in the architecture sense only, or add "(a design idea, not something this classifier proves)". |
| 5 | nit | N6 | cell 33 | The hourglass is shown as a table of units and time steps (8, 16, 32, 16, 8 and 88, 44, 22, 44, 88), which is correct and computed from the real model. N6 asks for the hourglass "as a picture"; a small bar plot would show the shape. | Optional: a bar chart of units per layer. |
| 6 | nit | N8 | cell 46 | After the sweep the knob stays at +6. Later cells do not use it, so nothing breaks, but a participant who then re-runs Part B in cell 43 gets its own `FORGET_KNOB = 0` line, so the leftover is harmless. | Optional: end the loop with `hands_on.FORGET_KNOB = 0`. |
| 7 | nit | zip | `notebook/make_bundle.py` | The zip also ships `prototype/rnn_ctc_demo.py`, `prototype/demo.py`, `demo/script.txt` and `demo/backup_meeting.wav`, none of which the notebook uses. | See "Answer to your zip question" below. |

## Your six "look hardest" points

1. **Every printed number comes from a cell: yes, and they reproduce.**
   - Loss example (cell 38): `Venue is confirmed...` p(correct) 0.0037, loss 5.61; `Yes, I'll email them today.` p 0.9957, loss 0.00. Identical on a transcript made with `base.en`, `small.en` and `medium.en` (all three give 9 turns, so the positional `label_turns_stub` gold stays aligned).
   - Keyword vs LSTM (cell 40): 12/18 vs 17/18, with the one LSTM miss ("No refreshments this time...", gold decision, got discussion). Reproduced in the notebook run and in my separate per-meeting count.
   - Knob sweep: 10/45 at -6, 35/45 at 0, 22/45 at +6, with the cell's own `part_b()` output. I ran the sequence 0, -6, 0, +6, 0, +6, -6, 0 to check the knob does not accumulate (`load_trained_model()` reads the weights fresh each call): the three numbers repeat exactly every time.
   - Hourglass (cell 33): 8/16/32/16/8 units, 88/44/22/44/88 steps.
2. **SPEC tension 2 (keep a lot is not better): pass.** Cell 45 says "'Keep a lot' (+6) is **not** 'better memory'". The numbers back it: both extremes score worse than 0 on the trained model (10/45 and 22/45 against 35/45), while the toy gap at +6 grows from 0.17 to 0.37 over 12 words (the first word still shows at the end) and at -6 it is gone by word 3.
   **SPEC tension 3 (no longer memory than an RNN): pass.** Cell 41: "We don't claim the trained LSTM remembers further back than a plain RNN on this task". A grep of every markdown cell for "better", "further back", "longer", "remembers" finds nothing else that claims it (see nit 4 for cell 32).
3. **Section order matches the Agenda: yes.** Cells 32-36 (LSTM, block 6), then cells 37-41 (teaching it and catching it out, block 7), then Hands-on B (cells 42-46, block 8), then section 5.
4. **Knob-sweep cell with the stub: sensible.** With the unwritten `lstm_cell_update`, cell 46 prints `PART B ... check: not written yet (it returned None)` three times and no traceback. With the solved line it prints, per setting, the knob value, the toy gap row, and the trained-model score (output below).
5. **Zip-only folder: yes.** Fresh zip (22 files, 3.8 MB), every cell run by `nbclient` from a folder holding only the zip and the notebook: 52 cells, no errors, 103 s on CPU (the builder reported 57 s; mine ran with other jobs). A second run with Part A and Part B solved inline also had no errors (101 s).
6. **Timing against blocks 6-8:** the new cells compute for well under a second each (hourglass 0.3 s, token demo 0.0 s, loss example 0.1 s, keyword-vs-LSTM 0.4 s, knob sweep 0.0 s unsolved and 0.6 s solved), against 20, 10 and 20 minutes. Nothing overruns.

## Requirement ID check against the PDFs

| ID | Result |
|---|---|
| N6 | Pass. Eraser (forget), pencil (input) and voice (output) are all named; `c = f·c_prev + i·g` appears in words and symbols; five layers shown with real unit and step counts; `<eot>` demonstrated in a two-turn example (`['can', 'you', ..., '?', '<eot>', 'yes', ...]`, `turn_end_indices = [9, 14]`) and explained as the only speaker-change signal. Nits 4-5. |
| N7 | Mostly. Label to prediction to loss to update is stated; the `-log p` example works but has the rounding mismatch (finding 2); the unseen-lines comparison is real but uses the best-case meeting without saying so (finding 1). The Agenda's failure case is shown honestly: the trained model misses "Venue is confirmed for the 14th" (p 0.0037) and the notebook says plainly that the keyword checklist gets it right because "confirmed" is a keyword. I verified that: `label_turns_keywords` labels that turn `decision` on all three transcripts. That differs from the Agenda's framing and the notebook reports it as found, which is correct. |
| N8 | Pass. Part B is unchanged and the check confirms the correct line; the knob changes the forget gate by -6, 0, +6; the sweep prints the toy gap and the trained score each time, and the honest "keep a lot is not better" text is there. |
| WP2 carry-over | Pass. Cell 23 now says all three rows are CPU runs and tells the facilitator to check the order on the day's Colab before the vote. |

## Answer to your zip question (`hands_on_reference.py`)

Nothing in the notebook needs it. `prototype/hands_on.py` imports it only inside `if SOLUTION:` (line 84), and `SOLUTION = "--solution" in sys.argv` (line 32), which a notebook never sets. I proved it: I removed `hands_on_reference.py`, `rnn_ctc_demo.py`, `demo.py`, `demo/script.txt` and `demo/backup_meeting.wav` from the zip (17 of 22 files left), solved Parts A and B inline, and ran all 52 cells from a zip-only folder: **0 errors**, 67 s. So I recommend dropping `hands_on_reference.py` from `PROTOTYPE` in `notebook/make_bundle.py`: it hands every participant the Part A and Part B answers (`np.tanh(W_x @ x + old_memory + b)` and `f * c_prev + i * g`). The facilitator still has it in the repo for `python prototype/hands_on.py --solution` and for `run_tests.py` (check f), which do not use the zip. `rnn_ctc_demo.py` and `demo.py` are also unused by the notebook and can go; `demo/script.txt` and the backup WAV are only needed if a participant is meant to rerun the facilitator notebook, which the orchestrator has said they are not.

## Commands I ran, with output

```
$ bash reviews/check_frozen.sh 2f2d470
frozen: none changed / outside allowed list: none        RESULT: OK (exit 0)
changed since 86d4b8e: HANDOFF-WP3.md, notebook/audio_workshop.ipynb, notebook/make_bundle.py

$ python notebook/make_bundle.py          -> 22 files, 3.8 MB (adds demo_student_council.txt)

# full notebook, folder with only the zip and the .ipynb, nbclient, base.en, CPU
[33] 0.3s  layer 1 8 units 88 steps / layer 2 16, 44 / layer 3 32, 22 / layer 4 16, 44 / layer 5 8, 88
[35] 0.0s  ['can','you','follow','up','with','catering','by','friday','?','<eot>','yes',"i'll",'email','them','today']  turn_end_indices = [9, 14]
[38] 0.1s  Venue is confirmed for the 14th, catering still pending.   gold = decision, p(correct) = 0.004, loss = 5.61
           Yes, I'll email them today.                                gold = action_item, p(correct) = 0.996, loss = 0.00
[40] 0.4s  keyword rules: 12/18 turns right / trained LSTM: 17/18 turns right / LSTM miss: 'No refreshments this time, ...' (gold decision, got discussion)
[43][46] with the unwritten stub: "check: not written yet (it returned None)" (4 times in total, cell 43 once and cell 46 three times)
[48] section 5 minutes unchanged, "Venue is confirmed" still labelled discussion
TOTAL 102.9 s, no cell errors (52 cells)
```

Knob sweep with the correct line (my script, `part_b()` called in the order 0, -6, 0, +6, 0, +6, -6, 0):
```
FORGET_KNOB = +0   gap 0.17 0.09 0.05 0.03 0.02 0.01 0.01 0.00 ...        Trained model, test meetings: 35/45 turns right (0.78)
FORGET_KNOB = -6   gap 0.17 0.01 0.00 0.00 ...                            Trained model, test meetings: 10/45 turns right (0.22)
FORGET_KNOB = +6   gap 0.17 0.18 0.19 0.20 0.21 0.23 0.25 0.27 ... 0.37   Trained model, test meetings: 22/45 turns right (0.49)
(the same three numbers on every repeat: no accumulation)
```

Held-out meetings, all of them (`holdout_data.HOLDOUT` and `HOLDOUT2`; each pair is turns right for keyword rules / trained LSTM, out of the turns in that meeting):
```
hiking_club 14 / 17     film_society 11 / 14    lab_group 12 / 14       student_council 12 / 17   hostel_floor_2 11 / 13
orchestra 12 / 13       cca_vote 12 / 13        startup_standup 12 / 16 waiting_room 15 / 14
HOLDOUT    149 turns: keywords 111 (0.74), LSTM 131 (0.88)
swimming_club 11 / 13   coding_bootcamp 12 / 15 charity_run 11 / 14     school_newspaper 11 / 12  cleanup_day 12 / 14
debate_prep 11 / 13     allotment_group 13 / 12 culture_night 12 / 11
HOLDOUT2   122 turns: keywords  93 (0.76), LSTM 104 (0.85)
```

Transcript stability for cell 38 (`get_transcript(sample_meeting.wav, pause_gap=1.5)` with each model): 9 turns on `base.en`, `small.en` and `medium.en`; p 0.0037 and 0.9957 each time; keyword label for the Venue turn: `decision` each time.

Skip test: setup plus cells 25 onward gives NameError for `np` (cell 26), `plt` (cell 31), `transcript` (cell 33), `turns` (cell 38).

```
$ python prototype/run_tests.py
```
Not re-run this round: `prototype/` is untouched since `86d4b8e` (`check_frozen.sh` and the file list show only the notebook, `make_bundle.py` and the handoff changed), and it passed there.

Not verified: Colab, a T4, and any audio playback.

## To get to PASS

Fix findings 1-3 (all notebook text, one decimal-places change and a wording change) and make the `make_bundle.py` change if the orchestrator agrees. I will then re-run the notebook from a zip-only folder and read the changed cells.
