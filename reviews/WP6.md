VERDICT: CHANGES

Verification of WP6 at `cb9a4a2` (parent `1b2dfea`), detached worktree, redoing the audit's key checks in the same two roles (facilitator, participant with no ML background). Everything below was run or read by me, on CPU only (no GPU, no Colab).

Short version: the audit's blockers are fixed and the notebook is much clearer. The Hands-on A contrast now appears, the gates are drawn, the retention graph has the third curve, the Sure. test and the own-meeting default work, the Stuck? notes are accurate, the hook has a GPU check, a text fallback and a big-print rendering, and the run-sheet stays operational. I found **one blocker and seven smaller items**, all inside `demo/` and `notebook/`. The blocker: on the backup path the minutes appear while the recording is still playing, which breaks the hook's "room calls out first, then press" order on the very path the facilitator page recommends. Nothing frozen was touched.

## Findings

| # | Severity | Block | Where | What a participant or facilitator experiences | Fix | Effort |
|---|---|---|---|---|---|---|
| V1 | **blocker** | 1 | `demo/hook_demo.ipynb` cell 8 | The cell is `display(Audio(backup_path))` followed by `run_demo(backup_path)` and `display(big_minutes(...))` **in one cell**. The player appears and the model starts at once. In my run the minutes appeared 48 s after the cell started (small.en, CPU; a T4 will be faster, so sooner). The recording is 53 s and the facilitator still has to click play. So the room sees the minutes before it has heard the recording, let alone called out the decisions and to-dos. That is the opposite of the Agenda ("let the room call out ... Then press one button") on the route `FACILITATOR.md` calls "the one path verified end to end". | Split it: cell 8 = the player only (`display(Audio(...))`); a new cell 9 = the press. Update the page ("cell 8 plays, cell 9 presses"). | S |
| V2 | should-fix | 4 | `notebook/audio_workshop.ipynb` cell 28 output vs cells 29-30 | F1 is fixed in the text (cell 30 now says to delete `+ W_h @ h_prev` **and** set the flag) and the "amnesia" cell 31 prints the contrast (`2.85 0.00 0.00 ...`). But the output of cell 28 itself still ends with `Now set MEMORY_ON = False and run again.` (that line comes from `prototype/hands_on.py`, frozen). A participant reads the printed line first, does only that, and gets the old `not quite ... Fix rnn_step() first`. Two instructions contradict each other, one of them on screen in the output. | Add a bold sentence in cell 30: "Ignore the 'Now set MEMORY_ON = False' line printed above: it is only half of the job; also delete the term, or just run the amnesia cell." | S |
| V3 | should-fix | 5 | cell 33 | F7 asked that the legend cover no curve. The legend (now three entries, lower right) sits over the tail of the 0.95 curve from about n = 30 to 47, including its 9.0% endpoint, which is the number the legend itself quotes. I checked the figure from my own executed notebook, and it matches the builder's screenshot 02. | `plt.legend(loc="center right", fontsize=8)` leaves both curves clear. | S |
| V4 | should-fix | 9 | cell 62 | "from the default's near-50-50 miss to a **confident**, correct `decision`". The flip is to `decision` at 56% against discussion at 43%, which is not confident; the caveat sentence beneath is good but this claim contradicts it. | "to a (still narrow) correct `decision`, 56% against 43%". | S |
| V5 | should-fix | 3 | cell 26, `demo/RUNSHEET.md` block 3 | F10 says each cell now imports what it uses. Cell 26 (block 3's only cell) still uses `np` with no `import numpy as np`; after Setup alone it gives `NameError: np` (I ran cells 2, 26). `RUNSHEET.md` says "this cell now imports what it needs", which is false. Cells 33 (graph) and 37 (gates) import numpy and pyplot now; cell 37 still needs `vocab`/`trained` from cell 35 (`NameError: vocab`), cell 59 needs `run_pipeline` from cell 56, and cell 28 after a restart still gives `ModuleNotFoundError: hands_on`; the boxed note under the title covers all of those. | Add `import numpy as np` to cell 26 and correct the run-sheet sentence. | S |
| V6 | should-fix | 1, 2 | `demo/RUNSHEET.md` (onboarding), block 2 timings | (a) Step 1 says to share the notebook **and the zip ahead of time**, then in bold "Do not hand out the zip until after the hook": contradictory on first read. (b) "Setup takes about a minute" and the block-2 timings (about 20 s for cells 4 and 12) are CPU timings with the model already cached; the first call in a fresh Colab downloads the model (about 140 MB for `base.en`, about 460 MB for `small.en`) and I could not measure that, so these numbers are optimistic. (c) The zip upload happens at 00:08 in the same minutes as Setup. | Reword (a): "share the notebook ahead of time; the zip after the hook". Add "(+ model download on first use; not measured)" to (b). | S |
| V7 | polish | 8 | cell 37 (gate plot) | The legend, upper left, covers the top of the "you" bars (the highest eraser and pencil values in the example). **What the plot shows:** the eraser (forget gate) dips at `<eot>` to 0.36, the lowest of the 8 tokens, against 0.57 to 0.83 on the words around it; so "the eraser dips at the speaker change" is visible. The pencil does **not** cleanly spike on content words: it is highest on "you" (0.74) and "book" (0.65) and low on "room" (0.32), so the audit's own earlier phrase about content words should not be repeated in narration. The notebook text does not claim it; the caveat (layer 1 only, one ensemble member, mean of 8 units) is stated. | Move the legend (`loc="upper right"` or outside), and say in the run-sheet "watch the eraser at `<eot>`". | S |
| V8 | polish | 6 | cell 34 | The dial framing is accurate for the LSTM: the knob adds a constant to the forget gate before the sigmoid, and `sigmoid(0)=0.5`, `sigmoid(3)=0.953` match the 0.49 and 0.95 curves I plotted in the audit. One phrase is a bit strong: an RNN has "no control over how much" it keeps; an RNN's weights do set a fixed rate. | "...with one fixed fade rate for every word". | S |
| V9 | polish | 10 | cell 63 | (a) "Take-home challenge: ... retrain": the participants' zip has `training_data.py` but not `train_lstm.py`, so a participant cannot retrain from what they were given. (b) Limit 2 says unseen words are "ignored, including most names"; cell 58 says names are dropped before the model sees the text. | (a) "the facilitators will share the retraining script", or omit; (b) delete "including most names". | S |
| V10 | polish | 9 | cell 61 | Prints two lines, both `Ravi: 'Sure.'` (the first `decision (56%)`, the second `action_item (99%)`), with nothing saying which is which. | Label them "first Sure." and "second Sure.". | S |
| V11 | polish | 8 | cells 53-54 | The sweep uses one ensemble member (as `hands_on.py` does), while the Sure. test uses the ensemble. Member 0 alone labels "Sure." after "Can you book the room?" as discussion 0.94 at knob 0 (the ensemble says action_item 0.80), so the two cells are not about the same model. The explanation in cell 53 for -6 is supported (see below); the caveat belongs next to the sweep. | One sentence: "this sweep runs one of the five models". | S |
| V12 | polish | 1 | `hook_demo.ipynb` cell 2 | A stale comment: "Both cell 3 and cell 4 need these ... (cell 4 used to NameError ...)". | Reword to "cells 6, 8, 10". | S |

## The audit items, re-checked

| Audit item | Result |
|---|---|
| **F1** block-4 text, literally | **Fixed** except V2. Following cell 30 literally (term deleted, flag off) gives `OK` and `gap 2.85 0.00 0.00 ...`; the amnesia cell 31 prints the same in one click (my solved run, cell 31). |
| **F2** gate plot | **Fixed.** Cell 37 draws eraser, pencil and voice per token from the shipped weights, with the caveat. Eraser dips at `<eot>`: yes (0.36). Pencil: see V7. Dial framing: accurate, V8. The section is retitled "an eraser you can set", and the "fixes the short memory" language is gone. Cell 53 links knob 0 to 0.49 and +3 to 0.95. |
| **F7** third curve | Computed in the cell (not typed): `toy RNN: r=0.578`, matching Part A's "58%". The legend problem is V3. The two sentences on why each extreme hurts are in cell 53; I tested the -6 claim: with the knob at -6 on member 0, "Sure." after "Can you book the room?" is action_item 0.44 against 0.54 alone (context mostly stops mattering); at 0 it is 0.94 discussion and at +6 0.98 discussion (see V11). |
| **F8** the Sure. test | Reproduced: `(no lead-in) -> action_item 0.41, discussion 0.32, decision 0.27`; `Can you book the room? -> action_item 0.80, discussion 0.19, decision 0.01`; `Shall we move it to Friday? -> discussion 0.98, decision 0.01, action_item 0.00`. The explanation cell states plainly that the third answer is wrong. |
| **F4** own-meeting default | Reproduced from the notebook's own cell 59: `decision (you) discussion (model, 51% sure) Ravi: Sure.  <- X`; all other lines match with 97% to 100%; unseen words `move`; the prompt and the default-specific hint follow. The "add a line before it" cell 61 flips the first "Sure." to `decision (56%)` (V4, V10). |
| **F5** hook | No GPU: cell 2 printed `GPU: no -- expect the button press to take about 1 minute (small.en, no GPU here)` and loaded `small.en`. The backup player appears (V1). The text-only cell 10 ran in 0.5 s and rendered the same minutes at large size (`<h1>`, 160% font, check boxes; builder's screenshot 01 shows it). Cell 2 took 14.8 s, cell 8 47.8 s on CPU. |
| **F6** Stuck? notes | I made five mistakes myself (three for Part A, two for Part B): `W_h * h_prev` gives a 2x2 grid; no tanh gives `-3.1549`; no `+ b` gives `-0.9874 0.5559` ("close"); Part B `f + c_prev + i + g` gives `2.2523 1.1199 ...` ("bigger"); `f * c_prev + i` gives `1.2686 0.6611 ...` ("close"). Each matches its note. |
| **F10** restart and skip | Partly (V5). Boxed note under the title: present (builder's screenshot 05). Whisper install prints "Installing Whisper, about a minute...". |
| **F3 / F11** run-sheet and facilitator page | Read as a facilitator on the day. `RUNSHEET.md` has onboarding, then blocks 2-10, each as a table of cells, one "Say" line, expected output, CPU time and what to do if it fails; plus notes. It is operational, and the "Say" lines are single sentences, so I see **no scope creep** into lecture content. `FACILITATOR.md` now opens with a six-step running order and a failure table, uses real cell indexes, and has the Rule H history at the bottom. Its remaining issues are V1, V6 and the stale hook comment (V12). Block 1 to 2 hand-off is covered by the onboarding steps. |
| Participants' notebook, both zips from empty folders | `workshop_files.zip` (17 files, 2.8 MB) with only the `.ipynb`: **unsolved 65 cells, 95.9 s, 0 errors; solved (Part A and Part B filled in), 92.0 s, 0 errors**. `facilitator_files.zip` (20 files, 3.7 MB) with `hook_demo.ipynb`: real cells 2, 8, 10 ran with no error. (One earlier attempt at running the hook cells timed out at 900 s while other jobs ran on this machine; the same three cells reran in 14.8, 47.8 and 0.5 s, and a plain-Python run took 17 s, 50 s and 0.4 s, so I treat it as a contended harness run, not a product defect.) |
| `run_tests.py`, frozen check | 12 passed, 0 failed, 1 skipped. `bash reviews/check_frozen.sh cb9a4a2`: **no frozen file changed**; the only flagged files are `plan/handoffs/HANDOFF-WP6.md` and the five screenshots (approved). `git diff --stat main cb9a4a2 -- prototype` is empty. |

## Commands I ran, with output

```
$ bash reviews/check_frozen.sh cb9a4a2      frozen: none changed; outside list: plan/handoffs/HANDOFF-WP6.md + 5 files in plan/handoffs/wp6/ (approved)
$ python prototype/run_tests.py             12 passed, 0 failed, 1 skipped in 30.8 s
$ python notebook/make_bundle.py            workshop_files.zip 17 files 2.8 MB ; facilitator_files.zip 20 files 3.7 MB ; git status clean

# participants' notebook, empty folder + zip only (nbclient, base.en, CPU, 65 cells)
unsolved   TOTAL 95.9 s  errors []
solved     TOTAL 92.0 s  errors []
 [28] OK, gap 2.85 1.85 0.69 0.31 0.16 0.09 0.04 0.03 0.02 0.01 0.01 0.01, "about 58% kept per step"; ends "Now set MEMORY_ON = False and run again."   (V2)
 [31] amnesia: OK, gap 2.85 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
 [33] 0.49^47 = 0.0000%   0.95^47 = 9.0%   toy RNN: r=0.578, 0.0000% left at n=47
 [37] can .41/.44/.46 | you .77/.74/.65 | book .63/.65/.46 | the .83/.26/.63 | room .57/.32/.42 | ? .58/.31/.75 | <eot> .36/.29/.74 | sure .71/.42/.57   (eraser/pencil/voice)
 [47] (no lead-in) -> action_item 0.41, discussion 0.32, decision 0.27 | Can you book the room? -> action_item 0.80 | Shall we move it to Friday? -> discussion 0.98
 [50]+[54] OK; knob -6 10/45 (gap dies at word 2), 0 35/45, +6 22/45 (gap grows 0.17 to 0.37)
 [59] default: "decision (you) discussion (model, 51% sure) Ravi: Sure.  <- X" ; 553 words
 [61] Ravi: 'Sure.' -> decision (56%) | Ravi: 'Sure.' -> action_item (99%)

# hook notebook, empty folder + facilitator_files.zip, real cells 2, 8, 10 (no GPU)
GPU: no -- expect the button press to take about 1 minute (small.en, no GPU here) ; Whisper (small.en) preloaded and ready.   cell 2 14.8 s
cell 8: display_data text/html <audio ...> then stdout "Press. One. Button." then display_data <div style="font-size:160%"> <h1>Meeting Minutes</h1> ...   47.8 s  (V1)
cell 10: text-only fallback, 0.5 s, same minutes

# Stuck? notes, five real mistakes (see table); restart/skip cases:
cell 28 alone -> ModuleNotFoundError: hands_on   | cells 2,26 -> NameError: np (V5) | cells 2,33 -> OK | cells 2,35 -> NameError: transcript
cells 2,37 -> NameError: vocab | cells 2,59 -> NameError: run_pipeline | cells 2,50 -> NameError: hands_on

# model 0 under the knob, "Sure." (prediction by hands_on._member_predict)
knob -6: alone action .54 | after request action .44, disc .35 | after Friday proposal action .67
knob  0: alone action .54 | after request discussion .94 | after Friday proposal discussion .99
knob +6: alone action .54 | after request discussion .98 | after Friday proposal discussion 1.00
```

Not verified (not possible here): Colab and a T4 (including the real press time and the first-run model download), the microphone recorder, the live skit, projector audio, and how anything sounds.

## To get to PASS

Fix V1 (split the backup cell) and V2 to V6 (all short text or one-line changes); V7 to V12 can ride along. Then I will re-run the notebook from both zips and read cells 8 to 10 of the hook notebook and cells 26, 30, 33, 37 and 62 of the participants' notebook.
