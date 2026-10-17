VERDICT: CHANGES

Final review of `cecf3da` (parent `198fee6`; base `main` = `3736eaf`), in a detached worktree. IDs claimed: N9, the integration items (bundle, zip-only run, minutes table), `progress.md`, `debug_cheatsheet.md`, the handoff move to `plan/handoffs/`, and the WP3 carry-over. I also did the whole-branch pass against `main` that you asked for.
Everything was re-run by me, on CPU only (no GPU, no Colab).

Short version: the branch is sound. Nothing frozen is touched, both zips run from empty folders, no requirement D1-D6 or N1-N10 has regressed, and the new N9 cell works on every ordinary input. There are two small should-fix items in the N9 cell itself, both quick: a leftover duplicate own-meeting cell, and a raw traceback when the participant labels only some of their lines (the most likely slip). Everything else is a nit. I would merge after those two are fixed.

## Findings

| # | Severity | ID | File:line | What is wrong | How to fix |
|---|---|---|---|---|---|
| 1 | should-fix | N9 | `notebook/audio_workshop.ipynb` cell 52 | The old unlabelled own-meeting cell (`MY_MEETING = '''Sam: Should we move the meeting to Friday? ...'''`) is still there, right after the new N9 cell 51, with no heading. It uses the same variable name `MY_MEETING`, overwrites the globals `turns`, `output` and `turn_ends`, and prints the old minutes. In the 15-minute hands-on a participant will see two cells that both say `MY_MEETING = ...` and may edit the wrong one. In the old (unlabelled) cell their edits would run, but none of the new checks (labels, mismatches, unknown words, the prompt) would. | Delete cell 52, or keep it under a clear heading ("Quick version, no labels") with a different variable name. |
| 2 | should-fix | N9 | cell 51 (`load_text` call) | If the participant labels only some of their lines, the cell crashes with a traceback: `ValueError: C:\\Users\\...\\tmpxxxx.txt: label every line (act / dec / dis) or none of them; 2 of 3 lines have one`. The same happens for a line with no `Name:` (`... line 2: expected 'Speaker: text', got "dis Let's start the meeting."`) and for an empty text (`... has no 'Speaker: text' lines`). The message itself is clear and the intro tells people to label every line, but a stack trace plus a temporary-file path is not a friendly message for a participant who just made the most natural mistake. | Wrap the `load_text(...)` call in `try: ... except ValueError as e:` and print the message with the file path removed (for example `str(e).split(": ", 1)[-1]`), then stop that cell cleanly (`raise SystemExit` is not appropriate in a notebook; skip the rest with an `else:` block or a small helper function). |
| 3 | nit | N9, honesty | cell 51 | "The model knows 553 words" is a typed constant in the print statement. It is true (I checked `len(load_trained_model()["vocab"])` = 553), but the Honesty rule is that printed numbers come from a command. | `print(f"The model knows {len(load_trained_model()['vocab'])} words; ...")`. |
| 4 | nit | N9 | cell 50 | "the model was tested on 17 held-out club and CCA meetings": the 17 include `startup_standup`, `hostel_floor_2`, `waiting_room`, `coding_bootcamp` and similar, which are not clubs or CCAs. The 0.85-0.88 figures match my own count (0.879 and 0.852 accuracy), only the description is loose. | "17 held-out meetings (clubs, projects, group work)". |
| 5 | nit | N9 | cell 51 | The unknown-word list also includes names (`lee`, `amy`, `b`), which is correct (they are unseen) but may confuse participants into thinking names matter. The model never reads speaker names (they are dropped before it sees the text). | Optional: add a half-sentence to the intro: "names are ignored by the model; they show up here only because they are not in the vocabulary". |

## The WP4 checks you listed

| Check | Result |
|---|---|
| **N9 cell** | Pass apart from findings 1-2. Cases I ran by executing the notebook's real cell 51 with different text: (a) the default labelled example: model labels printed beside mine, the one mismatch flagged with `<- X` (the "venue is confirmed" line, which the model calls discussion), minutes shown, unseen words listed, the 553-word vocabulary stated, and the "which EARLIER line should it have remembered?" prompt printed. (b) 5 unlabelled lines: model labels only, plus "No labels given -- add act/dec/dis at the start of every line above and run again". (c) lines full of invented words: ten unseen words listed, two mismatches flagged. (d) 3 labelled lines and 8 unlabelled lines: both fine (the cell does not insist on exactly 5). (e) blank lines in the middle: ignored, fine. (f) emoji and heavy punctuation: fine. (g) partly labelled, (h) no speaker name, (i) empty: ValueError tracebacks (finding 2). (j) a nonsense label word (`foo Sam: Hello.`): treated as a speaker called "foo Sam", no error, no flag (harmless). (k) one speaker only: works. The Agenda note ("subject to the training and whether the prototype handles other topics") is answered honestly in cell 50: 0.85-0.88 held out, a small vocabulary, unusual words read as `<unk>` and ignored. |
| **Headers vs the Agenda blocks and minutes** | Pass. The contents table and every header carry the block and minutes: section 1 about 6 + section 2 about 4 = block 2 (10); section 3 = block 3 (15); Hands-on A = block 4 (10); why memory fades = block 5 (7); section 4 = block 6 (20); teaching it = block 7 (10); Hands-on B = block 8 (20); section 5 + own meeting = block 9 (15). That is 107 notebook minutes, and with the hook (8) and wrap-up (5) it sums to the 120 minutes of `plan/SPEC.md`. Measured compute (my unsolved run, 80 s in total): sections 1-2 about 58 s, every other block under 1 s. Nothing overruns. |
| **`progress.md`** | Pass. One new entry ("Sun 18 Oct 2026 -- WP2-WP4"), 8 lines added. The one changed line is in the earlier WP1 entry (`HANDOFF-WP1.md` to `plan/handoffs/HANDOFF-WP1.md`), a path change only. `check_frozen.sh` flags it as "additive-only but 1 line changed": that is the approved handoff-move edit, nothing else. The entry's claims match what I saw (zips of 17 and 20 files, `run_tests.py` 12 passed, 0 failed, 1 skipped). |
| **`debug_cheatsheet.md`** | Pass. Additive only (8 added, 0 deleted). Each new row describes something that really happened: the `NameError` after skipping sections 1-2 (I reproduced it in WP3), the two zips (participants' versus facilitator's), and the embedded-newline syntax error in generated cells. |
| **Handoff move** | Pass. `HANDOFF-WP1.md`, `-WP2.md`, `-WP3.md` moved to `plan/handoffs/` with 0 lines changed, `HANDOFF-WP4.md` added there, and no `HANDOFF*` file is left in the root. I grepped every `.md`, `.py`, `.ipynb` and `.txt` for `HANDOFF-WP`: every reference outside `plan/handoffs/` now uses the new path (`progress.md`, `demo/ANSWER_KEY.md`, `demo/ATTEMPTS.md`, `demo/FACILITATOR.md`, `demo/sure_exchanges.txt`, `notebook/make_noisy_clips.py`, `plan/PROMPTS.md`, `plan/WORK_PACKAGES.md`), and none is broken. |
| **WP3 carry-over (cell 39 wording)** | Pass. It now says the LSTM figures come from `run_tests.py` check k and the keyword figures from "a per-meeting count over `holdout_data.py`, not printed by any existing script". That was the one nit left from WP3 round 2. |

## Whole-branch pass against `main` (`3736eaf`)

- **Frozen check:** `bash reviews/check_frozen.sh cecf3da`: **no frozen file changed** and `prototype/` has no change at all (`git diff --stat main cecf3da -- prototype` is empty). The script's FAIL lines are exactly the approved exceptions, nothing more: `.gitignore` (one added line, `notebook/facilitator_files.zip`) and the `plan/` edits. I read the `plan/PROMPTS.md` and `plan/WORK_PACKAGES.md` diffs: three lines in the first and one in the second, every one a `HANDOFF-WP*.md` to `plan/handoffs/HANDOFF-WP*.md` path change, no wording changed. `plan/SPEC.md` and `plan/RULES.md` are untouched. The `plan/handoffs/` files are the three renames plus the new WP4 handoff.
- **Participants' zip from an empty folder (17 files, 2.8 MB: no `hands_on_reference.py`, `rnn_ctc_demo.py`, `demo.py`, `demo/` or `ANSWER_KEY.md`):** the notebook with both YOUR-TURN cells unsolved ran all 54 cells, no errors, 87 s. The same notebook with Part A and Part B solved inline also ran, no errors, 91 s. Solved output: Part A `check ... OK`, Part B `check: OK`, knob sweep 10/45 at -6, 35/45 at 0, 22/45 at +6.
- **Facilitator notebook from an empty folder holding only `facilitator_files.zip` (20 files, 3.7 MB):** the real cell 2 then cell 8 (with `WHISPER_MODEL=small.en` for speed) ran with no error and printed both decisions and both action items with Ravi and Mei as owners.
- **`run_tests.py`:** 12 passed, 0 failed, 1 skipped. Leak check exits 0.

## Every requirement at the tip

| ID | Still holds at `cecf3da`? |
|---|---|
| D1 | Yes. Script and two-voice backup WAV unchanged since the WP1 PASS (`demo/script.txt`, the WAV and `prototype/` untouched since). |
| D2 | Yes. `FACILITATOR.md` still has the room calling out items before the button; the pacing wording is the WP2-fixed version. |
| D3 | Yes. `python prototype/demo.py` prints both decisions and both action items from text; the facilitator notebook's backup cell runs from its own zip. |
| D4 | Yes (facilitator page unchanged apart from path edits). |
| D5 | Yes. `demo/sure_exchanges.txt` differs from `main` only in one comment path. |
| D6 | Yes. The preparation check is the run list above. |
| N1 | Yes. Cells 5-10: waveform, spectrogram, forwards and backwards clip, MFCC on the same short clip. |
| N2 | Yes. Four clips regenerate byte-identically; the live key printed ideal 100, hiss 52, poor mic 35, second voice 61 on `base.en`, as in WP2 round 2. |
| N3 | Yes. Words-only section 3 with the symbol and word forms and the stepping cell. |
| N4 | Yes. Part A unsolved prints "not written yet"; solved prints OK. |
| N5 | Yes. Graph labelled an illustration; `0.49^47 = 0.0000%   0.95^47 = 9.0%`. |
| N6 | Yes. Gates, `c = f·c_prev + i·g`, hourglass table plus bar chart, `<eot>`. |
| N7 | Yes. Loss example and keyword vs LSTM (12/18 vs 17/18) with the best-case caveat. |
| N8 | Yes. Knob sweep at -6, 0 and +6 with both the toy result and the trained score; "keep a lot is not better" text intact. |
| N9 | Mostly. Works, subject to findings 1-2. |
| N10 | Yes. No `rnn_ctc`/CTC reference anywhere in the notebook; Whisper cells and the words-only RNN/LSTM sections stay separate. |

No requirement has regressed.

## Commands I ran, with output

```
$ bash reviews/check_frozen.sh cecf3da
frozen: none changed
FAIL outside list: .gitignore, plan/PROMPTS.md, plan/WORK_PACKAGES.md, plan/handoffs/HANDOFF-WP1..4.md     (all approved)
FAIL progress.md is additive-only but 1 line(s) ... changed (added: 8)    (the approved WP1-entry path edit)
debug_cheatsheet.md: additive only (added 8, deleted 0)
$ git diff --stat main cecf3da -- prototype        -> (empty)

$ python notebook/make_bundle.py ; git status --short
workshop_files.zip (2.8 MB, 17 files)   facilitator_files.zip (3.7 MB, 20 files)       (git status clean)
$ python notebook/make_noisy_clips.py ; git status --short     -> clean

$ nbclient, folder with only workshop_files.zip + notebook, unsolved      54 cells, 0 errors, TOTAL 87.0 s
$ nbclient, same, Part A and Part B solved inline                          54 cells, 0 errors, TOTAL 90.8 s
   [28] check (MEMORY_ON = True): OK    [43] check: OK, FORGET_KNOB = +0 ... 35/45 (0.78)    [46] FORGET_KNOB = -6 ... 10/45 (0.22) (and +6 22/45)
   [51] default N9 example: mismatch flagged on "Lee: The venue is confirmed for Friday."   553 words   prompt printed
   [52] the leftover old own-meeting cell prints the old minutes (finding 1)

$ facilitator_files.zip only, real hook_demo cells 2 then 8, WHISPER_MODEL=small.en
Decisions: Hold the showcase on the 21st (agreed) / Get black t -shirts (agreed)
Action Items: Book the hall by Friday -- Ravi (due Friday) / I'll send the order form to everyone tonight. -- Mei (due tonight)

$ python prototype/run_tests.py       12 passed, 0 failed, 1 skipped in 17.0 s
$ python prototype/training_data.py   exit 0
$ python prototype/pipeline.py --compare --text demo/script.txt     2 decisions, 2 action items, matching demo/ANSWER_KEY.md
$ python prototype/demo.py             same four items, boxed
$ python -c "len(load_trained_model()['vocab'])"                     553
```

N9 edge-case harness (`n9_cases.py`, runs the notebook's real cell 51 with the text swapped in):
```
a labelled default (mismatch)   -> flagged, minutes, unseen words, 553, prompt
b unlabelled 5                  -> model labels only + "No labels given ..."
c unseen words                  -> amy, blorp, frobnicate, gadget, gizmo, quokka, snorkel, synergy, wibble, zyzzyva listed; 2 mismatches
d 3 lines / d2 8 lines          -> fine
e blank lines inside            -> fine
f emoji and punctuation         -> fine (my first error here was my own console printing, not the cell)
g partly labelled               -> ValueError: ... label every line (act / dec / dis) or none of them; 2 of 3 lines have one
h no speaker name               -> ValueError: ... line 2: expected 'Speaker: text', got "dis Let's start the meeting."
i empty                         -> ValueError: ... has no 'Speaker: text' lines
j nonsense label word           -> no error (read as part of the speaker name)
k one speaker only              -> fine
```

Not verified: Colab, a T4, the microphone cells of the facilitator notebook, and any audio playback.

## To get to PASS

Fix findings 1 and 2 (delete or rename cell 52; catch the `ValueError` in cell 51), and ideally 3-4. I will then re-run the notebook from the zip-only folder and the three bad-input cases.
