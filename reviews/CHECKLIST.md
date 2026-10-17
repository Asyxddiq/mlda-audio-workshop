Reviewer branch: claude/review-workshop-gaps-53e548
(plan files call it `review/workshop-gaps`; this is the branch the app actually gave this worktree)
Builder branch per plan: `build/workshop-gaps` (the app's name may differ; I review the commit hash I am sent, not a branch name)

# Reviewer checklist, Workshop 4 demo + notebook

Sources: `plan/SPEC.md`, `plan/RULES.md`, `plan/WORK_PACKAGES.md`, and the two PDFs in `reference workshop docs/`
("Agenda" = WS 4 Agenda Proposal, "Programme" = Programme and Slide Plan). The PDFs win over the repo.
Verdict files go in `reviews/WPn.md`, first line `VERDICT: PASS` or `VERDICT: CHANGES`, then a findings table
(severity blocker / should-fix / nit, ID, file:line, what is wrong, how to fix), then commands run with their output.

## Setup, done once per review (every WP)

Let `$H` be the builder hash I was sent and `$SCRATCH` my scratchpad directory.

```bash
git worktree add --detach "$SCRATCH/rev-$H" $H         # never review in the builder's own worktree
bash reviews/check_frozen.sh $H                          # S1, from my own worktree (reviews/ lives on my branch)
cd "$SCRATCH/rev-$H" && python prototype/run_tests.py   # S2
python prototype/training_data.py                        # S3 leak check (RULES rule H)
```

| # | Check | Pass looks like |
|---|---|---|
| S1 | `bash reviews/check_frozen.sh $H` | Exit 0, "frozen: none changed", "outside allowed list: none", additive-only files report 0 deleted lines. Any exit 1 = blocker. |
| S2 | `python prototype/run_tests.py` in the detached worktree (I run it, not copy the handoff) | Last summary line says 0 failed. Record the exact pass/fail counts and compare with `main` (run the same on `main` once) so a drop is visible. |
| S3 | `python prototype/training_data.py` | The leak check passes. If any demo sentence appears in training data, blocker (Rule H). |
| S4 | `HANDOFF-WPn.md` exists, names the commit hash, lists the IDs covered, commands run with real output, anything not done | The hash in the handoff equals the hash I was sent, and `git rev-parse $H` matches. A claim of an ID with no evidence is a finding. |
| S5 | Commit hygiene: `git log main..$H --format='%h %s%n%b'` | Small commits; messages carry the attribution line the harness gives. No merge from or into main. Never pushed (I cannot check; I note it if `git branch -r` shows a builder ref). |
| S6 | Cross-OS: grep the new code for `C:\\`, `\\\\` paths, absolute user paths, `os.startfile`, `winsound`, `pyttsx3`/`sapi` imported unconditionally in files that are in the notebook bundle or run on Colab | A cell/script that only works on the builder's machine = should-fix (blocker if it is in the notebook or the bundle). Windows-only code is fine only inside the optional backup-WAV generator, clearly labelled, with the committed WAV as the thing participants use. |
| S7 | Any number printed in a notebook cell, `demo/` file or doc: grep the file for numbers, find the command that produced each, re-run it | Every figure reproduces (or is a stated, explained constant such as 0.49 / 0.95 / 47). A number I cannot reproduce = blocker (Honesty rule 4). |

## Honesty checks (RULES.md "Honesty" + Rule H), applied to every WP

| # | Check | Pass looks like |
|---|---|---|
| H1 | Retention graph / any `r^n` text: `grep -rniE "retention\|r\^n\|0\.49\|0\.95" notebook demo` | The cell and the caption say "illustration" and that it is not a measured property of the trained system (Programme p.2 note). It must not claim the trained LSTM keeps 95% or the RNN keeps 49%. |
| H2 | `grep -rniE "remember.*(further\|longer\|farther)\|long.range\|long.term" notebook demo` and read every hit | No claim that the trained LSTM remembers further back than the RNN. Long-memory claims attach to the toy experiments (N4/N8), not the classifier (SPEC tension 3; README found no clear RNN/LSTM difference). |
| H3 | `grep -rniE "keep a lot\|keep lots\|better memory\|more memory" notebook demo` | "Keep a lot" is never framed as "better". Both facts shown: toy result (high forget bias keeps first word) AND trained-model score drop at the extremes (SPEC tension 2, 35/45 at 0 baseline per SPEC; I re-measure). |
| H4 | Rule H, ordering: `git log --follow --format='%h %ad %s' -- demo/` (and the key file) | The answer-key commit is EARLIER than the first commit that runs the model on the script (look at commit order and handoff attempt log). A key added in the same commit as the first model run or after = blocker. |
| H5 | Rule H, key vs script: `git log -p -- demo/<key file>` | The key was never edited after the first model run to fit the model. Any change to the key after that point = blocker unless the handoff gives a human reason (e.g. typo) and the script still matches. |
| H6 | Rule H, attempts: compare the handoff's attempt table with `git log -p -- demo/<script file>` | Number of script revisions in git = attempts listed in the handoff (attempt no., what changed, result). An unlisted revision, or a rewording outside `labelling_guide.md` = blocker (cherry-picking). Read the final script against `labelling_guide.md` myself. |
| H7 | Rule H, "Sure." pair: read demo output and the "Sure." report | Both exchanges are in the demo input in the exact words of the Agenda ("Can you book the room?" / "Sure."; "Shall we move it to Friday?" / "Sure."), and the model's label for each is reported verbatim from the run. If the model gets one wrong it is presented as a finding for N7/N9, not reworded and not hidden. I re-run the model on those lines myself. |
| H8 | Rule H, facilitator disclosure: read `demo/FACILITATOR.md` | States plainly that the script was chosen after seeing how the model reads it. |
| H9 | No training-data contamination: `git diff main..$H --stat -- prototype/training_data.py prototype/scorecard_data.py prototype/holdout_data.py prototype/trained_lstm.npz` and `grep -rn "<each demo sentence>" prototype/*_data.py` | Empty diff; no demo sentence found in any `*_data.py`. |
| H10 | Claims vs outputs: every "works", "accuracy", "N/N correct", "X% " in handoff or docs | Backed by an output I reproduced. Where I get a different number, the finding states both. |
| H11 | Timings: for each notebook section, compare handoff's minutes to the SPEC timetable | Section minutes fit the block (2:10, 3:15, 4:10, 5:7, 6:20, 7:10, 8:20, 9:15), counting real run time of each cell I timed plus stated talk time. Overrun = should-fix, blocker if the hands-on blocks (4, 8, 9) cannot be done in time. |
| H12 | Scope: `git diff --stat main..$H` shows nothing for slides, Kahoot quiz file, take-home, worksheet, README | Out-of-scope files only appear if on the allowed list. (README is not on the allowed list, so S1 already catches it.) |

## WP1: The demo (D1-D6)

Files expected: `demo/` (script, answer key, backup WAV, `FACILITATOR.md`, a README or doc on how the WAV was made), `prototype/demo.py`, one notebook cell calling it, "Sure." report, `HANDOFF-WP1.md`.

| ID | Check I will run | Pass looks like |
|---|---|---|
| D1 | Read the script and answer key. Count spoken words (`wc -w`) and play/measure the WAV: `python -c "import wave;w=wave.open('demo/<f>.wav');print(w.getnframes()/w.getframerate())"`. Check speaker labels. | Two speakers; mock CCA meeting (a school club/CCA, not a corporate one); about 1 minute spoken (WAV between roughly 45 s and 90 s; text roughly 120-220 words); a backup WAV exists; the method for making it is documented and reproducible. I listen to / spot-check the WAV (two voices distinguishable, no truncation). |
| D2 | Read `demo/FACILITATOR.md` order of steps | Step order is: play/act the meeting, room calls out decisions and to-dos on their own, THEN press the button. The page says the answer key (their list) is not shown or run before the room answers. Agenda 1: "let the room call out the decisions and to-dos. Then press one button". |
| D3 | From a clean checkout: run the single default command exactly as documented, e.g. `python prototype/demo.py` with no flags, and again from the notebook cell. Count the commands in the facilitator page. | ONE command, no flags, no editing a file, no extra setup step between "press" and "minutes". Output is large and readable (headed sections, decisions and to-dos visually distinct, owners/deadlines where the model gives them). It works from a different cwd too (`cd / && python <abs path>/demo.py`) and with `KMP_DUPLICATE_LIB_OK` handled in the file, not by the user. |
| D3b | Diff the default run output against the answer key, item by item (I write the comparison table in the verdict) | Plan "Done when": every key item present, no wrong items, from text. A missing or extra item is a blocker unless the handoff records it honestly AND the key is untouched (then it is a Rule H finding, not a pass claim). |
| D3c | WAV path: run `python prototype/demo.py <wav>` (whatever the documented form is) on the backup WAV. Needs Whisper; if it is not installed in my env I say so and still check the handoff's numbers for plausibility rather than passing them on trust. | The handoff reports turns found, word accuracy, and whether the minutes match the key; I reproduce those numbers (word accuracy via an independent WER computed against the script text, not the builder's code). If the WAV result is worse than text it says so. Whisper stays separate from the RNN/LSTM stage (see N10). |
| D4 | Read `FACILITATOR.md` and the demo output | There is an explicit compare step: key / room's list vs generated minutes, with what to do when they differ (Programme slide 03 "Generated minutes compared with audience recall"). The "what to do if different" advice is honest (does not blame the audience, does not tell the facilitator to rerun until it matches). |
| D5 | Read the "Sure." report and the demo input; re-run the model on the four lines | Exact Agenda wording; model labels reported as actually produced; facilitator page frames it as a human exercise ("what does Sure mean each time"), explicitly not what the app is judged on. Reworded to pass = blocker (Rule H). |
| D6 | Run the whole of WP1 myself, in a fresh venv or at least a clean clone-like worktree: S2, D3, D3b, D3c, the notebook cell from the bundle | Everything works "as described" (D1-D4) on my run, not just the handoff's. Anything that needs a manual fix first = finding. |

## WP2: Notebook, sound and the RNN (N1-N5, N10)

Run for every notebook WP (after `python notebook/make_bundle.py`; see "Notebook run protocol" below).

| ID | Check I will run | Pass looks like |
|---|---|---|
| N1 | Execute the notebook; inspect the waveform cell, spectrogram cell, reversed-word cell, Whisper cell. Check axes with `nbformat` / render the PNG outputs. | Waveform plot with labelled time axis; spectrogram with time on x (left to right) and frequency/pitch on y with low at the bottom (check `origin="lower"` or equivalent, not flipped); an audio player for a word played backwards and the same word forwards so order is audible; Whisper output shown as words. Text explains "reads the picture left to right" in the Agenda sense. Agenda 2 also lists audio applications and challenges: a short list/markdown is present (nit if absent, the facilitator talks it). |
| N2 | Open the four clips; compute my own word error rate per clip vs the one sentence (re-implemented, not the builder's function); check the answer key | Four clips: three noisy scenarios + one ideal, SAME sentence; an answer key states which gives the most accurate transcript; the measured word accuracy per clip is printed by a cell and I reproduce it (small Whisper nondeterminism tolerated if the ranking is the same). The ideal clip is not mislabelled (it must not be beaten by a noisy one in my run; if it is, report). Kahoot not built (out of scope) but the four clips + key are there. |
| N3 | Read the RNN section, then run its step cell | The word-reading RNN is explained as a running summary reading one word at a time; the line `h = tanh(W_x . x + W_h . h_prev + b)` appears exactly in that form; the example input is words, not audio MFCCs. |
| N4 | Execute Hands-on A three ways: (a) blank/stub, (b) the correct line from `prototype/hands_on_reference.py`, (c) a deliberately wrong line | (a) check says not done yet, no crash trace; (b) check confirms; (c) check fails with a helpful message. Then set the old-memory term off: the output comparison (memory on vs off) is printed side by side and shows that with it off earlier words stop affecting the output; numbers come from the cell. Fits 10 minutes. If `prototype/hands_on.py` changed, S1 shows additive-only and `hands_on_reference.py` and `run_tests.py` still pass. |
| N5 | Run the retention cell; check the graph | Two curves r = 0.49 and r = 0.95 over 47 steps (x runs to 47, steps labelled). Printed values: 0.49^47 ~ 0 (I compute `0.49**47`) and 0.95^47 ~ 9% (`0.95**47` = 0.0895). The graph/caption says it is an illustration, not measured (H1). Fits 7 minutes. |
| N10 | `grep -nE "mfcc\|rnn_ctc" notebook/audio_workshop.ipynb`; read section order | Whisper cells and the RNN/LSTM section are separate: the downstream model's input is a text transcript/word ids, never Whisper features or MFCC frames. If the MFCC cell stayed it is under N1 as "what Whisper sees" and labelled as not the model's input (SPEC tension 1). The minutes pipeline function takes text/turns, so Whisper can be swapped. |
| N-sec | Read the heads of sections 1-2 | Marked facilitator-run or optional, with minutes stated per section (SPEC tension 5). |

## WP3: Notebook, LSTM and training (N6-N8)

| ID | Check I will run | Pass looks like |
|---|---|---|
| N6 | Read the LSTM section; run it; find the cell-state line, the gate analogy text, the hourglass picture, the `<eot>` explanation | `c = f * c_prev + i * g` appears in the form in the Programme; forget = eraser, input = pencil, output = voice; five stacked layers drawn/described as an hourglass (the model really has 5: I verify against `prototype/lstm_model.py` / the npz shapes, not just the builder's text); `<eot>` explained as "speaker changed" bookmark and shown on a real example turning into a token. |
| N7 | Run the training-in-one-picture cell and the comparison cell; recompute `-log(p)` by hand | Picture or cell goes label -> prediction -> loss -> update; the worked example's numbers are consistent (`-ln(p)` with the printed p; I recompute to 3 decimals; natural log vs log10 stated). Comparison: LSTM vs keyword checklist on lines the model never saw (I confirm these lines are in `holdout_data.py` / not in `training_data.py` via grep). "Venue is confirmed for the 14th" is shown as missed, with the reason (no "agree" / "let's"), and the cell prints whether the model and the keyword checklist actually miss it: if the model in fact gets it right, the claim "it is missed" is wrong and must be re-worded (check the "missed" statement is true for what the cell prints). Counts printed come from the cell. Fits 10 minutes. |
| N8 | Run Hands-on B: stub, correct line, wrong line (as N4). Turn the knob to -6, 0, +6. | Check confirms the correct `c = f * c_prev + i * g`. The knob changes forget bias; the toy two-sentence test shows what the facilitator describes (erase a lot: first word gone; keep a lot: first word shapes the end) AND the trained-model score is printed at -6, 0, +6 (I reproduce the three scores; SPEC cites 35/45 at 0). Text never says "keep a lot = better" (H3); it says both extremes lowered the trained score if that is what the run shows (and is honest if the run shows otherwise). Knob works without editing code outside the cell. Fits 20 minutes. |
| H2 (repeat) | Read the text at the end of N6/N7/N8 | No claim that the trained LSTM remembers further back than the RNN (SPEC tension 3). The classifier needs about one previous turn. |

## WP4: Your own meeting, integration (N9)

| ID | Check I will run | Pass looks like |
|---|---|---|
| N9 | In the notebook from the zip: (a) five ordinary lines about a topic not in training (e.g. a football club) with participant labels; (b) the same with labels left blank; (c) a line full of unseen words; (d) fewer/more than 5 lines; (e) blank line; (f) a line with weird punctuation or emoji | (a) model labels shown next to the participant's; mismatches flagged clearly; (b) works, prompts for labels but does not demand them; (c) the cell LISTS which of the participant's words were unseen (`<unk>`), the 553-word vocabulary is stated (verify 553 with `len(vocab)` myself) (SPEC tension 4); (d)(e)(f) no stack trace, a friendly message; always includes the prompt asking "which earlier line should it have remembered?". Agenda NOTE about topic flexibility is surfaced honestly (says the model only knows 553 words and meeting-style talk). Fits 15 minutes. Participant input is typed in the notebook (a worksheet is out of scope). |
| Bundle | `python notebook/make_bundle.py`, then `unzip -l notebook/workshop_files.zip` | The zip contains every file any cell imports or opens (`prototype/demo.py` if the demo cell calls it, new data files, new WAVs, new modules). Size reasonable (< ~25 MB). If the builder edited `make_bundle.py`, the change is within `notebook/**`. |
| Empty-folder run | `mkdir "$SCRATCH/empty" && cp notebook/workshop_files.zip notebook/audio_workshop.ipynb "$SCRATCH/empty"`; `python -I -m jupyter nbconvert --to notebook --execute` from that folder (copying only the .ipynb and zip) | Every cell runs top to bottom with no manual intervention other than the cells designed for input, in a folder holding only the zip and the .ipynb. No reference to `../prototype` or any path outside the folder; no `FileNotFoundError`. Cells that need the network (Whisper model download, pip) say so and the failure is graceful. Record total wall time and per-section time. |
| Re-run | Execute the notebook twice in the same folder | Second run still works (unzip cell is idempotent). |
| Timetable | The table of minutes per section vs the SPEC timetable | A table exists; sections sum to the blocks (10/15/10/7/20/10/20/15) and I checked its run-time figures against my own timings (H11). Sections 1-2 minutes stated. |
| Log | `git diff main..$H -- progress.md debug_cheatsheet.md` | `progress.md`: exactly one new log entry, nothing existing deleted. `debug_cheatsheet.md`: new rows only, each corresponding to a failure I can find in the build history; none deleted/edited. |
| Final | Re-run S1-S3, H1-H12 over the whole branch, not just the last commit | All pass, so the merge candidate is the commit I name in the verdict. |

## Notebook run protocol (WP2-WP4)

1. `python notebook/make_bundle.py` in the detached worktree (fresh zip, not the builder's committed one; also `git diff` the zip's listing against the committed zip if one is committed).
2. New empty folder holding only `workshop_files.zip` and `audio_workshop.ipynb`. Run with `KMP_DUPLICATE_LIB_OK=TRUE` set by the notebook, not by me.
3. Time each section; note cells that need input and how I drove them.
4. Inspect outputs for tracebacks, `nan`, stale outputs saved in the committed .ipynb that do not match a fresh run (committed outputs should be cleared or reproducible).
5. For every claim in a markdown cell about a number or behaviour: find the cell output that backs it.

## Findings severity

- **blocker**: frozen file touched, key/script cherry-picking, claim with no reproducible number, notebook cell fails from the empty folder, wrong equation, retention/LSTM claims violating Honesty, default demo misses a key item without honest disclosure.
- **should-fix**: timing overrun, missing facilitator guidance, non-portable code outside the optional generator, unclear instructions, missing minutes per section.
- **nit**: wording, formatting, unused imports.
