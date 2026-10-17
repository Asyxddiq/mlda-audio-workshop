VERDICT: CHANGES

Reviewed commit `efa5fd000e73396f363242b35656d9189344d3fd` (parent `6b7424c`, base `main` = `0a67bbc`), in a detached worktree. Requirement IDs claimed: D1-D6 plus the orchestrator's WP1 additions (W1a-W1f in `reviews/CHECKLIST.md`).
Everything below was re-run by me, not taken from `HANDOFF-WP1.md`. Whisper ran on CPU only (no GPU here), so no Colab / T4 timings and no live-mic test.

Short version: the text demo, the backup-WAV demo, the frozen-file check, `run_tests.py` and the "Sure." report are all correct and reproduce. Four blockers are left, all cheap to fix: a fallback cell that crashes, a live skit that will silently merge speakers, the PM attempts not listed, and a Rule H claim that overstates what happened.

## Findings

| # | Severity | ID | File:line | What is wrong | How to fix |
|---|---|---|---|---|---|
| 1 | blocker | D3, D6, W1b | `demo/hook_demo.ipynb` cell 8 (the backup cell) | The backup cell uses `run_demo`, `Markdown` and `display`, which are imported only in cell 6 (the live-recording button). The notebook's own text says to use cell 8 "if the recording fails". If the recording or cell 6 never ran, cell 8 raises `NameError: name 'run_demo' is not defined`. I reproduced it: setup cell, then cell 8 in a fresh interpreter, gives that exact error. The fallback therefore fails in the one situation it exists for. | Put `from demo import run_demo` and `from IPython.display import Markdown, display` at the top of cell 8 (or in the setup cell). Re-run the setup cell then cell 8 from a fresh kernel to confirm. |
| 2 | blocker | D1, D3, D6, W1a | `demo/script.txt:9`, `demo/FACILITATOR.md` (whole page), `demo/hook_demo.ipynb` cell 0 | The live skit is the primary delivery, but nothing tells the actors to leave a gap between turns. `script.txt:9` says "normal pace". The demo splits speakers on `pause_gap=1.5`, so a skit at normal pace has about 0.3-0.5 s between speakers, below 1.5 s, and Whisper will merge the turns into one speaker. The builder's own ATTEMPTS #6 shows a 1.2 s gap merged turns (10 said, 8 heard), and the WAV had to go to 2.0 s. The live-skit path was never run (handoff "Not done" says so), so D6's "works as described" is unproven for the delivery mode Asy chose. | State in `FACILITATOR.md` (step 2) and the `script.txt` header that actors must leave a clear pause of about 2 s between speakers and speak one sentence per turn. Replace "normal pace". Ideally record two real people (phone is fine) and run `python prototype/demo.py <file>`, then report the result. If that is not possible before the event, say plainly in `FACILITATOR.md` that the live path is untested and the backup WAV is the safe route. |
| 3 | blocker | W1f, H6 | `demo/ATTEMPTS.md` ("The showcase / T-shirt script" table, first row); `HANDOFF-WP1.md` | Rule H requires every attempt listed with number, what changed and result. The PM's four attempts are not listed. The table row says only "orchestrator's 4 attempts (not re-run; see its message)". The reader of the repo cannot see what was tried and discarded, so the script's selection cannot be audited. | Add a "PM attempts" table (4 rows: attempt number, exact wording or change, model result) supplied by the orchestrator, marked "reported by PM, not re-run here". Same table in the handoff. |
| 4 | blocker | H4, H5, H8 (honesty) | `progress.md` (new WP1 entry, 2nd bullet); `demo/ANSWER_KEY.md:5-8`; `demo/FACILITATOR.md:3-7` | Wrong claim about Rule H order. `progress.md` says the script and key were "written and committed before the model was run on it, per Rule H". That is true only of the superseded bake-sale script (`6b7424c`). The final key and script first appear in `efa5fd0`, together with every result (`git log main..efa5fd0 -- demo/script.txt demo/ANSWER_KEY.md` shows `6b7424c` and `efa5fd0` only), and the PM model-checked the script during drafting. `ANSWER_KEY.md` says "before this session re-ran the model", which is literally true but implies more than happened. The same `progress.md` bullet also says the script "changed once", while ATTEMPTS shows the builder's own three rewordings of the first script. The key content itself is a plain human reading and I found nothing in it that was bent to fit the model, but the record must not claim an ordering it does not have. | Reword `progress.md` and the `ANSWER_KEY.md` header to say: the first script and key were committed before any model run; the final PM script and key were drafted by the PM with model feedback, and the key was fixed from a plain reading. In `FACILITATOR.md` state plainly that the script was chosen after the PM watched how the model reads it, and note what that means: both decisions are phrased as agreement to a proposal ("Sounds good"), which scored 0.77 and 0.60 for `decision` in the builder's re-derived check, and plain-fact decisions such as "Venue is confirmed for the 14th" are the known weak spot and are not in the demo. |
| 5 | should-fix | D3, D4 (your point 1) | `demo/FACILITATOR.md` (step 5), output of `prototype/demo.py demo/backup_meeting.wav` | The on-screen minutes for the audio paths read "Get black t -shirts (agreed)", while the text path reads "Get black T-shirts". Reproduced with both `medium.en` and `small.en` through `demo.py`, and my independent transcript shows the recogniser writing "t -shirts" 3 times in both models. It is Whisper's rendering, not a model error. It is acceptable for the demo, but the room reads these minutes on screen and the facilitator is not warned. I would not post-process it in `demo.py`: that hides real recogniser output and would be a demo-only fix. I would also not reword the script just for this, since that reopens Rule H and the WAV. | Add one line to `FACILITATOR.md` step 5: "Whisper may write 'T-shirts' as 't -shirts'; that is the speech-to-text stage, not the minutes model." That is also a free link to block 2. Also soften the handoff's "same minutes" wording: `audio_check`'s `same_minutes` accepts items that are 80% word-similar (`prototype/audio_check.py:104-116`), so "Minutes match: True" does not mean text-identical. |
| 6 | should-fix | D2, W1d (your point 3) | `notebook/audio_workshop.ipynb` cells 3-4; `notebook/make_bundle.py:21`; `demo/FACILITATOR.md:57-60`; `demo/hook_demo.ipynb` cell 0 | The participants' notebook change works from the zip (see "Empty-folder run" below), but it contradicts the docs and weakens the hook. (a) `FACILITATOR.md` and `hook_demo.ipynb` cell 0 say the participants' notebook has "only a one-line pointer" / "a short cell that points here". It actually has a markdown cell plus a code cell that runs the demo and prints the full minutes. (b) The bundle ships `demo/ANSWER_KEY.md`, `demo/script.txt` and `demo/backup_meeting.wav` to every participant, but no code reads `ANSWER_KEY.md` (grep: only a print message in `prototype/demo.py:115` mentions it). Anyone who unzips before the hook has the answers; the point of D2 is that the room calls them out first. (c) The cell sits straight after Setup and says "the facilitator runs this live at the start", which is a handout cell telling participants to run it themselves. | Choose one, and make the docs match: either move the cell to after the hook (for example "Run this after the hook, never before") with a note that it is optional, or remove it and keep `hook_demo.ipynb` as the only runner. Drop `ANSWER_KEY.md` from `DEMO` in `make_bundle.py`, or accept it and state that the facilitator hands out the zip only after the hook (if kept, change the message at `prototype/demo.py:115` so it does not point at a file the participants lack). Fix the "one-line pointer" wording. |
| 7 | should-fix | D1 | `demo/script.txt` | D1 asks for a "~1-minute" mock meeting. The script is 68 spoken words (about 25-30 s of speech). The WAV is 53.5 s only because of nine 2.0 s silences (`wave`: 53.45 s). A live skit with the pauses required by finding 2 would land around 45-55 s, so it is roughly right in practice, but nothing says so. | Say in `FACILITATOR.md` that the skit runs about 50 s with the required pauses. Do not reword the script just for length unless you accept re-doing the Rule H attempt log. |
| 8 | nit | D3 | `prototype/demo.py:47-57` | The banner boxes raw Markdown (`# Meeting Minutes`, `**Attendees:**`, `- [ ]`). It is readable but is not "large". In `hook_demo.ipynb` cells 6 and 8 the minutes are shown twice (banner print, then `display(Markdown(...))`). | Optional: drop the banner in the notebook cells or the `display(Markdown)` line. |
| 9 | nit | D6 (your point 4) | `demo/hook_demo.ipynb` cell 4 | `record_skit()` logic reads correctly (see "Notes on your five points"). Small risks: `capture_output=True` hides ffmpeg's error text, so a failed conversion shows only `CalledProcessError`; the mic stream is never stopped, so the browser's recording indicator stays on. | Print `result.stderr` on failure; call `stream.getTracks().forEach(t => t.stop())` after `recorder.stop()`. |
| 10 | nit | S4 | `HANDOFF-WP1.md` (header, "Files changed") | Says nothing was committed after `6b7424c` and lists files as uncommitted. They are all in `efa5fd0`. | Update the header when the handoff is next touched. |
| 11 | nit | N-hygiene | `notebook/audio_workshop.ipynb` (diff) | The re-save rewrote `·` and `×` in the Part A and Part B markdown cells as `·` / `×` escapes. This renders the same, but it is noise in the diff. | None needed. |
| 12 | nit | W1e | `HANDOFF-WP1.md` | The laptop fallback reports accuracy but no run time. My `demo.py demo/backup_meeting.wav` took 3 min 37 s with `medium.en` on this CPU (while a second Whisper job ran at the same time, so the figure is pessimistic); the facilitator needs to know a laptop on `medium.en` is slow. | Report a time for `small.en` on CPU and add it to the fallback paragraph. |

## Notes on your five points

1. **"t -shirts"**: finding 5. Accept it with a facilitator line; do not post-process, do not reword.
2. **`progress.md` overclaim**: confirmed, finding 4 (blocker, wording only).
3. **Participants' notebook cell**: it does not break the notebook from the zip (I ran its setup cell and the demo cell from a folder holding only the zip and the `.ipynb`, and they print the correct minutes; `from demo import run_demo` resolves to `prototype/demo.py` even though a `demo/` folder exists beside it). It does weaken the hook and contradicts the docs: finding 6.
4. **`record_skit()` / `skit_path`**: read it twice. Order is correct: inject the JS, `eval_js("startRecording()")`, `input()`, `eval_js("stopRecording()")`, decode base64, `ffmpeg -y -i skit.webm skit.wav`. This is the standard Colab MediaRecorder pattern, and it needs the JS display and both `eval_js` calls in the same cell, which they are. `skit_path` is assigned at the last line of cell 4 and is defined whenever recording succeeds. If recording fails, cell 6 raises `NameError`, which is expected and is what the notebook tells the facilitator to expect, but then cell 8 fails too (finding 1). I cannot run it outside Colab, so the live-mic path and `medium.en` preload on a T4 are **unverified by me**, as in the handoff.
5. **`python prototype/demo.py demo/backup_meeting.wav`**: confirmed, from a fresh run, `medium.en` and `small.en` alike. Every key item, owners Ravi (book the hall by Friday, due Friday) and Mei (send the order form, due tonight). One cosmetic difference: "t -shirts" (finding 5). The two action-item lines keep the full sentence ("I'll send the order form to everyone tonight.") rather than the key's "Send the order form to everyone", which is wording only.

## Requirement ID check against the PDFs

| ID | Result |
|---|---|
| D1 | Partly. Two speakers, CCA showcase/T-shirt meeting, live skit plus a two-voice synthetic backup WAV, WAV generator documented (Windows SAPI, two female voices). About 1 minute holds only with the pauses (finding 7). Live skit untested (finding 2). |
| D2 | Pass. `FACILITATOR.md` order: question, skit, "Let the room call out the decisions and to-dos. Don't correct them yet", then press. Matches Agenda 1. The spoiler issue is in finding 6. |
| D3 | Pass for the default and WAV commands (one command, no flags, works from another working directory). The Colab button is one cell, but its backup variant is broken (finding 1). |
| D4 | Pass. `FACILITATOR.md` step 5 plus `ANSWER_KEY.md`; the "if the minutes differ" advice is honest ("the model isn't perfect"), and does not tell the facilitator to re-run until it matches. Matches Programme slide 03. |
| D5 | Pass. Exact Agenda wording in `demo/sure_exchanges.txt`, human exercise on the page, model labels reported as produced. I re-ran it: first "Sure." = `action_item`, second "Sure." = `discussion` (wrong; the guide says `decision`), and the run lists `move` as a word the model never saw (read as `<unk>`). Not reworded and not hidden. |
| D6 | Not yet. Text and WAV paths work as described; the live skit and the Colab-only cells are unverified, and the fallback cell is broken (findings 1-2). |
| W1a | Pass (skit script plus synthetic two-voice WAV, documented). |
| W1b | Partly: one-cell button and preload exist; facilitator-only; fallback bug (finding 1). Not executed on Colab or T4. |
| W1c | Pass. `get_transcript(..., pause_gap=1.5)` (`prototype/demo.py:80`), Speaker1/Speaker2 renamed to Mei/Ravi before labelling (owners in the minutes show Mei and Ravi), Whisper kept separate from the label stage. `transcriber.py` untouched. |
| W1d | Pass. `DEMO` list added to `make_bundle.py`; zip has 18 files and `demo.py` runs from the unzipped bundle. See finding 6 about `ANSWER_KEY.md`. |
| W1e | Pass for correctness (reproduced; see numbers below). No laptop time reported (finding 12). |
| W1f | Fails: PM attempts not listed (finding 3). The key's content is the same in `efa5fd0` as written; the ordering claim is wrong (finding 4). |
| N10 (preview) | Fine for WP1: `demo.py` calls `get_transcript` then `run_pipeline(transcript=...)`; Whisper and the RNN/LSTM stay separate. |

## Commands I ran, with output

```
$ git rev-parse efa5fd0
efa5fd000e73396f363242b35656d9189344d3fd

$ bash reviews/check_frozen.sh efa5fd0
base   : main (0a67bbc06f330a4b068cc48e84e146ede09afb38)
commit : efa5fd000e73396f363242b35656d9189344d3fd
files changed since merge-base:
  HANDOFF-WP1.md  demo/ANSWER_KEY.md  demo/ATTEMPTS.md  demo/FACILITATOR.md  demo/backup_meeting.wav
  demo/hook_demo.ipynb  demo/make_backup_wav.py  demo/script.txt  demo/sure_exchanges.txt
  notebook/audio_workshop.ipynb  notebook/make_bundle.py  progress.md  prototype/demo.py
frozen: none changed
outside allowed list: none
progress.md: additive only (added 8, deleted 0)
RESULT: OK            (exit 0)
```

```
$ python prototype/run_tests.py          (in the detached worktree)
PASS a-k, m (SKIP l: no --audio FILE given)
12 passed, 0 failed, 1 skipped in 16.1 s
$ python prototype/training_data.py      -> exit 0 (leak check passes)
```
The `main` baseline was not re-run; the 12/0/1 summary matches the one printed at the end of `progress.md` for before this work.

```
$ python prototype/demo.py               (default, text; also run from another working directory, exit 0)
Attendees: Mei, Ravi
Decisions:   Hold the showcase on the 21st (agreed) / Get black T-shirts (agreed)
Action Items: Book the hall by Friday -- Ravi (due Friday) / I'll send the order form to everyone tonight. -- Mei (due tonight)
```
Matches `demo/ANSWER_KEY.md`: all four items, no extra items, owners Ravi and Mei.

```
$ python prototype/demo.py demo/backup_meeting.wav          (medium.en, CPU; 3 min 37 s wall time)
Decisions:   Hold the showcase on the 21st (agreed) / Get black t -shirts (agreed)
Action Items: Book the hall by Friday -- Ravi (due Friday) / I'll send the order form to everyone tonight. -- Mei (due tonight)
$ WHISPER_MODEL=small.en python prototype/demo.py demo/backup_meeting.wav
(identical minutes, including "t -shirts")
```

My own word-accuracy check (`get_transcript(wav, pause_gap=1.5)`, reference = spoken text of `script.txt`, my own edit-distance word error rate, hyphens split, punctuation and case ignored; not the builder's `audio_check` code):
```
RESULT medium.en: turns said 10 heard 10; ref words 71; WER 0.014; accuracy 98.6%
RESULT small.en : turns said 10 heard 10; ref words 71; WER 0.000; accuracy 100.0%
```
This agrees with the handoff's 99% and 100%. The medium transcript differs from the script only in "OK" for "Okay" and "t -shirts", and it splits "Sounds good, white gets dirty too fast." into two sentences inside one turn without changing the turn count.

```
$ python prototype/pipeline.py --compare --text demo/sure_exchanges.txt
 1 Mei  Can you book the room?        keywords action_item  trained LSTM action_item
 2 Ravi Sure.                          keywords discussion   trained LSTM action_item   *
 3 Mei  Shall we move it to Friday?    keywords discussion   trained LSTM discussion
 4 Ravi Sure.                          keywords discussion   trained LSTM discussion
Words the model never saw in training (read as <unk>): move
```
Matches the handoff.

```
$ python -c "import wave; w=wave.open('demo/backup_meeting.wav'); print(w.getnframes()/w.getframerate())"
53.45 s (22050 Hz, mono)      script.txt spoken words: 68
```

Empty-folder run (a folder holding only `workshop_files.zip` and `audio_workshop.ipynb`, after `python notebook/make_bundle.py` gave 18 files, 2.6 MB): the participants' notebook setup cell and the new demo cell both run and print the minutes above. I did not run the rest of that notebook; it is unchanged by this commit apart from the escaped characters (finding 11), and the full from-zip run belongs to WP2-WP4.

Fresh-interpreter simulation of the facilitator notebook (zip only; setup cell without the 1.5 GB preload, then the backup cell):
```
NameError: name 'run_demo' is not defined          (finding 1)
```

Not run, and why: `demo/hook_demo.ipynb` on Colab (no Colab here), the browser-mic recorder, the `medium.en` preload on a T4, and a human-recorded live skit. All are stated as untested in the handoff, and none is verified here.

## To get to PASS

Fix findings 1-4 (all small), and ideally 5-7. Then send a new commit hash. I will re-run the empty-folder `hook_demo` fallback, `demo.py` on text and WAV, `run_tests.py`, and check the attempts table and the wording of `progress.md` / `ANSWER_KEY.md` / `FACILITATOR.md`.
