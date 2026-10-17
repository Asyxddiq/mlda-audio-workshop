# HANDOFF-WP2: notebook, sound and the RNN (N1-N5, N10)

**Branch:** `claude/build-workshop-gaps-756dec`.
**Last commit:** `3736eaf` (WP1 round 2, passed review as `3736eaf`). Nothing
committed since -- same protocol as WP1: the orchestrator alone commits.
**Status:** N1, N2, N3, N4 (unchanged, re-verified), N5 and N10 all done in
`notebook/audio_workshop.ipynb`, plus the WP1 carry-over fix below. Not
touched: notebook sections 4 and 5 (LSTM, meeting minutes -- WP3/WP4), and
no frozen file.

---

## WP1 carry-over (nit, fixed first)

"One sentence per turn" in `demo/script.txt`'s header, `demo/FACILITATOR.md`
step 2, and `demo/hook_demo.ipynb` cell 0 / cell 2 was wrong: turns 1
("Okay, let's start. Two things today...") and 7 ("Next, the T-shirts.
Should we get black T-shirts?") both have two sentences. The script and the
backup WAV already work (TTS happened not to pause long enough inside
either sentence to trip the splitter), so this was a documentation bug, not
a functional one -- but the instruction as written would have told a live
actor to do something the script itself doesn't follow.

Fixed in all three places: "say each turn in one go, with no long pause
inside it ... leave ~2 s before the other person speaks". Re-verified the
script still works after the wording change (wording-only edit, no script
content changed):

```
$ python prototype/pipeline.py --compare --text demo/script.txt
...
## Decisions
- Hold the showcase on the 21st (agreed)
- Get black T-shirts (agreed)
## Action Items
- [ ] Book the hall by Friday -- Ravi (due Friday)
- [ ] I'll send the order form to everyone tonight. -- Mei (due tonight)
```
Still matches `demo/ANSWER_KEY.md` exactly.

---

## Requirement IDs covered

| ID | Requirement | Where |
|---|---|---|
| N1 | Waveform + spectrogram; a word played backwards; Whisper reads the picture, writes words | New cells after the Whisper 3-liner in section 1: waveform/spectrogram plot, forwards/backwards clip, and the moved-and-reframed MFCC cell ("what Whisper-style models actually see") |
| N2 | Noisy-audio comparison: 3 noisy + 1 ideal, same sentence, answer key | New cells at the end of section 2: `notebook/make_noisy_clips.py` generates the 4 clips, a vote cell plays them, an answer-key cell measures `word_accuracy` live |
| N3 | RNN = a running summary, reading words one at a time | Section 3's intro rewritten around words (not audio frames); Hands-on A unchanged |
| N4 | Hands-on A: type the line, check confirms, compare with/without old memory | Unchanged in code, re-verified working |
| N5 | Memory fades: retained fraction = r^n; 0.49^47 ≈ 0%, 0.95^47 ≈ 9%, as a graph, labelled an illustration | New "Why simple memory fades" subsection after Hands-on A, before section 4 (LSTM) |
| N10 | Keep Whisper separate from the RNN/LSTM | Unaffected -- section 3 no longer feeds audio MFCC frames into the RNN at all; the only audio-to-RNN link (the old MFCC-as-RNN-input cell) was removed, not just reframed |

---

## What changed, cell by cell

(Cell numbers are the FINAL notebook's, 0-indexed, after all WP2 edits.)

**Section 1 (`## 1. Audio to text with Python`), cell 3:** added a
facilitator-run / minutes note (see "Sections 1-2 timing" below). Cell 4
(the Whisper 3-liner) unchanged.

**New cells 5-10, right after the Whisper 3-liner:**
- 5-6: "What Whisper is reading" -- waveform + spectrogram of
  `sample_short.wav` (`matplotlib`).
- 7-8: "Order matters" -- a ~0.5s clip played forwards then backwards
  (`IPython.display.Audio`).
- 9-10: "What Whisper-style models actually see" -- the MFCC cell, MOVED
  here from its old spot in section 3, reframed (same code, new
  surrounding text).

**Section 2 (`## 2. Where transcription struggles`), cell 13:** added a
facilitator-run / minutes note. Existing noise/poor-mic/accent cells
(14-20) unchanged.

**New cells 21-24, after the accent cell, before section 3:**
- 21-22: Kahoot vote -- plays the four clips (`NOISY_CLIPS`), asks for a
  vote before the next cell.
- 23-24: answer key -- measures `word_accuracy` for each clip live and
  prints the ranking. Not hardcoded: this is a real command that runs
  every time the notebook runs.

**Section 3 header + first cell (now cells 25-26), rewritten:**
- Old: "Sound is a long list of numbers... cut it into frames and read one
  frame at a time... That is an RNN," followed by an MFCC-on-audio cell.
- New: "When you read a sentence, you don't reread every word from
  scratch... An RNN reads one **word** at a time," with a bridging
  sentence back to Whisper's left-to-right spectrogram reading, followed by
  a cell that tokenizes one sentence with `lstm_model.model_words` into
  words, then into numbers via a vocabulary -- "words become numbers,"
  with no audio or RNN-on-audio-frames anywhere in this section (N10).

**Hands-on A (cells 27-29):** unchanged, code untouched. Re-verified it
still runs (see `run_tests.py` below, check f).

**New cells 30-31, after "break the memory", before `## 4. LSTMs`:**
"### Why simple memory fades" -- broken-telephone framing, labelled
explicitly as an illustration, not a measured property of the trained
model (which stacks 5 LSTM layers, not one RNN step); a graph of `r^n` for
r = 0.49 and r = 0.95 over 47 steps, with the exact numbers printed.

Sections 4 and 5 (LSTM, meeting minutes) and everything in them: untouched.

---

## Sections 1-2 timing (facilitator-run, Agenda block 2, 10 minutes total)

Agenda block 2 ("Turning sound into words") is a single 10-minute talk
covering N1 and N2 together, with no hands-on slot (SPEC tension 5). Split
roughly 6 min / 4 min between the notebook's sections 1 and 2 in the
markdown -- an estimate, not a measured rehearsal time, and said so in the
notebook text itself.

---

## New-cell CPU timings (from the full notebook run below, `base.en`, no GPU)

| Cell | What | Time |
|---|---|---|
| Waveform + spectrogram plot | matplotlib, one audio file | 3.2s |
| Forwards/backwards clip | just builds `Audio` widgets, no computation | 0.0s |
| MFCC cell (moved) | `audio_features.mfcc` on the 60s sample meeting | 0.4s |
| Kahoot vote cell | prints + builds 4 `Audio` widgets, no Whisper | 0.4s |
| Kahoot answer-key cell | 4 Whisper transcriptions (~15s clips each) | 13.2s |
| Tokenizer cell (words -> numbers) | pure Python, no model | 0.0s |
| Retention graph | matplotlib + two numpy power arrays | 0.4s |

On a GPU (Colab T4) the Whisper-dependent cells (Kahoot answer key) would
be faster; these numbers are CPU, as asked.

---

## Full end-to-end run, empty folder with only the zip

Per `plan/RULES.md`'s reviewer checklist ("every cell run from an empty
folder holding only the zip"), re-verified here with `nbclient` (no Colab
available, but this executes the real notebook file against a real
IPython kernel, cell by cell, the same as "Run All" would):

```
$ python notebook/make_bundle.py
wrote .../notebook/workshop_files.zip (3.7 MB, 21 files)

$ mkdir /tmp/ws4_test && cp workshop_files.zip audio_workshop.ipynb /tmp/ws4_test/
$ python -c "<nbclient script: NotebookClient(nb, timeout=600).execute(), from /tmp/ws4_test, nothing else present>"
```

```
[ 2]    0.3s  code     import os, sys, shutil, subprocess, importlib.util, zipfile ...
[ 4]   22.1s  code     import torch, whisper ...
[ 6]    3.2s  code     import numpy as np | import matplotlib.pyplot as plt ...
[ 8]    0.0s  code     from IPython.display import Audio, display ...
[10]    0.4s  code     from audio_features import mfcc ...
[12]   24.3s  code     os.environ["WHISPER_MODEL"] = MODEL ... (get_transcript on sample_meeting.wav)
[14]    6.5s  code     import numpy as np | from audio_check import word_accuracy ...
[16]    3.2s  code     NOISE = 0.1 ...
[18]    3.3s  code     poor = np.repeat(clean[::4], 4) ...
[20]    0.0s  code     MY_VOICE = None ... (skipped, no file set)
[22]    0.4s  code     NOISY_CLIPS = [ ...
[24]   13.2s  code     for label, filename in NOISY_CLIPS: heard = hear(...) ...
[26]    0.0s  code     from lstm_model import model_words ...
[28]    0.1s  code     import numpy as np, hands_on ...
[31]    0.4s  code     n = np.arange(48) ...
[33]    0.0s  code     from lstm_model import fold_time, unfold_time ...
[35]    0.0s  code     def lstm_cell_update(f, i, g, c_prev): ...
[38]    0.4s  code     from lstm_model import run_pipeline ...
[40]    0.3s  code     MY_MEETING = ''' ...

=== OK in 87.6s ===
```

Every cell ran without error. Outputs checked by hand:
- Waveform/spectrogram cell: produced a PNG (one benign matplotlib
  `RuntimeWarning: divide by zero encountered in log10`, from a silent
  stretch of the clip in the spectrogram's dB conversion -- a known,
  harmless matplotlib quirk, not a bug in this notebook's code).
- MFCC cell: `60 seconds of audio -> 5967 frames x 13 numbers`.
- Kahoot answer key, this run (`base.en`): ideal 100%, background hiss 52%,
  poor microphone 35%, second voice underneath 74% -- same relative order
  (ideal > second voice > hiss > poor mic) as a standalone check run
  earlier while building the clips (92/67/62/46 on a slightly different
  reference-text punctuation), confirming the ranking is stable even
  though the exact percentages move a little run to run.
- Tokenizer cell: `words: ['can', 'you', 'follow', 'up', 'with', 'catering', 'by', 'friday', '?']`,
  `numbers: [2, 8, 4, 6, 7, 3, 1, 5, 0]`.
- Retention graph: printed `0.49^47 = 0.0000%   0.95^47 = 9.0%`, matching
  the Agenda's "≈0%" / "≈9%" exactly.

**A real bug found and fixed during this check:** the Kahoot vote cell's
source had an actual embedded newline character inside a Python string
literal (a mistake made while generating the cell, not a logic bug),
which gave `SyntaxError: unterminated string literal` the first time this
notebook was run end to end. Fixed, re-verified (output above is from the
fixed version), and checked every other new cell's source for the same
mistake (none found).

---

## `run_tests.py` and frozen-file check

```
$ python prototype/run_tests.py
PASS  a. imports                  11 modules import  (0.7s)
PASS  b. weights                  5 members, labels discussion / decision / action_item  (0.1s)
PASS  c. shipped KPIs reproduce   dev 0.96/0.948, test 0.84/0.825, 36-turn test 0.86/0.853, sample 7/9, probes 9/12  (7.4s)
PASS  d. demo minutes             no decisions, one action item (catering by Friday, Priya)  (0.4s)
PASS  e. --text example           runs, model matches the hand labels 7/7  (0.5s)
PASS  f. hands-on                 solution: both parts OK, 35/45 at knob 0; unfilled version says 'not written yet'  (2.0s)
PASS  g. training data            leak checks pass  (4.7s)
PASS  h. scorecard                frozen scorecard runs (hash ignores line endings)  (2.1s)
PASS  i. smoke runs               5 scripts exit 0  (4.4s)
PASS  j. text clean-up            text clean-up: 31/31 clean-up checks pass  (0.8s)
PASS  k. holdout meetings         holdout 1 0.879/0.806, holdout 2 0.852/0.762 (accuracy / macro-F1)  (6.4s)
SKIP  l. audio (optional)         no --audio FILE given  (0.0s)
PASS  m. repo unchanged           git status and git diff identical before and after  (0.1s)

12 passed, 0 failed, 1 skipped in 29.8 s
```

```
$ git diff --stat main -- prototype/trained_lstm.npz training_data.py scorecard_data.py \
    holdout_data.py minutes_tests.py normalise_tests.py lstm_model.py train_lstm.py \
    meeting_minutes.py transcriber.py hands_on_reference.py testing/ prototype/hands_on.py
(no output -- nothing frozen was touched)
```

---

## Files changed / new

Working-tree diff from `3736eaf`:
```
 demo/FACILITATOR.md           |  13 +--
 demo/hook_demo.ipynb          |  16 ++--
 demo/script.txt               |  16 ++--
 notebook/audio_workshop.ipynb | 198 +++++++++++++++++++++++++++++++++++++++---
 notebook/make_bundle.py       |   5 +-
 5 files changed, 216 insertions(+), 32 deletions(-)
```
New (untracked):
- `notebook/make_noisy_clips.py` -- generates the 4 Kahoot clips
- `notebook/audio/noisy_1_ideal.wav`, `noisy_2_hiss.wav`, `noisy_3_poor_mic.wav`, `noisy_4_second_voice.wav`
- `HANDOFF-WP2.md` -- this file

`notebook/make_bundle.py`: added the 4 noisy clips to `AUDIO`. Bundle now
17 -> 21 files, 2.6 MB -> 3.7 MB.

---

## Not done / open

- **GPU/Colab timing** for the Kahoot answer-key cell (the only
  Whisper-dependent new cell) wasn't measured -- no GPU here, same
  limitation as WP1. The CPU numbers above are what's available.
- **The Kahoot clips' exact percentages will vary slightly machine to
  machine / Whisper-version to Whisper-version** (seen directly above: two
  different runs on this same machine gave different exact numbers but the
  same ranking). The notebook's answer-key markdown says this plainly and
  measures live rather than hardcoding a number, so this isn't considered
  a problem -- flagging it so the reviewer doesn't treat run-to-run
  percentage drift as a bug.
- Nothing else in WP2's deliverable list (N1-N5, N10,
  `plan/WORK_PACKAGES.md`) is outstanding. Sections 4 and 5 deliberately
  untouched (WP3/WP4).
