# Testing phase

Everything for the testing phase in one place: what to run, the scripts to read aloud, and the results, both from Claude's side and from manual runs by the team.

| | |
|---|---|
| **Started** | 16 Oct 2026 |
| **Status** | Claude's side done (16 Oct), including the model update (see [Model update](#model-update-16-oct)). Manual runs: none yet. |
| **Closes "Real transcriber swap"** | one manual solo or pair recording with `transcriber: PASS` |

---

## Quick start: test it on your own

You need Python, plus Whisper, a microphone library and ffmpeg. Run these from the repo folder (`python -m pip` works even when `pip` alone isn't found):

```
python -m pip install -r requirements.txt
python -m pip install openai-whisper sounddevice
winget install --id Gyan.FFmpeg -e
```

The last line is for Windows; on a Mac use `brew install ffmpeg`. Reopen the terminal afterwards so ffmpeg is found. Whisper pulls in PyTorch, so the install is a few GB.

Then, from the repo folder:

```
python testing/solo_test.py --name Asy
```

It shows one line at a time and records you. Read the line, press Enter, and stay quiet while the next line comes up (it waits 2 seconds, which is how one voice plays two people). At the end it runs the check and saves the result in `testing/results/manual/`. Push that `.txt` file, or send it to Claude, and it goes into the [manual results](#manual-runs-by-the-team) table below.

Recorded on your phone instead? Read [`scripts/solo_1_sample_meeting.txt`](scripts/solo_1_sample_meeting.txt) with a 2-second stop between lines, then:

```
python testing/solo_test.py --name Asy --file my_recording.m4a
```

Add `--script solo_2` for the second, harder script, or `--script solo_3` for a long reading of 50 short everyday lines (about 5 minutes).

Recordings (`.wav`, `.m4a`, `.mp3`) stay on your computer: `testing/.gitignore` keeps them out of git. Only the text result is pushed.

---

## What's in this folder

| Path | What it is |
|---|---|
| `testing.md` | This page: how to run each test, and all the results |
| [`model_update_16oct/`](model_update_16oct/model_update.md) | The 16 Oct model update: the rule, what shipped, what was tried, every score |
| [`try_it.py`](try_it.py) | The quickest test: talk (or type) like a meeting, no script, and see the transcript, each turn's label and the minutes |
| [`phases.py`](phases.py) | The chain one phase at a time (audio to text, RNN, LSTM, meeting minutes), a PASS or FAIL for each |
| [`solo_test.py`](solo_test.py) | The individual test: records one person (or checks a recording), runs the audio check, saves the result |
| [`scripts/`](scripts/) | The 9 scripts to read aloud: `solo_1`, `solo_2`, `solo_3` (one person) and `1`-`6` (two or three people) |
| [`testing script.md`](testing%20script.md) | The 67 meeting tests with the minutes each one should produce |
| [`results/claude/`](results/claude/) | Every result from Claude's side, one file per run |
| [`results/manual/`](results/manual/) | Results from manual runs by the team (filled in by `solo_test.py`) |

The Python checkers that load the model stay in `prototype/`, next to the code they import: `run_tests.py`, `minutes_tests.py`, `audio_check.py`. `solo_test.py` calls into them. The reading scripts are also still in `prototype/audio_tests/` for now; `testing/scripts/` is the copy to use.

---

## All the tests

| # | Test | What it checks | Command | Needs |
|---|---|---|---|---|
| 1 | Repo check (Nomt) | The demo still works: imports, weights, shipped scores, demo minutes, hands-on, scorecard, smoke runs | `python prototype/run_tests.py` | numpy only, about 25 s |
| 2 | Meeting tests | 67 typed meetings: do the minutes have the right decisions, action items, owners and due dates? | `python prototype/minutes_tests.py` (add `--keywords` for the fallback) | numpy only |
| 3 | Solo audio test | One person's recording: right turns, speakers and words (transcriber), and the same labels and minutes as the typed script (end to end) | `python testing/solo_test.py --name YOU` | Whisper, ffmpeg, a microphone or a recording |
| 4 | Pair audio tests | The same check with two or three people reading `scripts/1`-`6` (fast replies, a pause mid-turn, three speakers, a noisy room) | `python prototype/audio_check.py testing/scripts/1_sample_meeting.txt rec.m4a` | Whisper, ffmpeg, two people |
| 0 | Phase by phase | Each stage on its own: 1 audio to text (Whisper + pause rule), 2 the RNN teaching demo, 3 the trained LSTM's labels, 4 the minutes. Says what each result does not prove | `python testing/phases.py` (all four, automatic, no audio, about 15 s); **by hand: `python testing/phases.py --manual --name YOU`** (or add a phase number, e.g. `3 --manual`): you say 5 sentences (phase 1), type frames for the CTC decoder (2), type a short meeting (3 and 4), and each phase says PASS or FAIL; saves `results/manual/YOU_phases_<time>.txt`. Phase 1 on a recording: `python testing/phases.py 1 --audio rec.m4a` | numpy only; Whisper, sounddevice and ffmpeg for phase 1 |
| 5 | Checker self-test | `audio_check.py` passes when the transcript is exactly the script | `python prototype/audio_check.py SCRIPT.txt --heard TRANSCRIPT.txt` (a saved "Speaker1: ..." transcript instead of audio) | numpy only |
| 6 | Text clean-up | "I will" and "I'll", "let us" and "let's", "OK" and "okay" ... are read the same by the model, and names and possessives are left alone | `python prototype/normalise_tests.py` (also checks j in test 1) | numpy only |
| 7 | Holdout meetings | 17 fresh meetings (271 turns) the model was never trained on, written 16 Oct: labels right | part of test 1 (check k); data in `prototype/holdout_data.py` | numpy only |

The audio check gives two verdicts:
- **transcriber**: same number of turns, same back-and-forth, at least 90% of the words. This is the audio-to-text part.
- **end to end**: that, plus the same labels and minutes as the typed script. A transcriber PASS with an end-to-end FAIL means the words came through but the labeller reacted to a small wording change.

---

## Results summary

| Test | Claude's side (16 Oct) | Manual (team) |
|---|---|---|
| 1. Repo check | **13/13 pass** with the optional audio run (12 pass and 1 skipped without it) | not run yet |
| 2. Meeting tests | **45/67** trained model, 28/67 keyword fallback | not run yet |
| 3. Solo audio, a real person reading | can't be done here (no person to record) | **not run yet**: this is the one that closes "Real transcriber swap" |
| 3. Solo audio, synthetic voice, guided mode | **PASS / PASS** (`solo_1`, simulated microphone) | n/a |
| 3. Solo audio, synthetic voices from files | `solo_1`: PASS with 1-2 s gaps, FAIL with 0.4 s gaps or rough voices; `solo_2`: transcriber PASS, end to end FAIL | n/a |
| Real human speech (LibriSpeech) | **transcriber PASS**, 95% of words | n/a |
| 4. Pair audio tests | can't be done here | not run yet |
| 5. Checker self-test | **8/8 pass** | not run yet |
| 6. Text clean-up | **31/31 pass** | n/a |
| 7. Holdout meetings | **holdout 1: 131/149 (0.879 / 0.806); holdout 2: 104/122 (0.852 / 0.762)** | n/a |

---

## Claude's side

### How it was run

Claude works in a cloud workspace without a microphone, and `openai-whisper` can't be installed there (it needs PyTorch, which isn't available). So:

- **Whisper**: the same Whisper **base** weights the prototype uses, exported to ONNX by the k2-fsa/sherpa-onnx project, run with onnxruntime behind a small stand-in for `import whisper`. Everything in the repo ran unchanged on top of it: `transcriber.py` and its 0.8 s pause rule, `audio_check.py`, `solo_test.py`. The one difference: segment times came from a silence detector, not from Whisper's word timestamps. A manual run with the real package is still the only check of that part.
- **Voices**: synthetic speech from flite (an open-source text-to-speech engine), one voice per recording, the way one person would read a solo script. Three voices: `slt` (clearest), `awb` and `rms` (rougher), some with background noise added.
- **Real human speech**: 3 public-domain LibriSpeech recordings, joined with 2-second gaps.
- **Guided mode** (the microphone flow in `solo_test.py`): run with a simulated microphone. A scripted reader spoke each line as it appeared and pressed Enter. This tests the flow, the 2-second gaps and the saving; it can't test real microphone hardware.

### Each run

| Run | Script | transcriber | end to end | Notes | Result file |
|---|---|---|---|---|---|
| Real human speech | 3 LibriSpeech lines | **PASS** | PASS | 3/3 turns, 95% of words | [human_speech_librispeech](results/claude/human_speech_librispeech_2026-10-16.txt) |
| Guided mode, simulated mic, `slt` | solo_1 | **PASS** | **PASS** | 9/9 turns, 96% of words | [file](results/claude/Claude-guided-fakemic-slt_solo_1_sample_meeting_2026-10-16_1308.txt) |
| `slt`, 2 s gaps | solo_1 | **PASS** | **PASS** | 9/9 turns, 96% | [file](results/claude/Claude-synthetic-slt_solo_1_sample_meeting_2026-10-16_1304.txt) |
| `slt`, 1.5 s gaps | solo_1 | PASS | PASS | 9/9 turns, 96% | [file](results/claude/Claude-synthetic-slt-1.5s-gaps_solo_1_sample_meeting_2026-10-16_1304.txt) |
| `slt`, 1.0 s gaps | solo_1 | PASS | PASS | 9/9 turns, 96% | [file](results/claude/Claude-synthetic-slt-1.0s-gaps_solo_1_sample_meeting_2026-10-16_1305.txt) |
| `slt`, 0.4 s gaps | solo_1 | FAIL | FAIL | all 9 lines merged into 1 turn: the check catches a reading that is too fast | [file](results/claude/Claude-synthetic-slt-0.4s-gaps_solo_1_sample_meeting_2026-10-16_1305.txt) |
| `awb` | solo_1 | FAIL | FAIL | 88% of words; "Yes, I'll email them today." heard as "Yes, I leave me alone to date." | [file](results/claude/Claude-synthetic-awb_solo_1_sample_meeting_2026-10-16_1306.txt) |
| `rms` + noise | solo_1 | FAIL | FAIL | 85% of words | [file](results/claude/Claude-synthetic-rms-noise_solo_1_sample_meeting_2026-10-16_1306.txt) |
| `slt` | solo_2 | PASS | FAIL | 95% of words, but "Two things today" was heard as "do things today" and labelled an action item | [file](results/claude/Claude-synthetic-slt_solo_2_bake_sale_2026-10-16_1307.txt) |
| `rms` | solo_2 | PASS | FAIL | 97% of words, but "I'll bring" came out as "I will bring", which the model labels discussion | [file](results/claude/Claude-synthetic-rms_solo_2_bake_sale_2026-10-16_1307.txt) |
| `awb` + noise | solo_2 | FAIL | FAIL | 83% of words (the two "[BLANK_AUDIO]" marks come from Claude's stand-in) | [file](results/claude/Claude-synthetic-awb-noise_solo_2_bake_sale_2026-10-16_1307.txt) |

**Re-run with the text clean-up in place (16 Oct, afternoon):** same recordings, same stand-in Whisper.

| Run | Script | transcriber | end to end | Change | Result file |
|---|---|---|---|---|---|
| Guided mode, simulated mic, `slt` | solo_1 | PASS | PASS | same | [file](results/claude/Claude-cleanup-guided-fakemic-slt_solo_1_sample_meeting_2026-10-16_1418.txt) |
| `slt`, 2 / 1.5 / 1.0 s gaps | solo_1 | PASS | PASS | same | [2 s](results/claude/Claude-cleanup-slt_solo_1_sample_meeting_2026-10-16_1412.txt) |
| `slt`, 0.4 s gaps | solo_1 | FAIL | FAIL | same (one merged turn) | [file](results/claude/Claude-cleanup-slt-0.4s-gaps_solo_1_sample_meeting_2026-10-16_1413.txt) |
| `awb`, `rms` + noise | solo_1 | FAIL | FAIL | same (88% / 85% of words) | [awb](results/claude/Claude-cleanup-awb_solo_1_sample_meeting_2026-10-16_1414.txt) |
| `slt` | solo_2 | PASS | FAIL | same ("Two things today" heard as "do things today") | [file](results/claude/Claude-cleanup-slt_solo_2_bake_sale_2026-10-16_1415.txt) |
| `rms` | solo_2 | PASS | **PASS** | **was FAIL**: "I will bring" is now read as an action item | [file](results/claude/Claude-cleanup-rms_solo_2_bake_sale_2026-10-16_1415.txt) |
| `awb` + noise | solo_2 | FAIL | FAIL | same (83% of words) | [file](results/claude/Claude-cleanup-awb-noise_solo_2_bake_sale_2026-10-16_1416.txt) |

Other results from Claude's side:
- Repo check: [run_tests_2026-10-16.txt](results/claude/run_tests_2026-10-16.txt) (morning, 11 checks) and [run_tests_with_cleanup_2026-10-16.txt](results/claude/run_tests_with_cleanup_2026-10-16.txt) (afternoon, 13 checks, 13 passed, the audio check ran on the `slt` solo_1 recording).
- Clean-up tests: [normalise_tests_2026-10-16.txt](results/claude/normalise_tests_2026-10-16.txt), 31/31.
- Meeting tests: [trained model](results/claude/minutes_tests_model_2026-10-16.txt) (45/67), and [keyword fallback](results/claude/minutes_tests_keywords_2026-10-16.txt) (28/67).
- Checker self-test: [checker_selftest_2026-10-16.txt](results/claude/checker_selftest_2026-10-16.txt), 8/8.

---

## Manual runs (by the team)

Add a row for every manual run. `solo_test.py` saves the result file; link it here.

| Date | Who | Test | Recording | transcriber | end to end | Result file |
|---|---|---|---|---|---|---|
| 17 Oct | Asy | Solo `solo_1`, microphone, guided | `base`, old pause rule | FAIL (3 of 9 turns, 60% of words) | FAIL | [file](results/manual/Asy_solo_1_sample_meeting_2026-10-17_1810.txt) |
| 17 Oct | Asy | Same recording re-scored after the fixes below | `small` (multilingual, English forced), word-level pauses, 1.5 s solo gap | FAIL (9 of 9 turns, 71% of words: only the words fail) | FAIL (labels and minutes match) | not saved: a re-score of the line above |

**What the first manual run found (17 Oct), and what was changed:**
- Whisper guessed the language from a Malaysian accent and wrote Malay, which was also very slow. Fixed: `transcriber.py` now says `language="en"`.
- Whisper joined several spoken lines into one segment, so the 0.8 s pause rule saw 3 turns instead of 9. Fixed: the rule now runs on single words, which keep their own times (9 of 9 turns on the same recording).
- The `base` model heard 60% of the words ("catering" as "Patterning"); `small` hears 71% and got the labels and minutes right (a 461 MB download; 36 s for this 73 s recording on a laptop CPU, against 15 s for `base`).
- **English only, hard-set (17 Oct):** `transcriber.py` now uses `small.en`, a Whisper model trained on English alone, so it cannot write another language, plus `language="en"` and `task="transcribe"` so nothing is guessed or translated. On Asy's recording it hears the same 71% of the words as `small`. On a clean synthetic reading of `solo_1` (Windows voice Hazel, 2 s gaps) it hears **98%** and nothing but English.
- **In-line pauses (fixed for the solo test):** the pause rule splits at 0.8 s, so a pause at a full stop inside a line became a new turn (Hazel: 12 turns heard for 9; Asy's recording: 11). Measured: pauses inside a line were at most 1.1 s, gaps between lines at least 2.3 s (Hazel) and 3.2 s (Asy). `get_transcript` now takes `pause_gap`; `solo_test.py` uses 1.5 s, and the pair tests keep 0.8 s. Re-scored: **Hazel passes both verdicts** (9/9 turns, 98% of words, labels and minutes match; [result](results/claude/Claude-hazel-1.5s_solo_1_sample_meeting_2026-10-17_1843.txt)), and Asy's recording gets turns, speakers, labels and minutes right and fails only on words (71%).
- Tried and dropped: finding pauses from the audio's volume (worse), and a vocabulary hint to Whisper (+4 points of words, but 13 times slower).
- Still failing: the 90% word bar. Some misses are the model ("I'll email them" heard as "I will be more than"), some are spelling ("Q3" against "quarter three", "8%" against "8,000"), which the word counter treats as errors. For participants with thick accents, `medium` is the next size up (about 1.5 GB, slower); it was not tested.

When the first row shows `transcriber: PASS` for a real person, tick "Real transcriber swap" in `progress.md`.

---

## What the tests found so far

- **The model is the weak part, not the transcriber.** In every synthetic run where the words came through, any end-to-end failure was the labeller reacting to a small wording change: "I will" for "I'll", or one misheard word ("Two things today" as "do things today"). The "I will" kind is now fixed by the text clean-up (16 Oct); the misheard-word kind is not fixable in the model.
- **Decisions are missed most** in the meeting tests: 9/18 in group B, mostly short ones like "we're going with pizza" or "Saturday it is". Two retrains with new decision examples did not fix this without breaking something else (see [Model update](#model-update-16-oct)); real labelled meetings are the next step.
- **Pause rule limits** (from `transcriber.py`): a reply that starts less than 0.8 s after the last word merges into the turn before; a long pause inside a line splits it; a third speaker isn't told apart. The pair tests 3, 4 and 5 are there to measure these.
- **Claude's side can't test** the real `openai-whisper` package (its word timestamps), a real microphone, or a real person's voice. The manual runs cover those.

---

## Model update (16 Oct)

Asked: add a text clean-up and more decision examples. **The clean-up is shipped on the unchanged 13 Oct weights; the two retrains were not**, because they failed the rule written before training (worse on the team's test meetings, and a wrong decision in the demo minutes). The clean-up changes no earlier result above except the `solo_2` / `rms` run, which now passes. The rule, every score and the patch to reproduce the retrains: [`model_update_16oct/`](model_update_16oct/model_update.md).
