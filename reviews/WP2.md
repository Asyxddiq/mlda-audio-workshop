VERDICT: CHANGES

Reviewed commit `a8c7316ed399238980c0b1ea420043e27c304069` (parent `3736eaf`), in a detached worktree. IDs claimed: N1, N2, N3, N4 (unchanged), N5, N10, plus the WP1 carry-over wording fix, plus the orchestrator's WP2 additions (W2a-W2f in `reviews/CHECKLIST.md`).
Everything below was re-run by me, on CPU (no GPU, no Colab).

Short version: this is close. The notebook runs end to end from a folder holding only the zip with no errors, the N2 numbers reproduce exactly, section 3 has no RNN on audio frames, the retention graph is labelled an illustration, and nothing frozen was touched. There are no blockers. Three should-fix items remain, each a false or misleading statement in the notebook text that is quick to fix, so the verdict is CHANGES.

## Findings

| # | Severity | ID | File:line | What is wrong | How to fix |
|---|---|---|---|---|---|
| 1 | should-fix | N2, honesty | `notebook/audio_workshop.ipynb` cell 23 (answer-key markdown) | The cell says "your run may differ slightly machine to machine, but the ranking itself is stable". The ranking does hold, but "slightly" is wrong. I ran the four committed clips through three Whisper sizes with the notebook's own `word_accuracy` and `SAID` text: `base.en` ideal 100 / second voice 74 / hiss 52 / poor mic 35; `small.en` 100 / 100 / 83 / 22; `medium.en` 100 / 91 / 83 / 35. Hiss moves 31 points and second voice 26. On `small.en`, which is the model the notebook picks on a Colab GPU (cell 4: `MODEL = "small.en" if torch.cuda.is_available()`), the second-voice clip ties the ideal clip at 100%. A facilitator on Colab will therefore not see the 100/74/52/35 numbers the cell quotes, and "which clip is most accurate" is a tie between two of them. | Change the sentence to say the percentages depend strongly on the Whisper size (quote the three runs above or just "by 20 points or more"), that the order ideal, second voice, hiss, poor mic held on all three sizes I tried, and that on `small.en` the second-voice clip ties the ideal one. Optionally make the second voice louder (a larger factor than 0.25 in `make_noisy_clips.py`) so it separates from the ideal on `small.en`, but then re-measure on all three and regenerate the clips. |
| 2 | should-fix | N1 | `notebook/audio_workshop.ipynb` cells 5-6, 9-10 | Cell 5 says "the waveform and the spectrogram for the clip Whisper just read". Whisper just read `audio/sample_meeting.wav` (cell 4), but cell 6 plots `audio/sample_short.wav`. Cell 9 then says "This is the picture above, in the form a model reads", but cell 10 runs `mfcc` on `sample_meeting.wav`, not the clip in the picture. The two statements are not true as written. | Either plot the same file Whisper just transcribed (use the first few seconds of `sample_meeting.wav`, so the reversed-word clip can stay), or change cell 4 to transcribe `sample_short.wav` and keep the rest. Then make cell 10 use the same file. |
| 3 | should-fix | N3, N10 | `notebook/audio_workshop.ipynb` cell 0 (contents table) | The table at the top still lists "3 \| RNNs: reading audio as a sequence". Section 3 itself is now correctly about words (cells 25-26), so the contents table contradicts it, and it is the first thing participants read. SPEC N10 / tension 1 is exactly this: the RNN reads words, not audio. | Change the row to "RNNs: reading words as a sequence". |
| 4 | nit | N5 | cells 30-31 | The Programme asks to "link the example to long-range dependencies". The cell links only to the LSTM section ("see below"). Also the Part A output (cell 28, once solved) prints "about 58% kept per step" for the toy RNN, while the graph uses 0.49 for the "untrained model settings"; a participant may ask why the two differ. (The 0.49 is backed: `README.md:329` and `progress.md:111` report forget gates near 0.49 with random weights.) | One sentence: the toy RNN in Part A has its own number; 0.49 is what the untrained LSTM gates measured. Optionally add a line on why long-range dependencies (an early line the last line needs) are the cost of fading. |
| 5 | nit | N3 | cells 25-27 | The notebook states the recurrence in words (`new memory = tanh(W_x · x + W_h · old memory + b)`), but never in the Agenda's symbol form `h = tanh(W_x·x + W_h·h_prev + b)`. Section 3 also never shows the step being applied word after word (cell 26 only turns words into numbers). The Agenda gives both forms. | Add the `h = tanh(...)` line next to the word form. Optionally print the memory vector after each of the 9 words in cell 26, which would also show "reads one word at a time". |
| 6 | nit | N1 | cell 6 | The spectrogram cell prints a `RuntimeWarning: divide by zero ... log10` into the notebook output on Colab (the builder noted it). A facilitator may find it distracting on screen. | Optional: `import warnings` and silence it in that cell, or add a tiny epsilon. |
| 7 | nit | N2 | `notebook/make_bundle.py`, `make_noisy_clips.py` | `make_noisy_clips.py` is not in the zip (correct, participants only need the clips). The docstring says the clips are what "the facilitator reads out after the room votes", but the numbers are measured live, not read out. | None needed for the merge. |

## Your five "look hardest" points

1. **N2 key: measured, and do they reproduce?** Yes, exactly, for `base.en`. From a folder holding only the zip, the notebook's own cell 24 printed ideal 100%, hiss 52%, poor mic 35%, second voice 74%, the builder's numbers. Two further `base.en` runs gave identical numbers (the hiss uses a seeded generator). I also regenerated the clips with `python notebook/make_noisy_clips.py`: `git status` is clean, so the committed WAVs are exactly what the script writes. The numbers are measured, not hardcoded; but they depend on the model size (finding 1).
2. **§3 has no RNN on audio frames:** confirmed. Cells 25-26 tokenise a sentence into words and numbers; the MFCC cell moved to section 1 (cell 10) and is described as "what Whisper-style models actually see". `grep` for `rnn_ctc` in the notebook finds nothing. Hands-on A (cells 27-29) is on 12 word vectors, no audio. Contents table is stale (finding 3).
3. **Retention graph labelled an illustration:** yes, twice: the markdown (cell 30: "This is an **illustration** ... not a measured property of the trained model") and the plot title (cell 31). The printed numbers come from the cell: `0.49^47 = 0.0000%   0.95^47 = 9.0%`, which equals `0.49**47` and `0.95**47` (0.0895). Both curves run over 48 points (0 to 47 steps). Pass for N5.
4. **Runs from a zip-only folder:** yes. I built a fresh zip (21 files, 3.7 MB) and ran every cell with `nbclient` from a folder holding only `workshop_files.zip` and `audio_workshop.ipynb`: all 42 cells, no errors, 53 s total on CPU (the builder reported 88 s). The committed notebook has no saved outputs.
5. **Timings fit blocks 2-5:** yes. CPU compute per cell (my run): Whisper load and meeting transcription 13.3 s, spectrogram 2.1 s, MFCC 0.3 s, `get_transcript` 13.8 s, section-2 noise cells 3.6 + 1.8 + 1.9 s, Kahoot answer key 7.6 s, tokenizer 0.0 s, Part A 0.1 s, retention graph 0.3 s. Sections 1-2 total about 45 s of compute inside a 10-minute talk; blocks 3-5 (15, 10, 7 minutes) add under a second of compute each. Sections 1 and 2 are marked facilitator-run with minutes stated (about 6 and 4 of the 10; the notebook says it is an estimate). Nothing here overruns.

## Requirement ID check against the PDFs

| ID | Result |
|---|---|
| N1 | Mostly pass. Waveform with a time axis; spectrogram with time left to right and pitch bottom to top (matplotlib default puts low frequencies at the bottom); a reversed 0.5 s clip that is the first word of the clip ("Venue", 0.0-0.48 s by Whisper word timestamps), played forwards then backwards; Whisper writing words (cell 4). Cells 5 and 9 mis-describe which file is shown (finding 2). The Agenda's "list audio applications and challenges" is not in the notebook; I treat that as the facilitator's talk (no finding). |
| N2 | Pass on content: four clips of one sentence (ideal, hiss, poor mic, second voice), seeded and reproducible, a vote cell and a measured answer-key cell. The caveat on model size is finding 1. Kahoot itself not built (out of scope). |
| N3 | Partly. Words-based intro, tokenizer cell, step in words; the symbol form and a stepping display are missing (finding 5, nit). |
| N4 | Pass, unchanged code. I ran Part A myself: the reference line gives `check (MEMORY_ON = True): OK`; a line that forgets the old-memory term gives `not quite`; the unfilled stub says `not written yet`; deleting the old-memory term with `MEMORY_ON = False` gives `OK` and a gap of 0.00 from word 2 on ("the first word is gone. The memory really lives in that one term"). |
| N5 | Pass (see point 3). Nits in finding 4. |
| N10 | Pass in substance: Whisper cells (1-2) and the words-only RNN section are separate, and the minutes pipeline still takes a text transcript. The stale contents row is finding 3. |
| W2a | Pass (MFCC cell in section 1, "what Whisper-style models actually see"; no RNN on audio frames anywhere). |
| W2b | Pass with finding 1. |
| W2c | Pass. |
| W2d | Pass. |
| W2e | Pass. The three pacing texts (`demo/script.txt` header, `demo/FACILITATOR.md` step 2, `demo/hook_demo.ipynb` cells 0 and 4) now say "each turn in one go, no long pause inside it, ~2 s before the other person speaks", and mention that two turns have two sentences. `git diff 3736eaf a8c7316 -- demo/script.txt` shows only comment lines changed, so the script content and the WAV are untouched. |
| W2f | Pass. `notebook/audio_workshop.ipynb` has 0 saved outputs and 0 execution counts. `·` is the only encoding churn (`·` stored as a JSON escape, same character; checked with `ord`). |

## Commands I ran, with output

```
$ bash reviews/check_frozen.sh a8c7316
(base printed as main = 3736eaf, since main now holds WP1)
frozen: none changed / outside allowed list: none
files: HANDOFF-WP2.md, demo/FACILITATOR.md, demo/hook_demo.ipynb, demo/script.txt, notebook/audio/noisy_{1_ideal,2_hiss,3_poor_mic,4_second_voice}.wav,
       notebook/audio_workshop.ipynb, notebook/make_bundle.py, notebook/make_noisy_clips.py
RESULT: OK   (exit 0)

$ python prototype/run_tests.py
12 passed, 0 failed, 1 skipped in 38.0 s
$ python prototype/training_data.py      -> exit 0

$ python notebook/make_bundle.py         (detached worktree)
wrote ... workshop_files.zip (3.7 MB, 21 files)       # includes the four noisy_*.wav, demo/script.txt, demo/backup_meeting.wav; no ANSWER_KEY.md

$ python notebook/make_noisy_clips.py ; git status --short
(clean: regenerated WAVs identical to the committed ones)
```

Notebook from a zip-only folder (`nbclient`, `base.en`, CPU), key lines:
```
[ 4] 13.3s using base.en on the CPU ... (meeting transcript)
[10]  0.3s 60 seconds of audio  ->  5967 frames x 13 numbers (one frame every 10 ms)
[12] 13.8s Speaker1/Speaker2 turns from get_transcript(pause_gap=1.5)
[14]  3.6s clean       100%
[16]  1.8s noise 0.1: 52%
[18]  1.9s poor mic  : 35%
[24]  7.6s 1. ideal 100% / 2. background hiss 52% / 3. poor microphone 35% / 4. a second voice underneath 74%
[26]  0.0s words  : ['can','you','follow','up','with','catering','by','friday','?']  numbers: [2, 8, 4, 6, 7, 3, 1, 5, 0]
[28]  0.1s PART A ... check: not written yet (it returned None)      # the unfilled stub, as designed
[31]  0.3s 0.49^47 = 0.0000%   0.95^47 = 9.0%
TOTAL 53.3 s, no cell errors (42 cells)
```
(Part B, the minutes, and the "your own meeting" cells also ran: they are unchanged by WP2.)

N2 model-size check (my script `rank.py`, the notebook's `word_accuracy` and `SAID`, the four committed clips):
```
base.en   [ideal 100, hiss 52, poor_mic 35, second_voice 74]   ideal > second_voice > hiss > poor_mic   (twice, identical)
small.en  [ideal 100, hiss 83, poor_mic 22, second_voice 100]  ideal > second_voice > hiss > poor_mic   (ideal and second voice tie)
medium.en [ideal 100, hiss 83, poor_mic 35, second_voice 91]   ideal > second_voice > hiss > poor_mic
```

Part A, run by hand (`prototype/hands_on.py`):
```
correct line, MEMORY_ON=True   -> check: OK  got [-0.9964  0.5838], expected [-0.9964  0.5838]; gap shrinks "about 58% kept per step"
wrong line (no old memory), MEMORY_ON=True  -> check: not quite
old-memory term deleted, MEMORY_ON=False    -> check: OK; gap 2.85 then 0.00 from word 2 on
```

Whisper word timestamps on `sample_short.wav` (base.en): `Venue` 0.00-0.48 s, `is` 0.48-0.68 s, so the reversed `wave_samples[:8000]` clip is exactly the word "Venue".

Not verified: Colab and a T4 (so `small.en` timings and the GPU model choice are only checked through my CPU `small.en` run of the clips), and the audio playback itself (I did not listen to the reversed clip or the four noisy clips; the accuracy numbers and the waveform energy are what I checked).

## To get to PASS

Fix findings 1-3 (a few sentences and one table row in the notebook). I will then re-run the notebook from a zip-only folder and the clip ranking, and read the changed cells.
