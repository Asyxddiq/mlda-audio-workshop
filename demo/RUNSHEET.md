# Run-sheet: operating the demo and notebook

This page is for **operating** the workshop -- which cells to run, what to
expect, how long it takes, what to do if it fails. It is not a lecture
script or slide content (those are out of scope for this repo); each block
below gets exactly one pointer line for what to say on screen, nothing more.

Cell numbers are `notebook/audio_workshop.ipynb`'s actual indexes
(0-indexed, markdown and code counted together). For the hook (block 1),
see `demo/FACILITATOR.md` -- this page starts at participant onboarding
and block 2.

## Before the session: participant onboarding

The repo is private, so participants can't just open a GitHub link.

1. **Share the notebook ahead of time; the zip after the hook.** Send
   participants `notebook/audio_workshop.ipynb` (email, chat, whatever the
   group uses) before the session. **Do not hand out
   `notebook/workshop_files.zip`** (rebuild it first: `python
   notebook/make_bundle.py`) **until after the hook** (block 1) -- it
   contains `demo/script.txt`, which gives away the hook's answers. They
   can open the notebook file early; cell 2 (Setup) just won't run until
   they have the zip too.
2. Participants open Colab (colab.research.google.com), then **File > Upload
   notebook** and pick `audio_workshop.ipynb`.
3. **Runtime > Change runtime type > T4 GPU > Save** (the runtime restarts).
4. Participants run cell 2 (Setup) once they're told to -- **not before the
   hook is done**, since it needs the zip. Setup takes about a minute on
   Colab (`pip install`, no large download -- the Whisper model itself
   downloads lazily the first time it's used, in block 2; that first
   download isn't measured here).
5. When Setup finishes it prints `ready`. If it instead says "Upload
   workshop_files.zip...", they uploaded the notebook but not the zip (or
   to the wrong Files panel) -- have them upload the zip and re-run cell 2.

## Block 2: Turning sound into words (00:08-00:18, 10 min)

Facilitator-run -- participants follow along or run the cells themselves,
either is fine (cell 0's boxed note already says so).

| Cells | Say (one line) | Expect | Time (CPU) | If it fails |
|---|---|---|---|---|
| 4 | "Three lines of code, and it's already transcribing." | The meeting's text printed | ~20s (+ model download on first use; not measured) | `ModuleNotFoundError: whisper` -- cell 2 didn't finish; re-run it. |
| 6, 8 | "Here's what it's actually reading -- time left to right, pitch bottom to top." | Waveform + spectrogram image; two audio players (forwards/backwards) | ~3s | Blank plot -- re-run cell 2 then cell 4 first (needs `model`). |
| 10 | "Underneath, a spectrogram becomes a handful of numbers every 10 milliseconds." | A frame count printed | <1s | -- |
| 12 | "Now it's telling the two speakers apart too." | The meeting again, with `Speaker1`/`Speaker2` | ~20s | -- |
| 14, 16, 18, 20 | "Same sentence, four ways -- clean, noisy, a bad mic, and your own voice if you want to try it." | A word-accuracy percentage per cell | ~5s each | -- |
| 22, 24 | **Kahoot now:** play the four clips from cell 22's output (ideal, hiss, poor mic, second voice, in that order) in the live Kahoot quiz; the room votes on the ranking *in Kahoot*, not in the notebook. Then run cell 24 for the real numbers as the answer key. | Four accuracy percentages, same order as played | ~20s | -- |

**Total block 2 compute:** about 85-90s spread across the block's 10
minutes (plus the Whisper model's first-use download, not measured here --
do a throwaway transcription before the session so it's cached) -- not a
bottleneck, but don't start cell 4 after a long pause in the talk; start it
right as you begin speaking so the wait overlaps.

## Block 3: RNNs, a running summary (00:18-00:33, 15 min)

| Cells | Say | Expect | Time | If it fails |
|---|---|---|---|---|
| 26 | "Watch it read one word at a time, and turn each into a number." | The word list, then the memory vector after each word | <1s | Cell 26 imports `numpy` and `lstm_model` itself and needs nothing from earlier cells -- if it still fails, something's wrong with the environment, not the run order. |

## Block 4: Hands-on A -- build the memory, then break it (00:33-00:43, 10 min)

| Cells | Say | Expect | Time | If it fails |
|---|---|---|---|---|
| 28 | "Write that one line -- the check tells you right away." | `check (MEMORY_ON = True): OK` once they have it right | <1s | Wrong line -- point them at the "Stuck?" note right below the cell (cell 29), verified against real mistakes. |
| 31 | "Now run the amnesia version and watch the memory vanish." | The toy RNN's memory gap table dropping from `2.85` to `0.00` from word 2 on | <1s | -- |

## Block 5: Why simple memory fades (00:43-00:50, 7 min)

| Cells | Say | Expect | Time | If it fails |
|---|---|---|---|---|
| 33 | "Here's that same fading, as a graph -- including your own toy RNN from a minute ago." | A graph, three curves (0.49, 0.95, and the dotted toy-RNN curve) | <1s | `NameError: np`/`plt` -- this cell now imports both itself; if it still fails, cell 2 wasn't run. |

## Block 6: LSTMs -- an eraser you can set (00:50-01:10, 20 min)

| Cells | Say | Expect | Time | If it fails |
|---|---|---|---|---|
| 35 | "Five of these stacked, like an hourglass." | The layer-size table, then a bar chart (8/16/32/16/8) | <1s | `NameError: transcript` -- sections 1-2 weren't run; run from cell 2. |
| 37 | "And here's the eraser, pencil and voice actually at work, word by word." | A bar chart (3 bars per word) and a printed table; `<eot>` marked | <1s | -- |
| 39 | "That `<eot>` is the model's only clue someone new is talking." | A token list with `<eot>` visible between two turns | <1s | -- |
| 40 | "The hourglass folds four words into one step in the middle layer." | A fold/unfold shape check, `True` | <1s | -- |

## Block 7: Teaching it, and catching it out (01:10-01:20, 10 min)

| Cells | Say | Expect | Time | If it fails |
|---|---|---|---|---|
| 42 | "Confident and right costs almost nothing; confident and wrong costs a lot." | Two lines, `loss = 0.00` and `loss = 5.61` | <1s | -- |
| 44 | "Keyword rules versus the trained model, on a meeting it's never seen." | `keyword rules: 12/18`, `trained LSTM: 17/18`, one miss line | <1s | -- |
| 47 | "Same two words, three answers -- this is the hook's 'Sure.' again." | Three label/confidence lines | <1s | -- |

## Block 8: Hands-on B -- adjust the eraser (01:20-01:40, 20 min)

| Cells | Say | Expect | Time | If it fails |
|---|---|---|---|---|
| 50 | "Same deal -- one line, the check confirms it." | `check: OK` | <1s | Wrong line -- point at the "Stuck?" note below (cell 51). |
| 54 | "Now sweep the knob and watch both numbers move." | Three blocks (-6, 0, +6), toy gap row + trained score each | <1s | Stub unfilled -- prints "not written yet" three times, expected, not an error. |

## Block 9: Try it on your own meeting (01:40-01:55, 15 min)

| Cells | Say | Expect | Time | If it fails |
|---|---|---|---|---|
| 56 | "Now the whole pipeline, on a real two-speaker transcript." | Rendered minutes (headings, bullets) | <1s | -- |
| 59 | "Write your own 5 lines here, with labels if you want to check yourself." | Label comparison, rendered minutes, unseen words, the prompt | <1s | Bad input (unlabelled mix, no colon, empty) -- prints a clean one-line message, not a traceback; re-check the format hint it gives. |
| 61 | "Add one line of context and watch the model's answer flip." | Two label/confidence lines, the second now correct | <1s | -- |

## Block 10: Wrap-up (01:55-02:00, 5 min)

| Cells | Say | Expect | Time | If it fails |
|---|---|---|---|---|
| 63 | "Sound becomes words, memory turns words into meaning, meaning becomes minutes -- and here's what it still gets wrong." | The one-sentence recap and three honest limits | -- | -- |

## Notes

- Live-mic recording (`demo/hook_demo.ipynb` cells 4/6), a real T4 GPU, and
  how anything actually sounds are all untested from this repo's own
  checks (no Colab or microphone available here) -- rehearse these
  yourself before the day, same as the hook's own checklist in
  `demo/FACILITATOR.md`. The live-skit path specifically stays unverified,
  and its checklist item in `demo/FACILITATOR.md` stays unticked, until
  someone actually runs that rehearsal.
- The backup WAV's two voices (Zira as Mei, Hazel as Ravi) are both
  female -- the room may not be able to tell them apart by voice alone, so
  say the names out loud as you introduce the backup, not just rely on the
  recording.
- If a participant's Colab runtime restarts mid-session, point them at the
  boxed note under the notebook's title: run cell 2, then every cell above
  the one they want.
