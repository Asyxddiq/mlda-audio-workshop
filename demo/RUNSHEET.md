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

1. **Share both ahead of time.** Send participants `notebook/audio_workshop.ipynb`
   and `notebook/workshop_files.zip` (rebuild it first: `python
   notebook/make_bundle.py`) together, before the session (email, chat,
   whatever the group uses). `workshop_files.zip` has no demo answers in it
   (that's `demo/script.txt` and the backup WAV, both facilitator-only, in
   `facilitator_files.zip` instead) -- the participant zip has been safe to
   hand out early since the two zips split, so there's no reason to hold it
   back. They can run cell 2 (Setup) whenever they like, including before
   the hook.
2. Participants open Colab (colab.research.google.com), then **File > Upload
   notebook** and pick `audio_workshop.ipynb`.
3. **Runtime > Change runtime type > T4 GPU > Save** (the runtime restarts).
4. Participants run cell 2 (Setup) -- any time, including before the hook,
   now that the zip is shared early. Setup takes about a minute on Colab
   (`pip install`, no large download -- the Whisper model itself downloads
   lazily the first time it's used, in block 2; that first download isn't
   measured here).
5. When Setup finishes it prints `ready`. If it instead says "Upload
   workshop_files.zip...", they uploaded the notebook but not the zip (or
   to the wrong Files panel) -- have them upload the zip and re-run cell 2.

## Block 2: Turning sound into words (00:08-00:18, 10 min)

Facilitator-run -- participants follow along or run the cells themselves,
either is fine (cell 0's boxed note already says so).

**30 laptops, one room (every live-mic cell, here and in blocks 3-9):** speak
softly and close to the laptop's own mic -- don't project to the room, the
mic is a few inches away. Do live cells a row or a table at a time, not all
30 at once, so the noise floor stays low enough for everyone else's mic too.
Wait for the cell's own `done.` before speaking the next prompt. If a
laptop's mic misbehaves (permission denied, no mic, a bad recording), the
fallback clip runs automatically -- that participant follows along on the
printed/plotted result instead of their own.

| Cells | Say (one line) | Expect | Time (CPU) | If it fails |
|---|---|---|---|---|
| 4 | "Three lines of code, and it's already transcribing." | The meeting's text printed | ~20s (+ model download on first use; not measured) | `ModuleNotFoundError: whisper` -- cell 2 didn't finish; re-run it. |
| 6, 8 | "Here's what it's actually reading -- time left to right, pitch bottom to top." | Waveform + spectrogram image; two audio players (forwards/backwards) | ~3s | Blank plot -- re-run cell 2 then cell 4 first (needs `model`). |
| 10 | "Underneath, a spectrogram becomes a handful of numbers every 10 milliseconds." | A frame count printed | <1s | -- |
| 12 | "Now it's telling the two speakers apart too." | The meeting again, with `Speaker1`/`Speaker2` | ~20s | -- |
| 14, 16, 18, 20 | "Same sentence, four ways -- clean, noisy, a bad mic, and your own voice if you want to try it." | A word-accuracy percentage per cell | ~5s each | -- |
| 22, 24 | **Kahoot now:** play the four clips from cell 22's output (ideal, hiss, poor mic, second voice, in that order) in the live Kahoot quiz; the room votes on the ranking *in Kahoot*, not in the notebook. Then run cell 24 for the real numbers as the answer key. | Four accuracy percentages, same order as played | ~20s | -- |
| 26 | **Live mic, optional:** "Say a clear sentence -- see your own waveform and spectrogram." | Waveform + spectrogram of their own voice | ~15s (5s record + Whisper) | No mic / not Colab -- prints a plain sentence and uses the fallback clip automatically, not a traceback. |
| 28 | "Now say it again with real background noise -- tap the desk, rustle paper." | An accuracy percentage against their own clear recording -- often below 100%, but a short clear sentence can still come back 100% (the cell says so if it does) | ~15s | Same fallback as above. |
| 30 | "Record one word, then hear it backwards." | Audio player (forwards), audio player (backwards), Whisper's attempt at the reversed clip (usually garbled or empty) | ~10s | Same fallback as above. |

**Total block 2 compute:** about 85-90s spread across the block's 10
minutes (plus the Whisper model's first-use download, not measured here --
do a throwaway transcription before the session so it's cached), plus up
to ~40s more if the three live-mic cells (2b) run -- not a bottleneck, but
don't start cell 4 after a long pause in the talk; start it right as you
begin speaking so the wait overlaps. 2b is marked optional in the notebook
-- skip it first if the block is running long.

## Block 3: RNNs, a running summary (00:18-00:33, 15 min)

| Cells | Say | Expect | Time | If it fails |
|---|---|---|---|---|
| 33 | "Watch it read one word at a time, and turn each into a number." | The word list, then the memory vector after each word | <1s | Cell 33 imports `numpy` and `lstm_model` itself and needs nothing from earlier cells -- if it still fails, something's wrong with the environment, not the run order. |
| 35 | **Live mic:** "Say any sentence -- watch the memory bar move at every one of your words." | A bar chart, one bar per word they said, no flat stretches | ~15s | No mic / not Colab -- fallback clip, plain sentence, no traceback. |

## Block 4: Hands-on A -- build the memory, then break it (00:33-00:43, 10 min)

| Cells | Say | Expect | Time | If it fails |
|---|---|---|---|---|
| 38 | "Write that one line -- the check tells you right away." | `check (MEMORY_ON = True): OK` once they have it right | <1s | Wrong line -- point them at the "Stuck?" note right below the cell (cell 39), verified against real mistakes. |
| 41 | "Now run the amnesia version and watch the memory vanish." | The toy RNN's memory gap table dropping from `2.85` to `0.00` from word 2 on | <1s | -- |
| 43, 44 | **Live mic** *(bonus, if Part A left time)*: "Say 'Cats are sitting near the window this evening,' then the same line with 'Dogs.'" (both cells printed on screen; close is fine, it doesn't need to match word for word) | Two heard-back lines, then (cell 44) a plot: memory ON decays smoothly from 100% and never hits 0%, memory OFF is flat at 0% from word 2 | ~15s each | No mic / not Colab -- fallback clips, plain sentence, no traceback. |

## Block 5: Why simple memory fades (00:43-00:50, 7 min)

| Cells | Say | Expect | Time | If it fails |
|---|---|---|---|---|
| 47 | "Here's that same fading, as a graph -- including your own toy RNN from a minute ago." | A graph, three curves (0.49, 0.95, and the dotted toy-RNN curve) | <1s | `NameError: np`/`plt` -- this cell now imports both itself; if it still fails, cell 2 wasn't run. |
| 49, 50 | **Live mic:** "Say 'Red cars drive fast,' then 'Blue cars drive fast'" (read each line as shown) | Two heard-back lines | ~10s each | No mic / not Colab -- fallback clips, plain sentence, no traceback. |
| 51, 52 | **Live mic** *(bonus, if there's time)*: "Now the long Monday/Tuesday pair printed on screen" | Two more heard-back lines, then (cell 52) a two-bar chart: the short pair keeps about half its first word's effect, the long pair is down near a few percent, each number printed exactly | ~15-20s each (the long sentence takes longer to speak and transcribe) | No mic / not Colab -- fallback clips, plain sentence, no traceback. |

## Block 6: LSTMs -- an eraser you can set (00:50-01:10, 20 min)

| Cells | Say | Expect | Time | If it fails |
|---|---|---|---|---|
| 55 | "Five of these stacked, like an hourglass." | The layer-size table, then a bar chart (8/16/32/16/8) | <1s | `NameError: transcript` -- sections 1-2 weren't run; run from cell 2. |
| 57 | "And here's the eraser, pencil and voice actually at work, word by word." | A bar chart (3 bars per word) and a printed table; `<eot>` marked | <1s | -- |
| 59 | "That `<eot>` is the model's only clue someone new is talking." | A token list with `<eot>` visible between two turns | <1s | -- |
| 60 | "The hourglass folds four words into one step in the middle layer." | A fold/unfold shape check, `True` | <1s | -- |
| 62 | **Live mic:** "Say two short turns, ~2s apart -- a question, then a one-word answer." | A gate bar chart over YOUR words, `<eot>` visible in the labels | ~15s | No mic / not Colab -- fallback clip, plain sentence, no traceback. |
| 64 | **Live mic:** "Now put your Cats/Dogs pair from 4b through this same toy LSTM at three knob settings." | A 3-line plot, the gap surviving longest at `+6`, dying fastest at `-6` | <1s (reuses 4b's recording, no new prompt) | `NameError: CATS_LIVE` -- cell 43-44 (block 4b) weren't run; run from cell 2. |

## Block 7: Teaching it, and catching it out (01:10-01:20, 10 min)

| Cells | Say | Expect | Time | If it fails |
|---|---|---|---|---|
| 67 | "Confident and right costs almost nothing; confident and wrong costs a lot." | Two lines, `loss = 0.00` and `loss = 5.61` | <1s | -- |
| 69 | "Keyword rules versus the trained model, on a meeting it's never seen." | `keyword rules: 12/18`, `trained LSTM: 17/18`, one miss line | <1s | -- |
| 72, 73, 74 | **Live mic:** "Say a request, a proposal, then a plain fact -- one per cell." | Three lines: model label + confidence + loss, and the keyword rule's answer, for each | ~15s each | No mic / not Colab -- fallback clips, plain sentence, no traceback. |
| 77 | "Same two words, three answers -- this is the hook's 'Sure.' again." | Three label/confidence lines | <1s | -- |
| 80, 81 | **Live mic:** "Say 'Can you book the room? ... Sure.', then 'Shall we move it to Friday? ... Sure.' -- ~2s pause before each 'Sure.'" | Two label/confidence lines, one per exchange | ~20s each | No mic / not Colab -- fallback clips, plain sentence, no traceback. |

## Block 8: Hands-on B -- adjust the eraser (01:20-01:40, 20 min)

| Cells | Say | Expect | Time | If it fails |
|---|---|---|---|---|
| 84 | "Same deal -- one line, the check confirms it." | `check: OK` | <1s | Wrong line -- point at the "Stuck?" note below (cell 85). |
| 88 | "Now sweep the knob and watch both numbers move." | Three blocks (-6, 0, +6), toy gap row + trained score each | <1s | Stub unfilled -- prints "not written yet" three times, expected, not an error. |
| 90 | **Live mic:** "Say a short two-turn exchange -- a question, ~2s pause, then a short answer." | Three lines, one per knob setting, the label on YOUR exchange changing | ~15s | No mic / not Colab -- fallback clip, plain sentence, no traceback. |

## Block 9: Try it on your own meeting (01:40-01:55, 15 min)

| Cells | Say | Expect | Time | If it fails |
|---|---|---|---|---|
| 93 | "Now the whole pipeline, on a real two-speaker transcript." | Rendered minutes (headings, bullets) | <1s | -- |
| 96 | "Write your own 5 lines here, with labels if you want to check yourself." | Label comparison, rendered minutes, unseen words, the prompt | <1s | Bad input (unlabelled mix, no colon, empty) -- prints a clean one-line message, not a traceback; re-check the format hint it gives. |
| 98 | "Add one line of context and watch the model's answer flip." | Two label/confidence lines, the second now correct | <1s | -- |
| 101 | **Live mic:** "Talk a short 4-6 line meeting -- solo (playing both people) or with a partner, ~2s pause between lines." | Transcript with Speaker1/Speaker2, per-turn labels, rendered minutes | ~30-40s | No mic / not Colab -- fallback clip, plain sentence, no traceback. |

## Block 10: Wrap-up (01:55-02:00, 5 min)

| Cells | Say | Expect | Time | If it fails |
|---|---|---|---|---|
| 103 | "Sound becomes words, memory turns words into meaning, meaning becomes minutes -- and here's what it still gets wrong." | The one-sentence recap and three honest limits | -- | -- |

## Notes

- Live-mic recording (`demo/hook_demo.ipynb` cells 4/6, and `audio_workshop.ipynb`'s
  cells 26, 35, 43, 44, 49-52 added in WP7), a real T4 GPU, and how anything
  actually sounds are all untested from this repo's own checks (no Colab or
  microphone available here) -- rehearse these yourself before the day, same as
  the hook's own checklist in `demo/FACILITATOR.md`. `plan/handoffs/HANDOFF-WP7.md`
  lists exactly what to try on Colab with a real mic for the new live cells. The
  live-skit path specifically stays unverified, and its checklist item in
  `demo/FACILITATOR.md` stays unticked, until someone actually runs that
  rehearsal.
- The backup WAV's two voices (Zira as Mei, Hazel as Ravi) are both
  female -- the room may not be able to tell them apart by voice alone, so
  say the names out loud as you introduce the backup, not just rely on the
  recording.
- If a participant's Colab runtime restarts mid-session, point them at the
  boxed note under the notebook's title: run cell 2, then every cell above
  the one they want.
