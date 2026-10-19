# Run-sheet: operating the demo and notebook

Which cells to run, what to expect, how long, what to do if it fails. Not a lecture script. Each row has one line to say.

Cell numbers are positions in `notebook/audio_workshop.ipynb`, counting from 0, markdown and code together. The hook (block 1) is in `demo/FACILITATOR.md`. This page starts at onboarding and block 2.

## Before the session: participant onboarding

The repo is private. Participants can't open a GitHub link.

1. Send both files before the session, by email or chat:
   - `notebook/audio_workshop.ipynb`
   - `notebook/workshop_files.zip` (rebuild first: `python notebook/make_bundle.py`)

   The participant zip holds no demo answers. Those are in `facilitator_files.zip` (`demo/script.txt`, the backup WAV). Sending it early is safe.
2. Participants open Colab (colab.research.google.com). File > Upload notebook. Pick `audio_workshop.ipynb`.
3. Runtime > Change runtime type > T4 GPU > Save. The runtime restarts.
4. Participants run cell 2 (Setup). Any time, even before the hook. It takes about a minute (`pip install`). Whisper's model downloads on first use, in block 2. That download is not measured here.
5. Setup prints `ready` when done. If it says "Upload workshop_files.zip...", the zip is missing or in the wrong Files panel. Upload it. Re-run cell 2.

## Block 2: Turning sound into words (00:08-00:18, 10 min)

Facilitator-run. Participants can follow or run the cells. Cell 0's boxed note says so.

**30 laptops, one room** (every live-mic cell, blocks 2-9):

- Speak softly, close to your own laptop mic. Don't project to the room.
- Run live cells one row or table at a time, not all 30 at once. This keeps the noise low.
- Wait for the cell's own `done.` before the next prompt.
- If a mic fails (permission denied, no mic, bad recording), a fallback clip runs automatically. No traceback. That participant follows the printed or plotted result.

| Cells | Say | Expect | Time (CPU) | If it fails |
|---|---|---|---|---|
| 4 | "Three lines of code, and it's already transcribing." | The meeting's text | ~20s (+ model download on first use; not measured) | `ModuleNotFoundError: whisper`. Cell 2 didn't finish. Re-run it. |
| 6, 8 | "Here's what it's actually reading. Time left to right, pitch bottom to top." | Waveform and spectrogram. Two audio players (forwards, backwards) | ~3s | Blank plot. Re-run cell 2, then cell 4 (needs `model`). |
| 10 | "Underneath, a spectrogram becomes a handful of numbers every 10 milliseconds." | A frame count | <1s | - |
| 12 | "Now it tells the two speakers apart too." | The meeting again, as `Speaker1` / `Speaker2` | ~20s | - |
| 14, 16, 18, 20 | "Same sentence, four ways. Clean, noisy, a bad mic, and your own voice if you want." | A word-accuracy percentage per cell | ~5s each | - |
| 22, 24 | **Kahoot now.** Play the four clips from cell 22's output: ideal, hiss, poor mic, second voice. The room votes in Kahoot, not in the notebook. Then run cell 24 for the real numbers as the answer key. | Four accuracy percentages, in the order played | ~20s | - |
| 26 | **Live mic, optional.** "Say a clear sentence. See your own waveform and spectrogram." | Waveform and spectrogram of their voice | ~15s (5s record + Whisper) | No mic or not Colab: plain sentence, fallback clip. |
| 28 | "Say it again with real background noise. Tap the desk, rustle paper." | Accuracy % against their clear recording. Often under 100%. A short clear sentence can still hit 100% (the cell says so) | ~15s | Same fallback. |
| 30 | "Record one word, then hear it backwards." | Player (forwards), player (backwards), Whisper's attempt at the reversed clip (usually garbled or empty) | ~10s | Same fallback. |

**Block 2 compute:** about 85-90s across the 10 minutes. The optional live-mic cells (2b) add up to ~40s. The first Whisper download is not included. Do a throwaway transcription before the session so it is cached. Start cell 4 as you begin speaking, so the wait overlaps your talk. Skip 2b first if the block runs long.

## Block 3: RNNs, a running summary (00:18-00:33, 15 min)

| Cells | Say | Expect | Time | If it fails |
|---|---|---|---|---|
| 33 | "Watch it read one word at a time and turn each into a number." | The word list, then the memory vector after each word | <1s | Cell 33 imports `numpy` and `lstm_model` itself and needs nothing earlier. If it fails, the environment is wrong, not the run order. |
| 35 | **Live mic.** "Say any sentence. Watch the memory bar move at every word." | A bar chart, one bar per word, no flat stretches | ~15s | No mic or not Colab: plain sentence, fallback clip. |

## Block 4: Hands-on A, build the memory then break it (00:33-00:43, 10 min)

| Cells | Say | Expect | Time | If it fails |
|---|---|---|---|---|
| 38 | "Write that one line. The check tells you right away." | `check (MEMORY_ON = True): OK` | <1s | Wrong line. Point to the "Stuck?" note under the cell (cell 39). |
| 41 | "Now run the amnesia version. Watch the memory vanish." | The toy RNN's gap table falling from `2.85` to `0.00` from word 2 on | <1s | - |
| 43, 44 | **Live mic, bonus (if Part A left time).** "Say 'Cats are sitting near the window this evening.' Then the same line with 'Dogs.'" Both lines are on screen. Close is fine. | Two heard-back lines. Then (cell 44) a plot. Memory ON decays smoothly from 100% and never hits 0%. Memory OFF is flat at 0% from word 2 | ~15s each | No mic or not Colab: plain sentence, fallback clips. |
| 47 | *(Bonus)* "Write your own first-word-swap pair. The check confirms only the first word differs." | `check: OK` and a retained-% number on their words | <1s | Word counts differ, or a later word differs. The check says which. Nothing below depends on it. |

## Block 5: Why simple memory fades (00:43-00:50, 7 min)

| Cells | Say | Expect | Time | If it fails |
|---|---|---|---|---|
| 50 | "Here's that same fading as a graph. Your own toy RNN is in it." | A graph with three curves: 0.49, 0.95, and the dotted toy-RNN curve | <1s | `NameError: np`/`plt`. The cell imports both itself. If it still fails, cell 2 wasn't run. |
| 52, 53 | **Live mic.** "Say 'Red cars drive fast,' then 'Blue cars drive fast.'" Read each line as shown. | Two heard-back lines | ~10s each | No mic or not Colab: plain sentence, fallback clips. |
| 54, 55 | **Live mic, bonus (if there's time).** "Now the long Monday/Tuesday pair on screen." | Two more heard-back lines. Then (cell 52's successor) a two-bar chart: short keeps about half its first word's effect, long is down near a few percent | ~15-20s each (the long one takes longer to say and transcribe) | No mic or not Colab: plain sentence, fallback clips. |
| 58 | **Live mic, the headline.** "Say a SHORT 2-3 turn exchange. The first line holds one key fact. Pause ~2s between turns." | Turns printed back | ~20s | No mic or not Colab: plain sentence, fallback clip. |
| 60 | **Live mic.** "Now a LONG 8+ turn conversation, 40-60s, with the same first-line fact." | Two-bar chart: short keeps a clear %, long is near 0% | ~70-90s (a longer clip takes longer to transcribe) | Same fallback. |

Cells 58 and 60 are the headline Asy asked for. Don't cut them. If the block runs long, cut bonus cells 54-55 first.

## Block 6: LSTMs, an eraser you can set (00:50-01:10, 20 min)

| Cells | Say | Expect | Time | If it fails |
|---|---|---|---|---|
| 63 | "Five of these stacked, like an hourglass." | The layer-size table, then a bar chart (8/16/32/16/8) | <1s | `NameError: transcript`. Sections 1-2 weren't run. Run from cell 2. |
| 65 | "Here are the eraser, pencil and voice at work, word by word." | A bar chart (3 bars per word) and a printed table. `<eot>` marked | <1s | - |
| 67 | "That `<eot>` is the model's only clue that someone new is talking." | A token list with `<eot>` between two turns | <1s | - |
| 68 | "The hourglass folds four words into one step in the middle layer." | A fold/unfold shape check: `True` | <1s | - |
| 70 | **Live mic.** "Say two short turns, ~2s apart. A question, then a one-word answer." | A gate bar chart over YOUR words, `<eot>` in the labels | ~15s | No mic or not Colab: plain sentence, fallback clip. |
| 72 | **Live mic.** "Now put your Cats/Dogs pair from 4b through this same toy LSTM at three knob settings." | A 3-line plot. The gap lasts longest at `+6` and dies fastest at `-6` | <1s (reuses 4b's recording, no new prompt) | `NameError: CATS_LIVE`. Cells 43-44 (block 4b) weren't run. Run from cell 2. |

## Block 7: Teaching it, and catching it out (01:10-01:20, 10 min)

| Cells | Say | Expect | Time | If it fails |
|---|---|---|---|---|
| 75 | "Confident and right costs almost nothing. Confident and wrong costs a lot." | Two lines: `loss = 0.00` and `loss = 5.61` | <1s | - |
| 77 | "Keyword rules against the trained model, on a meeting it has never seen." | `keyword rules: 12/18`, `trained LSTM: 17/18`, one miss line | <1s | - |
| 80, 81, 82 | **Live mic.** "Say a request, a proposal, then a plain fact. One per cell." | Three lines: model label, confidence and loss, plus the keyword rule's answer | ~15s each | No mic or not Colab: plain sentence, fallback clips. |
| 85 | "Same two words, three answers. This is the hook's 'Sure.' again." | Three label/confidence lines | <1s | - |
| 88, 89 | **Live mic.** "Say 'Can you book the room? ... Sure.' Then 'Shall we move it to Friday? ... Sure.' Pause ~2s before each 'Sure.'" | Two label/confidence lines, one per exchange | ~20s each | No mic or not Colab: plain sentence, fallback clips. |

## Block 8: Hands-on B, adjust the eraser (01:20-01:40, 20 min)

| Cells | Say | Expect | Time | If it fails |
|---|---|---|---|---|
| 92 | "Same deal. One line, and the check confirms it." | `check: OK` | <1s | Wrong line. Point to the "Stuck?" note below (cell 93). |
| 96 | "Now sweep the knob. Watch both numbers move." | Three blocks (-6, 0, +6). Each has a toy gap row and a trained score | <1s | Stub unfilled: prints "not written yet" three times. Expected, not an error. |
| 98 | **Live mic.** "Say a short two-turn exchange. A question, ~2s pause, a short answer." | Three lines, one per knob setting. The label on YOUR exchange changes | ~15s | No mic or not Colab: plain sentence, fallback clip. |
| 101 | "Same idea through the toy LSTM's dial. It reuses 5c's recordings, no new prompt." | "KEPT" / "NOT KEPT" for short and long, not a raw percentage | <1s | `NameError: short_tokens`. Cells 58/60 (block 5c) weren't run. Run from cell 2. |
| 104 | **YOUR TURN.** "Set the dial so the long conversation comes back KEPT." | `check: OK` once the long one is "KEPT" | <1s | Stays "NOT KEPT". See the "Stuck?" note below (cell 105). Nothing below depends on it. |

## Block 9: Try it on your own meeting (01:40-01:55, 15 min)

| Cells | Say | Expect | Time | If it fails |
|---|---|---|---|---|
| 107 | "Now the whole pipeline, on a real two-speaker transcript." | Rendered minutes (headings, bullets) | <1s | - |
| 110 | "Write your own 5 lines. Add labels if you want to check yourself." | Label comparison, rendered minutes, unseen words, the prompt | <1s | Bad input (unlabelled mix, no colon, empty): a clean one-line message, no traceback. Re-check the format hint. |
| 112 | "Add one line of context. Watch the model's answer flip." | Two label/confidence lines, the second now correct | <1s | - |
| 115 | **Live mic.** "Talk a short 4-6 line meeting, solo (playing both people) or with a partner. Pause ~2s between lines. Up to 35s." | The transcript, as Speaker1 / Speaker2 | ~15-35s | No mic or not Colab: plain sentence, fallback clip. |
| 117 | **YOUR TURN.** "Label each of your turns act/dec/dis before you see the model's labels." | `check: OK` | <1s | Wrong count or a typo'd label. See the "Stuck?" note below (cell 118). The model's labels print either way. |
| 119 | (Continues from 117.) | Your label next to the model's for each turn, plus rendered minutes | <1s | - |
| 122 | "Does the TRAINED model's last answer move when you change line 1, or the line right before it?" | A two-bar chart. Both changes move it by a similar, modest amount. The trained model's context use is not a clean "only the previous line" rule | <1s | Falls back to the longer block 1-2 sample meeting if the spoken meeting (cell 115) had under 6 turns. |

## Block 10: Wrap-up (01:55-02:00, 5 min)

| Cells | Say | Expect | Time | If it fails |
|---|---|---|---|---|
| 124 | "Sound becomes words. Memory turns words into meaning. Meaning becomes minutes. And here's what it still gets wrong." | The one-sentence recap and three honest limits | - | - |
| 127 | *(Time permitting)* "In your own words: why does a long conversation's memory fade so much more than a short one's?" | What they wrote, printed back | <1s | - |
| 129 | "Here's what you built today." | A short list: prompts recorded, their dial value, their pair, their labels, their explanation (whichever they did) | <1s | A skipped piece just doesn't appear. Not an error. |

## Notes

- Not tested from this repo (no Colab or microphone here): live-mic recording (`demo/hook_demo.ipynb` cells 4/6 and every live cell in `audio_workshop.ipynb`), a real T4 GPU, and how anything sounds. Rehearse these before the day, like the hook checklist in `demo/FACILITATOR.md`. `plan/handoffs/HANDOFF-WP7.md` and `HANDOFF-WP8.md` list what to try on Colab with a real mic.
- The live-skit path is still unverified. Its checklist item in `demo/FACILITATOR.md` stays unticked until someone runs the rehearsal.
- The backup WAV's voices (Zira as Mei, Hazel as Ravi) are both female. Say the names aloud when you introduce the backup.
- If a participant's Colab runtime restarts, point them to the boxed note under the notebook's title. Run cell 2, then every cell above the one they want.
- **YOUR TURN cells (47, 104, 117, 127)** are optional and self-contained. Skipping one blocks nothing later. Cell 129's "What I built today" just lists fewer pieces.
