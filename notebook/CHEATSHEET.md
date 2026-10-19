# Workshop cheat sheet

*Print this one page (or keep it open in a second tab). Also the notebook's own cell 0.*

## Getting started

1. Open **colab.research.google.com** -> **File > Upload notebook** -> pick `audio_workshop.ipynb`.
2. **Runtime > Change runtime type > T4 GPU > Save** (restarts the runtime).
3. Upload `workshop_files.zip` to the Files panel (left sidebar) when asked.
4. Run cell 2 (Setup). It prints `ready` when done -- if it instead says "Upload workshop_files.zip...", you uploaded the notebook but not the zip (or to the wrong panel).
5. Run cells top to bottom from there.

## Running a live-mic cell

- Run the cell. It prints a countdown (`5... 4... 3...`) -- **that's your cue to speak**, close to the laptop, not loud.
- **Allow the microphone** the first time Chrome asks (top-left of the browser tab).
- Wait for `done.` before the next cell.
- No mic, denied permission, or not on Colab? The cell prints a plain sentence and uses a pre-recorded fallback clip automatically -- nothing breaks, just follow along on the printed/plotted result instead of your own voice.

## The live prompts, block by block (what you should see)

| Block | Cells | Say | You should see |
|---|---|---|---|
| 2 *(optional)* | 26, 28, 30 | a clear sentence; the same with noise; one word | your waveform/spectrogram; an accuracy %; forwards/backwards audio |
| 3 | 35 | any sentence, 5-15 words | a bar chart, the memory **changing** at every word |
| 4 *(bonus)* | 43-44 | "Cats are..." / "Dogs are..." (same ending) | ON decays smoothly, OFF flat at 0% from word 2 |
| 5 | 49-50, *(bonus)* 51-52 | "Red/Blue cars drive fast"; *(bonus)* the long Monday/Tuesday pair | a two-bar chart: short keeps much more than long |
| 6 | 62, 64 | a 2-turn exchange, ~2s pause | gate bar chart, `<eot>` marked; knob sweep reuses block 4's recording |
| 7 | 72-74 | a request; a proposal; a plain fact | label + confidence + loss for each |
| Sure. | 80-81 | "Can you book the room?.../Sure."; "Shall we move it to Friday?.../Sure." | the same word, two different labels |
| 8 | 90 | a 2-turn exchange | the label changes as the knob moves |
| 9 | 101 | your own 4-6 line meeting | transcript, labels, rendered minutes |
| **Headline** | 104, 106, 108, 110 | a short exchange (2-3 turns); then a long one (8+ turns, ~40-60s) | short keeps a lot, long keeps ~0% -- on the toy **and** the real trained model |

## If it goes wrong

| What happens | What to do |
|---|---|
| A red error (traceback) | Re-run cell 2, then every cell above the one you're stuck on, in order (the notebook reuses things earlier cells set up). |
| Colab disconnects / you jump ahead | Same fix: cell 2, then every cell above, top to bottom. |
| No mic, or you said no to the permission prompt | Nothing to do -- the cell already fell back to a pre-recorded clip and kept going. |
| "Whisper heard almost nothing..." | Say it again, closer to the mic and a bit louder, or just let the fallback run. |
| A `YOUR TURN` cell says `not written yet` | That's expected until you fill in the one line -- see the hint below. |

## YOUR TURN hints (not the answers)

- **Part A (block 4):** `new memory = tanh( W_x @ x + W_h @ h_prev + b )` -- one line, `@` is matrix multiply, `tanh` is `np.tanh`.
- **Part B (block 8):** `new notebook = (what you keep) * (old notebook) + (what you add) * (new note)` -- `c = f * c_prev + i * g`.

Stuck either way? Each cell has a collapsed **"Stuck?"** note right below it with the real mistakes people make and what they look like.

## What to take home

- The repo (ask the facilitators) -- the full code, including training, is with them, not in this zip.
- `notebook/live_mic.py` and `notebook/toy_memory.py` if you want to reuse the live-mic or toy-memory pattern elsewhere.
- The one-sentence version: **sound becomes words, memory turns words into meaning, meaning becomes minutes** -- and a real model's memory reaches about one turn back, not arbitrarily far.
