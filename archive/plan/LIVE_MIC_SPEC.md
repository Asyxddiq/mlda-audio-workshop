# Live-mic spec: every theory block gets "say it, see it" (WP7 and WP8)

Asked by Asy, 19 Oct 2026: "live-mic cells for ALL theory blocks ... 3-5 prompts ... RNN 5+ ... after testing and
they see the results ... they're like DAMN that's right (they can see first-hand the theory we taught them)."

Scope is still `notebook/` and `demo/` only. `prototype/`, the model and the data stay frozen.
New helper code goes in `notebook/live_mic.py`, bundled into `workshop_files.zip`.

## The pattern for every block (L1)

Each block's live section has four parts, in this order:

1. **Predict**: one markdown line asking the participant to guess first ("Will the end of the sentence still remember
   the first word?").
2. **Try it**: numbered spoken prompts. Each prompt is one cell that records the participant's mic (Colab
   `eval_js` + MediaRecorder, the same pattern as `demo/hook_demo.ipynb`), shows what Whisper heard, and runs
   the experiment on it.
3. **See it**: output built so the answer is visible at a glance. Prefer a plot or a 2-3 line comparison to a
   table of numbers. Put the key number in words, for example "the first word is 92% forgotten by word 12".
4. **Why**: one or two sentences tying what they saw to the theory slide.

## Rules

- **L2: Every live cell has a no-mic fallback.** A `TYPED = None` override, plus a pre-recorded synthetic clip for
  the same prompt in `notebook/audio/live/`. If there is no mic, it isn't Colab, or permission is denied, the cell
  still produces the aha. `record()` must fail with a plain sentence, never a traceback.
- **L3: One shared helper.**
  - `live_mic.record(seconds or press-to-stop)` returns a WAV path.
  - `live_mic.hear(path)` returns the text, using the already-loaded Whisper model.
  - `live_mic.say(prompt_id)` does both, plus the fallback.
  - Every block reuses these three, with no copy-pasted recorder code.
- **L4: Honest experiments only.**
  - The toy RNN and LSTM are untrained. Say so where it matters.
  - Words become vectors through a fixed, seeded lookup, for example a hash-seeded random vector per word: the same
    word gives the same vector every run. Say plainly that this stands in for learned embeddings.
  - Every "aha" must be true for ANY reasonable sentence a participant says, not only the scripted one. Test it on
    varied inputs: different lengths, a different first word, filler words.
  - Trained-classifier experiments use the real ensemble through `lstm_model.label_probabilities`.
  - Never claim the trained LSTM remembers further back than an RNN (SPEC tension 3).
- **L5: Time.** Each prompt takes about 15 s or less: about 5 s speaking plus Whisper on a T4 (`small.en`). The
  prompts must fit inside the block's minutes. Mark extra prompts "(bonus)" so a facilitator can skip them.
- **L6: Room reality.** 30 laptops in one room. Prompts are short, spoken close to the laptop, and the run-sheet says
  so. Two-speaker prompts (pause 2 s, then speak as the other person) are solo-friendly, since one person plays both.
- **L7: Keep the existing hands-on.** Parts A and B (YOUR TURN), the knob, and the typed own-meeting cell stay. The
  live cells sit next to them, not in place of them.

## Blocks and minimum prompts

| Block | Theory | Live prompts (minimum) | The "DAMN" moment it must deliver |
|---|---|---|---|
| 2 Sound to words | Spectrogram; Whisper reads it; noise and mic and accent hurt | **3-5**: say a clear sentence (see YOUR waveform and spectrogram, and your word accuracy); whisper it; say it fast; say it with noise (tap the desk, or music on); record a word and hear it backwards | Their own voice as a picture; accuracy drops in front of them when they whisper or add noise; their word backwards is unrecognisable, so order matters |
| 3 RNN running summary | Reads one word at a time, carries a summary | **5+** (blocks 3-5 together): say a sentence and watch the memory update word by word (one line per word, or a small heat strip) | The summary visibly changes at every word they said |
| 4 Hands-on A (memory lives in W_h·h_prev) | The memory term | Say two sentences that differ ONLY in the first word ("Cats..." / "Dogs..."), then compare memory on vs memory off | With memory on, the gap shrinks word by word. With memory off, it is 0 from word 2: the first word is gone instantly |
| 5 Memory fades | r^n | Say a SHORT pair (4 words) and a LONG pair (15+ words), same first-word swap | Short: the first word still matters at the end. Long: about 0% left. Overlay their measured fade on the r^n curve |
| 6 LSTM gates | eraser, pencil, voice; `<eot>` | **5+** (blocks 6-8 together): say two turns with a 2 s pause (for example "Can you book the room?" / "Sure."), plot eraser and pencil over THEIR words with `<eot>` marked; the same Cats/Dogs pair through the toy LSTM at knob 0, -6, +6 | The eraser dips where they changed speaker; the knob decides whether their first word survives |
| 7 Training and catching it out | loss = -log p; keyword vs model | Say 3 lines (a request, a proposal, a plain fact): show each label, its confidence, and the loss if it's wrong; keywords vs model | They catch the model out live on a decision stated as a fact |
| 8 Hands-on B (knob) | both extremes hurt | Say a 2-turn exchange; turn the knob; see the trained model's label on THEIR exchange change | Erasing too much loses the line before; keeping too much muddles it |
| "Sure." (inside block 6 or 7) | Context | Say "Can you book the room?" ... "Sure."; then "Shall we move it to Friday?" ... "Sure." | The same spoken word gets different labels. Honest: the second is wrong, and the notebook says so |
| 9 Your own meeting | End to end | Talk a 4-6 line meeting, alone or in pairs, then see the transcript, labels, minutes and mismatches; the typed cell stays as the fallback | Their spoken meeting becomes minutes |

## Packages

- **WP7:**
  - `live_mic.py` (record, hear, say, fallback);
  - fallback clips;
  - blocks 2, 3, 4 and 5 (at least 8 prompts in total, with at least 5 on the RNN side);
  - bundle;
  - run-sheet rows.
- **WP8:**
  - blocks 6, 7, 8, the "Sure." prompt, and 9;
  - the final full run;
  - the run-sheet;
  - the RUNSHEET zip-timing fix (the participants' zip has no demo answers, so it is safe to share early).

## Verification (both packages)

There is no mic in the build sessions. Test every prompt through the real code path by injecting WAV files in place
of `record()`:

- **Synthetic speech:** Windows TTS, with at least 2 voices.
- **Varied content:** at least 3 different sentences per experiment, to prove L4 holds beyond the script.
- **Bad audio:** silence, noise, and "no mic" (the fallback path).

Then:
- Run the whole notebook from `workshop_files.zip` in an empty folder (unsolved and solved).
- `compile()` every cell.
- Run `run_tests.py` and the frozen check.
- Screenshot every "See it" output in `plan/handoffs/wp7/` (or `wp8/`).

The live browser mic on Colab is checked by a person, and the handoff lists exactly what to try.
