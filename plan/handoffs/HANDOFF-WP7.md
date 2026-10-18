# HANDOFF-WP7: live-mic cells for blocks 2-5 (say it, see it)

**Branch:** `claude/build-workshop-gaps-756dec`, merged up to `main` at
`9cb479b` (`git merge main`, fast-forward -- `main` added `plan/LIVE_MIC_SPEC.md`
and the WP6 reviews). **Nothing committed this round** -- same protocol. Scope:
`notebook/` and `demo/` only; `prototype/`, the model and the data stay frozen.

**Source:** `plan/LIVE_MIC_SPEC.md`'s WP7 line -- `notebook/live_mic.py`
(record, hear, say, fallback), fallback clips, live sections for blocks 2, 3,
4 and 5 following the Predict/Try it/See it/Why pattern (L1), at least 8
prompts total with at least 5 on the RNN side, run-sheet rows for the new
cells. Every number below was independently re-derived in this session, not
assumed from the spec -- one real design bug was found and fixed during
verification (see "A bug found and fixed," below).

---

## What was built

### `notebook/live_mic.py` (new)

Three calls, reused by every live cell (L3):
- `record(prompt_id, seconds=5)` -- a fixed ~5s browser-mic recording
  (eval_js + MediaRecorder, same pattern as `demo/hook_demo.ipynb`'s mic-stop
  and ffmpeg-stderr fixes) with a visible per-second countdown printed to the
  cell's own output, not press-Enter-to-stop (L5/L6).
- `hear(path, model)` -- transcribes with the Whisper `model` already loaded
  in section 1; this module never loads a second one (per the orchestrator's
  instruction).
- `say(prompt_id, model, seconds=5, typed=None)` -- records (or uses `typed`,
  or falls back to the shipped clip), then transcribes. Returns
  `(text, wav_path, used_fallback)`.
- `word_vector(word, dim=8)` -- a fixed, hash-seeded (`zlib.crc32`) random
  vector per word: the same word always gets the same vector, every run, and
  it is explicitly documented as NOT a learned embedding (L4).

**No-mic fallback (L2):** `record()` raises a plain `RuntimeError` -- never an
uncaught JS or ffmpeg traceback -- when this isn't Colab, there's no mic, or
permission is denied; `say()` catches it and falls back automatically to
`audio/live/<prompt_id>.wav`.

### `notebook/make_live_clips.py` (new) + `notebook/audio/live/*.wav` (new, 10 files)

Fallback clips, one per prompt_id, via Windows SAPI (same technique as
`demo/make_backup_wav.py`), using both voices already on this machine
(Microsoft Zira, Microsoft Hazel) so the shipped set itself uses more than
one voice. `block2_noisy.wav` is synthesised clean, then has synthetic hiss
mixed in afterwards (same technique as the notebook's own cells 16/18) --
the one prompt whose whole point is a live mic picking up real noise can't
be faked any other way for the no-mic path, so this is a pre-built clip that
already sounds noisy, documented as such in the script's docstring.

### Live sections added to `notebook/audio_workshop.ipynb` (20 new cells, 65 -> 85)

| Block | New cells | Prompts | Pattern |
|---|---|---|---|
| 2b (after cell 24, the Kahoot answer key) | 25-31 (7 cells) | 3: say a clear sentence (waveform + spectrogram); say it again with real noise (accuracy vs. your own clear recording); record one word, hear it backwards | Predict / Try it / See it / Why, each step |
| 3b (after cell 33, the toy-RNN step-through) | 34-36 (3 cells) | 1: say any sentence (5-15 words) -- watch a bar chart of the toy RNN's memory update at every one of your words | Predict / Try it / See it (bar chart) / Why |
| 4b (after cell 41, the amnesia cell) | 42-45 (4 cells) | 2: say "Cats are sitting near the window this evening," then the same line with "Dogs" -- compare memory ON vs. OFF on your own voice | Predict / Try it (x2) / See it (plot) / Why |
| 5b (after cell 47, the retention graph) | 48-53 (6 cells) | 4: say a 4-word first-word-swap pair, then a 16-word one -- overlay your own fade on the r^n illustration | Predict / Try it (x4) / See it (plot) / Why |

**10 live prompts total, 7 on the RNN side (blocks 3-5)** -- both minimums
(8 total, 5+ RNN) cleared with margin. `notebook/live_mic.py` is bundled at
the zip root (same level as `audio_workshop.ipynb`); `notebook/make_bundle.py`
now also bundles `audio/live/*.wav` and `live_mic.py` into both zips.

Part A, Part B (YOUR TURN), the knob, and the typed own-meeting cell are all
still exactly where they were -- the live cells sit next to them (L7), not in
place of them; their own cell numbers simply shifted down by the number of
live cells inserted before them (see `demo/RUNSHEET.md`'s corrected table
below).

### `demo/RUNSHEET.md`: every cell number from block 3 onward corrected, new live-cell rows added

Inserting 20 cells shifts every later cell index. Recomputed every reference
in blocks 3-10 against the actual new notebook (not by arithmetic -- read the
real cell-type/source map), and added rows for cells 26-30 (block 2b), 35
(block 3b), 43-44 (block 4b), and 49-52 (block 5b), each with its own "if it
fails" note (no mic / not Colab -> fallback clip, plain sentence, never a
traceback).

---

## A bug found and fixed (cell 44, block 4b)

The first version of cell 44 printed:
```python
print(f"memory ON:  {gaps_on[-1]:.2f} apart at word {n_live} -- {'still tells them apart' if gaps_on[-1] > 0.05 else 'forgot too'}")
```
Verifying this against real recorded word pairs (not the artificial, scale-
controlled vectors the original retention-graph illustration uses) showed the
gap with memory ON reliably ends up tiny in absolute terms by word 8 (0.011,
0.0005 across the cases tested below) -- well under the 0.05 threshold, so
the cell would print **"forgot too"** for memory ON, directly contradicting
the point of the experiment (memory ON is supposed to look better than OFF,
not equally forgotten). The plot had the same problem: word 1's gap (the two
words "cats"/"dogs" themselves, no memory mixed in yet) is roughly an order
of magnitude bigger than every later word, so on a linear y-axis the ON/OFF
contrast from word 2 onward was visually flattened into the bottom of the
plot -- not "visible at a glance" as L1/the PM's priority #1 requires.

**Fixed:** the plot now starts at word 2 (word 1 is reported separately, in
the printed text) so the ON-shrinking vs. OFF-exactly-zero contrast is
visible at normal scale; the printed line now states the real, robust
distinction instead of a threshold that silently flips sign on tiny numbers:
```python
print(f"memory ON:  gap right after the swap (word 2) = {gaps_on[1]:.2f}, still {gaps_on[-1]:.3f} by word {n_live} -- shrinking, but never exactly zero")
print(f"memory OFF: gap right after the swap (word 2) = {gaps_off[1]:.2f}, {gaps_off[-1]:.3f} by word {n_live} -- exactly 0.000 from word 2 on, every time")
```
This is the TRUE, robust distinction (confirmed on 2 independent word pairs
below): memory ON is never exactly zero; memory OFF is exactly 0.000 from
word 2 on, always. Screenshot `03_block4_memory_on_off.png` shows the fixed
plot.

---

## L4 honesty verification (TTS-injected, 2 voices, varied inputs)

No mic in this build session -- every prompt tested through the real code
path by monkeypatching `live_mic.record` to return pre-synthesised WAVs
(Windows SAPI, both installed voices), per the spec's Verification section.

### Block 3b (memory updates at every word): 4 sentences, not just the shipped one

| Sentence (voice) | Heard | Words | Any flat step? |
|---|---|---|---|
| Shipped (Zira) | "The team decided to postpone the launch until March." | 9 | No |
| Alt 1 -- filler, different first word (Hazel) | "Um, so I think we should probably just wrap this up soon." | 11 | No |
| Alt 2 -- very short (Zira) | "Quick update." | 2 | No |
| Alt 3 -- long, different voice (Hazel) | "Honestly the whole thing could have been one short email instead of a meeting." | 14 | No |

The memory bar chart changed at every single word in all 4 cases -- 2 words
to 14, a filler ("Um"), a different first word each time. Screenshot
`02_block3_memory_bar.png` is the shipped sentence's chart.

### Block 4b (memory ON vs. OFF): 2 independent first-word-swap pairs

| Pair | Heard | ON word-2 gap | OFF word-2 gap | ON last gap | OFF last gap |
|---|---|---|---|---|---|
| Shipped: Cats/Dogs ... window this evening | matched exactly | 0.72 | 0.000 | 0.011 | 0.000 |
| Alt: Cats/Dogs ... nap in the sunniest spot | matched exactly | (confirmed nonzero) | 0.000 | 0.0005 | 0.000 |

Both pairs: memory OFF is **exactly** 0.000 from word 2 on (every element of
`gaps_off[1:]` is 0.0, asserted in the verification script, not eyeballed);
memory ON is **never** exactly 0, however small it gets. This holds
regardless of which first-word pair is spoken -- the honest, robust version
of the aha (see "A bug found and fixed" above for why the first draft's
framing was wrong).

### Block 5b (short vs. long fade): 2 independent short/long pairs

| Pair | Short heard | Long heard | Short last % | Long last % |
|---|---|---|---|---|
| Shipped | "Red/Blue cars drive fast." (4 words) | "Monday/Tuesday mornings ... this term." (16 words) | 3% | 0% |
| Alt | "Old/New clocks tick loud" (4 words, ASR misheard "ticked out" on one word -- still 2 words, didn't break the pairing) | "Winter/Summer evenings ... properly." (16 words) | 39% | 0% |

Short retains more than long in both cases, including the ASR-noisy alt
case -- the aha survives a real transcription slip, not just clean input.

### Bad audio: silence, noise, no-mic fallback

| Input | Result |
|---|---|
| 3s of silence | Whisper hallucinated a short word ("You"); no crash, no traceback |
| 3s of synthetic noise (no speech) | Same -- a hallucinated short word, no crash |
| `live_mic._is_colab()` forced `False` (no mic / not Colab) | `record()` raised the plain `RuntimeError` message; `say()` caught it, printed "Using the fallback clip instead: audio/live/block2_clear.wav", and returned the correct fallback text -- confirmed the returned path actually points at that file |

---

## Full verification

### `compile()`, every cell

```
notebook/audio_workshop.ipynb cells: 85 errs: 0
```

### Both zips, rebuilt

```
$ python notebook/make_bundle.py
wrote .../notebook/workshop_files.zip (4.0 MB, 28 files)
wrote .../notebook/facilitator_files.zip (5.0 MB, 31 files)
```
(28 = the WP6 17 + `live_mic.py` + 10 fallback clips.)

### Whole notebook from `workshop_files.zip`, empty folder, no-mic path forced (`live_mic._is_colab` patched `False` so every live cell takes its real fallback branch, same code path a no-mic participant gets) -- unsolved and solved

```
unsolved: 0 errors (all 85 cells)
solved:   0 errors (all 85 cells, cells 38/70 -- Part A/B -- filled with the reference lines)
```
Every live cell correctly printed "No live mic here ... using the fallback
clip instead" and used the matching `audio/live/<prompt_id>.wav`; every
existing cell (Whisper, the Kahoot clips, the gate plot, the Sure. test, the
knob, the own-meeting cell) still ran exactly as in WP6's last verified
state, with identical numbers (56%/44% "Sure." split, 9.0% retention at
n=47, etc.).

### `run_tests.py` and the frozen-file check

```
PASS  a. imports ... PASS  m. repo unchanged
12 passed, 0 failed, 1 skipped in 31.2s
```
```
$ git diff --stat main -- prototype/ data/
(no output -- nothing frozen was touched)
```

### Screenshots (`plan/handoffs/wp7/`, 4 files -- one per block's "See it")

```
01_block2_waveform_spectrogram.png  Your own waveform + spectrogram (block 2b, cell 26)
02_block3_memory_bar.png            The toy RNN's memory, one bar per word you said (block 3b, cell 35)
03_block4_memory_on_off.png         Memory ON vs. OFF on your own Cats/Dogs pair, fixed plot (block 4b, cell 44)
04_block5_fade_overlay.png          Your short vs. long pair overlaid on the r^n curve (block 5b, cell 52)
```

---

## What a person must try on Colab with a real mic

This build session has no microphone or Colab access -- everything above is
verified through the real code path with TTS-injected audio, not a real
browser mic. On the actual day (or in a rehearsal before it), a person with
Colab and a working mic should:

1. Upload `workshop_files.zip` and `audio_workshop.ipynb`, run cell 2 (Setup),
   confirm `ready`.
2. At cell 26 (block 2b): say a clear sentence out loud when the countdown
   starts -- confirm the countdown actually counts down in the cell's output
   (not just in theory), the recording stops on its own after ~5s with no
   press-Enter, and the waveform/spectrogram show YOUR voice.
3. At cell 28: say the same sentence with real background noise (tap the
   desk) -- confirm the printed accuracy is visibly below 100%.
4. At cell 30: say one word, confirm both the forwards and backwards audio
   players actually play in the browser.
5. At cells 35, 43-44, 49-52: say the prompted sentences -- confirm each
   bar chart / plot renders with real data, not a stale fallback (the printed
   "heard:" line should match what was actually said, not the shipped
   fallback text, when a real mic is used).
6. Deny microphone permission once, on purpose, at any one live cell --
   confirm it prints the plain fallback message (not a raw JS exception) and
   still produces a working plot from the shipped clip.
7. Confirm the Colab tab doesn't need anyone to click "Record" or press Enter
   to stop -- the recording should start and stop entirely on its own once
   the cell is run.

---

## Not checked / open

- No Colab, no real browser mic, no T4 GPU available from this build
  session -- everything above used TTS-injected WAVs through the real code
  path, not an actual recording. See "What a person must try" above.
- Timing on an actual T4: the ~5s record + Whisper transcribe estimate
  (~15s per prompt, L5) is based on this machine's CPU `base.en` runs: cell
  26 took ~4s end-to-end here (base.en, CPU, after the model was already
  warm) -- a T4 with `small.en` should be comparable or faster, but this
  hasn't been measured on one.
- The live cells are additive (20 new cells); this pushes blocks 2-5's total
  time up by roughly 1-2 minutes if all of them are run (2b is marked
  optional in the notebook and in the run-sheet for exactly this reason) --
  the PM/orchestrator may want to adjust the Agenda's block timings in
  `plan/SPEC.md` to reflect this, which is outside this package's scope
  (`demo/` and `notebook/` only).
- `block2_noisy`'s fallback clip is a synthesised "noisy" recording (clean
  TTS + added hiss), not a live mic actually picking up real noise -- flagged
  honestly in `make_live_clips.py`'s docstring; a live mic in the room will
  produce a different (and likely more convincing) noisy clip.
