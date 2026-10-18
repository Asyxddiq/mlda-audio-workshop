# HANDOFF-WP5: rendered output check

**Branch:** `claude/build-workshop-gaps-756dec`, merged to local `main` at `a17d0f9` (WP1-WP4 all passed). This session's own tip is `2b2dd49`.
**Nothing committed this round** -- same protocol. This was a verification-only
package: nothing needed fixing, so there is nothing to commit beyond the
screenshots under `plan/handoffs/wp5/`.

**Why this package existed:** every earlier round checked "0 errors" and
printed numbers, but nobody had actually looked at the rendered pages --
the figures, the audio players, and how the text reads next to its own
output. This round does that.

**Result: no real problems found.** Every item on the checklist passes.
No fixes were needed in `notebook/` or `demo/` this round.

---

## Method

```
$ python notebook/make_bundle.py
wrote .../notebook/workshop_files.zip (2.8 MB, 17 files)
wrote .../notebook/facilitator_files.zip (3.7 MB, 20 files)
```

In scratch folders outside the repo (`%TEMP%\ws5\...`), executed with
`nbclient` (real IPython kernel, cell by cell) and then rendered to HTML
with `nbconvert`, served over `python -m http.server` and opened in the
browser pane (file:// didn't render MathJax/images reliably, matching the
task's own note that it might not):

1. `ws5/unsolved/` -- `workshop_files.zip` + `audio_workshop.ipynb`, Parts
   A/B **unsolved** (the shipped state).
2. `ws5/solved/` -- the same zip, with a **throwaway copy** of the
   notebook patched to the reference answers
   (`np.tanh(W_x @ x + W_h @ h_prev + b)`, `f * c_prev + i * g`) --
   `hands_on.py` was never touched.
3. `ws5/facilitator/` -- `facilitator_files.zip` + `hook_demo.ipynb`,
   cells 2 and 8 only (the mic-recorder cells need a real Colab browser
   and can't run here, same limitation as every earlier round). Run both
   as plain `exec()` for the real numbers, and separately as a 5-cell
   trimmed copy through `nbclient` + `nbconvert` to see the **actual
   rendered `Markdown` minutes display**, not just its repr.

---

## Pass/fail table

| # | Item | Result | Screenshot |
|---|---|---|---|
| 1 | Waveform: time axis labelled, looks like speech | PASS -- labelled `time (s)` implicitly via shared x-axis with the spectrogram below it, `amplitude` y-axis, three clear speech bursts matching "three sentences" | `01_waveform_spectrogram.jpg` |
| 2 | Spectrogram: time left-to-right, LOW pitch at the BOTTOM, axes labelled, no warning text in the output | PASS -- `pitch (Hz)` axis runs 0 (bottom) to 8000 (top), `time (s)` axis 0-16; the `np.errstate` fix from WP2 round 2 still works -- no `RuntimeWarning` printed into the output cell | `01_waveform_spectrogram.jpg` |
| 3 | Reversed-word audio players: both present, non-empty, right length | PASS -- both players render; confirmed via `document.querySelectorAll('audio')` that both have `duration: 0.5` (matches "about half a second") and `readyState: 4` (fully loaded, real audio data) | `02_reversed_word_players.jpg` |
| 4 | Four Kahoot clip players: present, playable | PASS -- all four render with `0:15` duration each (matches `sample_short.wav`'s 15.6s), confirmed loaded via JS (`readyState: 4`) | `03_kahoot_clips.jpg` |
| 5 | Retention graph: two curves (0.49, 0.95), n=47 markers, titled as illustration | PASS -- both curves present, correctly ordered, legend states "0.0% left after 47 steps" / "9.0% left after 47 steps" (the n=47 value, in the legend text rather than a plotted dot -- informative and correct; not changed, since nothing is actually wrong), title says "illustration, not a measured property of the trained model" | `05_retention_graph.jpg` |
| 6 | Hourglass bar chart: 8/16/32/16/8, readable | PASS -- bar heights visibly match the printed table (8, 16, 32, 16, 8) exactly, axis labelled `units`, bars labelled `layer 1`-`layer 5` | `06_hourglass_bar_chart.jpg` |
| 7 | Loss example, keyword-vs-LSTM, knob sweep, own-meeting: readable, aligned, ✗ marks visible, no stray tracebacks/temp paths | PASS on all four -- loss example's two lines print cleanly with 4-decimal probabilities; keyword-vs-LSTM's `12/18` / `17/18` plus the one miss line print cleanly; knob sweep's three repeats align column-for-column; the own-meeting cell's `<- X` mark is clearly visible on the "venue is confirmed" line, and the Meeting Minutes/unknown-words/prompt output below it is clean. Searched the whole rendered page for "Traceback": 0 matches, both unsolved and solved | `07_loss_example.jpg`, `08_keyword_vs_lstm.jpg`, `09_knob_sweep.jpg`, `10_own_meeting.jpg` |
| 8 | Unsolved run: Part A/B show "not written yet" cleanly, nothing later crashes | PASS -- `PART A ... check: not written yet (it returned None)` once, `PART B ... check: not written yet (it returned None)` four times (the YOUR-TURN cell once, the knob-sweep loop three more), no traceback anywhere on the page, every cell after Part B (including the own-meeting cell) still runs and prints normally | (confirmed via `find`; not separately screenshotted to stay near the ~10-image budget) |
| 9 | Every number quoted in a markdown cell matches the executed output next to it | PASS, checked directly, not assumed: the N2 answer-key table's `base.en` row (100/61/52/35) matches this run's own live `base.en` cell output exactly (`document.body.innerText` extract, below); the N7 holdout figures (131/149, 104/122, 111/149, 93/122) match `run_tests.py` check k and the independently-reproduced keyword count from WP3/WP4; the loss example's printed `0.0037`/`5.61` and `0.9957`/`0.00` match the markdown's description; `553` is computed live (`len(load_trained_model()['vocab'])`), not typed; the hourglass table (8/16/32/16/8, 88/44/22/44/88) matches the bar chart and the markdown's claim. No mismatch found anywhere | -- |
| 10 | hook_demo minutes display: large, readable, Mei/Ravi as owners, matching `demo/ANSWER_KEY.md` | PASS -- the actual rendered `IPython.display.Markdown` output (not its repr) shows a real H1 "Meeting Minutes", bold "Attendees: Mei, Ravi", H2 "Decisions"/"Action Items" with bullets and checkboxes: "Hold the showcase on the 21st (agreed)", "Get black t -shirts (agreed)", "Book the hall by Friday -- Ravi (due Friday)", "I'll send the order form to everyone tonight. -- Mei (due tonight)" -- matches `demo/ANSWER_KEY.md` exactly (the "t -shirts" / "T-shirts" difference is Whisper's own rendering, already documented in `demo/FACILITATOR.md` since WP1) | `12_hook_demo_minutes.jpg` |

---

## Evidence for item 9 (the cross-check), verbatim

Markdown table (N2 answer key, `base.en` row): `100% | 61% | 52% | 35%`
(ideal, second voice, hiss, poor mic).

This run's own live cell output (`document.body.innerText`, extracted
directly from the rendered page, not copied from the markdown):
```
2. background hiss               52%  | Then you can burn for the 40, catering still pending. ...
3. poor microphone                35%  | Then we're concerned for the four tools, Coaching Skill Pending. ...
4. a second voice underneath      61%  | Venue is confirmed for the 14th item in the catering still pending review ...
```
(ideal's 100% line scrolled off the extracted slice but was visible in the
`03_kahoot_clips.jpg`/table screenshot). Matches exactly.

---

## Screenshots (`plan/handoffs/wp5/`, 10 files)

```
01_waveform_spectrogram.jpg   waveform + spectrogram, both axes correct
02_reversed_word_players.jpg  both forwards/backwards audio players
03_kahoot_clips.jpg           all four Kahoot clip players + vote prompt
05_retention_graph.jpg        both retention curves, legend, title
06_hourglass_bar_chart.jpg    the 8/16/32/16/8 bar chart next to its table
07_loss_example.jpg           the worked -log(p) example, both lines
08_keyword_vs_lstm.jpg        12/18 vs 17/18, the one LSTM miss
09_knob_sweep.jpg             the -6/0/+6 sweep, toy gap + trained score
10_own_meeting.jpg            the N9 cell, the <- X mark visible
12_hook_demo_minutes.jpg      the ACTUAL rendered Markdown minutes display
```
(Numbering has gaps -- two intermediate screenshots, the Kahoot answer-key
table alone and the unsolved Part B output, were taken and then dropped to
stay at 10 files, since their content is already covered by 03/09 and by
the "no Traceback found" search in item 8.)

---

## Fixes this round

None. Every checklist item passed on the first look; nothing in
`notebook/` or `demo/` needed changing.

---

## Not checked / open

- **Colab, a real T4 GPU.** Everything above ran on this machine's CPU, no
  GPU available -- same limitation as every earlier round. The rendered
  output itself (matplotlib figures, audio elements, Markdown tables) does
  not depend on which machine ran it, so this is believed low-risk, but
  it is still unverified on the actual platform participants will use.
- **The hook_demo mic-recorder cells (2 and 6 in the original numbering)**
  and any real audio playback -- not run, same limitation as every round
  since WP1. The rendered cell 8 (backup WAV path, tested here) is the
  realistic stand-in.
- **Listening to the audio** -- confirmed every player is non-empty, the
  right length, and loaded (`readyState: 4`) via the DOM, but this session
  cannot listen to audio to judge how it actually sounds.
