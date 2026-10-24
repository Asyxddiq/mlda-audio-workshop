# HANDOFF-WP5: rendered output check

**Branch:** `claude/build-workshop-gaps-756dec`, merged to local `main` at `a17d0f9` (WP1-WP4 all passed). This session's own tip is `2b2dd49`.
**Round 1 was committed by the orchestrator as `e89780b`** (11 files, all
under `plan/handoffs/`: this handoff plus the 10 screenshots -- no code
changed, same as the frozen check reported). Round 2 (below) is, as
always, left uncommitted for the orchestrator.

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
| 4 | Four Kahoot clip players: present, playable | PASS -- all four render with `0:15` duration each (matches `sample_short.wav`'s 15.6s), confirmed loaded via JS (`readyState: 4`). **Correction (round 2):** `03_kahoot_clips.jpg` itself is cropped to the vote prompt and only the first ("1. ideal") player -- the claim that all four load is true and was checked in the DOM, but the screenshot does not show all four; see the caption fix below | `03_kahoot_clips.jpg` (cropped -- see caption) |
| 5 | Retention graph: two curves (0.49, 0.95), n=47 markers, titled as illustration | PASS -- both curves present, correctly ordered. **Round 1's legend covered the start of both curves and the title overflowed the figure (reviewer nit 1, fixed in round 2 -- see below); round 2's image shows the fixed version.** | `05_retention_graph.png` |
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

## Screenshots (`plan/handoffs/wp5/`, 11 files after round 2)

```
01_waveform_spectrogram.jpg   waveform + spectrogram, both axes correct
02_reversed_word_players.jpg  both forwards/backwards audio players
03_kahoot_clips.jpg           the vote prompt + the first ("1. ideal") Kahoot player only --
                              cropped before the other three; all four were confirmed to
                              load, with different content, via the DOM, not from this image
05_retention_graph.png        both retention curves, legend, title (round 2: re-taken)
06_hourglass_bar_chart.jpg    the 8/16/32/16/8 bar chart next to its table
07_loss_example.jpg           the worked -log(p) example, both lines
08_keyword_vs_lstm.jpg        12/18 vs 17/18, the one LSTM miss
09_knob_sweep.jpg             the -6/0/+6 sweep, toy gap + trained score
10_own_meeting.jpg            the N9 cell, the <- X mark visible (round 2: re-taken, now
                              display(Markdown(...)))
11_section5_minutes.jpg       NEW (round 2): section 5's minutes, now display(Markdown(...))
12_hook_demo_minutes.jpg      the ACTUAL rendered Markdown minutes display (round 2:
                              re-taken, FP16 warning gone)
```
(Numbering has gaps -- two intermediate round-1 screenshots, the Kahoot
answer-key table alone and the unsolved Part B output, were taken and then
dropped to stay near 10 files, since their content is already covered by
03/09 and by the "no Traceback found" search in item 8. This now totals
11 files after round 2's addition of 11_section5_minutes.jpg.)

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

---

## Round 2 (review verdict: PASS on `e89780b`, 4 cosmetic nits; 3 fixed now, 1 was a handoff-only correction)

Reviewer's findings: `git show claude/review-workshop-gaps-53e548:reviews/WP5.md`.
Orchestrator asked for findings 1-3 to be fixed (visible to the room or
participants) and finding 4 (a handoff caption/header slip) corrected in
the text -- both done, independently re-verified. Nothing committed.

| # | Finding | Fix applied | Verified |
|---|---|---|---|
| 1 | Retention graph's legend covered the start of both curves (top-left, where they both start at 100%); the title overflowed the figure's width | First tried `loc="center right"` alone -- the legend box was still wide enough (long labels) to reach across most of the figure regardless of anchor side. Shortened the labels ("untrained setting" / "keeping 95%", "left at n=47" instead of the long parenthetical, which the surrounding markdown already explains) and moved to `loc="lower right"`, the one corner where both curves are already near zero by the time the plot gets there. Shortened the title to one line and dropped it to `fontsize=11` so it fits inside the axes | Extracted the actual PNG straight from the executed notebook's output (not a browser screenshot, to rule out any viewport cropping) -- legend sits cleanly in the empty bottom-right corner, doesn't touch either curve, title fits on one line, "illustration" kept |
| 2 | Participants' minutes cells (section 5, own meeting) printed raw Markdown (`# Meeting Minutes`, `**Attendees:**`, ...) instead of rendering it, unlike the facilitator notebook | Both cells now `from IPython.display import Markdown, display` and `display(Markdown(generate_minutes(...)))`; the label-comparison table above each stays as printed text (per the orchestrator's instruction) | Re-rendered both: section 5 and the own-meeting cell now show real H1/H2 headings, bold attendees, bulleted/checkboxed items -- same polish as `hook_demo.ipynb` |
| 3 | A red `UserWarning: FP16 is not supported on CPU` box appeared above the hook_demo minutes (CPU-only artifact; a T4 supports fp16) | Added `import warnings; warnings.filterwarnings("ignore", message="FP16")` to both `hook_demo.ipynb`'s and the participants' notebook's setup cells -- matches only that message, not a blanket warning silence | Searched both re-rendered pages: `UserWarning` 0 matches, `Traceback` 0 matches; the string `"FP16"` appears only inside the displayed source code of the `filterwarnings` call itself |
| 4 | `HANDOFF-WP5.md`: screenshot 03's caption claimed it shows all four Kahoot players (it's cropped to the first one); the header said "Nothing committed" while round 1's files are in `e89780b` | Fixed both: the pass/fail table and the screenshot list now say the image is cropped to the vote prompt + the first player, with the "all four load" claim attributed to the DOM check, not the image; the header now names `e89780b` as round 1's commit | Re-read the file |

### Re-render and the four changed spots, empty folders, fresh zips

```
$ python notebook/make_bundle.py
wrote .../notebook/workshop_files.zip (2.8 MB, 17 files)
wrote .../notebook/facilitator_files.zip (3.7 MB, 20 files)

$ <nbclient, empty folder + workshop_files.zip + the notebook, Part A/B solved inline (throwaway copy)>
... (53 cells)
=== OK in 78.4s ===

$ <nbclient, empty folder + facilitator_files.zip + a 5-cell trim of hook_demo.ipynb (cells 0,1,2,7,8 -- skipping the Colab-only mic cells)>
... (5 cells)
=== OK in 97.2s ===
```

Both rendered to HTML with `nbconvert` and opened in the browser pane,
same as round 1. New/replaced screenshots: `05_retention_graph.png` (the
fixed legend/title, pulled directly from the executed notebook's PNG
output rather than a page screenshot), `10_own_meeting.jpg` (now
`display(Markdown(...))`, `<- X` still visible), `11_section5_minutes.jpg`
(new -- section 5's minutes, now also `display(Markdown(...))`),
`12_hook_demo_minutes.jpg` (FP16 warning gone).

### `run_tests.py` and frozen-file check

```
$ python prototype/run_tests.py
PASS  a. imports                  11 modules import  (0.7s)
PASS  b. weights                  5 members, labels discussion / decision / action_item  (0.1s)
PASS  c. shipped KPIs reproduce   dev 0.96/0.948, test 0.84/0.825, 36-turn test 0.86/0.853, sample 7/9, probes 9/12  (3.3s)
PASS  d. demo minutes             no decisions, one action item (catering by Friday, Priya)  (0.2s)
PASS  e. --text example           runs, model matches the hand labels 7/7  (0.4s)
PASS  f. hands-on                 solution: both parts OK, 35/45 at knob 0; unfilled version says 'not written yet'  (1.0s)
PASS  g. training data            leak checks pass  (2.4s)
PASS  h. scorecard                frozen scorecard runs (hash ignores line endings)  (1.1s)
PASS  i. smoke runs               5 scripts exit 0  (2.5s)
PASS  j. text clean-up            text clean-up: 31/31 clean-up checks pass  (0.5s)
PASS  k. holdout meetings         holdout 1 0.879/0.806, holdout 2 0.852/0.762 (accuracy / macro-F1)  (3.2s)
SKIP  l. audio (optional)         no --audio FILE given  (0.0s)
PASS  m. repo unchanged           git status and git diff identical before and after  (0.1s)

12 passed, 0 failed, 1 skipped in 15.4 s
```

```
$ git diff --stat main -- prototype/trained_lstm.npz training_data.py scorecard_data.py \
    holdout_data.py minutes_tests.py normalise_tests.py lstm_model.py train_lstm.py \
    meeting_minutes.py transcriber.py hands_on_reference.py testing/ prototype/hands_on.py
(no output -- nothing frozen was touched)
```

### Files changed this round

```
 demo/hook_demo.ipynb           | setup cell: FP16 warning filter
 notebook/audio_workshop.ipynb  | retention-graph legend/title; cells 48, 51: display(Markdown(...)); setup cell: FP16 filter
 plan/handoffs/HANDOFF-WP5.md   | caption and header fixes (finding 4); this Round 2 section
 plan/handoffs/wp5/05_retention_graph.jpg -> .png  (replaced, re-taken)
 plan/handoffs/wp5/10_own_meeting.jpg             (replaced, re-taken)
 plan/handoffs/wp5/12_hook_demo_minutes.jpg       (replaced, re-taken)
 plan/handoffs/wp5/11_section5_minutes.jpg        (new)
```
