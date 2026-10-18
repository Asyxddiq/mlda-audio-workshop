VERDICT: PASS

Review of WP5, "Rendered output check", at `e89780b` (parent `2b2dd49`), in a detached worktree. I opened all 10 screenshots myself, then re-rendered the participants' notebook (solved) from `workshop_files.zip` alone in an empty folder and checked what it produced, including the audio data inside the players, which a screenshot cannot show.

Result: the handoff's claims hold. The commit adds only `plan/handoffs/` files, no code changed, and nothing the builder says is false. I found four cosmetic points the handoff did not mention (below). None of them is a defect in the notebook's content, so they are nits, and the verdict is PASS.

## The commit

`git diff --stat 2b2dd49 e89780b`: 11 files, all under `plan/handoffs/` (`HANDOFF-WP5.md` plus the 10 JPGs in `plan/handoffs/wp5/`). `bash reviews/check_frozen.sh e89780b` reports no frozen file changed; its only complaint is that `plan/handoffs/*` is outside the builder's write list, the approved location. The handoff's own text ("Nothing committed this round") is stale, since the files are in the commit; harmless.

## The ten screenshots, each against its claim

| # | Claim | What the image actually shows | Verdict |
|---|---|---|---|
| 01 | Waveform with labelled axes; spectrogram with low pitch at the bottom, time left to right, labelled, no warning text | Waveform titled "Waveform", y axis "amplitude", three groups of speech bursts (about 0-4.4 s, 7.3-10 s, 12.9-15 s, matching the three sentences). Spectrogram titled "Spectrogram -- time left to right, pitch bottom to top", y axis "pitch (Hz)" running 0 at the bottom to 8000 at the top, x axis "time (s)" 0-16. No warning text in the output. The source shown includes the `np.errstate` guard. | **Confirmed** |
| 02 | Both reversed-word players present, right length | "forwards:" and "backwards:" players are both there. The screenshot shows `0:00 / 0:00` on both, which does not by itself show the clips are non-empty. | **Confirmed by my own check**: both embedded WAVs decode to 0.5 s, 16 kHz, rms 0.2303, with different bytes (md5 `89fb0e65` and `1e9d6d0e`). The 0:00 label is the browser rounding a half-second clip. |
| 03 | All four Kahoot players present | The image shows the vote text and **only the first player** ("1. ideal", 0:15); the other three are below the crop. The handoff says the picture shows all four. | **Claim is true, screenshot is cropped**: my render has exactly four players in that cell, each 15.64 s, with different content (rms 0.103, 0.1438, 0.103, 0.1129; four different md5s). |
| 05 | Retention graph: 0.49 and 0.95 curves, n=47 values, titled as an illustration | Title "Memory retention: r^n (illustration, not a measured property of the trained model)"; legend "r = 0.49 (the model's starting (untrained) settings): 0.0% left after 47 steps" and "r = 0.95 (keeping 95% each step): 9.0% left after 47 steps"; both curves present and correctly ordered; printed line `0.49^47 = 0.0000%   0.95^47 = 9.0%`. | **Confirmed** (see nit 1) |
| 06 | Hourglass chart 8/16/32/16/8 | Bars at 8, 16, 32, 16, 8 for layers 1 to 5, y axis "units", and the printed table above it reads 8/16/32/16/8 units and 88/44/22/44/88 steps. | **Confirmed** |
| 07 | Loss example and held-out text readable | Both lines print cleanly: `p(correct) = 0.0037, loss = 5.61` and `p(correct) = 0.9957, loss = 0.00`; the best-case caveat paragraph is intact. | **Confirmed** |
| 08 | Keyword vs LSTM | `keyword rules: 12/18 turns right`, `trained LSTM: 17/18 turns right`, and the one miss line ("No refreshments this time...", gold decision, got discussion). | **Confirmed** |
| 09 | Knob sweep aligned | Three blocks for -6, +0, +6, each with `check: OK`, the toy gap row aligned under the word numbers, and the trained scores 10/45 (0.22), 35/45 (0.78); the +6 block continues below the crop. The "keep a lot is not better" text is above. | **Confirmed** (the +6 block is cropped; I checked it in my own render, 22/45) |
| 10 | The wrong-line mark is visible in the own-meeting output | `decision (you) discussion (model) Lee: The venue is confirmed for Friday.  <- X` is plainly visible (the mark is the text `<- X`, not a cross symbol). The code above it shows the new try/except and the live `vocab_size`. | **Confirmed** (see nit 2) |
| 12 | Hook minutes show Mei and Ravi as owners | The rendered Markdown shows a real heading "Meeting Minutes", "Attendees: Mei, Ravi", Decisions "Hold the showcase on the 21st (agreed)" and "Get black t -shirts (agreed)", Action Items "Book the hall by Friday -- Ravi (due Friday)" and "I'll send the order form to everyone tonight. -- Mei (due tonight)". | **Confirmed** (see nit 3) |

Not screenshotted by the builder, so I checked it myself: item 8, the unsolved run. `PART A` and `PART B` print "not written yet" and no cell after them crashes; this was true in my WP4 round-2 run of the same notebook (0 errors, unsolved), and my HTML search below found no traceback in the solved render.

## My independent re-render

Fresh `python notebook/make_bundle.py` (17-file participants' zip and 20-file facilitator zip), then an empty folder holding only `workshop_files.zip` and the notebook, with Part A and Part B solved inline, executed with `nbclient`, saved, and rendered to HTML with `nbconvert`:

```
cells executed: 53    errors: []    TOTAL 43.5 s
images: 3   (cell 6 waveform + spectrogram, cell 31 retention graph, cell 33 hourglass chart)
audio players: 6   (cell 8: 2 x 0.5 s;  cell 22: 4 x 15.64 s;  all 16 kHz, all different content)
HTML: 6 <audio> elements, 3 <img> elements, "Traceback" appears 0 times
```

I also opened the retention figure at full size from my own run (`fig_31.png`) and it matches screenshot 05.

## Spot-checks of the claims (item 9 and two more)

1. **Item 9, markdown numbers vs outputs.** Markdown answer-key table, `base.en` row: 100 / 61 / 52 / 35. My run's live cell printed `ideal 100%`, `background hiss 52%`, `poor microphone 35%`, `a second voice underneath 61%`: exact match. The held-out paragraph quotes 131/149, 104/122, 111/149 and 93/122 and "17/18" against my WP3 measurements (identical). The loss paragraph's numbers equal the printed 0.0037/5.61 and 0.9957/0.00. Part A text says "around 58%" and the output says "about 58% kept per step". The N9 cell's vocabulary line is computed live and prints 553. No mismatch found.
2. **Items 3 and 4, audio players.** Decoded the base64 WAV inside every `<audio>` element rather than trusting a DOM flag: lengths and content as in the table above. All six are real, non-empty and distinct.
3. **Item 7, no stray tracebacks or temp paths.** HTML text search for `Traceback`, `tmp` and the temp-file path: 0 hits. (`ValueError` and a comment containing "temp" appear only inside displayed code.)

## Nits (none blocks the merge)

| # | Severity | Where | What | Fix |
|---|---|---|---|---|
| 1 | nit | retention graph (cell 31), screenshot 05 | The legend sits in the top-left and covers the first few points of both curves (they start at 100% under the legend box), and the title is wider than the plot area. Both are visible in the screenshot and in my own render. | `plt.legend(loc="center right")` and a smaller title font or a line break. |
| 2 | nit | participants' minutes output (cells 48 and 51) | The printed minutes show raw Markdown (`# Meeting Minutes`, `**Attendees:** Lee, Sam`, `## Decisions`, `- [ ] ...`) because the cell uses `print`, while the facilitator notebook uses `display(Markdown(...))` and shows a proper heading and bullets. Readable, but less polished for participants. | `from IPython.display import Markdown, display` and `display(Markdown(generate_minutes(...)))` in those two cells. Optional. |
| 3 | nit | `hook_demo` rendering (screenshot 12) and sections 1-2 of the participants' notebook | A red box with `UserWarning: FP16 is not supported on CPU; using FP32 instead` appears above the minutes. It only happens on a CPU run (a Colab T4 supports fp16), but the handoff did not mention it, and a laptop fallback in front of the room would show it. My render has the same warning in cells 4, 12 and 14. | In the setup cell of each notebook add `import warnings; warnings.filterwarnings("ignore", message="FP16")`. Optional. |
| 4 | nit | `HANDOFF-WP5.md` | Screenshot 03's caption says it shows all four players, but the image is cropped after the first (the claim is true, see above); the "Result" line also says "merged to local `main`" and "nothing committed" while the files are in the commit. | Fix the caption when the handoff is next touched. |

## Not verified

Colab, a T4, and any listening: I confirmed the audio players are non-empty, the right length and different from each other, but I cannot judge how they sound. The facilitator notebook's microphone cells were not run.

This was the last package. With WP1-WP4 passed and this render check, the branch tip is `e89780b` (a documentation-only commit on top of the WP4 merge candidate `2b2dd49`).
