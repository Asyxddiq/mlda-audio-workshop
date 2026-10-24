# HANDOFF-WP8: live-mic cells for blocks 6-9 (say it, see it)

Branch `claude/build-workshop-gaps-756dec`, built on top of WP7 round 2 (toy_memory, hardened live_mic). Nothing committed. Scope: `notebook/` and `demo/` only; `prototype/` untouched.

## What's built

20 new cells across 4 sections, all reusing `live_mic.say`/`word_vector` (no new recorder code):

- **6b** (after cell 60): live 2-turn exchange -> toy LSTM (`SimpleLSTM`, untrained, same class the trained model uses) gate plot, eraser/pencil marked at `<eot>`. Then reuses 4b's `CATS_LIVE`/`DOGS_LIVE` (already W2-fixed) through the same toy LSTM at knob -6/0/+6.
- **7b** (after cell 65): request / proposal / plain-fact, one live_mic prompt each -> model label + confidence + loss, vs. keyword rule.
- **Sure. live** (after cell 68): "Can you book the room?.../Sure." and "Shall we move it to Friday?.../Sure." (2 prompts, 2s pause) -> label on the spoken "Sure." each time.
- **8b** (after cell 74): live 2-turn exchange -> trained model's label (member 0, same single-member setup as the existing knob sweep) at knob -6/0/+6, label changing visibly.
- **9b** (after cell 82): spoken 4-6 line meeting -> transcript (Whisper + `transcriber.segments_to_transcript`, pause-split, no second Whisper model loaded) -> labels -> rendered minutes. Typed cell (79) stays as the fallback.

8 new live prompts (block6 adds 1 new recording + reuses 2 from 4b; block7: 3; sure: 2; block8: 1; block9: 1) -- 18 total across WP7+WP8, well past the spec's minimums.

RUNSHEET.md and FACILITATOR.md cell numbers recomputed against the actual new notebook (105 cells); the stale "withhold the participant zip" line (fixed in both files -- `workshop_files.zip` has no demo answers since the zip split, safe to share early).

## L4 proof (TTS-injected, 2 voices)

- **Block6 gate plot:** eraser-is-global-minimum-at-`<eot>` held in 2 of 4 varied exchanges -- claim worded as a caveat ("doesn't reliably hit its lowest value... on every random init"), not an absolute.
- **Block6 knob sweep (reuses 4b's W2-fixed pair):** knob -6 collapses to ~0.0000 by word 8; knob 0 (default) ~0.02; knob +6 stays ~1.6 -- clean three-way separation, screenshot `wp8/02_block6_knob_sweep.png`.
- **Block7 catch-out:** 2 independent request/proposal/fact triplets. Both land the plain fact on "discussion" (arguably should be "decision", same weak spot as "Venue is confirmed..."), both get request/proposal sensibly.
- **Sure. live:** "Can you book the room?.../Sure." -> action_item 80% (correct); "Shall we move it to Friday?.../Sure." -> discussion 98% (wrong) -- matches the hook and the scripted Sure. test exactly.
- **Block8 knob sweep:** 2 independent exchanges, knob -6 flips the label vs. 0/+6 in both cases.
- **Block9:** ran end to end on a spoken 4-line meeting, real transcript -> labels -> rendered minutes, no crash.

## Full verification

`compile()` 105/105 cells clean. Both zips rebuilt (`workshop_files.zip` 5.0MB/37 files, `facilitator_files.zip` 5.9MB/40 files -- includes `toy_memory.py` now too). Unsolved + solved full notebook runs from `workshop_files.zip`, 0 errors each. `hook_demo.ipynb` cells 2/8/10/12 from `facilitator_files.zip`, 0 errors. `run_tests.py` 12 passed/0 failed/1 skipped. Frozen check empty. Screenshots in `plan/handoffs/wp8/`.

## Full Colab mic checklist (WP7 + WP8, notebook order)

1. Cell 26 (2b, clear sentence): countdown actually counts down; waveform/spectrogram show your voice.
2. Cell 28 (noisy): accuracy printed; if 100%, cell says so instead of claiming a drop.
3. Cell 30 (backward word): both audio players actually play; single-word OK (`minimum_words=1`).
4. Cell 35 (3b, any sentence): bar chart shows change per word, no flat bars.
5. Cells 43-44 (4b, Cats/Dogs): say them close to word-for-word but don't worry if they drift; percentage-decay plot, ON never 0%, OFF exactly 0% from word 2.
6. Cells 49-52 (5b, short + bonus long pair): two-bar chart, short noticeably taller than long.
7. Cell 62 (6b, 2-turn exchange, 2s pause): gate bar chart, `<eot>` visible in the word labels.
8. Cell 64 (6b knob sweep): reuses 4b's recording automatically, no new prompt.
9. Cells 72-74 (7b): request, proposal, plain fact -- one per cell.
10. Cells 80-81 (Sure. live, 2s pause before each "Sure."): two exchanges.
11. Cell 90 (8b, 2-turn exchange): label changes as you watch the knob move.
12. Cell 101 (9b, 4-6 line meeting, solo or paired, ~2s between lines): transcript, labels, rendered minutes.
13. **On every cell above:** deny mic permission once on purpose -- confirm the plain fallback message, not a raw error, and that the cell still produces a working plot/result from the shipped clip.
14. Confirm no cell needs a press-Enter or a button click to stop recording -- it should start and stop on its own.

## Not checked / open

- No Colab/real mic/T4 from this build session -- everything above is TTS-injected. Same caveat as WP7.
- Block6b's gate-plot caveat (eraser-at-`<eot>` only 2/4) is an honest limitation of the untrained toy LSTM's random init, not a bug to fix -- already worded as a caveat in the notebook.
- WP7/WP8 timing is additive; the orchestrator may want to revisit `plan/SPEC.md`'s block minutes (outside this package's scope).
