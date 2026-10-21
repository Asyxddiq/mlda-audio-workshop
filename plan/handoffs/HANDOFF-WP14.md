# HANDOFF-WP14: audio-themed demo UI

Nothing committed. Files: `demo/hook_ui.py` (my recorder: Record/Stop, timer, canvas waveform+bars, silence meter, eval_js), `demo/hook_theme.py` (qna's banner/progress/cards/audio panel), `demo/hook_demo_cli.py` (standalone non-Colab fallback), `demo/hook_demo.ipynb` cells 0,2,4,6,8,10,12 rewired to both, `notebook/make_bundle.py` adds all three to `facilitator_files.zip`.

Also bumped `hook_ui.py`'s recorder font sizes (title 26px, buttons 18px, timer 28px, status/silence label 16px) per the reviewer's readability flags -- didn't touch `hook_theme.py`.

Checks, re-run after the sizing fix: (1) recorder visualizer screenshot, oscillator-fed, mid-recording, big readable text: `plan/handoffs/wp14/recorder_preview_bars.jpg`. (2) cells 2, 8, 10, 12 from a fresh `facilitator_files.zip` extraction: **0 errors**. `compile()` on all 13 cells: OK. `run_tests.py` skipped per your instruction (prototype/ untouched, confirmed via `git diff --stat main -- prototype/`: empty except the additive `demo.py` function).

**Asy must try with the real mic in Colab:** cell 4's Record/Stop buttons, mic permission prompt, live waveform+bars while talking, the silence meter filling and flashing "next speaker" on a pause, then cell 6's progress strip and minutes cards on your own recording.

SendMessage sent.
