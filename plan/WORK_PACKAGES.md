# Work packages

Builder branch: `build/workshop-gaps`. Reviewer branch: `review/workshop-gaps`.
One package at a time: the builder stops after each one and waits for the review.

## WP1: The demo (D1-D6)

Deliver:
- A mock CCA meeting script, two speakers, about 1 minute spoken, with a written **answer key** (the decisions and to-dos a human hears). Files under `demo/`.
- A one-press runner, `prototype/demo.py` (and one notebook cell that calls it): takes the script text, or a WAV, and prints the minutes large and readable. One command, no flags needed for the default case.
- A backup WAV of the script in two synthetic voices (Windows text-to-speech is fine) for when nobody can act it out. Document how it was made.
- A short facilitator page, `demo/FACILITATOR.md`: what to say, when to press, what to do if the minutes differ from the room's list.
- A report on the two "Sure." exchanges as the model actually labels them (see rule H).

Done when:
- The default run lists every item in the answer key and no wrong items, from text.
- The WAV run's result is reported honestly: turns found, word accuracy, whether the minutes match the key.
- `python prototype/run_tests.py` reports 0 failed.

## WP2: Notebook, sound and the RNN (N1-N5, N10)

Deliver in `notebook/audio_workshop.ipynb`:
- Waveform, spectrogram, a reversed word (N1), and the four-scenario noisy set with answer key and measured word accuracy (N2).
- The RNN section rewritten around words (N3, N10).
- Hands-on A with the check, the old-memory-off comparison (N4).
- A memory-retention graph for r = 0.49 and r = 0.95 over 47 steps, labelled as an illustration (N5).
- Section 1-2 marked facilitator-run, with minutes per section.

## WP3: Notebook, LSTM and training (N6-N8)

Deliver:
- LSTM section: gates, cell-state line, hourglass picture, `<eot>` (N6).
- A training-in-one-picture cell with a worked `-log p` example, and a keyword-vs-LSTM comparison on unseen lines including "Venue is confirmed for the 14th" (N7).
- Hands-on B with the check and the knob, showing the toy result AND the trained-model score at -6, 0, +6 (N8, tension 2).

## WP4: Your own meeting, integration (N9)

Deliver:
- The own-meeting cell: five lines in, participant labels optional-but-prompted, model labels out, mismatches flagged, unknown words listed, a prompt asking which earlier line it should have remembered (N9).
- `python notebook/make_bundle.py` rebuilt; every cell run from an empty folder holding only the zip; a table of minutes per section against the timetable.
- One log entry in `progress.md`; new failures hit while building added to `debug_cheatsheet.md`.

## Review loop (per package)

1. Builder commits, writes `HANDOFF-WPn.md` (commit hash, requirement IDs covered, commands run and output, anything not done), then ends its turn.
2. The main session tells the reviewer the hash.
3. Reviewer writes `reviews/WPn.md` and ends its turn.
4. The main session tells the builder. Builder fixes in new commits and updates the handoff. Repeat until the verdict is PASS.
5. The user merges to `main`. Nobody pushes.
