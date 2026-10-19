# Workshop cheat sheet

## Setup

1. colab.research.google.com -> File > Upload notebook -> `audio_workshop.ipynb`.
2. Runtime > Change runtime type > T4 GPU > Save.
3. Upload `workshop_files.zip` when asked. Run cell 2. Wait for `ready`.

## Live-mic cells

Run the cell. Watch the countdown. Speak when it says so. Allow the mic once, first time.
No mic? It falls back to a recorded example automatically.

## Each block

| Block | Say | You should see |
|---|---|---|
| 2 | a clear sentence; then with noise; then one word | waveform, accuracy %, forwards/backwards |
| 3 | any sentence, 5-15 words | a bar for every word, changing |
| 4 | "Cats are..." / "Dogs are..." | memory ON fades, OFF drops to 0 |
| 5 | "Red/Blue cars drive fast"; a short exchange, one short reply each | short keeps more than long (the long one plays back a recording) |
| 6 | two short turns, 2s pause | gate bars, `<eot>` marked |
| 7 | a request; a proposal (not yet agreed); a plain fact that's really a decision | label + confidence for each |
| Sure. | "Can you book the room? / Sure."; "Shall we move it to Friday? / Sure." | same word, different labels |
| 8 | a two-turn exchange | the label changes as the knob moves |
| 9 | your own 4-6 line meeting | transcript, labels, minutes |
| 10 | your own words; nothing to say | your explanation; what you built |

## YOUR TURN hints (not the answers)

- Part A: use `@` for matrix multiply, `tanh` to squash the result.
- Part B: combine the old memory, what's forgotten, and what's new.
- Your pair (block 4): keep both sentences the same except the first word.
- The dial (block 8): bigger `KNOB` keeps more.
- Your labels (block 9): `act`, `dec`, or `dis`, one per turn.

## If something breaks

1. Red error? Run cell 2, then every cell above, in order.
2. No mic, or you said no? Nothing to do -- it already used the example.
3. "Heard almost nothing"? Try again, closer to the mic.
4. A `YOUR TURN` cell says "not yet"? See its "Stuck?" note.
5. Any cell marked `(bonus)` -- skip it if you're short on time.
