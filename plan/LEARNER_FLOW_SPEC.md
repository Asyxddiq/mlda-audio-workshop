# Learner-flow spec: WP9 (the day's flow and "see it first-hand") and WP10 ("make it yours")

Asked by Asy, 19 Oct 2026, to follow WP7-WP8:

> make sure the notebook is in a format that is suitable as a flow to learners on the day ... focus more on the
> live mic tests ... for RNN, they need to see first hand it works for sequential recordings, but for long and
> sustained convos, the memory is garbage. THEY NEED TO SEE THIS FIRSTHAND ... also provide a "cheatsheet" ...
> at the end, suggest which 20% we can remove ... so they can personalise the notebook ... 20% is just guidance.

## PM decisions

- **"Train it again" means another refinement pass of the notebook, not retraining the model.**
  - The model, training data and `prototype/` stay frozen.
  - Every earlier retrain on our own data failed its ship rule.
  - The workshop is in 3 days.
- **The scope is still `notebook/` and `demo/`.** Same agent loop and same L1-L7 rules as `plan/LIVE_MIC_SPEC.md`.

## WP9: the learner flow on the day

### F1: Live mic first in every block

Each block reads top to bottom as:

1. one short idea;
2. **Predict**;
3. **say it** (live mic);
4. **see it**;
5. **why**;
6. the code hands-on, where the block has one.

Theory text gets trimmed to what the slide doesn't already say. Long explanations move to collapsed
"More detail" cells.

### F2: RNN memory, short vs long, first-hand (the headline aha)

Participants must see, on their OWN recordings, that sequential memory works over a few words or turns and is
garbage over a long, sustained conversation. Two honest experiments, both live:

- **(a) Toy RNN, conversation length.**
  - The participant records a SHORT exchange (2-3 turns) and a LONG sustained conversation (8+ turns, about
    40-60 s). In both, the FIRST line holds a key fact (e.g. "The meeting is on Friday at 3").
  - Swap that first line for a different fact, keeping the same words otherwise.
  - Show, as one bar each, "how much of line 1 is still in the memory at the last word": short gives a clear
    percentage; long gives about 0%.
  - Then the same with the toy LSTM at a "keep" dial (knob +3) for contrast. It must be labelled honestly as a
    setting, not something our trained model learned.
- **(b) Trained model, memory reach** (real, measured; see the README "memory reach" result).
  - On the participant's long recorded meeting, change the FIRST line drastically. The last line's label
    probabilities barely move.
  - Then change the line JUST BEFORE the last. They move a lot.
  - The message: even our trained model really only uses about the previous turn.
  - Use `lstm_model.label_probabilities` and show it as a two-bar "how much did the last answer change?" chart.
  - Never claim the trained LSTM beats an RNN here.

Both experiments need L4 proof on varied recordings: different facts, different lengths, filler words.

### F3: Learner cheat sheet

`notebook/CHEATSHEET.md`, plus the same content as the notebook's first cell, short enough to print on one page
(A4, two columns):

- how to open the notebook, upload the zip, set the T4 and run setup;
- how to run a cell and what the live-mic cell does (countdown, Allow the mic);
- the prompt order per block, with "what you should see";
- what to do when it goes wrong (no mic, a red error, a restart);
- the two YOUR TURN lines' hints, but not the answers;
- what to take home.

Add it to `workshop_files.zip`.

### F4: Timing

Every block's live prompts plus the hands-on fit its minutes. Extras are marked "(bonus)". The RUNSHEET is
updated.

## WP10: "Make it yours" (personalisation)

**Step 1: proposal.** The builder proposes, then the reviewer assesses, then the PM decides.

The builder lists every candidate piece participants could build themselves at the end, after the live-mic tests.
For each candidate it gives:
- the minutes needed;
- the difficulty for a no-ML participant;
- which concept it reinforces;
- the risk of a broken notebook.

Example candidates:
- choosing their own prompts and sentences for each experiment;
- writing the labels for their own meeting;
- writing the minutes-formatting line;
- setting the dial to make the long conversation remembered;
- writing the "Why" in their own words;
- turning a built-in experiment into one they design.

The reviewer then checks that each candidate fits the workshop's minutes for a no-ML audience.

**Step 2: PM decision.** I decide the percentage (20% is guidance only; less if the time or the audience won't
carry it) and which pieces.

**Step 3: build and review.** Each removed piece becomes a YOUR TURN cell with:
- a one-line prompt;
- a check that confirms it works;
- a "Stuck?" hint;
- a working fallback, so the rest of the notebook still runs if they skip it.

The notebook ends with a "What I built today" summary cell that lists the participant's own pieces.
