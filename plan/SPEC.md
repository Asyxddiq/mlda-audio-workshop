# Spec: what the reference docs ask for (demo + notebook only)

Source: `reference workshop docs/WS 4 Agenda Proposal.pdf` ("Agenda") and
`reference workshop docs/Workshop_4_Programme_and_Slide_Plan (1).pdf` ("Programme").
Both PDFs win over anything the repo currently does. Quote the ID (e.g. `D3`) in every handoff and review.

**Out of scope for every session:** slides, Kahoot itself, the take-home retrain, the printable worksheet,
retraining or changing the model, README rewrites.

## The demo (Agenda block 1, 00:00-00:08)

| ID | Requirement | Source |
|---|---|---|
| D1 | Input is a ~1-minute mock CCA meeting: a recording, or two people acting it out | Agenda 1; Programme block 1 / slide 02 |
| D2 | The room calls out the decisions and to-dos first, before the app is shown | Agenda 1 |
| D3 | Then ONE button press shows the app producing the same list as minutes | Agenda 1 ("press one button") |
| D4 | The generated minutes are compared with the audience's recall | Programme slide 03 |
| D5 | The "Can you book the room?" / "Shall we move it to Friday?" -> "Sure." pair is read aloud to the room, who say what "Sure." means each time. It is a human exercise, not what the app is judged on | Agenda 1 |
| D6 | Preparation check: the demo works "as described" (D1-D4) | Programme p.2 |

## The notebook (Agenda blocks 2-9)

| ID | Requirement | Source |
|---|---|---|
| N1 | Show a waveform and a spectrogram (time left to right, pitch bottom to top); play a word backwards so order is audible; Whisper reads the picture and writes words | Agenda 2; Programme 2 |
| N2 | Noisy-audio comparison: 3 noisy scenarios + 1 ideal, same sentence, so the facilitator can ask "which gives the most accurate transcript?" (Kahoot itself is out of scope; the four clips + answer key are in scope) | Agenda 2 |
| N3 | RNN = a running summary: `h = tanh(W_x . x + W_h . h_prev + b)`, reading words one at a time | Agenda 3; Programme 3 |
| N4 | Hands-on A: type that one line, a check confirms it, then delete the old-memory term and compare the outputs | Agenda 4 |
| N5 | Memory fades: retained fraction after n steps is r^n; 0.49^47 is about 0, 0.95^47 is about 9%. A graph. It is an illustration, NOT a measured property of the trained system | Agenda 5; Programme 5 + equations note |
| N6 | LSTM: cell state with forget (eraser), input (pencil), output (voice) gates; `c = f * c_prev + i * g`; five layers as an hourglass; `<eot>` as the "speaker changed" bookmark | Agenda 6 |
| N7 | Training in one picture: label -> prediction -> `loss = -log(p(correct class))` -> update. Test on unseen lines against a keyword checklist. Show a failure: "Venue is confirmed for the 14th" is missed | Agenda 7; Programme 7 |
| N8 | Hands-on B: type the notebook update, a check confirms it, then turn the eraser knob (erase a lot / keep a lot) and observe how an early line affects later memory | Agenda 8 |
| N9 | Own meeting: participants write 5 lines, label each (decision / to-do / discussion), run the model, find one wrong prediction, and say which earlier line it should have remembered. Agenda notes this depends on whether the prototype handles other topics | Agenda 9; Programme 9 |
| N10 | Keep Whisper transcription separate from the downstream RNN/LSTM model | Programme p.2 |

## Known tensions between the PDFs and the repo (builder resolves; reviewer checks the resolution)

1. **N10 vs the current notebook section 3**, which feeds audio MFCC frames to an RNN. The PDFs have the RNN reading words. Resolve toward the PDFs; the MFCC cell can move to N1 as "what Whisper sees".
2. **N8 "keep a lot" wording.** On the toy two-sentence test, a high forget bias keeps the first word. On the trained model, both extremes lower the test score (35/45 at 0). Show both facts; never say "keep a lot = better".
3. **N7/N9.** The labelling task needs only about one previous turn (the RNN-vs-LSTM comparison in the README found no clear difference). Do not claim the trained LSTM remembers further back than an RNN. Teach the long-memory idea with the toy experiments (N4, N8), not with the classifier.
4. **N9 vocabulary.** The model knows 553 words; unknown words become `<unk>`. The cell must say which of the participant's words were unseen.
5. **Notebook sections 1-2 (Whisper install + noise tests)** have no hands-on slot in the timetable (block 2 is a 10-minute talk). Mark them facilitator-run or optional, and state the minutes each section needs.
6. **Hook script vs model.** The room calls out decisions; the model is weakest at decisions. See `RULES.md` rule H.

## Timetable the notebook must fit

| Block | Minutes | Notebook content |
|---|---|---|
| 2 Turning sound into words | 10 | N1, N2 (facilitator-run) |
| 3 RNN | 15 | N3 (talk; notebook shows the step) |
| 4 Hands-on A | 10 | N4 |
| 5 Memory fades | 7 | N5 |
| 6 LSTM | 20 | N6 |
| 7 Teaching + catching it out | 10 | N7 |
| 8 Hands-on B | 20 | N8 |
| 9 Your own meeting | 15 | N9 |
