# mlda-audio-workshop

Backend prototype for the MLDA Academics workshop **Deep Learning in Audio Applications**.

The pipeline is audio → transcript → LSTM → meeting minutes. It is written as small, readable NumPy code so that beginner-to-intermediate participants can follow every step and run it on free Google Colab.

| Component | Owner | Status (16 Oct 2026) |
|---|---|---|
| Real audio → transcript (Whisper, pause-based speakers) | Pasan, Rahini | Passes on synthetic speech and LibriSpeech clips; a team member's recording is still to test (see [testing](testing/testing.md)). A reply under 0.8 s can still be merged; more than two speakers are not told apart |
| RNN + CTC teaching demo | Pasan, Rahini | Working |
| 5-layer hourglass LSTM | Asy | Working |
| Training data: 40 hand-labelled meetings (577 turns) + 60 generated practice meetings, plus a frozen 6-meeting scorecard (105 turns) | Asy (drafted by Claude); Nomt checked his half (4 meetings) 8 Oct | Draft; the 29 meetings added 8 and 13 Oct are not yet checked by a person; Asy's blind check to do |
| Trained turn classifier (`train_lstm.py`, NumPy) | Asy | Working; ships the **tuned 13 Oct ensemble** (5 seeds) behind the 16 Oct text clean-up: test 0.84, probes 9/12, scorecard 0.90 (see [results](#step-4-training-the-classifier)). Two 16 Oct retrains failed their ship rule ([model update](testing/model_update_16oct/model_update.md)) |
| Meeting minutes (trained labels, keyword fallback) | Nomt | Working |
| Participants' hands-on (`hands_on.py`) | Asy | Ready 13 Oct: Part A writes the RNN step and switches its memory off; Part B writes the LSTM notebook update and turns the forget knob. `--solution` runs the answers |
| RNN vs LSTM comparison (`compare_rnn.py`) | Asy, for Pasan and Rahini | Done 13 Oct, with both the old and the tuned settings: **no clear difference** on our data by the rule written beforehand; both use about one turn of context |
| Whole chain in one command (`pipeline.py`) | All | Working on the sample meeting |

**Contents:** [Quick start](#quick-start) · [High-level alignment](#high-level-alignment) · [Architecture](#architecture) · [Key commands](#key-commands) · [Limitations](#simplifications-and-known-limitations) · [References](#references)

---

## Quick start

```bash
git clone https://github.com/Asyxddiq/mlda-audio-workshop.git
cd mlda-audio-workshop
pip install -r requirements.txt
python prototype/lstm_model.py
```

Expected output (abridged):

```text
Vocab size: 66 words (incl. <eot>)
Tokens in conversation: 88 (padded to 88 so it divides by 4)
Fold -> unfold round trip is lossless: True

Hourglass shape (compare with the paper's Table 1):
  layer      units   time steps
  layer 1        8           88
  layer 2       16           44
  layer 3       32           22
  layer 4       16           44
  layer 5        8           88

[Alex] Let's kick off. First item is the Q3 budget review.
  -> output after this turn (first 4 of 8 dims): [0.008 0.016 0.017 0.006]
...
```

The only dependency is `numpy`. No GPU is needed; the full run takes well under a second.

To run the whole chain (transcript → LSTM → minutes) in one go:

```bash
python prototype/pipeline.py                # sample meeting, no audio needed
python prototype/pipeline.py meeting.wav    # real audio: pip install openai-whisper, plus ffmpeg
python prototype/pipeline.py --meeting hackathon_1 --compare   # a hand-labelled meeting, hand labels alongside
python prototype/pipeline.py --compare --text prototype/my_meeting_example.txt   # your own lines from a text file
python prototype/hands_on.py                # the participants' hands-on (Parts A and B)
```

The trained weights (`prototype/trained_lstm.npz`) are in the repo, so the pipeline labels turns with the trained model straight away. To retrain them from [`training_data.py`](prototype/training_data.py) (a few minutes on a laptop CPU, numpy only):

```bash
python prototype/train_lstm.py
```

---

## High-level alignment

### Where this repo sits in the learning arc

The workshop teaches **Audio → Python → RNN → LSTM → Practical applications**. Each stage of the arc has an owner and a piece of this repo.

```mermaid
flowchart LR
    A["Audio"] --> B["Python"] --> C["RNN"] --> D["LSTM"] --> E["Practical applications"]
    C -.-> T1["Transcriber track<br/>Pasan + Rahini"]
    D -.-> T2["LSTM track<br/>Asy + Nomt"]
    E -.-> T3["Meeting minutes<br/>Nomt"]
    T1 --> P1["prototype/transcriber.py<br/>(Whisper; placeholder if no audio)"]
    T2 --> P2["prototype/lstm_model.py"]
    T3 --> P3["prototype/meeting_minutes.py"]
```

### Team tracks

| Track | People | Builds | Hands off |
|---|---|---|---|
| 1. Transcriber (RNN) | Pasan, Rahini | audio file → speaker-labelled transcript | `get_transcript(audio_path) -> str` |
| 2. LSTM | Asy, Nomt | transcript → hourglass LSTM → 8 numbers per word | `(tokens, 8)` array + turn boundaries |
| 3. Meeting minutes | Nomt | LSTM output → summary and/or action items | minutes text |

The tracks meet at two **contracts**: the transcript string, and the LSTM output array. As long as each side keeps its contract, the tracks can be built in parallel and swapped in without touching each other's code.

```mermaid
flowchart LR
    subgraph T1["Track 1: Pasan + Rahini"]
        R["Real RNN transcriber"]
        PH["Placeholder transcript<br/>(used until R is ready)"]
    end
    subgraph T2["Track 2: Asy"]
        L["Hourglass LSTM"]
    end
    subgraph T3["Track 3: Nomt"]
        M["Meeting minutes"]
    end
    R -. "later: same contract" .-> C1
    PH --> C1(["contract 1:<br/>get_transcript() -> str"])
    C1 --> L
    L --> C2(["contract 2:<br/>output (tokens x 8)<br/>+ turn_end_indices"])
    C2 --> M
```

### How the workshop session maps onto the code

```mermaid
flowchart TD
    P1["Part 1: Intro to audio + Python"] --> P2["Part 2: Challenges in audio + Kahoot"]
    P2 --> P3["Part 3: Case studies"]
    P3 --> P4["Part 4: RNN concepts"]
    P4 --> P5["Part 5: First hands-on demo<br/>80% given, 20% participants complete"]
    P5 --> P6["Part 6: LSTM + meeting minutes"]
    P6 --> P7["Part 7: Extend the MVP<br/>two-part backend"]
    P7 --> P8["Part 8: Try and test"]
    P8 --> P9["Part 9: Debug time"]
    P9 --> P10["Part 10: Wrap-up"]

    P5 -.-> C1["Transcriber code<br/>(Track 1)"]
    P6 -.-> C2["lstm_model.py<br/>SimpleLSTM, gates"]
    P7 -.-> C3["lstm_model.py -> meeting_minutes.py<br/>Part A given, Part B built"]
```

Which 20% participants complete in Part 5, and how the Part 7 backend splits into Part A and Part B, is decided **after** the MVP runs end to end (see [progress.md](progress.md)).

### Build plan

Three build weekends (26 Sep, 3 Oct, 10 Oct), then slides from 12 Oct. The dated timeline and what is done are in [progress.md](progress.md#checklist).

### Definition of done before slide planning starts

- [ ] Audio → transcript → meeting minutes runs end to end without manual steps
- [ ] The 80/20 hands-on split is testable by someone who isn't Pasan or Rahini
- [ ] The Part A/B extension is testable by someone who isn't Asy or Nomt
- [ ] Known failure modes are written down for debug time
- [ ] All four team members have run the full pipeline themselves

---

## Architecture

### Repository layout

```text
mlda-audio-workshop/
├── README.md                          you are here
├── progress.md                        Asy's LSTM-track progress tracker: log, decisions, next up
├── labelling_guide.md                 rules for labelling training turns
├── requirements.txt                   numpy
├── prototype/
│   ├── pipeline.py                    whole chain in one command (start here)
│   ├── transcriber.py                 audio → transcript with Whisper (Track 1)
│   ├── transcript_placeholder.py      sample meeting, used when no audio is given
│   ├── rnn_ctc_demo.py                RNN + CTC teaching demo (Part 5)
│   ├── lstm_model.py                  text clean-up, tokenize, embed, 5-layer hourglass LSTM (Track 2)
│   ├── meeting_minutes.py             label turns, write Markdown minutes (Track 3)
│   ├── training_data.py               hand-labelled meetings, probe set, practice-meeting generator, leak check
│   ├── train_lstm.py                  trains the hourglass + classifier head (NumPy backprop)
│   ├── trained_lstm.npz               the trained weights (made by train_lstm.py)
│   ├── hands_on.py                    participants' hands-on: RNN step, LSTM notebook update, forget knob
│   ├── hands_on_reference.py          reference answers for hands_on.py (facilitators)
│   ├── my_meeting_example.txt         example input for pipeline.py --text
│   ├── scorecard_data.py              6 frozen held-out meetings (never trained on or tuned against)
│   ├── scorecard.py                   aggregate scores on the scorecard
│   ├── holdout_data.py                17 fresh held-out meetings written 16 Oct (never trained on)
│   ├── compare_rnn.py                 the same hourglass with plain RNN layers, vs the LSTM
│   ├── run_tests.py                   all repo checks in one command
│   ├── minutes_tests.py               67 typed meeting tests with expected minutes
│   ├── normalise_tests.py             checks for the text clean-up
│   ├── audio_check.py                 scores a recording of a script: transcriber and end-to-end verdicts
│   ├── audio_tests/                   older copy of the reading scripts (use testing/scripts/)
│   └── train_classifier.py            earlier PyTorch scaffold (Nomt), kept for reference
└── testing/                           the testing phase: start at testing/testing.md
    ├── testing.md                     how to run each test, and every result
    ├── phases.py                      each stage on its own: audio to text, RNN, LSTM, minutes
    ├── solo_test.py                   one-person recording test
    ├── scripts/                       scripts to read aloud
    ├── testing script.md              the 67 meeting tests, written from minutes_tests.py
    ├── model_update_16oct/            the 16 Oct model update: rule, write-up, scores, patch
    └── results/                       result files (claude/ and manual/)
```

### End-to-end data flow

One run of `python prototype/lstm_model.py`:

```mermaid
sequenceDiagram
    autonumber
    participant Main as lstm_model.py (main)
    participant TP as transcript_placeholder
    participant Tok as tokenize + embed
    participant HG as HourglassLSTM
    participant Out as output

    Main->>TP: get_transcript()
    TP-->>Main: 9 speaker-labelled lines
    Main->>Tok: tokenize_transcript(turns)
    Tok-->>Main: 88 tokens (with <eot>) + turn_end_indices
    Main->>Tok: build_vocab, embed_tokens
    Tok-->>Main: 88 x 16 numbers
    Main->>Main: pad_to_multiple_of_4 -> 88 x 16 (already a multiple of 4)
    Main->>HG: forward(x_padded)
    HG->>HG: layer 1 -> fold -> layer 2 -> fold -> layer 3
    HG->>HG: unfold + skip -> layer 4 -> unfold + skip -> layer 5
    HG-->>Main: 88 x 8 output (+ every layer's output)
    Main->>Out: drop padding -> 88 x 8
    Out-->>Main: one 8-number vector per word, read off at each turn end
```

### Module interfaces

```mermaid
classDiagram
    class transcript_placeholder {
        +SAMPLE_TRANSCRIPT: str
        +get_transcript(audio_path=None) str
    }
    class SimpleLSTM {
        +input_size: int
        +hidden_size: int
        +W_f, W_i, W_c, W_o
        +b_f, b_i, b_c, b_o
        +step(x_t, h_prev, c_prev) (h_t, c_t)
        +forward(x_seq) ndarray
    }
    class HourglassLSTM {
        +layer1..layer5: SimpleLSTM
        +forward(x_seq) (output, layers)
    }
    class helpers {
        +tokenize_transcript(turns) (tokens, turn_end_indices)
        +build_vocab(tokens) dict
        +embed_tokens(tokens, vocab, embeddings) ndarray
        +pad_to_multiple_of_4(x_seq) ndarray
        +fold_time(x) ndarray
        +unfold_time(x) ndarray
    }
    HourglassLSTM *-- "5" SimpleLSTM : built from
    HourglassLSTM ..> helpers : fold / unfold
    helpers ..> transcript_placeholder : reads turns from
```

### Step 1: words become numbers

A computer can only do maths on numbers, so each word goes through three steps. Split the text into words, give each unique word an id, then swap the id for a row of 16 numbers from an embedding table.

```mermaid
flowchart LR
    A["'Venue is confirmed ...'"] -->|"lowercase, split into words;<br/>? and ! become their own tokens"| B["venue · is · confirmed · ..."]
    B -->|"look up / assign id"| C["ids: 25 · 5 · 26 · ..."]
    C -->|"pick row id of the<br/>66 x 16 embedding table"| D["16 numbers per word"]
    E["end of each turn"] -->|"insert <eot>"| B
```

- **Vocabulary:** 66 unique tokens in the sample meeting, including the `<eot>` end-of-turn marker. The trained model uses a fixed 553-word vocabulary from its training data instead, and any word outside it becomes `<unk>`.
- **Embeddings:** random in this walk-through (seed 42). In the trained model they are learned along with everything else.
- **The whole conversation is one sequence.** The LSTM never resets between speakers; `<eot>` is how it can tell a new person is talking.

### Step 2: the hourglass LSTM

Adapted from Li et al. (2024) (see [References](#references)). Going down, each layer **halves the time steps and doubles the units**; coming back up, it reverses. Skip connections carry early detail across to the later layers.

```mermaid
flowchart TD
    IN["input<br/>88 steps x 16 numbers"] --> L1["Layer 1<br/>88 steps · 8 units"]
    L1 -->|"fold: glue pairs<br/>88x8 -> 44x16"| L2["Layer 2<br/>44 steps · 16 units"]
    L2 -->|"fold<br/>44x16 -> 22x32"| L3["Layer 3 (the neck)<br/>22 steps · 32 units"]
    L3 -->|"unfold<br/>22x32 -> 44x16"| A4(("+"))
    L2 -. "skip" .-> A4
    A4 --> L4["Layer 4<br/>44 steps · 16 units"]
    L4 -->|"unfold<br/>44x16 -> 88x8"| A5(("+"))
    L1 -. "skip" .-> A5
    A5 --> L5["Layer 5<br/>88 steps · 8 units"]
    L5 --> OUT["output<br/>drop padding -> 88 x 8"]
```

| Layer | Our units | Our time steps | Paper units | Paper time steps |
|---|---|---|---|---|
| 1 | 8 | 88 | 256 | 512 |
| 2 | 16 | 44 | 512 | 256 |
| 3 | 32 | 22 | 1024* | 128 |
| 4 | 16 | 44 | 512 | 256 |
| 5 | 8 | 88 | 256 | 512 |

\* The paper prints 1014. Every other layer is exactly double or half its neighbour, so 1024 is almost certainly what was meant.

**Why folding loses nothing:** 4 time steps of 2 numbers become 2 time steps of 4 numbers. The numbers are the same, just arranged in fewer, longer rows. `unfold_time()` reverses it exactly, and the script checks this before it runs.

```mermaid
flowchart LR
    subgraph before["4 steps x 2 numbers"]
        direction TB
        r1["a b"] --- r2["c d"] --- r3["e f"] --- r4["g h"]
    end
    subgraph after["2 steps x 4 numbers"]
        direction TB
        s1["a b c d"] --- s2["e f g h"]
    end
    before -->|"fold_time()"| after
    after -->|"unfold_time()"| before
```

**Why the hourglass helps memory:** in layer 3, one step covers 4 words, so information crosses the whole conversation in 22 steps instead of 88. Every step multiplies stored memory by the forget gate *f*, so fewer steps means more survives. For example, "catering still pending" (turn 4) is 47 words before the end of the meeting:

| Forget gate f | Left in layer 1 (f^47) | Left in layer 3 (f^11.75) |
|---|---|---|
| 0.49 (our untrained gates) | about 0% | 0.02% |
| 0.95 (a trained gate) | 9% | 55% |

### Step 3: inside one LSTM step

All five layers are built from the same `SimpleLSTM.step()`. At every word, four small networks (gates) read the previous output `h(t-1)` joined with the current word `x(t)`:

```mermaid
flowchart LR
    H["h(t-1)<br/>8 numbers"] --> J["join<br/>[h, x] = 24 numbers"]
    X["x(t)<br/>16 numbers"] --> J
    J --> F["forget gate f<br/>sigmoid, 0 to 1"]
    J --> I["input gate i<br/>sigmoid, 0 to 1"]
    J --> CT["candidate c~<br/>tanh, -1 to 1"]
    J --> O["output gate o<br/>sigmoid, 0 to 1"]
    CP["c(t-1)<br/>memory line"] --> MF(("x"))
    F --> MF
    I --> MI(("x"))
    CT --> MI
    MF --> ADD(("+"))
    MI --> ADD
    ADD --> C["c(t)<br/>new memory"]
    C --> TH["tanh"]
    TH --> MO(("x"))
    O --> MO
    MO --> HN["h(t)<br/>output"]
    C -. "next word" .-> CP
    HN -. "next word" .-> H
```

```text
f_t  = sigmoid(W_f · [h_prev, x_t] + b_f)     what to keep
i_t  = sigmoid(W_i · [h_prev, x_t] + b_i)     how much to write
c~_t = tanh(W_c · [h_prev, x_t] + b_c)        what could be written
c_t  = f_t * c_prev + i_t * c~_t              update the memory line
o_t  = sigmoid(W_o · [h_prev, x_t] + b_o)     how much to show
h_t  = o_t * tanh(c_t)                        output for this word
```

Each gate has an 8 x 24 weight grid plus 8 biases: 800 numbers in layer 1 and 13,888 across all five layers. In `lstm_model.py`'s walk-through they are random, which is why every gate sits near 0.5; `train_lstm.py` learns them.

### Step 4: training the classifier

`train_lstm.py` teaches the hourglass to label each turn as **discussion**, **decision** or **action_item**. It trains the same network `lstm_model.py` runs, plus a small head on top, and saves the weights in the same shapes, so `lstm_model.py` loads them directly.

```mermaid
flowchart LR
    D["training_data.py<br/>labelled meetings"] --> E["embeddings<br/>(learned)"]
    E --> H["hourglass LSTM<br/>5 layers (learned)"]
    H -->|"8 numbers at each<br/>turn's last token"| HD["head: 8 -> 3 scores<br/>(learned)"]
    HD --> P["softmax -><br/>label per turn"]
    P -->|"compare with gold label:<br/>cross-entropy loss"| L(("loss"))
    L -. "backprop through time,<br/>Adam update" .-> E
    L -.-> H
    L -.-> HD
```

| Data | Meetings | Turns | Used for |
|---|---|---|---|
| Hand-written: group project, lab stand-up, CCA committee, club event, orientation planning (7 and 8 Oct) | 8 | 165 | training |
| Hand-written: 24 clubs and student groups, from a photo club to a blood drive (13 Oct) | 24 | 299 | training |
| Generated practice meetings (phrase lists in `training_data.py`) | 60 | 553 | training only |
| Hand-written: club event (a follow-up to a training meeting), plus music club, debate team, coding club, hostel floor (13 Oct) | 5 | 68 | dev: picks the best epoch and seed, and judges retrains |
| Hand-written: group project, hackathon, the sample meeting | 3 | 45 | **test**, scored once |
| Probe set: bare replies ("Sure.", "Okay.") after a request, a proposal or plain information | 12 | 12 replies | **cross-turn test**, never trained on |
| Scorecard (`scorecard_data.py`): hall council, tutoring, band, research group, football, charity drive | 6 | 105 | **second look**, frozen 13 Oct before any training; aggregate scores only |
| Holdout (`holdout_data.py`): two sets written 16 Oct | 17 | 271 | **fresh test** for the 16 Oct update, never trained on |

#### Current results

The shipped weights are the **tuned 13 Oct ensemble**: 5 seeds, each turn's label probabilities averaged, `<unk>` dropout 0.15, learning rate 0.01, 32-number embeddings, all settings chosen on dev only (`python prototype/train_lstm.py` reproduces it, about 12 minutes). Since 16 Oct a text clean-up runs in front of it ("I will" → "I'll", "let us" → "let's", "OK" → "okay", fillers dropped); it changes no dev or test turn. Every candidate was judged by a ship rule written before training. How earlier models got here (7 Oct, the 8 Oct rollback, the first 13 Oct retrain, the tuning runs) is in the [progress log](progress.md#log); the 16 Oct retrains that failed are in the [model update](testing/model_update_16oct/model_update.md).

| KPI | 7 Oct model | **Shipped** | Keyword rules | Always "discussion" |
|---|---|---|---|---|
| Dev accuracy / macro-F1 (68 turns) | 0.90 / 0.87 | **0.96 / 0.95** | | |
| Test accuracy / macro-F1 (45 turns) | 0.84 / 0.82 | 0.84 / 0.825 | 0.69 / 0.59 | 0.56 / 0.24 |
| The two test meetings nobody wrote rules against, accuracy (36 turns) | 0.86 | 0.86 | 0.64 | |
| Probes correct (of 12) | 9 | 9 | 3 | 3 |
| Sample meeting turns correct (of 9) | 7 | 7 | 8 | |
| Scorecard accuracy / macro-F1 (105 turns, scored once) | 0.78 / 0.72 | **0.90 / 0.87** | 0.76 / 0.61 | |
| Holdout 1 / holdout 2 accuracy (149 / 122 turns) | | 0.879 / 0.852 | | |
| 67 meeting tests passed ([testing](testing/testing.md)) | | 45 | 28 | |

- Single seeds of the tuned run range from 0.73 to 0.87 test accuracy; the ensemble is steadier than any one of them.
- Scorecard per label (precision / recall), 7 Oct vs shipped: decision 0.62 / 0.59 vs 0.71 / 0.88, action_item 0.76 / 0.65 vs 0.94 / 0.85, discussion 0.82 / 0.87 vs 0.94 / 0.91.
- What it gets right that keywords can't: "Sure." after "Kai, you handle the judging rubric" (action item) vs "Makes sense." after "How about we split the prizes?" (decision) vs "Okay." after "The bus was late again" (discussion).
- **Flagged failures:** decisions stated as facts ("Venue is confirmed for the 14th", "We have three sponsors confirmed so far") and "Noted, let's add that to the next agenda."; the chair settling a plan ("Okay, Thursday at four then."); 3 of the 4 agreement probes ("Sure.", "Sounds good.", "Fine by me." after a proposal); one missed request ("Could you order for seventy instead of sixty this time?"); two false action items ("Can you remind me where the form is?", "I won't be able to come on the second day"). Decisions are the weak class everywhere, including the meeting tests (9 of 18 found).
- On the sample meeting the minutes list the catering follow-up as an action item, no decisions, and no wrong items. `group_project_3` matches 16/18 hand labels (keyword rules 12/18), `hackathon_1` 15/18 (11/18).
- `python prototype/training_data.py` re-runs the leak check: no probe, test, scorecard, holdout or meeting-test sentence shares 5 words in a row with the training data.
- Caveat: all the data, including the scorecard and holdouts, was written by us (drafted by Claude). Real meetings would settle it.

#### RNN vs LSTM (`compare_rnn.py`)

Pasan and Rahini asked whether the LSTM actually beats a plain RNN here. `compare_rnn.py` builds the same hourglass (same sizes, folds, skips, data, settings and 5 seeds) with plain RNN layers and trains both. The rule was written before the first run: "LSTM better" needs at least +1 probe **and** +0.05 macro-F1 on 5-seed means.

| 5-seed means, test (45 turns) | Accuracy | Macro-F1 | Probes (of 12) |
|---|---|---|---|
| LSTM hourglass, old settings | 0.82 | 0.80 | 7.6 |
| RNN hourglass, old settings | 0.80 | 0.76 | 9.0 |
| LSTM hourglass, tuned settings | 0.80 | 0.77 | 7.8 |
| RNN hourglass, tuned settings | 0.80 | 0.78 | 8.0 |

Verdict, both times: **no clear difference** (old settings: LSTM +0.046 macro-F1 but 1.4 fewer probes; tuned: LSTM −0.01 macro-F1, 0.2 fewer probes). These are single-seed means; the shipped ensemble scores higher than any of them. Memory reach, i.e. how much blanking out the turn d turns back moves the current turn's prediction (0 = no effect, 1 = completely), tuned settings: LSTM 0.053 / 0.006 / 0.002 / 0.001 for 1, 2, 3 and 4 turns back; RNN 0.075 / 0.013 / 0.004 / 0.000. Both rely almost entirely on the previous turn, which is all this labelling task needs, so the workshop shouldn't claim our trained LSTM remembers further back than an RNN. The memory difference is shown with `hands_on.py` instead, where the forget gate is set by hand.

---

## Key commands

### Run

| What | Command |
|---|---|
| Install the one dependency | `pip install -r requirements.txt` |
| Run the whole chain | `python prototype/pipeline.py` |
| Run it on real audio | `python prototype/pipeline.py meeting.wav` (needs `pip install openai-whisper` and ffmpeg) |
| Run all checks before a push or the workshop | `python prototype/run_tests.py` (about 25 s, no training, no internet; `--audio FILE` adds an audio run) |
| Just try it: talk like a meeting and see the result | `python testing/try_it.py` (or `--text` to type instead) |
| Test one stage at a time | `python testing/phases.py` (all four phases, automatic) or `python testing/phases.py --manual --name YOU` (by hand: you give each phase its input) |
| Meeting tests, clean-up tests, audio tests | `python prototype/minutes_tests.py`, `python prototype/normalise_tests.py`, `python testing/solo_test.py --name YOU`: all explained in [testing/testing.md](testing/testing.md) |
| Compare keyword rules with the trained LSTM | `python prototype/pipeline.py --compare` (shows both labels for every turn; also works with `meeting.wav`) |
| Run a hand-labelled meeting | `python prototype/pipeline.py --meeting hackathon_1 --compare` (any meeting name in `training_data.py`; adds the hand labels to the comparison) |
| Run the LSTM stage only | `python prototype/lstm_model.py` |
| Train the LSTM classifier | `python prototype/train_lstm.py` (numpy only, a few minutes) |
| Check the training data | `python prototype/training_data.py` |
| Run your own lines | `python prototype/pipeline.py --compare --text my_meeting.txt` (one "Speaker: text" per line; optional `act` / `dec` / `dis` labels in front; lists words the model never saw) |
| Participants' hands-on | `python prototype/hands_on.py` (`--solution` runs the reference answers) |
| Score the frozen scorecard | `python prototype/scorecard.py` (or `python prototype/scorecard.py other.npz`) |
| RNN vs LSTM | `python prototype/compare_rnn.py` (trains 5 seeds of each, about 15 to 20 minutes) |
| Earlier PyTorch scaffold | `python prototype/train_classifier.py` (needs `pip install torch`) |
| RNN + CTC teaching demo | `python prototype/rnn_ctc_demo.py` |
| Print the placeholder transcript only | `python prototype/transcript_placeholder.py` |
| Check numpy is available | `python -c "import numpy; print(numpy.__version__)"` |

### Google Colab (free tier)

The repo is private, so the simplest route in Colab is to upload the files:

1. Open a new notebook at colab.research.google.com.
2. In the Files panel on the left, upload `transcript_placeholder.py`, `transcriber.py` and `lstm_model.py` (`lstm_model.py` imports the other two). Add `trained_lstm.npz` too if you want the trained weights.
3. Run in a cell:

```python
!python lstm_model.py
```

For the hands-on, also upload `hands_on.py`, `hands_on_reference.py` and `training_data.py` (plus `trained_lstm.npz`), then run `!python hands_on.py`.

numpy is preinstalled on Colab, and the CPU runtime is enough.

### Git workflow

```mermaid
gitGraph
    commit id: "Initial commit"
    commit id: "Prototype + docs"
    branch nomt/meeting-minutes
    checkout nomt/meeting-minutes
    commit id: "minutes v1"
    checkout main
    branch asy/lstm-training
    checkout asy/lstm-training
    commit id: "training loop"
    checkout main
    merge nomt/meeting-minutes
    merge asy/lstm-training
```

```bash
# once
git clone https://github.com/Asyxddiq/mlda-audio-workshop.git
cd mlda-audio-workshop

# every time you start working
git checkout main
git pull

# work on your own branch
git checkout -b yourname/short-topic      # e.g. nomt/meeting-minutes
# ...edit files...
python prototype/lstm_model.py            # make sure it still runs
git add prototype/your_file.py
git commit -m "Short description of the change"
git push -u origin yourname/short-topic

# then open a Pull Request on GitHub and ask a teammate to review
```

Conventions:

- One branch per person per topic: `asy/...`, `nomt/...`, `pasan/...`, `rahini/...`.
- Run `python prototype/lstm_model.py` before every push. If it breaks, fix it before merging.
- Keep the two contracts stable (`get_transcript() -> str`, and the `(tokens, 8)` LSTM output). If one has to change, tell the other track first.
- Update [progress.md](progress.md) when you finish something.

---

## Simplifications and known limitations

| Limitation | Why it's fine for now | What changes later |
|---|---|---|
| Training data is small and written by us (577 hand-written turns, all drafted by Claude, + generated practice meetings) | Enough to show training works and beats keyword rules | Blind cross-check, then real meetings: all the data, including the scorecard and holdouts, is still ours |
| Our trained LSTM and a plain RNN score about the same here, and both use about one turn of context | The labels mostly depend on the previous turn only | Examples where a turn depends on something several turns back, if we want the trained model to show the LSTM's edge |
| Speakers are told apart only by pauses (0.8 s) | Simple and teachable; works for two people taking turns | Replies under 0.8 s merge, a third speaker isn't separated; a diarisation model if needed |
| No attention gates on the skips (the paper has them) | Plain addition is easier to teach | Optional stretch topic |
| Folding lets a word see up to 3 words ahead | Minutes are written from the full transcript anyway | Only matters for live, word-by-word use |
| Simple tokenizer: lowercase words, `?` and `!` as tokens, other punctuation dropped | Simple enough to explain in one slide | A library tokenizer, if ever needed |
| Training is hand-written NumPy backprop, not a framework | No port between frameworks; one dependency; checked against numerical gradients | PyTorch scaffold (`train_classifier.py`) is there if we switch |
| The trained model misses many decisions (reported confirmations like "Venue is confirmed", short ones like "Saturday it is") | Visible, honest failure case for *Try and test* | Retrains with more of our own examples (8, 13 and 16 Oct) did not fix it without breaking something else. Next: real meetings, or a small decision-phrase rule layer |

---

## Interactive explainers

Two interactive pages walk through this backend with the real numbers from the code:

- **Minutes Pipeline**: the four stages, the hourglass layer by layer, and the output turn by turn.
- **Inside the LSTM Cell**: one word at a time through the six steps of `SimpleLSTM.step()`, plus a forget-gate memory demo.

They are private to Asy until shared, so ask Asy for the links.

---

## References

- Li Z, Basit A, Daraz A, Jan A (2024). *Deep causal speech enhancement and recognition using efficient long-short term memory Recurrent Neural Network.* PLOS ONE 19(1): e0291240. https://doi.org/10.1371/journal.pone.0291240
  - Used for: the hourglass layer layout (their Table 1 and Fig 2), lossless reshaping between layers, and skip connections. The paper applies it to cleaning noisy speech; we apply the same shape to transcript tokens.

## Team

MLDA Academics: Asy (Main Comm, LSTM), Nomt (LSTM, meeting minutes), Pasan and Rahini (RNN transcriber).
