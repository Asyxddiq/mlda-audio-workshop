# mlda-audio-workshop

Backend prototype for the MLDA Academics workshop **Deep Learning in Audio Applications**.

The pipeline is audio → transcript → LSTM → meeting minutes. It is written as small, readable NumPy code so that beginner-to-intermediate participants can follow every step and run it on free Google Colab.

| Component | Owner | Status (26 Sep 2026) |
|---|---|---|
| Placeholder transcript | Asy | Working |
| 5-layer hourglass LSTM (forward pass, untrained) | Asy | Working |
| Real audio → transcript (RNN) | Pasan, Rahini | In progress, separate track |
| Meeting minutes from LSTM output | Nomt | Not started, next |

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
Vocab size: 69 words (incl. <eot>)
Tokens in conversation: 86 (padded to 88 so it divides by 4)
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
    T1 --> P1["prototype/transcript_placeholder.py<br/>(stand-in for now)"]
    T2 --> P2["prototype/lstm_model.py"]
    T3 --> P3["prototype/meeting_minutes.py<br/>(to be written)"]
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

### Three-weekend build plan

Dates assume the weekends of 26 Sep, 3 Oct and 10 Oct. Adjust them in this file if the team's calendar differs.

```mermaid
gantt
    title MVP build, then slides
    dateFormat YYYY-MM-DD
    axisFormat %d %b
    section Weekend 1 - build pieces
    Placeholder + LSTM + hourglass (Asy)      :done,   a1, 2026-09-25, 3d
    Audio to transcript (Pasan + Rahini)     :active, t1, 2026-09-26, 2d
    Minutes pipeline sketched (Nomt)          :        n1, 2026-09-26, 2d
    Sync on transcript format                 :milestone, m1, 2026-09-27, 0d
    section Weekend 2 - connect
    Real transcript into the LSTM             :        c1, 2026-10-03, 2d
    Decide 80/20 and Part A/B split           :        c2, 2026-10-03, 2d
    Full chain runs once                      :milestone, m2, 2026-10-04, 0d
    section Weekend 3 - freeze
    Participant starter code + debug notes    :        f1, 2026-10-10, 2d
    MVP frozen                                :milestone, m3, 2026-10-11, 0d
    section After
    Slides and learning materials             :        s1, 2026-10-12, 7d
```

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
├── progress.md                        Asy's LSTM-track progress tracker
├── requirements.txt                   numpy
└── prototype/
    ├── transcript_placeholder.py      stand-in for the transcriber (Track 1)
    └── lstm_model.py                  tokenize, embed, 5-layer hourglass LSTM (Track 2)
```

`prototype/meeting_minutes.py` does not exist yet. Nomt owns it.

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
    Tok-->>Main: 86 tokens (with <eot>) + turn_end_indices
    Main->>Tok: build_vocab, embed_tokens
    Tok-->>Main: 86 x 16 numbers
    Main->>Main: pad_to_multiple_of_4 -> 88 x 16
    Main->>HG: forward(x_padded)
    HG->>HG: layer 1 -> fold -> layer 2 -> fold -> layer 3
    HG->>HG: unfold + skip -> layer 4 -> unfold + skip -> layer 5
    HG-->>Main: 88 x 8 output (+ every layer's output)
    Main->>Out: drop padding -> 86 x 8
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
    A["'Venue is confirmed ...'"] -->|"split on spaces,<br/>lowercase"| B["venue · is · confirmed · ..."]
    B -->|"look up / assign id"| C["ids: 27 · 5 · 28 · ..."]
    C -->|"pick row id of the<br/>69 x 16 embedding table"| D["16 numbers per word"]
    E["end of each turn"] -->|"insert <eot>"| B
```

- **Vocabulary:** 69 unique tokens, including the `<eot>` end-of-turn marker.
- **Embeddings:** random for now (seed 42). Training would move words used in similar ways closer together.
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
    L5 --> OUT["output<br/>drop padding -> 86 x 8"]
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

Each gate has an 8 x 24 weight grid plus 8 biases: 800 numbers in layer 1 and 13,888 across all five layers. They are random until the model is trained, which is why every gate currently sits near 0.5.

### Component status

```mermaid
stateDiagram-v2
    direction LR
    state "Transcript" as T {
        [*] --> Placeholder
        Placeholder --> RealTranscriber: Track 1 ready (Weekend 2)
    }
    state "LSTM" as L {
        [*] --> Untrained
        Untrained --> Trained: data + training decided
    }
    state "Minutes" as M {
        [*] --> NotStarted
        NotStarted --> Built: Nomt
    }
    T --> L: contract 1
    L --> M: contract 2
```

Current state: **Placeholder**, **Untrained**, **NotStarted**.

---

## Key commands

### Run

| What | Command |
|---|---|
| Install the one dependency | `pip install -r requirements.txt` |
| Run the whole prototype | `python prototype/lstm_model.py` |
| Print the placeholder transcript only | `python prototype/transcript_placeholder.py` |
| Check numpy is available | `python -c "import numpy; print(numpy.__version__)"` |

### Google Colab (free tier)

The repo is private, so the simplest route in Colab is to upload the two files:

1. Open a new notebook at colab.research.google.com.
2. In the Files panel on the left, upload `transcript_placeholder.py` and `lstm_model.py` (both are needed; one imports the other).
3. Run in a cell:

```python
!python lstm_model.py
```

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
| Weights are random (no training) | Proves shapes and wiring end to end | Choose a task + data, then train |
| Transcript is a hard-coded sample | Lets the LSTM track work before the transcriber exists | Swap in Track 1 behind the same `get_transcript()` |
| No attention gates on the skips (the paper has them) | Plain addition is easier to teach | Optional stretch topic |
| Folding lets a word see up to 3 words ahead | Minutes are written from the full transcript anyway | Only matters for live, word-by-word use |
| Whitespace tokenizer ("14th," keeps its comma) | Simple enough to explain in one slide | Strip punctuation or use a library tokenizer |
| NumPy forward pass only | Gates stay visible as plain maths; runs anywhere | Training may move to PyTorch or Keras on Colab |

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
