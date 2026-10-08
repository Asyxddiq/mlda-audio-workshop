# Labelling guide (LSTM track)

Shared rules for hand-labelling meeting turns, so Asy and Nomt label the
training set the same way. One label per turn.

| | |
|---|---|
| **Owners** | Asy, Nomt |
| **Status** | Rules locked by Asy (7 Oct 2026); blind cross-check still to do |
| **Feeds** | `prototype/training_data.py` (every labelled meeting), `train_lstm.py`, `meeting_minutes.py` |
| **Last updated** | 8 Oct 2026 |

> **Locked 7 Oct 2026.** Asy confirmed both proposals: action items are
> labelled on both the request and the reply, and the cross-turn probe set
> is in. The first full dataset (`prototype/training_data.py`) follows these
> rules. Still to do: the blind cross-check in step 3 of the process.
>
> Clarified 8 Oct (Nomt's check of his half): echoed replies, ownerless
> "let's", finished tasks vs settled plans. No labels changed.

---

## The three labels

Every turn is exactly one of:

| Label | What it is | Rough test |
|---|---|---|
| `action_item` | Someone commits to (or is asked to) do a specific task | "Who has to do something after this meeting?" |
| `decision` | The group settles something: a choice, an approval, a confirmation | "Did something get agreed or finalised here?" |
| `discussion` | Everything else: updates, questions, context, chit-chat | Default when neither of the above fits |

When two could apply, prefer `action_item` over `decision` over `discussion`.

In `training_data.py` the labels are written as `act`, `dec` and `dis`.

---

## Action items: label the request AND the reply

**[LOCKED 7 Oct]** Label **both** the request and the acceptance as
`action_item`, and merge them into one item when rendering the minutes
(task + deadline from the request, owner from the acceptor).
`generate_minutes()` does this merge.

Why both, rather than one turn:
- The **task and deadline** live in the request ("Can you follow up with
  catering by Friday?").
- The **owner and the commitment** live in the acceptance ("Sure." /
  "Yes, I'll email them today.").
- A bare acceptance like "Sure." has no keyword, so classifying it correctly
  needs the previous turn — which is exactly the cross-turn case that gives
  the LSTM an honest reason to exist (keyword rules can't do it).

A self-assigned action in a single turn ("I'll email them today") is request
and acceptance collapsed into one — label it a single `action_item`.

A reply that **declines** ("Sorry, I can't this week.") is `discussion`; the
request itself is still `action_item`.

> Alternatives considered: **request-only** (kills the cross-turn / memory
> lesson, since requests are keyword-friendly) and **acceptance-only** (the
> acceptance turn is contentless, and requests with no reply get missed).

---

## Decisions: proposals, agreements and the chair

- A proposal still waiting for an answer ("How about we move it to
  Friday?", "I suggest we cap sign-ups at 80.") is `discussion`.
- The reply that agrees to it ("Agreed.", "Sure.", "Fine by me.") is
  `decision`. A reply that objects ("I'm not sure about that.") is
  `discussion`.
- The chair settling a choice ("Okay, Thursday it is.", "Let's go with the
  cheaper one.") is `decision`. Procedural "let's" ("Let's start.", "Let's
  keep monitoring that.") stays `discussion`. A "Let's X" with no owner that
  settles what will happen ("Let's prepare ponchos just in case.") is also
  `decision`; it becomes `action_item` only when someone takes it on
  ("I'll ...") or is named ("Mei, please ...").
- Reporting that something is now final ("The permit was approved on
  Tuesday.", "Venue is confirmed for the 14th.") is `decision`.
- Reporting a task someone has finished ("I submitted the abstract
  yesterday.", "We bought sixty ponchos.") is `discussion`, a status update.
  The test is whether a plan or approval is now settled, not whether
  something happened in the past.
- A reply that just echoes another ("Same.", "Me too.", "Same here.") takes
  the label of the line it echoes: an echoed agreement is `decision`, an
  echoed objection is `discussion`.
- Acknowledging plain information ("The venue closes at ten." → "Okay.") is
  `discussion`.

---

## Tricky cases

| Turn | Label | Note |
|---|---|---|
| "Can you book the room?" → "Sure." | both `action_item` | The "Sure." is the cross-turn case — no keyword, needs memory |
| "How about we move it to Friday?" → "Sure." | proposal `discussion`, "Sure." `decision` | Same reply word, different label: only the previous turn tells them apart |
| "The bus was late again." → "Okay." | both `discussion` | Acknowledging information is not a decision |
| "Let's move it to Friday." → "Agreed." | both `decision` | The chair settling it, then agreement |
| "I won't follow up on that." | `discussion` | Contains "follow up" but is a negation — a keyword rule wrongly flags it |
| "Can you remind me where the form is?" | `discussion` | A question for right now, not a task for after the meeting — a keyword rule wrongly flags it |
| "Priya will sort out catering." | `action_item` | Third-person assignment, no "I'll" — a keyword rule misses it |
| "Venue is confirmed for the 14th." | `decision` | Reporting that something is final |
| "We have three sponsors confirmed so far." | `decision` | Borderline (it's also a running count); kept as `decision` for consistency with the line above |
| "Noted, let's add that to the next agenda." | `decision` | Closing/agreeing to carry forward |
| "I propose we buy tripods." → "Agreed." → "Same here." | `discussion`, `decision`, `decision` | "Same" takes the label of the line it echoes; here an agreement |
| "Should we give certificates?" → "I'd rather wait." → "Same, people like the medals." | all `discussion` | Here "Same" echoes an objection; the decision is the chair's next line, "Okay, we'll wait then." |
| "Let's prepare ponchos just in case." | `decision` | Ownerless "let's" settles a course of action; `action_item` only with "I'll" or a named person |
| "The ponchos are in the storeroom, we bought sixty." | `discussion` | A finished task is a status update; compare "The permit was approved on Tuesday." (`decision`: an approval now settled) |

_Add real examples here as they come up while labelling._

---

## Cross-turn probe set

**[LOCKED 7 Oct]** A curated set of examples whose label provably depends
on the previous turn, held **entirely out of training** and reported as its
own line next to the keyword and majority-class baselines.

This is what carries the "the hourglass earns its keep" claim — a targeted
slice where keyword rules score ~0 and the trained model can win — rather
than aggregate held-out accuracy, which is too small and noisy to trust.

**Done:** 12 two-turn probes in `training_data.py` (`PROBES`). The reply
is scored: 5 after requests (`action_item`), 4 after proposals
(`decision`), 3 after plain information (`discussion`). The same reply
words ("Sure.", "Okay.", "Got it.", "Sounds good.") appear under all three
labels, so memorising "Sure. = action_item" doesn't score well.
`python prototype/training_data.py` checks that no probe or test sentence
shares 5 words in a row with the training data.

---

## Process

1. ~~Agree this guide (resolve the two proposals with Asy).~~ Done 7 Oct.
2. Each write your half of the turns (Nomt: CCA committee + club event;
   Asy: group project + lab stand-up), spreading `action_item` / `decision` /
   `discussion` realistically, with ~15–20% cross-turn cases.
   **First draft of all four meeting types written 7 Oct** (by Claude, for
   Asy): 11 meetings, 211 turns, in `training_data.py`. Nomt to check his
   half.
3. Each label 10 of the **other's** turns blind, then compare — mismatches
   mean this guide needs a clearer rule.

---

## Consistency note (resolved 7 Oct)

The code now follows the locked rule: `training_data.py`,
`label_turns_stub()` in `meeting_minutes.py` (turn 6 is `action_item`),
`train_classifier.py`'s `GOLD_LABELS`, and `generate_minutes()`, which
merges a request and the reply that accepts it into one action item.
