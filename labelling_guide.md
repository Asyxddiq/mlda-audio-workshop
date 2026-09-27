# Labelling guide (LSTM track)

Shared rules for hand-labelling meeting turns, so Asy and Nomt label the
~100-turn training set the same way. One label per turn.

| | |
|---|---|
| **Owners** | Asy, Nomt |
| **Status** | Draft stub — some rules **proposed, pending Asy** (flagged below) |
| **Feeds** | `train_classifier.py` (`GOLD_LABELS`), `meeting_minutes.py` |
| **Last updated** | 27 Sep 2026 |

> **Draft.** The three categories are agreed. The action-item rule and the
> probe set are proposals from the RESEARCHER session, not yet confirmed by
> Asy. Nothing in the code has been changed to match the proposed rule yet —
> see the note at the end.

---

## The three labels

Every turn is exactly one of:

| Label | What it is | Rough test |
|---|---|---|
| `action_item` | Someone commits to (or is asked to) do a specific task | "Who has to do something after this meeting?" |
| `decision` | The group settles something: a choice, an approval, a confirmation | "Did something get agreed or finalised here?" |
| `discussion` | Everything else: updates, questions, context, chit-chat | Default when neither of the above fits |

When two could apply, prefer `action_item` over `decision` over `discussion`.

---

## Action items: request, acceptance, or both?

**[PROPOSED, pending Asy]** Label **both** the request and the acceptance as
`action_item`, and merge them into one item when rendering the minutes
(task + deadline from the request, owner from the acceptor).

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

> Alternatives considered: **request-only** (kills the cross-turn / memory
> lesson, since requests are keyword-friendly) and **acceptance-only** (the
> acceptance turn is contentless, and requests with no reply get missed).

---

## Tricky cases (to expand with Asy)

| Turn | Label | Note |
|---|---|---|
| "Can you book the room?" → "Sure." | both `action_item` | The "Sure." is the cross-turn case — no keyword, needs memory |
| "Let's move it to Friday." → "Agreed." | request `discussion`/`decision`, "Agreed." = `decision` | Acceptance of a proposal is the decision point |
| "I won't follow up on that." | `discussion` | Contains "follow up" but is a negation — a keyword rule wrongly flags it |
| "Priya will sort out catering." | `action_item` | Third-person assignment, no "I'll" — a keyword rule misses it |
| "Noted, let's add that to the next agenda." | `decision` | Closing/agreeing to carry forward |

_Add real examples here as they come up while labelling._

---

## Cross-turn probe set

**[PROPOSED, pending Asy]** Curate 8–10 examples whose label provably depends
on the previous turn (bare acceptances like "Sure." / "Agreed." after a
request or proposal). Hold them **entirely out of training** and report the
model's score on them as its own line, next to the keyword and majority-class
baselines.

This is what carries the "the hourglass earns its keep" claim — a targeted
slice where keyword rules score ~0 and the trained model can win — rather
than aggregate held-out accuracy, which is too small and noisy to trust.

---

## Process

1. Agree this guide (resolve the two proposals with Asy).
2. Each write your half of the ~100 turns (Nomt: CCA committee + club event;
   Asy: group project + lab stand-up), spreading `action_item` / `decision` /
   `discussion` realistically, with ~15–20% cross-turn cases.
3. Each label 10 of the **other's** turns blind, then compare — mismatches
   mean this guide needs a clearer rule.

---

## Consistency note (open follow-up)

Until the action-item rule is locked, the code deliberately still uses the
older **request-only** labelling:

- `train_classifier.py` → `GOLD_LABELS` (turn 5 `action_item`, turn 6 `discussion`)
- `meeting_minutes.py` → `label_turns_stub()`

When the rule is locked, update **both** of those and add the
request→acceptance merge to `generate_minutes()` in the same change, so the
documents and the code never disagree.
