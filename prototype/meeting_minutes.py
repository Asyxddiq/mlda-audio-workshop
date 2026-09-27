"""
Meeting minutes (Nomt) -- last step of the pipeline.

What this file does
--------------------
Turns labeled turns into a readable Markdown meeting-minutes document.

The catch (see progress.md): the hourglass LSTM in lstm_model.py is
untrained, so its output vectors don't mean anything yet -- nothing has
ever taught it what "this turn is a decision" or "this turn is an action
item" looks like. Turning that into per-turn labels needs a trained
classifier on top of the LSTM output, which isn't built yet.

So this file is split in two:
  - label_turns_stub(): a hand-labeled stand-in for that not-yet-trained
    classifier, specific to the current sample transcript. This is the
    same kind of placeholder trick as transcript_placeholder.py -- it
    lets generate_minutes() be built and tested end to end today.
  - generate_minutes(): the real, permanent part. Once a trained
    classifier exists, its predictions replace label_turns_stub()'s
    output -- generate_minutes() doesn't change.

Label scheme
------------
One label dict per turn, aligned by index with `turns`:
    {"category": "decision" | "action_item" | "discussion",
     "owner": str,            # required for "action_item"
     "deadline": str | None}  # optional, "action_item" only
"""

from lstm_model import run_pipeline

Turn = tuple[str, str]  # (speaker, text)


def label_turns_stub(turns: list[Turn]) -> list[dict]:
    """Hand-labeled stand-in for the not-yet-trained classifier.

    Specific to the 9-turn sample transcript in transcript_placeholder.py --
    it will need updating (or replacing with real predictions) if that
    transcript changes.
    """
    stub_labels = [
        {"category": "discussion"},
        {"category": "discussion"},
        {"category": "discussion"},
        {"category": "decision"},
        {"category": "action_item", "owner": "Priya", "deadline": "Friday"},
        {"category": "discussion"},
        {"category": "discussion"},
        {"category": "discussion"},
        {"category": "decision"},
    ]

    if len(stub_labels) != len(turns):
        raise ValueError(
            f"Stub has {len(stub_labels)} labels but the transcript has "
            f"{len(turns)} turns -- update label_turns_stub() to match it."
        )
    return stub_labels


def generate_minutes(turns: list[Turn], labels: list[dict]) -> str:
    """Render labeled turns as a Markdown meeting-minutes document.

    `labels` must be the same length as `turns`, aligned by index.
    """
    if len(labels) != len(turns):
        raise ValueError(f"Got {len(labels)} labels for {len(turns)} turns.")

    attendees = sorted({speaker for speaker, _ in turns})

    decisions = [
        text for (_, text), label in zip(turns, labels)
        if label["category"] == "decision"
    ]
    action_items = [
        (text, label["owner"], label.get("deadline"))
        for (_, text), label in zip(turns, labels)
        if label["category"] == "action_item"
    ]

    lines = ["# Meeting Minutes", "", f"**Attendees:** {', '.join(attendees)}", ""]

    lines.append("## Decisions")
    if decisions:
        lines.extend(f"- {text}" for text in decisions)
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Action Items")
    if action_items:
        for text, owner, deadline in action_items:
            due = f" (due {deadline})" if deadline else ""
            lines.append(f"- [ ] {text} -- {owner}{due}")
    else:
        lines.append("- (none)")

    return "\n".join(lines)


if __name__ == "__main__":
    # output and turn_end_indices are part of the hand-off contract but
    # unused here for now -- the LSTM is untrained, so there's nothing
    # meaningful to read from them yet. Importing run_pipeline() anyway
    # proves the wiring works end to end against the real contract.
    turns, _output, _turn_end_indices = run_pipeline()

    labels = label_turns_stub(turns)
    print(generate_minutes(turns, labels))
