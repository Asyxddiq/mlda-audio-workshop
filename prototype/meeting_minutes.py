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

So this file is split in three:
  - label_turns_keywords(): keyword rules that label ANY transcript. This
    is the default labeller and the agreed fallback (see progress.md
    decisions), so real transcripts from transcriber.py work today.
  - label_turns_stub(): the original hand-labeled stand-in, specific to
    the 9-turn sample transcript. Kept as a reference answer for that
    meeting.
  - generate_minutes(): the real, permanent part. Once a trained
    classifier exists, its predictions replace the keyword labels --
    generate_minutes() doesn't change.

Label scheme
------------
One label dict per turn, aligned by index with `turns`:
    {"category": "decision" | "action_item" | "discussion",
     "owner": str,            # required for "action_item"
     "deadline": str | None}  # optional, "action_item" only
"""

import re
import sys

from lstm_model import run_pipeline

Turn = tuple[str, str]  # (speaker, text)

# Same lists as the keyword baseline in train_classifier.py, so the
# fallback and the baseline the classifier is measured against agree.
# They follow the current request-only action-item rule; update both
# places together when the labelling guide's rule is locked.
_DECISION_KEYWORDS = ["confirmed", "agreed", "approved", "decided", "noted"]
_ACTION_KEYWORDS = ["can you", "could you", "please", "follow up"]
_REQUEST_KEYWORDS = ["can you", "could you", "please"]

_DEADLINE = re.compile(
    r"\b(?:by|before)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|tomorrow|tonight|next week|end of (?:the )?(?:day|week|month))\b"
    r"|\b(today|tomorrow|tonight)\b",
    re.IGNORECASE,
)


def label_turns_keywords(turns: list[Turn]) -> list[dict]:
    """Label every turn with keyword rules. Works on any transcript.

    - category: action_item if an action keyword appears, else decision if
      a decision keyword appears, else discussion (same order as the
      keyword baseline in train_classifier.py).
    - owner (action items): for a request ("can you ..."), the next person
      to speak; otherwise the speaker. "unassigned" if nobody replies.
    - deadline (action items): "by Friday", "by end of week", "today", ...
    """
    labels = []
    for i, (speaker, text) in enumerate(turns):
        lowered = text.lower()
        if any(keyword in lowered for keyword in _ACTION_KEYWORDS):
            if any(keyword in lowered for keyword in _REQUEST_KEYWORDS):
                next_speaker = turns[i + 1][0] if i + 1 < len(turns) else None
                owner = next_speaker if next_speaker and next_speaker != speaker else "unassigned"
            else:
                owner = speaker
            match = _DEADLINE.search(text)
            deadline = (match.group(1) or match.group(2)) if match else None
            labels.append({"category": "action_item", "owner": owner, "deadline": deadline})
        elif any(keyword in lowered for keyword in _DECISION_KEYWORDS):
            labels.append({"category": "decision"})
        else:
            labels.append({"category": "discussion"})
    return labels


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
    # Optional audio file: python prototype/meeting_minutes.py meeting.wav
    # output and turn_end_indices are part of the hand-off contract but
    # unused here for now -- the LSTM is untrained, so there's nothing
    # meaningful to read from them yet.
    audio_path = sys.argv[1] if len(sys.argv) > 1 else None
    turns, _output, _turn_end_indices = run_pipeline(audio_path=audio_path)

    labels = label_turns_keywords(turns)
    print(generate_minutes(turns, labels))
