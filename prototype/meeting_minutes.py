"""
Meeting minutes (Nomt) -- last step of the pipeline.

What this file does
--------------------
Turns labeled turns into a readable Markdown meeting-minutes document.

Where the labels come from
--------------------------
  - label_turns_model(): the TRAINED hourglass LSTM (train_lstm.py) plus
    a small classifier head. The default whenever trained_lstm.npz exists.
  - label_turns_keywords(): keyword rules that label ANY transcript. The
    agreed fallback when there are no trained weights. Can't see the
    previous turn, so it misses replies like "Sure." to a request.
  - label_turns_stub(): the original hand-labelled stand-in for the 9-turn
    sample meeting, kept as a reference answer.
Both real labellers only decide the CATEGORY; add_owners_and_deadlines()
then works out who owns each action item and by when, the same way for
both.

generate_minutes() renders the result. Following labelling_guide.md
(action-item rule locked 7 Oct 2026: label the request AND the reply), a
request and the reply that accepts it become one line in the minutes:
task from the request, owner = the person who accepted.

Label scheme
------------
One label dict per turn, aligned by index with `turns`:
    {"category": "decision" | "action_item" | "discussion",
     "owner": str,            # required for "action_item"
     "deadline": str | None,  # optional, "action_item" only
     "part_of": int}          # optional: this reply accepts the request at that index
"""

import re
import sys

from lstm_model import label_probabilities, load_trained_model, run_pipeline

Turn = tuple[str, str]  # (speaker, text)

# Same lists as the keyword baseline in train_lstm.py and
# train_classifier.py, so the fallback and the baseline the trained model
# is measured against agree.
_DECISION_KEYWORDS = ["confirmed", "agreed", "approved", "decided", "noted"]
_ACTION_KEYWORDS = ["can you", "could you", "please", "follow up"]
_REQUEST_KEYWORDS = ["can you", "could you", "please"]

_DEADLINE = re.compile(
    r"\b(?:by|before)\s+((?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|tomorrow|tonight|next week|(?:the\s+)?end of (?:the\s+)?(?:day|week|month)"
    r"|the\s+\d{1,2}(?:st|nd|rd|th))\b"
    r"|\b(today|tomorrow|tonight|this (?:week|weekend|afternoon|evening))\b",
    re.IGNORECASE,
)


_SELF_ASSIGN = re.compile(r"\b(i'll|i will|i can(?!['’]t)|let me|i'm going to|leave it (?:to|with) me)\b",
                          re.IGNORECASE)
_REQUEST_START = re.compile(
    r"^(?:\w+,\s*)?(?:can you|could you|would you(?: be able to)?|please|"
    r"can someone|could someone|someone needs to|who can|you)\s+", re.IGNORECASE)


def _deadline(text: str) -> str | None:
    match = _DEADLINE.search(text)
    return (match.group(1) or match.group(2)) if match else None


def _is_request(text: str) -> bool:
    """Asking someone else to do it (not 'I'll do it')."""
    lowered = text.lower()
    if _SELF_ASSIGN.search(lowered):
        return False
    return ("?" in text or any(k in lowered for k in _REQUEST_KEYWORDS)
            or re.search(r"\b(you|someone|somebody)\b", lowered) is not None)


def _named_attendee(text: str, attendees: set[str], speaker: str) -> str | None:
    """'Nice, thanks Mei. Arjun, can you ...' -> 'Arjun': the name in the
    LAST sentence that mentions anyone, since that's where the ask is."""
    for sentence in reversed(re.split(r"(?<=[.?!])\s+", text.strip())):
        names = [w for w in re.findall(r"[A-Za-z]+", sentence) if w in attendees and w != speaker]
        if names:
            return names[0]
    return None


def add_owners_and_deadlines(turns: list[Turn], categories: list[str]) -> list[dict]:
    """Turn a list of categories into full label dicts (owner, deadline).

    Works the same whichever labeller produced the categories.

    - An action item asked of someone ("Can you book the room?", "Kai will
      handle security.") followed by an action_item reply from someone
      else ("Sure.") is ONE task: the reply gets "part_of": <request
      index>, and the person who replied is the owner.
    - Otherwise the owner is: the speaker for "I'll ..."; a named attendee
      for "Kai will ..." or "Kai, can you ..."; for an unanswered request,
      the next person to speak; else the speaker.
    - Deadline: "by Friday", "by end of week", "today", ... from the request
      or its reply.
    """
    attendees = {speaker for speaker, _ in turns}
    labels: list[dict] = [{"category": c} for c in categories]
    i = 0
    while i < len(turns):
        speaker, text = turns[i]
        if categories[i] != "action_item":
            i += 1
            continue
        nxt = i + 1
        if (nxt < len(turns) and categories[nxt] == "action_item"
                and turns[nxt][0] != speaker
                and not _SELF_ASSIGN.search(text)        # "I'll do X" needs no acceptance
                and len(text.split()) > 3                # a bare "Sure." isn't a request
                and not _is_request(turns[nxt][1])):     # the reply isn't a new request
            labels[i].update(owner=turns[nxt][0],
                             deadline=_deadline(text) or _deadline(turns[nxt][1]))
            labels[nxt].update(owner=turns[nxt][0], deadline=labels[i]["deadline"], part_of=i)
            i += 2
            continue
        named = _named_attendee(text, attendees, speaker)
        if _SELF_ASSIGN.search(text):
            owner = speaker
        elif named:
            owner = named
        elif _is_request(text):
            later = turns[nxt][0] if nxt < len(turns) else None
            owner = later if later and later != speaker else "unassigned"
        else:
            owner = speaker
        labels[i].update(owner=owner, deadline=_deadline(text))
        i += 1
    return labels


def label_turns_keywords(turns: list[Turn]) -> list[dict]:
    """Label every turn with keyword rules. Works on any transcript.

    The fallback when the trained model (trained_lstm.npz) isn't there.
    category: action_item if an action keyword appears, else decision if a
    decision keyword appears, else discussion (same order as the keyword
    baseline in train_lstm.py). Keywords can't see the previous turn, so a
    bare "Sure." after a request stays discussion here -- the trained model
    is what catches those.
    """
    categories = []
    for _speaker, text in turns:
        lowered = text.lower()
        if any(keyword in lowered for keyword in _ACTION_KEYWORDS):
            categories.append("action_item")
        elif any(keyword in lowered for keyword in _DECISION_KEYWORDS):
            categories.append("decision")
        else:
            categories.append("discussion")
    return add_owners_and_deadlines(turns, categories)


def label_turns_model(turns: list[Turn], output, turn_end_indices: list[int]) -> list[dict] | None:
    """Label turns with the TRAINED hourglass LSTM (see train_lstm.py).

    `output` must come from run_pipeline() with the trained weights loaded.
    The classifier head maps the 8 numbers at each turn's last token to 3
    scores; the highest wins. If trained_lstm.npz holds an ensemble, each
    member's label probabilities are averaged first. Returns None if there
    are no trained weights.
    """
    trained = load_trained_model()
    if trained is None:
        return None
    if len(trained["members"]) > 1:
        # An ensemble (13 Oct): average every member's label probabilities.
        scores = label_probabilities(turns, trained)
    else:
        scores = output[turn_end_indices] @ trained["head_W"].T + trained["head_b"]
    categories = [trained["labels"][k] for k in scores.argmax(axis=1)]
    return add_owners_and_deadlines(turns, categories)


def label_turns_stub(turns: list[Turn]) -> list[dict]:
    """Hand-labelled reference answer for the sample meeting.

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
        {"category": "action_item", "owner": "Priya", "deadline": "Friday", "part_of": 4},
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


def _task_text(text: str) -> str:
    """'Great. Ben, can you put out a call in the group?' -> 'Put out a call in the group'

    For requests, keep just the asking sentence and drop "can you" etc.
    Anything else (e.g. "I'll book the room.") is left as it was said.
    """
    sentences = re.split(r"(?<=[.?!])\s+", text.strip())
    for sentence in sentences:
        match = _REQUEST_START.match(sentence)
        if match:
            task = sentence[match.end():].rstrip(" ?.!")
            return task[:1].upper() + task[1:]
    for sentence in sentences:                        # "Update first. Today I'll do X." -> "Today I'll do X."
        if _SELF_ASSIGN.search(sentence) or re.search(r"\b[A-Z][a-z]+ (?:will|is going to)\b", sentence):
            return sentence
    return text


_PROPOSAL_START = re.compile(
    r"^(?:\w+,\s*)?(?:(?:then|so|and|but|okay|ok|well)\s+)?(?:how about(?: we)?|shall we|i suggest we|i propose we|maybe we should|should we|"
    r"what if we|why don't we|can we|i think we should|then we should|we should|let's)\s+",
    re.IGNORECASE)


_AGREEMENT_WORDS = {"sure", "okay", "ok", "agreed", "sounds", "good", "yeah", "yes", "yep", "fine",
                    "by", "with", "me", "makes", "sense", "works", "for", "idea", "call", "that",
                    "same", "here", "why", "not", "let's", "do", "it", "too", "great", "alright",
                    "absolutely", "definitely", "go", "agree", "i"}


def _is_agreement(words: str) -> bool:
    found = re.findall(r"[a-z']+", words.lower())
    return bool(found) and all(w in _AGREEMENT_WORDS for w in found)


def _decision_text(turns: list[Turn], labels: list[dict], i: int) -> str | None:
    """What was decided, or None if it's already listed.

    An agreement ("Agreed.", "Makes sense.", "Agreed, the old ones are
    falling apart.") says nothing on its own, so show the proposal it
    answers instead:
        "How about we split the prize money?" -> "Makes sense."
        => "Split the prize money (agreed)"
    A second agreement to the same thing ("Fine with me.") is skipped.
    """
    text = turns[i][1]
    parts = re.split(r"[,.!?]", text, maxsplit=1)
    first_clause, rest = parts[0], (parts[1] if len(parts) > 1 else "")
    bare = _is_agreement(text)
    # "Agreed, the old ones are falling apart." agrees and gives a reason;
    # "Okay, we'll wait for the replacements then." states its own decision.
    agrees_with_reason = (_is_agreement(first_clause) and not bare and
                          not re.search(r"\b(we|we'll|let's|then|it is|it's)\b", rest.lower()))
    if not (bare or agrees_with_reason):
        return text
    if i > 0 and labels[i - 1]["category"] == "decision":
        return None if bare else text                 # already covered by the turn before
    for j in range(i - 1, max(-1, i - 4), -1):        # look back up to 3 turns for the proposal
        for sentence in reversed(re.split(r"(?<=[.?!])\s+", turns[j][1].strip())):
            match = _PROPOSAL_START.match(sentence)
            if match:
                proposal = sentence[match.end():].rstrip(" ?.!")
                return proposal[:1].upper() + proposal[1:] + " (agreed)"
        # stop at an objection or new point, so an agreement isn't pinned on a rejected proposal
        if not (_is_agreement(turns[j][1]) or labels[j]["category"] == "decision"):
            break
    if not bare or i == 0:
        return text
    return turns[i - 1][1].rstrip(" ?.!") + " (agreed)"


def generate_minutes(turns: list[Turn], labels: list[dict]) -> str:
    """Render labeled turns as a Markdown meeting-minutes document.

    `labels` must be the same length as `turns`, aligned by index.
    """
    if len(labels) != len(turns):
        raise ValueError(f"Got {len(labels)} labels for {len(turns)} turns.")

    attendees = sorted({speaker for speaker, _ in turns})

    decisions: list[str] = []
    for i, label in enumerate(labels):
        if label["category"] == "decision":
            text = _decision_text(turns, labels, i)
            if text and text not in decisions:
                decisions.append(text)
    # A reply that accepts a request ("Sure.") carries "part_of" and is
    # already covered by the request's line, so it isn't listed again.
    action_items = [
        (_task_text(text), label["owner"], label.get("deadline"))
        for (_, text), label in zip(turns, labels)
        if label["category"] == "action_item" and "part_of" not in label
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
    audio_path = sys.argv[1] if len(sys.argv) > 1 else None
    turns, output, turn_end_indices = run_pipeline(audio_path=audio_path)

    # Trained model if train_lstm.py has been run, keyword rules otherwise.
    labels = label_turns_model(turns, output, turn_end_indices) or label_turns_keywords(turns)
    print(generate_minutes(turns, labels))
