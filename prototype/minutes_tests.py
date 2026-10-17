"""
Meeting tests: give the prototype things people actually say in meetings and
check the minutes that come out (14 Oct 2026).

    python prototype/minutes_tests.py              # all cases, trained model
    python prototype/minutes_tests.py -v           # also print each failing case's minutes
    python prototype/minutes_tests.py A07 E02      # just these cases (always verbose)
    python prototype/minutes_tests.py --keywords   # same cases with the keyword fallback

Like testing a food-ordering app with "one burger, add fries, less salt":
each case is a short typed transcript plus the minutes a person would write
for it. Words only, no audio (audio_check.py is the audio test). A case
PASSES when the minutes have exactly the expected decisions and action
items, each action with the right owner and due date.

Categories
  A  action items: requests, offers, assignments, declines, due dates
  B  decisions: proposals agreed, chair decides, confirmations, rejected ideas
  C  discussion only: nothing should reach the minutes (keyword traps)
  D  the same short reply ("Sure.", "Okay.") after a request / proposal / info
  E  whole meetings

Expected items are matched loosely: a decision passes if the expected words
appear in one of the listed decisions (case ignored); an action passes if
the task words appear in its line AND the owner and due date are exact
(due None = no due date shown). Extra items fail the case.

The full list, with the expected minutes for each case, is in
"testing/testing script.md" (written from this file: python prototype/minutes_tests.py --markdown).

None of these lines are in training_data.py, scorecard_data.py or the
generator (checked: no 5 words in a row shared), so they are new to the model.
Not to be used for tuning: like the scorecard, they test, they don't train.
"""

import re
import sys

from lstm_model import run_pipeline
from meeting_minutes import generate_minutes, label_turns_keywords, label_turns_model

CATEGORIES = {"A": "Action items", "B": "Decisions", "C": "Discussion only (nothing in the minutes)",
              "D": "Same reply, different meaning", "E": "Whole meetings"}

# (id, what it tests, transcript lines, expected minutes)
# expected = {"decisions": [words], "actions": [(task words, owner, due or None)]}
CASES = [
    # ---- A: action items ------------------------------------------------
    ("A01", "request + yes", [
        "Mei: Ben, can you book the seminar room for Thursday?",
        "Ben: Sure, I'll do it now."],
     {"decisions": [], "actions": [("book the seminar room", "Ben", None)]}),
    ("A02", "request + bare 'Okay'", [
        "Mei: Could you send the slides to the group tonight?",
        "Ben: Okay."],
     {"decisions": [], "actions": [("send the slides", "Ben", "tonight")]}),
    ("A03", "offer with a due date", [
        "Ben: I'll write to the sponsor by Friday."],
     {"decisions": [], "actions": [("write to the sponsor", "Ben", "Friday")]}),
    ("A04", "assigned to someone not in the meeting", [
        "Mei: Kai will handle the registration desk."],
     {"decisions": [], "actions": [("registration desk", "Kai", None)]}),
    ("A05", "conditional offer", [
        "Ben: I can redo the poster tonight if nobody minds."],
     {"decisions": [], "actions": [("redo the poster", "Ben", "tonight")]}),
    ("A06", "request turned down", [
        "Mei: Can you get the name tags printed tomorrow?",
        "Ben: Sorry, I can't, I'm away tomorrow."],
     {"decisions": [], "actions": [("name tags", "unassigned", "tomorrow")]}),
    ("A07", "'Will do.' + 'before the next meeting'", [
        "Hana: Please update the member list before the next meeting.",
        "Omar: Will do."],
     {"decisions": [], "actions": [("update the member list", "Omar", "the next meeting")]}),
    ("A08", "'Who can ...?' + 'I can.'", [
        "Hana: Who can collect the sign-in sheets?",
        "Omar: I can."],
     {"decisions": [], "actions": [("collect the sign-in sheets", "Omar", None)]}),
    ("A09", "'Let me ...'", [
        "Omar: Let me look at the booking calendar after this."],
     {"decisions": [], "actions": [("booking calendar", "Omar", None)]}),
    ("A10", "'Someone has to ...' + 'Leave it with me.'", [
        "Hana: Someone has to ring the coach company.",
        "Omar: Leave it with me."],
     {"decisions": [], "actions": [("ring the coach company", "Omar", None)]}),
    ("A11", "direct order, no 'please'", [
        "Lina: Arjun, you take the opening speech.",
        "Arjun: Okay."],
     {"decisions": [], "actions": [("opening speech", "Arjun", None)]}),
    ("A12", "two tasks to two people", [
        "Lina: Dev, could you buy the decorations by Friday?",
        "Dev: Yes.",
        "Lina: And Sara, please sort out the music.",
        "Sara: No problem."],
     {"decisions": [], "actions": [("buy the decorations", "Dev", "Friday"),
                                   ("sort out the music", "Sara", None)]}),
    ("A13", "due date given in the reply", [
        "Lina: Can you pull together a feedback summary?",
        "Dev: Yes, I'll have it ready by Monday."],
     {"decisions": [], "actions": [("feedback summary", "Dev", "Monday")]}),
    ("A14", "'X is going to ...'", [
        "Sara: Nadia is going to prepare a welcome note."],
     {"decisions": [], "actions": [("welcome note", "Nadia", None)]}),
    ("A15", "'Could someone ...?' + 'I'll do it.'", [
        "Lina: Could someone reserve the library discussion room?",
        "Dev: I'll do it."],
     {"decisions": [], "actions": [("reserve the library discussion room", "Dev", None)]}),
    ("A16", "'I'm going to ...' + tonight", [
        "Dev: I'm going to post the recording online tonight."],
     {"decisions": [], "actions": [("post the recording", "Dev", "tonight")]}),
    ("A17", "'are you able to' + this week", [
        "Hana: Omar, are you able to email the caterer this week?",
        "Omar: Yeah, no problem."],
     {"decisions": [], "actions": [("email the caterer", "Omar", "this week")]}),
    ("A18", "'can't today, but I'll ...' is still a yes", [
        "Hana: Can you test the projector before Friday?",
        "Omar: I can't today, but I'll do it tomorrow."],
     {"decisions": [], "actions": [("test the projector", "Omar", "Friday")]}),
    ("A19", "turned down, then someone else takes it", [
        "Hana: Can you print the handouts?",
        "Omar: Sorry, I'm swamped.",
        "Lina: I'll print them."],
     {"decisions": [], "actions": [("print", "Lina", None)]}),
    ("A20", "'chase' + 'Sure.'", [
        "Mei: Ben, can you chase the venue about parking?",
        "Ben: Sure."],
     {"decisions": [], "actions": [("chase the venue", "Ben", None)]}),

    # ---- B: decisions ---------------------------------------------------
    ("B01", "proposal + 'Sounds good.'", [
        "Mei: How about we move the meeting to Friday?",
        "Ben: Sounds good."],
     {"decisions": ["move the meeting to Friday"], "actions": []}),
    ("B02", "chair decides after a comparison", [
        "Ben: The big hall costs more but fits everyone.",
        "Mei: Okay, let's go with the big hall then."],
     {"decisions": ["big hall"], "actions": []}),
    ("B03", "decision stated as a fact", [
        "Ben: Venue's confirmed for the 14th."],
     {"decisions": ["confirmed for the 14th"], "actions": []}),
    ("B04", "approval reported", [
        "Mei: The dean approved our budget this morning."],
     {"decisions": ["approved our budget"], "actions": []}),
    ("B05", "proposal + objection = no decision", [
        "Mei: Should we cancel the dry run?",
        "Ben: I'd rather not, we still need the practice."],
     {"decisions": [], "actions": []}),
    ("B06", "proposal + 'Yeah, let's do that.'", [
        "Aisha: What if the banner uses the green design?",
        "Tom: Yeah, let's do that."],
     {"decisions": ["green design"], "actions": []}),
    ("B07", "two people agree = still one decision", [
        "Aisha: I suggest we cap sign-ups at forty.",
        "Tom: Agreed.",
        "Lina: Fine by me."],
     {"decisions": ["cap sign-ups at forty"], "actions": []}),
    ("B08", "chair's 'let's' on its own", [
        "Aisha: Let's keep the club meeting on Thursdays."],
     {"decisions": ["Thursdays"], "actions": []}),
    ("B09", "'we're going with'", [
        "Tom: Okay, final answer, we're going with pizza."],
     {"decisions": ["pizza"], "actions": []}),
    ("B10", "'that's settled'", [
        "Aisha: So that's settled, the trip is on the 21st."],
     {"decisions": ["trip is on the 21st"], "actions": []}),
    ("B11", "proposal, objection, chair settles", [
        "Tom: Should we hire a photographer?",
        "Lina: It's too expensive.",
        "Aisha: Okay, no photographer this year."],
     {"decisions": ["no photographer"], "actions": []}),
    ("B12", "booking came through", [
        "Tom: The booking for the sports hall came through yesterday."],
     {"decisions": ["sports hall"], "actions": []}),
    ("B13", "agreement + echo ('Same here.')", [
        "Aisha: Shall we kick off at nine?",
        "Tom: Works for me.",
        "Lina: Same here."],
     {"decisions": ["kick off at nine"], "actions": []}),
    ("B14", "'We'll hold ... in ...'", [
        "Aisha: We'll hold the bake sale in the canteen."],
     {"decisions": ["bake sale in the canteen"], "actions": []}),
    ("B15", "proposal ignored, topic changes", [
        "Tom: Maybe we should change the logo?",
        "Lina: Anyway, about the budget."],
     {"decisions": [], "actions": []}),
    ("B16", "negative decision", [
        "Aisha: Right, we're not doing the raffle."],
     {"decisions": ["not doing the raffle"], "actions": []}),
    ("B17", "show of hands", [
        "Aisha: Hands up for Saturday. Okay, that's most of you, Saturday it is."],
     {"decisions": ["Saturday"], "actions": []}),
    ("B18", "'Can we ...?' is a proposal, not a task", [
        "Tom: Can we push the deadline to Monday?",
        "Aisha: Yes, let's do Monday."],
     {"decisions": ["Monday"], "actions": []}),

    # ---- C: discussion only ---------------------------------------------
    ("C01", "information + 'Okay.'", [
        "Ben: The level three printer is broken again.",
        "Mei: Okay."],
     {"decisions": [], "actions": []}),
    ("C02", "status update on a finished task", [
        "Ben: I sent the treasurer the invoice yesterday."],
     {"decisions": [], "actions": []}),
    ("C03", "a question for right now", [
        "Ben: Can you tell me where the sign-up form went?",
        "Mei: It's on the module page."],
     {"decisions": [], "actions": []}),
    ("C04", "'follow up' in a negative sentence", [
        "Ben: I'm not going to follow up on it, it's sorted already."],
     {"decisions": [], "actions": []}),
    ("C05", "procedural 'let's'", [
        "Mei: Okay, let's start. First item is the budget."],
     {"decisions": [], "actions": []}),
    ("C06", "'decided' in 'haven't decided'", [
        "Tom: We still haven't decided the theme."],
     {"decisions": [], "actions": []}),
    ("C07", "'confirmed' in 'hasn't confirmed'", [
        "Lina: The sponsor still hasn't confirmed anything."],
     {"decisions": [], "actions": []}),
    ("C08", "'Could you repeat that?'", [
        "Tom: Could you repeat that?",
        "Lina: Sure, I said we need more chairs."],
     {"decisions": [], "actions": []}),
    ("C09", "'Please, go ahead.'", [
        "Tom: Can I add something?",
        "Aisha: Please, go ahead."],
     {"decisions": [], "actions": []}),
    ("C10", "'confirmed now?' + 'Not yet.'", [
        "Tom: Is the venue confirmed now?",
        "Aisha: Not yet."],
     {"decisions": [], "actions": []}),
    ("C11", "an opinion", [
        "Lina: I think the old logo looks nicer."],
     {"decisions": [], "actions": []}),
    ("C12", "closing", [
        "Aisha: Thanks everyone, see you next week."],
     {"decisions": [], "actions": []}),
    ("C13", "a 'what if' wish", [
        "Tom: If we had more money we could hire a band."],
     {"decisions": [], "actions": []}),
    ("C14", "a decision from last year", [
        "Ben: Last year we agreed on a cap and it didn't really work."],
     {"decisions": [], "actions": []}),
    ("C15", "'I'll' in a figure of speech", [
        "Lina: I'll believe it when I see it."],
     {"decisions": [], "actions": []}),
    ("C16", "question about a done task", [
        "Tom: Did you manage to email the sponsor?",
        "Lina: Yes, I emailed them on Monday."],
     {"decisions": [], "actions": []}),

    # ---- D: same reply, different meaning --------------------------------
    ("D01", "'Sure.' after a request = action", [
        "Aisha: Tom, can you collect the banner from the print shop?",
        "Tom: Sure."],
     {"decisions": [], "actions": [("collect the banner", "Tom", None)]}),
    ("D02", "'Sure.' after a proposal = decision", [
        "Aisha: Shall we meet at the cafe instead?",
        "Tom: Sure."],
     {"decisions": ["meet at the cafe"], "actions": []}),
    ("D03", "'Okay.' after information = nothing", [
        "Aisha: The lift is out of order today.",
        "Tom: Okay."],
     {"decisions": [], "actions": []}),
    ("D04", "'Got it.' after a request = action", [
        "Aisha: Lina, please send the photos to the committee.",
        "Lina: Got it."],
     {"decisions": [], "actions": [("send the photos", "Lina", None)]}),
    ("D05", "'Got it.' after information = nothing", [
        "Aisha: The bus leaves at seven sharp.",
        "Lina: Got it."],
     {"decisions": [], "actions": []}),
    ("D06", "'Sounds good.' after a proposal = decision", [
        "Tom: Why don't we split into two teams?",
        "Aisha: Sounds good."],
     {"decisions": ["split into two teams"], "actions": []}),
    ("D07", "'Sounds good.' after information = nothing", [
        "Tom: The caterer can do vegetarian options too.",
        "Aisha: Sounds good."],
     {"decisions": [], "actions": []}),
    ("D08", "'Okay.' after a request = action", [
        "Aisha: Tom, could you check the microphones before Saturday?",
        "Tom: Okay."],
     {"decisions": [], "actions": [("check the microphones", "Tom", "Saturday")]}),
    ("D09", "'Okay.' after a proposal = decision", [
        "Tom: How about we finish at five?",
        "Aisha: Okay."],
     {"decisions": ["finish at five"], "actions": []}),

    # ---- E: whole meetings ----------------------------------------------
    ("E01", "the sample meeting (transcript_placeholder.py)", [
        "Alex: Let's kick off. First item is the Q3 budget review.",
        "Priya: I've sent the numbers over, we're currently 8% under budget.",
        "Alex: Good, let's keep monitoring that. Next, the workshop venue.",
        "Priya: Venue is confirmed for the 14th, catering still pending.",
        "Alex: Can you follow up with catering by Friday?",
        "Priya: Yes, I'll email them today.",
        "Alex: Great, last item, feedback from the last session.",
        "Priya: Mostly positive, a few people wanted more hands-on time.",
        "Alex: Noted, let's add that to the next agenda. Meeting adjourned."],
     {"decisions": ["Venue is confirmed", "next agenda"],
      "actions": [("follow up with catering", "Priya", "Friday")]}),
    ("E02", "club welcome party", [
        "Aisha: Okay everyone, welcome party planning first.",
        "Tom: We've booked the function room for the 3rd.",
        "Aisha: Great. What should we serve?",
        "Lina: How about noodles? They're cheap and everyone likes them.",
        "Tom: Fine with me.",
        "Aisha: Lina, can you place the food order by Wednesday?",
        "Lina: Sure.",
        "Tom: I'll make the poster tonight.",
        "Aisha: Perfect, thanks all."],
     {"decisions": ["booked the function room", "noodles"],
      "actions": [("food order", "Lina", "Wednesday"),
                  ("make the poster", "Tom", "tonight")]}),
    ("E03", "project demo prep", [
        "Farah: Quick check on the demo. It crashed twice yesterday.",
        "Arjun: I think it's the login page.",
        "Farah: Can you fix it before Friday?",
        "Arjun: Yes, I'll fix it.",
        "Farah: Also, please send me the error logs today.",
        "Arjun: Okay.",
        "Farah: And we'll present in person, not online.",
        "Arjun: Agreed."],
     {"decisions": ["present in person"],
      "actions": [("fix it", "Arjun", "Friday"),
                  ("send me the error logs", "Arjun", "today")]}),
    ("E04", "budget meeting with a no", [
        "Hana: First item, the budget. We have about two hundred left.",
        "Omar: Should we spend it on new jerseys?",
        "Dev: I'd rather keep some for the finals trip.",
        "Hana: Okay, we'll keep half for the trip.",
        "Omar: Makes sense.",
        "Hana: Dev, could you get quotes for the jerseys by next week?",
        "Dev: Sorry, I can't, exams.",
        "Omar: I can do it.",
        "Hana: Thanks Omar. That's all for today."],
     {"decisions": ["keep half for the trip"],
      "actions": [("quotes for the jerseys", "Omar", "next week")]}),
]


# ---------------------------------------------------------------------------
# Running a case
# ---------------------------------------------------------------------------

_ACTION_LINE = re.compile(r"^- \[ \] (.*) -- (\S+)(?: \(due (.*)\))?$")


def minutes_for(lines: list[str], keywords: bool = False) -> str:
    turns, output, ends = run_pipeline(transcript="\n".join(lines))
    labels = None if keywords else label_turns_model(turns, output, ends)
    return generate_minutes(turns, labels or label_turns_keywords(turns))


def read_minutes(minutes: str) -> tuple[list[str], list[tuple[str, str, str | None]]]:
    decisions, actions, section = [], [], None
    for line in minutes.splitlines():
        if line.startswith("## "):
            section = line[3:]
        elif line.startswith("- ") and line != "- (none)":
            if section == "Decisions":
                decisions.append(line[2:])
            elif section == "Action Items":
                m = _ACTION_LINE.match(line)
                if m:
                    actions.append((m.group(1), m.group(2), m.group(3)))
    return decisions, actions


def check(expected: dict, minutes: str) -> list[str]:
    """Empty list = pass; otherwise what's wrong, in plain words."""
    decisions, actions = read_minutes(minutes)
    problems = []
    left = list(decisions)
    for words in expected["decisions"]:
        hit = next((d for d in left if words.lower() in d.lower()), None)
        if hit is None:
            problems.append(f"missing decision: {words!r}")
        else:
            left.remove(hit)
    problems += [f"extra decision: {d!r}" for d in left]
    left = list(actions)
    for task, owner, due in expected["actions"]:
        same_task = [a for a in left if task.lower() in a[0].lower()]
        exact = next((a for a in same_task if a[1] == owner and a[2] == due), None)
        if exact:
            left.remove(exact)
        elif same_task:
            got = same_task[0]
            left.remove(got)
            problems.append(f"action {task!r}: got owner {got[1]}, due {got[2]} "
                            f"(expected {owner}, due {due})")
        else:
            problems.append(f"missing action: {task!r} -- {owner}")
    problems += [f"extra action: {a[0]!r} -- {a[1]}" for a in left]
    return problems


def run(ids: list[str], verbose: bool, keywords: bool) -> int:
    chosen = [c for c in CASES if not ids or c[0] in ids]
    if ids and len(chosen) != len(set(ids)):
        sys.exit(f"Unknown case id(s): {sorted(set(ids) - {c[0] for c in chosen})}")
    print(f"Meeting tests, {'keyword fallback' if keywords else 'trained model'}: {len(chosen)} cases\n")
    totals = {}
    for cid, what, lines, expected in chosen:
        minutes = minutes_for(lines, keywords)
        problems = check(expected, minutes)
        ok = not problems
        t = totals.setdefault(cid[0], [0, 0])
        t[0] += ok
        t[1] += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {cid}  {what}")
        for p in problems:
            print(f"          {p}")
        if not ok and (verbose or ids):
            print("          transcript:")
            for line in lines:
                print(f"            {line}")
            print("          minutes:")
            for line in minutes.splitlines():
                if line.startswith(("## ", "- ")):
                    print(f"            {line}")
    print()
    passed = sum(t[0] for t in totals.values())
    for cat, (p, n) in sorted(totals.items()):
        print(f"  {cat}  {CATEGORIES[cat]:<42} {p:>2}/{n}")
    print(f"\n  Total {passed}/{len(chosen)} passed")
    return 0 if passed == len(chosen) else 1


# ---------------------------------------------------------------------------
# The Markdown list ("testing/testing script.md")
# ---------------------------------------------------------------------------

def markdown() -> str:
    out = ["# Testing script: meeting tests", "",
           "Things people say in meetings, and the minutes the prototype should write for each.",
           "Words only (typed transcripts); `audio_check.py` is the audio test.", "",
           "Run them all with one command:", "",
           "```", "python prototype/minutes_tests.py        # trained model",
           "python prototype/minutes_tests.py -v     # show the minutes for failing cases",
           "python prototype/minutes_tests.py A07    # one case",
           "python prototype/minutes_tests.py --keywords   # keyword fallback instead", "```", "",
           "A case passes when the minutes have exactly these decisions and action items, each",
           "action with the right owner and due date. The cases live in `minutes_tests.py`;",
           "this file is written from it (`python prototype/minutes_tests.py --markdown`).",
           "They are for testing only, never for training or tuning.", ""]
    for cat, title in CATEGORIES.items():
        out += [f"## {cat}: {title}", ""]
        for cid, what, lines, expected in (c for c in CASES if c[0][0] == cat):
            out += [f"### {cid}: {what}", "", "```"] + lines + ["```", "", "Expected minutes:", ""]
            for d in expected["decisions"]:
                out.append(f"- Decision mentioning \"{d}\"")
            for task, owner, due in expected["actions"]:
                out.append(f"- Action \"{task}\" -- owner **{owner}**" + (f", due **{due}**" if due else ", no due date"))
            if not expected["decisions"] and not expected["actions"]:
                out.append("- Nothing (no decisions, no action items)")
            out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--markdown" in args:
        print(markdown())
        sys.exit(0)
    flags = {a for a in args if a.startswith("-")}
    ids = [a.upper() for a in args if not a.startswith("-")]
    sys.exit(run(ids, verbose="-v" in flags, keywords="--keywords" in flags))
