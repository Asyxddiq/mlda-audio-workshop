"""
Scorecard: 6 held-out meetings for a second, final look at the model (Asy).

    python prototype/scorecard.py

Written 13 Oct 2026 (rebuild), BEFORE the new training and dev meetings
and before any training run; three lines were reworded (before any score)
because they shared 5 words with old training lines; then frozen: scorecard.py refuses to score if
this file's hash changes. It is never trained on, never used to pick
settings, and scorecard.py prints only aggregate scores, never which turns
were wrong, so it can't be tuned against turn by turn.

Caveat: these were written by Claude after seeing the earlier model's weak
spots (decisions stated as facts, agreements after proposals), so they are
weaker evidence than meetings written by someone who hasn't seen the model.
Real meetings from the team would be a better scorecard.

Same format and labelling rules as training_data.py / labelling_guide.md.
"""

from training_data import _parse

HALL_COUNCIL = """
dis  Grace: Okay, let's get started, there are four things on the list tonight.
dis  Omar: The laundry machines on level two are broken again.
dis  Grace: How many of them?
dis  Omar: Three out of five, since Sunday.
act  Grace: Omar, could you report it to the hall office tomorrow morning?
act  Omar: Sure, I'll go before my class.
dis  Leah: Next is the movie night. Last time only twenty people came.
dis  Ivan: I think the timing was bad, it was the week before exams.
dis  Leah: Should we push it to the Friday after the break?
dec  Grace: Yes, let's do that.
dec  Ivan: Works for me.
act  Leah: I'll put up the posters on Monday.
dis  Omar: For the budget, we spent about two hundred on the welcome party.
dec  Grace: The hall office approved our request for another three hundred.
dis  Ivan: Nice, that's more than I expected.
dis  Leah: Can someone explain what the second part of the form means?
dis  Grace: It's just asking for receipts, nothing else.
dis  Omar: Okay.
dis  Grace: Alright, I think we're done. Thanks all.
"""

TUTORING_PROGRAMME = """
dis  Hana: Hi everyone, quick check in on the tutoring programme.
dis  Felix: We have fourteen tutors signed up and about forty students.
dis  Hana: Got it.
dis  Rosa: The library said we can use the group rooms on Tuesday and Thursday evenings.
dis  Felix: Do we need to book them every week?
dis  Rosa: No, the booking is fixed for the whole term.
dec  Hana: Great, then Tuesday and Thursday evenings it is.
dis  Felix: Some students asked for help with statistics, but none of the tutors cover it.
dis  Rosa: Maybe we can ask the maths society if anyone is interested.
act  Hana: Good idea. Rosa, can you contact their president this week?
act  Rosa: Okay, I'll message her tonight.
dis  Felix: Also, two tutors dropped out because of their internships.
dis  Hana: That's a shame.
dis  Felix: I already updated the timetable to cover their slots.
dis  Hana: Thanks Felix. One more thing, should we give tutors a certificate at the end?
dec  Rosa: Sure.
dec  Felix: Yeah, they'd appreciate that.
act  Rosa: I'll draft one and send it around.
dis  Hana: That's everything for today.
"""

BAND_PRACTICE = """
dis  Zara: Alright, before we play, a few things about the gig.
dis  Theo: Is it still on the twelfth?
dec  Zara: Yes, the cafe confirmed the twelfth, from eight to ten.
dis  Theo: Okay.
dis  Ming: We only have six songs ready, that's not enough for two hours.
dis  Zara: How about we add two covers and do a longer break?
dec  Ming: That works.
dec  Theo: Same.
dis  Ming: My amp keeps cutting out, I think it's the cable.
act  Zara: Can you bring a spare cable next time?
act  Ming: Yep.
act  Theo: The drum stool is also broken, I'll buy a new one this weekend.
dis  Zara: Who's doing the setlist?
act  Ming: I can do it, if you send me the order you want.
dis  Zara: Cool. Let's run through the first song now.
"""

RESEARCH_GROUP = """
dis  Prof Wong: Good afternoon. Let's go around, Aaron first.
dis  Aaron: I reran the experiment with the new dataset, the results are similar to before.
dis  Prof Wong: Similar how?
dis  Aaron: About two percent better on average, but the variance is higher.
act  Prof Wong: Could you run it with three more seeds so we can be sure?
act  Aaron: Okay, I'll start them today.
dis  Bella: I finished the related work section of the paper last night.
dis  Prof Wong: Good. Chen?
dis  Chen: The ethics form for the user study is still with the review board.
dis  Prof Wong: When did you submit it?
dis  Chen: About three weeks ago.
act  Prof Wong: Please email them and ask for an update.
act  Chen: Will do.
dis  Bella: About the conference, should we aim for the March deadline or wait for June?
dis  Aaron: March is tight with the extra runs.
dec  Prof Wong: Then we'll aim for June and do it properly.
dec  Bella: Okay, makes sense.
dis  Prof Wong: Alright, see you all next week.
"""

FOOTBALL_TEAM = """
dis  Coach Dev: Okay team, quick meeting before training.
dis  Sam: The pitch is booked for the friendly on Saturday, right?
dec  Coach Dev: Yes, the sports centre confirmed it yesterday.
dis  Kofi: Do we have enough players? A few people are away.
dis  Sam: We have fifteen so far.
dis  Coach Dev: Okay, that's enough.
dis  Kofi: The new jerseys still haven't arrived.
dis  Sam: I emailed the supplier last week and they said next Wednesday.
dis  Coach Dev: Should we just wear the old ones for the friendly?
dec  Kofi: Yeah, fine by me.
dec  Sam: Agreed.
act  Coach Dev: Kofi, you collect the old jerseys from the storeroom on Friday.
act  Kofi: Sure.
dis  Sam: Someone has to bring the first aid kit.
act  Coach Dev: Sam, please bring it on Saturday.
act  Sam: No problem.
dis  Kofi: Also, the captain's armband is missing.
dis  Coach Dev: We'll look for it later. Let's start the warm up.
"""

CHARITY_DRIVE = """
dis  Isla: Thanks for coming, let's look at the book donation drive.
dis  Marco: We've collected about three hundred books so far.
dis  Isla: Nice.
dis  Yuki: The children's home said they mostly need picture books and textbooks.
dis  Marco: Most of what we got is novels, though.
dis  Isla: What if we sell the novels at a stall and use the money to buy picture books?
dec  Yuki: Good idea.
dec  Marco: Yes, let's go for it.
dis  Yuki: Where would we run the stall?
dis  Marco: The student union lets clubs use the atrium on Wednesdays.
act  Isla: Yuki, could you book the atrium for next Wednesday?
act  Yuki: Okay, I'll book it this afternoon.
dis  Marco: I sorted the books into boxes by type yesterday.
dis  Isla: Thanks Marco.
dec  Yuki: The children's home agreed to collect the books on the twentieth.
dis  Isla: Perfect. Anything else? No? Okay, see you Wednesday.
"""

SCORECARD = [
    {"name": "hall_council_1",       "split": "scorecard", "turns": _parse(HALL_COUNCIL)},
    {"name": "tutoring_programme_1", "split": "scorecard", "turns": _parse(TUTORING_PROGRAMME)},
    {"name": "band_practice_1",      "split": "scorecard", "turns": _parse(BAND_PRACTICE)},
    {"name": "research_group_1",     "split": "scorecard", "turns": _parse(RESEARCH_GROUP)},
    {"name": "football_team_1",      "split": "scorecard", "turns": _parse(FOOTBALL_TEAM)},
    {"name": "charity_drive_1",      "split": "scorecard", "turns": _parse(CHARITY_DRIVE)},
]
