"""
Holdout meetings, written 16 Oct 2026 BEFORE the clean-up step and the new
decision examples were added (Asy + Claude).

These 9 meetings are never trained on and never used to pick an epoch, a seed
or a setting. They exist to answer one question: did the 16 Oct update
(text clean-up + more decision styles) help on meetings the model has not seen?
See testing/model_update_16oct/ship_rule.md for the rule that was fixed before training.

Labels follow labelling_guide.md like training_data.py:
    dis = discussion   dec = decision   act = action_item
Some lines deliberately use the long forms Whisper often writes ("I will",
"let us", "OK", digits) next to the short ones, because a real transcript has both.
"""

from training_data import _parse

HIKING_CLUB = """
dis  Zara: Alright everyone, let us get started. Three items today, the route, the carpool and the gear list.
dis  Omar: Before we begin, has anyone checked the weather for the weekend?
dis  Zara: The forecast says light rain on Saturday and clear on Sunday.
dis  Omar: Then Sunday is better for the long trail.
dec  Zara: Agreed, we are doing the long trail on Sunday then.
dis  Lee: How about we set off at seven so we beat the heat?
dec  Omar: Seven works, let's lock that in.
dis  Zara: For the carpool, I have room for three people in my car.
act  Omar: Lee, can you drive the second car and pick up the group from the north gate?
act  Lee: I will, no problem.
dis  Zara: The permit for the reserve costs fifteen dollars per person.
dis  Omar: That is a bit steep for a student group.
dis  Lee: We could ask the union to pay half of the fee.
act  Zara: Good idea, Omar, please email the union about the permit fee this week.
act  Omar: I will send it tonight.
dis  Zara: One more thing, the first aid kit has expired bandages.
act  Lee: I'll replace them before Saturday.
dis  Zara: Great, that is all for today.
"""

FILM_SOCIETY = """
dis  Priya: Welcome back. Tonight we need to pick the screening for next month.
dis  Jun: I have shortlisted three, a heist film, a documentary and an old musical.
dis  Priya: What did the survey say?
dis  Jun: Most people voted for the documentary.
dis  Alina: The only concern is that it runs almost three hours.
dis  Priya: What if we split it over two evenings?
dec  Alina: Yes please, that solves it.
dec  Jun: OK, documentary over two evenings, final.
act  Priya: Jun, could you ask the film office whether we can borrow the projector twice?
act  Jun: Will do.
dis  Alina: Also, the poster from last time had a typo in the date.
dis  Priya: Yes, it said the twelfth instead of the second.
act  Alina: I will proofread the new one before it goes to print.
dis  Priya: Thank you. And the popcorn machine is broken.
dec  Jun: Forget the machine, we are buying pre-popped bags instead.
act  Alina: Fine, I will compare the prices this week.
dis  Priya: Alright, that is everything. See you all on Thursday.
"""

LAB_GROUP = """
dis  Nina: Morning all. Quick update on the sequencing runs, two finished and one is still queued.
dis  Hamid: The queued one should start tonight.
dis  Nina: Good. We also need to talk about the paper deadline.
dis  Hamid: Is it still the end of the month?
dis  Nina: Yes, the call for papers says the thirtieth.
dis  Rosa: I can draft the methods section, but the figures are not ready.
act  Nina: Hamid, can you regenerate figures two and three by Wednesday?
act  Hamid: Yeah, I can do that.
dis  Rosa: Should we also add the extra control experiment?
dis  Nina: It would take another two weeks.
dis  Hamid: Then we should skip it and mention it as future work.
dec  Nina: Right, we are skipping the control and listing it as future work.
dec  Rosa: Fine with me.
act  Nina: Rosa, please send me the methods draft by Friday.
act  Rosa: Okay, I will.
dis  Nina: Great, thanks both.
"""

STUDENT_COUNCIL = """
dis  Marcus: Let us begin. First, the Friday assembly.
dis  Ella: The hall is booked from two to four.
dis  Marcus: Are we doing the awards before or after the performances?
dis  Ella: After, I think, so people stay until the end.
dec  Marcus: That is settled, awards go last.
dis  Tariq: Should the juniors sit at the front?
dec  Ella: Yes, juniors at the front.
dis  Marcus: Next, the budget. We have 600 dollars left.
dis  Tariq: The sound system rental alone is 200.
dis  Ella: Could we borrow the speakers from the music department instead?
dec  Marcus: Good, we will borrow theirs and drop the rental.
act  Ella: I will talk to the music teacher tomorrow.
dis  Tariq: The banner design is still not finished.
act  Marcus: Tariq, could you finish the banner by Wednesday?
act  Tariq: Got it, I will.
dis  Ella: What about refreshments?
dec  Marcus: No refreshments this time, we are saving the money for the trip.
dis  Tariq: That is everything from my side, thanks.
"""

HOSTEL_FLOOR = """
dis  Dev: Okay, floor meeting. Mainly about the kitchen.
dis  Sam: The fridge smells terrible again.
dis  Dev: Somebody left food in there for two weeks.
dis  Mira: We need a rule for that.
dis  Sam: What if we clean it out every Sunday?
dec  Mira: Yes, every Sunday, starting this weekend.
dis  Dev: Who does it though?
dec  Sam: A rotating schedule, we go in room order.
act  Dev: Okay, I will make the schedule and stick it on the fridge.
dis  Mira: And the common room lights are flickering.
act  Sam: Dev, could you tell the maintenance office about the lights?
act  Dev: Sure, I will email them.
dis  Mira: Also the quiet hours start at 11, but the music is still loud.
dis  Sam: Yeah, it was loud last night.
dec  Dev: We will hold to 11 from now on, no exceptions.
dis  Mira: Okay, that is all I had.
"""

ORCHESTRA = """
dis  Hana: Let us go through the concert plan. We have eleven weeks left.
dis  Luis: The conductor wants to add a fourth piece.
dis  Hana: That is a lot with only eleven weeks.
dis  Beth: I do not mind if the piece is short.
dis  Luis: It is only six minutes.
dec  Hana: Okay then, we are adding it.
dis  Beth: Who takes the solo part?
dec  Luis: The conductor chose Ahmad for the solo.
dis  Hana: Is Ahmad in today?
dis  Beth: No, he has an exam.
act  Hana: Beth, could you tell him about the solo and bring the sheet music?
act  Beth: Yes, I will do that after class.
dis  Luis: We also need to rent the hall for the dress rehearsal.
dis  Hana: How about the Thursday before the concert, at 5 p.m.?
dec  Luis: Thursday works, let us book it.
act  Beth: I will call the hall office tomorrow.
dis  Hana: Perfect, thank you everyone.
"""

CCA_VOTE = """
dis  Chen: Item three, the new membership fee. Currently it is five dollars.
dis  Farid: Some members find even that too much.
dis  Chen: The proposal is to make it free and fund the club from donations.
dis  Yara: I am worried donations will not cover the costs.
dis  Chen: Understood. Can we have a show of hands, who is in favour?
dis  Farid: Four in favour.
dis  Yara: Two against.
dec  Chen: The motion passes, the membership will be free from next term.
dis  Farid: What do we do about the printing costs then?
dis  Yara: We could charge for the newsletter.
dis  Chen: I do not think people would pay for it.
dec  Farid: Let us leave it as it is for now and review at the end of term.
act  Chen: Yara, can you draft the announcement about the free membership?
act  Yara: Yes, I will have it ready by Monday.
dis  Chen: Thanks. Meeting closed.
"""

STARTUP_STANDUP = """
dis  Ivan: Standup time. What are we blocked on?
dis  Joy: The payment API keeps timing out in staging.
dis  Ivan: Is it our code or theirs?
dis  Joy: Looks like theirs, their status page shows an outage.
dis  Noor: We could build a fallback that retries three times.
dec  Ivan: Do it, retries it is.
act  Noor: I will add the retry logic this afternoon.
dis  Joy: For the demo on Friday, should we show the mobile app or the web version?
dis  Ivan: The web version is more stable.
dec  Joy: Then web it is, the mobile app stays out of the demo.
dis  Noor: One more thing, the logo on the landing page is blurry.
act  Ivan: Joy, please swap in the high resolution logo before the demo.
act  Joy: Done, I will do it now.
dis  Noor: I was thinking we should rename the product too.
dis  Ivan: Not now, we are three days from the demo.
dec  Noor: Fine, we keep the current name.
dis  Ivan: Good. Talk tomorrow.
"""

WAITING_ROOM = """
dis  Grace: Hi all. Before we start, did anyone see the email from the principal?
dis  Tom: I saw it, it is about the new timetable.
dis  Grace: Do we know whether it is final?
dis  Tom: Not yet, they said it might still change.
dis  Elena: Last term they promised the same thing and nothing happened.
dis  Grace: True. Should we plan around the old timetable for now?
dis  Elena: I think we should wait for the official notice.
dis  Tom: The office usually posts these on Fridays.
dis  Grace: Then we will see on Friday.
dis  Elena: What about the guest talk? Is it still on?
dis  Tom: I do not know, the speaker has not replied.
dis  Grace: Let us not worry until we hear back.
dis  Elena: There is nothing to do until then.
dis  Tom: Right. Anything else?
dis  Grace: No, that is it. Thanks everyone.
"""

HOLDOUT = [
    {"name": "hiking_club",     "split": "holdout", "turns": _parse(HIKING_CLUB)},
    {"name": "film_society",    "split": "holdout", "turns": _parse(FILM_SOCIETY)},
    {"name": "lab_group",       "split": "holdout", "turns": _parse(LAB_GROUP)},
    {"name": "student_council", "split": "holdout", "turns": _parse(STUDENT_COUNCIL)},
    {"name": "hostel_floor_2",  "split": "holdout", "turns": _parse(HOSTEL_FLOOR)},
    {"name": "orchestra",       "split": "holdout", "turns": _parse(ORCHESTRA)},
    {"name": "cca_vote",        "split": "holdout", "turns": _parse(CCA_VOTE)},
    {"name": "startup_standup", "split": "holdout", "turns": _parse(STARTUP_STANDUP)},
    {"name": "waiting_room",    "split": "holdout", "turns": _parse(WAITING_ROOM)},
]


# ---------------------------------------------------------------------------
# Holdout 2: written later the same day (16 Oct), after round 1 of the update failed the
# ship rule and BEFORE round 2 was trained. Round 1's holdout above had been looked at in
# aggregate while deciding what to change, so the round 2 decision is judged on these.
# ---------------------------------------------------------------------------

SWIMMING_CLUB = """
dis  Carla: Right, swimming club. Gala preparation first.
dis  Dmitri: The pool is only free on Tuesday nights.
dis  Carla: Then we have two practices before the gala.
dis  Eun: Why don't we add a Sunday morning practice as well?
dec  Dmitri: Yes, that gives people more time.
dec  Carla: Alright, Sunday mornings are added.
dis  Eun: The team caps are the wrong colour this year.
act  Carla: Eun, could you exchange them at the shop this week?
act  Eun: Yes, I'll go on Thursday.
dis  Dmitri: Parents want to know what time the gala starts.
dis  Carla: It starts at nine and runs until about one.
dis  Eun: Should we send the timetable by email or put it on the notice board?
dec  Dmitri: Email, it reaches everyone.
act  Carla: Dmitri, please write the email and send it to the parents list.
act  Dmitri: Will do.
dis  Eun: That is all I had.
"""

CODING_BOOTCAMP = """
dis  Felix: Standup. Who is blocked?
dis  Gwen: I am waiting for the API keys from the vendor.
dis  Felix: They said they would arrive today, right?
dis  Gwen: That was the promise, yes.
dis  Hiro: In the meantime, we could use the mock server.
dec  Felix: Good, use the mock server until the keys arrive.
dis  Hiro: There's also the question of the database. Postgres or SQLite?
dis  Gwen: SQLite is simpler for the demo.
dec  Felix: SQLite then.
dis  Hiro: The login page looks different on phones.
act  Felix: Gwen, can you fix the layout before Thursday?
act  Gwen: Yes, I will look at it tomorrow morning.
dis  Hiro: Should we add dark mode?
dis  Felix: We do not have time for that.
dec  Gwen: Right, no dark mode for the demo.
dis  Felix: Okay, talk tomorrow.
"""

CHARITY_RUN = """
dis  Haley: Thanks for coming. We are three weeks from the charity run.
dis  Ivor: Registrations are at 140 so far.
dis  Haley: The target was 200, so we need a push.
dis  June: What if we offer a free t-shirt to everyone who signs up this week?
dec  Haley: Yes please, let's do that.
dis  Ivor: The route passes the main road, which needs a permit.
act  Haley: Ivor, can you apply for the road permit by Friday?
act  Ivor: Okay, I'll fill in the form tonight.
dis  June: Do we want a five kilometre route or ten?
dis  Haley: Last year most people preferred the shorter one.
dec  Ivor: Five kilometres, we are keeping it simple.
dis  June: What about water stations?
dis  Haley: We can place one at the halfway point.
dec  June: Good, one station at halfway.
dis  Ivor: Perfect. See you next week.
"""

SCHOOL_NEWSPAPER = """
dis  Kofi: Production meeting. The next issue goes to print on the 12th.
dis  Lara: I have the sports page nearly done.
dis  Kofi: What about the interview with the principal?
dis  Lara: Still waiting for her to approve the quotes.
dis  Marco: If she does not reply by Monday, we could run it without the last paragraph.
dec  Lara: Okay, that is the fallback.
act  Kofi: Lara, can you chase her office tomorrow?
act  Lara: Sure, I'll call them.
dis  Marco: The cover photo is a bit dark.
dis  Kofi: We could brighten it, or choose the second photo.
dec  Marco: The second photo, it is much sharper.
dis  Lara: Should we run the crossword this time?
dec  Kofi: Yes, the crossword goes in.
dis  Marco: Fine. Anything else?
dis  Kofi: No, that's it. Good work, everyone.
"""

CLEANUP_DAY = """
dis  Nora: Okay, the neighbourhood clean-up. We have the second Saturday of the month.
dis  Oscar: I counted around 30 people who said yes.
dis  Nora: That is a good number. Gloves and bags are the main cost.
dis  Pearl: The council might supply them for free.
dis  Nora: Has anyone asked them?
dis  Oscar: Not yet.
act  Nora: Pearl, could you email the council this week?
act  Pearl: I will, tonight.
dis  Oscar: Where do we start? The park or the riverside?
dis  Pearl: The riverside is much dirtier.
dec  Nora: Riverside first, then the park if there is time.
dis  Oscar: Is lunch on us?
dec  Nora: No lunch, but we will have water and fruit.
dis  Pearl: Okay, I think that covers it.
dis  Oscar: Great, thanks everyone.
"""

DEBATE_PREP = """
dis  Quinn: Debate prep. The motion is about school uniforms.
dis  Rosa: We are proposing, so we speak first.
dis  Quinn: Right. Who takes the opening?
dis  Sandeep: I think Rosa should open, she is the strongest speaker.
dec  Quinn: Agreed, Rosa opens.
act  Sandeep: I'll take the closing speech then.
dis  Rosa: We need statistics on how many schools have dropped uniforms.
act  Quinn: Rosa, could you find three good sources by Wednesday?
act  Rosa: Yes, I will start tonight.
dis  Sandeep: What if we begin with a personal story?
dec  Quinn: Good idea, we open with a story.
dis  Rosa: The other team will probably argue about cost.
dis  Sandeep: Then we should prepare a short answer on that.
dec  Quinn: Yes, short answer on cost, and keep it under a minute.
dis  Rosa: Okay, that is enough for today.
"""

ALLOTMENT_GROUP = """
dis  Tabitha: Allotment meeting. Seeds first.
dis  Ugo: The tomato seeds from last year did not grow well.
dis  Tabitha: Maybe the soil was too cold.
dis  Vera: I read that they need a warmer start indoors.
dis  Ugo: We could try that this year.
dis  Tabitha: Or we could buy seedlings instead.
dis  Vera: Seedlings are expensive.
dis  Ugo: I'm leaning towards starting them indoors.
dis  Tabitha: Let's sleep on it and talk next week.
dis  Vera: The water butt is leaking, by the way.
dis  Ugo: It has been like that since March.
dis  Tabitha: It lost a lot of water last week.
dis  Vera: Right. That's all from me for today.
"""

CULTURE_NIGHT = """
dis  Wanjiru: Culture night. We have 12 acts so far.
dis  Xin: Ten minutes each is too long, we would be there until midnight.
dis  Wanjiru: What if we cap each act at six minutes?
dec  Xin: Six it is.
dis  Yusra: And the order of acts?
dis  Wanjiru: We could put the dance groups at the end.
dec  Yusra: Dance groups last, definitely.
dis  Xin: Food stalls are next. The union allows at most five.
dis  Yusra: We have eight applications.
dis  Wanjiru: We should pick by lottery to be fair.
dec  Xin: Lottery, I like that.
act  Wanjiru: Yusra, please run the draw on Monday.
act  Yusra: Okay, will do.
dis  Xin: The sound check is at four.
dis  Wanjiru: Everyone has to be there.
dec  Xin: No exceptions, we start without anyone who is late.
dis  Yusra: Great, see you all then.
"""

HOLDOUT2 = [
    {"name": "swimming_club",    "split": "holdout2", "turns": _parse(SWIMMING_CLUB)},
    {"name": "coding_bootcamp",  "split": "holdout2", "turns": _parse(CODING_BOOTCAMP)},
    {"name": "charity_run",      "split": "holdout2", "turns": _parse(CHARITY_RUN)},
    {"name": "school_newspaper", "split": "holdout2", "turns": _parse(SCHOOL_NEWSPAPER)},
    {"name": "cleanup_day",      "split": "holdout2", "turns": _parse(CLEANUP_DAY)},
    {"name": "debate_prep",      "split": "holdout2", "turns": _parse(DEBATE_PREP)},
    {"name": "allotment_group",  "split": "holdout2", "turns": _parse(ALLOTMENT_GROUP)},
    {"name": "culture_night",    "split": "holdout2", "turns": _parse(CULTURE_NIGHT)},
]


if __name__ == "__main__":
    from collections import Counter
    for name, ms in (("holdout", HOLDOUT), ("holdout2", HOLDOUT2)):
        counts = Counter(label for m in ms for _s, _t, label in m["turns"])
        n = sum(counts.values())
        print(f"{name} {len(ms)} meetings, {n} turns  "
              + "  ".join(f"{k} {v} ({v / n:.0%})" for k, v in sorted(counts.items())))
