"""
Hand-labelled meetings for training the LSTM classifier (Asy + Nomt).

STATUS: first draft, written and labelled by Claude for Asy on 7 Oct 2026,
following labelling_guide.md with the action-item rule locked as "label
BOTH the request and the reply". Still to do (guide, "Process" step 3):
Asy and Nomt each label 10 turns blind and compare. Edit freely -- the
training script reads whatever is here.

Format: one turn per line,  <label>  <Speaker>: <text>
    dis  = discussion     dec = decision     act = action_item

Labelling rules used (see labelling_guide.md for the full version)
-----------------------------------------------------------------
act  A request for a task ("Can you ...?", "Ben, you take ...") AND the
     reply that accepts it ("Sure.", "Okay, will do."). Also self-assigned
     ("I'll book the room") and third-person ("Kai will handle security").
dec  The group settles something: agreeing to a proposal ("Agreed.",
     "Yeah, let's do that."), the chair settling a choice ("Okay, Thursday
     it is.", "Let's go with the cheaper one."), or reporting that
     something is now final ("The permit was approved on Tuesday.", like
     "Venue is confirmed" in the sample). Procedural "let's" ("Let's
     start.", "Let's keep monitoring that.") stays dis.
dis  Everything else: updates, questions, proposals still waiting for an
     answer, acknowledging information ("Okay.", "Got it." after an update).
     When two apply: act > dec > dis.

Bare replies ("Sure.", "Okay.", "Got it.", "Sounds good.") appear under
all three labels on purpose: after a request they are act, after a
proposal dec, after plain information dis. Only the previous turn tells
them apart, which is the job keyword rules can't do and the LSTM's memory
can.

Splits (always by whole meeting, never by turn)
-----------------------------------------------
train  8 meetings: the model learns from these
dev    1 meeting:  picks the best epoch (early stopping)
test   3 meetings: scored once at the end; includes the 9-turn sample
                   meeting the pipeline demos, so the demo shows held-out
                   performance, not memorised answers
PROBES 12 two-turn exchanges, never trained on (guide: cross-turn probe
       set). Only the reply is scored.
"""

LABEL_CODES = {"dis": "discussion", "dec": "decision", "act": "action_item"}


def _parse(block: str) -> list[tuple[str, str, str]]:
    """'act  Arjun: Sure.' -> ('Arjun', 'Sure.', 'action_item')"""
    turns = []
    for line in block.strip().splitlines():
        code, rest = line.split(maxsplit=1)
        speaker, text = rest.split(":", 1)
        turns.append((speaker.strip(), text.strip(), LABEL_CODES[code]))
    for (a, _, _), (b, _, _) in zip(turns, turns[1:]):
        # A real transcript merges back-to-back lines from one speaker.
        assert a != b, f"two turns in a row by {a}: merge them into one line"
    return turns


# ---------------------------------------------------------------------------
# Group project (Asy's half)
# ---------------------------------------------------------------------------

GROUP_PROJECT_1 = """
dis  Wei: Okay everyone, let's start. We have three things today, the report, the demo and the slides.
dis  Farah: Before that, did anyone hear back from Professor Tan about the extension?
dis  Arjun: Not yet, I emailed him on Monday.
dis  Wei: Alright. For the report, we're at about sixty percent of the methods section.
dis  Mei: I finished the literature review last night, it's in the shared folder.
act  Wei: Nice, thanks Mei. Arjun, can you write up the results section by Sunday?
act  Arjun: Sure.
dis  Farah: For the demo, should we use the webcam version or the recorded video?
dis  Mei: The webcam one crashed twice during testing.
dis  Wei: How about we use the recorded video as the main demo and keep the webcam as a backup?
dec  Arjun: Yeah, let's do that.
act  Farah: I'll trim the video down to two minutes.
dis  Wei: Good. On the slides, the template from last semester looks a bit dated.
act  Mei: I can redo the template tonight if nobody minds.
dis  Wei: That would be great.
dis  Arjun: One thing, the dataset link in the appendix is broken.
act  Wei: Could you fix the link when you do the results?
act  Arjun: Okay, will do.
dis  Farah: Also, we haven't decided who presents which part.
dec  Wei: Let's keep it simple, each person presents the section they wrote.
dec  Mei: Agreed.
dec  Farah: Fine by me.
dec  Wei: Okay, that's settled then. Same time next week.
"""

GROUP_PROJECT_2 = """
dis  Ravi: Hey, can everyone hear me?
dis  Siti: Yes, loud and clear.
dis  Ben: Yep.
dis  Ravi: So the prototype is mostly working, but the login page still throws an error.
dis  Ben: That's the token expiry bug, I know where it is.
act  Ravi: Can you push a fix before Wednesday's check-in?
act  Ben: Yeah, no problem.
dis  Siti: I tested the search feature yesterday and it's quite slow on large files.
dis  Ravi: How slow are we talking?
dis  Siti: About eight seconds for a fifty page document.
dis  Ravi: Okay.
dis  Ben: We could cache the results instead of searching every time.
dec  Ravi: Let's try caching first and see if that's enough.
act  Siti: Okay, I'll add the cache and time it again.
dis  Ravi: Next, the user survey. We only have twelve responses so far.
dis  Ben: Can you believe only twelve people out of the whole cohort replied?
dis  Siti: Maybe we should post it in the course forum as well.
act  Ravi: Sure, let's do that. Siti, please post it there tonight.
act  Siti: Got it.
dis  Ben: Oh, and the client meeting got moved to Friday afternoon.
dis  Ravi: Okay, good to know. I think that's everything. Thanks guys.
"""

GROUP_PROJECT_3 = """
dis  Arjun: Let's start, we only have the room for half an hour.
dis  Nadia: I merged everyone's code into the main branch last night.
dis  Ben: Did the tests pass?
dis  Nadia: All except the export function.
act  Arjun: Ben, could you look at the export tests today?
act  Ben: Okay, I'll check them after lunch.
dis  Arjun: For the final report, the lecturer wants it in the conference template.
dis  Nadia: That template has a strict four page limit.
dis  Ben: Then we should move the user study details to the appendix.
dec  Arjun: Agreed.
act  Nadia: I'll move that section over tonight.
dis  Arjun: Also, the peer review form is due on Monday.
dis  Ben: Can you remind me where the form is?
dis  Arjun: It's on the course page under assignments.
dis  Nadia: Should we meet again on Thursday before submission?
dec  Arjun: Sure.
dec  Ben: Thursday works for me too.
dec  Arjun: Okay, Thursday at four then. Thanks everyone.
"""

# ---------------------------------------------------------------------------
# Lab stand-up (Asy's half)
# ---------------------------------------------------------------------------

LAB_STANDUP_1 = """
dis  Daniel: Morning all, quick round of updates. Nadia, you start.
act  Nadia: I finished training the baseline, accuracy is about seventy one percent. Today I'll go through the misclassified samples.
dis  Daniel: Good. Kai?
dis  Kai: Still stuck on the data loader, it runs out of memory on the full set.
dis  Hui: I had the same issue last month, it's the image cache.
act  Daniel: Hui, could you pair with Kai on that after this?
act  Hui: Sure.
dis  Kai: Thanks, that would help a lot.
dis  Daniel: Hui, your update?
dis  Hui: I submitted the abstract for the conference yesterday.
dis  Daniel: Great. Was the figure approved by Professor Lim before you sent it?
dec  Hui: Yes, he signed off on it on Friday.
dis  Daniel: Okay. On my side, the GPU booking for next week is open.
dis  Nadia: Should we book the whole Tuesday for the big run?
act  Daniel: Yeah, let's do that, I'll put in the booking this afternoon.
dis  Kai: One question, are we still doing the reading group on Thursday?
dec  Daniel: Let's keep it on Thursday, same room.
dec  Nadia: Sounds good.
dis  Daniel: Alright, that's it, back to work.
"""

LAB_STANDUP_2 = """
dis  Daniel: Okay, let's go. Kai first today.
dis  Kai: The data loader works now, thanks to Hui. The full training run started last night.
dis  Hui: Glad it worked.
dis  Daniel: How long will the run take?
dis  Kai: Probably until tomorrow morning.
act  Daniel: Can you send the loss curves to the group chat once it finishes?
act  Kai: Will do.
dis  Nadia: I looked at the misclassified samples, most of them are blurry images.
dis  Hui: We could filter out anything below a sharpness threshold.
dis  Nadia: Or we keep them and add a blur augmentation.
dec  Daniel: Let's try the augmentation, filtering throws away data.
act  Nadia: Okay, I'll set that up today.
dis  Hui: The conference sent the review dates, reviews come back on the twentieth.
act  Daniel: Good to know. Also, please remember to log your GPU hours in the sheet.
act  Nadia: Sure.
dis  Kai: I won't be in on Friday, I have a medical appointment.
act  Daniel: No problem, just update the board before you go.
act  Kai: Okay.
dis  Daniel: That's all, thanks everyone.
"""

# ---------------------------------------------------------------------------
# CCA committee (Nomt's half -- drafted here, Nomt to check)
# ---------------------------------------------------------------------------

CCA_COMMITTEE_1 = """
dis  Marcus: Alright, let's begin. First, the budget for next semester.
dis  Priya: We have about eleven hundred dollars left after the camp.
dis  Alex: Does that include the deposit we paid for the studio?
dis  Priya: Yes, the deposit is already deducted.
dis  Marcus: Okay. Then I propose we spend four hundred on new tripods.
dec  Farah: Agreed, the old ones are falling apart.
dec  Alex: Same here.
act  Marcus: Okay, tripods are approved. Priya, could you get three quotes from suppliers by next Friday?
act  Priya: Sure, I'll ask around.
dis  Farah: For publicity, the instagram page has been quiet for a month.
dis  Alex: Maybe we can post the photos from the camp.
act  Farah: Good idea, I'll post a few every week starting Monday.
act  Marcus: Please make sure the people in the photos are okay with it.
act  Farah: Of course.
dis  Alex: Next item is the annual general meeting date.
dis  Marcus: Last year it was the second week of March.
dis  Alex: Can we do the first week instead? The second week clashes with midterms.
dec  Priya: Fine by me.
dec  Marcus: Okay, first week of March it is.
act  Alex: I'll book the lecture theatre then.
dis  Marcus: Great, anything else? No? Then we're done.
"""

CCA_COMMITTEE_2 = """
dis  Marcus: Okay, let's get started, we don't have much time.
dis  Priya: I got the three quotes for the tripods. The cheapest is three hundred and sixty.
dis  Alex: Is that the one with the carrying bags?
dis  Priya: Yes, bags are included.
dec  Marcus: Then let's go with that one.
dec  Farah: Sounds good.
act  Marcus: Priya, please place the order this week.
act  Priya: Okay.
dis  Farah: The instagram posts are doing well, we got forty new followers.
dis  Alex: Nice, that's more than last semester in total.
dis  Marcus: Should we run a photo contest to keep the momentum going?
dec  Alex: Yeah, why not.
act  Farah: I can handle the posters for the contest.
dis  Marcus: Thanks Farah.
dec  Alex: For the AGM, the lecture theatre is booked for the third of March.
dis  Priya: Do we need to send out a notice to members?
dis  Alex: Yes, the constitution says two weeks before.
act  Marcus: Alex, could you draft the notice and send it to me first?
act  Alex: Sure.
dis  Farah: I won't follow up on the sponsor email, they replied that they're not interested.
dis  Marcus: That's a pity, but okay. Meeting adjourned.
"""

# ---------------------------------------------------------------------------
# Club event (Nomt's half -- drafted here, Nomt to check)
# ---------------------------------------------------------------------------

CLUB_EVENT_1 = """
dis  Siti: Thanks for coming. The charity run is in three weeks, so let's check where we are.
dis  Ben: Registration has a hundred and twenty sign ups so far.
dis  Siti: Got it.
dis  Kai: That's already more than last year.
dis  Siti: What about the route permit?
dec  Mei: The permit was approved on Tuesday.
act  Siti: Great, we still need volunteers for the water stations. Ben, can you put out a call in the telegram group?
act  Ben: Yep, I'll do it after this.
dis  Kai: For the finish line, do we want medals or certificates?
dis  Mei: Medals cost more, but people like them.
dis  Siti: How about medals for the top fifty and certificates for everyone?
dec  Kai: That works.
dec  Ben: Agreed.
act  Siti: Okay, medals plus certificates. Mei, could you order the medals?
act  Mei: Sure, I'll check the supplier from last year.
dis  Kai: Should someone contact the campus security office?
act  Siti: Good point. Kai will handle the security office.
act  Kai: Okay.
dis  Mei: Last thing, the forecast says it might rain that morning.
dec  Siti: Let's prepare ponchos just in case.
dec  Ben: Good call.
dis  Siti: Okay, that's all for today.
"""

CLUB_EVENT_2 = """
dis  Siti: Okay, one week to go. Quick updates please.
dis  Ben: We have eighteen volunteers now, which covers all the water stations.
act  Siti: Perfect. Can you send them the briefing document by Thursday?
act  Ben: Sure.
dis  Mei: The medals arrived yesterday, but two boxes have the wrong year printed.
dis  Kai: Can the supplier replace them in time?
dis  Mei: They said four working days.
dis  Siti: That's cutting it close. Should we just give certificates to those runners?
dis  Ben: I'd rather wait for the replacements.
dis  Kai: Same, people really like the medals.
dec  Siti: Okay, we'll wait for the replacements then.
act  Mei: I'll call the supplier again tomorrow to check.
dec  Kai: The security office confirmed two officers for the morning.
dis  Siti: Great, thanks Kai.
dis  Ben: Should we do a dry run of the registration booth?
act  Siti: Yes, let's do it on Friday evening. Mei, please book the tables from the student union.
act  Mei: Okay, will do.
dis  Kai: Do we have a plan if it rains?
dis  Ben: The ponchos are in the storeroom, we bought sixty.
dis  Siti: Okay. See everyone on race day.
"""

HACKATHON_1 = """
dis  Hui: Okay, let's get going. The hackathon is in a month.
dec  Ravi: We have three sponsors confirmed so far.
dis  Farah: Do we know how much prize money that gives us?
dis  Ravi: About two thousand dollars in total.
dis  Hui: How about we split it into first, second and third prizes?
dec  Kai: Makes sense.
dec  Farah: Fine with me.
act  Hui: Okay. Kai, you handle the judging rubric.
act  Kai: Sure.
dis  Ravi: The venue team asked if we need the auditorium or the smaller rooms.
dis  Farah: The auditorium is too big for sixty people.
dec  Hui: Let's take the smaller rooms then.
act  Ravi: I'll let the venue team know.
dis  Farah: For food, last year we ran out of dinner.
act  Hui: Could you order for seventy instead of sixty this time?
act  Farah: Yes, I'll update the order by Wednesday.
dis  Kai: I won't be able to come on the second day, by the way.
dis  Hui: That's fine, we have enough helpers. Alright, thanks everyone.
"""

# Added 8 Oct (training only): settled plans reported as facts (dec) next to
# finished tasks and still-pending items (dis), following the 8 Oct guide.
ORIENTATION_1 = """
dis  Lina: Okay, let's begin. Orientation week is in nine days.
dec  Tom: The sports hall booking came through yesterday, so day one is sorted.
dis  Aisha: Nice. What about the welcome packs?
dis  Jun: I finished packing two hundred of them last night.
dis  Lina: Great, thanks Jun.
dec  Tom: The dean also approved the extra budget for snacks.
dis  Aisha: How much did we get?
dis  Tom: Three hundred dollars.
dis  Jun: We still haven't heard back about the second hall.
act  Lina: Tom, could you chase the facilities office tomorrow?
act  Tom: Okay.
dis  Aisha: I printed the campus maps this morning, they're in the club room.
dec  Lina: Noted, we'll hand them out at registration.
dis  Jun: The bus company replied but their quote is too high.
dis  Aisha: Should we just walk the groups over instead?
dec  Tom: Yeah, let's do that.
dec  Lina: Okay, walking it is. The safety briefing is fixed for nine on the first morning.
dis  Jun: That's all from me.
dis  Lina: Thanks everyone, see you on Monday.
"""

# ---------------------------------------------------------------------------
# Added 13 Oct (training only): 24 more meetings in new settings, so the
# model sees more ways of saying the same thing. Written by Claude, same
# rules as labelling_guide.md. Batch 1 (12) then batch 2 (12).
# ---------------------------------------------------------------------------

PHOTO_CLUB_1 = """
dis  Jia: Hi all, let's start. The exhibition is in two weeks.
dis  Ken: I printed the first twenty photos this afternoon.
dis  Jia: Looks great.
dis  Lara: Do we have frames for all of them?
dis  Ken: Only for twelve, the shop ran out.
act  Jia: Lara, could you check the other branch tomorrow?
act  Lara: Sure, it's on my way home.
dis  Ken: Should we print captions under each photo or put a list at the door?
dis  Lara: A list is cheaper.
dec  Jia: Okay, a list at the door then.
dec  Ken: Fine with me.
dec  Lara: The arts centre booked us the small gallery for the whole week.
dis  Jia: Brilliant. That's it for now.
"""

DRAMA_CLUB_1 = """
dis  Nora: Okay, quick production meeting before rehearsal.
dis  Eli: The script changes are done, I sent them to the cast last night.
dis  Nora: Thanks.
dis  Pia: The lights in the hall flicker during the second act.
dis  Eli: Is that the bulbs or the dimmer?
dis  Pia: Probably the dimmer, the technician said it's old.
act  Nora: Can you ask the technician to look at it before Friday?
act  Pia: Okay.
act  Eli: For costumes, we're still missing the two soldier jackets. I'll borrow them from the theatre society.
dis  Pia: Should we sell tickets online this year?
dec  Nora: Yes, let's do it online.
dec  Eli: Agreed, it's easier for everyone.
dis  Nora: Alright, places for act one please.
"""

ROBOTICS_TEAM_1 = """
dis  Raj: Morning, let's go through the robot before the qualifier.
dis  Sofia: The left motor overheats after ten minutes.
dis  Tim: I replaced the fan on Monday, it didn't help.
dis  Raj: Hmm, okay.
dis  Sofia: Maybe we lower the speed limit in the code.
dis  Tim: I'd rather not, we need the speed for the ramp.
dis  Raj: What if we add a heat sink instead?
dec  Tim: Yeah, that's better.
dec  Sofia: Okay, heat sink it is.
act  Raj: Tim, you order the heat sink today.
act  Tim: Got it.
dec  Sofia: The organisers confirmed our slot for nine thirty.
dis  Raj: Good. Back to testing then.
"""

NEWSPAPER_1 = """
dis  Ava: Let's begin, we need to lock the next issue by Thursday.
dis  Ben: I have the interview with the dean, it's about eight hundred words.
dis  Ava: That's a bit long.
dis  Ben: I can cut it to six hundred.
dis  Cara: The sports page is empty, nobody covered the finals.
act  Ava: Cara, would you write a short piece on the finals by Wednesday?
act  Cara: Yes, I'll do it tonight.
dis  Ben: Should we keep the crossword?
dec  Ava: Let's keep it, people like it.
dis  Cara: The printer sent the new price list, it went up again.
dis  Ava: Okay, noted for now.
act  Ben: I'll compare it with the other printer and report back.
dis  Ava: Thanks, see you Thursday.
"""

RECYCLING_DRIVE_1 = """
dis  Ella: Okay, let's talk about the recycling drive.
dis  Finn: We collected forty bags of plastic last weekend.
dis  Ella: That's a lot more than I expected.
dis  Gita: The collection bins near the canteen are always full by noon.
dis  Finn: How about we put two more bins there?
dec  Ella: Sure.
dec  Gita: Same.
act  Ella: Finn, can you ask facilities for the extra bins?
act  Finn: No problem.
dis  Gita: Did the town council reply about the pickup?
dec  Ella: Yes, they agreed to collect every Thursday from next week.
dis  Gita: Nice.
dis  Finn: That's all from me.
"""

STUDY_GROUP_1 = """
dis  Ivy: Okay, let's plan the revision for the final.
dis  Jon: The exam covers chapters one to nine.
dis  Kim: Can you remind me which chapters were cancelled?
dis  Jon: Only chapter ten, the lecturer dropped it.
dis  Ivy: Should we split the chapters and each make a summary?
dec  Kim: Good idea.
dec  Jon: Yeah, okay.
act  Ivy: I'll take chapters one to three.
act  Kim: I can do four to six.
act  Jon: Then I'll do seven to nine.
dis  Kim: When should we share them?
dec  Ivy: Let's share them by Sunday night.
dis  Jon: Okay, see you then.
"""

DANCE_CLUB_1 = """
dis  Lily: Hi everyone, quick one about the showcase.
dis  Max: The studio is free on Saturday mornings only.
dis  Lily: Okay.
dis  Nia: We still don't have the music for the group piece.
act  Lily: Nia, please send me three options by tomorrow.
act  Nia: Will do.
dis  Max: The costume budget is around two hundred dollars.
dis  Nia: That's not much for twelve dancers.
dis  Lily: Maybe we should wear black and just buy matching scarves.
dec  Max: Works for me.
dec  Nia: Sounds good.
dec  Lily: The student union approved our showcase date this morning.
dis  Max: Finally. Okay, let's warm up.
"""

CAREER_FAIR_1 = """
dis  Omar: Okay, let's start, the career fair is in ten days.
dis  Pam: We have eighteen companies registered so far.
dis  Omar: Got it.
dis  Quinn: Three of them asked for a power socket at their booth.
act  Omar: Can you check the floor plan for sockets?
act  Quinn: Sure, I'll look at it after lunch.
dis  Pam: Should we give the companies lunch as well?
dis  Quinn: The budget is tight this year.
dis  Pam: Hmm, okay.
dec  Omar: Let's just give them coffee and snacks then.
dec  Pam: Fair enough.
dis  Quinn: I sent the reminder email to all students yesterday.
dis  Omar: Great, thanks. That's all.
"""

CHESS_CLUB_1 = """
dis  Ray: Alright, let's go through the tournament plans.
dis  Sara: We have twenty eight players signed up.
dis  Ray: Do we have enough boards?
dis  Sara: We have fifteen, so yes.
dis  Ray: Okay.
dis  Toby: Should we use the rapid format so it finishes on time?
dec  Sara: Yeah, let's do that.
dec  Ray: Agreed.
act  Sara: Toby, you prepare the pairings the night before.
act  Toby: Okay, will do.
dec  Sara: The library confirmed we can use the reading room all day.
dis  Toby: Nice. Anything else?
dis  Ray: No, that's it.
"""

WELCOME_NIGHT_1 = """
dis  Uma: Okay, welcome night planning.
dis  Vic: We expect about sixty new students.
dis  Wes: The hall holds eighty, so we're fine.
dis  Uma: Good.
dis  Vic: For food, should we order pizza or noodles?
dis  Wes: Noodles are cheaper and more people can eat them.
dec  Uma: Then we'll go with noodles.
dec  Vic: Okay.
act  Wes: I'll call the restaurant tomorrow to place the order.
dis  Vic: We still need someone to run the games.
act  Uma: Vic, can you plan two icebreaker games?
act  Vic: Sure.
dis  Uma: Okay, that's everything.
"""

FOOD_FAIR_1 = """
dis  Xena: Let's start. The food fair is next Saturday.
dis  Yusuf: Nine stalls have confirmed, we wanted twelve.
dis  Xena: Hmm, okay.
dis  Zoe: The health office asked for a list of all the stalls.
act  Xena: Zoe, could you send them the list by Wednesday?
act  Zoe: Okay, I'll do it tomorrow.
dis  Yusuf: The tents from last year are torn.
dis  Zoe: We could rent new ones, it's about a hundred dollars.
dec  Xena: Okay, let's rent them.
dec  Yusuf: Fine by me.
dec  Zoe: The campus police signed off on the road closure yesterday.
dis  Xena: Great news. See you all on Saturday.
"""

JOURNAL_CLUB_1 = """
dis  Adam: Okay, let's pick the paper for next week.
dis  Beth: I suggested the one on speech recognition in noisy rooms.
dis  Carl: I read it, it's quite long.
dis  Adam: How about we just cover the first half?
dec  Beth: Sure.
dec  Carl: That works for me.
act  Adam: Beth, you lead the discussion since you picked it.
act  Beth: Okay.
dis  Carl: I finished the slides for last week's paper, they're on the drive.
dis  Adam: Thanks Carl.
dis  Beth: Is the room still the same?
dis  Adam: Yes, same room, same time.
dis  Carl: Okay, see you then.
"""

MODEL_UN_1 = """
dis  Dina: Hi all, let's prepare for the conference.
dis  Eric: Our delegation got the climate committee.
dis  Dina: Okay.
dis  Fay: We need position papers by the fifteenth.
act  Dina: Eric, can you draft ours this weekend?
act  Eric: Yes, I'll start on Saturday.
dis  Fay: Should we practise speeches on Thursday?
dec  Dina: Good idea, let's practise on Thursday.
dec  Eric: Sounds good.
dis  Fay: The registration fee is forty dollars each.
dec  Dina: The faculty agreed to cover half of it.
dis  Eric: Oh, that helps.
dis  Dina: Okay, that's all for today.
"""

CAMPUS_RADIO_1 = """
dis  Gus: Okay, let's go, short meeting today.
dis  Hope: The new microphones came in on Tuesday.
dis  Gus: Nice.
dis  Iris: Our Friday show has almost no listeners.
dis  Hope: What if we move it to the lunchtime slot?
dis  Iris: I'm not sure, the lunch slot clashes with the news.
dis  Gus: Let's think about it and decide next week.
dis  Hope: Someone needs to update the show schedule online.
act  Iris: I can do it tonight.
act  Gus: Thanks. Also, the studio key is missing again. Hope, please ask security for a spare.
act  Hope: Okay.
dis  Gus: Alright, see you next week.
"""

GARDENING_CLUB_1 = """
dis  Jade: Morning everyone, let's check on the garden.
dis  Kurt: The tomatoes are doing well, the lettuce not so much.
dis  Lena: I think the lettuce needs more shade.
dis  Jade: Should we put up a net over that bed?
dec  Kurt: Yeah, okay.
dec  Lena: Agreed.
act  Jade: Kurt, could you buy a shade net this week?
act  Kurt: Sure.
dis  Lena: I watered everything on Sunday and Monday.
dis  Jade: Thank you.
dec  Kurt: The estate office said yes to the second garden plot.
dis  Lena: That's great news.
dis  Jade: Okay, let's get to work.
"""

ALUMNI_DINNER_1 = """
dis  Mark: Okay, let's go through the alumni dinner.
dis  Nina: Fifty two alumni have replied yes.
dis  Mark: Got it.
dis  Owen: The hotel wants the final numbers by Monday.
act  Mark: Nina, please send them the numbers on Monday morning.
act  Nina: Okay, will do.
dis  Owen: Should we invite the old club advisor as a guest of honour?
dec  Nina: Yes, let's invite her.
dec  Mark: Same.
act  Owen: I'll write to her tonight.
dis  Nina: We printed the name tags on Friday.
dis  Mark: Great, that's all from me.
"""

PEER_MENTORING_1 = """
dis  Pria: Let's start, we have the mentor matching to discuss.
dis  Ravi: Thirty mentors and forty five first years signed up.
dis  Pria: So some mentors will have two mentees.
dis  Sean: Should we ask for volunteers to take two?
dec  Pria: Yes, let's ask for volunteers first.
dec  Ravi: Okay.
act  Sean: I'll send the message to the mentors today.
dis  Ravi: The training session for mentors is fixed for the third.
dis  Pria: Do we have a room?
dec  Ravi: Yes, the counselling centre booked their seminar room for us.
dis  Sean: Good.
dis  Pria: Okay, thanks everyone.
"""

ESPORTS_CLUB_1 = """
dis  Troy: Okay team, quick meeting about the league.
dis  Una: We lost two of our three matches last week.
dis  Troy: Yeah, I saw.
dis  Vik: Our practice times keep changing, that's the main problem.
dis  Una: How about fixed practice every Tuesday and Thursday?
dec  Vik: Works for me.
act  Troy: Okay, Tuesday and Thursday it is. Una, can you post the schedule in the group?
act  Una: Yep.
dis  Vik: The lab computers were updated, the game runs faster now.
dis  Troy: Good to know. Alright, that's it.
"""

STARTUP_TEAM_1 = """
dis  Walt: Let's start, we have the pitch in nine days.
dis  Xia: I finished the first version of the deck yesterday.
dis  Walt: Thanks, I'll read it tonight.
dis  Yan: The judges want to see a working demo.
dis  Xia: The demo crashes when two people log in at once.
act  Walt: Yan, could you look at that bug this week?
act  Yan: Okay, I'll try tomorrow.
dis  Xia: Should we mention the pilot with the cafe?
dec  Walt: Yes, let's put it on the second slide.
dec  Yan: Agreed.
dec  Xia: The incubator accepted us for the spring programme.
dis  Walt: Wow, that's great. Okay, back to work.
"""

CHOIR_CONCERT_1 = """
dis  Zoey: Okay, let's plan the winter concert.
dis  Alan: We have eight songs ready.
dis  Zoey: Got it.
dis  Bree: The piano in the hall is out of tune.
act  Zoey: Bree, can you call the tuner this week?
act  Bree: Sure, I'll call on Monday.
dis  Alan: Do we sell tickets or keep it free?
dis  Bree: Last time it was free and the hall was full.
dec  Zoey: Then let's keep it free.
dec  Alan: Okay.
dec  Bree: The music department lent us the hall for the evening of the ninth.
dis  Zoey: Perfect. Let's start with the first song.
"""

BLOOD_DRIVE_1 = """
dis  Cleo: Hi all, the blood drive is in three weeks.
dis  Dan: The blood bank sent the forms this morning.
dis  Cleo: Okay.
dis  Emma: Last time a lot of people didn't show up.
dis  Dan: Maybe we should send a text reminder the day before.
dec  Cleo: Yes, let's do that.
dec  Emma: Good call.
act  Cleo: Emma, could you set up the reminders?
act  Emma: Okay, I'll set them up this weekend.
dec  Dan: The sports hall is booked for the whole day.
dis  Emma: That's a relief.
dis  Cleo: Okay, see you next week.
"""

EXCHANGE_INFO_1 = """
dis  Fred: Okay, let's plan the exchange info session.
dis  Gina: Four partner universities said they can join online.
dis  Fred: Nice.
dis  Hugo: Do we have a room with a good camera?
dis  Gina: The seminar room on level five has one.
act  Fred: Can you book it for the twentieth?
act  Gina: Okay.
dis  Hugo: Should we record the session for people who can't come?
dec  Fred: Yes, let's record it.
dec  Gina: Sure.
dis  Hugo: I made the poster last night, it's in the drive.
dis  Fred: Thanks Hugo. That's all.
"""

ASTRONOMY_CLUB_1 = """
dis  Iko: Okay, let's talk about the stargazing night.
dis  Jay: The forecast says clear skies on Friday.
dis  Iko: Okay.
dis  Kia: We only have two telescopes working.
dis  Jay: The third one needs a new lens, about eighty dollars.
dis  Iko: Should we buy the lens?
dec  Kia: Yeah, why not.
dec  Jay: Agreed.
act  Iko: Jay, you order the lens today.
act  Jay: Will do.
dec  Kia: Security agreed to keep the rooftop open until midnight.
dis  Iko: Great. See you Friday.
"""

YEARBOOK_1 = """
dis  Lou: Let's start, the yearbook goes to print in a month.
dis  Mia: I collected photos from eleven clubs so far.
dis  Lou: Good.
dis  Ned: Some clubs haven't sent anything yet.
act  Lou: Ned, please email the remaining clubs this week.
act  Ned: Okay.
dis  Mia: Should we add a page for the graduating staff?
dec  Lou: Yes, let's add one page.
dec  Ned: Fine by me.
act  Mia: I'll ask the office for the list of staff.
dis  Ned: I already fixed the typos in the first ten pages.
dis  Lou: Thanks Ned. Okay, that's it.
"""

# ---------------------------------------------------------------------------
# Added 13 Oct (dev only): 4 more dev meetings, so dev has enough turns to
# tell models apart (the 8 Oct retrain showed one 20-turn dev meeting
# saturates). Never trained on.
# ---------------------------------------------------------------------------

MUSIC_CLUB_1 = """
dis  Abel: Okay, let's get started on the open mic.
dis  Bina: Fourteen acts have signed up.
dis  Abel: Nice.
dis  Cruz: The sound system from the union is booked for the evening.
dec  Abel: Great, so the sound is sorted.
dis  Bina: The stage is quite small for the band acts.
dis  Cruz: Maybe the band acts should all go in the second half?
dec  Bina: Sounds good.
act  Abel: Agreed. Cruz, can you put together the running order?
act  Cruz: Sure.
dis  Bina: I printed the flyers yesterday.
dis  Abel: Thanks. Okay, that's all.
"""

DEBATE_TEAM_1 = """
dis  Dara: Let's begin, the regional debate is in two weeks.
dis  Emil: We got the motion on public transport.
dis  Dara: Okay.
dis  Faye: We need a third speaker, Leon is sick.
dis  Emil: Maybe we can ask Tasha, she debated last year.
act  Dara: Faye, could you message Tasha today?
act  Faye: Yes, I'll text her now.
dis  Emil: Should we practise against the second team on Saturday?
dec  Dara: Sure.
dec  Faye: Okay.
dec  Emil: The coach confirmed the bus for the competition.
dis  Dara: Good. See you Saturday.
"""

CODING_CLUB_1 = """
dis  Gabe: Okay, let's plan the beginner workshop.
dis  Hiro: Thirty students registered.
dis  Gabe: Got it.
dis  Ines: The lab only has twenty five computers.
dis  Hiro: Some people can bring their own laptops.
dis  Gabe: Should we ask everyone to bring a laptop just in case?
dec  Ines: Yeah, let's do that.
dec  Hiro: Same.
act  Gabe: Ines, please add that to the registration email.
act  Ines: Okay.
act  Hiro: I'll prepare the exercises for the first hour.
dis  Gabe: Thanks. That's everything.
"""

HOSTEL_FLOOR_1 = """
dis  Jana: Hi everyone, floor meeting, it won't take long.
dis  Kris: The kitchen is getting really messy.
dis  Liam: Yeah, someone left dishes for three days.
dis  Jana: What if we make a cleaning roster?
dec  Kris: Good idea.
dec  Liam: Okay.
act  Jana: Kris, can you draw up the roster?
act  Kris: Sure, I'll put it on the fridge.
dis  Liam: Also the fire drill is on Wednesday at ten.
dis  Jana: Okay.
dec  Kris: The warden agreed to fix the shower on our floor this week.
dis  Jana: Finally. Alright, that's it.
"""

# The 9-turn sample meeting from transcript_placeholder.py, relabelled with
# the locked rule (turn 6 "Yes, I'll email them today." is now act).
SAMPLE_MEETING = """
dis  Alex: Let's kick off. First item is the Q3 budget review.
dis  Priya: I've sent the numbers over, we're currently 8% under budget.
dis  Alex: Good, let's keep monitoring that. Next, the workshop venue.
dec  Priya: Venue is confirmed for the 14th, catering still pending.
act  Alex: Can you follow up with catering by Friday?
act  Priya: Yes, I'll email them today.
dis  Alex: Great, last item, feedback from the last session.
dis  Priya: Mostly positive, a few people wanted more hands-on time.
dec  Alex: Noted, let's add that to the next agenda. Meeting adjourned.
"""

MEETINGS = [
    {"name": "group_project_1", "split": "train", "turns": _parse(GROUP_PROJECT_1)},
    {"name": "group_project_2", "split": "train", "turns": _parse(GROUP_PROJECT_2)},
    {"name": "lab_standup_1",   "split": "train", "turns": _parse(LAB_STANDUP_1)},
    {"name": "lab_standup_2",   "split": "train", "turns": _parse(LAB_STANDUP_2)},
    {"name": "cca_committee_1", "split": "train", "turns": _parse(CCA_COMMITTEE_1)},
    {"name": "cca_committee_2", "split": "train", "turns": _parse(CCA_COMMITTEE_2)},
    {"name": "club_event_1",    "split": "train", "turns": _parse(CLUB_EVENT_1)},
    {"name": "orientation_1",   "split": "train", "turns": _parse(ORIENTATION_1)},
    # 13 Oct, batch 1 (training)
    {"name": "photo_club_1",      "split": "train", "turns": _parse(PHOTO_CLUB_1)},
    {"name": "drama_club_1",      "split": "train", "turns": _parse(DRAMA_CLUB_1)},
    {"name": "robotics_team_1",   "split": "train", "turns": _parse(ROBOTICS_TEAM_1)},
    {"name": "newspaper_1",       "split": "train", "turns": _parse(NEWSPAPER_1)},
    {"name": "recycling_drive_1", "split": "train", "turns": _parse(RECYCLING_DRIVE_1)},
    {"name": "study_group_1",     "split": "train", "turns": _parse(STUDY_GROUP_1)},
    {"name": "dance_club_1",      "split": "train", "turns": _parse(DANCE_CLUB_1)},
    {"name": "career_fair_1",     "split": "train", "turns": _parse(CAREER_FAIR_1)},
    {"name": "chess_club_1",      "split": "train", "turns": _parse(CHESS_CLUB_1)},
    {"name": "welcome_night_1",   "split": "train", "turns": _parse(WELCOME_NIGHT_1)},
    {"name": "food_fair_1",       "split": "train", "turns": _parse(FOOD_FAIR_1)},
    {"name": "journal_club_1",    "split": "train", "turns": _parse(JOURNAL_CLUB_1)},
    # 13 Oct, batch 2 (training)
    {"name": "model_un_1",        "split": "train", "turns": _parse(MODEL_UN_1)},
    {"name": "campus_radio_1",    "split": "train", "turns": _parse(CAMPUS_RADIO_1)},
    {"name": "gardening_club_1",  "split": "train", "turns": _parse(GARDENING_CLUB_1)},
    {"name": "alumni_dinner_1",   "split": "train", "turns": _parse(ALUMNI_DINNER_1)},
    {"name": "peer_mentoring_1",  "split": "train", "turns": _parse(PEER_MENTORING_1)},
    {"name": "esports_club_1",    "split": "train", "turns": _parse(ESPORTS_CLUB_1)},
    {"name": "startup_team_1",    "split": "train", "turns": _parse(STARTUP_TEAM_1)},
    {"name": "choir_concert_1",   "split": "train", "turns": _parse(CHOIR_CONCERT_1)},
    {"name": "blood_drive_1",     "split": "train", "turns": _parse(BLOOD_DRIVE_1)},
    {"name": "exchange_info_1",   "split": "train", "turns": _parse(EXCHANGE_INFO_1)},
    {"name": "astronomy_club_1",  "split": "train", "turns": _parse(ASTRONOMY_CLUB_1)},
    {"name": "yearbook_1",        "split": "train", "turns": _parse(YEARBOOK_1)},
    {"name": "club_event_2",    "split": "dev",   "turns": _parse(CLUB_EVENT_2)},
    # 13 Oct (dev only)
    {"name": "music_club_1",    "split": "dev",   "turns": _parse(MUSIC_CLUB_1)},
    {"name": "debate_team_1",   "split": "dev",   "turns": _parse(DEBATE_TEAM_1)},
    {"name": "coding_club_1",   "split": "dev",   "turns": _parse(CODING_CLUB_1)},
    {"name": "hostel_floor_1",  "split": "dev",   "turns": _parse(HOSTEL_FLOOR_1)},
    {"name": "group_project_3", "split": "test",  "turns": _parse(GROUP_PROJECT_3)},
    {"name": "hackathon_1",     "split": "test",  "turns": _parse(HACKATHON_1)},
    {"name": "sample_meeting",  "split": "test",  "turns": _parse(SAMPLE_MEETING)},
]

# ---------------------------------------------------------------------------
# Cross-turn probe set: the reply (last turn) is scored, nothing else.
# Same kinds of bare reply under all three labels, so a model that just
# memorises "Sure. = action_item" can't score well.
# ---------------------------------------------------------------------------

PROBES = [_parse(block) for block in [
    # after a request -> act
    "act  Mei: Could you send the slides to everyone tonight?\nact  Arjun: Sure.",
    "act  Ravi: Can you check the projector before the talk?\nact  Ben: Okay.",
    "act  Nadia: Please print twenty copies of the agenda.\nact  Kai: Got it.",
    "act  Hui: Siti, you take care of the name tags.\nact  Siti: No problem.",
    "act  Daniel: Could someone book the meeting room for Monday?\nact  Farah: I can.",
    # after a proposal -> dec
    "dis  Marcus: How about we move the social to Saturday?\ndec  Priya: Sure.",
    "dis  Alex: Shall we go with the green design?\ndec  Wei: Yeah, let's do that.",
    "dis  Siti: I suggest we cap registration at a hundred.\ndec  Ben: Sounds good.",
    "dec  Arjun: Let's push the internal deadline to Friday.\ndec  Mei: Fine by me.",
    # after plain information -> dis
    "dis  Kai: The bus was late again this morning.\ndis  Hui: Okay.",
    "dis  Farah: We had about fifty people at the last session.\ndis  Ravi: Sounds good.",
    "dis  Ben: The venue has a projector and two microphones.\ndis  Siti: Got it.",
]]
# Only the reply is scored; the first turn's label is there for completeness.


# ---------------------------------------------------------------------------
# Data augmentation: short practice meetings built from phrase lists
# ---------------------------------------------------------------------------
# 146 hand-written training turns is too few: the model memorises them in
# ~10 epochs. generate_meetings() adds short meetings stitched together from
# the lists below, mostly to show the model many more examples of the
# cross-turn pattern (same reply, different label depending on the turn
# before). Used for TRAINING ONLY -- dev, test and the probes stay
# hand-written, and none of the probe sentences appear in these lists.

NAMES = ["Wei", "Farah", "Arjun", "Mei", "Ravi", "Siti", "Ben", "Daniel", "Nadia",
         "Kai", "Hui", "Marcus", "Priya", "Alex", "Jun", "Aisha", "Tom", "Lina"]

TASKS = ["reserve the seminar room", "share the minutes in the group chat", "update the budget sheet",
         "email the sponsor", "fix the broken link", "print the name cards",
         "collect the feedback forms", "draft the poster", "order the snacks",
         "renew the software licence", "set up the registration form",
         "test the demo on the lab laptop", "upload the recording to the drive",
         "call the bus company", "reply to the dean's office", "clean up the shared folder",
         "write the introduction", "review the pull request", "ask the lecturer about the rubric",
         "buy the extension cables", "update the website", "contact the alumni",
         "prepare the quiz questions", "check the room booking system"]
DEADLINES = ["", "", " by Friday", " today", " before the next meeting", " by tomorrow",
             " this week", " by Monday", " tonight", " by the end of the week"]

REQUESTS = ["Can you {task}{deadline}?", "Could you {task}{deadline}?",
            "{name}, can you {task}{deadline}?", "{name}, could you {task}{deadline}?",
            "Please {task}{deadline}.", "{name}, please {task}{deadline}.",
            "{name}, you {task}{deadline}.", "Someone needs to {task}{deadline}.",
            "Who can {task}{deadline}?", "{name}, would you be able to {task}{deadline}?"]
ACCEPTS = ["Sure.", "Okay.", "Will do.", "Got it.", "No problem.", "Yep.", "Okay, will do.",
           "Sure, I'll do it later.", "Yes, I'll handle it.", "I can do it.", "On it.",
           "Alright.", "Sure thing.", "Can do.", "Leave it with me.", "Yeah, okay."]

PROPOSALS = ["hold the meeting in the library", "use the bigger room", "limit each team to four people",
             "drop the second session", "reuse the old banner", "split the work by chapter",
             "start at nine instead of ten", "order pizza instead of bento", "keep the old logo",
             "invite the alumni as well", "run it online this time", "skip the dry run",
             "share the slides after the talk", "combine the two events", "meet every two weeks",
             "give out vouchers instead of medals"]
PROPOSE = ["How about we {p}?", "Shall we {p}?", "I suggest we {p}.", "Maybe we should {p}.",
           "Should we {p}?", "What if we {p}?", "I think we should {p}.",
           "Why don't we {p}?", "Can we {p}?"]
AGREES = ["Sure.", "Okay.", "Agreed.", "Sounds good.", "Yeah, let's do that.", "Fine by me.",
          "Makes sense.", "Works for me.", "Good idea.", "Yes, let's go with that.",
          "That works.", "Good call.", "Why not.", "Okay, let's do it.", "Yeah, okay."]
CHAIR_DECIDES = ["Okay, let's {p} then.", "Alright, we'll {p}.", "So we {p}, that's settled.",
                 "Okay, final answer, we {p}."]

INFO = ["The room is free all afternoon.", "We have about {n} sign ups so far.",
        "The printer on level three is working again.", "Last year's event had {n} people.",
        "The lecturer moved the deadline last week.", "Our budget is around {n} dollars.",
        "The venue closes at ten.", "Most people filled in the survey on their phones.",
        "Two people dropped out yesterday.", "The new laptops arrived this morning.",
        "The library is closed on Sunday.", "Our post got {n} likes.",
        "The weather looks fine for Saturday.", "The sponsor replied but hasn't given an answer yet.",
        "The canteen near the hall is under renovation.", "I tried the new app, it's a bit slow."]
ACKS = ["Okay.", "Got it.", "Sounds good.", "Nice.", "Good to know.", "Ah, okay.", "Alright.",
        "Thanks.", "Oh, that's good.", "Hmm, okay.", "Cool.", "Sure.", "Yeah, okay."]

SELF_ASSIGN = ["I'll {task}{deadline}.", "I can {task}{deadline}.", "Let me {task}{deadline}.",
               "I'm going to {task}{deadline}.", "Okay, I'll {task}{deadline}."]
THIRD_PERSON = ["{name} will {task}{deadline}.", "{name} is going to {task}{deadline}."]
REPORTED_FINAL = ["The hall booking went through this morning.", "Our budget got approved yesterday.",
                  "The permit came through on Monday.", "The date is fixed now, it's the second Saturday.",
                  "The sponsorship was signed off last night.", "The bus company confirmed the pickup time.",
                  "The dean approved our proposal.", "The guest speaker confirmed she's coming."]
OBJECTIONS = ["Hmm, I'm not sure about that.", "I'd rather not, it's a bit late to change.",
              "Can we think about it first?", "Maybe, but let's check the budget before deciding."]
DECLINES = ["Sorry, I can't this week.", "I'm a bit swamped, maybe someone else?",
            "Not sure I can, I'm away on Friday."]

# Added 8 Oct: the model's weak spot was decisions reported as facts. These
# templates show settled plans (decision) next to finished tasks and
# still-pending items (discussion), so the model learns the difference
# rather than "past tense = decision".
SETTLED = ["{thing} is confirmed for {when}.", "{thing} is booked for {when}.",
           "{thing} got approved this morning.",
           "{org} approved {thing_l}.", "{org} confirmed {thing_l} for {when}.",
           "Good news, {thing_l} is locked in for {when}.", "{thing} was signed off yesterday.",
           "{thing} is finalised now, {pending}.", "{org} said yes to {thing_l}."]
THINGS = ["The hall booking", "Our budget", "The second session", "The room change", "The guest list",
          "The menu", "The poster design", "The sponsorship", "The extra tables", "The prize money",
          "The trip date", "The bus booking"]
ORGS = ["The student union", "The dean's office", "The library", "The sports centre", "The AV team",
        "The caterer", "Facilities"]
WHENS = ["Saturday", "the 21st", "next Tuesday", "week twelve", "the afternoon slot", "the 3rd"]
PENDINGS = ["but we're waiting on the speakers", "though the menu is open",
            "but the poster needs another pass"]
STATUS_DONE = ["I sent the invoice to the treasurer yesterday.", "We printed the name cards this morning.",
               "I've uploaded the photos to the drive.", "We bought the extension cables last week.",
               "I finished the slides for the talk.", "The posters went up on Monday.",
               "I emailed the caterer to confirm the numbers.", "I submitted the claim form on Tuesday.",
               "We moved the chairs back after the session.", "I replied to the dean's office already."]
NOT_YET = ["The dean hasn't approved it yet.", "We're still waiting for the booking to come through.",
           "Nothing is confirmed yet for the trip.", "The sponsor still hasn't signed anything.",
           "The room might change, it isn't fixed yet."]
NOTED = ["Noted, we'll bring it up again next week.", "Noted, we'll park that for now.",
         "Okay, noted, we'll raise it with the dean.", "Noted, we'll keep the old format then."]

QUESTIONS = [("Did anyone hear back from the sponsor?", "Not yet."),
             ("How many people are coming?", "Around {n}."),
             ("When is the report due?", "Next Friday, I think."),
             ("Where are we meeting next time?", "Same room as today."),
             ("Is the projector in that room working?", "It was last week."),
             ("Was the budget approved in the end?", "Not yet, still waiting."),
             ("Is the venue confirmed yet?", "No, they haven't replied.")]
TRAPS = ["Can you believe it's already week ten?", "I won't follow up on that, it's sorted.",
         "We haven't decided on the theme yet.", "Please, go ahead.",
         "Could you repeat that? I missed it.", "I'm not sure that was agreed last time.",
         "Honestly, can you imagine doing this every week?"]
OPENERS = ["Okay, let's start.", "Hi everyone, let's begin.", "Alright, shall we get going?",
           "Thanks for joining, let's start."]
CLOSERS = ["That's all for today, thanks.", "Okay, see you next week.", "Alright, thanks everyone.",
           "Great, let's end here."]


def generate_meetings(n_meetings: int, seed: int = 0) -> list[dict]:
    """Short practice meetings (6-14 turns) built from the phrase lists."""
    import random
    rng = random.Random(seed)

    def fill(template: str, name: str) -> str:
        thing = rng.choice(THINGS)
        return template.format(task=rng.choice(TASKS), deadline=rng.choice(DEADLINES),
                               name=name, p=rng.choice(PROPOSALS), n=rng.choice([12, 30, 45, 80, 120]),
                               thing=thing, thing_l=thing[0].lower() + thing[1:], org=rng.choice(ORGS),
                               when=rng.choice(WHENS), pending=rng.choice(PENDINGS))

    meetings = []
    for m in range(n_meetings):
        people = rng.sample(NAMES, rng.choice([3, 4]))
        chair = people[0]
        turns: list[tuple[str, str, str]] = []

        def say(speaker, text, label):
            if turns and turns[-1][0] == speaker:          # keep speakers alternating
                speaker = rng.choice([p for p in people if p != speaker])
            turns.append((speaker, text, label))

        def someone_else(*avoid):
            return rng.choice([p for p in people if p not in avoid])

        say(chair, rng.choice(OPENERS), "discussion")
        for _ in range(rng.randint(3, 6)):
            kind = rng.choices(["request", "propose", "info", "self", "third", "final",
                                "question", "trap", "chair", "objection", "decline",
                                "settled", "status", "notyet", "noted"],
                               weights=[4, 3, 4, 1.5, 1, 1, 1.5, 1.5, 1, 1, 0.7,
                                        2, 2, 1, 0.8])[0]
            asker = someone_else(turns[-1][0])
            if kind == "request":
                target = someone_else(asker)
                say(asker, fill(rng.choice(REQUESTS), target), "action_item")
                say(target, rng.choice(ACCEPTS), "action_item")
            elif kind == "propose":
                say(asker, fill(rng.choice(PROPOSE), ""), "discussion")
                say(someone_else(asker), rng.choice(AGREES), "decision")
            elif kind == "info":
                say(asker, fill(rng.choice(INFO), ""), "discussion")
                say(someone_else(asker), rng.choice(ACKS), "discussion")
            elif kind == "self":
                say(asker, fill(rng.choice(SELF_ASSIGN), ""), "action_item")
            elif kind == "third":
                say(asker, fill(rng.choice(THIRD_PERSON), someone_else(asker)), "action_item")
            elif kind == "final":
                say(asker, fill(rng.choice(REPORTED_FINAL), ""), "decision")
            elif kind == "question":
                q, a = rng.choice(QUESTIONS)
                say(asker, q, "discussion")
                say(someone_else(asker), fill(a, ""), "discussion")
            elif kind == "trap":
                say(asker, rng.choice(TRAPS), "discussion")
            elif kind == "objection":            # proposal not agreed: both stay discussion
                say(asker, fill(rng.choice(PROPOSE), ""), "discussion")
                say(someone_else(asker), rng.choice(OBJECTIONS), "discussion")
            elif kind == "decline":              # request still asked (act), reply declines (dis)
                target = someone_else(asker)
                say(asker, fill(rng.choice(REQUESTS), target), "action_item")
                say(target, rng.choice(DECLINES), "discussion")
            elif kind == "settled":              # a plan or approval reported as settled
                say(asker, fill(rng.choice(SETTLED), ""), "decision")
                if rng.random() < 0.5:
                    say(someone_else(asker), rng.choice(ACKS), "discussion")
            elif kind == "status":               # a finished task: a status update
                say(asker, rng.choice(STATUS_DONE), "discussion")
                if rng.random() < 0.5:
                    say(someone_else(asker), rng.choice(ACKS), "discussion")
            elif kind == "notyet":               # not settled yet
                say(asker, rng.choice(NOT_YET), "discussion")
            elif kind == "noted":                # closing a point by settling what happens next
                say(asker, fill(rng.choice(INFO), ""), "discussion")
                say(chair, rng.choice(NOTED), "decision")
            else:  # the chair settles a proposal without asking
                say(chair, fill(rng.choice(CHAIR_DECIDES), ""), "decision")
        say(chair, rng.choice(CLOSERS), "discussion")
        meetings.append({"name": f"generated_{m}", "split": "train", "turns": turns})
    return meetings


GENERATED = generate_meetings(60)


if __name__ == "__main__":
    from collections import Counter
    for split in ("train", "dev", "test"):
        ms = [m for m in MEETINGS if m["split"] == split]
        counts = Counter(label for m in ms for _s, _t, label in m["turns"])
        n = sum(counts.values())
        print(f"{split:5s} {len(ms)} meetings, {n:3d} turns  "
              + "  ".join(f"{k} {v} ({v / n:.0%})" for k, v in sorted(counts.items())))
    counts = Counter(label for m in GENERATED for _s, _t, label in m["turns"])
    n = sum(counts.values())
    print(f"generated {len(GENERATED)} meetings, {n} turns  "
          + "  ".join(f"{k} {v} ({v / n:.0%})" for k, v in sorted(counts.items())))
    print(f"probes {len(PROBES)} exchanges (reply scored)")

    # Leak check: no probe or test sentence may share a run of 5 words with
    # the training/dev meetings or with anything the generator can produce
    # (2,000 sample meetings cover its phrase lists). Short replies like
    # "Sure." are meant to repeat, and are too short to have 5 words anyway.
    import re

    def five_grams(text):
        words = re.findall(r"[a-z0-9']+", text.lower())
        return {tuple(words[k:k + 5]) for k in range(len(words) - 4)}

    seen = set()
    for m in [m for m in MEETINGS if m["split"] != "test"] + generate_meetings(2000, seed=99):
        for _s, text, _l in m["turns"]:
            seen |= five_grams(text)
    from scorecard_data import SCORECARD     # 13 Oct: the frozen scorecard is held out too
    held_out = [t for p in PROBES for _s, t, _l in p]
    held_out += [t for m in MEETINGS if m["split"] == "test" for _s, t, _l in m["turns"]]
    held_out += [t for m in SCORECARD for _s, t, _l in m["turns"]]
    leaks = [(t, sorted(five_grams(t) & seen)) for t in held_out if five_grams(t) & seen]
    for text, grams in leaks:
        print("  LEAK:", text, "<-", [" ".join(g) for g in grams])
    assert not leaks, "probe/test/scorecard sentences overlap the training data"
    print("no probe, test or scorecard sentence shares 5 words in a row with training data: OK")

    # Dev picks the epoch and seed (and judges retrains), so it must not
    # overlap the training-only data either.
    train_only = set()
    for m in [m for m in MEETINGS if m["split"] == "train"] + generate_meetings(2000, seed=99):
        for _s, text, _l in m["turns"]:
            train_only |= five_grams(text)
    dev_leaks = [t for m in MEETINGS if m["split"] == "dev" for _s, t, _l in m["turns"]
                 if five_grams(t) & train_only]
    for text in dev_leaks:
        print("  DEV LEAK:", text)
    assert not dev_leaks, "dev sentences overlap the training data"
    print("no dev sentence shares 5 words in a row with training data: OK\n")
    for speaker, text, label in GENERATED[0]["turns"]:
        print(f"  {label:12s} {speaker}: {text}")
