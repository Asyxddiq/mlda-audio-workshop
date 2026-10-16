# Testing script: meeting tests

Things people say in meetings, and the minutes the prototype should write for each.
Words only (typed transcripts); `audio_check.py` is the audio test.

Run them all with one command:

```
python prototype/minutes_tests.py        # trained model
python prototype/minutes_tests.py -v     # show the minutes for failing cases
python prototype/minutes_tests.py A07    # one case
python prototype/minutes_tests.py --keywords   # keyword fallback instead
```

A case passes when the minutes have exactly these decisions and action items, each
action with the right owner and due date. The cases live in `minutes_tests.py`;
this file is written from it (`python prototype/minutes_tests.py --markdown`).
They are for testing only, never for training or tuning.

## A: Action items

### A01: request + yes

```
Mei: Ben, can you book the seminar room for Thursday?
Ben: Sure, I'll do it now.
```

Expected minutes:

- Action "book the seminar room" -- owner **Ben**, no due date

### A02: request + bare 'Okay'

```
Mei: Could you send the slides to the group tonight?
Ben: Okay.
```

Expected minutes:

- Action "send the slides" -- owner **Ben**, due **tonight**

### A03: offer with a due date

```
Ben: I'll write to the sponsor by Friday.
```

Expected minutes:

- Action "write to the sponsor" -- owner **Ben**, due **Friday**

### A04: assigned to someone not in the meeting

```
Mei: Kai will handle the registration desk.
```

Expected minutes:

- Action "registration desk" -- owner **Kai**, no due date

### A05: conditional offer

```
Ben: I can redo the poster tonight if nobody minds.
```

Expected minutes:

- Action "redo the poster" -- owner **Ben**, due **tonight**

### A06: request turned down

```
Mei: Can you get the name tags printed tomorrow?
Ben: Sorry, I can't, I'm away tomorrow.
```

Expected minutes:

- Action "name tags" -- owner **unassigned**, due **tomorrow**

### A07: 'Will do.' + 'before the next meeting'

```
Hana: Please update the member list before the next meeting.
Omar: Will do.
```

Expected minutes:

- Action "update the member list" -- owner **Omar**, due **the next meeting**

### A08: 'Who can ...?' + 'I can.'

```
Hana: Who can collect the sign-in sheets?
Omar: I can.
```

Expected minutes:

- Action "collect the sign-in sheets" -- owner **Omar**, no due date

### A09: 'Let me ...'

```
Omar: Let me look at the booking calendar after this.
```

Expected minutes:

- Action "booking calendar" -- owner **Omar**, no due date

### A10: 'Someone has to ...' + 'Leave it with me.'

```
Hana: Someone has to ring the coach company.
Omar: Leave it with me.
```

Expected minutes:

- Action "ring the coach company" -- owner **Omar**, no due date

### A11: direct order, no 'please'

```
Lina: Arjun, you take the opening speech.
Arjun: Okay.
```

Expected minutes:

- Action "opening speech" -- owner **Arjun**, no due date

### A12: two tasks to two people

```
Lina: Dev, could you buy the decorations by Friday?
Dev: Yes.
Lina: And Sara, please sort out the music.
Sara: No problem.
```

Expected minutes:

- Action "buy the decorations" -- owner **Dev**, due **Friday**
- Action "sort out the music" -- owner **Sara**, no due date

### A13: due date given in the reply

```
Lina: Can you pull together a feedback summary?
Dev: Yes, I'll have it ready by Monday.
```

Expected minutes:

- Action "feedback summary" -- owner **Dev**, due **Monday**

### A14: 'X is going to ...'

```
Sara: Nadia is going to prepare a welcome note.
```

Expected minutes:

- Action "welcome note" -- owner **Nadia**, no due date

### A15: 'Could someone ...?' + 'I'll do it.'

```
Lina: Could someone reserve the library discussion room?
Dev: I'll do it.
```

Expected minutes:

- Action "reserve the library discussion room" -- owner **Dev**, no due date

### A16: 'I'm going to ...' + tonight

```
Dev: I'm going to post the recording online tonight.
```

Expected minutes:

- Action "post the recording" -- owner **Dev**, due **tonight**

### A17: 'are you able to' + this week

```
Hana: Omar, are you able to email the caterer this week?
Omar: Yeah, no problem.
```

Expected minutes:

- Action "email the caterer" -- owner **Omar**, due **this week**

### A18: 'can't today, but I'll ...' is still a yes

```
Hana: Can you test the projector before Friday?
Omar: I can't today, but I'll do it tomorrow.
```

Expected minutes:

- Action "test the projector" -- owner **Omar**, due **Friday**

### A19: turned down, then someone else takes it

```
Hana: Can you print the handouts?
Omar: Sorry, I'm swamped.
Lina: I'll print them.
```

Expected minutes:

- Action "print" -- owner **Lina**, no due date

### A20: 'chase' + 'Sure.'

```
Mei: Ben, can you chase the venue about parking?
Ben: Sure.
```

Expected minutes:

- Action "chase the venue" -- owner **Ben**, no due date

## B: Decisions

### B01: proposal + 'Sounds good.'

```
Mei: How about we move the meeting to Friday?
Ben: Sounds good.
```

Expected minutes:

- Decision mentioning "move the meeting to Friday"

### B02: chair decides after a comparison

```
Ben: The big hall costs more but fits everyone.
Mei: Okay, let's go with the big hall then.
```

Expected minutes:

- Decision mentioning "big hall"

### B03: decision stated as a fact

```
Ben: Venue's confirmed for the 14th.
```

Expected minutes:

- Decision mentioning "confirmed for the 14th"

### B04: approval reported

```
Mei: The dean approved our budget this morning.
```

Expected minutes:

- Decision mentioning "approved our budget"

### B05: proposal + objection = no decision

```
Mei: Should we cancel the dry run?
Ben: I'd rather not, we still need the practice.
```

Expected minutes:

- Nothing (no decisions, no action items)

### B06: proposal + 'Yeah, let's do that.'

```
Aisha: What if the banner uses the green design?
Tom: Yeah, let's do that.
```

Expected minutes:

- Decision mentioning "green design"

### B07: two people agree = still one decision

```
Aisha: I suggest we cap sign-ups at forty.
Tom: Agreed.
Lina: Fine by me.
```

Expected minutes:

- Decision mentioning "cap sign-ups at forty"

### B08: chair's 'let's' on its own

```
Aisha: Let's keep the club meeting on Thursdays.
```

Expected minutes:

- Decision mentioning "Thursdays"

### B09: 'we're going with'

```
Tom: Okay, final answer, we're going with pizza.
```

Expected minutes:

- Decision mentioning "pizza"

### B10: 'that's settled'

```
Aisha: So that's settled, the trip is on the 21st.
```

Expected minutes:

- Decision mentioning "trip is on the 21st"

### B11: proposal, objection, chair settles

```
Tom: Should we hire a photographer?
Lina: It's too expensive.
Aisha: Okay, no photographer this year.
```

Expected minutes:

- Decision mentioning "no photographer"

### B12: booking came through

```
Tom: The booking for the sports hall came through yesterday.
```

Expected minutes:

- Decision mentioning "sports hall"

### B13: agreement + echo ('Same here.')

```
Aisha: Shall we kick off at nine?
Tom: Works for me.
Lina: Same here.
```

Expected minutes:

- Decision mentioning "kick off at nine"

### B14: 'We'll hold ... in ...'

```
Aisha: We'll hold the bake sale in the canteen.
```

Expected minutes:

- Decision mentioning "bake sale in the canteen"

### B15: proposal ignored, topic changes

```
Tom: Maybe we should change the logo?
Lina: Anyway, about the budget.
```

Expected minutes:

- Nothing (no decisions, no action items)

### B16: negative decision

```
Aisha: Right, we're not doing the raffle.
```

Expected minutes:

- Decision mentioning "not doing the raffle"

### B17: show of hands

```
Aisha: Hands up for Saturday. Okay, that's most of you, Saturday it is.
```

Expected minutes:

- Decision mentioning "Saturday"

### B18: 'Can we ...?' is a proposal, not a task

```
Tom: Can we push the deadline to Monday?
Aisha: Yes, let's do Monday.
```

Expected minutes:

- Decision mentioning "Monday"

## C: Discussion only (nothing in the minutes)

### C01: information + 'Okay.'

```
Ben: The level three printer is broken again.
Mei: Okay.
```

Expected minutes:

- Nothing (no decisions, no action items)

### C02: status update on a finished task

```
Ben: I sent the treasurer the invoice yesterday.
```

Expected minutes:

- Nothing (no decisions, no action items)

### C03: a question for right now

```
Ben: Can you tell me where the sign-up form went?
Mei: It's on the module page.
```

Expected minutes:

- Nothing (no decisions, no action items)

### C04: 'follow up' in a negative sentence

```
Ben: I'm not going to follow up on it, it's sorted already.
```

Expected minutes:

- Nothing (no decisions, no action items)

### C05: procedural 'let's'

```
Mei: Okay, let's start. First item is the budget.
```

Expected minutes:

- Nothing (no decisions, no action items)

### C06: 'decided' in 'haven't decided'

```
Tom: We still haven't decided the theme.
```

Expected minutes:

- Nothing (no decisions, no action items)

### C07: 'confirmed' in 'hasn't confirmed'

```
Lina: The sponsor still hasn't confirmed anything.
```

Expected minutes:

- Nothing (no decisions, no action items)

### C08: 'Could you repeat that?'

```
Tom: Could you repeat that?
Lina: Sure, I said we need more chairs.
```

Expected minutes:

- Nothing (no decisions, no action items)

### C09: 'Please, go ahead.'

```
Tom: Can I add something?
Aisha: Please, go ahead.
```

Expected minutes:

- Nothing (no decisions, no action items)

### C10: 'confirmed now?' + 'Not yet.'

```
Tom: Is the venue confirmed now?
Aisha: Not yet.
```

Expected minutes:

- Nothing (no decisions, no action items)

### C11: an opinion

```
Lina: I think the old logo looks nicer.
```

Expected minutes:

- Nothing (no decisions, no action items)

### C12: closing

```
Aisha: Thanks everyone, see you next week.
```

Expected minutes:

- Nothing (no decisions, no action items)

### C13: a 'what if' wish

```
Tom: If we had more money we could hire a band.
```

Expected minutes:

- Nothing (no decisions, no action items)

### C14: a decision from last year

```
Ben: Last year we agreed on a cap and it didn't really work.
```

Expected minutes:

- Nothing (no decisions, no action items)

### C15: 'I'll' in a figure of speech

```
Lina: I'll believe it when I see it.
```

Expected minutes:

- Nothing (no decisions, no action items)

### C16: question about a done task

```
Tom: Did you manage to email the sponsor?
Lina: Yes, I emailed them on Monday.
```

Expected minutes:

- Nothing (no decisions, no action items)

## D: Same reply, different meaning

### D01: 'Sure.' after a request = action

```
Aisha: Tom, can you collect the banner from the print shop?
Tom: Sure.
```

Expected minutes:

- Action "collect the banner" -- owner **Tom**, no due date

### D02: 'Sure.' after a proposal = decision

```
Aisha: Shall we meet at the cafe instead?
Tom: Sure.
```

Expected minutes:

- Decision mentioning "meet at the cafe"

### D03: 'Okay.' after information = nothing

```
Aisha: The lift is out of order today.
Tom: Okay.
```

Expected minutes:

- Nothing (no decisions, no action items)

### D04: 'Got it.' after a request = action

```
Aisha: Lina, please send the photos to the committee.
Lina: Got it.
```

Expected minutes:

- Action "send the photos" -- owner **Lina**, no due date

### D05: 'Got it.' after information = nothing

```
Aisha: The bus leaves at seven sharp.
Lina: Got it.
```

Expected minutes:

- Nothing (no decisions, no action items)

### D06: 'Sounds good.' after a proposal = decision

```
Tom: Why don't we split into two teams?
Aisha: Sounds good.
```

Expected minutes:

- Decision mentioning "split into two teams"

### D07: 'Sounds good.' after information = nothing

```
Tom: The caterer can do vegetarian options too.
Aisha: Sounds good.
```

Expected minutes:

- Nothing (no decisions, no action items)

### D08: 'Okay.' after a request = action

```
Aisha: Tom, could you check the microphones before Saturday?
Tom: Okay.
```

Expected minutes:

- Action "check the microphones" -- owner **Tom**, due **Saturday**

### D09: 'Okay.' after a proposal = decision

```
Tom: How about we finish at five?
Aisha: Okay.
```

Expected minutes:

- Decision mentioning "finish at five"

## E: Whole meetings

### E01: the sample meeting (transcript_placeholder.py)

```
Alex: Let's kick off. First item is the Q3 budget review.
Priya: I've sent the numbers over, we're currently 8% under budget.
Alex: Good, let's keep monitoring that. Next, the workshop venue.
Priya: Venue is confirmed for the 14th, catering still pending.
Alex: Can you follow up with catering by Friday?
Priya: Yes, I'll email them today.
Alex: Great, last item, feedback from the last session.
Priya: Mostly positive, a few people wanted more hands-on time.
Alex: Noted, let's add that to the next agenda. Meeting adjourned.
```

Expected minutes:

- Decision mentioning "Venue is confirmed"
- Decision mentioning "next agenda"
- Action "follow up with catering" -- owner **Priya**, due **Friday**

### E02: club welcome party

```
Aisha: Okay everyone, welcome party planning first.
Tom: We've booked the function room for the 3rd.
Aisha: Great. What should we serve?
Lina: How about noodles? They're cheap and everyone likes them.
Tom: Fine with me.
Aisha: Lina, can you place the food order by Wednesday?
Lina: Sure.
Tom: I'll make the poster tonight.
Aisha: Perfect, thanks all.
```

Expected minutes:

- Decision mentioning "booked the function room"
- Decision mentioning "noodles"
- Action "food order" -- owner **Lina**, due **Wednesday**
- Action "make the poster" -- owner **Tom**, due **tonight**

### E03: project demo prep

```
Farah: Quick check on the demo. It crashed twice yesterday.
Arjun: I think it's the login page.
Farah: Can you fix it before Friday?
Arjun: Yes, I'll fix it.
Farah: Also, please send me the error logs today.
Arjun: Okay.
Farah: And we'll present in person, not online.
Arjun: Agreed.
```

Expected minutes:

- Decision mentioning "present in person"
- Action "fix it" -- owner **Arjun**, due **Friday**
- Action "send me the error logs" -- owner **Arjun**, due **today**

### E04: budget meeting with a no

```
Hana: First item, the budget. We have about two hundred left.
Omar: Should we spend it on new jerseys?
Dev: I'd rather keep some for the finals trip.
Hana: Okay, we'll keep half for the trip.
Omar: Makes sense.
Hana: Dev, could you get quotes for the jerseys by next week?
Dev: Sorry, I can't, exams.
Omar: I can do it.
Hana: Thanks Omar. That's all for today.
```

Expected minutes:

- Decision mentioning "keep half for the trip"
- Action "quotes for the jerseys" -- owner **Omar**, due **next week**

