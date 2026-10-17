# Model update, 16 Oct 2026

Asy asked why the model labelled decisions badly in the tests and asked for two fixes: (1) a text clean-up in
front of the model, (2) more training examples for decisions. Both were built and tested against a rule
written before any training ([`ship_rule.md`](ship_rule.md)).

## Result in three lines

- **Shipped: the text clean-up (1), applied to the existing 13 Oct weights.** Nothing that was measured before
  got worse, and long-form wording ("I will", "let us", "do not", "OK") is now read like the short form.
- **Not shipped: the retrained models (2).** Two retrains with new decision data both failed the rule. The
  second one looks better on the 67 meeting tests but is worse on the team's test meetings, and its demo
  minutes list a wrong decision, so the 13 Oct weights stay.
- **The real limit is the amount and variety of data.** A model this small, trained on about 1,000 turns written
  by a few people, does not generalise to new wording on decisions however I re-arrange those turns.

## What was shipped

`lstm_model.normalise()` / `model_words()`: before the model reads a turn, the text is lowercased and the long
and short spellings are made the same: "I will" / "I'll", "we will", "do not" / "don't", "it is" / "it's",
"let us" / "let's", "OK" / "okay", "all right" / "alright", "gonna", and filler sounds ("um", "uh") are
dropped. Names are not touched ("Kai will ..." stays), and neither are possessives ("I have the keys").

- Used in training, in `pipeline.py` (including the "words the model never saw" list) and, for the minutes
  comparison only, in `audio_check.py`. `audio_check.py` still compares **raw** words for the transcriber
  verdict, so "I will" for "I'll" still counts as a transcription difference there.
- The 13 Oct weights were trained on raw words, so the clean-up changes **no** turn of dev or test (0 of 113).
  That is why dev, test, probes, sample meeting, the 67 meeting tests and the frozen scorecard are all exactly
  as before. It changes 52 of 149 turns in holdout 1 and 23 of 122 in holdout 2, which are the meetings I wrote
  with many long forms because Whisper writes them.
- New: [`prototype/normalise_tests.py`](../../prototype/normalise_tests.py) (31 checks),
  [`prototype/holdout_data.py`](../../prototype/holdout_data.py) (17 fresh meetings, never trained on), and two
  new checks in `run_tests.py` (j: clean-up, k: holdout). The leak check in `training_data.py` now compares
  sentences the way the model reads them, and also covers the holdouts and the 67 meeting tests.
- Audio (Claude's side, synthetic voices): the one failure caused by "I will bring..." now passes end to end
  (`solo_2`, voice `rms`: transcriber PASS, end to end PASS, was FAIL). Everything else is unchanged.

## What was tried and not shipped

All rows use the same network and settings as 13 Oct (5 seeds, ensemble, learning rate 0.01, embeddings 32).
Full numbers: [`scores.txt`](scores.txt).

| | Dev (68) | Test (45) | Probes (/12) | Sample (/9) | 67 tests | Holdout 1 (149) | Holdout 2 (122) |
|---|---|---|---|---|---|---|---|
| 13 Oct weights, no clean-up | 65 | 38 | 9 | 7 | 45 | 125 | 104 |
| **Shipped: same weights + clean-up** | 65 | 38 | 9 | 7 | 45 | **131** | 104 |
| Round 1a: retrain, clean-up only | 65 | 38 | 8 | 7 | 40 | 128 | 104 |
| Round 1b: retrain, clean-up + 12 new meetings + new decision templates | 64 | 36 | 8 | 7 | 44 | 133 | 102 |
| Round 2: retrain, much wider content (about 80 proposals, 70 tasks, 40 choices) | 67 | 34 | 8 | 6 | **50** | 129 | 105 |

(Numbers are turns or tests right. The frozen scorecard was scored once, for the shipped state only: 0.90
accuracy / 0.87 macro-F1, identical to before.)

- **Round 1b** looked like a win on holdout 1 (0.893 / 0.833 against 0.839 / 0.716) and was then **worse** on
  holdout 2, which I wrote afterwards (0.836 against 0.852). The holdout 1 gain came from my writing the training
  meetings and holdout 1 in the same sitting, not from better generalisation. This is why a second holdout was
  written before round 2.
- **Round 2** is the best retrain: it finds 12 of 18 decisions in the 67 meeting tests (was 9) and 67 of 68
  dev turns. But the team's test meetings fall from 38 to 34 of 45, the sample meeting from 7 to 6 of 9, and the
  **demo minutes list a wrong decision** ("Good, let's keep monitoring that. Next, the workshop venue."), which
  breaks the team rule of no wrong items. Fails ship-rule items 1 (holdout 2: 0.861 / 0.809, needed 0.902 /
  0.818), 2 (pooled 109, needed 111) and 3.
- Only the 13 Oct weights and the clean-up are in the repo. To rerun rounds 1 and 2: from the repo root,
  `patch -p1 < testing/model_update_16oct/round2.patch`, then `python prototype/train_lstm.py` (about 12 minutes).
  Rounds 1 and 2 also replaced every number by one `<num>` token; that code is in the patch.

## Why it is not working better (what the numbers say)

- Every retrain moved some measures up and others down by 2 to 5 turns. With dev at 68 turns, test at 45 and
  probes at 12, that is about the size of the noise, so the data and the model, not the clean-up, set the ceiling.
- Decisions are the weak class in every version: on holdout 1 the 13 Oct weights find 6 of 22 decisions, the
  clean-up alone raises that to 10 (and action items from 22 to 26 of 27), and round 2 finds 11. Many decisions have no marker word ("Mondays then.", "Seven works.") and depend on
  the turn before; 8 to 32 LSTM units trained on this many turns learn the common wording, not the idea.
- More of my own writing helps on things like my own tests and hurts on the team's test meetings. The data needs
  a different source, not more of the same.

## What would help next (not done)

1. **Real meetings.** Five to ten real club or project meetings, labelled by Asy and Nomt (already item 2 under
   "Next up" in `progress.md`). This is the only change likely to move decisions.
2. **A rule layer for decisions the model cannot see.** For the workshop, a short, readable list of decision
   phrases ("we're going with", "X it is", "we're not doing", "that's settled", "Carried") checked before the
   model, shown as a separate step in the demo. Easy to explain, easy to test on the 67 meeting tests.
3. **Demo choice.** `group_project_3` scores 16 of 18 turns, and the sample meeting 7 of 9. Show one of those
   and say plainly that decisions are the known weakness.
4. **Pretrained word vectors** would help with unseen words, but break the "built from scratch" teaching goal.

## Files in this folder

| File | What it is |
|---|---|
| [`ship_rule.md`](ship_rule.md) | The rule, written before training, plus the round 1 result and the round 2 rule |
| [`scores.txt`](scores.txt) | Every candidate scored on every set, as printed |
| [`round2.patch`](round2.patch) | Changes to `training_data.py` and `lstm_model.py` that reproduce the round 1 and 2 training |
