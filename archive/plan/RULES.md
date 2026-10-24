# Rules for both sessions

## Scope
Only the demo and the notebook. Nothing about slides, Kahoot, the take-home, the worksheet or the README.

## Where the builder may write
- `notebook/**`
- `demo/**`, `prototype/demo.py`
- `progress.md` (one log entry), `debug_cheatsheet.md` (new rows only)
- `prototype/hands_on.py`: additive changes only, and only if `hands_on_reference.py` and `run_tests.py` still pass. Say so in the handoff.

## Frozen: the builder must not touch these (the reviewer checks with `git diff --stat main`)
`prototype/trained_lstm.npz`, `training_data.py`, `scorecard_data.py`, `holdout_data.py`, `minutes_tests.py`, `normalise_tests.py`,
`lstm_model.py`, `train_lstm.py`, `meeting_minutes.py`, `transcriber.py`, `hands_on_reference.py`, anything under `testing/`.
No retraining. No new model. If the demo needs a model change, stop and say so; do not make it.

## Rule H: the demo script and the model
- Write the mock script and the answer key FIRST, from what a human hears. Commit the key before running the model.
- Run the model on it. If it misses an item, you may reword the script, within `labelling_guide.md`, but every attempt is listed in the handoff (attempt number, what changed, result). Never change the key to match the model.
- Never add demo sentences to the training data. `python prototype/training_data.py` (leak check) must still pass.
- The two "Sure." exchanges are reported exactly as the model labels them. Do not reword them to make them pass. If the model gets one wrong, that is a finding for the notebook (N7/N9), not a bug to hide.
- The facilitator page says plainly that the script was chosen after seeing how the model reads it.

## Honesty
- Retention graph = an illustration, say so on the cell.
- No claim that the trained LSTM remembers further back than an RNN.
- "Keep a lot" is not "better memory" (see SPEC tension 2).
- Any number printed in a notebook cell or document must come from a command you ran. Paste the output in the handoff.

## Git
- Each session works only in its own branch/worktree. Commit small and often. Never push, never merge to `main`.
- Commit messages end with the attribution line the harness gives you.
- Windows: use `python`, forward slashes in paths in code, `KMP_DUPLICATE_LIB_OK=TRUE` (the notebook already sets it).

## Reviewer-only rules
- Write only under `reviews/` on `review/workshop-gaps`. Never edit builder files.
- Do not trust the handoff. Re-run: `python prototype/run_tests.py`, the demo command, `python notebook/make_bundle.py`, and the notebook from an empty folder holding only the zip (use a detached worktree of the builder's commit).
- Check every requirement ID the handoff claims against the PDF text in `reference workshop docs/`.
- Verdict file format for `reviews/WPn.md`: first line `VERDICT: PASS` or `VERDICT: CHANGES`; then a table of findings (severity blocker / should-fix / nit, requirement ID, file:line, what is wrong, how to fix); then the commands you ran with their output.
- Look hardest for: cherry-picked demo text, frozen files changed, claims not backed by a printed number, a cell that only works on the builder's machine, timings that overrun the block.
