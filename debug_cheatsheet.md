# Debug cheat sheet

For **Part 9: Debug time**. Every row is something that really went wrong while we built and tested this. Find your symptom, try the fix.

## Setting up

| What you see | Why | Fix |
|---|---|---|
| `pip is not recognized` | Python is installed but not on the PATH | Use `python -m pip install ...` instead of `pip install ...` |
| `ffmpeg is not recognized`, right after installing it | The terminal you have open was started before the install | Close it and open a new one. In the same window: `$env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")` |
| `OMP: Error #15 ... libiomp5md.dll already initialized` | Anaconda on Windows loaded two copies of one maths library | Set `KMP_DUPLICATE_LIB_OK=TRUE` before running (`os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"`). The notebook and test scripts already do |
| `No module named whisper` | Whisper isn't installed | `python -m pip install openai-whisper` (a few GB with PyTorch, so do it before the day) |
| `No module named sounddevice` | Only needed to record from the microphone | `python -m pip install sounddevice`, or record on your phone and give the file instead |
| Colab: "Upload workshop_files.zip" | The code zip isn't in the Files panel | Drag `workshop_files.zip` into the Files panel (left) and run the cell again. The files vanish when the runtime restarts, so upload again |
| First run takes a long time | Whisper downloads its model once (139 MB `base.en`, 461 MB `small.en`) | Wait; it's cached afterwards. Needs internet |

## Audio to text

| What you see | Why | Fix |
|---|---|---|
| Text in another language, or gibberish, and very slow | Whisper guessed the language from an accent | Always set `language="en"`, and use an English-only model (`base.en`, `small.en`) |
| Whisper is slow | No GPU | Use `base.en`, or on Colab choose **Runtime > Change runtime type > T4 GPU** |
| One odd word for a whole recording ("Classroom") | The recording is almost silent, too short, or stopped early | Check the microphone isn't muted or set to the wrong device, speak louder and closer, and keep recording to the end |
| **Fewer turns** than people spoke | Pauses between speakers too short, or a noisy room hides them | Pause a full 2 seconds between turns, in a quiet room. The pause that counts as a new speaker is `pause_gap` in `get_transcript` |
| **More turns** than people spoke | A breath or pause longer than `pause_gap` inside one person's turn | Speak each turn in one go. Raise `pause_gap` (1.5 s for one person reading, 0.8 s for two people talking) |
| Wrong words ("catering" heard as "Patterning") | Accent, background noise or a poor microphone | Quiet room, mic close, a bigger model (`small.en`, `medium.en`). This is section 2 of the notebook: it is expected, not a bug |
| Speakers are `Speaker1` and `Speaker2`, not names | Whisper only writes words; speakers are guessed from pauses | Expected. More than two speakers are not told apart |

## The model and the minutes

| What you see | Why | Fix |
|---|---|---|
| A decision is missing from the minutes ("Venue is confirmed", a bare "Sounds good.") | The model is weakest at decisions: it learned from a few hundred turns | Known limit, not a bug. Try the same line after a clear proposal ("Shall we move it to Friday?" then "Sounds good.") |
| An action item that isn't one ("Can you remind me where the form is?") | A question for right now looks like a request | Known limit. Say what you'd change in the labelling rules |
| Words listed as "never seen" | The model only knows its training vocabulary | Expected for rare words; they are ignored |
| `I will` and `I'll` give different results | They used to; the text clean-up now treats them the same | Make sure you have the latest `lstm_model.py` |

## The hands-on (`hands_on.py` and the notebook)

| What you see | Why | Fix |
|---|---|---|
| `not written yet (it returned None)` | Your function still ends with `return None` | Write the line and return it |
| `not quite  got [...], expected [...]` | The formula is slightly wrong | Part A: `np.tanh(W_x @ x + W_h @ h_prev + b)`. Part B: `f * c_prev + i * g` (`@` is matrix times vector, `*` is element by element) |
| Part B's test score collapses | Your line runs inside the trained model, so a wrong line breaks it | Fix the formula until the check says OK |
| `FORGET_KNOB` at -6 or +6 makes the score worse | The trained model learned its own settings at 0 | Expected: both extremes hurt. Negative erases the notebook, positive never lets it go |

## Before the workshop

* `python prototype/run_tests.py` (about 25 s) must say `0 failed`.
* Try the whole thing once on the laptop you will present from, with its microphone: `python testing/try_it.py --name YOU`.
* Open the notebook on Colab once with a fresh runtime, with a clean copy of `workshop_files.zip`, and run every cell.
