"""
Make the two zips this workshop hands out:

    python notebook/make_bundle.py

Writes:
  - `workshop_files.zip`    -- for participants. Only what
                               `audio_workshop.ipynb` needs. Uploaded to the
                               Files panel in Colab; the notebook's first
                               cell unpacks it.
  - `facilitator_files.zip` -- for the facilitator running the hook. The
                               participant zip's contents, plus
                               `prototype/demo.py`, `demo/script.txt` and
                               `demo/backup_meeting.wav`, so
                               `demo/hook_demo.ipynb` works from this zip
                               alone (see WP3 review, "Answer to your zip
                               question" -- the facilitator notebook is the
                               only thing that needs `demo.py`).

Both are gitignored (rebuild, don't commit) -- see `.gitignore`. Rebuild
whenever anything in `prototype/`, `notebook/audio/` or `demo/` changes.

Deliberately NOT bundled, in either zip:
  - `prototype/hands_on_reference.py` -- the Part A / Part B answers. No
    notebook cell imports it (`hands_on.py` only does, inside
    `if SOLUTION`, which a notebook never sets); shipping it would hand
    every participant both answers. Facilitators keep it from the repo
    (`python prototype/hands_on.py --solution`); `run_tests.py` also reads
    it straight from `prototype/`, not from a zip.
  - `prototype/rnn_ctc_demo.py` -- not imported by any notebook cell.
  - `demo/ANSWER_KEY.md` -- the facilitator's reference; see the DEMO
    comment below for why.
"""
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# Everything audio_workshop.ipynb (participants) actually imports or reads.
PROTOTYPE = ["audio_check.py", "audio_features.py", "demo_student_council.txt", "hands_on.py",
             "lstm_model.py", "meeting_minutes.py", "pipeline.py", "training_data.py",
             "transcriber.py", "transcript_placeholder.py", "trained_lstm.npz"]
# demo_student_council.txt: N7's held-out meeting (keyword rules vs the trained LSTM,
# section 4). Never in training_data.py -- see that file's own header comment.
AUDIO = ["sample_meeting.wav", "sample_short.wav",
        # N2's Kahoot clips (notebook/make_noisy_clips.py) -- rebuild those before this if
        # sample_short.wav or sample_meeting.wav change.
        "noisy_1_ideal.wav", "noisy_2_hiss.wav", "noisy_3_poor_mic.wav", "noisy_4_second_voice.wav"]
# WP7's live-mic no-mic fallback clips (notebook/make_live_clips.py) -- one per live
# prompt_id, read by live_mic.say() when there's no mic, it isn't Colab, or permission
# was denied.
AUDIO_LIVE = ["block2_clear.wav", "block2_noisy.wav", "block2_backward.wav",
              "block3_sentence.wav", "block4_cats.wav", "block4_dogs.wav",
              "block5_short_a.wav", "block5_short_b.wav", "block5_long_a.wav", "block5_long_b.wav",
              "block6_gates.wav", "block7_request.wav", "block7_proposal.wav", "block7_fact.wav",
              "block8_exchange.wav", "sure_request.wav", "sure_proposal.wav", "block9_meeting.wav",
              "headline_short_conv.wav", "headline_long_conv.wav"]
# live_mic.py and toy_memory.py sit at the zip root (same level as
# audio_workshop.ipynb), not under prototype/ -- that is where every live cell
# imports them from. CHEATSHEET.md (WP9 F3) is there for participants to read,
# not imported by anything.
NOTEBOOK_TOP = ["live_mic.py", "toy_memory.py", "CHEATSHEET.md"]

# Only for the facilitator notebook (demo/hook_demo.ipynb), on top of the above.
# ANSWER_KEY.md is deliberately NOT bundled, even here: it is read from the
# repo by the facilitator, not needed by any notebook cell, and leaving it
# out of every zip means a participant who somehow got a facilitator zip
# still wouldn't have the answers (WP1 review finding 6).
FACILITATOR_PROTOTYPE = ["demo.py"]
FACILITATOR_DEMO = ["script.txt", "backup_meeting.wav"]


def _write(out_name: str, prototype: list[str], audio: list[str], demo: list[str],
           audio_live: list[str], notebook_top: list[str]) -> str:
    out = os.path.join(HERE, out_name)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in prototype:
            z.write(os.path.join(REPO, "prototype", name), f"prototype/{name}")
        for name in audio:
            z.write(os.path.join(HERE, "audio", name), f"audio/{name}")
        for name in demo:
            z.write(os.path.join(REPO, "demo", name), f"demo/{name}")
        for name in audio_live:
            z.write(os.path.join(HERE, "audio", "live", name), f"audio/live/{name}")
        for name in notebook_top:
            z.write(os.path.join(HERE, name), name)
    count = len(prototype) + len(audio) + len(demo) + len(audio_live) + len(notebook_top)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB, {count} files)")
    return out


if __name__ == "__main__":
    _write("workshop_files.zip", PROTOTYPE, AUDIO, [], AUDIO_LIVE, NOTEBOOK_TOP)
    _write("facilitator_files.zip", PROTOTYPE + FACILITATOR_PROTOTYPE, AUDIO, FACILITATOR_DEMO,
           AUDIO_LIVE, NOTEBOOK_TOP)
