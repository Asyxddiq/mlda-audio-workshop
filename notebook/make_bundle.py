"""
Make workshop_files.zip: the code and audio the participants' notebook needs, in one file.

    python notebook/make_bundle.py

Participants upload this zip (and open audio_workshop.ipynb) in Google Colab; the notebook's first
cell unpacks it. The repo is private, so this is the way to hand the files out. Rebuild it whenever
anything in prototype/ changes.
"""
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PROTOTYPE = ["audio_check.py", "audio_features.py", "hands_on.py", "hands_on_reference.py", "lstm_model.py",
             "meeting_minutes.py", "pipeline.py", "rnn_ctc_demo.py", "training_data.py", "transcriber.py",
             "transcript_placeholder.py", "trained_lstm.npz"]
AUDIO = ["sample_meeting.wav", "sample_short.wav"]

out = os.path.join(HERE, "workshop_files.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for name in PROTOTYPE:
        z.write(os.path.join(REPO, "prototype", name), f"prototype/{name}")
    for name in AUDIO:
        z.write(os.path.join(HERE, "audio", name), f"audio/{name}")
print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB, {len(PROTOTYPE) + len(AUDIO)} files)")
