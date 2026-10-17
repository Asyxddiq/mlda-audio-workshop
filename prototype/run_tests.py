"""
One command that proves the demo still works: python prototype/run_tests.py

Runs every check below, prints one line each (PASS / FAIL / SKIP), then a
summary and the total time. Exits 0 only if nothing FAILED. Needs only the
standard library and numpy; no training, no internet, no Whisper.

    python prototype/run_tests.py                  # everything except audio
    python prototype/run_tests.py --audio meeting.wav
                                                   # also run the pipeline on a recording
                                                   # (needs openai-whisper and ffmpeg)

Run it before every push, and on the presenter laptop the day before the
workshop. Each check is isolated: one failure does not stop the others.
"""

import argparse
import hashlib
import importlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LABELS = ["discussion", "decision", "action_item"]
WEIGHTS_HINT = ("trained_lstm.npz differs from the shipped 13 Oct ensemble; "
                "if a retrain overwrote it: git checkout prototype/trained_lstm.npz")


class Skip(Exception):
    pass


class Fail(Exception):
    pass


def run_script(args, timeout=180):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([sys.executable, *args], cwd=HERE, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout, env=env)
    return done.returncode, done.stdout, done.stderr


def tail(text, n=3):
    return " | ".join(text.strip().splitlines()[-n:])


def expect_ok(args, timeout=180):
    try:
        code, out, err = run_script(args, timeout)
    except subprocess.TimeoutExpired:
        raise Fail(f"python {' '.join(args)} took more than {timeout} s")
    if code != 0:
        raise Fail(f"python {' '.join(args)} exited {code}: {tail(err or out)}")
    return out


def labels_for(turns):
    from lstm_model import run_pipeline
    from meeting_minutes import label_turns_model
    transcript = "\n".join(f"{s}: {t}" for s, t in turns)
    parsed, output, ends = run_pipeline(transcript=transcript)
    labelled = label_turns_model(parsed, output, ends)
    if labelled is None:
        raise Fail("no trained weights loaded. " + WEIGHTS_HINT)
    return [label["category"] for label in labelled]


# ---------------------------------------------------------------------------
# The checks. Each returns a short reason on success, raises Fail or Skip.
# ---------------------------------------------------------------------------

MODULES = ["lstm_model", "meeting_minutes", "pipeline", "training_data", "transcriber",
           "scorecard", "scorecard_data", "train_lstm", "compare_rnn", "rnn_ctc_demo",
           "hands_on_reference"]


def check_imports():
    broken = []
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as error:
            broken.append(f"{name} ({type(error).__name__}: {error})")
    if broken:
        raise Fail("could not import " + "; ".join(broken)
                   + ". Run from a full checkout with numpy installed: pip install -r requirements.txt")
    return f"{len(MODULES)} modules import"


def check_weights():
    import numpy as np
    from lstm_model import TRAINED_WEIGHTS, load_trained_model
    trained = load_trained_model()
    if trained is None:
        raise Fail(f"no weights at {TRAINED_WEIGHTS}. " + WEIGHTS_HINT)
    saved = np.load(TRAINED_WEIGHTS)
    missing = [n for n in range(5) if f"m{n}_embeddings" not in saved.files]
    if len(trained["members"]) != 5 or missing:
        raise Fail(f"expected 5 ensemble members m0..m4, found {len(trained['members'])}. " + WEIGHTS_HINT)
    if trained["labels"] != LABELS:
        raise Fail(f"labels are {trained['labels']}, expected {LABELS}. " + WEIGHTS_HINT)
    return "5 members, labels discussion / decision / action_item"


def check_kpis():
    from scorecard import score
    from training_data import MEETINGS, PROBES

    def run(meetings):
        gold, pred = [], []
        for m in meetings:
            gold += [label for _s, _t, label in m["turns"]]
            pred += labels_for([(s, t) for s, t, _l in m["turns"]])
        return gold, pred

    dev = [m for m in MEETINGS if m["split"] == "dev"]
    test = [m for m in MEETINGS if m["split"] == "test"]
    other = [m for m in test if m["name"] != "sample_meeting"]
    sample = [m for m in test if m["name"] == "sample_meeting"]

    problems = []
    # (name, meetings, meeting count, turn count, accuracy, macro-F1)
    for name, meetings, n_meet, n_turns, acc, f1 in [
            ("dev", dev, 5, 68, 0.96, 0.948),
            ("test", test, 3, 45, 0.84, 0.825),
            ("test without sample_meeting", other, 2, 36, 0.86, 0.853)]:
        gold, pred = run(meetings)
        got = score(gold, pred)
        if len(meetings) != n_meet or len(gold) != n_turns:
            problems.append(f"{name}: {len(meetings)} meetings / {len(gold)} turns, expected {n_meet} / {n_turns}")
        elif abs(got["accuracy"] - acc) > 0.005 or abs(got["macro_f1"] - f1) > 0.005:
            problems.append(f"{name}: accuracy {got['accuracy']:.3f} / macro-F1 {got['macro_f1']:.3f}, "
                            f"expected {acc} / {f1}")

    gold, pred = run(sample)
    right = sum(g == p for g, p in zip(gold, pred))
    if (right, len(gold)) != (7, 9):
        problems.append(f"sample_meeting: {right}/{len(gold)} turns, expected 7/9")

    probe_right = sum(ex[-1][2] == labels_for([(s, t) for s, t, _l in ex])[-1] for ex in PROBES)
    if (probe_right, len(PROBES)) != (9, 12):
        problems.append(f"probes: {probe_right}/{len(PROBES)}, expected 9/12")

    if problems:
        raise Fail("; ".join(problems) + ". " + WEIGHTS_HINT)
    return "dev 0.96/0.948, test 0.84/0.825, 36-turn test 0.86/0.853, sample 7/9, probes 9/12"


def check_demo_minutes():
    import pipeline
    minutes = pipeline.run(verbose=False)
    decisions, _, actions = minutes.partition("## Action Items")
    items = [line for line in actions.splitlines() if line.startswith("- ")]
    if "- (none)" not in decisions or any(line.startswith("- ") and line != "- (none)"
                                          for line in decisions.splitlines()):
        raise Fail("the sample meeting should list no decisions (team rule: no wrong items), got:\n" + decisions
                   + "\n" + WEIGHTS_HINT)
    if len(items) != 1 or not items[0].endswith("Follow up with catering by Friday -- Priya (due Friday)"):
        raise Fail(f"expected exactly one action item 'Follow up with catering by Friday -- Priya "
                   f"(due Friday)', got {items}. " + WEIGHTS_HINT)
    return "no decisions, one action item (catering by Friday, Priya)"


def check_text_example():
    import pipeline
    path = os.path.join(HERE, "my_meeting_example.txt")
    transcript, gold = pipeline.load_text(path)
    if not gold:
        raise Fail("my_meeting_example.txt has no hand labels any more; each line should start with act / dec / dis")
    pipeline.run(text=path, verbose=False)
    pred = labels_for([tuple(line.split(": ", 1)) for line in transcript.splitlines()])
    right = sum(g == p for g, p in zip(gold, pred))
    if right != len(gold):
        raise Fail(f"model matches {right}/{len(gold)} hand labels on my_meeting_example.txt, expected 7/7. "
                   + WEIGHTS_HINT)
    return f"runs, model matches the hand labels {right}/{len(gold)}"


def check_hands_on():
    out = expect_ok(["hands_on.py", "--solution"])
    if out.count(": OK") != 2:
        raise Fail("hands_on.py --solution should print OK for both Part A and Part B. If you edited "
                   "hands_on.py or hands_on_reference.py, compare them with git diff")
    if "FORGET_KNOB = +0" not in out or "35/45 turns right" not in out:
        score_line = next((line.strip() for line in out.splitlines() if "turns right" in line), tail(out, 2))
        raise Fail(f"hands_on.py --solution should report 35/45 test turns at FORGET_KNOB = 0, got: "
                   f"{score_line}. " + WEIGHTS_HINT)
    plain = expect_ok(["hands_on.py"])
    if plain.count("not written yet") != 2 or "Traceback" in plain:
        raise Fail("the unfilled hands_on.py should say 'not written yet' twice and not crash, got: "
                   + tail(plain))
    return "solution: both parts OK, 35/45 at knob 0; unfilled version says 'not written yet'"


def check_training_data():
    expect_ok(["training_data.py"])
    return "leak checks pass"


def check_scorecard():
    out = expect_ok(["scorecard.py"])
    if "Scorecard:" not in out:
        raise Fail("scorecard.py ran but printed no scorecard: " + tail(out))
    return "frozen scorecard runs (hash ignores line endings)"


def check_cleanup():
    out = expect_ok(["normalise_tests.py"])
    last = out.strip().splitlines()[-1]
    done, _, total = last.split(" ")[0].partition("/")
    if not done or done != total:
        raise Fail("normalise_tests.py: " + tail(out, 4))
    return f"text clean-up: {last}"


def check_holdout():
    """16 Oct: two sets of fresh meetings the model was never trained on (holdout_data.py)."""
    from holdout_data import HOLDOUT, HOLDOUT2
    from scorecard import score
    problems, shown = [], []
    for name, meetings, n_turns, right_expected, acc, f1 in [
            ("holdout 1", HOLDOUT, 149, 131, 0.879, 0.806),
            ("holdout 2", HOLDOUT2, 122, 104, 0.852, 0.762)]:
        gold, pred = [], []
        for m in meetings:
            gold += [label for _s, _t, label in m["turns"]]
            pred += labels_for([(s, t) for s, t, _l in m["turns"]])
        got = score(gold, pred)
        right = sum(g == p for g, p in zip(gold, pred))
        shown.append(f"{name} {got['accuracy']:.3f}/{got['macro_f1']:.3f}")
        if len(gold) != n_turns:
            problems.append(f"{name}: {len(gold)} turns, expected {n_turns}")
        elif abs(got["accuracy"] - acc) > 0.005 or abs(got["macro_f1"] - f1) > 0.005:
            problems.append(f"{name}: accuracy {got['accuracy']:.3f} / macro-F1 {got['macro_f1']:.3f} "
                            f"({right}/{len(gold)}), expected {acc} / {f1}")
    if problems:
        raise Fail("; ".join(problems) + ". " + WEIGHTS_HINT)
    return ", ".join(shown) + " (accuracy / macro-F1)"


def check_smoke():
    runs = [["lstm_model.py"], ["rnn_ctc_demo.py"], ["transcriber.py"], ["pipeline.py", "--compare"],
            ["pipeline.py", "--meeting", "group_project_3", "--compare"]]
    for args in runs:
        expect_ok(args)
    return f"{len(runs)} scripts exit 0"


def check_audio(path):
    if not path:
        raise Skip("no --audio FILE given")
    if not os.path.isfile(path):
        raise Fail(f"no audio file at {path}")
    if importlib.util.find_spec("whisper") is None:
        raise Skip("openai-whisper isn't installed (pip install openai-whisper)")
    if shutil.which("ffmpeg") is None:
        raise Skip("ffmpeg isn't on PATH (winget install --id Gyan.FFmpeg -e)")
    out = expect_ok(["pipeline.py", os.path.abspath(path)], timeout=900)
    if "# Meeting Minutes" not in out:
        raise Fail("pipeline.py ran on the audio but printed no minutes: " + tail(out))
    return f"pipeline ran on {os.path.basename(path)}"


# ---------------------------------------------------------------------------
# m. The tests must leave the repo exactly as they found it
# ---------------------------------------------------------------------------

def repo_snapshot():
    if shutil.which("git") is None:
        return None
    try:
        status = subprocess.run(["git", "-C", HERE, "status", "--porcelain"], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=60)
        diff = subprocess.run(["git", "-C", HERE, "diff"], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if status.returncode != 0 or diff.returncode != 0:
        return None
    lines = [line for line in status.stdout.splitlines() if "__pycache__" not in line]
    return lines, hashlib.sha256(diff.stdout).hexdigest()


def check_repo_unchanged(before):
    if before is None:
        raise Skip("not a git checkout, or git isn't available")
    after = repo_snapshot()
    if after != before:
        raise Fail("running the checks changed files in the repo (git status or git diff differs from "
                   "before the run). Look at git status and git diff")
    return "git status and git diff identical before and after"


def main():
    parser = argparse.ArgumentParser(description="Run every check on the demo.")
    parser.add_argument("--audio", metavar="FILE", help="also run the pipeline on this audio file")
    args = parser.parse_args()

    start = time.time()
    print("Training scripts (train_lstm.py, compare_rnn.py) are intentionally not run: they take "
          "12-20 minutes and train_lstm.py overwrites the shipped weights.\n")
    before = repo_snapshot()

    checks = [
        ("a", "imports", check_imports),
        ("b", "weights", check_weights),
        ("c", "shipped KPIs reproduce", check_kpis),
        ("d", "demo minutes", check_demo_minutes),
        ("e", "--text example", check_text_example),
        ("f", "hands-on", check_hands_on),
        ("g", "training data", check_training_data),
        ("h", "scorecard", check_scorecard),
        ("i", "smoke runs", check_smoke),
        ("j", "text clean-up", check_cleanup),
        ("k", "holdout meetings", check_holdout),
        ("l", "audio (optional)", lambda: check_audio(args.audio)),
        ("m", "repo unchanged", lambda: check_repo_unchanged(before)),
    ]

    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for key, name, check in checks:
        began = time.time()
        try:
            status, reason = "PASS", check()
        except Skip as why:
            status, reason = "SKIP", str(why)
        except Fail as why:
            status, reason = "FAIL", str(why)
        except Exception as error:
            status, reason = "FAIL", f"{type(error).__name__}: {error}. " + WEIGHTS_HINT
        counts[status] += 1
        print(f"{status}  {key}. {name:<24} {reason}  ({time.time() - began:.1f}s)")

    print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped "
          f"in {time.time() - start:.1f} s")
    sys.exit(1 if counts["FAIL"] else 0)


if __name__ == "__main__":
    main()
