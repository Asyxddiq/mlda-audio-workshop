"""
Test the pipeline one phase at a time, with a PASS or FAIL for each.

    python testing/phases.py                       # phases 1 (logic only), 2, 3, 4: no audio, about 15 s
    python testing/phases.py 3                     # one phase
    python testing/phases.py 1 --audio rec.m4a     # phase 1 on a real recording (needs Whisper + ffmpeg)
    python testing/phases.py 1 --audio rec.m4a --script solo_2 --pause-gap 1.5

The phases follow the chain, in order:

  1  Audio -> text      Whisper + the pause rule (prototype/transcriber.py)
  2  RNN                the Part 5 teaching demo (prototype/rnn_ctc_demo.py, prototype/hands_on.py Part A)
  3  LSTM               the trained hourglass labelling each turn (prototype/lstm_model.py)
  4  Meeting minutes    labels -> decisions and action items (prototype/meeting_minutes.py)

What each one does NOT prove is printed under its result. Phase 1 without --audio only checks the
pause logic on made-up segments; it says so. The RNN in phase 2 is a teaching demo with random
weights: the real audio -> text is Whisper, not that RNN.

Exit code 0 only if every phase that ran passed.
"""

import argparse
import contextlib
import io
import os
import re
import subprocess
import sys

# Anaconda on Windows can load two copies of the OpenMP library (see solo_test.py)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PROTO = os.path.join(REPO, "prototype")
SCRIPTS = os.path.join(HERE, "scripts")
sys.path.insert(0, PROTO)

SHIPPED_MEETING_TESTS = 45      # what the shipped model passes of the 67 typed meetings (testing.md)
EXAMPLE_TEXT = os.path.join(PROTO, "my_meeting_example.txt")


class Phase:
    def __init__(self, number: int, name: str):
        self.number, self.name = number, name
        self.checks: list[tuple[bool, str]] = []
        self.note = ""

    def check(self, ok: bool, text: str) -> bool:
        self.checks.append((bool(ok), text))
        print(f"    [{'ok' if ok else 'FAIL'}]  {text}")
        return bool(ok)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(ok for ok, _ in self.checks)


def run_script(args: list[str], timeout: int = 300) -> tuple[int, str]:
    """Run a prototype script in its own process; returns (exit code, all output)."""
    done = subprocess.run([sys.executable, *args], cwd=REPO, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)
    return done.returncode, done.stdout + done.stderr


# ---------------------------------------------------------------------------
# Phase 1: audio -> text
# ---------------------------------------------------------------------------

def find_script(name: str) -> str:
    if os.path.isfile(name):
        return name
    matches = sorted(f for f in os.listdir(SCRIPTS) if f.startswith(name) and f.endswith(".txt"))
    if not matches:
        sys.exit(f"No script called {name!r} in {SCRIPTS}")
    return os.path.join(SCRIPTS, matches[0])


def phase1(args) -> Phase:
    p = Phase(1, "Audio -> text (Whisper + pause rule)")
    code, out = run_script([os.path.join("prototype", "transcriber.py")])
    p.check(code == 0 and "self-test passed" in out,
            "pause rule on made-up segments: speakers alternate, same-speaker pieces merge, "
            "word-by-word input splits at a pause")
    if not args.audio:
        p.note = ("Logic only: NO audio was tested. Give a recording to test Whisper: "
                  "python testing/phases.py 1 --audio rec.m4a")
        return p
    if not os.path.isfile(args.audio):
        p.check(False, f"audio file not found: {args.audio}")
        return p
    script = find_script(args.script)
    gap = args.pause_gap if args.pause_gap is not None else (1.5 if os.path.basename(script).startswith("solo") else 0.8)
    print(f"    ...transcribing {os.path.basename(args.audio)} with Whisper, pause gap {gap} s "
          f"(first run downloads the model)")
    from audio_check import check as audio_check
    from transcriber import WHISPER_MODEL, get_transcript
    heard = get_transcript(args.audio, pause_gap=gap)
    print("    heard:")
    for line in heard.strip().splitlines():
        print(f"      {line}")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        audio_check(script, heard)
    report = buf.getvalue()
    rows = {m.group(1): (m.group(2) == "OK", m.group(3)) for m in
            re.finditer(r"^\s+(Turns|Speakers|Words)\s+(OK|FAIL)\s+(.*)$", report, re.M)}
    for name in ("Turns", "Speakers", "Words"):
        ok, note = rows.get(name, (False, "no result"))
        p.check(ok, f"{name.lower()}: {note}")
    p.note = (f"Model {WHISPER_MODEL}, script {os.path.basename(script)}. Passing means the audio became the "
              f"script's turns, speakers and 90% of its words; it does not test the labels or minutes (phases 3 and 4).")
    return p


# ---------------------------------------------------------------------------
# Phase 2: RNN
# ---------------------------------------------------------------------------

def phase2(args) -> Phase:
    p = Phase(2, "RNN (Part 5 teaching demo)")
    import numpy as np
    import rnn_ctc_demo as demo
    model = demo.SimpleRNN(demo.F, demo.H, len(demo.CHARS))
    frames = np.random.default_rng(0).normal(size=(50, demo.F))
    probs = model.forward(frames)
    p.check(probs.shape == (50, len(demo.CHARS)), f"50 audio frames in -> {probs.shape} probabilities out (one row per frame)")
    p.check(np.allclose(probs.sum(axis=1), 1), "every row of probabilities sums to 1")
    ix = {c: i for i, c in enumerate(demo.CHARS)}

    def onehot(seq):
        m = np.zeros((len(seq), len(demo.CHARS)))
        for t, c in enumerate(seq):
            m[t, c] = 1
        return m

    hello = onehot([ix["h"], ix["h"], ix["e"], 0, ix["l"], ix["l"], 0, ix["l"], ix["o"]])
    p.check(demo.ctc_greedy_decode(hello) == "hello", "CTC decoder: h h e _ l l _ l o -> 'hello' (the blank keeps the double l)")
    p.check(demo.ctc_greedy_decode(onehot([ix["b"], ix["b"], ix["b"]])) == "b", "CTC decoder: b b b -> 'b' (repeats merge)")
    p.check(demo.ctc_greedy_decode(onehot([0, 0, 0])) == "", "CTC decoder: all blanks -> empty text")
    code, out = run_script([os.path.join("prototype", "hands_on.py"), "--solution"])
    p.check(code == 0, "hands_on.py Part A (the participants' RNN step, reference answer) runs and checks out")
    p.note = ("The weights are random, so this proves the shapes and the decoder, not that it hears speech. "
              "The real audio -> text in the pipeline is Whisper (phase 1), not this RNN.")
    return p


# ---------------------------------------------------------------------------
# Phase 3: LSTM
# ---------------------------------------------------------------------------

def load_example():
    from pipeline import load_text
    from lstm_model import run_pipeline
    transcript, gold = load_text(EXAMPLE_TEXT)
    turns, output, ends = run_pipeline(transcript=transcript)
    return transcript, gold, turns, output, ends


def phase3(args) -> Phase:
    p = Phase(3, "LSTM (trained hourglass labels each turn)")
    import numpy as np
    import lstm_model
    from meeting_minutes import label_turns_keywords, label_turns_model
    trained = lstm_model.load_trained_model()
    p.check(trained is not None and len(trained["members"]) == 5,
            f"trained weights load: {len(trained['members']) if trained else 0} models in the ensemble (expect 5)")
    x = np.random.default_rng(1).normal(size=(88, 8))
    p.check(np.array_equal(lstm_model.unfold_time(lstm_model.fold_time(x)), x), "fold -> unfold is lossless")
    transcript, gold, turns, output, ends = load_example()
    p.check(output.shape[1] == 8 and len(ends) == len(turns),
            f"hourglass output is {output.shape[0]} words x 8 numbers, with {len(ends)} turn ends for {len(turns)} turns")
    model = [l["category"] for l in label_turns_model(turns, output, ends)]
    kw = [l["category"] for l in label_turns_keywords(turns)]
    right = sum(m == g for m, g in zip(model, gold))
    kwright = sum(k == g for k, g in zip(kw, gold))
    print(f"\n      {'turn':<48} {'hand':<12} {'model':<12} keywords")
    for (sp, tx), g, m, k in zip(turns, gold, model, kw):
        print(f"      {(sp + ': ' + tx)[:46]:<48} {g:<12} {m:<12} {k}{'' if k == g else '  x'}")
    print()
    p.check(right == len(gold), f"trained model matches the hand labels on {right}/{len(gold)} turns "
                                f"(keyword rules: {kwright}/{len(gold)})")
    p.note = ("Seven turns is a smoke test, not a score. The scores on held-out meetings "
              "(dev, test, probes, scorecard, holdouts) are in: python prototype/run_tests.py")
    return p


# ---------------------------------------------------------------------------
# Phase 4: meeting minutes
# ---------------------------------------------------------------------------

def phase4(args) -> Phase:
    p = Phase(4, "Meeting minutes")
    from meeting_minutes import generate_minutes, label_turns_model
    transcript, gold, turns, output, ends = load_example()
    minutes = generate_minutes(turns, label_turns_model(turns, output, ends))
    print("      minutes for prototype/my_meeting_example.txt:")
    for line in minutes.strip().splitlines():
        print(f"        {line}")
    print()
    actions = [l for l in minutes.splitlines() if l.startswith("- [ ]")]
    p.check(len(actions) == 1 and "cups" in actions[0].lower() and "Bea" in actions[0],
            "request + 'Okay.' becomes ONE action item: buy the cups, owner Bea (the person who accepted)")
    p.check("drinks" in minutes.lower() or "sell" in minutes.lower(),
            "'Sure, let's do that.' after a proposal is listed as a decision about the drinks")
    code, out = run_script([os.path.join("prototype", "minutes_tests.py")])
    m = re.search(r"Total (\d+)/(\d+) passed", out)
    passed, total = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    for line in out.splitlines():
        if re.match(r"\s+[A-E]\s+\S", line):
            print(f"      {line.strip()}")
    p.check(passed >= SHIPPED_MEETING_TESTS,
            f"{passed}/{total} typed meeting tests pass (the shipped model passes {SHIPPED_MEETING_TESTS}; "
            f"all 67 is not expected: decisions are the known weak spot)")
    p.note = ("Typed text only: the words are exactly right, so this is the labels -> minutes step in "
              "isolation. The expected minutes for every case are in testing/testing script.md.")
    return p


PHASES = {1: phase1, 2: phase2, 3: phase3, 4: phase4}


def main() -> int:
    ap = argparse.ArgumentParser(description="Test the pipeline one phase at a time.")
    ap.add_argument("phases", nargs="*", type=int, choices=[1, 2, 3, 4], help="which phases (default: all)")
    ap.add_argument("--audio", help="a recording for phase 1 (needs openai-whisper and ffmpeg)")
    ap.add_argument("--script", default="solo_1", help="the script that was read aloud (default solo_1)")
    ap.add_argument("--pause-gap", type=float, help="seconds of silence that count as a new speaker "
                    "(default 1.5 for solo scripts, 0.8 otherwise)")
    args = ap.parse_args()
    chosen = args.phases or [1, 2, 3, 4]

    results = []
    for n in chosen:
        print(f"\n=== Phase {n}: " + {1: "Audio -> text", 2: "RNN", 3: "LSTM", 4: "Meeting minutes"}[n])
        try:
            phase = PHASES[n](args)
        except ModuleNotFoundError as e:
            phase = Phase(n, "")
            phase.check(False, f"missing package: {e.name}" + (" (pip install openai-whisper, and install ffmpeg)"
                                                              if e.name == "whisper" else ""))
        except Exception as e:  # a crash is a FAIL, with the reason
            phase = Phase(n, "")
            phase.check(False, f"crashed: {type(e).__name__}: {e}")
        print(f"  {'PASS' if phase.passed else 'FAIL'}  phase {n}" + (f"\n  Not proven: {phase.note}" if phase.note else ""))
        results.append((n, phase))

    print("\n=== Summary")
    names = {1: "Audio -> text", 2: "RNN", 3: "LSTM", 4: "Meeting minutes"}
    for n, ph in results:
        extra = "  (pause logic only, no audio tested)" if n == 1 and not args.audio else ""
        print(f"  {'PASS' if ph.passed else 'FAIL'}  {n}. {names[n]}{extra}")
    return 0 if all(ph.passed for _, ph in results) else 1


if __name__ == "__main__":
    sys.exit(main())
