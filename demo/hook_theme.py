"""Dark/purple presentation helpers for the hook demo (facilitator notebook).

Presentation only: nothing here labels turns, writes minutes or touches `prototype/`.
Every function returns an `IPython.display.HTML` object (self-contained CSS, no JavaScript, a
Google Fonts `@import` for Poppins that falls back to Segoe UI / system-ui when offline), so a
cell can end with the call, or wrap it in `display(...)`.

    banner(title="Who remembers the meeting?", subtitle="Workshop 4 · Deep Learning in Audio Applications")
    progress(stage)                  # "listening" | "transcribing" | "labelling" | "minutes" | "done"
    ProgressStrip(stage="listening") # shows once, then .update(stage) redraws in place; .done() marks all green
    minutes_cards(minutes_md)        # generate_minutes() Markdown -> Decisions / Action items cards
    audio_panel(path, title="Recording")  # waveform + spectrogram picture and an embedded <audio> player
    parse_minutes(minutes_md)        # the parser behind minutes_cards, if you need the pieces

Palette (from the workshop deck): #0D0D12 / #15151C backgrounds, #2E2E3C cards, purple #8B5CF6 to
#A855F7 (one soft glow), pink #EC4899, green #22C55E for done.
"""
import base64
import html as _html
import io
import os
import re
import wave

from IPython.display import HTML, display

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # avoids the Windows OpenMP clash when whisper (torch) and numpy both load

BG0, BG1, CARD = "#0D0D12", "#15151C", "#2E2E3C"
PURPLE, PURPLE2, PINK, GREEN = "#8B5CF6", "#A855F7", "#EC4899", "#22C55E"
TEXT, MUTED, DIM = "#F4F4F8", "#9CA3AF", "#6B7280"

STAGES = ("listening", "transcribing", "labelling", "minutes")
_STAGE_LABEL = {"listening": "Listening", "transcribing": "Transcribing", "labelling": "Labelling", "minutes": "Minutes"}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
.hk, .hk * { box-sizing: border-box; font-family: 'Poppins', 'Segoe UI', system-ui, -apple-system, sans-serif; }
.hk { color: __TEXT__; }

.hk-banner { position: relative; overflow: hidden; margin: 8px 0 16px 0; padding: 34px 40px;
  background: linear-gradient(135deg, __BG1__ 0%, __BG0__ 100%); border: 1px solid __CARD__; border-radius: 20px; }
.hk-banner::before { content: ""; position: absolute; right: -90px; top: -130px; width: 440px; height: 440px;
  background: radial-gradient(circle, rgba(139,92,246,.50) 0%, rgba(139,92,246,0) 66%); pointer-events: none; }
.hk-banner h1 { position: relative; margin: 0; padding: 0; font-size: 46px; line-height: 1.15; font-weight: 700; color: #fff; letter-spacing: -0.2px; }
.hk-banner p { position: relative; margin: 12px 0 0 0; font-size: 20px; line-height: 1.4; font-weight: 400; color: __MUTED__; }

.hk-prog { display: flex; align-items: center; margin: 10px 0; padding: 16px 18px; background: __BG1__;
  border: 1px solid __CARD__; border-radius: 20px; }
.hk-step { flex: 1 1 0; text-align: center; padding: 16px 10px; border-radius: 14px; background: __BG0__;
  border: 1px solid __CARD__; color: __MUTED__; font-size: 22px; font-weight: 600; white-space: nowrap; }
.hk-step.done { color: __GREEN__; border-color: rgba(34,197,94,.55); }
.hk-step.now { color: #fff; border-color: transparent; box-shadow: 0 0 30px rgba(168,85,247,.55);
  background: linear-gradient(135deg, __PURPLE__ 0%, __PURPLE2__ 55%, __PINK__ 100%); animation: hk-pulse 1.7s ease-in-out infinite; }
.hk-link { flex: none; width: 26px; height: 3px; background: __CARD__; }
.hk-link.done { background: rgba(34,197,94,.6); }
@keyframes hk-pulse { 0%, 100% { box-shadow: 0 0 18px rgba(168,85,247,.35); } 50% { box-shadow: 0 0 38px rgba(168,85,247,.75); } }

.hk-min { margin: 8px 0; padding: 26px 28px; background: __BG0__; border: 1px solid __CARD__; border-radius: 20px; }
.hk-head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px 24px; }
.hk-title { margin: 0; padding: 0; font-size: 40px; line-height: 1.15; font-weight: 700; color: #fff; }
.hk-att { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; font-size: 18px; color: __MUTED__; }
.hk-chip { padding: 4px 14px; border-radius: 999px; background: __CARD__; color: #E5E7EB; font-size: 18px; font-weight: 500; }
.hk-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
@media (max-width: 760px) { .hk-cols { grid-template-columns: 1fr; } }
.hk-col { padding: 18px 18px 6px 18px; background: __BG1__; border-radius: 16px; }
.hk-colhead { display: flex; align-items: center; gap: 10px; margin: 0 0 14px 2px; font-size: 28px; font-weight: 700; color: #fff; }
.hk-dot { width: 14px; height: 14px; border-radius: 50%; flex: none; }
.hk-count { padding: 2px 14px; border-radius: 999px; background: __CARD__; color: #fff; font-size: 18px; font-weight: 600; }
.hk-item { display: flex; gap: 14px; align-items: flex-start; margin: 0 0 12px 0; padding: 16px 18px; border-radius: 14px; background: __CARD__; }
.hk-mark { flex: none; width: 26px; margin-top: 2px; font-size: 22px; line-height: 1.3; }
.hk-mark.dec { color: __PURPLE2__; }
.hk-mark.act { color: __PINK__; }
.hk-text { font-size: 26px; line-height: 1.35; font-weight: 500; color: __TEXT__; word-break: break-word; }
.hk-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.hk-pill { display: inline-flex; align-items: center; gap: 8px; padding: 4px 16px 4px 5px; border-radius: 999px; font-size: 19px; font-weight: 600; }
.hk-pill.due { padding-left: 14px; }
.hk-pill.own { background: rgba(139,92,246,.22); color: #D8CCFF; border: 1px solid rgba(139,92,246,.55); }
.hk-pill.due { background: rgba(236,72,153,.16); color: #FBC3DE; border: 1px solid rgba(236,72,153,.55); }
.hk-av { width: 28px; height: 28px; border-radius: 50%; background: __PURPLE__; color: #fff; font-size: 16px; line-height: 28px; text-align: center; }
.hk-empty { margin: 0 0 12px 0; padding: 20px; border: 1px dashed __CARD__; border-radius: 14px; color: __DIM__; font-size: 20px; text-align: center; }
.hk-raw { margin: 8px 0; padding: 22px 26px; background: __BG1__; border: 1px solid __CARD__; border-radius: 16px; }
.hk-raw pre { margin: 0; white-space: pre-wrap; font-family: Consolas, 'Courier New', monospace; font-size: 20px; line-height: 1.5; color: __TEXT__; background: transparent; border: 0; padding: 0; }

.hk-audio { margin: 8px 0; padding: 18px 20px 20px 20px; background: __BG1__; border: 1px solid __CARD__; border-radius: 20px; }
.hk-audio-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin: 0 0 12px 2px; }
.hk-audio-title { font-size: 22px; font-weight: 600; color: #fff; }
.hk-audio img { display: block; width: 100%; height: auto; border-radius: 12px; }
.hk-audio audio { display: block; width: 100%; height: 46px; margin-top: 14px; color-scheme: dark; }
.hk-note { margin-top: 10px; font-size: 16px; color: __MUTED__; }
</style>
"""


def _css():
    out = _CSS
    for key, val in (("BG0", BG0), ("BG1", BG1), ("CARD", CARD), ("PURPLE2", PURPLE2), ("PURPLE", PURPLE), ("PINK", PINK),
                     ("GREEN", GREEN), ("TEXT", TEXT), ("MUTED", MUTED), ("DIM", DIM)):
        out = out.replace(f"__{key}__", val)
    return out


def _esc(text):
    return _html.escape(str(text), quote=True)


# ---------------------------------------------------------------- banner
def banner(title="Who remembers the meeting?", subtitle="Workshop 4 · Deep Learning in Audio Applications"):
    """The big title card at the top of the demo."""
    return HTML(f'{_css()}<div class="hk"><div class="hk-banner"><h1>{_esc(title)}</h1><p>{_esc(subtitle)}</p></div></div>')


# ---------------------------------------------------------------- progress strip
def _progress_html(stage):
    stage = re.sub(r"[^a-z ]", "", str(stage or "").lower()).strip()    # "Listening..." -> "listening"
    if "minutes" in stage:                                              # run_demo_staged says "Writing minutes..."
        stage = "minutes"
    all_done = stage == "done"
    idx = STAGES.index(stage) if stage in STAGES else (len(STAGES) if all_done else -1)
    parts = []
    for i, name in enumerate(STAGES):
        state = "done" if i < idx else ("now" if i == idx else "")
        label = ("✓ " if state == "done" else "") + _STAGE_LABEL[name]
        parts.append(f'<div class="hk-step {state}">{label}</div>')
        if i < len(STAGES) - 1:
            parts.append(f'<div class="hk-link {"done" if i < idx else ""}"></div>')
    return f'{_css()}<div class="hk"><div class="hk-prog">{"".join(parts)}</div></div>'


def progress(stage):
    """A horizontal strip of the four stages with `stage` highlighted (earlier ones green).
    Unknown stages highlight nothing; "done" turns all four green."""
    return HTML(_progress_html(stage))


class ProgressStrip:
    """Shows the strip once, then redraws it in place:

        strip = ProgressStrip("listening")
        strip.update("transcribing")
        strip.done()
    """

    def __init__(self, stage="listening"):
        self.stage = stage
        self.handle = display(HTML(_progress_html(stage)), display_id=True)

    def update(self, stage):
        self.stage = stage
        if self.handle is not None:                    # plain Python has no display handle: nothing to update
            self.handle.update(HTML(_progress_html(stage)))
        return self

    def done(self):
        return self.update("done")


# ---------------------------------------------------------------- minutes cards
def parse_minutes(minutes_md):
    """generate_minutes() Markdown -> {"title", "attendees", "decisions": [text], "actions": [{"text","owner","due"}]}.
    Item text is kept exactly as written; only the trailing "-- Owner (due X)" is split off. Raises ValueError
    when no Decisions / Action Items section is found."""
    title, attendees, current = "Meeting Minutes", [], None
    found = {"decisions": False, "actions": False}
    decisions, actions = [], []
    for raw in str(minutes_md).splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# "):
            title = line[2:].strip() or title
            continue
        m = re.match(r"\*\*attendees:?\*\*:?\s*(.*)$", line, re.I)
        if m:
            attendees = [a.strip() for a in m.group(1).split(",") if a.strip()]
            continue
        if line.startswith("## "):
            head = line[3:].strip().lower()
            current = "decisions" if head.startswith("decision") else ("actions" if head.startswith("action") else None)
            if current:
                found[current] = True
            continue
        if current and line.startswith("- "):
            body = line[2:].strip()
            if current == "actions":
                body = re.sub(r"^\[[ xX]?\]\s*", "", body)
            if body.lower() == "(none)" or not body:
                continue
            if current == "decisions":
                decisions.append(body)
            else:
                text, sep, rest = body.rpartition(" -- ")
                owner = due = None
                if sep and text:
                    m2 = re.match(r"^(.*?)(?:\s*\(due ([^)]*)\))?$", rest.strip())
                    owner = (m2.group(1).strip() if m2 else rest.strip()) or None
                    due = (m2.group(2) or "").strip() if m2 else None
                    due = due or None
                    if owner and owner.lower() == "none":
                        owner = None
                else:
                    text = body
                actions.append({"text": text, "owner": owner, "due": due})
    if not (found["decisions"] or found["actions"]):
        raise ValueError("no Decisions or Action Items section found")
    return {"title": title, "attendees": attendees, "decisions": decisions, "actions": actions}


def _cards_html(items, mark_cls, mark, empty_text):
    if not items:
        return f'<div class="hk-empty">{_esc(empty_text)}</div>'
    out = []
    for it in items:
        if isinstance(it, str):
            it = {"text": it, "owner": None, "due": None}
        pills = ""
        if it["owner"]:
            pills += f'<span class="hk-pill own"><span class="hk-av">{_esc(it["owner"][:1].upper())}</span>{_esc(it["owner"])}</span>'
        if it["due"]:
            pills += f'<span class="hk-pill due">due {_esc(it["due"])}</span>'
        pills = f'<div class="hk-pills">{pills}</div>' if pills else ""
        out.append(f'<div class="hk-item"><div class="hk-mark {mark_cls}">{mark}</div>'
                   f'<div><div class="hk-text">{_esc(it["text"])}</div>{pills}</div></div>')
    return "".join(out)


def minutes_cards(minutes_md):
    """generate_minutes() Markdown as two columns of cards: Decisions and Action items (owner and due date
    as pills). Never raises: if the text cannot be parsed it is shown as-is in a dark card."""
    try:
        data = parse_minutes(minutes_md)
        chips = "".join(f'<span class="hk-chip">{_esc(a)}</span>' for a in data["attendees"])
        att = f'<div class="hk-att">Attendees {chips}</div>' if chips else ""
        dec = _cards_html(data["decisions"], "dec", "◆", "No decisions found")
        act = _cards_html(data["actions"], "act", "☐", "No action items found")
        body = (f'<div class="hk-min"><div class="hk-head"><h2 class="hk-title">{_esc(data["title"])}</h2>{att}</div>'
                f'<div class="hk-cols">'
                f'<div class="hk-col"><div class="hk-colhead"><span class="hk-dot" style="background:{PURPLE2}"></span>Decisions'
                f'<span class="hk-count">{len(data["decisions"])}</span></div>{dec}</div>'
                f'<div class="hk-col"><div class="hk-colhead"><span class="hk-dot" style="background:{PINK}"></span>Action items'
                f'<span class="hk-count">{len(data["actions"])}</span></div>{act}</div></div></div>')
    except Exception:                                  # a facilitator must never see a traceback here
        body = f'<div class="hk-raw"><pre>{_esc(minutes_md)}</pre></div>'
    return HTML(f'{_css()}<div class="hk">{body}</div>')


# ---------------------------------------------------------------- audio panel
_MIME = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".mp4": "audio/mp4", ".ogg": "audio/ogg",
         ".webm": "audio/webm", ".flac": "audio/flac", ".aac": "audio/aac"}
_MAX_EMBED_BYTES = 8_000_000


def _load_16k(path):
    """Mono float32 at 16 kHz in [-1, 1], through whisper.load_audio (ffmpeg); plain `wave` as a fallback."""
    try:
        import whisper
        return whisper.load_audio(path)
    except Exception:
        import numpy as np
        with wave.open(path, "rb") as w:
            raw, rate, ch, width = w.readframes(w.getnframes()), w.getframerate(), w.getnchannels(), w.getsampwidth()
        if width != 2:
            raise
        x = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)
        if rate != 16000:
            x = np.interp(np.arange(0, len(x) * 16000 / rate), np.arange(len(x)) * 16000 / rate, x).astype("float32")
        return x


def _figure_png(x):
    """Purple waveform above a magma spectrogram, dark background, seconds on the time axis."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    rate, dur = 16000, max(len(x) / 16000.0, 0.01)
    fig = Figure(figsize=(11, 4.8), dpi=130, facecolor=BG1)
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.7], hspace=0.12)
    a1 = fig.add_subplot(gs[0])
    a2 = fig.add_subplot(gs[1], sharex=a1)

    block = max(1, len(x) // 1800)                      # min/max envelope: fast and still crisp for long clips
    n = len(x) // block
    if n:
        env = x[: n * block].reshape(n, block)
        t = (np.arange(n) + 0.5) * block / rate
        a1.fill_between(t, env.min(axis=1), env.max(axis=1), color=PURPLE2, linewidth=0)
    a1.set_ylim(-1.0, 1.0)
    a1.set_yticks([])
    a1.set_ylabel("loudness", color=MUTED, fontsize=12)
    a1.tick_params(labelbottom=False, length=0)

    xs = x + np.random.default_rng(0).normal(0, 2e-5, len(x))      # a tiny floor: digital silence has no log
    a2.specgram(xs, NFFT=512, Fs=rate, noverlap=384, cmap="magma", vmin=-110, vmax=-30)
    a2.set_ylim(0, 5000)
    a2.set_ylabel("pitch (Hz)", color=MUTED, fontsize=12)
    a2.set_xlabel("time (seconds)", color=MUTED, fontsize=12)
    a2.set_xlim(0, dur)
    for ax in (a1, a2):
        ax.set_facecolor(BG0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(CARD)
        ax.tick_params(colors=MUTED, labelsize=11)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG1, bbox_inches="tight")
    return buf.getvalue()


def _wav_bytes_16k(x):
    import numpy as np
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def audio_panel(path, title="Recording"):
    """A dark card: waveform and spectrogram picture (embedded PNG), then an <audio controls> player with the
    file embedded as base64, so it plays in a saved notebook. Falls back to the player alone if the picture fails."""
    path = str(path)
    try:
        x = _load_16k(path)
    except Exception as err:
        return HTML(f'{_css()}<div class="hk"><div class="hk-audio"><div class="hk-audio-title">{_esc(title)}</div>'
                    f'<div class="hk-note">Could not read this audio file ({_esc(type(err).__name__)}).</div></div></div>')
    seconds = len(x) / 16000.0
    picture = ""
    try:
        png = base64.b64encode(_figure_png(x)).decode("ascii")
        picture = f'<img alt="Waveform and spectrogram of {_esc(title)}" src="data:image/png;base64,{png}">'
    except Exception:
        picture = '<div class="hk-note">The waveform picture could not be drawn; the player below still works.</div>'
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in _MIME and os.path.getsize(path) <= _MAX_EMBED_BYTES:
            data, mime = open(path, "rb").read(), _MIME[ext]
        else:                                           # unknown or large file: embed a compact 16 kHz mono copy
            data, mime = _wav_bytes_16k(x), "audio/wav"
        player = f'<audio controls preload="metadata" src="data:{mime};base64,{base64.b64encode(data).decode("ascii")}"></audio>'
    except Exception:
        player = '<div class="hk-note">The audio could not be embedded.</div>'
    return HTML(f'{_css()}<div class="hk"><div class="hk-audio"><div class="hk-audio-head">'
                f'<span class="hk-audio-title">{_esc(title)}</span><span class="hk-chip">{seconds:.1f} s</span></div>'
                f'{picture}{player}</div></div>')
