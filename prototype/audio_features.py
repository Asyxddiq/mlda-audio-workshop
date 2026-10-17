"""
Audio -> numbers: the first step before an RNN can read speech.

    from audio_features import mfcc
    features = mfcc(samples)          # (frames, 13)

`samples` are floats at 16 kHz (whisper.load_audio gives exactly that). Every 10 ms we look at the
last 25 ms of sound and keep 13 numbers that describe its colour (the standard "MFCC" features).
The RNN in rnn_ctc_demo.py reads one such row per time step. Plain NumPy.
"""
import numpy as np

RATE = 16000


def mfcc(x, sr=RATE, n_fft=400, hop=160, n_mels=26, n_ceps=13):
    """Audio samples -> (frames, 13) numbers. 25 ms windows every 10 ms, mel filterbank, log, DCT."""
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])                         # pre-emphasis: boost the highs
    n = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    power = np.abs(np.fft.rfft(x[idx] * np.hamming(n_fft), 512)) ** 2  # how much of each frequency
    mel = lambda f: 2595 * np.log10(1 + f / 700)                       # ears hear low pitches in more detail
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = inv(np.linspace(mel(0), mel(sr / 2), n_mels + 2))
    bins = np.floor((512 + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, 257))
    for m in range(1, n_mels + 1):
        a, b, c = bins[m - 1], bins[m], bins[m + 1]
        fb[m - 1, a:b] = (np.arange(a, b) - a) / max(b - a, 1)
        fb[m - 1, b:c] = (c - np.arange(b, c)) / max(c - b, 1)
    logmel = np.log(power @ fb.T + 1e-10)
    k = np.arange(n_ceps)[None, :]
    dct = np.cos(np.pi / n_mels * (np.arange(n_mels)[:, None] + 0.5) * k)
    return logmel @ dct


if __name__ == "__main__":
    t = np.arange(RATE) / RATE                                          # one second of a 440 Hz tone
    f = mfcc(np.sin(2 * np.pi * 440 * t))
    assert f.shape == (98, 13) and np.isfinite(f).all()
    print("1 s of audio ->", f.shape, "(98 frames x 13 numbers): OK")
