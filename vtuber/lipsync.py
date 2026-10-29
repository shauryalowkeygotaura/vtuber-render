"""lipsync.py -- mouth shapes straight from the audio, no phoneme recogniser.

WHY NOT RHUBARB
---------------
Rhubarb's recogniser is English-only and Shaurya talks Hinglish. A wrong
phoneme guess looks worse than no guess, so this reads the two things that
are language-free: how LOUD the voice is (how open the mouth is) and where
its energy sits in the spectrum (what shape the mouth is).

  - F1 band (250-900 Hz) heavy, little above 2 kHz  -> open back vowel  "aa"/"oh"
  - energy pushed up into 2-4 kHz (high F2)         -> spread vowel     "ee"/"ih"
  - everything low and quiet in the middle           -> rounded         "ou"

That is the same trick VTuber apps use for live mic lip sync; at 30 fps nobody
can tell it from a phoneme track.
"""
from __future__ import annotations

import subprocess

import numpy as np

SR = 16_000
VISEMES = ("aa", "ih", "ou", "ee", "oh")


def load_audio(path: str) -> np.ndarray:
    """Any audio/video file -> mono float32 at 16 kHz, via ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR),
         "-f", "s16le", "-"],
        capture_output=True, check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg could not decode {path}: {proc.stderr.decode(errors='replace')[:300]}")
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _band(spec: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    m = (freqs >= lo) & (freqs < hi)
    return spec[:, m].sum(axis=1)


def _smooth(x: np.ndarray, attack: float, release: float) -> np.ndarray:
    """Fast to open, slower to close: a mouth snapping shut every frame flickers."""
    out = np.empty_like(x)
    prev = 0.0
    for i, v in enumerate(x):
        k = attack if v > prev else release
        prev = prev + (v - prev) * k
        out[i] = prev
    return out


def visemes(audio: np.ndarray, fps: int) -> dict:
    """Per-frame viseme weights (0..1) plus a loudness track for gestures."""
    # Exact sample position per frame: an integer hop (16000 // 30 = 533)
    # drifts the mouth ~40 ms behind the voice per minute of audio.
    win = int(SR * 0.04)
    n_frames = max(1, int(np.ceil(len(audio) * fps / SR)))
    padded = np.pad(audio, (win // 2, win + SR // fps + 1))
    window = np.hanning(win).astype(np.float32)
    starts = (np.arange(n_frames) * SR / fps).round().astype(int)
    frames = np.stack([padded[s:s + win] * window for s in starts])
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(win, 1 / SR)

    rms = np.sqrt((frames ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-6)
    # Normalise loudness to this recording: phone mic and studio mic both
    # end up using the full mouth range.
    voiced = db[db > np.percentile(db, 20)]
    ceil = np.percentile(voiced, 95) if voiced.size else db.max()
    # at least a 24 dB range, or a steady voice (or a held note) never opens
    floor = min(np.percentile(voiced, 10) if voiced.size else db.min(), ceil - 24)
    open_ = np.clip((db - floor) / max(ceil - floor, 1e-3), 0, 1)
    open_[db < floor] = 0.0

    low = _band(spec, freqs, 250, 900)
    mid = _band(spec, freqs, 900, 2000)
    high = _band(spec, freqs, 2000, 4000)
    total = low + mid + high + 1e-9
    front = high / total          # spread lips
    back = low / total            # open jaw
    round_ = np.clip(1 - front * 3 - (mid / total), 0, 1)

    w = {
        "aa": open_ * np.clip(back * 1.6, 0, 1),
        "ee": open_ * np.clip(front * 3.0, 0, 1),
        "ih": open_ * np.clip(front * 2.0, 0, 1) * 0.6,
        "oh": open_ * np.clip(back * 1.2, 0, 1) * np.clip(round_ * 1.5, 0, 1),
        "ou": open_ * np.clip(round_ * 1.2, 0, 1) * 0.7,
    }
    # Share one "jaw budget" so shapes don't stack into a gaping mouth.
    stacked = np.stack([w[v] for v in VISEMES])
    s = stacked.sum(axis=0)
    scale = np.where(s > 1, 1 / np.maximum(s, 1e-9), 1)
    out = {}
    for i, v in enumerate(VISEMES):
        out[v] = np.round(_smooth(stacked[i] * scale, 0.65, 0.35), 3).tolist()
    out["energy"] = np.round(_smooth(open_, 0.3, 0.08), 3).tolist()
    out["frames"] = n_frames
    return out
