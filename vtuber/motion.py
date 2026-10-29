"""motion.py -- turn audio features + directed beats into one pose per frame.

All the "acting" lives here, in plain numbers, so the browser side is a dumb
puppet that applies a frame and takes a picture. That keeps the render
deterministic (same input, same video) and lets this file be tested without a
GPU.

Layers, summed per frame:
  idle      slow seeded sway + breathing: a still head reads as a frozen video
  beat      a small nod on each loudness onset: how people punctuate speech
  gesture   what performance.py asked for on that line (nod, tilt, ...)
  face      expression weights eased between beats, blinks on a random clock
"""
from __future__ import annotations

import math
import random

import numpy as np

from .performance import EXPRESSIONS


def _ease(x: np.ndarray, tau_frames: float) -> np.ndarray:
    k = 1 - math.exp(-1 / max(tau_frames, 1e-6))
    out = np.empty_like(x)
    prev = x[0]
    for i, v in enumerate(x):
        prev = prev + (v - prev) * k
        out[i] = prev
    return out


def _env(n: int, start: int, length: int, attack: int, release: int) -> np.ndarray:
    """0..1 hold envelope with eased edges."""
    e = np.zeros(n)
    for i in range(max(0, start), min(n, start + length)):
        k = i - start
        a = min(1.0, k / max(attack, 1))
        r = min(1.0, (length - k) / max(release, 1))
        e[i] = 0.5 - 0.5 * math.cos(math.pi * min(a, r))
    return e


def build(lip: dict, beats: list[dict], fps: int, seed: int = 7) -> dict:
    n = lip["frames"]
    t = np.arange(n) / fps
    rng = random.Random(seed)
    energy = np.asarray(lip["energy"])

    # idle sway: three incommensurate sines per axis never visibly loop
    def sway(amp: float) -> np.ndarray:
        s = np.zeros(n)
        for f in (0.11, 0.23, 0.37):
            s += np.sin(2 * np.pi * f * rng.uniform(0.8, 1.2) * t + rng.uniform(0, 6.28))
        return s / 3 * amp

    talk = 1 + 1.5 * energy
    pitch = sway(0.035) * talk
    yaw = sway(0.05) * talk
    roll = sway(0.025) * talk
    spine = 0.012 * np.sin(2 * np.pi * 0.24 * t)  # breathing

    # beat nods on loudness onsets
    d = np.diff(energy, prepend=0)
    last = -fps
    for i in range(1, n - 1):
        if d[i] > 0.12 and d[i] >= d[i - 1] and d[i] >= d[i + 1] and i - last > fps * 0.35:
            last = i
            length = int(fps * 0.35)
            pitch += 0.05 * _env(n, i, length, 3, length - 3)

    look_x = np.zeros(n)
    weights = {e: np.zeros(n) for e in EXPRESSIONS if e != "neutral"}
    for b in beats:
        s = int(b["start"] * fps)
        L = max(1, int((b["end"] - b["start"]) * fps))
        k = b["intensity"]
        g = b["gesture"]
        if g == "nod":
            w = min(L, int(fps * 0.9))
            pitch += 0.10 * k * np.sin(np.clip((np.arange(n) - s) / w, 0, 1) * 4 * np.pi) * _env(n, s, w, 2, 6)
        elif g == "shake":
            w = min(L, int(fps * 1.0))
            yaw += 0.13 * k * np.sin(np.clip((np.arange(n) - s) / w, 0, 1) * 4 * np.pi) * _env(n, s, w, 2, 6)
        elif g == "tilt":
            roll += 0.13 * k * rng.choice((-1, 1)) * _env(n, s, L, 8, 10)
        elif g == "lean_in":
            spine += 0.07 * k * _env(n, s, L, 10, 12)
            pitch += 0.05 * k * _env(n, s, L, 10, 12)
        elif g == "look_away":
            w = min(L, int(fps * 1.4))
            side = rng.choice((-1, 1))
            yaw += 0.22 * k * side * _env(n, s, w, 6, 10)
            look_x += 0.8 * side * _env(n, s, w, 4, 8)
        if b["expression"] in weights:
            weights[b["expression"]][s:s + L] = 0.85 * k

    face = {e: _ease(w, fps * 0.25) for e, w in weights.items()}
    # A wide-open smile fights the mouth shapes: give the jaw priority.
    open_ = np.asarray(lip["aa"]) + np.asarray(lip["oh"])
    face["happy"] = face["happy"] * (1 - 0.45 * np.clip(open_, 0, 1))

    blink = np.zeros(n)
    i = int(fps * rng.uniform(0.5, 2.0))
    while i < n:
        dur = int(fps * 0.16)
        for k in range(dur):
            if i + k < n:
                blink[i + k] = max(blink[i + k], 1 - abs(2 * k / dur - 1))
        if rng.random() < 0.18:  # occasional double blink
            i += dur + int(fps * 0.12)
            continue
        i += int(fps * rng.uniform(2.2, 5.5))
    blink *= 1 - np.clip(face["happy"] * 1.2, 0, 1)  # happy already squints

    r3 = lambda a: np.round(a, 4).tolist()  # noqa: E731
    return {
        "fps": fps,
        "frames": n,
        "head": {"pitch": r3(pitch), "yaw": r3(yaw), "roll": r3(roll)},
        "spine": r3(spine),
        "lookX": r3(look_x),
        "visemes": {v: lip[v] for v in ("aa", "ih", "ou", "ee", "oh")},
        "face": {e: r3(w) for e, w in face.items()},
        "blink": r3(blink),
    }
