"""moves.py -- the move library the director picks from (code does the motion).

Every move is a pure function of local time u (seconds since the beat started)
and a seeded rng, so the same edit.json always renders the same video, while
different seeds give different overshoots, angles and sides.

  ENTRANCES  where the avatar comes from at the start of a beat
  SPOTS      where it lands (scale + anchor on the 1080x1920 canvas)
  CAMERA     shake / zoom punches applied to the whole frame
  ARMS       arm poses and loops (angles for puppet.Puppet)
"""
from __future__ import annotations

import math

import numpy as np

W, H = 1080, 1920


def clamp01(x):
    return min(1.0, max(0.0, x))


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def back(x, s=1.7):
    """Ease-out that swings past the target and settles."""
    x = clamp01(x) - 1
    return 1 + x * x * ((s + 1) * x + s)


def bounce(x):
    x = clamp01(x)
    for a, b, c in ((1 / 2.75, 0, 0), (2 / 2.75, 1.5 / 2.75, 0.75), (2.5 / 2.75, 2.25 / 2.75, 0.9375)):
        if x < a:
            return 7.5625 * (x - b) ** 2 + c if b else 7.5625 * x * x
    return 7.5625 * (x - 2.625 / 2.75) ** 2 + 0.984375


# ── where the avatar stands: (scale, x of body centre, y of the feet) ───────
SPOTS = {
    "center_big":    (1.55, W * 0.50, H * 1.28),   # waist-up, fills the frame
    "center":        (1.15, W * 0.50, H * 1.02),
    "corner_left":   (0.75, W * 0.27, H * 1.00),
    "corner_right":  (0.75, W * 0.73, H * 1.00),
    "half_left":     (1.30, W * 0.12, H * 1.15),   # half off the left edge
    "half_right":    (1.30, W * 0.88, H * 1.15),
    "tiny_top":      (0.45, W * 0.70, H * 0.42),   # small, up beside the caption
}


# ── entrances: offset (dx, dy), rotation (deg), squash (sx, sy) over u ──────
def enter(kind: str, u: float, rng: np.random.Generator) -> tuple[float, float, float, float, float]:
    side = -1 if rng.random() < 0.5 else 1
    if kind == "drop_from_sky":
        k = bounce(u / 0.7)
        dy = -(1 - k) * H * 1.3
        land = max(0.0, 1 - abs(u - 0.26) / 0.1) if u < 0.7 else 0  # squash on first impact
        return 0, dy, 0, 1 + 0.18 * land, 1 - 0.2 * land
    if kind in ("pop_in_left", "pop_in_right"):
        s = -1 if kind == "pop_in_left" else 1
        k = back(u / 0.45, 2.4)
        return s * (1 - k) * W * 0.9, 0, s * (1 - k) * 25, 1, 1
    if kind in ("slide_in_left", "slide_in_right"):
        s = -1 if kind == "slide_in_left" else 1
        k = smooth(u / 0.35)
        lean = math.sin(clamp01(u / 0.6) * math.pi) * 12 * -s
        return s * (1 - k) * W * 1.1, 0, lean, 1, 1
    if kind == "peek_from_bottom":
        k = back(u / 0.5, 1.4)
        return 0, (1 - k) * H * 0.7, 0, 1, 1
    if kind == "spin_in":
        k = back(u / 0.55, 1.2)
        return 0, 0, (1 - k) * 360 * side, k, k
    return 0, 0, 0, 1, 1  # "none": already there


ENTRANCES = ("drop_from_sky", "pop_in_left", "pop_in_right", "slide_in_left", "slide_in_right",
             "peek_from_bottom", "spin_in", "none")


# ── camera: frame offset (dx, dy), zoom, rotation (deg) ──────────────────────
def camera(kind: str, u: float, rng_seed: int) -> tuple[float, float, float, float]:
    if kind == "shake_heavy" or kind == "shake_light":
        amp = 38 if kind == "shake_heavy" else 12
        decay = math.exp(-u * 3.5)
        r = np.random.default_rng(rng_seed + int(u * 30))
        return amp * decay * (r.random() * 2 - 1), amp * decay * (r.random() * 2 - 1), 1, 2 * decay * (r.random() * 2 - 1)
    if kind == "zoom_punch":
        z = 1 + 0.22 * (math.exp(-((u - 0.08) / 0.12) ** 2) if u > 0.08 else smooth(u / 0.08))
        return 0, 0, z, 0
    if kind == "slow_push":
        return 0, 0, 1 + 0.06 * u, 0
    if kind == "whip_pan":
        k = smooth(u / 0.25)
        return (1 - k) * W * 0.6, 0, 1, (1 - k) * -8
    return 0, 0, 1, 0


CAMERAS = ("zoom_punch", "shake_heavy", "shake_light", "slow_push", "whip_pan", "none")


# ── arms: angles for puppet.Puppet (+ shoulder = out, + elbow = forearm up) ─
def arms(kind: str, u: float) -> dict:
    k = back(u / 0.35)
    if kind == "wave":
        return {"B_shoulder": 115 * k, "B_elbow": 70 * k + 25 * math.sin(u * 14) * smooth(u / 0.5),
                "B_wrist": 10 * math.sin(u * 14), "B_hand_kind": "open"}
    if kind == "fan_spin":   # forearms whirl like a fan: dance
        spin = u * 720
        return {"A_shoulder": 90 * k, "B_shoulder": 90 * k, "A_elbow": spin, "B_elbow": spin + 180,
                "A_hand_kind": "fist", "B_hand_kind": "fist"}
    if kind == "flex":
        pump = back(max(0.0, u - 0.2) / 0.3) * (0.85 + 0.15 * math.sin(u * 9))  # biceps swell after the arms land
        return {"A_shoulder": 90 * k, "B_shoulder": 90 * k, "A_elbow": 125 * k, "B_elbow": 125 * k,
                "A_bulge": pump, "B_bulge": pump, "A_hand_kind": "fist", "B_hand_kind": "fist"}
    if kind == "point_left":
        return {"A_shoulder": 75 * k, "A_elbow": 8 * k, "A_hand_kind": "point"}
    if kind == "point_right":
        return {"B_shoulder": 75 * k, "B_elbow": 8 * k, "B_hand_kind": "point"}
    if kind == "both_up":
        w = 8 * math.sin(u * 10)
        return {"A_shoulder": 160 * k + w, "B_shoulder": 160 * k - w, "A_elbow": 10, "B_elbow": 10,
                "A_hand_kind": "open", "B_hand_kind": "open"}
    if kind == "shrug":
        return {"A_shoulder": 25 * k, "A_elbow": 75 * k, "A_wrist": 60 * k, "B_shoulder": 25 * k,
                "B_elbow": 75 * k, "B_wrist": 60 * k, "A_hand_kind": "open", "B_hand_kind": "open"}
    if kind == "rock_on":
        pump = 10 * math.sin(u * 12)
        return {"B_shoulder": 140 * k + pump, "B_elbow": 30, "B_hand_kind": "rock", "A_hand_kind": "fist"}
    if kind == "thumbs_up":
        return {"B_shoulder": 70 * k, "B_elbow": 110 * k, "B_hand_kind": "thumbs"}
    return {"A_hand_kind": "open", "B_hand_kind": "open"}  # idle


ARMS = ("idle", "wave", "fan_spin", "flex", "point_left", "point_right", "both_up", "shrug", "rock_on", "thumbs_up")
