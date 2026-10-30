"""mechhand.py -- robotic hands drawn in code, crisp at any size.

hand(kind, L, mirror) -> (RGBA image, wrist point in that image)
The hand hangs DOWN from the wrist (like the drawing's rest pose), palm to the
viewer, thumb toward +x. mirror=True flips it for the other arm. The puppet then
rotates it with the forearm, so one sprite covers every angle.

kinds: open, fist, point, peace, thumbs, rock, claw
"""
from __future__ import annotations

import math

from PIL import Image, ImageDraw

SS = 4
INK = (22, 18, 22, 255)
PLATE = (92, 101, 116, 255)
PLATE_HI = (150, 160, 176, 255)
DARK = (46, 50, 60, 255)
GLOW = (0, 229, 255, 255)

KINDS = ("open", "fist", "point", "peace", "thumbs", "rock", "claw")
# per finger (index, middle, ring, pinky): extension 0 = curled .. 1 = straight, and spread angle (deg)
FINGERS = {
    "open":   ((1, -10), (1, -3), (1, 4), (1, 11)),
    "fist":   ((0, 0), (0, 0), (0, 0), (0, 0)),
    "point":  ((1, -2), (0, 0), (0, 0), (0, 0)),
    "peace":  ((1, -14), (1, 6), (0, 0), (0, 0)),
    "thumbs": ((0, 0), (0, 0), (0, 0), (0, 0)),
    "rock":   ((1, -8), (0, 0), (0, 0), (1, 10)),
    "claw":   ((0.55, -12), (0.55, -4), (0.55, 4), (0.55, 12)),
}
THUMB = {"open": (1, 50), "fist": (0, 0), "point": (0.3, 20), "peace": (0, 0),
         "thumbs": (1, 150), "rock": (1, 55), "claw": (0.6, 40)}


def _capsule(d, a, b, w, fill):
    d.line([a, b], fill=INK, width=int(w + 2 * SS * 1.6))
    for p in (a, b):
        r = w / 2 + SS * 1.6
        d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=INK)
    d.line([a, b], fill=fill, width=int(w))
    for p in (a, b):
        r = w / 2
        d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=fill)


def _joint(d, p, r):
    d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=DARK, outline=INK, width=SS)
    g = r * 0.45
    d.ellipse((p[0] - g, p[1] - g, p[0] + g, p[1] + g), fill=GLOW)


def hand(kind: str, L: float, mirror: bool = False) -> tuple[Image.Image, tuple[float, float]]:
    if kind not in KINDS:
        kind = "open"
    S = int(L * 2.2)
    im = Image.new("RGBA", (S * SS, S * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    u = L * SS                        # one hand-length in supersampled px
    cx, wy = S * SS / 2, S * SS * 0.30  # wrist point
    # cuff
    cw, ch = u * 0.50, u * 0.16
    d.rounded_rectangle((cx - cw / 2, wy - ch * 0.4, cx + cw / 2, wy + ch * 0.6), radius=u * 0.05,
                        fill=DARK, outline=INK, width=int(SS * 2))
    d.line((cx - cw * 0.4, wy + ch * 0.1, cx + cw * 0.4, wy + ch * 0.1), fill=GLOW, width=int(SS * 2.5))
    # palm
    pw, ph = u * 0.60, u * 0.42
    py0 = wy + ch * 0.55
    d.rounded_rectangle((cx - pw / 2, py0, cx + pw / 2, py0 + ph), radius=u * 0.10,
                        fill=PLATE, outline=INK, width=int(SS * 2.2))
    d.rounded_rectangle((cx - pw * 0.32, py0 + ph * 0.18, cx + pw * 0.32, py0 + ph * 0.62), radius=u * 0.05,
                        fill=PLATE_HI)
    d.ellipse((cx - u * 0.05, py0 + ph * 0.32, cx + u * 0.05, py0 + ph * 0.32 + u * 0.10), fill=GLOW, outline=INK,
              width=SS)
    # fingers along the bottom edge, index on the thumb side (+x)
    fw = u * 0.115
    base_y = py0 + ph - fw * 0.3
    xs = [cx + pw * f for f in (0.33, 0.11, -0.11, -0.33)]
    seg = u * 0.27
    for (ext, spread), x in zip(FINGERS[kind], xs):
        a = (x, base_y)
        if ext <= 0.05:  # curled: a knuckle stub folded toward the palm
            _capsule(d, (x, base_y - fw * 0.1), (x, base_y + fw * 0.45), fw, PLATE)
            _joint(d, (x, base_y + fw * 0.1), fw * 0.32)
            continue
        ang = math.radians(90 - spread)       # 90 = straight down
        b = (a[0] + math.cos(ang) * seg, a[1] + math.sin(ang) * seg)
        bend = (1 - ext) * 70
        ang2 = math.radians(90 - spread - bend * (1 if x > cx else -1) * 0 + bend)
        c = (b[0] + math.cos(ang2) * seg * 0.85, b[1] + math.sin(ang2) * seg * 0.85)
        _capsule(d, a, b, fw, PLATE)
        _capsule(d, b, c, fw * 0.9, PLATE_HI)
        _joint(d, a, fw * 0.36)
        _joint(d, b, fw * 0.32)
    # thumb from the palm's +x side
    ext, ang_deg = THUMB[kind]
    t0 = (cx + pw * 0.47, py0 + ph * 0.30)
    if ext <= 0.05:
        t1 = (cx + pw * 0.20, py0 + ph * 0.80)   # folded across the fingers
        _capsule(d, t0, t1, fw * 1.05, PLATE_HI)
        _joint(d, t0, fw * 0.36)
    else:
        a1 = math.radians(90 - ang_deg)  # 0 = sideways, 90 = down, negative = up toward the wrist
        L1 = seg * (0.9 + 0.4 * ext)
        t1 = (t0[0] + math.cos(a1) * L1 * 0.6 + u * 0.02, t0[1] + math.sin(a1) * L1 * 0.6)
        t2 = (t1[0] + math.cos(a1) * L1 * 0.6, t1[1] + math.sin(a1) * L1 * 0.6)
        if ang_deg < 60:  # angle out-and-down from the palm side
            t1 = (t0[0] + u * 0.10, t0[1] + u * 0.12)
            t2 = (t1[0] + math.cos(math.radians(ang_deg)) * seg * ext, t1[1] + math.sin(math.radians(ang_deg)) * seg * ext)
        _capsule(d, t0, t1, fw * 1.05, PLATE)
        _capsule(d, t1, t2, fw * 0.95, PLATE_HI)
        _joint(d, t0, fw * 0.38)
        _joint(d, t1, fw * 0.32)
    im = im.resize((S, S), Image.LANCZOS)
    wrist = (S / 2, S * 0.30)
    if mirror:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
        wrist = (S - wrist[0], wrist[1])
    return im, wrist
