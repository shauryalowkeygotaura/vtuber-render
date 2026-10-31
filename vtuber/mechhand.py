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
# swap-in tool parts (the director picks these by meaning: build -> drill, announce -> megaphone, ...)
TOOLS = {"drill": "build, fix, grind, work, code, hard", "megaphone": "announce, listen, loud, attention, follow",
         "rocket": "fast, launch, growth, 10x, blew up, viral", "wrench": "fix, debug, repair, tweak"}
FLAME = (255, 150, 30, 255)
FLAME_HI = (255, 236, 120, 255)
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


def hand(kind: str, L: float, mirror: bool = False, phase: int = 0) -> tuple[Image.Image, tuple[float, float]]:
    """phase (0-3) animates tools: the drill bit's stripes turn, the rocket flame flickers."""
    if kind not in KINDS and kind not in TOOLS:
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
    if kind in TOOLS:
        _tool(d, kind, cx, wy + ch * 0.55, u, phase)
        return _finish(im, S, mirror)
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
    return _finish(im, S, mirror)


def _finish(im: Image.Image, S: int, mirror: bool):
    im = im.resize((S, S), Image.LANCZOS)
    wrist = (S / 2, S * 0.30)
    if mirror:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
        wrist = (S - wrist[0], wrist[1])
    return im, wrist


def _tool(d, kind: str, cx: float, y0: float, u: float, phase: int):
    """Tool parts hang from the cuff, pointing down the forearm's line."""
    lw = int(SS * 2.2)
    if kind == "drill":
        bw, bh = u * 0.42, u * 0.30
        d.rounded_rectangle((cx - bw / 2, y0, cx + bw / 2, y0 + bh), radius=u * 0.06, fill=(240, 196, 40, 255),
                            outline=INK, width=lw)
        d.rectangle((cx - bw * 0.3, y0 + bh * 0.35, cx + bw * 0.3, y0 + bh * 0.6), fill=DARK)
        cy = y0 + bh
        tip = cy + u * 0.62
        d.polygon([(cx - bw * 0.36, cy), (cx + bw * 0.36, cy), (cx, tip)], fill=PLATE_HI, outline=INK)
        d.line([(cx - bw * 0.36, cy), (cx, tip), (cx + bw * 0.36, cy)], fill=INK, width=lw)
        for j in range(5):  # spiral flutes, shifted by phase so it spins
            f = (j + phase / 4) / 5
            yy = cy + (tip - cy) * f
            half = bw * 0.36 * (1 - f)
            d.line([(cx - half, yy), (cx + half, yy + u * 0.07)], fill=DARK, width=int(SS * 1.6))
    elif kind == "wrench":
        sw = u * 0.13
        d.rounded_rectangle((cx - sw / 2, y0, cx + sw / 2, y0 + u * 0.62), radius=sw / 2, fill=PLATE_HI,
                            outline=INK, width=lw)
        hy = y0 + u * 0.62
        r = u * 0.2
        d.ellipse((cx - r, hy - r * 0.4, cx + r, hy + r * 1.6), fill=PLATE_HI, outline=INK, width=lw)
        d.rectangle((cx - r * 0.38, hy + r * 0.4, cx + r * 0.38, hy + r * 1.8), fill=(0, 0, 0, 0))
        d.line([(cx - r * 0.38, hy + r * 0.45), (cx - r * 0.38, hy + r * 1.5)], fill=INK, width=lw)
        d.line([(cx + r * 0.38, hy + r * 0.45), (cx + r * 0.38, hy + r * 1.5)], fill=INK, width=lw)
        d.line([(cx - r * 0.38, hy + r * 0.45), (cx + r * 0.38, hy + r * 0.45)], fill=INK, width=lw)
    elif kind == "megaphone":
        hw = u * 0.12
        d.rounded_rectangle((cx - hw, y0, cx + hw, y0 + u * 0.18), radius=hw * 0.5, fill=DARK, outline=INK, width=lw)
        top, bot = y0 + u * 0.16, y0 + u * 0.78
        d.polygon([(cx - u * 0.10, top), (cx + u * 0.10, top), (cx + u * 0.34, bot), (cx - u * 0.34, bot)],
                  fill=(232, 72, 60, 255), outline=INK)
        d.line([(cx - u * 0.10, top), (cx - u * 0.34, bot)], fill=INK, width=lw)
        d.line([(cx + u * 0.10, top), (cx + u * 0.34, bot)], fill=INK, width=lw)
        d.ellipse((cx - u * 0.34, bot - u * 0.06, cx + u * 0.34, bot + u * 0.06), fill=PLATE_HI, outline=INK, width=lw)
        d.line([(cx - u * 0.2, top + u * 0.3), (cx + u * 0.2, top + u * 0.3)], fill=(255, 255, 255, 200), width=lw)
    elif kind == "rocket":
        # a fist with a booster: flame bursts out of the wrist end (toward the elbow), flickering
        fw, fh = u * 0.56, u * 0.44
        d.rounded_rectangle((cx - fw / 2, y0, cx + fw / 2, y0 + fh), radius=u * 0.12, fill=PLATE, outline=INK, width=lw)
        for j in range(4):
            x = cx - fw * 0.36 + j * fw * 0.24
            d.ellipse((x - u * 0.07, y0 + fh * 0.62, x + u * 0.07, y0 + fh * 0.62 + u * 0.14), fill=PLATE_HI,
                      outline=INK, width=int(SS * 1.6))
        d.ellipse((cx - u * 0.05, y0 + fh * 0.22, cx + u * 0.05, y0 + fh * 0.22 + u * 0.10), fill=GLOW, outline=INK,
                  width=SS)
        flick = (0.85, 1.0, 0.9, 1.08)[phase % 4]
        fl = u * 0.55 * flick
        base = y0 - u * 0.12
        d.polygon([(cx - fw * 0.36, base), (cx + fw * 0.36, base), (cx, base - fl)], fill=FLAME)
        d.polygon([(cx - fw * 0.18, base), (cx + fw * 0.18, base), (cx, base - fl * 0.6)], fill=FLAME_HI)
