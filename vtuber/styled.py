"""styled.py -- code-drawn background scenes that stand in for stock b-roll.

    python -m vtuber.styled papercut "ship it today" rocket 3    # one test clip -> assets/broll/
    python -m vtuber.styled sketch "your idea" bulb 3 --still     # middle frame as PNG, to eyeball

Two looks, ported from the installed style skills (no image/video model, no Node, no browser):
  papercut  cut paper on a coloured wall: flat fills, paper grain, soft drop shadows, back-eased pops
            (palette + layout from paper-reel's references/style.md)
  sketch    soft hand-drawn ink on cream paper: wobbly strokes, offset watercolour fill, lines that
            "boil" (3 jittered redraws cycled at 7.5 fps), drawn on with a wipe

Each scene = a headline + one icon from ICONS + light decor, kept above the caption band
(captions sit at 66% height in compose.py; the avatar stands in front of all of it).
Groq only picks the headline words and an icon NAME from the whitelist; it never writes drawing code.
Clips are 1080x1920 / 30 fps / no audio, cached in assets/broll/ like the Pexels clips.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "assets" / "broll"
FONTS = ROOT / "assets" / "fonts"
W, H, FPS = 1080, 1920, 30
VERSION = 1            # bump to invalidate cached clips after changing the look
STYLES = ("papercut", "sketch")
MODES = ("stock", "styled", "mix")


def _bgr(hexs: str) -> tuple[float, float, float]:
    h = hexs.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


# ── icons: unit-square primitives, drawn by both styles ──────────────────────
# ("poly", pts, role) closed fill | ("circle", (cx, cy, r), role) | ("rect", (x0, y0, x1, y1), role)
# ("line", pts, width, role) open stroke
def _heart() -> list:
    pts = []
    for k in range(48):
        t = 2 * math.pi * k / 48
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((0.5 + x / 38, 0.47 - y / 38))
    return [("poly", pts, "red")]


def _star(n=8, r0=0.42, r1=0.18) -> list:
    pts = [(0.5 + (r0 if k % 2 == 0 else r1) * math.cos(math.pi * k / n - math.pi / 2),
            0.5 + (r0 if k % 2 == 0 else r1) * math.sin(math.pi * k / n - math.pi / 2)) for k in range(2 * n)]
    return [("poly", pts, "accent")]


def _arc(cx, cy, r, a0, a1, n=18) -> list:
    return [(cx + r * math.cos(a0 + (a1 - a0) * k / n), cy + r * math.sin(a0 + (a1 - a0) * k / n))
            for k in range(n + 1)]


ICONS: dict[str, list] = {
    "spark": _star(),
    "bulb": [("circle", (0.5, 0.4, 0.28), "accent"), ("rect", (0.38, 0.62, 0.62, 0.77), "dark"),
             ("rect", (0.42, 0.79, 0.58, 0.86), "dark"), ("line", [(0.42, 0.42), (0.5, 0.52), (0.58, 0.42)], 0.03, "light")],
    "phone": [("rect", (0.28, 0.06, 0.72, 0.94), "dark"), ("rect", (0.32, 0.14, 0.68, 0.8), "light"),
              ("circle", (0.5, 0.87, 0.03), "light"), ("rect", (0.38, 0.24, 0.62, 0.3), "main"),
              ("rect", (0.38, 0.36, 0.56, 0.41), "accent")],
    "laptop": [("rect", (0.15, 0.2, 0.85, 0.66), "dark"), ("rect", (0.2, 0.25, 0.8, 0.61), "light"),
               ("poly", [(0.05, 0.7), (0.95, 0.7), (0.88, 0.8), (0.12, 0.8)], "main"),
               ("rect", (0.27, 0.33, 0.6, 0.38), "accent"), ("rect", (0.27, 0.44, 0.5, 0.49), "main")],
    "clock": [("circle", (0.5, 0.5, 0.42), "dark"), ("circle", (0.5, 0.5, 0.35), "light"),
              ("line", [(0.5, 0.5), (0.5, 0.25)], 0.05, "dark"), ("line", [(0.5, 0.5), (0.67, 0.6)], 0.05, "main")],
    "coin": [("circle", (0.5, 0.5, 0.4), "accent"), ("circle", (0.5, 0.5, 0.31), "main"),
             ("rect", (0.455, 0.3, 0.545, 0.7), "light")],
    "chart_up": [("rect", (0.14, 0.6, 0.3, 0.86), "main"), ("rect", (0.4, 0.45, 0.56, 0.86), "main"),
                 ("rect", (0.66, 0.26, 0.82, 0.86), "main"),
                 ("line", [(0.1, 0.5), (0.38, 0.32), (0.58, 0.38), (0.84, 0.14)], 0.05, "green"),
                 ("poly", [(0.92, 0.06), (0.9, 0.24), (0.76, 0.12)], "green")],
    "chart_down": [("rect", (0.14, 0.26, 0.3, 0.86), "main"), ("rect", (0.4, 0.45, 0.56, 0.86), "main"),
                   ("rect", (0.66, 0.6, 0.82, 0.86), "main"),
                   ("line", [(0.1, 0.14), (0.38, 0.34), (0.58, 0.3), (0.84, 0.52)], 0.05, "red"),
                   ("poly", [(0.92, 0.6), (0.74, 0.58), (0.86, 0.44)], "red")],
    "check": [("circle", (0.5, 0.5, 0.42), "green"), ("line", [(0.3, 0.52), (0.45, 0.66), (0.72, 0.36)], 0.08, "light")],
    "cross": [("circle", (0.5, 0.5, 0.42), "red"), ("line", [(0.33, 0.33), (0.67, 0.67)], 0.08, "light"),
              ("line", [(0.67, 0.33), (0.33, 0.67)], 0.08, "light")],
    "warning": [("poly", [(0.5, 0.08), (0.94, 0.86), (0.06, 0.86)], "accent"),
                ("rect", (0.465, 0.34, 0.535, 0.62), "dark"), ("circle", (0.5, 0.73, 0.045), "dark")],
    "chat": [("poly", [(0.1, 0.16), (0.9, 0.16), (0.9, 0.64), (0.42, 0.64), (0.24, 0.84), (0.27, 0.64), (0.1, 0.64)], "light"),
             ("circle", (0.32, 0.4, 0.05), "dark"), ("circle", (0.5, 0.4, 0.05), "dark"), ("circle", (0.68, 0.4, 0.05), "dark")],
    "code": [("rect", (0.06, 0.14, 0.94, 0.86), "dark"),
             ("line", [(0.36, 0.32), (0.2, 0.5), (0.36, 0.68)], 0.06, "accent"),
             ("line", [(0.64, 0.32), (0.8, 0.5), (0.64, 0.68)], 0.06, "accent"),
             ("line", [(0.56, 0.26), (0.44, 0.74)], 0.06, "light")],
    "lock": [("line", _arc(0.5, 0.44, 0.17, math.pi, 2 * math.pi), 0.07, "dark"),
             ("line", [(0.33, 0.44), (0.33, 0.5)], 0.07, "dark"), ("line", [(0.67, 0.44), (0.67, 0.5)], 0.07, "dark"),
             ("rect", (0.24, 0.48, 0.76, 0.9), "accent"), ("circle", (0.5, 0.64, 0.05), "dark"),
             ("rect", (0.48, 0.64, 0.52, 0.78), "dark")],
    "heart": _heart(),
    "arrow": [("poly", [(0.08, 0.4), (0.56, 0.4), (0.56, 0.2), (0.92, 0.5), (0.56, 0.8), (0.56, 0.6), (0.08, 0.6)], "main")],
    "rocket": [("poly", [(0.5, 0.04), (0.66, 0.3), (0.66, 0.68), (0.34, 0.68), (0.34, 0.3)], "light"),
               ("poly", [(0.34, 0.46), (0.18, 0.72), (0.34, 0.68)], "main"),
               ("poly", [(0.66, 0.46), (0.82, 0.72), (0.66, 0.68)], "main"),
               ("circle", (0.5, 0.34, 0.07), "dark"),
               ("poly", [(0.38, 0.7), (0.62, 0.7), (0.5, 0.96)], "accent")],
}

# keyword -> icon, used when Groq is down or returns something off the list
KEYWORDS = [
    (r"money|price|cost|paid|pay|rupee|dollar|cash|earn|revenue|\$|₹", "coin"),
    (r"code|bug|build|deploy|repo|github|script|api|server", "code"),
    (r"phone|call|ring|dial", "phone"),
    (r"laptop|computer|screen|website|app|dashboard", "laptop"),
    (r"time|hour|minute|second|late|fast|slow|wait|today", "clock"),
    (r"idea|think|smart|learn|why|how", "bulb"),
    (r"grow|more|up|scale|win|best|better", "chart_up"),
    (r"fail|down|drop|lose|lost|less|worse", "chart_down"),
    (r"safe|secure|password|private|lock", "lock"),
    (r"yes|works|done|right|correct|ship", "check"),
    (r"\bno\b|wrong|broke|broken|never|stop", "cross"),
    (r"warn|careful|risk|danger|mistake", "warning"),
    (r"talk|chat|message|say|tell|ask|reply|dm", "chat"),
    (r"love|like|heart|fan", "heart"),
    (r"launch|start|go|rocket|boost", "rocket"),
    (r"next|then|so|because|step", "arrow"),
]

STOP = set("""a an the and or but so to of in on at for with from by is are was were be been it its this that
these those i you he she we they my your our their me him her us them do does did not just like really very
can could will would should have has had get got make made what when where who which there here all any some
one about into than then also if as up out over more most""".split())

PAPER = {  # paper-reel palette (references/style.md)
    "main": "#d67555", "accent": "#e6b82a", "dark": "#1c2b56", "light": "#f4f1e8",
    "red": "#d2453a", "green": "#2f9e5b", "card": "#192833", "ink": "#1c2b56",
    "walls": ["#869997", "#2b3b5e", "#583656", "#6c9e8f", "#c9b28a", "#4a4566", "#c79a5b"],
    "scraps": ["#f4f1e8", "#e6b82a", "#d67555", "#95a9a7", "#f2c94c"],
}
SKETCH = {  # soft pastels under dark ink
    "main": "#f2a68a", "accent": "#f6d06b", "dark": "#8fa6c9", "light": "#fffdf7",
    "red": "#ee8a80", "green": "#9fd3a8", "ink": "#2b2a33", "paper": "#f6f0e2", "rule": "#c9d6e8",
}


# ── planning: which shots, which words, which icon ───────────────────────────
def _clean_word(s: str) -> str:
    """Whitelist (the text is drawn verbatim), max 3 whole words / 22 chars.
    ₹ becomes 'Rs' because the bundled fonts have no rupee glyph."""
    s = re.sub(r"₹\s*", "Rs ", s or "")
    s = re.sub(r"[^A-Za-z0-9 '&%$?!+-]", " ", s)
    out = ""
    for w in s.split()[:3]:
        nxt = f"{out} {w}".strip()
        if len(nxt) > 22:
            break
        out = nxt
    return out or s.strip()[:22]


def _guess_icon(text: str) -> str:
    t = (text or "").lower()
    for pat, icon in KEYWORDS:
        if re.search(pat, t):
            return icon
    return "spark"


def _fallback(line: str) -> dict:
    words = [w for w in re.findall(r"[A-Za-z0-9']+", line or "") if w.lower() not in STOP]
    best = sorted(words, key=len, reverse=True)[:2]
    keep = [w for w in words if w in best][:2]       # keep spoken order
    return {"word": _clean_word(" ".join(keep)) or "listen", "icon": _guess_icon(line)}


DECIDE = (
    "Also decide 'use' for each line, picking whichever would look BEST behind it. \"stock\": real stock footage "
    "can show it, and that covers more than it seems: people and their reactions (frustrated at a laptop, "
    "celebrating, thinking), everyday places and objects, screens, code, cities, work. Real footage feels human "
    "and is the default for story moments. \"drawn\": a designed paper/sketch card with a short headline and an "
    "icon, better when the line's point is a number, a price, a short verdict or punchline (\"Also dead.\"), a "
    "lesson, a list, a call to action, or an idea footage could only show loosely. There is no quota and no "
    "fixed pattern; judge each line. But texture matters: a long run of the same kind gets boring, so where a "
    "line could go either way, choose the one that breaks a run of 3 or more.")


def plan(lines: list[str], decide: bool = False) -> list[dict]:
    """One Groq call for every line: [{word, icon, use}] where use = 'stock' | 'drawn' (only with
    decide=True, else None). Fails soft to keyword rules with use=None (caller falls back to pick())."""
    specs = [{**_fallback(x), "use": None} for x in lines]
    if not lines:
        return specs
    try:
        from vtuber import groq_pool  # noqa: PLC0415
        menu = "\n".join(f"{i}: {x}" for i, x in enumerate(lines))
        shape = '{"i": 0, "word": "...", "icon": "..."' + (', "use": "stock|drawn"}' if decide else "}")
        # the stock-vs-drawn call is a judgement across the whole reel: worth the stronger tier
        resp = groq_pool.chat(groq_pool.SMART if decide else groq_pool.FAST, messages=[
            {"role": "system", "content":
                "You design the background for each numbered line of a fast vertical reel. For each line give "
                "'word': the 1-3 word punch of the line, the phrase a viewer remembers (no filler, no full sentence), "
                f"and 'icon': the ONE icon from this list that shows the idea: {sorted(ICONS)}. "
                + (DECIDE + " " if decide else "")
                + f'Reply JSON {{"scenes": [{shape}]}}'},
            {"role": "user", "content": menu}],
            max_tokens=60 + 50 * len(lines), response_format={"type": "json_object"}, require_content=True)
        data = json.loads(resp.choices[0].message.content)
        for s in data.get("scenes", []):
            if not isinstance(s, dict):
                continue
            i = s.get("i")
            if not isinstance(i, int) or not 0 <= i < len(lines):
                continue
            word = _clean_word(s.get("word") if isinstance(s.get("word"), str) else "")
            icon = s.get("icon") if s.get("icon") in ICONS else _guess_icon(lines[i])
            use = s.get("use") if decide and s.get("use") in ("stock", "drawn") else None
            specs[i] = {"word": word or specs[i]["word"], "icon": icon, "use": use}
    except Exception as e:  # noqa: BLE001 - the cards are still drawn from the keyword rules
        print(f"  styled: Groq plan unavailable ({str(e)[:80]}), using keyword rules", file=sys.stderr)
    return specs


def decide(beats: list[dict], specs: list[dict]) -> set[int]:
    """mix: Groq's per-line call (spec['use']), no quota. Halves of a split line follow their first
    half. Lines Groq didn't answer fall back to pick()'s alternation, so one bad reply can't empty it."""
    fallback = pick(beats, "alternate")
    out: set[int] = set()
    for i, b in enumerate(beats):
        if i > 0 and (b.get("cont") or (b.get("broll") and b.get("broll") == beats[i - 1].get("broll"))):
            if i - 1 in out:
                out.add(i)
            continue
        use = specs[i].get("use") if i < len(specs) else None
        if use == "drawn" or (use is None and i in fallback):
            out.add(i)
    return out


def pick(beats: list[dict], mode: str) -> set[int]:
    """Fixed rules: stock -> none, styled -> all, anything else -> the fallback alternation
    (hook keeps real footage, then every other line drawn, split lines kept together).
    mix normally uses decide() instead; this is what it falls back to when Groq can't answer."""
    if mode == "stock":
        return set()
    if mode == "styled":
        return set(range(len(beats)))
    out: set[int] = set()
    k = 0   # counts lines, not shots: halves of a split line share their b-roll term
    for i, b in enumerate(beats):
        same_line = i > 0 and (b.get("cont") or (b.get("broll") and b.get("broll") == beats[i - 1].get("broll")))
        if same_line:
            if i - 1 in out:
                out.add(i)
            continue
        if k % 2 == 1:
            out.add(i)
        k += 1
    return out


def choose_style(style: str, seed_text: str) -> str:
    """'auto' = one look per reel (stable for re-renders), never mixed inside a reel."""
    if style in STYLES:
        return style
    return STYLES[int(hashlib.sha1(seed_text.encode("utf-8")).hexdigest(), 16) % len(STYLES)]


# ── drawing helpers ──────────────────────────────────────────────────────────
def _noise(h: int, w: int, seed: int, octaves=((6, 1.0), (24, 0.5), (96, 0.3), (300, 0.15))) -> np.ndarray:
    """Multi-octave value noise in [-1, 1]: the paper grain."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    for cells, amp in octaves:
        g = rng.random((max(2, cells * h // w), cells)).astype(np.float32)
        acc += cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC) * amp
    acc -= acc.mean()
    return acc / (np.abs(acc).max() + 1e-6)


def _vignette(strength: float = 0.22) -> np.ndarray:
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H * 0.45) / (H * 0.6)) ** 2)
    return (1 - strength * np.clip(d - 0.35, 0, 1)).astype(np.float32)[..., None]


def _to_px(prim, S: int):
    """Primitive -> (closed?, points in px, stroke width px, role)."""
    kind = prim[0]
    if kind == "poly":
        return True, [(x * S, y * S) for x, y in prim[1]], 0, prim[2]
    if kind == "circle":
        cx, cy, r = prim[1]
        n = max(24, int(r * S / 4))
        return True, [((cx + r * math.cos(2 * math.pi * k / n)) * S, (cy + r * math.sin(2 * math.pi * k / n)) * S)
                      for k in range(n)], 0, prim[2]
    if kind == "rect":
        x0, y0, x1, y1 = prim[1]
        return True, [(x0 * S, y0 * S), (x1 * S, y0 * S), (x1 * S, y1 * S), (x0 * S, y1 * S)], 0, prim[2]
    return False, [(x * S, y * S) for x, y in prim[1]], max(2, prim[2] * S), prim[3]


def _mask(closed: bool, pts, width: float, size: tuple[int, int], ss: int = 2) -> np.ndarray:
    """Antialiased float mask (supersampled PIL draw)."""
    w, h = size
    im = Image.new("L", (w * ss, h * ss), 0)
    d = ImageDraw.Draw(im)
    p = [(x * ss, y * ss) for x, y in pts]
    if closed:
        d.polygon(p, fill=255)
    else:
        lw = int(width * ss)
        d.line(p, fill=255, width=lw, joint="curve")
        for x, y in (p[0], p[-1]):  # round caps
            d.ellipse((x - lw / 2, y - lw / 2, x + lw / 2, y + lw / 2), fill=255)
    return np.asarray(im.resize((w, h), Image.LANCZOS), np.float32) / 255.0


def _subdivide(pts, closed: bool, step: float = 14.0):
    out = []
    seq = list(pts) + ([pts[0]] if closed else [])
    for (x0, y0), (x1, y1) in zip(seq, seq[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
        out += [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(n)]
    out.append(seq[-1])
    return out


def _jitter(pts, rng: np.random.Generator, amt: float):
    """Smooth hand wobble: low-frequency offsets along the path, not per-point noise."""
    n = len(pts)
    if n < 2:
        return pts
    knots = rng.normal(0, amt, (max(2, n // 6 + 2), 2))
    xs = np.linspace(0, len(knots) - 1, n)
    off = np.stack([np.interp(xs, np.arange(len(knots)), knots[:, k]) for k in range(2)], 1)
    return [(x + dx, y + dy) for (x, y), (dx, dy) in zip(pts, off)]


class Sprite:
    """Premultiplied BGR + alpha, float32, blitted with scale/rotation about its centre."""

    def __init__(self, rgb: np.ndarray, a: np.ndarray):
        self.rgb, self.a = rgb, a

    @classmethod
    def empty(cls, w: int, h: int) -> "Sprite":
        return cls(np.zeros((h, w, 3), np.float32), np.zeros((h, w), np.float32))

    def over(self, color, m: np.ndarray, tex: np.ndarray | None = None, tex_amt: float = 0.0):
        col = np.asarray(color, np.float32)[None, None, :] * np.ones_like(self.rgb)
        if tex is not None:
            col *= 1 + tex_amt * tex[..., None]
        m3 = m[..., None]
        self.rgb = self.rgb * (1 - m3) + col * m3
        self.a = self.a * (1 - m) + m

    def shadow(self, m: np.ndarray, dx: int, dy: int, blur: float, amt: float):
        sh = np.roll(np.roll(m, dy, 0), dx, 1)
        sh = cv2.GaussianBlur(sh, (0, 0), blur) * amt
        self.rgb *= (1 - sh)[..., None]
        self.a = self.a + sh * (1 - self.a)

    def blit(self, canvas: np.ndarray, cx: float, cy: float, scale: float = 1.0, angle: float = 0.0,
             reveal: float | None = None):
        if scale <= 0.01:
            return
        h, w = self.a.shape
        side = int(math.ceil(math.hypot(w, h) * scale)) + 4
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += side / 2 - w / 2
        M[1, 2] += side / 2 - h / 2
        rgb = cv2.warpAffine(self.rgb, M, (side, side), flags=cv2.INTER_LINEAR, borderValue=0)
        a = cv2.warpAffine(self.a, M, (side, side), flags=cv2.INTER_LINEAR, borderValue=0)
        if reveal is not None and reveal < 1:   # drawn-on wipe, left to right with a soft edge
            ramp = np.clip((reveal * (side + 120) - np.arange(side)) / 120, 0, 1).astype(np.float32)
            rgb *= ramp[None, :, None]
            a *= ramp[None, :]
        x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        dx1, dy1 = min(W, x0 + side), min(H, y0 + side)
        if dx1 <= dx0 or dy1 <= dy0:
            return
        sub_a = a[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0][..., None]
        sub_rgb = rgb[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]
        roi = canvas[dy0:dy1, dx0:dx1]
        roi *= 1 - sub_a
        roi += sub_rgb


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / name), size)


def _fit_font(text: str, name: str, max_w: int, start: int, lo: int = 40) -> ImageFont.FreeTypeFont:
    size = start
    while size > lo:
        f = _font(name, size)
        if f.getbbox(text)[2] - f.getbbox(text)[0] <= max_w:
            return f
        size -= 6
    return _font(name, lo)


def _text_mask(text: str, font: ImageFont.FreeTypeFont, pad: int = 10) -> np.ndarray:
    l, t, r, b = font.getbbox(text)
    im = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad - l, pad - t), text, font=font, fill=255)
    return np.asarray(im, np.float32) / 255.0


def _ease_back(u: float, s: float = 1.9) -> float:
    u = min(1.0, max(0.0, u)) - 1
    return u * u * ((s + 1) * u + s) + 1


# ── papercut ─────────────────────────────────────────────────────────────────
ICON_S = 600
ICON_Y = 800
SIGN_Y = 300


def _paper_icon(icon: str, tex: np.ndarray) -> Sprite:
    pad = 40
    S = ICON_S
    sp = Sprite.empty(S + 2 * pad, S + 2 * pad)
    for prim in ICONS[icon]:
        closed, pts, width, role = _to_px(prim, S)
        pts = [(x + pad, y + pad) for x, y in pts]
        m = _mask(closed, pts, width, (S + 2 * pad, S + 2 * pad))
        sp.shadow(m, 6, 10, 9, 0.42)
        sp.over(_bgr(PAPER[role]), m, tex, 0.10)
    return sp


def _paper_sign(word: str, rng: np.random.Generator) -> Sprite:
    text = word.upper()
    font = _fit_font(text, "ArchivoBlack-Regular.ttf", 780, 132)
    tm = _text_mask(text, font)
    th, tw = tm.shape
    cw, ch = tw + 90, th + 70
    off = 16
    pad = 40
    sw, sh = cw + off + 2 * pad, ch + off + 2 * pad
    sp = Sprite.empty(sw, sh)
    tex = _noise(sh, sw, int(rng.integers(1 << 30)), ((4, 1.0), (40, 0.5), (200, 0.3)))
    card = np.zeros((sh, sw), np.float32)
    cv2.rectangle(card, (pad + off, pad), (pad + off + cw, pad + ch), 1.0, -1)
    sp.shadow(card, 6, 10, 10, 0.4)
    sp.over(_bgr(PAPER["card"]), card, tex, 0.08)
    paper = np.zeros((sh, sw), np.float32)
    cv2.rectangle(paper, (pad, pad + off), (pad + cw, pad + off + ch), 1.0, -1)
    sp.shadow(paper, 5, 9, 9, 0.38)
    sp.over(_bgr(PAPER["light"]), paper, tex, 0.07)
    tx, ty = pad + 45, pad + off + 35
    big = np.zeros((sh, sw), np.float32)
    big[ty:ty + th, tx:tx + tw] = tm
    sp.over(_bgr(PAPER["ink"]), big)
    return sp


def _paper_scrap(rng: np.random.Generator, tex: np.ndarray) -> Sprite:
    s = int(rng.integers(50, 110))
    pad = 24
    sp = Sprite.empty(s + 2 * pad, s + 2 * pad)
    kind = rng.integers(3)
    if kind == 0:
        pts = [(pad + s / 2 + s / 2 * math.cos(a), pad + s / 2 + s / 2 * math.sin(a))
               for a in np.linspace(0, 2 * math.pi, 28, endpoint=False)]
    elif kind == 1:
        pts = [(pad, pad + s), (pad + s / 2, pad), (pad + s, pad + s)]
    else:
        pts = [(pad, pad + s * 0.3), (pad + s, pad), (pad + s, pad + s * 0.7), (pad, pad + s)]
    m = _mask(True, pts, 0, (s + 2 * pad, s + 2 * pad))
    sp.shadow(m, 4, 7, 6, 0.35)
    col = PAPER["scraps"][int(rng.integers(len(PAPER["scraps"])))]
    sp.over(_bgr(col), m, tex[: s + 2 * pad, : s + 2 * pad], 0.1)
    return sp


def _render_papercut(spec: dict, seconds: float, seed: int, sink) -> None:
    rng = np.random.default_rng(seed)
    wall = _bgr(PAPER["walls"][seed % len(PAPER["walls"])])
    grain = _noise(H, W, seed)
    bg = np.ones((H, W, 3), np.float32) * np.asarray(wall, np.float32)
    bg *= (1 + 0.07 * grain)[..., None]
    floor_y = int(H * 0.79)                      # a darker paper strip the avatar stands on
    bg[floor_y:] *= 0.82
    bg[floor_y:floor_y + 6] *= 0.7
    vig = _vignette()
    tex = _noise(ICON_S + 80, ICON_S + 80, seed + 1, ((5, 1.0), (40, 0.5), (220, 0.3)))
    icon = _paper_icon(spec["icon"], tex)
    sign = _paper_sign(spec["word"], rng)
    scraps = []
    for _ in range(7):
        x = float(rng.uniform(60, W - 60))
        y = float(rng.uniform(120, H * 0.62))
        if abs(x - W / 2) < 330 and 520 < y < 1100:   # keep the hero clear
            x = 90 if x < W / 2 else W - 90
        scraps.append((_paper_scrap(rng, tex), x, y, float(rng.uniform(0, 6.28)), float(rng.uniform(-25, 25))))
    tilt = float(rng.uniform(-2.5, 2.5))
    n = max(1, int(round(seconds * FPS)))
    for f in range(n):
        t = f / FPS
        canvas = bg.copy()
        for sp, x, y, ph, rot in scraps:
            sp.blit(canvas, x + 6 * math.sin(t * 1.3 + ph), y + 9 * math.sin(t * 0.9 + ph),
                    _ease_back((t - 0.05) / 0.3), rot + 4 * math.sin(t + ph))
        s_icon = _ease_back((t - 0.12) / 0.38)
        icon.blit(canvas, W / 2, ICON_Y + 7 * math.sin(t * 2 * math.pi / 2.4), s_icon,
                  1.4 * math.sin(t * 2 * math.pi / 3.1))
        sign.blit(canvas, W / 2, SIGN_Y, _ease_back(t / 0.32), tilt)
        canvas *= vig
        sink(canvas)


# ── sketch ───────────────────────────────────────────────────────────────────
BOIL = 3          # redraws per element
BOIL_EVERY = 4    # frames per redraw (30 / 4 = 7.5 fps boil)


def _ink_layer(prims, S: int, pad: int, rng: np.random.Generator, wobble: float, ink_w: float) -> Sprite:
    size = (S + 2 * pad, S + 2 * pad)
    sp = Sprite.empty(*size)
    wash = _noise(size[1], size[0], int(rng.integers(1 << 30)), ((3, 1.0), (12, 0.6), (60, 0.3)))
    strokes = []
    for prim in prims:
        closed, pts, width, role = _to_px(prim, S)
        pts = [(x + pad, y + pad) for x, y in pts]
        fine = _subdivide(pts, closed)
        if closed:   # watercolour fill, offset from the line like a quick colouring-in
            fill_pts = _jitter([(x + 9, y + 7) for x, y in fine], rng, wobble * 1.6)
            m = cv2.GaussianBlur(_mask(True, fill_pts, 0, size), (0, 0), 2.2) * (0.78 + 0.18 * wash)
            sp.over(_bgr(SKETCH[role]), np.clip(m, 0, 1))
            strokes.append((True, _jitter(fine, rng, wobble), ink_w))
        else:        # open strokes are thick marker lines in the role colour, inked on top
            line = _jitter(fine, rng, wobble)
            m = _mask(False, line, width * 0.8, size)
            sp.over(_bgr(SKETCH[role] if role != "light" else SKETCH["ink"]), m)
            strokes.append((False, _jitter(fine, rng, wobble * 0.6), max(ink_w, width * 0.25)))
    for closed, pts, w_ in strokes:
        seq = pts + ([pts[0]] if closed else [])
        sp.over(_bgr(SKETCH["ink"]), _mask(False, seq, w_, size))
    return sp


def _sketch_title(word: str, rng: np.random.Generator) -> Sprite:
    font = _fit_font(word, "PatrickHand-Regular.ttf", 840, 190)
    tm = _text_mask(word, font, pad=30)
    th, tw = tm.shape
    pad = 20
    sp = Sprite.empty(tw + 2 * pad, th + 60 + 2 * pad)
    big = np.zeros(sp.a.shape, np.float32)
    big[pad:pad + th, pad:pad + tw] = tm
    y = pad + th + 10   # marker underline, slightly drunk
    line = _jitter(_subdivide([(pad + 30, y), (pad + tw - 30, y + rng.uniform(-14, 14))], False), rng, 4)
    um = _mask(False, line, 16, (sp.a.shape[1], sp.a.shape[0]))
    sp.over(_bgr(SKETCH["accent"]), um * 0.9)
    sp.over(_bgr(SKETCH["ink"]), big)
    return sp


DOODLES = [
    [("line", _arc(0.5, 0.5, 0.35, 0, 5.5, 20), 0.06, "ink")],                                # swirl
    [("line", [(0.5, 0.1), (0.5, 0.9)], 0.07, "ink"), ("line", [(0.1, 0.5), (0.9, 0.5)], 0.07, "ink")],  # plus
    [("line", [(0.1, 0.6), (0.3, 0.4), (0.5, 0.6), (0.7, 0.4), (0.9, 0.6)], 0.06, "ink")],  # zigzag
    [("circle", (0.5, 0.5, 0.18), "accent")],                                                # dot
]


def _render_sketch(spec: dict, seconds: float, seed: int, sink) -> None:
    rng = np.random.default_rng(seed)
    paper = np.ones((H, W, 3), np.float32) * np.asarray(_bgr(SKETCH["paper"]), np.float32)
    paper *= (1 + 0.045 * _noise(H, W, seed))[..., None]
    for y in range(240, H, 76):   # faint notebook rules
        cv2.line(paper, (0, y), (W, y), _bgr(SKETCH["rule"]), 2, cv2.LINE_AA)
    cv2.line(paper, (118, 0), (118, H), _bgr("#efb3ad"), 2, cv2.LINE_AA)
    vig = _vignette(0.14)
    ink_w = 7.0
    icons = [_ink_layer(ICONS[spec["icon"]], ICON_S, 40, np.random.default_rng(seed + 10 + k), 5.0, ink_w)
             for k in range(BOIL)]
    titles = [_sketch_title(spec["word"], np.random.default_rng(seed + 20 + k)) for k in range(BOIL)]
    doodles = []
    for _ in range(5):
        d = DOODLES[int(rng.integers(len(DOODLES)))]
        x = float(rng.choice([rng.uniform(150, 300), rng.uniform(W - 300, W - 150)]))
        y = float(rng.uniform(520, H * 0.6))
        doodles.append(([_ink_layer(d, 90, 16, np.random.default_rng(seed + 40 + k + int(x)), 3.0, 5.0)
                         for k in range(BOIL)], x, y, float(rng.uniform(-30, 30))))
    n = max(1, int(round(seconds * FPS)))
    for f in range(n):
        t = f / FPS
        v = (f // BOIL_EVERY) % BOIL
        canvas = paper.copy()
        for frames, x, y, rot in doodles:
            frames[v].blit(canvas, x, y, 1.0, rot, reveal=min(1.0, max(0.0, (t - 0.35) / 0.4)))
        icons[v].blit(canvas, W / 2, ICON_Y, 1.0 + 0.012 * math.sin(t * 2.2), 0.0,
                      reveal=min(1.0, max(0.0, (t - 0.1) / 0.6)))
        titles[v].blit(canvas, W / 2, SIGN_Y, 1.0, -1.5, reveal=min(1.0, t / 0.4))
        canvas *= vig
        sink(canvas)


# ── output ───────────────────────────────────────────────────────────────────
def render(style: str, spec: dict, seconds: float, seed: int = 7) -> Path:
    """One scene clip (cached). Raises RuntimeError if ffmpeg fails."""
    style = style if style in STYLES else "papercut"
    spec = {"word": _clean_word(spec.get("word", "")) or "listen",
            "icon": spec.get("icon") if spec.get("icon") in ICONS else "spark"}
    seconds = max(0.5, round(float(seconds), 1))
    key = hashlib.sha1(json.dumps([VERSION, style, spec, seconds, seed], sort_keys=True).encode()).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"styled_{style}_{key}.mp4"
    if out.exists() and out.stat().st_size > 10_000:
        return out
    tmp = out.with_suffix(".part.mp4")
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
                             "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(tmp)],
                            stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def sink(canvas: np.ndarray) -> None:
        proc.stdin.write(np.clip(canvas, 0, 255).astype(np.uint8).tobytes())

    try:
        (_render_sketch if style == "sketch" else _render_papercut)(spec, seconds, int(seed), sink)
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"styled: ffmpeg failed: {err[-300:]}")
    except BaseException:
        proc.kill()
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(out)
    print(f"  styled: {style} '{spec['word']}' + {spec['icon']} ({seconds:g}s) -> {out.name}")
    return out


def still(style: str, spec: dict, at: float, out: Path, seed: int = 7) -> Path:
    """One frame as PNG, to eyeball a look without encoding video."""
    frames: list[np.ndarray] = []
    n_target = int(at * FPS)

    def sink(canvas: np.ndarray) -> None:
        if len(frames) == 0 and sink.i == n_target:
            frames.append(np.clip(canvas, 0, 255).astype(np.uint8))
        sink.i += 1
    sink.i = 0
    (_render_sketch if style == "sketch" else _render_papercut)(spec, at + 1 / FPS, seed, sink)
    cv2.imwrite(str(out), frames[0])
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("style", choices=STYLES)
    ap.add_argument("word")
    ap.add_argument("icon", choices=sorted(ICONS))
    ap.add_argument("seconds", type=float, nargs="?", default=3.0)
    ap.add_argument("--still", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    spec = {"word": a.word, "icon": a.icon}
    if a.still:
        p = ROOT / "output" / f"still_{a.style}_{a.icon}.png"
        p.parent.mkdir(exist_ok=True)
        print(still(a.style, spec, a.seconds / 2, p, a.seed))
    else:
        print(render(a.style, spec, a.seconds, a.seed))
