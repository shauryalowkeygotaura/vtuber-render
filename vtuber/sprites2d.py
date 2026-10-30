"""sprites2d.py -- the 2D (PNGtuber) sprite pack: one PNG per emotion x mouth x eyes.

A sprite pack is just a folder:

    <pack>/<emotion>_<mouth>.png         eyes open
    <pack>/<emotion>_<mouth>_blink.png   eyes shut   (optional per emotion)

emotion in EMOTIONS, mouth in MOUTHS. Any missing file falls back to the
neutral one, so a hand-drawn pack can start with 4 neutral mouths and grow.
Draw your own (any size, transparent PNG, same canvas for every file) and
pass --pack <folder>.

The default pack ("rev") is generated here as SVG and rasterised once with
the local Chromium, then cached in assets/sprites/rev/. It is a chibi bust
in the Revengine kit: burgundy hoodie, cream strings, near-black hair.
"""
from __future__ import annotations

import base64
from pathlib import Path

EMOTIONS = ("neutral", "happy", "excited", "sad", "angry", "surprised",
            "thinking", "smug", "relaxed", "nervous")
MOUTHS = ("closed", "small", "open", "round")
W, H = 800, 1000

INK = "#22161a"
SKIN = "#f3d5c0"
SKIN_SHADE = "#e3b9a0"
HAIR = "#1b1416"
HAIR_HI = "#3a2a2f"
HOODIE = "#722F37"
HOODIE_DARK = "#561f27"
CREAM = "#F0E4CC"
IRIS = "#5a2a22"
MOUTH_IN = "#5b1e26"
TONGUE = "#d9707a"
BLUSH = "#ec8f8f"

# brow angle (deg, + = inner end lower) and lift (px, + = higher) per emotion
BROWS = {
    "neutral": (0, 0), "happy": (-6, 6), "excited": (-10, 14), "sad": (-18, 4),
    "angry": (22, -6), "surprised": (-4, 22), "thinking": (8, 0),
    "smug": (10, -2), "relaxed": (-4, 0), "nervous": (-16, 8),
}
# mouths that read as smiles when the mouth is closed
CLOSED = {
    "neutral": "M370 600 Q400 608 430 600",
    "happy": "M355 592 Q400 632 445 592",
    "excited": "M350 590 Q400 640 450 590",
    "sad": "M365 612 Q400 590 435 612",
    "angry": "M362 610 Q400 596 438 610",
    "surprised": None,           # small o, drawn below
    "thinking": "M385 604 Q408 598 430 606",
    "smug": "M365 604 Q410 612 440 590",
    "relaxed": "M370 598 Q400 614 430 598",
    "nervous": "M362 604 Q375 596 388 604 Q400 612 412 604 Q425 596 438 604",
}
SMILEY = {"happy", "excited", "smug", "relaxed"}


def _eye(cx: float, emotion: str, blink: bool, side: int) -> str:
    """side -1 = left eye (viewer's left), +1 = right."""
    cy = 480
    if blink or emotion == "happy":
        if emotion in SMILEY or emotion == "happy":
            return f'<path d="M{cx-34} {cy+6} Q{cx} {cy-30} {cx+34} {cy+6}" fill="none" stroke="{INK}" stroke-width="9" stroke-linecap="round"/>'
        return f'<path d="M{cx-34} {cy} Q{cx} {cy+18} {cx+34} {cy}" fill="none" stroke="{INK}" stroke-width="9" stroke-linecap="round"/>'
    rx, ry, ir = 36, 46, 27
    if emotion == "surprised":
        rx, ry, ir = 40, 52, 18
    if emotion == "nervous":
        ir = 20
    gx, gy = 0, 4
    if emotion == "thinking":
        gx, gy = 10 * side + 6, -14
    if emotion == "nervous":
        gx = -8
    parts = [
        f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="#fff" stroke="{INK}" stroke-width="6"/>',
        f'<ellipse cx="{cx+gx}" cy="{cy+gy}" rx="{ir}" ry="{ir*1.25:.0f}" fill="{IRIS}"/>',
        f'<ellipse cx="{cx+gx}" cy="{cy+gy+4}" rx="{ir*0.5:.0f}" ry="{ir*0.65:.0f}" fill="{INK}"/>',
        f'<circle cx="{cx+gx-9}" cy="{cy+gy-12}" r="8" fill="#fff"/>',
        f'<circle cx="{cx+gx+10}" cy="{cy+gy+12}" r="4" fill="#fff"/>',
    ]
    if emotion == "excited":  # star catchlight
        sx, sy = cx + gx + 8, cy + gy - 10
        parts.append(f'<path d="M{sx} {sy-14} L{sx+4} {sy-4} L{sx+14} {sy} L{sx+4} {sy+4} L{sx} {sy+14} '
                     f'L{sx-4} {sy+4} L{sx-14} {sy} L{sx-4} {sy-4} Z" fill="#fff"/>')
    # lids: a skin-coloured cap over the top of the eye
    lid = {"smug": 0.55, "relaxed": 0.45, "angry": 0.35, "sad": 0.25}.get(emotion)
    if lid:
        slant = {"angry": 14 * side * -1, "sad": 12 * side}.get(emotion, 0)
        top = cy - ry - 6
        cut = cy - ry + 2 * ry * lid
        parts.append(f'<path d="M{cx-rx-6} {top} L{cx+rx+6} {top} L{cx+rx+6} {cut - slant} '
                     f'L{cx-rx-6} {cut + slant} Z" fill="{SKIN}"/>')
        parts.append(f'<path d="M{cx-rx-2} {cut + slant} L{cx+rx+2} {cut - slant}" stroke="{INK}" '
                     f'stroke-width="8" stroke-linecap="round"/>')
    else:
        parts.append(f'<path d="M{cx-rx-4} {cy-ry*0.55} Q{cx} {cy-ry-14} {cx+rx+4} {cy-ry*0.55}" fill="none" '
                     f'stroke="{INK}" stroke-width="10" stroke-linecap="round"/>')
    return "\n".join(parts)


def _brow(cx: float, emotion: str, side: int) -> str:
    ang, lift = BROWS[emotion]
    y = 402 - lift
    # inner end is toward the face centre: for the left eye that's +x
    import math
    dx = 34
    dy = math.tan(math.radians(ang)) * dx
    inner_x, outer_x = cx - side * dx, cx + side * dx
    return (f'<path d="M{outer_x} {y - dy} Q{cx} {y - 10} {inner_x} {y + dy}" fill="none" stroke="{HAIR}" '
            f'stroke-width="12" stroke-linecap="round"/>')


def _mouth(emotion: str, mouth: str) -> str:
    if mouth == "closed":
        if CLOSED[emotion] is None:
            return f'<ellipse cx="400" cy="604" rx="13" ry="16" fill="{MOUTH_IN}" stroke="{INK}" stroke-width="5"/>'
        return f'<path d="{CLOSED[emotion]}" fill="none" stroke="{INK}" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>'
    smile = emotion in SMILEY or emotion == "happy"
    frown = emotion in ("sad", "angry")
    if mouth == "round":
        return (f'<ellipse cx="400" cy="606" rx="20" ry="25" fill="{MOUTH_IN}" stroke="{INK}" stroke-width="6"/>'
                f'<ellipse cx="400" cy="620" rx="11" ry="7" fill="{TONGUE}"/>')
    if mouth == "small":
        h = 14
        if smile:
            d = f"M356 596 Q400 600 444 596 Q400 {632} 356 596 Z"
        elif frown:
            d = f"M366 612 Q400 590 434 612 Q400 {616} 366 612 Z"
        else:
            d = f"M362 600 Q400 597 438 600 Q400 {600 + h * 2} 362 600 Z"
        return (f'<path d="{d}" fill="{MOUTH_IN}" stroke="{INK}" stroke-width="6" stroke-linejoin="round"/>'
                f'<path d="M372 601 Q400 604 428 601" fill="none" stroke="#fff" stroke-width="5" stroke-linecap="round"/>')
    # open
    if smile:
        d = "M350 590 Q400 596 450 590 Q446 660 400 664 Q354 660 350 590 Z"
    elif frown:
        d = "M362 618 Q400 572 438 618 Q432 654 400 656 Q368 654 362 618 Z"
    else:
        d = "M358 594 Q400 586 442 594 Q440 652 400 656 Q360 652 358 594 Z"
    return (f'<path d="{d}" fill="{MOUTH_IN}" stroke="{INK}" stroke-width="6" stroke-linejoin="round"/>'
            f'<ellipse cx="400" cy="642" rx="26" ry="11" fill="{TONGUE}"/>'
            f'<path d="M366 597 Q400 602 434 597" fill="none" stroke="#fff" stroke-width="6" stroke-linecap="round"/>')


def _extras(emotion: str) -> str:
    out = []
    if emotion in ("happy", "excited", "nervous", "smug"):
        a = 0.55 if emotion != "smug" else 0.3
        for x in (282, 518):
            out.append(f'<ellipse cx="{x}" cy="560" rx="34" ry="14" fill="{BLUSH}" opacity="{a}"/>')
            for k in (-14, 0, 14):
                out.append(f'<path d="M{x+k-5} 566 L{x+k+3} 552" stroke="{BLUSH}" stroke-width="3" opacity="0.9"/>')
    if emotion == "nervous":
        out.append('<path d="M585 360 Q608 400 596 418 Q584 430 572 418 Q562 402 585 360 Z" fill="#bfe3f5" '
                   f'stroke="{INK}" stroke-width="4"/>')
    if emotion == "sad":
        out.append('<path d="M498 528 Q508 560 498 574 Q488 560 498 528 Z" fill="#bfe3f5" '
                   f'stroke="{INK}" stroke-width="3"/>')
    if emotion == "angry":
        g = (f'<g transform="translate(560 300)" stroke="#d1343f" stroke-width="9" stroke-linecap="round" fill="none">'
             '<path d="M-22 -6 Q-8 -8 -6 -22"/><path d="M6 -22 Q8 -8 22 -6"/>'
             '<path d="M22 6 Q8 8 6 22"/><path d="M-6 22 Q-8 8 -22 6"/></g>')
        out.append(g)
    if emotion == "excited":
        for (x, y, s) in ((170, 300, 1.0), (640, 250, 0.8), (660, 600, 0.6)):
            out.append(f'<path transform="translate({x} {y}) scale({s})" d="M0 -30 L8 -8 L30 0 L8 8 L0 30 L-8 8 '
                       f'L-30 0 L-8 -8 Z" fill="{CREAM}" stroke="{INK}" stroke-width="4"/>')
    if emotion == "surprised":
        out.append(f'<g transform="translate(610 250) rotate(12)"><rect x="-9" y="-46" width="18" height="56" rx="9" '
                   f'fill="{CREAM}" stroke="{INK}" stroke-width="5"/><circle cx="0" cy="34" r="10" fill="{CREAM}" '
                   f'stroke="{INK}" stroke-width="5"/></g>')
    if emotion == "thinking":
        for (x, y, r) in ((600, 360, 10), (628, 318, 15), (664, 268, 22)):
            out.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{CREAM}" stroke="{INK}" stroke-width="4"/>')
    return "\n".join(out)


def svg(emotion: str, mouth: str, blink: bool = False) -> str:
    body = f"""
  <!-- back hair -->
  <path d="M196 480 Q176 236 400 200 Q624 236 604 480 L596 540 Q590 470 560 440 L240 440 Q210 470 204 540 Z" fill="{HAIR}"/>
  <!-- hoodie -->
  <path d="M120 1000 Q120 790 260 745 L400 720 L540 745 Q680 790 680 1000 Z" fill="{HOODIE}" stroke="{INK}" stroke-width="7"/>
  <path d="M250 760 Q400 860 550 760 Q520 735 400 730 Q280 735 250 760 Z" fill="{HOODIE_DARK}" stroke="{INK}" stroke-width="6"/>
  <path d="M352 800 L344 930" stroke="{CREAM}" stroke-width="9" stroke-linecap="round"/>
  <path d="M448 800 L456 930" stroke="{CREAM}" stroke-width="9" stroke-linecap="round"/>
  <circle cx="344" cy="936" r="9" fill="{CREAM}"/><circle cx="456" cy="936" r="9" fill="{CREAM}"/>
  <text x="400" y="985" text-anchor="middle" font-family="Consolas, monospace" font-weight="700" font-size="30" fill="{CREAM}" letter-spacing="4">[REV]</text>
  <!-- neck -->
  <path d="M352 650 L350 760 Q400 790 450 760 L448 650 Z" fill="{SKIN_SHADE}" stroke="{INK}" stroke-width="6"/>
  <!-- ears -->
  <ellipse cx="208" cy="500" rx="26" ry="40" fill="{SKIN}" stroke="{INK}" stroke-width="6"/>
  <ellipse cx="592" cy="500" rx="26" ry="40" fill="{SKIN}" stroke="{INK}" stroke-width="6"/>
  <!-- face -->
  <path d="M215 430 Q212 600 300 660 Q400 715 500 660 Q588 600 585 430 Q585 270 400 262 Q215 270 215 430 Z" fill="{SKIN}" stroke="{INK}" stroke-width="7"/>
  <!-- fringe -->
  <path d="M200 440 Q190 250 400 225 Q610 250 600 440 Q585 360 548 330 Q560 390 520 420 Q500 350 455 330 Q470 380 430 400 Q410 340 360 330 Q370 380 320 410 Q300 350 262 340 Q250 400 200 440 Z" fill="{HAIR}" stroke="{INK}" stroke-width="5" stroke-linejoin="round"/>
  <path d="M300 270 Q360 250 420 262" stroke="{HAIR_HI}" stroke-width="10" fill="none" stroke-linecap="round"/>
  <path d="M400 225 Q420 170 470 160 Q440 200 452 232" fill="{HAIR}" stroke="{INK}" stroke-width="5"/>
  {_brow(320, emotion, -1)}
  {_brow(480, emotion, 1)}
  {_eye(320, emotion, blink, -1)}
  {_eye(480, emotion, blink, 1)}
  <path d="M400 540 Q394 556 402 560" fill="none" stroke="{SKIN_SHADE}" stroke-width="5" stroke-linecap="round"/>
  {_mouth(emotion, mouth)}
  {_extras(emotion)}
"""
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">{body}</svg>'


def build_default_pack(out_dir: Path, chromium: str | None = None, force: bool = False) -> Path:
    """Rasterise every sprite to PNG with the local Chromium (cached)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(e, m, b) for e in EMOTIONS for m in MOUTHS for b in (False, True)]
    todo = [j for j in jobs if force or not (out_dir / _name(*j)).exists()]
    if not todo:
        return out_dir
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        kw = {"executable_path": chromium} if chromium else {}
        browser = pw.chromium.launch(**kw)
        page = browser.new_page(viewport={"width": W, "height": H})
        for e, m, b in todo:
            data = base64.b64encode(svg(e, m, b).encode()).decode()
            page.set_content(f'<html><body style="margin:0;background:transparent">'
                             f'<img src="data:image/svg+xml;base64,{data}"></body></html>')
            page.screenshot(path=str(out_dir / _name(e, m, b)), omit_background=True,
                            clip={"x": 0, "y": 0, "width": W, "height": H})
        browser.close()
    return out_dir


def _name(emotion: str, mouth: str, blink: bool) -> str:
    return f"{emotion}_{mouth}{'_blink' if blink else ''}.png"


def load_pack(folder: Path) -> dict[tuple[str, str, bool], Path]:
    """(emotion, mouth, blink) -> file, with fallbacks so a partial pack works."""
    have = {p.name: p for p in folder.glob("*.png")}
    if "neutral_closed.png" not in have:
        raise FileNotFoundError(f"{folder} needs at least neutral_closed.png")
    out = {}
    for e in EMOTIONS:
        for m in MOUTHS:
            for b in (False, True):
                for cand in (_name(e, m, b), _name(e, m, False), _name("neutral", m, b),
                             _name("neutral", m, False), _name(e, "closed", False), "neutral_closed.png"):
                    if cand in have:
                        out[(e, m, b)] = have[cand]
                        break
    return out
