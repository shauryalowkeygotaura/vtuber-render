"""cartoon.py -- real cut-out photo, drawn cartoon face on top.

The photo never warps (that was what looked uncanny). Instead:
  cut-out   rembg removes the background, a white sticker outline goes round you
  eyes      big drawn circle eyes over your real ones; blink = squash to a line
  mouth     a drawn cartoon mouth over your real one, shape per viseme
Placement comes from MediaPipe face landmarks (photo2d.landmarks), so the
features follow each pose photo's head size and tilt.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from vtuber import photo2d

INK = (22, 18, 20, 255)
WHITE = (255, 255, 255, 255)
MOUTH_IN = (92, 18, 30, 255)
TONGUE = (226, 92, 110, 255)
SS = 4  # supersample factor for smooth outlines


def cutout(img_bgr: np.ndarray) -> Image.Image:
    from rembg import new_session, remove
    rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return remove(rgb, session=new_session("u2net_human_seg"), post_process_mask=True)


def sticker(cut: Image.Image, px: int = 14) -> Image.Image:
    """White outline round the cut-out, like a sticker."""
    a = cut.getchannel("A").point(lambda v: 255 if v > 128 else 0)
    ring = a.filter(ImageFilter.MaxFilter(px * 2 + 1)).filter(ImageFilter.GaussianBlur(1.2))
    out = Image.new("RGBA", cut.size, (255, 255, 255, 0))
    out.putalpha(ring)
    out.alpha_composite(cut)
    return out


def _patch(size: int):
    im = Image.new("RGBA", (size * SS, size * SS), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)


def _paste(dst: Image.Image, patch: Image.Image, centre, angle_deg: float):
    p = patch.resize((patch.width // SS, patch.height // SS), Image.LANCZOS).rotate(angle_deg, Image.BICUBIC, expand=True)
    dst.alpha_composite(p, (int(centre[0] - p.width / 2), int(centre[1] - p.height / 2)))


def eye(r: float, open_: float = 1.0, look=(0.0, 0.0), emotion: str = "neutral") -> Image.Image:
    """One eye, radius r px. open_ 0 = closed line, 1 = full circle."""
    S = int(r * 3)
    im, d = _patch(S)
    c = S * SS / 2
    R = r * SS
    lw = max(3, int(r * 0.16)) * SS
    if emotion in ("happy", "excited") and open_ > 0.3:
        # ^ ^ happy arcs
        d.arc((c - R, c - R * 0.6, c + R, c + R * 1.4), 200, 340, fill=INK, width=lw)
        return im
    ry = R * max(0.06, open_)
    if open_ < 0.15:
        d.line((c - R, c, c + R, c), fill=INK, width=lw)
        return im
    d.ellipse((c - R, c - ry, c + R, c + ry), fill=WHITE, outline=INK, width=lw)
    pr = R * (0.30 if emotion == "surprised" else 0.46)
    px, py = c + look[0] * R * 0.35, c + look[1] * ry * 0.35
    pry = min(pr, ry * 0.9)
    d.ellipse((px - pr, py - pry, px + pr, py + pry), fill=INK)
    hl = pr * 0.32
    d.ellipse((px - pr * 0.45 - hl, py - pry * 0.45 - hl, px - pr * 0.45 + hl, py - pry * 0.45 + hl), fill=WHITE)
    return im


def mouth(w: float, open_: float = 0.0, widen: float = 0.0, emotion: str = "neutral") -> Image.Image:
    """Cartoon mouth, w = width in px. open_ 0..1, widen -1 (oo) .. 1 (ee)."""
    S = int(w * 1.8)
    im, d = _patch(S)
    c = S * SS / 2
    W = w * SS / 2 * (1 + 0.25 * widen)
    lw = max(3, int(w * 0.07)) * SS
    smile = {"happy": 1, "excited": 1, "smug": 0.6, "relaxed": 0.5, "sad": -0.8, "angry": -0.5}.get(emotion, 0.25)
    if open_ < 0.12:
        # closed: a curved line
        bend = smile * W * 0.45
        d.arc((c - W, c - W * 0.6 - bend, c + W, c + W * 0.6 - bend), 20 if smile >= 0 else 200,
              160 if smile >= 0 else 340, fill=INK, width=lw) if abs(smile) > 0.05 else \
            d.line((c - W, c, c + W, c), fill=INK, width=lw)
        return im
    H = W * (0.25 + 1.1 * open_) * (1 - 0.35 * widen)
    if widen < -0.3:  # oo / oh: round
        W = W * 0.6
    top = c - H * 0.35
    # D-shaped open mouth: flat-ish top, round bottom (happy) or round both
    box = (c - W, top - H * 0.3, c + W, top + H)
    d.chord(box, 0, 180, fill=MOUTH_IN, outline=INK, width=lw) if smile > 0.3 else \
        d.ellipse((c - W, top, c + W, top + H), fill=MOUTH_IN, outline=INK, width=lw)
    if open_ > 0.35:
        ty = top + H * 0.72
        d.ellipse((c - W * 0.5, ty - H * 0.18, c + W * 0.5, top + H * 0.98), fill=TONGUE)
    if smile > 0.3:
        d.rectangle((c - W * 0.7, top - H * 0.02 + lw / 2, c + W * 0.7, top + H * 0.12), fill=WHITE)
        d.line((c - W, top - H * 0.02 + lw / 4, c + W, top - H * 0.02 + lw / 4), fill=INK, width=lw)
    return im


def face_geom(face: photo2d.Face):
    p = face.pts
    eyes = []
    for o, i, t, b in (photo2d.EYE_L, photo2d.EYE_R):
        eyes.append(((p[o] + p[i]) / 2, float(np.linalg.norm(p[o] - p[i]))))
    gap = float(np.linalg.norm(eyes[0][0] - eyes[1][0]))
    angle = -math.degrees(math.atan2(face.right[1], face.right[0]))
    return eyes, gap, angle


def draw_face(img: Image.Image, face: photo2d.Face, offset=(0, 0), scale: float = 1.0, *,
              eye_open: float = 1.0, look=(0.0, 0.0), mouth_open: float = 0.0, widen: float = 0.0,
              emotion: str = "neutral"):
    eyes, gap, angle = face_geom(face)
    ox, oy = offset
    r = gap * 0.42 * scale  # eye radius: big, nearly touching
    for (ctr, _), side in zip(eyes, (-1, 1)):
        c = (ox + ctr[0] * scale, oy + ctr[1] * scale - r * 0.1)
        _paste(img, eye(r, eye_open, look, emotion), c, angle)
    m = face.mouth * scale + (ox, oy)
    _paste(img, mouth(gap * 0.95 * scale, mouth_open, widen, emotion), m, angle)


def toonify(cut: Image.Image, colours: int = 9, line: int = 3) -> Image.Image:
    """Turn the cut-out photo into a flat cartoon: smooth, few flat colours, ink lines.
    Run it AFTER scaling to final size so lines stay crisp."""
    rgba = np.asarray(cut)
    rgb, a = rgba[..., :3].copy(), rgba[..., 3]
    sm = rgb
    for _ in range(4):
        sm = cv2.bilateralFilter(sm, 9, 60, 9)
    sm = cv2.medianBlur(sm, 7)
    lab = cv2.cvtColor(sm, cv2.COLOR_RGB2LAB)
    m = a > 128
    px = lab[m].reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    sample = px[np.random.default_rng(0).choice(len(px), min(len(px), 60000), replace=False)]
    _, _, centres = cv2.kmeans(sample, colours, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    d = ((px[:, None, :] - centres[None]) ** 2).sum(-1)
    q = lab.copy()
    q[m] = centres[d.argmin(1)].astype(np.uint8)
    flat = cv2.cvtColor(q, cv2.COLOR_LAB2RGB)
    flat = cv2.medianBlur(flat, 5)
    # ink: edges between colour regions + the silhouette
    g = cv2.cvtColor(flat, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(g, 30, 90)
    edges |= cv2.Canny(a, 50, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line, line)))
    flat[edges > 0] = INK[:3]
    # crisp alpha (no soft fringe), then the outer ink line
    a2 = np.where(a > 128, 255, 0).astype(np.uint8)
    out = np.dstack([flat, a2])
    return Image.fromarray(out, "RGBA")
