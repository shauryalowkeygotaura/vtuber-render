"""avatar.py -- a drawn avatar (generated illustration of you) that blinks and talks.

The art is never warped. Each frame:
  eyes   open = the original drawing; closed = the eye painted out (inpaint from
         the surrounding skin) and a curved lash line drawn in the same ink
  mouth  the drawn mouth painted out once, then a mouth shape per viseme drawn
         in the art's own style: thin ink outline, dark interior, teeth, tongue
Placement comes from MediaPipe face landmarks, which find drawn faces fine.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw

from vtuber import photo2d

INK = (24, 16, 18, 255)
INSIDE = (70, 22, 28, 255)
TONGUE = (196, 96, 98, 255)
TEETH = (250, 248, 244, 255)
SS = 3  # supersample for face lines (4 cost 180 ms/frame for no visible gain at reel size)


def key_green(img_bgr: np.ndarray, tol: float = 42) -> np.ndarray:
    """Alpha mask for a flat green backdrop: flood from the borders so green
    inside the character never gets keyed."""
    h, w = img_bgr.shape[:2]
    bg = np.median(np.concatenate([img_bgr[:8].reshape(-1, 3), img_bgr[:, :8].reshape(-1, 3)]), axis=0)
    close = (np.linalg.norm(img_bgr.astype(np.float32) - bg, axis=2) < tol).astype(np.uint8)
    n, lab = cv2.connectedComponents(close)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bgmask = np.isin(lab, list(border))
    # enclosed backdrop (gap between arm and body): big blobs of nearly-exact bg colour
    tight = np.linalg.norm(img_bgr.astype(np.float32) - bg, axis=2) < tol * 0.6
    n2, lab2, stats, _ = cv2.connectedComponentsWithStats(tight.astype(np.uint8))
    for j in range(1, n2):
        if stats[j, cv2.CC_STAT_AREA] > 400:
            bgmask |= lab2 == j
    alpha = np.where(bgmask, 0, 255).astype(np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    alpha = cv2.erode(alpha, np.ones((3, 3), np.uint8))  # drop the green fringe
    return cv2.GaussianBlur(alpha, (3, 3), 0)


def despill(img_bgr: np.ndarray) -> np.ndarray:
    """Pull green spill off edges: green may not exceed the larger of red/blue."""
    out = img_bgr.copy()
    b, g, r = out[..., 0].astype(np.int16), out[..., 1].astype(np.int16), out[..., 2].astype(np.int16)
    out[..., 1] = np.minimum(g, np.maximum(r, b)).astype(np.uint8)
    return out


def find_face(img_bgr: np.ndarray) -> photo2d.Face:
    """Landmarks on the whole image; for full-body art the face is too small, so
    retry on an upscaled crop of the top of the frame and map the points back."""
    try:
        return photo2d.landmarks(img_bgr)
    except ValueError:
        pass
    h, w = img_bgr.shape[:2]
    x0, x1, y1 = int(w * 0.15), int(w * 0.85), int(h * 0.35)
    k = 3.0
    crop = cv2.resize(img_bgr[:y1, x0:x1], None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    f = photo2d.landmarks(crop)
    off = np.array([x0, 0], np.float32)
    return photo2d.Face(f.pts / k + off, f.up, f.right, f.mouth / k + off, f.half_w / k,
                        f.chin_d / k, f.centre / k + off, f.size / k)


@dataclass
class Rig:
    base: np.ndarray       # BGRA art, mouth painted out
    closed: np.ndarray     # BGRA art, mouth AND eyes painted out
    face: photo2d.Face
    eyes: list             # [(outer, inner, top, bottom)] per eye, float pts
    mouth_w: float

    @classmethod
    def build(cls, img_bgr: np.ndarray) -> "Rig":
        face = find_face(img_bgr)
        p = face.pts
        alpha = key_green(img_bgr)
        img_bgr = despill(img_bgr)
        eyes = [tuple(p[j] for j in idx) for idx in (photo2d.EYE_L, photo2d.EYE_R)]
        mw = float(np.linalg.norm(p[photo2d.CORNER_R] - p[photo2d.CORNER_L]))

        def paint_out(src, masks):
            m = np.zeros(src.shape[:2], np.uint8)
            for poly in masks:
                cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 255)
            m = cv2.dilate(m, np.ones((5, 5), np.uint8))
            return cv2.inpaint(src, m, 7, cv2.INPAINT_TELEA)

        # mouth region: the lip outline plus a margin
        lips = np.array([p[i] for i in (61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91)], np.float32)
        c = lips.mean(0)
        lips = c + (lips - c) * np.array([1.25, 1.9])
        no_mouth = paint_out(img_bgr, [lips])
        eye_polys = []
        for o, i, t, b in eyes:
            ctr = (o + i) / 2
            rx = np.linalg.norm(o - i) / 2 * 1.35
            ry = max(np.linalg.norm(t - b) / 2 * 1.9, rx * 0.5)
            ang = np.arctan2(*(i - o)[::-1])
            pts = [ctr + rx * np.cos(a) * np.array([np.cos(ang), np.sin(ang)])
                   + ry * np.sin(a) * np.array([-np.sin(ang), np.cos(ang)]) for a in np.linspace(0, 2 * np.pi, 24)]
            eye_polys.append(np.array(pts))
        no_eyes = no_mouth.copy()
        for poly, (o, i, t, b) in zip(eye_polys, eyes):
            ctr = (o + i) / 2
            span = np.linalg.norm(o - i)
            cy = int(ctr[1] + span * 0.55)
            patch = no_mouth[cy - 3:cy + 4, int(ctr[0] - span * 0.3):int(ctr[0] + span * 0.3)]
            skin = np.median(patch.reshape(-1, 3), axis=0)
            m = np.zeros(no_mouth.shape[:2], np.uint8)
            cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 255)
            m = cv2.GaussianBlur(cv2.dilate(m, np.ones((3, 3), np.uint8)), (7, 7), 0).astype(np.float32)[..., None] / 255
            no_eyes = (no_eyes * (1 - m) + skin * m).astype(np.uint8)
        base = np.dstack([no_mouth, alpha])
        closed = np.dstack([no_eyes, alpha])
        return cls(base, closed, face, eyes, mw)

    # ── drawing ─────────────────────────────────────────────────────────────
    @property
    def _box(self):
        """Drawing area for face features: the head plus room for marks."""
        if not hasattr(self, "_box_cache"):
            c, r = self.face.centre, self.face.size * 1.3
            h, w = self.base.shape[:2]
            self._box_cache = (max(0, int(c[0] - r)), max(0, int(c[1] - r)), min(w, int(c[0] + r)), min(h, int(c[1] + r)))
        return self._box_cache

    def _layer(self):
        x0, y0, x1, y1 = self._box
        im = Image.new("RGBA", ((x1 - x0) * SS, (y1 - y0) * SS), (0, 0, 0, 0))
        return im, ImageDraw.Draw(im)

    # emotion -> (eyes, mouth, marks). eyes: open (the drawing) | arcs | shock | smug | sad | stars | closed
    LOOKS = {
        "neutral":   ("open", "rest", ()),
        "happy":     ("arcs", "grin", ("blush",)),
        "excited":   ("stars", "grin", ("sparkle", "blush")),
        "surprised": ("shock", "o", ("shock",)),
        "angry":     ("open", "grit", ("vein",)),
        "smug":      ("smug", "smirk", ()),
        "sad":       ("sad", "frown", ("gloom",)),
        "nervous":   ("open", "wobble", ("sweat",)),
        "thinking":  ("smug", "flat", ("dots",)),
        "relaxed":   ("arcs", "rest", ()),
    }

    def frame(self, open_: float = 0.0, widen: float = 0.0, blink: float = 0.0, smile: float = 0.3,
              emotion: str = "neutral", t: float = 0.0) -> Image.Image:
        """open_ 0..1 jaw, widen -1 (oo) .. 1 (ee), blink 0..1, smile -1..1, t = seconds (wobble/sparkle)."""
        eyes, mouth_kind, marks = self.LOOKS.get(emotion, self.LOOKS["neutral"])
        if blink > 0.5 and eyes in ("open", "shock", "smug", "sad", "stars"):
            eyes = "closed"
        src = self.base if eyes == "open" else self.closed
        art = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGRA2RGBA))
        lay, d = self._layer()
        k = SS
        lw = 2.4 * k

        def Q(v):
            return ((float(v[0]) - self._box[0]) * k, (float(v[1]) - self._box[1]) * k)

        def bez(a, c, b, n=18):
            return [Q((1 - s) ** 2 * a + 2 * (1 - s) * s * c + s ** 2 * b) for s in np.linspace(0, 1, n)]

        for o, i, tp, bt in self.eyes:
            ctr = (o + i) / 2
            span = float(np.linalg.norm(o - i))
            hgt = float(np.linalg.norm(tp - bt))
            if eyes == "closed":
                d.line(bez(o, ctr + [0, hgt * 0.25 + 6], i), fill=INK, width=int(lw * 1.4), joint="curve")
            elif eyes == "arcs":   # ^ ^
                d.line(bez(o + [0, 3], ctr + [0, -hgt * 1.4], i + [0, 3]), fill=INK, width=int(lw * 1.6), joint="curve")
            elif eyes == "sad":    # droopy closed lid + worried brow, inner end high
                d.line(bez(o + [0, 2], ctr + [0, hgt * 0.7], i + [0, -3]), fill=INK, width=int(lw * 1.4), joint="curve")
            elif eyes in ("shock", "stars"):
                r = span * 0.5
                c = ctr + [0, -hgt * 0.2]
                d.ellipse((*Q(c - [r, r * 1.15]), *Q(c + [r, r * 1.15])), fill=TEETH, outline=INK, width=int(lw * 1.3))
                if eyes == "shock":
                    pr = r * 0.16
                    d.ellipse((*Q(c - pr), *Q(c + pr)), fill=INK)
                else:
                    pts = []
                    for j in range(10):
                        a = -np.pi / 2 + j * np.pi / 5
                        rr = r * (0.75 if j % 2 == 0 else 0.32) * (1 + 0.08 * np.sin(t * 14))
                        pts.append(Q(c + rr * np.array([np.cos(a), np.sin(a)])))
                    d.polygon(pts, fill=(255, 196, 40, 255), outline=INK)
            elif eyes == "smug":   # half-lidded: flat heavy lid over the lower half of the eye
                c = ctr + [0, hgt * 0.1]
                d.chord((*Q(c - [span * 0.5, hgt * 0.7]), *Q(c + [span * 0.5, hgt * 0.7])), 0, 180,
                        fill=TEETH, outline=INK, width=int(lw))
                pr = hgt * 0.6
                d.chord((*Q(c - [pr, pr]), *Q(c + [pr, pr])), 0, 180, fill=(40, 26, 24, 255))
                d.line([Q(o + [-2, hgt * 0.1]), Q(i + [1, hgt * 0.1])], fill=INK, width=int(lw * 1.8))

        self._mouth(d, mouth_kind, open_, widen, smile, t)
        self._marks(d, marks, t)
        x0, y0, x1, y1 = self._box
        lay = lay.resize((x1 - x0, y1 - y0), Image.LANCZOS)
        art.alpha_composite(lay, (x0, y0))
        return art

    def _mouth(self, d, kind, open_, widen, smile, t):
        k = SS
        lw = 2.4 * k
        m = self.face.mouth
        right = self.face.right
        down = np.array([-right[1], right[0]])

        def P(u, v):
            q = m + right * u + down * v
            return ((q[0] - self._box[0]) * k, (q[1] - self._box[1]) * k)

        W = self.mouth_w / 2 * (1 + 0.22 * widen)
        if kind == "grin":
            open_, smile, W = max(open_, 0.45), 1.0, W * 1.25
        elif kind == "o":
            open_, widen = max(open_, 0.7), -1
        elif kind == "frown":
            smile = -1.0
        if widen < -0.3:
            W *= 0.62
        Hh = self.mouth_w * (0.06 + 0.62 * open_) * (1 - 0.3 * max(0, widen))
        xs = np.linspace(-W, W, 24)

        if kind == "grit":  # clenched teeth, wide
            Wg, Hg = W * 1.15, max(8.0, Hh * 0.6)
            box = [P(u, -Hg / 2 + 1.5 * (u / Wg) ** 2) for u in np.linspace(-Wg, Wg, 16)] + \
                  [P(u, Hg / 2 - 1.5 * (u / Wg) ** 2) for u in np.linspace(Wg, -Wg, 16)]
            d.polygon(box, fill=TEETH)
            d.line(box + [box[0]], fill=INK, width=int(lw))
            d.line([P(-Wg, 0), P(Wg, 0)], fill=INK, width=int(lw * 0.7))
            for u in np.linspace(-Wg * 0.6, Wg * 0.6, 4):
                d.line([P(u, -Hg / 2 + 1), P(u, Hg / 2 - 1)], fill=INK, width=int(lw * 0.6))
            return
        if open_ < 0.1:
            if kind == "smirk":  # one corner up
                pts = [P(u, -max(0.0, u / W) ** 2 * 7 + 1.5 * (1 - (u / W) ** 2)) for u in xs]
            elif kind == "wobble":
                pts = [P(u, 2.2 * np.sin(u / W * 9 + t * 20)) for u in xs]
            elif kind == "flat":
                pts = [P(-W * 0.7, 0), P(W * 0.6, -1)]
            else:
                pts = [P(u, smile * 4 * (1 - (u / W) ** 2) - smile * 2) for u in xs]
            d.line(pts, fill=INK, width=int(lw), joint="curve")
            return
        frown = smile < -0.3
        if frown:   # arched top, shallow bottom
            top = [P(u, -Hh * 0.5 * (1 - (u / W) ** 2) + 3 * (u / W) ** 2) for u in xs]
            bot = [P(u, Hh * 0.6 * (1 - (u / W) ** 2) ** 0.7) for u in xs[::-1]]
        else:
            top = [P(u, -Hh * 0.15 * (1 - (u / W) ** 2) - smile * 3 * (u / W) ** 2) for u in xs]
            bot = [P(u, Hh * (1 - (u / W) ** 2) ** 0.7) for u in xs[::-1]]
        d.polygon(top + bot, fill=INSIDE)
        if not frown and open_ > 0.3:
            tw = W * 0.55
            tong = [P(u, Hh * (0.95 - 0.35 * (1 - (u / tw) ** 2))) for u in np.linspace(-tw, tw, 16)] + \
                   [P(u, Hh * (1 - (u / W) ** 2) ** 0.7 - 1) for u in np.linspace(tw, -tw, 16)]
            d.polygon(tong, fill=TONGUE)
        if not frown and open_ > 0.2 and widen > -0.3:
            tw = W * 0.75
            teeth = [P(u, -Hh * 0.15 * (1 - (u / W) ** 2) - smile * 3 * (u / W) ** 2 + 1) for u in np.linspace(-tw, tw, 16)] + \
                    [P(u, min(Hh * 0.18, 6)) for u in np.linspace(tw, -tw, 16)]
            d.polygon(teeth, fill=TEETH)
        d.line(top + bot + [top[0]], fill=INK, width=int(lw), joint="curve")

    def _marks(self, d, marks, t):
        """Manhwa reaction marks around the face."""
        k = SS
        f = self.face
        up, right = f.up, f.right
        s = f.size
        bob = np.sin(t * 7) * s * 0.02

        def Q(v):
            return ((float(v[0]) - self._box[0]) * k, (float(v[1]) - self._box[1]) * k)

        for mk in marks:
            if mk == "blush":
                for (o, i, _, _) in self.eyes:
                    c = (o + i) / 2 - up * s * 0.17
                    for j in range(-2, 3):
                        a = c + right * j * s * 0.035
                        d.line([Q(a + right * s * 0.02 - up * s * 0.02), Q(a - right * s * 0.01 + up * s * 0.02)],
                               fill=(214, 70, 80, 230), width=int(2.6 * k))
            elif mk == "sweat":
                c = f.centre + right * s * 0.62 + up * s * 0.3 - up * bob * 3
                r = s * 0.06
                drop = [Q(c + up * r * 2.2)] + [Q(c + r * np.array([np.cos(a), np.sin(a)]))
                                                for a in np.linspace(-0.3, np.pi + 0.3, 14)]
                d.polygon(drop, fill=(150, 210, 255, 255))
                d.line(drop + [drop[0]], fill=INK, width=int(2.4 * k))
            elif mk == "vein":
                c = f.centre + up * s * 0.62 + right * s * 0.45
                r = s * 0.09 * (1 + 0.12 * np.sin(t * 16))
                for ang in (45, 135, 225, 315):
                    a = np.radians(ang)
                    p0 = c + r * 0.35 * np.array([np.cos(a), np.sin(a)])
                    p1 = c + r * np.array([np.cos(a + 0.5), np.sin(a + 0.5)])
                    p2 = c + r * np.array([np.cos(a - 0.5), np.sin(a - 0.5)])
                    d.line([Q(p1), Q(p0), Q(p2)], fill=(225, 40, 50, 255), width=int(5 * k), joint="curve")
            elif mk == "shock":   # vertical lines on the forehead
                for j in range(-3, 4):
                    a = f.centre + up * s * 0.52 + right * j * s * 0.07
                    d.line([Q(a), Q(a - up * s * (0.12 + 0.03 * (j % 2)))], fill=(60, 70, 140, 200), width=int(3 * k))
            elif mk == "gloom":   # dark hatching over the upper face
                for j in range(-6, 7):
                    a = f.centre + up * s * 0.32 + right * j * s * 0.04
                    d.line([Q(a), Q(a - up * s * 0.22)], fill=(40, 30, 70, 150), width=int(2.2 * k))
            elif mk == "sparkle":
                for (dx, dy, rr, ph) in ((0.7, 0.55, 0.09, 0), (-0.75, 0.4, 0.06, 1.7), (0.85, 0.1, 0.05, 3.1)):
                    c = f.centre + right * s * dx + up * s * dy
                    r = s * rr * (0.75 + 0.25 * np.sin(t * 9 + ph))
                    pts = [Q(c + (r if j % 2 == 0 else r * 0.25) * np.array([np.cos(j * np.pi / 4), np.sin(j * np.pi / 4)]))
                           for j in range(8)]
                    d.polygon(pts, fill=(255, 236, 160, 255), outline=INK)
            elif mk == "dots":    # ... thinking
                for j in range(3):
                    c = f.centre + right * s * (0.62 + 0.11 * j) + up * s * (0.5 + 0.05 * j)
                    r = s * 0.025 * (1.3 if int(t * 3) % 3 == j else 1)
                    d.ellipse((*Q(c - r), *Q(c + r)), fill=TEETH, outline=INK, width=int(2 * k))
