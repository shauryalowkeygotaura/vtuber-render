"""puppet.py -- cut-out puppet from one full-body drawing.

rig-base.png + rig-base.json (joint pixels) -> pieces with pivots:
  legs+torso (root), head (pivot = neck), per arm: upper (shoulder), fore (elbow), hand (wrist)
Every frame the face is drawn on the whole art by avatar.Rig (expressions, talk,
blink), then each piece is cut from that frame with a fixed mask and placed by a
chain of 2D transforms: root -> shoulder -> elbow -> wrist. Joints get round caps
so bends never show a gap, and cut edges get an ink line like the drawing's own.

Pose = dict of angles in degrees (0 = the drawing's rest pose):
  head, A_shoulder, A_elbow, A_wrist, B_shoulder, B_elbow, B_wrist
  For both arms, + raises the arm OUT to that arm's side and + elbow bends the
  forearm up and inward, so poses can be mirrored by swapping A/B.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from vtuber.avatar import Rig, key_green

INK = (24, 16, 18)


def _seg_dist(xx, yy, a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    ab = b - a
    t = np.clip(((xx - a[0]) * ab[0] + (yy - a[1]) * ab[1]) / float(ab @ ab), 0, 1)
    return np.hypot(xx - (a[0] + t * ab[0]), yy - (a[1] + t * ab[1]))


def _rot(pivot, deg):
    """2x3 -> 3x3 rotation about pivot, + = counter-clockwise on screen."""
    m = cv2.getRotationMatrix2D((float(pivot[0]), float(pivot[1])), deg, 1.0)
    return np.vstack([m, [0, 0, 1]])


def _widen(a, b, f: float) -> np.ndarray:
    """3x3 scale by f perpendicular to the line a->b (1 = no change)."""
    if abs(f - 1) < 1e-3:
        return np.eye(3)
    d = np.subtract(b, a).astype(np.float64)
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    L = np.eye(2) + (f - 1) * np.outer(n, n)
    M = np.eye(3)
    M[:2, :2] = L
    M[:2, 2] = np.asarray(a, np.float64) - L @ np.asarray(a, np.float64)
    return M


@dataclass
class Puppet:
    rig: Rig
    j: dict
    masks: dict                       # piece -> float32 alpha (0..1), full art size
    caps: dict                        # piece -> (centre, radius, colour BGR) drawn under the piece
    order: list = field(default_factory=list)
    bbox: dict = field(default_factory=dict)

    @classmethod
    def build(cls, art_path: str | Path) -> "Puppet":
        art_path = Path(art_path)
        img = cv2.imread(str(art_path), cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4:
            img = img[..., :3]
        j = json.loads(art_path.with_suffix(".json").read_text())
        rig = Rig.build(img)
        # keep only the character (drops stray marks like a generator's sparkle logo)
        a = key_green(img)
        n, lab, st, _ = cv2.connectedComponentsWithStats((a > 128).astype(np.uint8))
        keep = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        fg = lab == keep
        rig.base[..., 3] = np.where(fg, rig.base[..., 3], 0)
        rig.closed[..., 3] = np.where(fg, rig.closed[..., 3], 0)

        h, w = fg.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = j["radius"]
        bones = {"torso": (j["torso"], r["torso"]), "legs": (j["legs"], r["legs"])}
        for s in "AB":
            S = j[s]
            bones[f"{s}_upper"] = ((S["shoulder"], S["elbow"]), r["upper"])
            bones[f"{s}_fore"] = ((S["elbow"], S["wrist"]), r["fore"])
            bones[f"{s}_hand"] = ((S["wrist"], S["tip"]), r["hand"])
        names = list(bones)
        D = np.stack([_seg_dist(xx, yy, *bones[k][0]) / bones[k][1] for k in names])
        lab = np.argmin(D, 0)
        # head: everything above the neck line within the head's width
        face = rig.face
        chin_y = float(face.pts[152][1]) + 4
        head = fg & (yy < chin_y) & (np.abs(xx - j["neck"][0]) < 190)
        # the body keeps a 4 px strip under the head's bottom edge, so a head tilt never opens a gap
        head_core = cv2.erode(head.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        masks = {}
        for i, k in enumerate(names):
            masks[k] = fg & (lab == i) & ~(head_core if k in ("torso", "legs") else head)
        masks["torso"] |= masks.pop("legs")
        # stray specks of hand/watch that ended up labelled torso would stay pinned at the hip when
        # the arm lifts: keep only the torso's main body
        n_, lab_, st_, _ = cv2.connectedComponentsWithStats(masks["torso"].astype(np.uint8))
        if n_ > 2:
            masks["torso"] = lab_ == 1 + int(np.argmax(st_[1:, cv2.CC_STAT_AREA]))
        masks["head"] = head

        caps = {}
        for s in "AB":
            S = j[s]
            # shoulder: a sleeve-coloured ball; elbow: a skin ball under the cuff; wrist: skin ball
            def colour(pt, dy):
                y, x = int(pt[1] + dy), int(pt[0])
                return tuple(int(c) for c in np.median(img[y - 2:y + 3, x - 2:x + 3].reshape(-1, 3), 0))
            caps[f"{s}_upper"] = (S["shoulder"], r["upper"] * 0.95, colour(S["shoulder"], 20))
            caps[f"{s}_fore"] = (S["elbow"], r["fore"] * 0.95, colour(S["elbow"], 25))
            caps[f"{s}_hand"] = (S["wrist"], r["hand"] * 0.62, colour(S["wrist"], 18))
            # pieces overlap a little at the joints so seams never open
            for p in ("upper", "fore", "hand"):
                m = masks[f"{s}_{p}"].astype(np.uint8)
                masks[f"{s}_{p}"] = (cv2.dilate(m, np.ones((5, 5), np.uint8)) > 0) & fg
        fm = {k: cv2.GaussianBlur(v.astype(np.float32), (3, 3), 0) for k, v in masks.items()}
        order = ["torso", "A_fore", "B_fore", "A_upper", "B_upper", "A_hand", "B_hand", "head"]  # cuffs overlap forearms
        p = cls(rig, j, fm, caps, order)
        p._ink_cuts(fg)
        p.bbox = {}
        for k, m in fm.items():
            ys, xs = np.nonzero(m > 0.01)
            p.bbox[k] = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
        return p

    def _ink_cuts(self, fg):
        """Cut edges inside the silhouette get an ink stroke (stored per piece)."""
        self.cut_ink = {}
        for k, m in self.masks.items():
            if k != "torso":  # arm joints are hidden by caps; only the torso's shoulder cut needs a line
                self.cut_ink[k] = np.zeros(m.shape, bool)
                continue
            mb = (m > 0.5).astype(np.uint8)
            edge = mb - cv2.erode(mb, np.ones((3, 3), np.uint8))
            inner = edge.astype(bool) & cv2.erode(fg.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            self.cut_ink[k] = cv2.dilate(inner.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)

    # ── posing ──────────────────────────────────────────────────────────────
    def _mech(self, out, M, side, kind, scale: float = 1.0, launch: float = 0.0, phase: int = 0):
        """Draw a mechanical hand / tool in place of the drawn one, riding the wrist transform.
        scale pops it in on a swap; launch pushes it out along the forearm (rocket fist)."""
        from vtuber.mechhand import TOOLS, hand  # noqa: PLC0415
        S = self.j[side]
        L = float(np.linalg.norm(np.subtract(S["tip"], S["wrist"]))) * 1.15
        phase = phase % 4 if kind in TOOLS else 0
        key = (side, kind, phase)
        cache = self.__dict__.setdefault("_mech_cache", {})
        if key not in cache:
            im, wr = hand(kind, L, mirror=(side == "A"), phase=phase)
            a = np.asarray(im).astype(np.float32)[..., [2, 1, 0, 3]]
            a[..., :3] *= a[..., 3:4] * (1 / 255.0)   # premultiplied, like every sprite _blit takes
            cache[key] = (a, wr)
        spr, wr = cache[key]
        if scale <= 0.02:
            return
        if kind in TOOLS:
            scale *= 1.9   # tools are props: big enough to read at phone size
        wx, wy = S["wrist"]
        d = np.subtract(S["tip"], S["wrist"]).astype(np.float64)
        d /= np.linalg.norm(d)
        # sprite px -> art px: wrist on the drawing's wrist, scaled about it, pushed along the hand
        to_art = np.array([[scale, 0, wx - wr[0] * scale + d[0] * launch],
                           [0, scale, wy - wr[1] * scale + d[1] * launch], [0, 0, 1]], np.float32)
        M3 = np.vstack([M, [0, 0, 1]]) @ to_art
        _blit(out, spr, M3[:2])

    def transforms(self, pose: dict) -> dict:
        """piece -> 3x3 matrix in art space (before the root/camera transform)."""
        g = lambda k: float(pose.get(k, 0.0))  # noqa: E731
        T = {"torso": np.eye(3), "head": _rot(self.j["neck"], g("head"))}
        for s, sign in (("A", -1), ("B", 1)):
            S = self.j[s]
            up = _rot(S["shoulder"], sign * g(f"{s}_shoulder"))
            fo = up @ _rot(S["elbow"], sign * g(f"{s}_elbow"))
            ha = fo @ _rot(S["wrist"], sign * g(f"{s}_wrist"))
            # bicep bulge: widen the upper arm across its own bone (points on the bone stay put,
            # so shoulder and elbow don't move); only the upper piece gets it, not the chain
            T[f"{s}_upper"] = up @ _widen(S["shoulder"], S["elbow"], 1 + 0.5 * g(f"{s}_bulge"))
            T[f"{s}_fore"], T[f"{s}_hand"] = fo, ha
        return T

    def _piece(self, k: str, art: np.ndarray | None) -> np.ndarray:
        """Premultiplied BGRA float sprite for piece k. Body pieces never change (only the face
        does), so they're cut once and cached; the head is re-cut from this frame's art."""
        cache = self.__dict__.setdefault("_piece_cache", {})
        if k != "head" and k in cache:
            return cache[k]
        y0, y1, x0, x1 = self.bbox[k]
        src = art if art is not None else self._rest_art()
        piece = src[y0:y1, x0:x1].astype(np.float32)[..., [2, 1, 0, 3]]
        piece[..., 3] *= self.masks[k][y0:y1, x0:x1]
        ink = self.cut_ink[k][y0:y1, x0:x1] & (self.masks[k][y0:y1, x0:x1] > 0.5)
        piece[ink] = (*INK, 255)
        piece[..., :3] *= piece[..., 3:4] * (1 / 255.0)
        if k != "head":
            cache[k] = piece
        return piece

    def _rest_art(self) -> np.ndarray:
        if not hasattr(self, "_rest"):
            self._rest = np.asarray(self.rig.frame())
        return self._rest

    def render(self, pose: dict, root: np.ndarray, size: tuple[int, int], face: dict | None = None,
               canvas: np.ndarray | None = None) -> np.ndarray:
        """Draw the posed puppet. root = 3x3 art->canvas matrix. Returns BGRA float canvas."""
        W, H = size
        out = np.zeros((H, W, 4), np.float32) if canvas is None else canvas
        T = self.transforms(pose)
        for k in self.order:
            M = (root @ T[k])[:2]
            kind = pose.get(f"{k[0]}_hand_kind") if k.endswith("_hand") else None
            if kind and kind != "real":
                sd = k[0]
                self._mech(out, M, sd, kind, float(pose.get(f"{sd}_hand_scale", 1.0)),
                           float(pose.get(f"{sd}_hand_launch", 0.0)), int(pose.get(f"{sd}_hand_phase", 0)))
                continue
            if k in self.caps:  # joint ball first, under the piece, with a soft shadow rim
                c, r, col = self.caps[k]
                cc = M @ np.array([c[0], c[1], 1.0])
                s = math.sqrt(abs(np.linalg.det(M[:, :2])))
                shade = col if k.endswith("_upper") else tuple(int(v * 0.8) for v in col)  # shoulder: plain sleeve
                cv2.circle(out, (int(cc[0]), int(cc[1])), int(r * s), (*shade, 255), -1, cv2.LINE_AA)
                cv2.circle(out, (int(cc[0]), int(cc[1])), int(r * s * 0.8), (*col, 255), -1, cv2.LINE_AA)
            art = np.asarray(self.rig.frame(**(face or {}))) if k == "head" else None
            y0, x0 = self.bbox[k][0], self.bbox[k][2]
            off = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], np.float32)
            _blit(out, self._piece(k, art), (np.vstack([M, [0, 0, 1]]) @ off)[:2])
        return out


def _blit(out: np.ndarray, src: np.ndarray, M: np.ndarray):
    """Warp a PREMULTIPLIED BGRA float sprite by 2x3 M and composite it onto out
    (over operator), touching only the destination box the warped sprite covers."""
    h, w = src.shape[:2]
    corners = M @ np.array([[0, w, 0, w], [0, 0, h, h], [1, 1, 1, 1]], np.float32)
    H, W = out.shape[:2]
    X0, Y0 = max(0, int(corners[0].min()) - 2), max(0, int(corners[1].min()) - 2)
    X1, Y1 = min(W, int(corners[0].max()) + 3), min(H, int(corners[1].max()) + 3)
    if X1 <= X0 or Y1 <= Y0:
        return
    Ml = M.copy()
    Ml[0, 2] -= X0
    Ml[1, 2] -= Y0
    warped = cv2.warpAffine(src, Ml, (X1 - X0, Y1 - Y0), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    roi = out[Y0:Y1, X0:X1]
    a = warped[..., 3] * (1 / 255.0)
    inv = cv2.merge([1 - a, 1 - a, 1 - a])
    roi[..., :3] = cv2.add(cv2.multiply(np.ascontiguousarray(roi[..., :3]), inv), np.ascontiguousarray(warped[..., :3]))
    roi[..., 3] = np.maximum(roi[..., 3], warped[..., 3])
