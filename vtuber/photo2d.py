"""photo2d.py -- make one still photo talk: a 2D rig built from face landmarks.

No 3D model, no drawn sprites: MediaPipe finds the face in the photo once,
then every frame is the same photo with a few local warps.

  mouth   jaw-drop warp below the lip seam + a painted mouth interior and teeth,
          widened for ee/ih, pinched for oh/ou (visemes from lipsync.py)
  blink   lids painted over each eye with skin sampled just above it
  face    smile / frown corners, brow raise / knit, per emotion
  camera  handheld drift from motion.py's head track, punch-in on big beats
  extras  small emotion marks near the head (sparkle, sweat, !, ?, anger)

Works on any photo with one clearly visible face. Small faces still work
because the camera punches in, but a selfie with the face filling a third of
the frame looks best.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL = Path(__file__).resolve().parent.parent / "assets" / "models" / "face_landmarker.task"

EMOTIONS = ("neutral", "happy", "excited", "sad", "angry", "surprised",
            "thinking", "smug", "relaxed", "nervous")

# MediaPipe face-mesh indices
UPPER_IN = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)
LOWER_IN = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308)
EYE_L = (33, 133, 159, 145)   # outer, inner, top, bottom (subject's right eye, viewer's left)
EYE_R = (263, 362, 386, 374)
BROW_L, BROW_R = 105, 334
CORNER_L, CORNER_R = 61, 291
CHIN, FOREHEAD, NOSE = 152, 10, 1

# per-emotion face shape: smile (+up / -down), brow lift (px-ish units), one-sided smile
SHAPE = {
    "neutral":   (0.0, 0.0, 0),  "happy":   (1.0, 0.25, 0), "excited": (1.0, 0.9, 0),
    "sad":       (-0.8, 0.35, 0), "angry":   (-0.5, -0.8, 0), "surprised": (0.0, 1.0, 0),
    "thinking":  (-0.1, 0.5, 1),  "smug":    (0.8, -0.2, 1), "relaxed": (0.5, 0.0, 0),
    "nervous":   (0.3, 0.6, 0),
}
GRADE = {  # BGR multipliers, subtle
    "happy": (0.97, 1.0, 1.04), "excited": (0.96, 1.0, 1.06), "sad": (1.06, 0.99, 0.94),
    "angry": (0.94, 0.96, 1.08), "relaxed": (0.98, 1.0, 1.02),
}


@dataclass
class Face:
    pts: np.ndarray        # (468+, 2) pixel coords in the photo
    up: np.ndarray         # unit vector chin -> forehead
    right: np.ndarray      # unit vector across the face (corner L -> R)
    mouth: np.ndarray      # lip-seam centre
    half_w: float          # half mouth width
    chin_d: float          # seam -> chin distance along -up
    centre: np.ndarray     # face centre
    size: float            # forehead -> chin distance


def landmarks(img_bgr: np.ndarray) -> Face:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    if not MODEL.exists():
        raise FileNotFoundError(
            f"{MODEL} missing. Download: https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
    det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)), num_faces=1))
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
    if not res.face_landmarks:
        raise ValueError("no face found in the photo")
    h, w = img_bgr.shape[:2]
    pts = np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]], dtype=np.float32)
    up = pts[FOREHEAD] - pts[CHIN]
    size = float(np.linalg.norm(up))
    up /= size
    right = pts[CORNER_R] - pts[CORNER_L]
    half_w = float(np.linalg.norm(right)) / 2
    right /= np.linalg.norm(right)
    mouth = (pts[13] + pts[14]) / 2
    chin_d = float(np.dot(mouth - pts[CHIN], up))
    centre = (pts[FOREHEAD] + pts[CHIN]) / 2
    return Face(pts, up, right, mouth, half_w, chin_d, centre, size)


def _smooth(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


class Rig:
    """Precomputes the face ROI and per-pixel face coordinates once."""

    def __init__(self, img_bgr: np.ndarray, face: Face):
        self.img = img_bgr
        self.f = face
        h, w = img_bgr.shape[:2]
        r = face.size * 0.9
        x0, y0 = (face.centre - r).astype(int)
        x1, y1 = (face.centre + r).astype(int)
        self.x0, self.y0 = max(0, x0), max(0, y0)
        self.x1, self.y1 = min(w, x1), min(h, y1)
        ys, xs = np.mgrid[self.y0:self.y1, self.x0:self.x1].astype(np.float32)
        self.X, self.Y = xs, ys
        rel_x = xs - face.mouth[0]
        rel_y = ys - face.mouth[1]
        down = -face.up
        self.a = rel_x * face.right[0] + rel_y * face.right[1]       # across the mouth
        self.b = rel_x * down[0] + rel_y * down[1]                    # below the seam
        hw, cd = face.half_w, face.chin_d
        lat = np.exp(-(self.a / (1.5 * hw)) ** 2)
        below = _smooth(self.b, -1.0, 1.5)
        past_chin = 1 - _smooth(self.b, cd, cd * 1.7)
        self.w_jaw = (lat * below * past_chin).astype(np.float32)
        self.w_lat = (np.exp(-(self.b / (hw * 0.9)) ** 2) * np.exp(-(self.a / (2.2 * hw)) ** 2)).astype(np.float32)
        corner = np.exp(-((np.abs(self.a) - hw) / (0.55 * hw)) ** 2) * np.exp(-(self.b / (0.7 * hw)) ** 2)
        self.w_corner = corner.astype(np.float32)
        self.side = np.sign(self.a).astype(np.float32)
        eye_w = float(np.linalg.norm(face.pts[EYE_L[0]] - face.pts[EYE_L[1]]))
        self.eye_w = eye_w
        self.w_brow = np.zeros_like(self.a)
        self.w_brow_r = np.zeros_like(self.a)
        for i, store in ((BROW_L, "w_brow"), (BROW_R, "w_brow_r")):
            c = face.pts[i]
            g = np.exp(-(((xs - c[0]) ** 2 + (ys - c[1]) ** 2) / (2 * (eye_w * 1.1) ** 2)))
            setattr(self, store, g.astype(np.float32))
        self.lid_colour = [self._skin_above(e) for e in (EYE_L, EYE_R)]

    def _skin_above(self, eye) -> np.ndarray:
        p = self.f.pts
        top = p[eye[2]]
        q = top + self.f.up * self.eye_w * 0.35
        x, y = int(q[0]), int(q[1])
        k = max(2, int(self.eye_w * 0.15))
        patch = self.img[max(0, y - k):y + k, max(0, x - k):x + k].reshape(-1, 3)
        return np.median(patch, axis=0)

    def frame(self, open_: float, widen: float, emotion: str, intensity: float,
              blink: float, mouth_bias: float = 0.0) -> np.ndarray:
        f = self.f
        smile, brow, one_side = SHAPE.get(emotion, SHAPE["neutral"])
        smile *= intensity
        brow *= intensity
        open_ = float(np.clip(open_ + mouth_bias, 0, 1))
        hw = f.half_w
        drop = open_ * hw * 0.48
        down = -f.up

        # displacement field (where content moves TO); remap samples from X - d
        dx = np.zeros_like(self.a)
        dy = np.zeros_like(self.a)
        dx += self.w_jaw * drop * down[0]
        dy += self.w_jaw * drop * down[1]
        lat = widen * hw * 0.10
        dx += self.w_lat * lat * (self.a / hw) * f.right[0]
        dy += self.w_lat * lat * (self.a / hw) * f.right[1]
        cw = self.w_corner
        if one_side:
            cw = cw * (self.side > 0)
        lift = smile * hw * 0.16
        dx += cw * lift * f.up[0]
        dy += cw * lift * f.up[1]
        bl = brow * self.eye_w * 0.22
        wb = self.w_brow + (self.w_brow_r if not one_side else self.w_brow_r * 1.8)
        if one_side:
            wb = self.w_brow * 0.3 + self.w_brow_r * 1.4
        dx += wb * bl * f.up[0]
        dy += wb * bl * f.up[1]

        roi = self.img[self.y0:self.y1, self.x0:self.x1]
        mx = (self.X - dx - self.x0).astype(np.float32)
        my = (self.Y - dy - self.y0).astype(np.float32)
        warped = cv2.remap(roi, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        if drop > 0.6:
            warped = self._mouth_interior(warped, drop, lat, lift, one_side)
        if blink > 0.05:
            warped = self._lids(warped, blink, (dx, dy))

        out = self.img.copy()
        out[self.y0:self.y1, self.x0:self.x1] = warped
        g = GRADE.get(emotion)
        if g and intensity > 0:
            k = np.array(g, dtype=np.float32) ** min(1.0, intensity)
            out = np.clip(out.astype(np.float32) * k, 0, 255).astype(np.uint8)
        return out

    def _moved(self, p: np.ndarray, drop: float, lat: float, lift: float, one_side: int, lower: bool) -> np.ndarray:
        f = self.f
        r = p - f.mouth
        a = float(np.dot(r, f.right))
        hw = f.half_w
        q = p.copy()
        if lower:
            w = math.exp(-(a / (1.5 * hw)) ** 2)
            q = q + (-f.up) * drop * w
        q = q + f.right * lat * (a / hw) * math.exp(-(a / (2.2 * hw)) ** 2)
        cw = math.exp(-((abs(a) - hw) / (0.55 * hw)) ** 2)
        if not one_side or a > 0:
            q = q + f.up * lift * cw
        return q

    def _mouth_interior(self, roi: np.ndarray, drop: float, lat: float, lift: float, one_side: int) -> np.ndarray:
        p = self.f.pts
        up_pts = [self._moved(p[i], drop, lat, lift, one_side, False) for i in UPPER_IN]
        lo_pts = [self._moved(p[i], drop, lat, lift, one_side, True) for i in LOWER_IN]
        poly = np.array(up_pts + lo_pts[::-1], dtype=np.float32) - (self.x0, self.y0)
        s = 4  # supersample the masks: the mouth is often only ~40 px wide
        h, w = roi.shape[:2]
        m = np.zeros((h * s, w * s), np.uint8)
        cv2.fillPoly(m, [np.round(poly * s).astype(np.int32)], 255, lineType=cv2.LINE_AA)
        teeth_h = min(drop * 0.2, self.f.half_w * 0.1)
        t_poly = np.array(up_pts + [q + (-self.f.up) * teeth_h for q in up_pts[::-1]], dtype=np.float32) - (self.x0, self.y0)
        t = np.zeros_like(m)
        cv2.fillPoly(t, [np.round(t_poly * s).astype(np.int32)], 255, lineType=cv2.LINE_AA)
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        t = cv2.resize(t, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        m = cv2.GaussianBlur(m, (0, 0), 0.8)
        t = cv2.GaussianBlur(t, (0, 0), 0.8) * m
        dark = np.array((28, 22, 52), np.float32)       # BGR: deep red-brown
        teeth = np.array((176, 184, 192), np.float32)
        out = roi.astype(np.float32)
        out = out * (1 - m[..., None]) + dark * m[..., None]
        out = out * (1 - t[..., None] * 0.6) + teeth * t[..., None] * 0.6
        return out.astype(np.uint8)

    def _lids(self, roi: np.ndarray, blink: float, disp) -> np.ndarray:
        p = self.f.pts
        out = roi.astype(np.float32)
        h, w = roi.shape[:2]
        for eye, col in zip((EYE_L, EYE_R), self.lid_colour):
            o, i, t, b = (p[k] - (self.x0, self.y0) for k in eye)
            c = (o + i) / 2
            ang = math.degrees(math.atan2(i[1] - o[1], i[0] - o[0]))
            ax = float(np.linalg.norm(i - o)) / 2 + 2
            ay = max(float(np.linalg.norm(t - b)) / 2 + 3, ax * 0.42)
            s = 4
            m = np.zeros((h * s, w * s), np.uint8)
            cv2.ellipse(m, (int(c[0] * s), int(c[1] * s)), (int(ax * s), int(ay * s)), ang, 0, 360, 255, -1, cv2.LINE_AA)
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            m = cv2.GaussianBlur(m, (0, 0), 1.2) * blink
            out = out * (1 - m[..., None]) + col.astype(np.float32) * m[..., None]
            # lash line along the lower half of the closed lid
            lash = np.zeros((h * s, w * s), np.uint8)
            cv2.ellipse(lash, (int(c[0] * s), int((c[1] + ay * 0.15) * s)), (int(ax * s), int(ay * 0.6 * s)),
                        ang, 20, 160, 255, max(1, int(s * 1.6)), cv2.LINE_AA)
            lash = cv2.resize(lash, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 * blink
            out = out * (1 - lash[..., None] * 0.9) + np.array((30, 28, 32), np.float32) * lash[..., None] * 0.9
        return out.astype(np.uint8)
