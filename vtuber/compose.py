"""compose.py -- render an edit.json: b-roll + puppet + moves (+ optional audio).

edit.json (written by the director, editable by hand):
{
  "seed": 7,
  "audio": "output/x.voice.mp3" | null,
  "beats": [
    {"start": 0.0, "end": 2.2, "text": "...", "emotion": "happy", "talk": [0..1 per frame] (optional),
     "spot": "center_big", "enter": "drop_from_sky", "camera": "zoom_punch",
     "arms": "wave", "broll": "neon city night"}
  ]
}
Unknown names fall back to safe defaults, so a hand edit can never crash a render.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from vtuber import broll, moves
from vtuber.avatar import Rig
from vtuber.puppet import Puppet

W, H, FPS = moves.W, moves.H, 30
FEET = (390.0, 1335.0)     # body centre x and feet y in rig-base.png
EMOTIONS = tuple(Rig.LOOKS)


def _m(tx=0.0, ty=0.0, rot=0.0, sx=1.0, sy=1.0):
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
    S = np.diag([sx, sy, 1]).astype(np.float32)
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], np.float32)
    return T @ R @ S


def clean(beat: dict) -> dict:
    """Whitelist every field: anything unknown becomes the safe default."""
    b = dict(beat)
    b["spot"] = b.get("spot") if b.get("spot") in moves.SPOTS else "center"
    b["enter"] = b.get("enter") if b.get("enter") in moves.ENTRANCES else "none"
    b["camera"] = b.get("camera") if b.get("camera") in moves.CAMERAS else "none"
    b["arms"] = b.get("arms") if b.get("arms") in moves.ARMS else "idle"
    b["emotion"] = b.get("emotion") if b.get("emotion") in EMOTIONS else "neutral"
    return b


class BrollReader:
    def __init__(self, path: Path | None):
        self.cap = cv2.VideoCapture(str(path)) if path else None
        self.last = None

    def frame(self) -> np.ndarray:
        if self.cap is not None:
            ok, fr = self.cap.read()
            if not ok:  # loop the clip if the beat outlasts it
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, fr = self.cap.read()
            if ok:
                self.last = cv2.resize(fr, (W, H)) if fr.shape[:2] != (H, W) else fr
        if self.last is None:
            self.last = np.full((H, W, 3), (55, 47, 114), np.uint8)
        return self.last


RIM = (236, 246, 255)   # BGR, cool white


def _rim(canvas: np.ndarray, layer: np.ndarray, px: int = 7):
    """Comic sticker edge + soft glow behind the avatar so it reads on any b-roll."""
    a = layer[..., 3]
    ys, xs = np.nonzero(a > 8)
    if not len(ys):
        return
    y0, y1 = max(0, ys.min() - 40), min(H, ys.max() + 41)
    x0, x1 = max(0, xs.min() - 40), min(W, xs.max() + 41)
    sub = a[y0:y1, x0:x1]
    edge = cv2.dilate(sub, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1)))
    glow = cv2.GaussianBlur(edge, (0, 0), 14) * 0.55
    rim = np.maximum(edge, glow)[..., None] / 255.0
    roi = canvas[y0:y1, x0:x1]
    roi[..., :3] = np.asarray(RIM, np.float32) * rim + roi[..., :3] * (1 - rim)
    la = layer[y0:y1, x0:x1, 3:4] / 255.0
    roi[..., :3] = layer[y0:y1, x0:x1, :3] * la + roi[..., :3] * (1 - la)


def render(edit: dict, out: Path, puppet: Puppet | None = None, use_broll: bool = True) -> Path:
    p = puppet or Puppet.build(Path(__file__).resolve().parent.parent / "assets/avatar/rig-base.png")
    beats = [clean(b) for b in edit["beats"]]
    seed = int(edit.get("seed", 7))
    used: set[str] = set()
    readers = []
    for b in beats:
        clip = None
        if use_broll and b.get("broll"):
            try:
                clip = broll.fetch(b["broll"], b["end"] - b["start"] + 0.2, used)
            except RuntimeError as e:
                print(f"  broll off: {e}", file=sys.stderr)
                use_broll = False
        readers.append(BrollReader(clip))
    n = int(round(beats[-1]["end"] * FPS))
    audio = edit.get("audio")
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-framerate", str(FPS), "-i", "-"]
    if audio:
        cmd += ["-i", str(audio), "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-t", f"{n / FPS:.3f}", str(out)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    bi = 0
    for i in range(n):
        t = i / FPS
        while bi < len(beats) - 1 and t >= beats[bi]["end"]:
            bi += 1
        b = beats[bi]
        u = t - b["start"]
        rng = np.random.default_rng(seed * 1000 + bi)

        canvas = np.zeros((H, W, 4), np.float32)
        canvas[..., :3] = readers[bi].frame()
        canvas[..., 3] = 255

        scale, sx_, fy = moves.SPOTS[b["spot"]]
        dx, dy, rot, qx, qy = moves.enter(b["enter"], u, rng)
        breathe = 1 + 0.012 * math.sin(t * 3.1)
        bob = 10 * math.sin(t * 4.4)
        root = _m(sx_ + dx, fy + dy + bob, rot + 1.5 * math.sin(t * 1.7), scale * qx, scale * qy * breathe) @ \
            _m(-FEET[0], -FEET[1])
        pose = moves.arms(b["arms"], u)
        pose["head"] = pose.get("head", 0) + 4 * math.sin(t * 2.3)
        talk = b.get("talk")
        mouth = float(talk[min(len(talk) - 1, int(u * FPS))]) if talk else 0.0
        face = {"emotion": b["emotion"], "t": t, "open_": mouth,
                "blink": 1.0 if (t % 3.1) < 0.09 else 0.0}
        layer = p.render(pose, root, (W, H), face)
        _rim(canvas, layer)

        cdx, cdy, z, crot = moves.camera(b["camera"], u, seed * 7919 + bi)
        if (cdx, cdy, z, crot) != (0, 0, 1, 0):
            M = cv2.getRotationMatrix2D((W / 2, H / 2), crot, z)
            M[0, 2] += cdx
            M[1, 2] += cdy
            canvas = cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        ff.stdin.write(canvas[..., :3].clip(0, 255).astype(np.uint8).tobytes())
        if i and i % (FPS * 2) == 0:
            print(f"  {i}/{n}", flush=True)
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    return out


if __name__ == "__main__":
    e = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(render(e, Path(sys.argv[2])))
