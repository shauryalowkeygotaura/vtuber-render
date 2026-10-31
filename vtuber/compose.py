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
from vtuber.mechhand import TOOLS
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
    b["hand"] = b.get("hand") if b.get("hand") in TOOLS else "none"
    if b["hand"] != "none" and b["spot"] in ("center_big", "half_left", "half_right"):
        b["spot"] = "center"   # a tool at the end of a raised arm is off-frame in close-ups
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


FONT = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "JetBrainsMono-ExtraBold.ttf"
_fonts: dict = {}


def _caption(frame: np.ndarray, text: str, pop: float) -> np.ndarray:
    """Word-chunk caption, cream on a heavy outline, popping in."""
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415
    size = int(84 * pop)
    if size not in _fonts:
        _fonts[size] = ImageFont.truetype(str(FONT), max(8, size))
    y0 = int(H * 0.66)
    band = Image.fromarray(cv2.cvtColor(frame[y0:y0 + 260], cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(band).text((W / 2, 130), text.upper(), font=_fonts[size], fill=(240, 228, 204),
                              stroke_width=10, stroke_fill=(16, 12, 14), anchor="mm")
    frame[y0:y0 + 260] = cv2.cvtColor(np.asarray(band), cv2.COLOR_RGB2BGR)
    return frame


RIM = (236, 246, 255)   # BGR, cool white


def _rim(canvas: np.ndarray, layer: np.ndarray, px: int = 7):
    """Comic sticker edge + soft glow behind the avatar so it reads on any b-roll.
    layer is premultiplied BGRA (puppet.render output). The outline is built at half
    resolution and the glow at 1/4 (full-res morphology cost ~700 ms a frame)."""
    a = layer[..., 3]
    ys, xs = np.nonzero(a[::4, ::4] > 8)
    if not len(ys):
        return
    y0, y1 = max(0, ys.min() * 4 - 44), min(H, ys.max() * 4 + 48)
    x0, x1 = max(0, xs.min() * 4 - 44), min(W, xs.max() * 4 + 48)
    sub = a[y0:y1, x0:x1]
    h, w = sub.shape
    half = cv2.resize(sub, (w // 2, h // 2), interpolation=cv2.INTER_AREA).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px + 1, px + 1))
    edge = cv2.dilate(half, k)
    quarter = cv2.resize(edge, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    glow = cv2.resize(cv2.GaussianBlur(quarter, (0, 0), 3.5), (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    rim = cv2.max(edge, (glow.astype(np.float32) * 0.55).astype(np.uint8))
    rim = cv2.resize(rim, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) * (1 / 255.0)
    roi = canvas[y0:y1, x0:x1]
    rgb = np.ascontiguousarray(roi[..., :3])
    r3 = cv2.merge([rim, rim, rim])
    rgb = cv2.add(cv2.multiply(rgb, 1 - r3), r3 * np.asarray(RIM, np.float32))
    la = layer[y0:y1, x0:x1, 3] * (1 / 255.0)
    rgb = cv2.add(cv2.multiply(rgb, cv2.merge([1 - la, 1 - la, 1 - la])),
                  np.ascontiguousarray(layer[y0:y1, x0:x1, :3]))
    roi[..., :3] = rgb


def _puff(canvas: np.ndarray, p: Puppet, pose: dict, root: np.ndarray, side: str, u: float):
    """Cartoon smoke puff at the wrist while a tool snaps on."""
    T = p.transforms(pose)[f"{side}_hand"]
    w = p.j[side]["wrist"]
    c = (root @ T) @ np.array([w[0], w[1], 1.0])
    k = u / 0.4
    s = math.sqrt(abs(np.linalg.det(root[:2, :2])))
    alpha = (1 - k) ** 1.5
    for j in range(7):
        ang = j * 2 * math.pi / 7 + 0.4
        r = (18 + 70 * k) * s * (0.8 + 0.4 * ((j * 37) % 5) / 5)
        cx, cy = int(c[0] + math.cos(ang) * r), int(c[1] + math.sin(ang) * r)
        rad = max(2, int((26 - 10 * k) * s))
        layer = canvas[..., :3].copy()
        cv2.circle(layer, (cx, cy), rad, (245, 245, 250), -1, cv2.LINE_AA)
        cv2.circle(layer, (cx, cy), rad, (40, 36, 44), max(1, int(2 * s)), cv2.LINE_AA)
        canvas[..., :3] = canvas[..., :3] * (1 - alpha) + layer * alpha


RIG = Path(__file__).resolve().parent.parent / "assets/avatar/rig-base.png"
_P: Puppet | None = None


def _frames(edit: dict, beats: list[dict], clips: list, i0: int, i1: int, out: Path) -> Path:
    """Render frames [i0, i1) to a silent segment. Runs in a worker process."""
    global _P
    if _P is None:
        _P = Puppet.build(RIG)
    p = _P
    seed = int(edit.get("seed", 7))
    caps = edit.get("captions") or []
    mtrack, wtrack = edit.get("mouth") or [], edit.get("widen") or []
    readers: dict[int, BrollReader] = {}
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-framerate", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                           "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    bi = 0
    for i in range(i0, i1):
        t = i / FPS
        while bi < len(beats) - 1 and t >= beats[bi]["end"]:
            bi += 1
        b = beats[bi]
        u = max(0.0, t - b["start"])
        if bi not in readers:  # open this beat's clip at the right frame for this segment
            readers[bi] = BrollReader(clips[bi])
            if readers[bi].cap is not None:
                readers[bi].cap.set(cv2.CAP_PROP_POS_FRAMES, int(u * FPS))
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
        side = None
        if b["hand"] != "none":   # tool swap: pops onto the free arm with a smoke puff
            side = "A" if b["arms"] == "point_left" else "B"
            if b["arms"] in ("idle", "shrug") and b["hand"] != "rocket":
                pose.update({f"{side}_shoulder": 70 * moves.back(u / 0.35), f"{side}_elbow": 100 * moves.back(u / 0.35)})
            if b["hand"] == "rocket":   # fist fires off along the arm and flies back
                pose.update({f"{side}_shoulder": 90 * moves.back(u / 0.3), f"{side}_elbow": 0})
                out_k = moves.smooth((u - 0.45) / 0.35) - moves.smooth((u - 1.3) / 0.5)
                pose[f"{side}_hand_launch"] = 950 * out_k
            pose[f"{side}_hand_kind"] = b["hand"]
            pose[f"{side}_hand_scale"] = moves.back(u / 0.28, 2.2)
            pose[f"{side}_hand_phase"] = int(t * 15)
        mouth = float(mtrack[i]) * 1.1 if i < len(mtrack) else 0.0
        widen = float(wtrack[i]) if i < len(wtrack) else 0.0
        face = {"emotion": b["emotion"], "t": t, "open_": min(1.0, mouth), "widen": widen,
                "blink": 1.0 if (t % 3.1) < 0.09 else 0.0}
        layer = p.render(pose, root, (W, H), face)
        _rim(canvas, layer)
        if side and u < 0.4:
            _puff(canvas, p, pose, root, side, u)
        cdx, cdy, z, crot = moves.camera(b["camera"], u, seed * 7919 + bi)
        if (cdx, cdy, z, crot) != (0, 0, 1, 0):
            M = cv2.getRotationMatrix2D((W / 2, H / 2), crot, z)
            M[0, 2] += cdx
            M[1, 2] += cdy
            canvas = cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        frame = canvas[..., :3].clip(0, 255).astype(np.uint8)
        cap = next((c for c in caps if c["start"] - 0.02 <= t <= c["end"] + 0.12), None)
        if cap:
            frame = _caption(frame, cap["text"], min(1.0, (t - cap["start"]) * 8 + 0.6))
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on segment {out.name}")
    return out


def render(edit: dict, out: Path, puppet: Puppet | None = None, use_broll: bool = True,
           workers: int | None = None) -> Path:
    """Fetch b-roll once, render the frames in parallel segments, then join + add audio."""
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor

    t0 = time.time()
    beats = [clean(b) for b in edit["beats"]]
    used: set[str] = set()
    clips: list = []
    for b in beats:
        clip = None
        if use_broll and b.get("broll"):
            try:
                clip = broll.fetch(b["broll"], b["end"] - b["start"] + 0.2, used, line=b.get("text", ""))
            except RuntimeError as e:
                print(f"  broll off: {e}", file=sys.stderr)
                use_broll = False
        clips.append(clip)
    print(f"  b-roll ready in {time.time() - t0:.0f}s")
    n = int(round(beats[-1]["end"] * FPS))
    # each worker holds a puppet (~0.6 GB peak); the laptop has run out of memory before, so cap it
    workers = workers or max(1, min(6, (os.cpu_count() or 2) // 3))
    seg_dir = out.parent / f".{out.stem}_segs"
    seg_dir.mkdir(exist_ok=True)
    bounds = [round(n * k / workers) for k in range(workers + 1)]
    segs = [seg_dir / f"seg{k:02d}.mp4" for k in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_frames, edit, beats, clips, bounds[k], bounds[k + 1], segs[k]) for k in range(workers)]
        for f in futs:
            f.result()
    print(f"  frames done in {time.time() - t0:.0f}s ({workers} workers)")
    lst = seg_dir / "list.txt"
    lst.write_text("".join(f"file '{s_.name}'\n" for s_ in segs), encoding="utf-8")
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst)]
    audio = edit.get("audio")
    if audio:
        cmd += ["-i", str(audio), "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "copy", "-movflags", "+faststart", "-t", f"{n / FPS:.3f}", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg join failed: {r.stderr[-300:]}")
    for s_ in segs + [lst]:
        s_.unlink(missing_ok=True)
    seg_dir.rmdir()
    print(f"  total {time.time() - t0:.0f}s")
    return out


if __name__ == "__main__":
    e = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(render(e, Path(sys.argv[2])))
