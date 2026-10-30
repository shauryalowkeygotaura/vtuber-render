"""talk.py -- a photo of you + a script (or a voice take) in, a talking reel out.

    doppler run -p philosopher-pipeline -c dev -- python talk.py --photo me.jpg --script script.txt
    doppler run -p philosopher-pipeline -c dev -- python talk.py --photo me.jpg --audio take.m4a
    python talk.py --photo me.jpg --audio take.m4a --no-llm      # offline: lip sync + idle only

--script  synthesises the voice: --tts edge (free, default en-IN-PrabhatNeural)
          or --tts orpheus (Groq, --voice troy). Your cloned voice slots in here later.
--audio   uses a real recording instead (your own voice take).

Pipeline: voice -> lipsync.py (mouth per frame) -> performance.py (emotion +
gesture per sentence, 10 emotions) -> motion.py (head track, blinks) ->
photo2d.py (warps the photo) -> camera + emotion marks + word captions -> mp4.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vtuber import lipsync, motion, performance, photo2d

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
FONT = ROOT / "assets" / "fonts" / "JetBrainsMono-ExtraBold.ttf"
CREAM = (240, 228, 204)
BURGUNDY = (114, 47, 55)
PUNCH = {"excited", "surprised", "angry"}


# ── voice ────────────────────────────────────────────────────────────────────
def _orpheus():
    sys.path.insert(0, str(ROOT.parent / "paralang"))
    from paralang import orpheus  # noqa: PLC0415 - stdlib-only TTS, vendored by path
    return orpheus


def synth_script(text: str, voice: str, wav_path: Path) -> list[tuple[str, float, float]]:
    """Sentence by sentence (short requests are what Orpheus does best), joined
    with a breath of silence. Returns word timings for captions."""
    orp = _orpheus()
    sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s.strip()]
    gap = b"\x00\x00" * int(orp.SR * 0.16)
    pcm = b""
    for i, s in enumerate(sentences, 1):
        if not orp.speakable(s):
            raise ValueError(f"Orpheus is English-only; rewrite: {s[:60]}")
        pcm += orp.synth(s, voice=voice) + gap
        print(f"  voice {i}/{len(sentences)}", flush=True)
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(orp.SR); w.writeframes(pcm)
    try:
        return orp.align(pcm, text)
    except Exception as e:  # noqa: BLE001 - captions fall back to sentence timing
        print(f"  alignment failed ({str(e)[:100]}); captions from sentences", file=sys.stderr)
        return []


def synth_edge(text: str, voice: str, audio_path: Path) -> list[tuple[str, float, float]]:
    """Free Microsoft Edge neural voice, no key. Word timings come with it."""
    import asyncio

    import edge_tts

    async def go():
        c = edge_tts.Communicate(text.strip(), voice, boundary="WordBoundary")
        audio, words = bytearray(), []
        async for ch in c.stream():
            if ch["type"] == "audio":
                audio.extend(ch["data"])
            elif ch["type"] == "WordBoundary":
                s = ch["offset"] / 1e7
                words.append((ch["text"], s, s + ch["duration"] / 1e7))
        return bytes(audio), words

    audio, words = asyncio.run(go())
    if not audio:
        raise RuntimeError("edge-tts returned no audio")
    audio_path.write_bytes(audio)
    print(f"  voice: edge {voice}, {len(words)} words")
    script = text.split()
    if len(script) == len(words):   # keep the script's punctuation: captions break on it
        words = [(s, a, e) for s, (_, a, e) in zip(script, words)]
    return words


def words_from_beats(beats: list[dict]) -> list[tuple[str, float, float]]:
    out = []
    for b in beats:
        ws = b["text"].split()
        if not ws:
            continue
        total = sum(len(w) + 1 for w in ws)
        t = b["start"]
        for w in ws:
            dt = (b["end"] - b["start"]) * (len(w) + 1) / total
            out.append((w, t, t + dt))
            t += dt
    return out


def caption_chunks(words, max_words: int = 3, max_chars: int = 16):
    chunks, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or len(" ".join(x[0] for x in cur + [w])) > max_chars
                    or re.search(r"[.?!,]$", cur[-1][0]) or w[1] - cur[-1][2] > 0.3):
            chunks.append(cur); cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return [(" ".join(x[0] for x in c), c[0][1], c[-1][2]) for c in chunks]


# ── per-frame tracks ─────────────────────────────────────────────────────────
def emotion_track(beats: list[dict], n: int, fps: int):
    emo = ["neutral"] * n
    k = np.zeros(n)
    for b in beats:
        s, e = int(b["start"] * fps), int(b["end"] * fps)
        for i in range(max(0, s), min(n, e)):
            emo[i] = b["expression"]
            k[i] = b["intensity"]
    # ease intensity in and out so expressions don't snap
    out = np.zeros(n)
    prev = 0.0
    for i in range(n):
        tgt = k[i] if emo[i] != "neutral" else 0.0
        if i and emo[i] != emo[i - 1]:
            prev = 0.0
        prev += (tgt - prev) * 0.18
        out[i] = prev
    return emo, out


def punch_track(beats: list[dict], n: int, fps: int) -> np.ndarray:
    z = np.zeros(n)
    for b in beats:
        strong = b["expression"] in PUNCH or b["gesture"] == "lean_in" or b["intensity"] >= 0.8
        if not strong:
            continue
        s, e = int(b["start"] * fps), int(b["end"] * fps)
        z[max(0, s):min(n, e)] = 1.0
    out, prev = np.zeros(n), 0.0
    for i in range(n):
        prev += (z[i] - prev) * (0.35 if z[i] > prev else 0.08)   # snap in, ease out
        out[i] = prev
    return out


# ── drawing ──────────────────────────────────────────────────────────────────
def _star(img, c, r, col):
    pts = []
    for j in range(8):
        a = j * math.pi / 4
        rr = r if j % 2 == 0 else r * 0.3
        pts.append((c[0] + rr * math.cos(a), c[1] + rr * math.sin(a)))
    cv2.fillPoly(img, [np.round(pts).astype(np.int32)], col, cv2.LINE_AA)
    cv2.polylines(img, [np.round(pts).astype(np.int32)], True, (20, 20, 24), 3, cv2.LINE_AA)


def emotion_mark(frame: np.ndarray, emotion: str, amt: float, anchor, t: float, draw: ImageDraw.ImageDraw | None = None):
    """Small comic marks beside the head. Drawn in BGR with cv2."""
    if amt < 0.15:
        return
    x, y = int(anchor[0]), int(anchor[1])
    s = 0.6 + 0.4 * amt
    bob = int(6 * math.sin(t * 6))
    cream = CREAM[::-1]
    if emotion == "excited":
        for dx, dy, r in ((0, 0, 34), (70, 60, 22), (-40, 90, 16)):
            _star(frame, (x + dx, y + dy + bob), r * s * (0.9 + 0.1 * math.sin(t * 9 + dx)), cream)
    elif emotion == "happy":
        _star(frame, (x, y + bob), 24 * s, cream)
    elif emotion == "nervous":
        pts = np.array([(x, y - 40), (x + 22, y), (x, y + 18), (x - 22, y)], np.int32) + (0, bob)
        cv2.fillPoly(frame, [pts], (245, 220, 170), cv2.LINE_AA)
        cv2.polylines(frame, [pts], True, (20, 20, 24), 3, cv2.LINE_AA)
    elif emotion == "angry":
        red = (63, 52, 209)
        for ang in (45, 135, 225, 315):
            a = math.radians(ang)
            c = (int(x + 22 * math.cos(a)), int(y + bob + 22 * math.sin(a)))
            cv2.ellipse(frame, c, (16, 16), ang + 180, -60, 60, red, 8, cv2.LINE_AA)
    elif emotion in ("surprised", "thinking") and draw is not None:
        pass  # text marks are drawn in the PIL pass


def text_mark(draw: ImageDraw.ImageDraw, emotion: str, amt: float, anchor, font, t: float):
    if amt < 0.15 or emotion not in ("surprised", "thinking"):
        return
    ch = "!" if emotion == "surprised" else "?"
    x, y = anchor[0], anchor[1] + 6 * math.sin(t * 6)
    draw.text((x, y), ch, font=font, fill=CREAM, stroke_width=6, stroke_fill=(20, 18, 20), anchor="mm")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photo", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--script", help="text, or a path to a .txt")
    src.add_argument("--audio")
    ap.add_argument("--tts", choices=("edge", "orpheus"), default="edge",
                    help="edge = free, no key; orpheus = Groq (needs its terms accepted on the key's org)")
    ap.add_argument("--voice", help="edge: en-IN-PrabhatNeural (default), en-US-AndrewNeural...; orpheus: troy, daniel...")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--out")
    a = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("ffmpeg not on PATH", file=sys.stderr)
        return 2
    OUT.mkdir(exist_ok=True)
    img = cv2.imread(a.photo)
    if img is None:
        print(f"cannot read photo {a.photo}", file=sys.stderr)
        return 2
    stem = Path(a.out).stem if a.out else Path(a.photo).stem + "-talk"

    words: list = []
    if a.script:
        text = Path(a.script).read_text(encoding="utf-8") if Path(a.script).is_file() else a.script
        if a.tts == "edge":
            audio = OUT / f"{stem}.voice.mp3"
            words = synth_edge(text, a.voice or "en-IN-PrabhatNeural", audio)
        else:
            audio = OUT / f"{stem}.voice.wav"
            words = synth_script(text, a.voice or "troy", audio)
    else:
        audio = Path(a.audio)

    fps = a.fps
    lip = lipsync.visemes(lipsync.load_audio(str(audio)), fps)
    beats = performance.direct(str(audio), use_llm=not a.no_llm, expressions=photo2d.EMOTIONS)
    tl = motion.build(lip, [{**b, "expression": b["expression"] if b["expression"] in performance.EXPRESSIONS
                             else "happy" if b["expression"] in ("excited", "smug") else "neutral"} for b in beats], fps)
    (OUT / f"{stem}.beats.json").write_text(json.dumps(beats, indent=1, ensure_ascii=False), encoding="utf-8")
    if not words:
        words = words_from_beats(beats)
    caps = [] if a.no_captions else caption_chunks(words)
    n = tl["frames"] if not a.seconds else min(tl["frames"], int(a.seconds * fps))
    print(f"  {n} frames, {len(beats)} beats, {len(caps)} captions")

    face = photo2d.landmarks(img)
    rig = photo2d.Rig(img, face)
    emo, emo_k = emotion_track(beats, n, fps)
    punch = punch_track(beats, n, fps)
    vis = {v: np.asarray(tl["visemes"][v]) for v in ("aa", "ih", "ou", "ee", "oh")}
    head = {k: np.asarray(v) for k, v in tl["head"].items()}
    blink = np.asarray(tl["blink"])

    W, H = 1080, 1920
    h, w = img.shape[:2]
    base = max(W / w, H / h)
    img_c = np.array((w / 2, h / 2))
    font = ImageFont.truetype(str(FONT), 78)
    mark_font = ImageFont.truetype(str(FONT), 120)

    out = Path(a.out or OUT / f"{stem}.mp4")
    ff = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-framerate", str(fps), "-i", "-", "-i", str(audio), "-map", "0:v", "-map", "1:a",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", f"{n / fps:.3f}", str(out)],
        stdin=subprocess.PIPE)
    ci = 0
    for i in range(n):
        t = i / fps
        open_ = min(1.0, vis["aa"][i] + 0.8 * vis["oh"][i] + 0.55 * vis["ou"][i] + 0.45 * vis["ih"][i] + 0.45 * vis["ee"][i])
        widen = float(np.clip(vis["ee"][i] + 0.6 * vis["ih"][i] - vis["ou"][i] - 0.6 * vis["oh"][i], -1, 1))
        e = emo[i]
        bias = 0.15 * emo_k[i] if e in ("surprised", "excited") else 0.0
        fr = rig.frame(open_ * 1.15, widen, e, float(emo_k[i]), float(blink[i]), bias)

        # camera: mid shot on the face, punch in on big lines, handheld drift from the head track
        z = 1.35 + 0.45 * punch[i]
        focus = img_c + (face.centre - img_c) * (0.55 + 0.45 * punch[i])
        ang = math.degrees(head["roll"][i]) * 0.6
        S = base * z
        M = cv2.getRotationMatrix2D((float(focus[0]), float(focus[1])), ang, S)
        M[0, 2] += W / 2 - focus[0] + head["yaw"][i] * 260
        M[1, 2] += H * 0.42 - focus[1] + head["pitch"][i] * 260
        frame = cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

        anchor_src = face.centre + face.up * face.size * 0.75 + face.right * face.size * 0.7
        anchor = M @ np.array([anchor_src[0], anchor_src[1], 1.0])
        anchor = (min(W - 90, max(90, anchor[0])), min(H - 300, max(150, anchor[1])))
        emotion_mark(frame, e, float(emo_k[i]), anchor, t)

        while ci < len(caps) and caps[ci][2] < t:
            ci += 1
        cap = caps[ci] if ci < len(caps) and caps[ci][1] <= t + 0.05 else None
        if cap or (e in ("surprised", "thinking") and emo_k[i] > 0.15):
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            d = ImageDraw.Draw(pil)
            text_mark(d, e, float(emo_k[i]), anchor, mark_font, t)
            if cap:
                pop = min(1.0, (t - cap[1]) * 8 + 0.6)
                f = font if pop >= 1 else ImageFont.truetype(str(FONT), int(78 * pop))
                d.text((W / 2, H * 0.74), cap[0].upper(), font=f, fill=CREAM, stroke_width=9,
                       stroke_fill=(16, 12, 14), anchor="mm")
            frame = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        ff.stdin.write(frame.tobytes())
        if i and i % (fps * 5) == 0:
            print(f"  {i}/{n}", flush=True)
    ff.stdin.close()
    if ff.wait() != 0:
        print("ffmpeg failed", file=sys.stderr)
        return 1
    print(f"  video -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
