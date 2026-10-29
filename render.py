"""render.py -- voice recording in, talking avatar video out.

    doppler run -p philosopher-pipeline -c dev -- python render.py samples/take.m4a
    python render.py take.m4a --no-llm            # skip Groq: lip sync + idle only
    python render.py take.m4a --alpha             # transparent .webm to lay over b-roll
    python render.py take.m4a --avatar me.vrm     # your own VRoid export
    python render.py take.m4a --still 2.5         # one PNG at 2.5 s, to eyeball framing

Shaurya only records his voice. Everything the face and head do is decided
here: lipsync.py (mouth from the audio), performance.py (expression + gesture
from the words), motion.py (all of it as one pose per frame), web/index.html
(three-vrm puppet that renders a frame on request).
"""
from __future__ import annotations

import argparse
import base64
import functools
import glob
import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from vtuber import lipsync, motion, performance

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
DEFAULT_AVATAR = ROOT / "assets" / "sample.vrm"


def _serve() -> tuple[http.server.ThreadingHTTPServer, int]:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):  # noqa: D401 - silence per-request logs
            pass

    handler = functools.partial(Quiet, directory=str(ROOT))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _chromium_path() -> str | None:
    """Python playwright pins a revision that may not be downloaded here."""
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright"
    hits = sorted(glob.glob(str(base / "chromium-*" / "chrome-win64" / "chrome.exe")))
    return hits[-1] if hits else None


def _launch(pw):
    args = ["--ignore-gpu-blocklist", "--enable-gpu", "--use-angle=d3d11", "--enable-unsafe-swiftshader"]
    try:
        return pw.chromium.launch(args=args)
    except Exception:  # noqa: BLE001 - pinned revision missing; use what's installed
        exe = _chromium_path()
        if not exe:
            raise
        return pw.chromium.launch(executable_path=exe, args=args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--avatar", default=str(DEFAULT_AVATAR))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1080x1920")
    ap.add_argument("--alpha", action="store_true", help="transparent VP9 .webm")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--seconds", type=float, help="render only the first N seconds")
    ap.add_argument("--still", type=float, help="write one PNG at this time and stop")
    ap.add_argument("--out")
    a = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("ffmpeg not on PATH", file=sys.stderr)
        return 2
    w, h = (int(x) for x in a.size.lower().split("x"))
    OUT.mkdir(exist_ok=True)
    stem = Path(a.audio).stem

    t0 = time.time()
    lip = lipsync.visemes(lipsync.load_audio(a.audio), a.fps)
    beats = performance.direct(a.audio, use_llm=not a.no_llm)
    tl = motion.build(lip, beats, a.fps)
    (OUT / f"{stem}.beats.json").write_text(json.dumps(beats, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  analysed {tl['frames']} frames, {len(beats)} directed beats in {time.time() - t0:.1f}s")

    avatar = Path(a.avatar).resolve()
    if not avatar.is_relative_to(ROOT):
        shutil.copy(avatar, OUT / "_avatar.vrm")
        avatar = OUT / "_avatar.vrm"
    avatar_url = "/" + avatar.relative_to(ROOT).as_posix()

    n = tl["frames"]
    if a.seconds:
        n = min(n, int(a.seconds * a.fps))
    first = 0
    if a.still is not None:
        first = min(int(a.still * a.fps), tl["frames"] - 1)

    from playwright.sync_api import sync_playwright

    srv, port = _serve()
    try:
        with sync_playwright() as pw:
            browser = _launch(pw)
            page = browser.new_page(viewport={"width": w, "height": h})
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{port}/web/index.html")
            page.wait_for_function("window.stageReady === true", timeout=60_000)
            info = page.evaluate("([u,w,h,t]) => init(u,w,h,t)", [avatar_url, w, h, a.alpha])
            page.evaluate("t => loadTimeline(t)", tl)
            if errors:
                raise RuntimeError("stage error: " + errors[0])
            print(f"  stage ready (head at {info['headY']:.2f} m)")

            if a.still is not None:
                # physics needs the frames before it, so step up to the target
                for i in range(first):
                    page.evaluate("i => renderAt(i, 'image/jpeg')", i)
                data = page.evaluate("i => renderAt(i, 'image/png')", first)
                path = Path(a.out or OUT / f"{stem}.still.png")
                path.write_bytes(base64.b64decode(data.split(",", 1)[1]))
                print(f"  still -> {path}")
                browser.close()
                return 0

            out = Path(a.out or OUT / f"{stem}.{'webm' if a.alpha else 'mp4'}")
            vcodec = (["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "4M", "-c:a", "libopus"]
                      if a.alpha else
                      ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"])
            ff = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-y", "-f", "image2pipe", "-framerate", str(a.fps), "-i", "-",
                 "-i", a.audio, "-map", "0:v", "-map", "1:a", *vcodec, "-t", f"{n / a.fps:.3f}", str(out)],
                stdin=subprocess.PIPE,
            )
            t1 = time.time()
            for i in range(n):
                data = page.evaluate("i => renderAt(i)", i)
                ff.stdin.write(base64.b64decode(data.split(",", 1)[1]))
                if i and i % (a.fps * 5) == 0:
                    rate = i / (time.time() - t1)
                    print(f"  {i}/{n} frames  {rate:.1f} fps  eta {(n - i) / rate:.0f}s", flush=True)
            ff.stdin.close()
            code = ff.wait()
            browser.close()
            if code != 0:
                print(f"ffmpeg failed ({code})", file=sys.stderr)
                return 1
            print(f"  video -> {out}  ({time.time() - t0:.0f}s total)")
    finally:
        srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
