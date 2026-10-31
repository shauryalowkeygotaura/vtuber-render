"""broll.py -- high-quality portrait b-roll from Pexels, for the avatar reels.

    doppler run -p creative-studio -c dev -- python -m vtuber.broll "city night" 4

Quality rules (the old b-roll was rejected as low quality):
  * portrait search, and the file must be at least 1080x1920 (a true full-HD
    vertical); landscape is accepted only if its height is >= 2160, so the
    centre crop is still full HD
  * the smallest file that clears that bar is downloaded (a 4K 60fps original is
    120 MB of nothing extra once scaled to 1080x1920)
  * clips shorter than the beat are skipped, clip ids never repeat in a reel
Downloads are cached in assets/broll/ (gitignored) by Pexels id.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "assets" / "broll"
W, H = 1080, 1920
API = "https://api.pexels.com/videos/search"
STOP = {"the", "and", "with", "for", "screen", "video", "background", "on", "of"}


def _key() -> str:
    k = (os.environ.get("PEXELS_API_KEY") or "").split(",")[0].strip()
    if not k:
        raise RuntimeError("PEXELS_API_KEY missing: run through `doppler run -p creative-studio -c dev --`")
    return k


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": _key(), "User-Agent": "vtuber-render"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            if r.status != 200:
                raise RuntimeError(f"Pexels HTTP {r.status}")
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Pexels HTTP {e.code}: {e.read()[:200]!r}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Pexels returned non-JSON: {e}") from e


def _best_file(video: dict) -> dict | None:
    """Smallest file that is full-HD vertical after a centre crop."""
    ok = []
    for f in video.get("video_files", []):
        w, h = int(f.get("width") or 0), int(f.get("height") or 0)
        if not f.get("link") or f.get("file_type") != "video/mp4":
            continue
        portrait_ok = h >= w and w >= W and h >= H
        landscape_ok = w > h and h >= 2160
        if portrait_ok or landscape_ok:
            ok.append(f)
    return min(ok, key=lambda f: int(f["width"]) * int(f["height"])) if ok else None


def search(term: str, min_seconds: float, used: set[str] | None = None, pages: int = 2) -> list[dict]:
    """Candidates [{id, url, w, h, duration, page}] that pass the quality bar, best first."""
    used = used or set()
    out = []
    for page in range(1, pages + 1):
        q = urllib.parse.urlencode({"query": term, "orientation": "portrait", "size": "large",
                                    "per_page": 40, "page": page})
        data = _get(f"{API}?{q}")
        for v in data.get("videos", []):
            cid = f"pexels:{v.get('id')}"
            if cid in used or float(v.get("duration") or 0) < min_seconds:
                continue
            f = _best_file(v)
            if f:
                out.append({"id": cid, "url": f["link"], "w": int(f["width"]), "h": int(f["height"]),
                            "duration": float(v["duration"]), "page": v.get("url", "")})
        if len(out) >= 6 or not data.get("next_page"):
            break
    # Pexels search is loose ("github pages homepage" returned a girl on a phone): rank by how many
    # query words appear in the clip's title slug, and drop clips that match none when others do
    words = {w for w in term.lower().split() if len(w) > 2} - STOP
    for c in out:
        slug = c["page"].lower()
        c["score"] = sum(w.rstrip("s") in slug for w in words)
    out.sort(key=lambda c: -c["score"])
    if out and out[0]["score"] > 0:
        out = [c for c in out if c["score"] > 0]
    return out


def _title(c: dict) -> str:
    """Pexels page slug -> readable title ('a-person-using-a-laptop-7126537' -> 'a person using a laptop')."""
    slug = c["page"].rstrip("/").rsplit("/", 1)[-1]
    words = slug.split("-")
    return " ".join(words[:-1] if words and words[-1].isdigit() else words)[:120]


SAFE = ("dark abstract technology background", "neon light particles", "code on screen dark",
        "blurred city lights night", "abstract gradient motion")


def rng_safe(term: str) -> str:
    """A neutral backdrop query, stable per term so re-renders pick the same one."""
    return SAFE[sum(map(ord, term)) % len(SAFE)]


def judge(line: str, term: str, cands: list[dict]) -> tuple[int | None, str]:
    """Groq reads the spoken line and the candidate clip titles: returns the best index
    (None = nothing fits) and a better search query to try if nothing fits."""
    if not line or not cands:
        return 0 if cands else None, ""
    try:
        from vtuber import groq_pool  # noqa: PLC0415
        menu = "\n".join(f"{i}: {_title(c)}" for i, c in enumerate(cands[:10]))
        resp = groq_pool.chat(groq_pool.FAST, messages=[
            {"role": "system", "content":
                "You pick background stock footage for one line of a vertical reel. The clip must show the "
                "line's idea literally or as an obvious visual metaphor; a clip that only shares a word "
                "(a jungle pipeline for a 'deployment pipeline') is WRONG. Reply JSON "
                '{"pick": <index or -1>, "alt": "<2-4 word concrete search query to try if -1>"}'},
            {"role": "user", "content": f"Line: {line}\nSearch: {term}\nClips:\n{menu}"}],
            max_tokens=120, response_format={"type": "json_object"}, require_content=True)
        data = json.loads(resp.choices[0].message.content)
        pick = int(data.get("pick", -1))
        alt = data.get("alt") if isinstance(data.get("alt"), str) else ""
        alt = " ".join("".join(ch for ch in w if ch.isalnum()) for w in alt.lower().split()[:5]).strip()
        return (pick if 0 <= pick < min(10, len(cands)) else None), alt
    except Exception as e:  # noqa: BLE001 - judging is a bonus; fall back to title ranking
        print(f"  broll: judge unavailable ({str(e)[:80]})", file=sys.stderr)
        return 0, ""


def fetch(term: str, seconds: float, used: set[str], line: str = "") -> Path | None:
    """Download (cached) and conform one clip to 1080x1920/30fps/no audio, `seconds` long.
    With `line`, Groq judges the candidates; if none fit it tries its better query once."""
    cands = search(term, seconds, used)
    pick, alt = judge(line, term, cands)
    if pick is None and alt and alt != term:
        alt_cands = search(alt, seconds, used)
        apick, _ = judge(line, alt, alt_cands)
        if apick is not None:
            print(f"  broll: '{term}' didn't fit, using '{alt}'")
            term, cands, pick = alt, alt_cands, apick
    if pick is None and line:
        # nothing on-topic exists: a neutral moving background beats a wrong one (CN Tower for "no internet")
        print(f"  broll: nothing fits '{term}', using a neutral background", file=sys.stderr)
        term = rng_safe(term)
        cands = search(term, seconds, used)
        pick = 0
    if pick:
        cands = [cands[pick]] + cands[:pick] + cands[pick + 1:]
    for c in cands:
        CACHE.mkdir(parents=True, exist_ok=True)
        raw = CACHE / f"{c['id'].replace(':', '_')}.mp4"
        if not raw.exists() or raw.stat().st_size < 100_000:
            tmp = raw.with_suffix(".part")
            try:
                req = urllib.request.Request(c["url"], headers={"User-Agent": "vtuber-render"})
                with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
                    while chunk := r.read(1 << 20):
                        fh.write(chunk)
                tmp.replace(raw)
            except (urllib.error.URLError, OSError) as e:
                print(f"  broll: download failed {c['id']}: {e}", file=sys.stderr)
                tmp.unlink(missing_ok=True)
                continue
        out = CACHE / f"{c['id'].replace(':', '_')}_{seconds:.1f}s.mp4"
        if not out.exists():
            vf = f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H},fps=30"
            r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-t", f"{seconds:.3f}", "-an",
                                "-vf", vf, "-c:v", "libx264", "-crf", "16", "-preset", "fast",
                                "-pix_fmt", "yuv420p", str(out)], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  broll: conform failed {c['id']}: {r.stderr[-200:]}", file=sys.stderr)
                continue
        used.add(c["id"])
        print(f"  broll: '{term}' -> {c['id']} ({c['w']}x{c['h']}) {c['page']}")
        return out
    print(f"  broll: nothing full-HD for '{term}'", file=sys.stderr)
    return None


if __name__ == "__main__":
    term = sys.argv[1] if len(sys.argv) > 1 else "city night"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 4
    print(fetch(term, secs, set()))
