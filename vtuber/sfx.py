"""sfx.py -- real recorded sound effects from Openverse, placed on the acting.

Openverse indexes CC audio (mostly Freesound) behind a free, keyless API. We
only take CC0 and CC BY, keep the shortest clip that fits, cache it under
assets/sfx/ and record the credit line in assets/sfx/credits.json (CC BY needs
the author named in the post caption).

Placement is deliberately sparse: a sound on every line is the "voice + stock
slop" Shaurya rejected. Only beats that already carry a big move get one.
"""
from __future__ import annotations

import json
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.openverse.org/v1/audio/"
CACHE = Path(__file__).resolve().parent.parent / "assets" / "sfx"
ALLOWED_LICENSES = {"cc0", "by"}
UA = {"User-Agent": "vtuber-render/0.1 (personal reel tool)"}

# beat -> (search query, longest clip in ms, gain in dB under the voice)
CUES = {
    ("surprised", None): ("pop", 600, -14),
    (None, "lean_in"): ("soft whoosh", 1200, -18),
    (None, "shake"): ("record scratch", 900, -16),
}


def _safe_query(q: str) -> str:
    """Whitelist, not escaping: letters, digits and single spaces only."""
    if not re.fullmatch(r"[a-z0-9]+( [a-z0-9]+)*", q):
        raise ValueError(f"refusing sfx query {q!r}")
    return q


def _get(url: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status} from {url[:80]}")
        return r.read()


def fetch(query: str, max_ms: int) -> Path | None:
    """Best short CC0/CC-BY clip for `query`, cached. None if nothing fits."""
    query = _safe_query(query)
    CACHE.mkdir(parents=True, exist_ok=True)
    slug = query.replace(" ", "-")
    hit = next(iter(CACHE.glob(f"{slug}.*")), None)
    if hit:
        return hit
    url = API + "?" + urllib.parse.urlencode({"q": query, "license": ",".join(sorted(ALLOWED_LICENSES)), "page_size": 20})
    try:
        data = json.loads(_get(url))
    except (urllib.error.URLError, json.JSONDecodeError, RuntimeError, TimeoutError) as exc:
        print(f"  sfx search failed for {query!r}: {exc}")
        return None
    fits = [r for r in data.get("results", [])
            if r.get("license") in ALLOWED_LICENSES and r.get("url") and 80 <= (r.get("duration") or 0) <= max_ms]
    if not fits:
        return None
    best = min(fits, key=lambda r: r["duration"])
    ext = (best.get("filetype") or Path(urllib.parse.urlparse(best["url"]).path).suffix.lstrip(".") or "mp3")[:4]
    dest = CACHE / f"{slug}.{ext}"
    try:
        dest.write_bytes(_get(best["url"], timeout=40))
    except (urllib.error.URLError, RuntimeError, TimeoutError) as exc:
        print(f"  sfx download failed for {query!r}: {exc}")
        return None
    credits_path = CACHE / "credits.json"
    try:
        credits = json.loads(credits_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        credits = {}
    credits[dest.name] = {
        "title": best.get("title"), "creator": best.get("creator"), "license": best.get("license"),
        "license_url": best.get("license_url"), "source": best.get("foreign_landing_url"),
    }
    credits_path.write_text(json.dumps(credits, indent=1, ensure_ascii=False), encoding="utf-8")
    return dest


def cues(beats: list[dict]) -> list[tuple[float, Path, int]]:
    """(time_s, file, gain_db) for each beat that earns a sound; max one per 2 s."""
    out: list[tuple[float, Path, int]] = []
    last = -10.0
    for b in beats:
        for (expr, gest), (query, max_ms, gain) in CUES.items():
            if (expr and b["expression"] == expr and b["intensity"] >= 0.5) or (gest and b["gesture"] == gest):
                if b["start"] - last >= 2.0:
                    clip = fetch(query, max_ms)
                    if clip:
                        out.append((b["start"], clip, gain))
                        last = b["start"]
                break
    return out


def mix(voice: str, placed: list[tuple[float, Path, int]], dest: Path) -> Path:
    """Voice + placed effects -> one wav. Voice level is untouched."""
    if not placed:
        return Path(voice)
    args = ["ffmpeg", "-v", "error", "-y", "-i", voice]
    chains = []
    for i, (t, clip, gain) in enumerate(placed, start=1):
        args += ["-i", str(clip)]
        ms = int(t * 1000)
        chains.append(f"[{i}:a]aformat=channel_layouts=mono,volume={gain}dB,adelay={ms}:all=1[s{i}]")
    labels = "".join(f"[s{i}]" for i in range(1, len(placed) + 1))
    graph = ";".join(chains) + f";[0:a]{labels}amix=inputs={len(placed) + 1}:duration=first:normalize=0[out]"
    args += ["-filter_complex", graph, "-map", "[out]", str(dest)]
    proc = subprocess.run(args, capture_output=True, check=False)
    if proc.returncode != 0:
        print(f"  sfx mix failed, using dry voice: {proc.stderr.decode(errors='replace')[:200]}")
        return Path(voice)
    return dest
