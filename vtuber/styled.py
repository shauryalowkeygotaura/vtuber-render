"""styled.py -- vtuber-render's adapter over the shared drawn-scenes engine (drawn_scenes/, vendored).

    python -m vtuber.styled papercut "ship it today" rocket 3 --still    # eyeball one frame

The drawing (papercut / sketch scenes, any size) and the LLM stock-vs-drawn mix live in drawn_scenes/,
shared with format-render, football-shorts-autopilot and philosopher-pipeline. Edit it in
Code/drawn-scenes and run its sync.sh. This file only adds what is specific to the avatar reels:
portrait size, the assets/broll cache, our Groq key pool, and the beat-aware pick()/decide().
"""
from __future__ import annotations

import sys
from pathlib import Path

import drawn_scenes as ds
from drawn_scenes.core import fallback_spec as _fallback  # noqa: F401  (kept for tests/back-compat)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "assets" / "broll"
W, H = 1080, 1920
STYLES = ds.STYLES
MODES = ("stock", "styled", "mix")
ICONS = ds.ICONS
_clean_word = ds.clean_word
choose_style = ds.choose_style

VIDEO = "a fast, funny vertical reel about tech and building things, starring a cartoon avatar"
STOCK = "real portrait stock footage from Pexels that shows what the line says"


def _chat(messages: list, max_tokens: int) -> str:
    from vtuber import groq_pool  # noqa: PLC0415
    resp = groq_pool.chat(groq_pool.SMART, messages=messages, max_tokens=max_tokens,
                          response_format={"type": "json_object"}, require_content=True)
    return resp.choices[0].message.content


def plan(lines: list[str], decide: bool = False) -> list[dict]:
    return ds.plan(lines, decide=decide, video=VIDEO, stock=STOCK, chat=_chat)


def _same_as_prev(beats: list[dict]) -> list[bool]:
    """Halves of a split line share their b-roll term (director.py), so they stay one background type."""
    return [i > 0 and bool(b.get("cont") or (b.get("broll") and b.get("broll") == beats[i - 1].get("broll")))
            for i, b in enumerate(beats)]


def pick(beats: list[dict], mode: str) -> set[int]:
    """Fixed rules: stock -> none, styled -> all, otherwise the alternation fallback."""
    if mode == "stock":
        return set()
    if mode == "styled":
        return set(range(len(beats)))
    return ds.alternate(len(beats), _same_as_prev(beats))


def decide(beats: list[dict], specs: list[dict]) -> set[int]:
    """mix: Groq's per-line stock/drawn call, no quota; alternation only for lines it didn't answer."""
    return ds.decide(specs, _same_as_prev(beats))


def render(style: str, spec: dict, seconds: float, seed: int = 7) -> Path:
    return ds.render(style, spec, seconds, CACHE, size=(W, H), seed=seed)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("style", choices=STYLES)
    ap.add_argument("word")
    ap.add_argument("icon", choices=sorted(ICONS))
    ap.add_argument("seconds", type=float, nargs="?", default=3.0)
    ap.add_argument("--still", action="store_true")
    a = ap.parse_args()
    spec = {"word": a.word, "icon": a.icon}
    if a.still:
        (ROOT / "output").mkdir(exist_ok=True)
        print(ds.still(a.style, spec, ROOT / "output" / f"still_{a.style}_{a.icon}.png", size=(W, H)))
    else:
        print(render(a.style, spec, a.seconds))
    sys.exit(0)
