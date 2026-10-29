"""performance.py -- what the avatar FEELS and DOES, read from what was said.

Lip sync covers the mouth. Everything else a face does (smile on the joke,
brows up on the surprise, a nod on the point being landed) comes from the
words, so: Groq Whisper for timed segments, then one small Groq call that tags
each segment with an expression and a gesture.

Fails soft: no key, no network, bad JSON -> every beat is neutral and the
render still happens. A video with a calm face beats no video.
"""
from __future__ import annotations

import json
import os
import re
import sys

from . import groq_pool

EXPRESSIONS = ("neutral", "happy", "sad", "angry", "surprised", "relaxed")
GESTURES = ("none", "nod", "shake", "tilt", "lean_in", "look_away")

SYSTEM = (
    "You direct a VTuber avatar. For each numbered transcript segment pick the "
    "facial expression and one head gesture that a lively, confident 17-year-old "
    "creator would naturally make while saying it. Mostly neutral or happy; save "
    "surprised/angry/sad for lines that clearly carry them. Gestures: nod = "
    "agreeing or landing a point, shake = denying or 'no', tilt = questions or "
    "irony, lean_in = secrets, stakes, 'listen', look_away = thinking or recalling. "
    f"expression in {list(EXPRESSIONS)}, gesture in {list(GESTURES)}, intensity "
    "0.2-1.0. Reply JSON only: {\"beats\":[{\"i\":0,\"expression\":\"happy\","
    "\"intensity\":0.6,\"gesture\":\"nod\"}]}"
)


def transcribe(path: str) -> list[dict]:
    with open(path, "rb") as fh:
        data = fh.read()
    client = groq_pool.get_client()
    res = client.audio.transcriptions.create(
        file=(os.path.basename(path), data),
        model="whisper-large-v3-turbo",
        response_format="verbose_json",
    )
    segs = getattr(res, "segments", None) or (res.get("segments") if isinstance(res, dict) else None) or []
    out = []
    for s in segs:
        g = s.get if isinstance(s, dict) else (lambda k, d=None, _s=s: getattr(_s, k, d))
        text = (g("text") or "").strip()
        if text:
            out.extend(_sentences(float(g("start")), float(g("end")), text))
    return out


def _sentences(start: float, end: float, text: str) -> list[dict]:
    """Whisper often returns one segment for a whole breath group; one gesture
    per 13 seconds looks dead. Split on sentence marks (incl. Devanagari '।')
    and share the time by character count, which tracks speech length."""
    parts = [p.strip() for p in re.split(r"(?<=[.?!।])\s+", text) if p.strip()]
    if len(parts) < 2:
        return [{"start": start, "end": end, "text": text}]
    total = sum(len(p) for p in parts)
    out, t = [], start
    for p in parts:
        dt = (end - start) * len(p) / total
        out.append({"start": round(t, 3), "end": round(t + dt, 3), "text": p})
        t += dt
    return out


def _label_chunk(chunk: list[dict], offset: int) -> list[dict]:
    lines = "\n".join(f"{offset + i}: {s['text']}" for i, s in enumerate(chunk))
    resp = groq_pool.chat(
        groq_pool.FAST,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": lines}],
        max_tokens=min(900, 60 + 40 * len(chunk)),
        response_format={"type": "json_object"},
        require_content=True,
    )
    raw = resp.choices[0].message.content
    beats = json.loads(raw).get("beats", [])
    return beats if isinstance(beats, list) else []


def _clean(beat: dict, seg: dict) -> dict:
    expr = beat.get("expression") if beat.get("expression") in EXPRESSIONS else "neutral"
    gest = beat.get("gesture") if beat.get("gesture") in GESTURES else "none"
    try:
        inten = float(beat.get("intensity", 0.5))
    except (TypeError, ValueError):
        inten = 0.5
    return {**seg, "expression": expr, "gesture": gest, "intensity": max(0.0, min(1.0, inten))}


def direct(path: str, use_llm: bool = True) -> list[dict]:
    """Timed beats: [{start, end, text, expression, intensity, gesture}]."""
    try:
        segs = transcribe(path) if use_llm else []
    except Exception as exc:  # noqa: BLE001 - fail soft, see module doc
        print(f"  transcription failed ({exc.__class__.__name__}: {str(exc)[:120]}); neutral face", file=sys.stderr)
        return []
    beats: dict[int, dict] = {}
    for off in range(0, len(segs), 20):
        try:
            for b in _label_chunk(segs[off:off + 20], off):
                if isinstance(b, dict) and isinstance(b.get("i"), int):
                    beats[b["i"]] = b
        except Exception as exc:  # noqa: BLE001
            print(f"  direction failed for segments {off}+ ({str(exc)[:120]}); neutral", file=sys.stderr)
    return [_clean(beats.get(i, {}), s) for i, s in enumerate(segs)]
