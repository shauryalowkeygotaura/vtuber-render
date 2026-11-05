"""director.py -- Groq picks the moves, rules keep it watchable, code does the motion.

Input: timed beats from performance.direct ([{start, end, text, expression, intensity}]).
Output: edit.json beats with spot / enter / camera / arms / broll, ready for compose.

Groq only chooses names from fixed menus (the part language models are good at).
Every answer is whitelist-checked; a bad or missing answer falls back to the
rules below, so a render never depends on the LLM behaving.
"""
from __future__ import annotations

import json
import random
import sys

from vtuber import groq_pool, moves
from vtuber.soundtrack import CUES
from vtuber.mechhand import TOOLS
from vtuber.avatar import Rig

MAX_SHOT = 3.2   # seconds; longer beats are cut into two shots so the frame keeps moving

SYSTEM = f"""You direct a fast, funny vertical reel starring a cartoon avatar (a manhwa-style guy in a white shirt
and black tie, with robot hands). For EACH numbered line, choose:
- spot: one of {list(moves.SPOTS)}
- enter: one of {list(moves.ENTRANCES)}  (how he appears for this shot; "none" = he's already there)
- camera: one of {list(moves.CAMERAS)}
- arms: one of {list(moves.ARMS)}
- emotion: one of {list(Rig.LOOKS)}
- hand: "none" or a robot-hand tool that swaps in, ONLY when the line clearly earns it:
  {"; ".join(f"{k} = {v}" for k, v in TOOLS.items())}. Use it on at most a third of the lines.
- sound: a sound effect that plays what the line SAYS (like a party popper on a win), one of:
  {"; ".join(f"{k} = {v[1]}" for k, v in CUES.items())}; or "none". Only when a word in the line clearly
  earns it; about half the lines, never the same sound on neighbouring lines. His voice itself stays clean.
- broll: 2-4 word stock-footage search for a VERTICAL background clip that shows the line's idea literally
  (concrete, filmable things: "laptop code screen", "delhi traffic night", not abstractions)
Rules: the first line is the hook, make it loud (drop_from_sky or spin_in, zoom_punch or shake_heavy).
Match arms to meaning (pointing at the viewer for "you", shrug for doubt, flex for wins, fan_spin for hype,
thumbs_up for approval, wave for hello/bye). Vary everything: never repeat an entrance or spot back to back.
Reply JSON: {{"shots": [{{"i": 0, "spot": ..., "enter": ..., "camera": ..., "arms": ..., "emotion": ..., "hand": ..., "sound": ..., "broll": ...}}]}}"""


def _split(beats: list[dict]) -> list[dict]:
    out = []
    for b in beats:
        n = max(1, round((b["end"] - b["start"]) / MAX_SHOT))
        words = b["text"].split()
        for k in range(n):
            s = b["start"] + (b["end"] - b["start"]) * k / n
            e = b["start"] + (b["end"] - b["start"]) * (k + 1) / n
            part = " ".join(words[len(words) * k // n: len(words) * (k + 1) // n])
            out.append({**b, "start": round(s, 3), "end": round(e, 3), "text": part or b["text"], "cont": k > 0})
    return out


def _ask(shots: list[dict]) -> dict[int, dict]:
    lines = "\n".join(f"{i}: {s['text']}" for i, s in enumerate(shots))
    try:
        resp = groq_pool.chat(groq_pool.SMART,
                              messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": lines}],
                              max_tokens=min(4000, 120 + 90 * len(shots)),
                              response_format={"type": "json_object"}, require_content=True)
        data = json.loads(resp.choices[0].message.content)
        got = data.get("shots", []) if isinstance(data, dict) else []
        return {int(g["i"]): g for g in got if isinstance(g, dict) and str(g.get("i", "")).isdigit()}
    except Exception as e:  # noqa: BLE001 - rules take over
        print(f"  director: Groq unavailable ({str(e)[:120]}); rules only", file=sys.stderr)
        return {}


def _safe_term(t) -> str:
    """Whitelist a b-roll query: letters, digits, spaces; 2-5 words."""
    if not isinstance(t, str):
        return ""
    words = ["".join(ch for ch in w if ch.isalnum()) for w in t.lower().split()]
    words = [w for w in words if w][:5]
    return " ".join(words) if len(words) >= 2 else ""


def direct(beats: list[dict], seed: int = 7, use_llm: bool = True) -> list[dict]:
    rng = random.Random(seed)
    shots = _split(beats)
    # shots must tile the timeline: a pause between sentences belongs to the shot before it,
    # otherwise the next shot's entrance (e.g. drop_from_sky) sits off-screen through the gap
    for a, b in zip(shots, shots[1:]):
        a["end"] = b["start"]
    if shots:
        shots[0]["start"] = 0.0
    picks = _ask(shots) if use_llm else {}
    out, prev = [], {}
    for i, s in enumerate(shots):
        g = picks.get(i, {})
        b = {"start": s["start"], "end": s["end"], "text": s["text"]}
        b["emotion"] = g.get("emotion") if g.get("emotion") in Rig.LOOKS else \
            s.get("expression") if s.get("expression") in Rig.LOOKS else "neutral"
        b["spot"] = g.get("spot") if g.get("spot") in moves.SPOTS else rng.choice(list(moves.SPOTS))
        b["enter"] = g.get("enter") if g.get("enter") in moves.ENTRANCES else rng.choice(moves.ENTRANCES[:-1])
        b["camera"] = g.get("camera") if g.get("camera") in moves.CAMERAS else rng.choice(moves.CAMERAS)
        b["arms"] = g.get("arms") if g.get("arms") in moves.ARMS else rng.choice(moves.ARMS)
        b["hand"] = g.get("hand") if g.get("hand") in TOOLS else "none"
        # a swap only lands if it's rare: never two shots in a row, never the same tool within ~5 s
        if b["hand"] != "none" and prev and (prev.get("hand") != "none" or
                                             any(o.get("hand") == b["hand"] and s["start"] - o["start"] < 5
                                                 for o in out)):
            b["hand"] = "none"
        # meaning sound; "auto" = soundtrack keyword rules decide. Voice FX are not the director's call.
        b["cue"] = g.get("sound") if g.get("sound") in CUES or g.get("sound") == "none" else "auto"
        b["broll"] = _safe_term(g.get("broll")) or (prev.get("broll") if s.get("cont") else "") or "city night lights"
        # ── rules layer ─────────────────────────────────────────────────────
        if i == 0:   # hook: loud entrance, hard camera
            if b["enter"] not in ("drop_from_sky", "spin_in", "pop_in_left", "pop_in_right"):
                b["enter"] = "drop_from_sky"
            if b["camera"] not in ("zoom_punch", "shake_heavy"):
                b["camera"] = "zoom_punch"
        if prev:
            if b["enter"] == prev["enter"] and b["enter"] != "none":
                b["enter"] = rng.choice([e for e in moves.ENTRANCES[:-1] if e != prev["enter"]])
            if b["spot"] == prev["spot"]:
                b["spot"] = rng.choice([p for p in moves.SPOTS if p != prev["spot"]])
        if s.get("cont"):     # second half of a long line: same background, new framing
            b["broll"] = prev.get("broll", b["broll"])
            if b["enter"] in ("drop_from_sky", "spin_in"):
                b["enter"] = rng.choice(["slide_in_left", "slide_in_right", "pop_in_left", "pop_in_right"])
        if i == len(shots) - 1:   # end on a spike
            b["camera"] = "shake_heavy" if b["camera"] in ("none", "slow_push") else b["camera"]
        out.append(b)
        prev = b
    return out
