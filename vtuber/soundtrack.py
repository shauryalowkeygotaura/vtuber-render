"""soundtrack.py -- the playful audio layer: sounds on what's SAID, sounds on the moves.

Shaurya's rules (2026-11-05 review): the voice stays clean, sound effects carry the
fun (a party popper on a win, a cash register on "paying"), no background music.

Everything is driven by edit.json, so a hand edit or the Groq director can change it:

  per beat    "cue": "auto" | "none" | a CUES name | [names]   meaning sounds, on their word
              "sfx": "auto" | "none" | a SFX name                movement sound at the beat start
              "vfx": "none" | a VFX name   voice effect; auto = only a short punchline echo, once
              "mfx": music trick (drop / muffle / stutter), only matters with music
  top level   "music": null (default) | {"mood": "lofi"} | {"file": "x.mp3"}   opt-in

Missing fields are filled by plan() from the words (caption timings) and the acting
(entrance, camera, robot-hand tool), so old edits get the treatment too.

Rendering is one ffmpeg graph. The voice stays dry except inside effect windows:
there the dry voice is muted and a processed copy of the same slice plays at the
same time (pitch effects keep tempo, so lip sync holds). Reverb and echo tails
ring past the window on purpose. The music is ducked under the voice with a
sidechain compressor, and a limiter on the master stops any effect from clipping.

Audio-only preview (seconds, not minutes):
    python -m vtuber.soundtrack edits/zero-dollar-v4.json            # -> output/<name>.mix.wav
    python -m vtuber.soundtrack edits/x.json --music lofi --print
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import wave
from pathlib import Path

import numpy as np

from vtuber import sfx

ROOT = Path(__file__).resolve().parent.parent
MUSIC_DIR = ROOT / "assets" / "music"
IR_DIR = ROOT / "assets" / "ir"

# ── voice effects ────────────────────────────────────────────────────────────
# name -> (ffmpeg chain on the slice, reverb IR seconds or 0, tail seconds, what it's for)
VFX: dict[str, tuple[str, float, float, str]] = {
    "chipmunk":  ("rubberband=pitch=1.65:formant=shifted,volume=-1dB", 0, 0, "tiny squeaky voice: cute, shocked, small things"),
    "giant":     ("rubberband=pitch=0.62:formant=shifted,bass=g=4,volume=1dB", 0, 0, "huge slow demon voice: menace, big claims, villain"),
    "robot":     ("afftfilt=real='hypot(re,im)*sin(0)':imag='hypot(re,im)*cos(0)':win_size=512:overlap=0.75,"
                  "chorus=0.6:0.9:25:0.4:0.25:2,volume=16dB", 0, 0, "monotone machine: tech, AI, code, robot hands"),
    "megaphone": ("highpass=f=700,lowpass=f=3200,volume=14dB,asoftclip=type=hard,volume=-9dB", 0, 0,
                  "blown-out loudspeaker: announcements, shouting"),
    "phone":     ("highpass=f=350,lowpass=f=3300,acrusher=bits=8:mix=0.6,volume=2dB", 0, 0, "old phone line: calls, quotes"),
    "crunch":    ("volume=20dB,asoftclip=type=tanh,acrusher=bits=6:mode=log:mix=0.5,volume=-17dB", 0, 0,
                  "distorted overdrive: rage, hype"),
    "echo":      ("aecho=0.8:0.75:210|420|630:0.5|0.32|0.18", 0, 1.1, "slap echo: punchlines, the hook"),
    "cave":      ("", 2.8, 2.6, "huge reverb hall: dramatic, epic, the last line"),
    "dream":     ("lowpass=f=2600,chorus=0.5:0.9:50|60:0.4|0.32:0.25|0.4:2|2.3", 1.6, 1.8, "soft washed memory: sad, thinking"),
    "wobble":    ("vibrato=f=6.5:d=0.75", 0, 0, "seasick wobble: nervous, confused"),
    "glitch":    ("tremolo=f=17:d=0.95,acrusher=bits=7:mix=0.4,volume=2dB", 0, 0, "choppy broken signal: errors, bugs"),
    "alien":     ("rubberband=pitch=1.25,flanger=delay=4:depth=6:speed=0.6:regen=40,volume=1dB", 0, 0, "weird space voice"),
    "none":      ("", 0, 0, "dry"),
}

# what the music does under a beat (ffmpeg timeline filters on the music bus)
MFX = {
    "none": "",
    "drop": "volume=0",                    # music cuts out: the line lands in silence, then the beat slams back
    "muffle": "lowpass=f=420",             # music goes underwater / next room
    "stutter": "tremolo=f=8:d=1",          # music chops on the beat
}

# sfx menu: name -> (Openverse query, longest clip ms, peak dBFS, offset s after the hit)
SFX = {
    # movement sounds (land on entrances, camera hits, tool swaps)
    "whoosh": ("swoosh", 900, -17, 0.0),
    "soft_whoosh": ("soft whoosh", 1200, -20, 0.0),
    "pop": ("pop", 600, -16, 0.05),
    "boing": ("boing", 900, -17, 0.15),
    "slide_whistle": ("slide whistle", 1500, -19, 0.0),
    "impact": ("impact", 1200, -15, 0.12),
    "scratch": ("record scratch", 900, -16, 0.0),
    "air_horn": ("air horn", 1500, -18, 0.05),
    "drill": ("power drill", 1500, -19, 0.1),
    "rocket": ("rocket launch", 2000, -17, 0.1),
    "ratchet": ("ratchet", 1200, -18, 0.1),
    # meaning sounds (land on the word that earns them)
    "party_popper": ("party popper", 1500, -13, 0.0),
    "cash": ("cash register", 1500, -13, 0.0),
    "coins": ("coins", 1500, -14, 0.0),
    "phone_ring": ("telephone ring", 3000, -16, 0.0),
    "ding": ("ding", 1000, -14, 0.0),
    "notification": ("notification", 1000, -13, 0.0),
    "rip": ("paper rip", 1200, -13, 0.0),
    "typing": ("typing", 2500, -17, 0.0),
    "stamp": ("stamp", 800, -12, 0.0),
    "sad_trombone": ("sad trombone", 3000, -15, 0.0),
    "buzzer": ("wrong buzzer", 1500, -15, 0.0),
    "clock": ("clock ticking", 2000, -17, 0.0),
    "shutter": ("camera shutter", 800, -14, 0.0),
    "cheer": ("cheer", 3000, -16, 0.0),
    "glass": ("glass shatter", 2000, -16, 0.0),
    # meme sounds (CC0 / CC BY look-alikes; the real trending audio gets added in the app at post time)
    "vine_boom": ("deep boom", 2500, -11, 0.0),
    "bruh": ("bruh", 1500, -12, 0.0),
    "crickets": ("crickets", 3000, -16, 0.0),
    "fail_horn": ("fail horn", 2500, -14, 0.0),
    "oof": ("oof", 1000, -12, 0.0),
    "dramatic": ("dramatic sting", 3000, -15, 0.0),
    "correct": ("ding ding", 1500, -14, 0.0),
    "zoom_run": ("cartoon run", 1500, -16, 0.0),
}
ENTER_SFX = {"drop_from_sky": "slide_whistle", "pop_in_left": "pop", "pop_in_right": "pop",
             "slide_in_left": "soft_whoosh", "slide_in_right": "soft_whoosh",
             "peek_from_bottom": "boing", "spin_in": "whoosh"}
CAMERA_SFX = {"zoom_punch": "vine_boom", "whip_pan": "whoosh", "shake_heavy": "scratch"}
HAND_SFX = {"drill": "drill", "rocket": "rocket", "megaphone": "air_horn", "wrench": "ratchet"}

# meaning cues: the words that trigger it, and what it means (the director's menu text).
# Sounds play in spoken order; when two cues hit the same word, the one listed first wins
# (memes first: they are the funniest option).
CUES: dict[str, tuple[str, str]] = {
    "bruh": (r"\b(bruh|seriously|stupid|dumb|why would|who does that)\b", "meme: bruh, something dumb"),
    "oof": (r"\b(ouch|oof|expensive|hurts?|painful|billing you)\b", "meme: oof, that hurts"),
    "crickets": (r"\b(nobody|no one|silence|quiet|crickets)\b", "meme: crickets, awkward silence"),
    "fail_horn": (r"\b(fail\w*|wasted?|lost|broke|flop\w*)\b", "meme: clown horn, a fail"),
    "vine_boom": (r"\b(actually|turns out|the truth|secret|plot twist|zero|literally)\b",
                  "meme: vine boom, a reveal or shock"),
    "dramatic": (r"\b(but here'?s|the problem|dramatic|uh oh)\b", "meme: dramatic sting"),
    "correct": (r"\b(exactly|correct|nailed it|bingo)\b", "meme: correct ding"),
    "zoom_run": (r"\b(run\w*|fast\w*|quick\w*|rush\w*|switch\w*)\b", "meme: cartoon zoom, speed"),
    "cash": (r"\b(pay\w*|paid|charg\w*|bill\w*|money|price\w*|cost\w*|rupees?|dollars?|per minute)\b",
             "money, paying, billing"),
    "coins": (r"\b(free|save[sd]?|saving|cheap)\b", "free, saving money"),
    "phone_ring": (r"\b(call\w*|phones?|picks? up|ring\w*)\b", "a phone call, picking up"),
    "stamp": (r"\b(book\w*|approved|confirm\w*|sign\w*|appointment)\b", "booked, confirmed, approved"),
    "rip": (r"\b(ripp?\w*|tore|tear\w*|scrap\w*|ditch\w*)\b", "ripping something out"),
    "typing": (r"\b(cod(e|ing)|rebuil\w*|built|script\w*|program\w*)\b", "coding, building"),
    "notification": (r"\b(comment\w*|dm\w*|messag\w*|send you|subscribe|follow)\b", "comment, DM, notification"),
    "party_popper": (r"\b(that'?s it|whole thing|done|launch\w*|shipp?ed|finally|win\w*|congrat\w*|celebrat\w*)",
                     "celebration, a win, ta-da"),
    "sad_trombone": (r"\b(fail\w*|stupid|wasted?|lost|sad|broke)\b", "failure, embarrassing, sad"),
    "buzzer": (r"\b(wrong|mistakes?|errors?|bugs?|nope)\b", "wrong, error"),
    "clock": (r"\b(wait\w*|minutes?|hours?|time|slow)\b", "waiting, time"),
    "shutter": (r"\b(photo\w*|screenshot\w*|picture\w*)\b", "photo, screenshot"),
    "cheer": (r"\b(viral|crush\w*|insane|amazing)\b", "big hype"),
    "glass": (r"\b(crash\w*|smash\w*|shatter\w*)\b", "crash, smash"),
}


# ── your own meme clips (e.g. Clash of Clans) ────────────────────────────────
# Drop audio files in assets/sfx/memes/ and list them in memes.json:
#   {"hog_rider": {"file": "hog-rider.mp3", "words": ["let's go", "charge", "attack"], "gain_db": -12}}
# They join the cue menu AHEAD of everything else, so they win their words.
# The clip files are git-ignored: they are other people's audio, kept local.
MEME_DIR = ROOT / "assets" / "sfx" / "memes"
_SAFE_NAME = re.compile(r"[a-z0-9_]{1,30}")
_SAFE_WORD = re.compile(r"[a-z0-9' ]{1,40}")
_SAFE_FILE = re.compile(r"[A-Za-z0-9_.-]{1,80}\.(mp3|wav|ogg|m4a)")


def load_user_memes(meme_dir: Path = MEME_DIR) -> dict[str, dict]:
    """Whitelisted entries from memes.json whose file exists. Bad entries are skipped, loudly."""
    try:
        raw = json.loads((meme_dir / "memes.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        print(f"  memes.json is not valid JSON ({exc}); no custom memes", file=sys.stderr)
        return {}
    out = {}
    for name, m in (raw.items() if isinstance(raw, dict) else []):
        if not (isinstance(m, dict) and _SAFE_NAME.fullmatch(str(name)) and _SAFE_FILE.fullmatch(str(m.get("file", "")))):
            print(f"  memes.json: skipping {name!r} (name a-z0-9_, file like clip.mp3)", file=sys.stderr)
            continue
        given = m.get("words")
        given = [given] if isinstance(given, str) else given if isinstance(given, list) else []
        words = [w.strip().lower() for w in given if isinstance(w, str)]
        words = [w for w in words if _SAFE_WORD.fullmatch(w)]
        path = meme_dir / m["file"]
        if not words or not path.is_file():
            print(f"  memes.json: skipping {name!r} (needs words and {path.name} in {meme_dir.name}/)", file=sys.stderr)
            continue
        gain = m.get("gain_db", -12)
        out[name] = {"path": path, "words": words,
                     "gain_db": float(gain) if isinstance(gain, (int, float)) and -40 <= gain <= 0 else -12.0}
    return out


USER_MEMES = load_user_memes()
for _name, _m in USER_MEMES.items():
    SFX[_name] = ("file:" + str(_m["path"]), 0, _m["gain_db"], 0.0)
CUES = {**{n: (r"\b(" + "|".join(re.escape(w) for w in m["words"]) + r")\b", f"your meme clip: {', '.join(m['words'])}")
           for n, m in USER_MEMES.items()}, **{k: v for k, v in CUES.items() if k not in USER_MEMES}}

CUE_GAP = 0.3         # seconds between any two meaning sounds (speech is fast; clips are cut to CUE_MAX_S)
CUE_MAX_S = 0.9       # a meaning sound plays at most this long, then fades
CUE_REPEAT = 8.0      # the same meaning sound never twice inside this window
MOVE_CLEAR = 0.9      # a movement sound steps aside if a meaning sound lands this close

MOODS = {   # mood -> Openverse search (CC0 / CC BY, loopable beds). Music is opt-in: --music <mood>
    "funky": "upbeat funky loop",
    "lofi": "lofi melody loop",
    "chiptune": "chiptune",
    "hiphop": "hip hop beat",
    "piano": "upbeat piano loop",
}


# ── planning: fill vfx / mfx / sfx / cue from the acting and the words ───────
def pick_vfx(beat: dict, i: int, n: int, used: int) -> str:
    """Voice stays dry. The one exception: a short punchline (<= 4 words) that already
    carries a big move gets a slap echo, once per reel (Shaurya liked it on
    "In Hinglish, even." and rejected robot / wobble / giant / cave on normal lines)."""
    words = len(beat.get("text", "").split())
    big = beat.get("hand") == "rocket" or beat.get("camera") == "zoom_punch" or beat.get("enter") == "spin_in"
    if used == 0 and 0 < i < n - 1 and words <= 4 and big:
        return "echo"
    return "none"


def _word_time(beat: dict, pattern: str, captions: list[dict]) -> float | None:
    """Start of the caption chunk inside this beat whose words match `pattern`."""
    for c in captions:
        if beat["start"] - 0.05 <= c["start"] < beat["end"] and re.search(pattern, c["text"].lower()):
            return c["start"]
    return None


def plan(edit: dict, seed: int | None = None) -> list[dict]:
    """Beats with vfx / mfx / sfx / cue filled in and whitelisted. Hand-set values win.

    cues = [[name, time], ...]: meaning sounds ("cash", "party_popper" ...) timed to their words.
    """
    beats = edit["beats"]
    captions = edit.get("captions") or []
    n = len(beats)
    out, echoes = [], 0
    last_cue, last_by_name = -9.0, {}
    for i, b in enumerate(beats):
        b = dict(b)
        b["vfx"] = b["vfx"] if b.get("vfx") in VFX else pick_vfx(b, i, n, echoes)
        echoes += b["vfx"] != "none"
        b["mfx"] = b["mfx"] if b.get("mfx") in MFX else "none"
        # meaning sounds: a hand / director choice ("stamp" or ["phone_ring", "stamp"]) wins;
        # otherwise every cue word in the line, in the order they are spoken, max 2 per line
        want = b.get("cue", "auto")
        want = [want] if isinstance(want, str) else list(want) if isinstance(want, list) else ["auto"]
        text = b.get("text", "").lower()
        if want == ["auto"]:
            order = {name: k for k, name in enumerate(CUES)}
            found = sorted((m.start(), order[name], name) for name, (pat, _) in CUES.items()
                           if (m := re.search(pat, text)))
            want = [name for _, _, name in found]
        picked = []
        for name in [w for w in want if w in CUES]:
            t = _word_time(b, CUES[name][0], captions)
            t = b["start"] + 0.1 if t is None else t
            if len(picked) < 2 and t - last_cue >= CUE_GAP and t - last_by_name.get(name, -99.0) >= CUE_REPEAT:
                picked.append([name, round(t, 3)])
                last_cue, last_by_name[name] = t, t
        b["cues"] = picked
        # movement sound
        s = b.get("sfx", "auto")
        if s not in SFX and s != "none":
            s = (HAND_SFX.get(b.get("hand", "none")) or ENTER_SFX.get(b.get("enter", "none"))
                 or CAMERA_SFX.get(b.get("camera", "none")) or "none")
        b["sfx"] = s
        out.append(b)
    # movement sounds: one every 0.9 s at most, and they step aside for meaning sounds
    cue_times = [t for b in out for _, t in b["cues"]]
    last = -9.0
    for b in out:
        if b["sfx"] != "none":
            if b["start"] - last < 0.9 or any(abs(b["start"] - ct) < MOVE_CLEAR for ct in cue_times):
                b["sfx"] = "none"
            else:
                last = b["start"]
    return out


# ── assets: reverb impulse, music bed ────────────────────────────────────────
def impulse(seconds: float, sr: int) -> Path:
    """Synthetic room: decaying stereo-free noise with early reflections. Cached per size."""
    IR_DIR.mkdir(parents=True, exist_ok=True)
    path = IR_DIR / f"room_{seconds:.1f}s_{sr}.wav"
    if path.exists():
        return path
    rng = np.random.default_rng(int(seconds * 10))
    n = int(seconds * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / seconds)          # -60 dB at `seconds`
    ir = np.convolve(ir, np.ones(6) / 6, mode="same")                 # dull the top end a little
    for d, g in ((0.011, 0.6), (0.019, 0.45), (0.029, 0.35)):         # early reflections
        ir[int(d * sr)] += g
    ir[0] = 1.0
    ir /= np.max(np.abs(ir))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((ir * 32000).astype(np.int16).tobytes())
    return path


def music_track(mood: str) -> Path | None:
    """A CC0/CC-BY music bed for `mood`, cached in assets/music/ with its credit."""
    if mood not in MOODS:
        return None
    MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    hit = next(iter(MUSIC_DIR.glob(f"{mood}.*")), None)
    if hit:
        return hit
    query = sfx._safe_query(MOODS[mood])
    url = sfx.API + "?" + urllib.parse.urlencode({"q": query, "license": "cc0,by", "page_size": 20})
    try:
        data = json.loads(sfx._get(url))
    except (urllib.error.URLError, json.JSONDecodeError, RuntimeError, TimeoutError) as exc:
        print(f"  music search failed for {mood!r}: {exc}", file=sys.stderr)
        return None
    fits = [r for r in data.get("results", []) if r.get("license") in ("cc0", "by") and r.get("url")
            and 20_000 <= (r.get("duration") or 0) <= 240_000]
    if not fits:
        return None
    # short loops beat long songs (no intro, loop cleanly); CC0 beats CC BY (no credit needed)
    best = max(fits, key=lambda r: ("loop" in r.get("title", "").lower()) * 2 + (r["license"] == "cc0")
               - abs((r.get("duration") or 0) - 60_000) / 120_000)
    ext = (best.get("filetype") or "mp3")[:4]
    dest = MUSIC_DIR / f"{mood}.{ext}"
    try:
        dest.write_bytes(sfx._get(best["url"], timeout=60))
    except (urllib.error.URLError, RuntimeError, TimeoutError) as exc:
        print(f"  music download failed for {mood!r}: {exc}", file=sys.stderr)
        return None
    credits_path = MUSIC_DIR / "credits.json"
    try:
        credits = json.loads(credits_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        credits = {}
    credits[dest.name] = {k: best.get(k) for k in ("title", "creator", "license", "license_url")}
    credits[dest.name]["source"] = best.get("foreign_landing_url")
    credits_path.write_text(json.dumps(credits, indent=1, ensure_ascii=False), encoding="utf-8")
    return dest


def auto_mood(beats: list[dict]) -> str:
    """Pick a bed from how the reel feels overall."""
    score = {"funky": 0, "lofi": 0, "chiptune": 0, "hiphop": 0}
    for b in beats:
        e = b.get("emotion")
        score["funky"] += e in ("happy", "excited")
        score["lofi"] += e in ("relaxed", "thinking", "sad")
        score["hiphop"] += e in ("smug", "angry")
        score["chiptune"] += b.get("hand", "none") != "none"
    return max(score, key=score.get)


def resolve_music(edit: dict, beats: list[dict]) -> tuple[Path | None, float]:
    m = edit.get("music", {"mood": "auto"})   # on by default (Shaurya asked for it back 2026-11-05)
    if not m:
        return None, 0.0
    gain = float(m.get("gain_db", -17))
    if m.get("file"):
        p = Path(m["file"])
        return (p if p.is_file() else None), gain
    mood = m.get("mood", "auto")
    return music_track(auto_mood(beats) if mood == "auto" else mood), gain


# ── render ───────────────────────────────────────────────────────────────────
def _probe_sr(path: str) -> int:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return 24000


def _between(wins: list[tuple[float, float]]) -> str:
    return "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in wins)


def build(edit: dict, dest: Path) -> Path:
    """Render voice + effects + SFX + ducked music to `dest` (wav). Falls back to dry voice."""
    voice = edit["audio"]
    beats = plan(edit)
    total = beats[-1]["end"]
    sr = _probe_sr(voice)
    music, gain = resolve_music(edit, beats)
    inputs = ["-i", voice] + (["-stream_loop", "-1", "-i", str(music)] if music else [])   # music is input 1
    graph: list[str] = [f"[0:a]aformat=sample_rates={sr}:channel_layouts=mono,asplit=3[vdry][vfxsrc][vkey0]"]
    fx = [(b["start"], b["end"], b["vfx"]) for b in beats if b["vfx"] != "none"]
    mix = []
    if fx:
        graph.append(f"[vdry]volume=0:enable='{_between([(s, e) for s, e, _ in fx])}'[dry]")
        graph.append(f"[vfxsrc]asplit={len(fx)}" + "".join(f"[src{k}]" for k in range(len(fx))) if len(fx) > 1
                     else "[vfxsrc]anull[src0]")
        for k, (s, e, name) in enumerate(fx):
            chain, ir_s, tail, _ = VFX[name]
            d = e - s
            pre = (f"[src{k}]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS,"
                   f"afade=t=in:d=0.012,afade=t=out:st={max(0.0, d - 0.025):.3f}:d=0.025,apad=pad_dur={tail + 0.05:.2f}")
            if ir_s:
                inputs += ["-i", str(impulse(ir_s, sr))]
                ir_idx = inputs.count("-i") - 1
                graph.append(f"{pre}[pre{k}]")
                graph.append(f"[{ir_idx}:a]aformat=sample_rates={sr}:channel_layouts=mono[ir{k}]")
                wet = f"[pre{k}][ir{k}]afir=dry=6:wet=9:length=1" + (f",{chain}" if chain else "")
                graph.append(f"{wet},volume=-2dB,atrim=0:{d + tail:.3f},adelay={int(s * 1000)}:all=1[w{k}]")
            else:
                graph.append(f"{pre},{chain},atrim=0:{d + tail:.3f},adelay={int(s * 1000)}:all=1[w{k}]")
            mix.append(f"[w{k}]")
        mix.insert(0, "[dry]")
    else:
        mix.append("[vdry]")

    # SFX: movement sounds at the beat start, meaning sounds on their word
    hits = [(b["start"], b["sfx"], "") for b in beats if b["sfx"] != "none"]
    # meaning sounds are cut short so fast speech can carry two in a row (ring -> stamp)
    hits += [(t, name, f"atrim=0:{CUE_MAX_S},afade=t=out:st={CUE_MAX_S - 0.18:.2f}:d=0.18,")
             for b in beats for name, t in b["cues"]]
    for t, name, trim in hits:
        q, max_ms, peak, off = SFX[name]
        clip = Path(q[5:]) if q.startswith("file:") else sfx.fetch(q, max_ms)
        if not clip:
            continue
        inputs += ["-i", str(clip)]
        idx = inputs.count("-i") - 1
        level = peak - sfx._peak_db(clip)
        graph.append(f"[{idx}:a]aformat=sample_rates={sr}:channel_layouts=mono,{trim}volume={level:.1f}dB,"
                     f"adelay={int((t + off) * 1000)}:all=1[x{idx}]")
        mix.append(f"[x{idx}]")

    # voice bus (dry + effect windows) feeds the music ducker
    graph.append("".join(mix) + f"amix=inputs={len(mix)}:duration=longest:normalize=0[vbus]")
    if music:
        mtl = [f"aformat=sample_rates={sr}:channel_layouts=mono", f"atrim=0:{total + 1.5:.3f}", f"volume={gain:.1f}dB",
               "afade=t=in:d=0.25", f"afade=t=out:st={total:.3f}:d=1.4"]
        for name, filt in MFX.items():
            wins = [(b["start"], b["end"]) for b in beats if b["mfx"] == name]
            if filt and wins:
                mtl.append(f"{filt}:enable='{_between(wins)}'")
        graph.append(f"[1:a]{','.join(mtl)}[mus]")
        graph.append("[vkey0]apad[vkey]")
        graph.append("[mus][vkey]sidechaincompress=threshold=0.04:ratio=6:attack=15:release=350:makeup=1[duck]")
        graph.append("[vbus][duck]amix=inputs=2:duration=longest:normalize=0[pre]")
    else:
        graph.append("[vkey0]anullsink")
        graph.append("[vbus]anull[pre]")
    graph.append(f"[pre]apad=whole_dur={total + 1.5:.3f},atrim=0:{total + 1.5:.3f},alimiter=limit=0.93:level=disabled[out]")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(graph), "-map", "[out]",
           "-ar", "48000", str(dest)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  soundtrack failed, using dry voice:\n{r.stderr[-600:]}", file=sys.stderr)
        return Path(voice)
    return dest


def describe(beats: list[dict]) -> str:
    return "\n".join(f"{b['start']:6.2f}s  vfx={b['vfx']:<5} move={b['sfx']:<13} "
                     f"cues={' '.join(f'{n}@{t:.2f}' for n, t in b['cues']) or 'none':<32} "
                     f"{b.get('text', '')[:44]}" for b in beats)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="audio-only preview of an edit.json")
    ap.add_argument("edit")
    ap.add_argument("--music", help="mood (" + ", ".join(MOODS) + ", auto), a file path, or none")
    ap.add_argument("--print", action="store_true", help="show the per-beat plan")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    e = json.loads(Path(a.edit).read_text(encoding="utf-8"))
    if a.music:
        e["music"] = None if a.music == "none" else {"file": a.music} if Path(a.music).is_file() else {"mood": a.music}
    if a.print:
        print(describe(plan(e)))
    out = Path(a.out) if a.out else ROOT / "output" / f"{Path(a.edit).stem}.mix.wav"
    print(build(e, out))
