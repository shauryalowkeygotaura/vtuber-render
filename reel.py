"""reel.py -- script in, finished avatar reel out (voice, lip sync, director, b-roll, captions).

    doppler run -p creative-studio -c dev -- doppler run -p philosopher-pipeline -c dev -- \
        python reel.py --script scripts/dashboard-not-broken.txt

  1. voice      Groq Orpheus (default, --voice troy) or --tts edge; --audio uses a real take instead
  2. beats      performance.direct: Whisper + Groq emotion per sentence
  3. director   Groq picks spot / entrance / camera / arms / emotion / b-roll per shot
  4. edit.json  written to edits/<name>.json: edit any line by hand and re-run with --edit
  5. compose    Pexels b-roll + puppet + moves + rim + captions + audio -> output/<name>.mp4

Re-render a hand-edited plan without calling Groq again:
    python reel.py --edit edits/<name>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import talk
from vtuber import compose, director, lipsync, motion, performance
from vtuber.avatar import Rig

ROOT = Path(__file__).resolve().parent


def mouth_tracks(audio: Path, fps: int) -> tuple[list[float], list[float]]:
    lip = lipsync.visemes(lipsync.load_audio(str(audio)), fps)
    tl = motion.build(lip, [], fps)
    v = {k: np.asarray(tl["visemes"][k]) for k in ("aa", "ih", "ou", "ee", "oh")}
    open_ = np.minimum(1.0, v["aa"] + 0.8 * v["oh"] + 0.55 * v["ou"] + 0.45 * v["ih"] + 0.45 * v["ee"])
    widen = np.clip(v["ee"] + 0.6 * v["ih"] - v["ou"] - 0.6 * v["oh"], -1, 1)
    return [round(float(x), 3) for x in open_], [round(float(x), 3) for x in widen]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--script", help="text or a .txt path")
    src.add_argument("--audio", help="a real voice take")
    src.add_argument("--edit", help="re-render an existing edits/*.json")
    ap.add_argument("--tts", choices=("orpheus", "edge"), default="orpheus",
                    help="orpheus = Groq Orpheus (default, expressive); edge = free Microsoft voice")
    ap.add_argument("--voice", help="orpheus: troy (default), austin, daniel...; edge: en-IN-PrabhatNeural")
    ap.add_argument("--workers", type=int, help="parallel render processes (default 4)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-broll", action="store_true")
    ap.add_argument("--name")
    a = ap.parse_args()

    (ROOT / "output").mkdir(exist_ok=True)
    (ROOT / "edits").mkdir(exist_ok=True)
    if a.edit:
        edit_path = Path(a.edit)
        edit = json.loads(edit_path.read_text(encoding="utf-8"))
        name = edit_path.stem
    else:
        name = a.name or (Path(a.script).stem if a.script and Path(a.script).is_file() else
                          Path(a.audio).stem if a.audio else "reel")
        words = []
        if a.script:
            text = Path(a.script).read_text(encoding="utf-8") if Path(a.script).is_file() else a.script
            if a.tts == "orpheus":
                audio = ROOT / "output" / f"{name}.voice.wav"
                words = talk.synth_script(text, a.voice or "troy", audio)
            else:
                audio = ROOT / "output" / f"{name}.voice.mp3"
                words = talk.synth_edge(text, a.voice or "en-IN-PrabhatNeural", audio)
        else:
            audio = Path(a.audio)
        beats = performance.direct(str(audio), use_llm=not a.no_llm, expressions=tuple(Rig.LOOKS))
        if not beats:
            print("no speech found in the audio", file=sys.stderr)
            return 1
        shots = director.direct(beats, seed=a.seed, use_llm=not a.no_llm)
        mouth, widen = mouth_tracks(audio, compose.FPS)
        caps = talk.caption_chunks(words or talk.words_from_beats(beats))
        edit = {"seed": a.seed, "audio": str(audio), "beats": shots,
                "captions": [{"text": c[0], "start": round(c[1], 3), "end": round(c[2], 3)} for c in caps],
                "mouth": mouth, "widen": widen}
        edit_path = ROOT / "edits" / f"{name}.json"
        # beats first and readable; the long per-frame tracks last
        edit_path.write_text(json.dumps(edit, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  plan -> {edit_path} ({len(shots)} shots)")
    out = compose.render(edit, ROOT / "output" / f"{name}.mp4", use_broll=not a.no_broll, workers=a.workers)
    print(f"  video -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
