import numpy as np

from vtuber import lipsync, motion, performance


def tone(freq: float, secs: float) -> np.ndarray:
    t = np.arange(int(lipsync.SR * secs)) / lipsync.SR
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_silence_keeps_mouth_shut_and_voice_opens_it():
    audio = np.concatenate([np.zeros(lipsync.SR), tone(600, 1.0)])
    v = lipsync.visemes(audio, 30)
    assert v["frames"] == 60
    jaw = np.array(v["aa"]) + np.array(v["oh"]) + np.array(v["ee"])
    assert jaw[:25].max() < 0.05
    assert jaw[40:].max() > 0.3


def test_bright_sound_reads_as_spread_lips_not_open_jaw():
    v = lipsync.visemes(np.concatenate([np.zeros(8000), tone(3000, 1.0)]), 30)
    assert max(v["ee"]) > max(v["aa"])


def test_long_segment_splits_on_hindi_and_english_sentence_marks():
    parts = performance._sentences(0.0, 10.0, "Haan. Kya baat hai? ठीक है। Done!")
    assert [p["text"] for p in parts] == ["Haan.", "Kya baat hai?", "ठीक है।", "Done!"]
    assert parts[0]["start"] == 0.0 and abs(parts[-1]["end"] - 10.0) < 0.01


def test_build_is_deterministic_and_frame_aligned():
    lip = lipsync.visemes(tone(500, 2.0), 30)
    beats = [{"start": 0.2, "end": 1.5, "text": "x", "expression": "happy", "intensity": 0.8, "gesture": "nod"}]
    a, b = motion.build(lip, beats, 30), motion.build(lip, beats, 30)
    assert a == b
    assert len(a["head"]["pitch"]) == len(a["blink"]) == lip["frames"]
    assert max(a["face"]["happy"]) > 0.3


def test_sfx_only_on_big_beats_and_spaced(monkeypatch):
    from pathlib import Path

    from vtuber import sfx
    monkeypatch.setattr(sfx, "fetch", lambda q, m: Path(f"{q}.mp3"))
    beats = [
        {"start": 0.5, "expression": "surprised", "intensity": 0.9, "gesture": "none"},
        {"start": 1.0, "expression": "neutral", "intensity": 0.5, "gesture": "lean_in"},   # too close
        {"start": 4.0, "expression": "happy", "intensity": 0.6, "gesture": "nod"},        # no cue
        {"start": 6.0, "expression": "neutral", "intensity": 0.5, "gesture": "lean_in"},
    ]
    placed = sfx.cues(beats)
    assert [(t, p.name) for t, p, _ in placed] == [(0.5, "pop.mp3"), (6.0, "soft whoosh.mp3")]


def test_sfx_query_is_whitelisted():
    import pytest

    from vtuber import sfx
    for bad in ("pop;rm", "../x", "Pop", "a  b"):
        with pytest.raises(ValueError):
            sfx._safe_query(bad)


def test_captions_break_on_sentences_and_pauses():
    import talk
    words = [("Rewrote", 0.0, 0.4), ("nothing.", 0.4, 0.9), ("Checked", 1.0, 1.4),
             ("every", 1.4, 1.6), ("deploy.", 1.6, 2.0), ("All", 2.6, 2.8), ("green.", 2.8, 3.1)]
    caps = [c[0] for c in talk.caption_chunks(words)]
    assert caps == ["Rewrote nothing.", "Checked every", "deploy.", "All green."]


def test_photo_rig_moves_only_the_lower_face():
    import numpy as np
    from vtuber import photo2d
    pts = np.zeros((478, 2), np.float32) + (100, 100)
    pts[photo2d.FOREHEAD] = (100, 40); pts[photo2d.CHIN] = (100, 160)
    pts[photo2d.CORNER_L] = (85, 120); pts[photo2d.CORNER_R] = (115, 120)
    pts[13] = pts[14] = (100, 120)
    for i, x in zip(photo2d.EYE_L, (70, 90, 80, 80)): pts[i] = (x, 85)
    for i, x in zip(photo2d.EYE_R, (130, 110, 120, 120)): pts[i] = (x, 85)
    img = (np.random.default_rng(0).random((220, 200, 3)) * 255).astype(np.uint8)
    face = photo2d.Face(pts, np.array([0, -1.0]), np.array([1.0, 0]), pts[13].copy(), 15.0, 40.0,
                        np.array([100, 100.0]), 120.0)
    rig = photo2d.Rig(img, face)
    shut = rig.frame(0, 0, "neutral", 0, 0)
    talking = rig.frame(1, 0, "neutral", 0, 0)
    assert np.array_equal(shut[:60], img[:60])            # above the face: untouched
    assert not np.array_equal(talking[118:150, 85:115], shut[118:150, 85:115])  # mouth opened


def _edit(beats, **kw):
    return {"seed": 7, "audio": None, "beats": beats, **kw}


def test_soundtrack_keeps_voice_dry_and_puts_sounds_on_meaning():
    from vtuber import soundtrack as st
    beats = [
        {"start": 0, "end": 2, "text": "voice AI is charging you", "emotion": "angry", "enter": "drop_from_sky",
         "camera": "zoom_punch", "hand": "none"},
        {"start": 2, "end": 4, "text": "loud bit", "emotion": "smug", "enter": "none", "camera": "none", "hand": "megaphone"},
        {"start": 4, "end": 5.2, "text": "In Hinglish, even.", "emotion": "excited", "hand": "rocket", "camera": "zoom_punch"},
        {"start": 5.2, "end": 7, "text": "It picks up, it books the appointment", "vfx": "rm -rf", "mfx": "evil",
         "sfx": "../../x", "cue": "evil"},
        {"start": 7, "end": 9, "text": "bye", "emotion": "neutral", "vfx": "chipmunk"},
    ]
    caps = [{"text": "voice AI is", "start": 0.0, "end": 0.9}, {"text": "charging you", "start": 0.9, "end": 2.0},
            {"text": "It picks up,", "start": 5.2, "end": 5.6}, {"text": "it books the", "start": 5.6, "end": 6.0},
            {"text": "appointment", "start": 6.0, "end": 6.6}]
    p = st.plan(_edit(beats, captions=caps))
    assert all(b["vfx"] in st.VFX and b["mfx"] in st.MFX and (b["sfx"] in st.SFX or b["sfx"] == "none") for b in p)
    assert [b["vfx"] for b in p] == ["none", "none", "echo", "none", "chipmunk"]   # dry, one punchline echo, hand edit wins
    assert p[0]["cues"] == [["cash", 0.9]]                    # cash register on "charging", not the line start
    assert p[3]["cues"] == []                                 # an unknown hand-set cue is dropped, not guessed
    auto = st.plan(_edit([dict(beats[3], cue="auto")], captions=caps))
    assert [c[0] for c in auto[0]["cues"]] == ["phone_ring", "stamp"]   # spoken order, two per line
    assert st.resolve_music(_edit(beats, music=None), p) == (None, 0.0)   # "music": null turns the bed off
    meme = st.plan(_edit([{"start": 0, "end": 2, "text": "it's kind of stupid"}], captions=[]))
    assert meme[0]["cues"][0][0] == "bruh"                   # memes win a shared word


def test_speed_edit_keeps_everything_on_the_same_words(tmp_path):
    import wave
    import reel
    sr = 16000
    voice = tmp_path / "v.wav"
    with wave.open(str(voice), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(np.zeros(sr * 3, np.int16).tobytes())
    e = {"audio": str(voice), "beats": [{"start": 0.0, "end": 1.5}, {"start": 1.5, "end": 3.0}],
         "captions": [{"text": "hi", "start": 1.2, "end": 1.8}], "mouth": [0.0] * 45 + [1.0] * 45, "widen": [0.0] * 90}
    f = reel.speed_edit(e, 1.5, tmp_path / "fast.wav")
    assert f["beats"][1] == {"start": 1.0, "end": 2.0} and f["captions"][0]["start"] == 0.8
    assert len(f["mouth"]) == 60 and f["mouth"][29] == 0.0 and f["mouth"][31] == 1.0   # mouth opens at 1.0 s, not 1.5
    assert e["beats"][1]["start"] == 1.5                     # the original plan is untouched


def test_soundtrack_renders_offline_and_keeps_timing(tmp_path):
    import subprocess
    import wave
    from vtuber import soundtrack as st
    sr = 24000
    t = np.arange(int(sr * 4)) / sr
    tone = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    voice = tmp_path / "v.wav"
    with wave.open(str(voice), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(tone.tobytes())
    beats = [{"start": 0, "end": 1.5, "text": "a", "vfx": "chipmunk", "sfx": "none"},
             {"start": 1.5, "end": 3, "text": "b", "vfx": "none", "sfx": "none"},
             {"start": 3, "end": 4, "text": "c", "vfx": "cave", "sfx": "none"}]
    out = st.build({"seed": 1, "audio": str(voice), "beats": beats, "music": None}, tmp_path / "m.wav")
    assert out == tmp_path / "m.wav"
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                               capture_output=True, text=True).stdout)
    assert 4.0 <= dur <= 5.6
