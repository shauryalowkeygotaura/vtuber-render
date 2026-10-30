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
