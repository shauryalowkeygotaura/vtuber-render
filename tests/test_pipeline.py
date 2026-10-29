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
