"""Pose test sheet for the cut-out puppet: python scripts/puppet_sheet.py out.png"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vtuber.puppet import Puppet  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
t0 = time.time()
p = Puppet.build(ROOT / "assets/avatar/rig-base.png")
print(f"built in {time.time() - t0:.1f}s")
POSES = [
    ("rest (mech)", {"A_hand_kind": "open", "B_hand_kind": "open"}, {}),
    ("both up", {"A_shoulder": 160, "B_shoulder": 160, "A_elbow": 10, "B_elbow": 10, "A_hand_kind": "rock", "B_hand_kind": "peace"}, {"emotion": "excited"}),
    ("wave", {"B_shoulder": 110, "B_elbow": 70, "head": -6, "B_hand_kind": "open", "A_hand_kind": "fist"}, {"emotion": "happy"}),
    ("point", {"A_shoulder": 80, "A_elbow": 5, "head": 4, "A_hand_kind": "point"}, {"emotion": "smug"}),
    ("shrug", {"A_shoulder": 25, "A_elbow": 75, "A_wrist": 60, "B_shoulder": 25, "B_elbow": 75, "B_wrist": 60, "head": 8, "A_hand_kind": "open", "B_hand_kind": "open"}, {"emotion": "nervous"}),
    ("flex", {"A_shoulder": 90, "A_elbow": 120, "B_shoulder": 90, "B_elbow": 120, "A_hand_kind": "fist", "B_hand_kind": "fist"}, {"emotion": "angry"}),
    ("fan 1", {"A_shoulder": 90, "A_elbow": 0, "B_shoulder": 90, "B_elbow": 180, "A_hand_kind": "claw", "B_hand_kind": "thumbs"}, {"emotion": "excited"}),
    ("fan 2", {"A_shoulder": 90, "A_elbow": 90, "B_shoulder": 90, "B_elbow": 270}, {"emotion": "excited"}),
]
W, H = 768, 1100
tiles = []
for name, pose, face in POSES:
    bg = np.zeros((H, W, 4), np.float32)
    bg[..., :3] = (55, 47, 114)
    root = np.array([[1, 0, 0], [0, 1, -60], [0, 0, 1]], np.float32)
    out = p.render(pose, root, (W, H), face, bg)
    img = out[..., :3].clip(0, 255).astype(np.uint8)
    cv2.putText(img, name, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
    tiles.append(cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA))
rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
cv2.imwrite(sys.argv[1], np.vstack(rows))
print(f"{sys.argv[1]} in {time.time() - t0:.1f}s")
