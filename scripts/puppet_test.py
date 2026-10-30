"""6 s arm-motion test of the cut-out puppet (no audio): python scripts/puppet_test.py out.mp4"""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vtuber.puppet import Puppet  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FPS, W, H = 30, 1080, 1920
p = Puppet.build(ROOT / "assets/avatar/rig-base.png")


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def back(x, s=1.7):  # overshoot ease-out: swings past and settles
    x = min(1.0, max(0.0, x)) - 1
    return 1 + x * x * ((s + 1) * x + s)


def pose_at(t):
    """Keyframed acts: wave 0-1.6, fan spin 1.6-3.6, flex 3.6-4.8, point 4.8-6."""
    P = {"A_hand_kind": "open", "B_hand_kind": "open", "head": 3 * math.sin(t * 2.2)}
    face = {"emotion": "neutral", "t": t, "blink": 1.0 if (t % 2.7) < 0.1 else 0.0}
    if t < 1.6:
        k = back(t / 0.45)
        P.update(B_shoulder=115 * k, B_elbow=70 * k + 25 * math.sin(t * 14) * ease(t / 0.5), B_wrist=10 * math.sin(t * 14))
        face["emotion"] = "happy"
    elif t < 3.6:
        u = t - 1.6
        k = back(u / 0.35)
        spin = u * 720  # two turns a second
        P.update(A_shoulder=90 * k, B_shoulder=90 * k, A_elbow=spin, B_elbow=spin + 180,
                 A_hand_kind="fist", B_hand_kind="fist", head=8 * math.sin(u * 9))
        face["emotion"] = "excited"
    elif t < 4.8:
        u = t - 3.6
        k = back(u / 0.3)
        P.update(A_shoulder=90 * k, B_shoulder=90 * k, A_elbow=125 * k, B_elbow=125 * k,
                 A_hand_kind="fist", B_hand_kind="fist", head=-4)
        face["emotion"] = "angry"
    else:
        u = t - 4.8
        k = back(u / 0.3)
        P.update(A_shoulder=75 * k, A_elbow=8 * k, A_hand_kind="point", head=6 * k)
        face["emotion"] = "smug"
    return P, face


out = sys.argv[1] if len(sys.argv) > 1 else "output/cartoon/puppet-test.mp4"
scale = 1.25
root = np.array([[scale, 0, W / 2 - 390 * scale], [0, scale, 130], [0, 0, 1]], np.float32)
ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                       "-framerate", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out],
                      stdin=subprocess.PIPE)
n = FPS * 6
for i in range(n):
    t = i / FPS
    pose, face = pose_at(t)
    bg = np.zeros((H, W, 4), np.float32)
    bg[..., :3] = (55, 47, 114)
    bob = 8 * math.sin(t * 4.4)
    r = root.copy()
    r[1, 2] += bob
    fr = p.render(pose, r, (W, H), face, bg)
    ff.stdin.write(fr[..., :3].clip(0, 255).astype(np.uint8).tobytes())
    if i % 30 == 0:
        print(f"{i}/{n}", flush=True)
ff.stdin.close()
ff.wait()
print(out)
