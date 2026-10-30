"""Contact sheet of the avatar's states: python scripts/avatar_sheet.py art.png out.png"""
import sys
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vtuber.avatar import Rig  # noqa: E402

BG = (114, 47, 55, 255)
r = Rig.build(cv2.imread(sys.argv[1]))
font = ImageFont.truetype(str(Path(__file__).resolve().parent.parent / "assets/fonts/JetBrainsMono-ExtraBold.ttf"), 30)
states = [(e, {"emotion": e, "t": 0.3}) for e in Rig.LOOKS] + [("angry, talking", {"emotion": "angry", "open_": 0.6}), ("happy, talking", {"emotion": "happy", "open_": 0.7})]
fx, fy = r.face.centre
s = r.face.size * 1.6
crop = (int(fx - s * 0.6), int(fy - s * 0.75), int(fx + s * 0.6), int(fy + s * 0.65))
tiles = []
for name, kw in states:
    f = r.frame(**kw)
    bg = Image.new("RGBA", f.size, BG)
    bg.alpha_composite(f)
    t = bg.crop(crop).resize((408, 480), Image.LANCZOS)
    ImageDraw.Draw(t).text((12, 10), name, fill="white", font=font)
    tiles.append(t)
full = Image.new("RGBA", r.base.shape[1::-1], BG)
full.alpha_composite(r.frame(open_=0.5))
full = full.resize((int(full.width * 1440 / full.height), 1440), Image.LANCZOS)
sheet = Image.new("RGB", (408 * 4 + full.width, 1440), "black")
sheet.paste(full, (0, 0))
for i, t in enumerate(tiles):
    sheet.paste(t, (full.width + (i % 4) * 408, (i // 4) * 480))
sheet.save(sys.argv[2])
print(sys.argv[2])
