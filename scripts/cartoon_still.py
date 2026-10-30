"""One test frame of the cartoon avatar: python scripts/cartoon_still.py <photo> [out.png]"""
import sys
from pathlib import Path

import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vtuber import cartoon, photo2d  # noqa: E402

W, H = 1080, 1920
BG = (114, 47, 55, 255)  # burgundy

src = sys.argv[1]
out = Path(sys.argv[2] if len(sys.argv) > 2 else "output/cartoon/still.png")
img = cv2.imread(src)
face = photo2d.landmarks(img)
cut = cartoon.cutout(img)
x0, y0, x1, y1 = cut.getbbox()
cut = cut.crop((x0, y0, x1, y1))
# frame on the face: head fills ~40% of the width, face centre at 40% height
scale = W * 0.40 / face.size
cut = cut.resize((int(cut.width * scale), int(cut.height * scale)), Image.LANCZOS)
fx, fy = (face.centre[0] - x0) * scale, (face.centre[1] - y0) * scale
ox, oy = W / 2 - fx, H * 0.40 - fy
oy = max(oy, H - cut.height)  # never leave a gap under the body
cut = cartoon.sticker(cartoon.toonify(cut), 10)
frame = Image.new("RGBA", (W, H), BG)
if len(sys.argv) > 3:  # b-roll still behind
    bg = Image.open(sys.argv[3]).convert("RGBA")
    k = max(W / bg.width, H / bg.height)
    bg = bg.resize((int(bg.width * k) + 1, int(bg.height * k) + 1), Image.LANCZOS)
    frame.alpha_composite(bg.crop(((bg.width - W) // 2, (bg.height - H) // 2, (bg.width - W) // 2 + W, (bg.height - H) // 2 + H)))
frame.alpha_composite(cut, (int(ox), int(oy)))
# face points live in the original photo: shift by the crop, then scale
offset = (ox - x0 * scale, oy - y0 * scale)
cartoon.draw_face(frame, face, offset, scale, eye_open=1.0, look=(0.2, 0.1),
                  mouth_open=0.6, widen=0.1, emotion="neutral")
out.parent.mkdir(parents=True, exist_ok=True)
frame.convert("RGB").save(out)
print(out)
