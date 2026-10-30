# Pose pack prompts (Gemini app)

Upload `manhwa-v1.png` (this folder) with EVERY prompt so the character stays identical.
Prefix each prompt with:

> Same character, same art style, same outfit as the attached image (white shirt sleeves rolled, slim black tie, silver crescent-moon chain over the tie, black watch on left wrist, silver star bracelet on right wrist, black trousers). Full body visible from head to knees, whole body in the frame, nothing cropped. Flat plain solid green background, no shadow on the floor, no text. 9:16.

Then add ONE pose line. Save each as `poses/<name>.png` with the name shown:

| file | pose line |
|---|---|
| `point.png` | Pointing straight at the viewer with his right arm fully extended, leaning forward, other hand on hip. |
| `crossed.png` | Arms crossed over his chest, weight on one leg, head tilted slightly, unimpressed. |
| `think.png` | One hand on his chin, other arm across his waist supporting the elbow, looking up and to the side. |
| `shock.png` | Both hands on his head, leaning back, knees slightly bent, mouth open in shock. |
| `hype.png` | Jumping with one fist raised high in the air, other arm pumped at his side, legs bent mid-jump. |
| `shrug.png` | Classic shrug: both palms up at shoulder height, shoulders raised, head tilted. |
| `lean.png` | Leaning sideways with one arm resting on an invisible wall, legs crossed at the ankle, cool and relaxed. |
| `walk.png` | Side view (facing right), walking confidently mid-stride, one hand in pocket. |
| `facepalm.png` | One hand covering his face in a facepalm, other hand on his hip, shoulders slumped. |
| `explain.png` | Both hands open in front of his chest, gesturing as if explaining something, slight lean forward. |

Tips: if the face drifts, add "keep the exact same face, hairstyle and skin tone as the attached image".
Generate 2-3 and keep the best. The rig picks up each new drawing automatically (it finds the face itself).
