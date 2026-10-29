# vtuber-render

Voice recording in, talking 3D avatar video out. No camera, no face tracking:
you record audio, the pipeline does all the acting.

```bash
doppler run -p philosopher-pipeline -c dev -- python render.py take.m4a            # 1080x1920 mp4
doppler run -p philosopher-pipeline -c dev -- python render.py take.m4a --alpha    # transparent webm for b-roll overlay
python render.py take.m4a --no-llm                                                 # offline: lip sync + idle only
python render.py take.m4a --avatar me.vrm                                          # your VRoid export
python render.py take.m4a --still 2.5                                              # one PNG to check framing
```

## How it acts
| Layer | File | Source |
|---|---|---|
| Mouth (aa/ih/ou/ee/oh) | `vtuber/lipsync.py` | loudness + spectral balance of the audio. Language-free, so Hinglish works |
| Expression + gesture per sentence | `vtuber/performance.py` | Groq Whisper segments, split into sentences, tagged by a Groq model |
| Head sway, beat nods, blinks, breathing | `vtuber/motion.py` | seeded procedural motion, deterministic |
| Rendering | `web/index.html` | three.js 0.184 + @pixiv/three-vrm 3.5.5, one frame per call, spring-bone hair physics |

Renders at ~4-9 fps (GPU via ANGLE when available): a 60 s reel takes 2-4 minutes.
Every run writes `output/<name>.beats.json`, so you can see what the director chose.

## Avatar
`assets/sample.vrm` is pixiv's VRM1 sample (commercial use + redistribution allowed,
no credit needed, per its embedded VRM meta). It is a placeholder. To make it *you*:
build a character in **VRoid Studio** (free), export VRM 1.0, pass `--avatar`.

## Notes
- Rhubarb Lip Sync was considered and dropped: English-only recogniser, and running
  its downloaded binary was blocked here.
- `vtuber/groq_pool.py` is vendored from `Vault/Scripts/groq_pool.py`. Do not edit here.
