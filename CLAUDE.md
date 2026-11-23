# vtuber-render: project rules

## Rendering: code or HyperFrames only
- **Never use Higgsfield**, or any paid image/video model (fal.ai, Kling, Seedance, Runway, Gemini/OpenAI image, Nano Banana, MaxFusion, Atlas Cloud). Every frame is drawn in code: HyperFrames, Remotion, SVG/canvas, Paper.js, PixiJS, Three.js.
- If an installed skill offers an "AI-video escape hatch" or image-model branch (e.g. pasteup section 6, hand-drawn-explainer-video-nikola's Flow/Nano Banana prompt mode), skip it and take the code path.
- Voice: our own stack (Groq + Orpheus via paralang, or Shaurya's recording). Ignore skills' ElevenLabs/OpenAI TTS options.

## Drawn b-roll on CI (`vtuber/styled.py`)
The unattended path. Actions -> render -> `broll`: `mix` (default: Groq reads every line and picks stock vs drawn per shot, no quota; footage for story moments, drawn cards for numbers/verdicts/punchlines/CTAs; any shot whose Pexels clip fails the judge is drawn too; alternation only if Groq is down), `styled` (all drawn), `stock` (old behaviour). `style`: `papercut`, `sketch`, `auto` (one look per reel). Locally: `python reel.py --script x.txt --broll mix --style sketch`. Saved in the edit plan as `broll_mode` / `broll_style`.
Groq only picks a 1-3 word headline + an icon name from `styled.ICONS`; all drawing is Python. New icon = add primitives to `ICONS` + a `KEYWORDS` row. New look = a `_render_<style>` + add to `STYLES`, bump `VERSION`.

## Style skills (`.claude/skills/`, installed by `python scripts/install_skills.py`)
Pinned in `skills.lock.json`; vendored code lives in `vendor/skill-repos/`. Both are gitignored (third-party). Each skill installs its own node deps on first use; read its setup step before running.

| Want | Skill |
|---|---|
| Paper cut-out reel, 9:16 | `paper-reel`, `papercut-explainer` (full 9-skill pipeline) |
| Paper collage / scrapbook (HyperFrames) | `hyperframes-tactile-collage`, `paper-collage`, `vox-collage`, `gbro-collage-info`, `pasteup` |
| Soft hand-drawn / sketch | `hand-drawn-cartoon`, `claude-animation`, `javascript-animation`, `line-art-animation`, `explainer-studio`, `whiteboard-video` |
| Many styles in one | `anidoodle`, `kalem`, `papertrail-make` |
| Explainer object / diagram | `anatomy`, `archscribe`, `lanshu-animated-architecture-diagram`, `system-diagram-animation` |
| Named looks | `kurzgesagt`, `stick-figure`, `low-poly`, `minecraft`, `dan-koe`, `zach-d-films`, `map-route` |
| Type / data | `kinetic-typography`, `data-story-charts`, `anim-graph` |
