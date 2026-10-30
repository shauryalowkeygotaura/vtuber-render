"""Generate a drawn avatar of you from reference photos with Gemini image models.

    doppler run -p portfolio -c dev -- python scripts/gen_avatar.py out.png "prompt" ref1.jpg [ref2.jpg ...]

Needs GEMINI_API_KEY with image models enabled (billing on at aistudio.google.com).
"""
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

MODELS = ("gemini-2.5-flash-image", "gemini-3.1-flash-image", "gemini-3-pro-image")


def main() -> int:
    out, prompt, refs = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("GEMINI_API_KEY missing (run through doppler)", file=sys.stderr)
        return 2
    parts = [{"text": prompt}] + [
        {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(Path(r).read_bytes()).decode()}}
        for r in refs]
    body = json.dumps({"contents": [{"parts": parts}],
                       "generationConfig": {"responseModalities": ["IMAGE"],
                                            "imageConfig": {"aspectRatio": "9:16"}}}).encode()
    for m in MODELS:
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
            data=body, headers={"Content-Type": "application/json", "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            print(f"{m}: HTTP {e.code} {e.read()[:200]!r}", file=sys.stderr)
            continue
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            print(f"{m}: {e}", file=sys.stderr)
            continue
        for c in data.get("candidates", []):
            for p in c.get("content", {}).get("parts", []):
                if "inlineData" in p:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(base64.b64decode(p["inlineData"]["data"]))
                    print(f"{m} -> {out}")
                    return 0
        print(f"{m}: no image in response: {json.dumps(data)[:300]}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
