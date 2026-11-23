"""Install the third-party video-style skills pinned in skills.lock.json.

    python scripts/install_skills.py                 # fetch pinned commits from GitHub
    python scripts/install_skills.py --from DIR      # reuse clones named owner_repo in DIR

Layout it produces (both gitignored, this repo is public):
    .claude/skills/<name>/          one folder per skill, auto-discovered by Claude Code here
    vendor/skill-repos/<owner_repo> full repo, for skills that import code from their repo root

Only skills it installed are removed on re-run (tracked in vendor/skill-repos/.installed.json),
so hand-written project skills in .claude/skills/ are never touched.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "skills.lock.json"
SKILLS = ROOT / ".claude" / "skills"
VENDOR = ROOT / "vendor" / "skill-repos"
STATE = VENDOR / ".installed.json"

SKIP_DIRS = {".git", ".github", "node_modules", "__pycache__", ".venv"}
SKIP_EXT = {".mp4", ".mov", ".webm", ".gif", ".mkv", ".avi"}  # demo renders, not needed to run
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")  # whitelist, never shell-interpolated
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")  # skill names come from untrusted SKILL.md


def safe_dest(name: str) -> Path:
    """Whitelist a skill name before it becomes a path (blocks ../ and absolute paths)."""
    if not NAME_RE.match(name):
        raise ValueError(f"unsafe skill name: {name!r}")
    return SKILLS / name


def _ignore(_dir: str, names: list[str]) -> list[str]:
    return [n for n in names if n in SKIP_DIRS or Path(n).suffix.lower() in SKIP_EXT]


def fetch(repo: str, sha: str, dest: Path) -> None:
    dest.mkdir(parents=True)
    git = ["git", "-C", str(dest)]
    for cmd in (git + ["init", "-q"],
                git + ["fetch", "-q", "--depth", "1", f"https://github.com/{repo}", sha],
                git + ["checkout", "-q", "FETCH_HEAD"]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"{repo}: {' '.join(cmd[3:5])} failed: {r.stderr.strip()}")


def skill_name(md: Path) -> str:
    m = re.search(r"^name:\s*\"?([^\"\n]+?)\"?\s*$", md.read_text(encoding="utf-8"), re.M)
    if not m:
        raise ValueError(f"{md}: no name: in frontmatter")
    return m.group(1).strip()


def patch(md: Path, new_name: str | None, vendor_note: str | None) -> None:
    text = md.read_text(encoding="utf-8")
    if new_name:
        text = re.sub(r"^name:.*$", f"name: {new_name}", text, count=1, flags=re.M)
    if vendor_note:
        # after the closing frontmatter fence, so the description still parses
        parts = text.split("---", 2)
        if len(parts) == 3:
            text = f"---{parts[1]}---\n\n> {vendor_note}\n{parts[2]}"
    md.write_text(text, encoding="utf-8")


def install_repo(entry: dict, src: Path, rename: dict) -> list[str]:
    repo = entry["repo"]
    slug = repo.replace("/", "_")
    found = sorted((p for p in src.rglob("SKILL.md") if not SKIP_DIRS & set(p.parts)),
                   key=lambda p: len(p.parts))
    installed: list[str] = []

    if (src / "SKILL.md").exists():  # the whole repo is one skill
        name = skill_name(src / "SKILL.md")
        final = rename.get(name, name)
        dest = safe_dest(final)
        if dest.exists():
            raise RuntimeError(f"skill name collision: {final} (from {repo})")
        shutil.copytree(src, dest, ignore=_ignore)
        patch(dest / "SKILL.md", final if final != name else None, None)
        return [final]

    vendored = VENDOR / slug
    shutil.copytree(src, vendored, ignore=_ignore)
    seen: set[str] = set()
    for md in found:
        name = skill_name(md)
        if name in seen:  # e.g. anidoodle ships the same skill twice (plugin + skills/)
            continue
        seen.add(name)
        final = rename.get(name, name)
        dest = safe_dest(final)
        if dest.exists():
            raise RuntimeError(f"skill name collision: {final} (from {repo})")
        shutil.copytree(md.parent, dest, ignore=_ignore)
        rel = md.parent.relative_to(src).as_posix()
        note = (f"Installed from {repo}@{entry['sha'][:7]} ({rel}). Paths in this skill that point at "
                f"the source repo resolve under `{vendored.as_posix()}/`; run repo scripts from there.")
        patch(dest / "SKILL.md", final if final != name else None, note)
        installed.append(final)
    return installed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_dir", type=Path, help="dir of existing owner_repo clones")
    args = ap.parse_args()

    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for e in lock["repos"]:
        if not (REPO_RE.match(e["repo"]) and SHA_RE.match(e["sha"])):
            print(f"refusing bad lock entry: {e}", file=sys.stderr)
            return 1

    # uninstall only what we installed last time
    if STATE.exists():
        for name in json.loads(STATE.read_text(encoding="utf-8")):
            if NAME_RE.match(name):
                shutil.rmtree(safe_dest(name), ignore_errors=True)
    if VENDOR.exists():
        shutil.rmtree(VENDOR)
    SKILLS.mkdir(parents=True, exist_ok=True)
    VENDOR.mkdir(parents=True)

    all_names: list[str] = []
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for e in lock["repos"]:
            slug = e["repo"].replace("/", "_")
            try:
                if args.from_dir:
                    src = args.from_dir / slug
                    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"],
                                          capture_output=True, text=True).stdout.strip()
                    if head != e["sha"]:
                        raise RuntimeError(f"clone is at {head[:7] or '?'}, lock wants {e['sha'][:7]}")
                else:
                    src = Path(tmp) / slug
                    fetch(e["repo"], e["sha"], src)
                names = install_repo(e, src, lock.get("rename", {}))
                all_names += names
                print(f"ok  {e['repo']:<48} -> {', '.join(names)}")
            except Exception as ex:  # keep going, report at the end
                failures.append(f"{e['repo']}: {ex}")
                print(f"ERR {e['repo']}: {ex}", file=sys.stderr)

    STATE.write_text(json.dumps(sorted(all_names), indent=1), encoding="utf-8")
    print(f"\n{len(all_names)} skills installed into {SKILLS}")
    if failures:
        print(f"{len(failures)} repo(s) failed:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
