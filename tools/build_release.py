"""Build LocalTrainer-<version>.zip (app code only) and its SHA-256 file.

Used by .github/workflows/release.yml and runnable locally:
    python tools/build_release.py --out dist [--tag v1.2.3]
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# tracked files that are not part of the app package
EXCLUDE = [".github/*", "tests/*", "tools/*", "docs/*", ".gitignore", ".gitattributes",
           "requirements-dev.txt", "DECISIONS.md", "PUBLISH.md"]


def read_version() -> str:
    text = (ROOT / "app" / "version.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    if not match:
        sys.exit("Could not read __version__ from app/version.py")
    return match[1]


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    files = [f for f in out.decode("utf-8").split("\0") if f]
    return sorted(f for f in files if not any(fnmatch.fnmatch(f, pat) for pat in EXCLUDE))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    ap.add_argument("--tag", help="Fail unless the tag matches app/version.py (vX.Y.Z)")
    args = ap.parse_args()
    version = read_version()
    if args.tag and args.tag.lstrip("v") != version:
        sys.exit(f"Tag {args.tag} does not match app/version.py ({version}). Bump the version first.")
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"LocalTrainer-{version}.zip"
    zip_path = args.out / name
    prefix = f"LocalTrainer-{version}/"
    files = tracked_files()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in files:
            zf.write(ROOT / rel, prefix + rel)
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    (args.out / f"{name}.sha256").write_text(f"{digest}  {name}\n", encoding="utf-8")
    print(f"{zip_path} ({len(files)} files, {zip_path.stat().st_size / 1000:.0f} kB)\nsha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
