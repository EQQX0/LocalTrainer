"""Build LocalTrainer-Setup-<version>.exe around the release zip, plus its SHA-256 file. Windows only.

Used by .github/workflows/release.yml after tools/build_release.py, and runnable locally:
    python tools/build_installer.py --zip dist/LocalTrainer-1.2.3.zip --out dist

It compiles tools/installer/Setup.cs with the csc.exe of .NET Framework 4, which every Windows 10/11 has.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tools" / "installer" / "Setup.cs"
ICON = ROOT / "ui" / "assets" / "icon.ico"
PAYLOAD = "payload.zip"  # resource name Setup.cs reads


def find_csc() -> Path:
    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    for arch in ("Framework64", "Framework"):
        csc = windir / "Microsoft.NET" / arch / "v4.0.30319" / "csc.exe"
        if csc.is_file():
            return csc
    sys.exit("csc.exe from .NET Framework 4 was not found. Build the installer on Windows.")


def build(zip_path: Path, out_dir: Path) -> Path:
    match = re.fullmatch(r"LocalTrainer-(\d+\.\d+\.\d+)\.zip", zip_path.name)
    if not match:
        sys.exit(f"{zip_path.name} is not named LocalTrainer-X.Y.Z.zip")
    version = match[1]
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = out_dir / f"LocalTrainer-Setup-{version}.exe"
    with tempfile.TemporaryDirectory() as tmp:
        info = Path(tmp) / "AssemblyInfo.cs"
        info.write_text(
            "using System.Reflection;\n"
            '[assembly: AssemblyTitle("LocalTrainer Setup")]\n'
            '[assembly: AssemblyProduct("LocalTrainer")]\n'
            '[assembly: AssemblyDescription("Installs LocalTrainer and starts it")]\n'
            f'[assembly: AssemblyVersion("{version}.0")]\n'
            f'[assembly: AssemblyFileVersion("{version}.0")]\n', encoding="utf-8")
        cmd = [str(find_csc()), "/nologo", "/optimize+", "/target:exe", "/platform:anycpu",
               "/r:System.IO.Compression.dll", "/r:System.IO.Compression.FileSystem.dll",
               f"/win32icon:{ICON}", f"/resource:{zip_path},{PAYLOAD}", f"/out:{exe}", str(SOURCE), str(info)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            sys.exit(f"csc failed:\n{res.stdout}{res.stderr}")
    digest = hashlib.sha256(exe.read_bytes()).hexdigest()
    (out_dir / f"{exe.name}.sha256").write_text(f"{digest}  {exe.name}\n", encoding="utf-8")
    print(f"{exe} ({exe.stat().st_size / 1000:.0f} kB)\nsha256 {digest}")
    return exe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    args = ap.parse_args()
    build(args.zip.resolve(), args.out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
