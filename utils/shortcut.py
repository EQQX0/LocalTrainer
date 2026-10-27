"""Desktop shortcut that starts LocalTrainer without a console window (Windows only).

setup.bat runs ``python -m utils.shortcut --replace``; the app creates the shortcut once on its first start.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from utils.paths import APP_ROOT, ASSETS_DIR

SHORTCUT_NAME = "LocalTrainer.lnk"
ICON_FILE = ASSETS_DIR / "icon.ico"

# PowerShell asks Windows for the desktop folder (it may live in OneDrive) and writes the .lnk through
# WScript.Shell. Paths travel in environment variables, so quotes or apostrophes in them need no escaping.
_PS_SCRIPT = (
    "$ErrorActionPreference = 'Stop'; "
    "$p = Join-Path ([Environment]::GetFolderPath('Desktop')) $env:LT_LNK_NAME; "
    "if ((Test-Path -LiteralPath $p) -and $env:LT_LNK_REPLACE -ne '1') { 'exists ' + $p; exit 0 }; "
    "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($p); "
    "$s.TargetPath = $env:LT_LNK_TARGET; $s.Arguments = $env:LT_LNK_ARGS; "
    "$s.WorkingDirectory = $env:LT_LNK_DIR; $s.IconLocation = $env:LT_LNK_ICON; "
    "$s.Description = 'Train YOLO models on your GPU'; $s.Save(); 'created ' + $p"
)


def shortcut_fields(app_root: Path = APP_ROOT, python: str = sys.executable) -> dict[str, str]:
    """Environment for the PowerShell script: pythonw.exe from the running venv, main.py, the icon."""
    exe = Path(python)
    pythonw = exe.with_name("pythonw.exe")
    target = pythonw if pythonw.is_file() else exe
    return {"LT_LNK_NAME": SHORTCUT_NAME, "LT_LNK_TARGET": str(target),
            "LT_LNK_ARGS": f'"{Path(app_root) / "main.py"}"', "LT_LNK_DIR": str(app_root),
            "LT_LNK_ICON": f"{ICON_FILE},0" if ICON_FILE.is_file() else f"{target},0"}


def create_desktop_shortcut(replace: bool = False, app_root: Path = APP_ROOT,
                            python: str = sys.executable, timeout: float = 30) -> tuple[str, str]:
    """Create LocalTrainer.lnk on the desktop. Returns (outcome, detail).

    outcome is "created", "exists" (an existing shortcut was left alone), "unsupported" or "failed".
    It never raises: a missing shortcut must not stop the app or setup.
    """
    if os.name != "nt":
        return "unsupported", "desktop shortcuts are made on Windows only"
    env = {**os.environ, **shortcut_fields(app_root, python), "LT_LNK_REPLACE": "1" if replace else "0"}
    try:
        res = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                              "-Command", _PS_SCRIPT], env=env, capture_output=True, text=True,
                             timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError) as e:
        return "failed", str(e)
    out = res.stdout.strip()
    if res.returncode == 0 and out.startswith(("created ", "exists ")):
        outcome, _, path = out.partition(" ")
        return outcome, path
    return "failed", (res.stderr or out).strip()[-500:]


def main() -> int:
    outcome, detail = create_desktop_shortcut(replace="--replace" in sys.argv[1:])
    if outcome == "created":
        print(f"Created the desktop shortcut {detail}")
    elif outcome == "exists":
        print(f"Kept the existing desktop shortcut {detail}")
    else:
        print(f"Could not create a desktop shortcut: {detail}")
    return 0  # a missing shortcut is not a setup failure


if __name__ == "__main__":
    sys.exit(main())
