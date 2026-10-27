"""Standalone update helper (standard library only).

Started by the app right before it quits. It waits for the app to exit, backs up
the current app files, copies the new files in, runs pip if requirements.txt
changed, and relaunches. Any failure rolls the app files back. User data
(projects/, settings.json, the venv, downloaded weights) is never touched.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

PROTECTED = {
    "projects", "settings.json", ".venv", "venv", "env", "weights", ".ultralytics", ".cache",
    "update_backups", "update_status.json", "update.log", ".git",
}
KEEP_BACKUPS = 3
WORK_DIR_PREFIX = "LocalTrainer-update-"  # same as core.updater; this file cannot import the app

LogFn = Callable[[str], None]


def _retry(fn: Callable[[], object], attempts: int = 10, delay: float = 0.5) -> None:
    """File operations on Windows can fail briefly while antivirus or Explorer holds a handle."""
    for i in range(attempts):
        try:
            fn()
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)


def wait_for_pid(pid: int, timeout_s: float = 120) -> None:
    if pid <= 0:
        return
    if os.name == "nt":
        synchronize = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, pid)
        if handle:
            ctypes.windll.kernel32.WaitForSingleObject(handle, int(timeout_s * 1000))
            ctypes.windll.kernel32.CloseHandle(handle)
    else:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(0.25)
    time.sleep(1.0)  # let the OS release file handles


def requirements_hash(path: Path) -> str | None:
    """Hash of requirements.txt that ignores line endings, blank lines and comments."""
    if not path.is_file():
        return None
    lines = [ln.split("#", 1)[0].strip() for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    return hashlib.sha256("\n".join(ln for ln in lines if ln).encode()).hexdigest()


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        _retry(lambda: shutil.rmtree(path))
    elif path.exists():
        _retry(path.unlink)


def _copy(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(src, dst)


def update_items(source: Path) -> list[str]:
    """Top-level names in the new package that the updater may replace."""
    return sorted(p.name for p in Path(source).iterdir() if p.name not in PROTECTED)


def apply_update(source: Path, target: Path, backup_dir: Path, log: LogFn = print,
                 copy: Callable[[Path, Path], None] = _copy) -> tuple[list[str], list[str]]:
    """Move current items to backup_dir, copy new ones in. Rolls back and re-raises on failure.

    Returns (moved names, added names) for a later rollback.
    """
    source, target, backup_dir = Path(source), Path(target), Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=False)
    moved: list[str] = []
    added: list[str] = []
    try:
        for name in update_items(source):
            dst = target / name
            if dst.exists():
                _retry(lambda d=dst, n=name: shutil.move(str(d), str(backup_dir / n)))
                moved.append(name)
            else:
                added.append(name)
            copy(source / name, dst)
            log(f"Replaced {name}")
    except Exception as exc:
        log(f"Copy failed ({exc}). Rolling back.")
        rollback(target, backup_dir, moved, added, log)
        raise
    return moved, added


def rollback(target: Path, backup_dir: Path, moved: list[str], added: list[str], log: LogFn = print) -> None:
    for name in added + moved:
        p = Path(target) / name
        if p.exists():
            _remove(p)
    for name in moved:
        src = Path(backup_dir) / name
        if src.exists():
            _retry(lambda s=src, n=name: shutil.move(str(s), str(Path(target) / n)))
    log("Rollback complete. The previous version is restored.")


def find_venv_python(target: Path) -> Path | None:
    for venv in (".venv", "venv", "env"):
        p = Path(target) / venv / "Scripts" / "python.exe"
        if p.is_file():
            return p
        p = Path(target) / venv / "bin" / "python"
        if p.is_file():
            return p
    return None


def run_pip(python: Path, requirements: Path, log: LogFn) -> bool:
    log(f"requirements.txt changed. Running pip install with {python}")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0  # type: ignore[attr-defined]
    proc = subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements)],
                          capture_output=True, text=True, creationflags=flags)
    log(proc.stdout[-4000:])
    if proc.returncode != 0:
        log(proc.stderr[-4000:])
    return proc.returncode == 0


def write_status(target: Path, ok: bool, version: str, message: str) -> None:
    status = {"ok": ok, "version": version, "message": message, "time": datetime.now().isoformat(timespec="seconds")}
    (Path(target) / "update_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


def prune_backups(root: Path, keep: int = KEEP_BACKUPS) -> None:
    if not root.is_dir():
        return
    for old in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)[:-keep]:
        shutil.rmtree(old, ignore_errors=True)


def remove_work_dir(work_dir: Path, log: LogFn) -> None:
    """Delete the temp folder that held the download and this script (Python has already read it)."""
    work_dir = Path(work_dir)
    if not work_dir.name.startswith(WORK_DIR_PREFIX) or not work_dir.is_dir():
        return
    shutil.rmtree(work_dir, ignore_errors=True)
    log(f"Could not delete {work_dir}." if work_dir.exists() else f"Deleted {work_dir}.")


def relaunch(target: Path, python: Path | None) -> None:
    run_bat = Path(target) / "run.bat"
    if os.name == "nt" and run_bat.is_file():
        subprocess.Popen(["cmd.exe", "/c", str(run_bat)], cwd=str(target),
                         creationflags=subprocess.CREATE_NEW_CONSOLE)  # type: ignore[attr-defined]
    elif python is not None:
        subprocess.Popen([str(python), str(Path(target) / "main.py")], cwd=str(target))


def run(source: Path, target: Path, version: str, pid: int = 0, do_relaunch: bool = True,
        pip: Callable[[Path, Path, LogFn], bool] = run_pip, work_dir: Path | None = None) -> bool:
    """Full update. Returns True on success. Always leaves a status file for the app."""
    target = Path(target)
    log_file = target / "update.log"

    def log(msg: str) -> None:
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")

    log(f"Updating to {version} from {source}")
    wait_for_pid(pid)
    python = find_venv_python(target)
    old_req = requirements_hash(target / "requirements.txt")
    backup_root = target / "update_backups"
    backup_dir = backup_root / f"{datetime.now():%Y%m%d_%H%M%S}"
    ok, message = False, ""
    try:
        moved, added = apply_update(Path(source), target, backup_dir, log)
        new_req = requirements_hash(target / "requirements.txt")
        if new_req != old_req and new_req is not None:
            if python is None:
                log("No virtual environment found; skipping pip install.")
            elif not pip(python, target / "requirements.txt", log):
                rollback(target, backup_dir, moved, added, log)
                raise RuntimeError("Installing the new requirements failed (see update.log).")
        ok, message = True, f"Updated to version {version}."
        prune_backups(backup_root)
    except Exception as exc:
        message = f"The update failed and the previous version was restored: {exc}"
        log(message)
    write_status(target, ok, version, message)
    log(message)
    if work_dir is not None:
        remove_work_dir(work_dir, log)
    if do_relaunch:
        relaunch(target, python)
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LocalTrainer update helper")
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--target", required=True, type=Path)
    ap.add_argument("--version", required=True)
    ap.add_argument("--pid", type=int, default=0)
    ap.add_argument("--no-relaunch", action="store_true")
    ap.add_argument("--work-dir", type=Path, help="temp folder to delete when done")
    args = ap.parse_args(argv)
    ok = run(args.source, args.target, args.version, args.pid, not args.no_relaunch, work_dir=args.work_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
