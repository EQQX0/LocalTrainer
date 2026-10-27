"""In-app updater: check GitHub releases, download, verify SHA-256, extract, hand off to the helper.

Network calls go through an ``opener`` argument so tests can pass a fake one.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from app.version import APP_NAME, GITHUB_REPO, __version__

API_LATEST = "https://api.github.com/repos/{repo}/releases/latest"
USER_AGENT = f"{APP_NAME}-Updater/{__version__}"
TIMEOUT_S = 15
CHUNK = 256 * 1024
WORK_DIR_PREFIX = f"{APP_NAME}-update-"
STALE_WORK_DIR_S = 3600  # startup cleanup skips staging folders younger than this

Opener = Callable[..., Any]          # same signature as urllib.request.urlopen
ProgressFn = Callable[[int, int], None]  # (bytes done, bytes total or 0)


class UpdateError(Exception):
    """Base class. str(exc) is a friendly message for the user."""


class NoInternetError(UpdateError):
    pass


class RateLimitError(UpdateError):
    pass


class NoReleaseError(UpdateError):
    pass


class MissingAssetError(UpdateError):
    pass


class ChecksumError(UpdateError):
    pass


class CancelledError(UpdateError):
    pass


# ----------------------------------------------------------------- versions

_SEMVER = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$")


def parse_version(text: str) -> tuple[Any, ...]:
    """Sortable key for a semantic version ('v1.2.3', '1.2.3-beta.1'). Raises ValueError."""
    m = _SEMVER.match(text.strip())
    if not m:
        raise ValueError(f"Not a semantic version: {text!r}")
    major, minor, patch, pre = int(m[1]), int(m[2]), int(m[3]), m[4]
    # a release sorts after its pre-releases; numeric identifiers sort before alphanumeric ones
    pre_key = () if pre is None else tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in pre.split("."))
    return (major, minor, patch, pre is None, pre_key)


def is_newer(remote: str, local: str = __version__) -> bool:
    try:
        return parse_version(remote) > parse_version(local)
    except ValueError:
        return False


# ------------------------------------------------------------------ release

@dataclass
class ReleaseInfo:
    version: str
    tag: str
    notes: str
    html_url: str
    zip_name: str
    zip_url: str
    zip_size: int
    sha_url: str


def _request(url: str, accept: str = "application/vnd.github+json") -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})


def _friendly_http_error(exc: urllib.error.HTTPError) -> UpdateError:
    headers = exc.headers or {}
    remaining = headers.get("X-RateLimit-Remaining")
    if exc.code == 429 or (exc.code == 403 and remaining == "0"):
        reset = headers.get("X-RateLimit-Reset")
        when = ""
        if reset and str(reset).isdigit():
            when = f" Try again after {datetime.fromtimestamp(int(reset)):%H:%M}."
        return RateLimitError("GitHub's limit of 60 update checks per hour was reached." + (when or " Try again later."))
    if exc.code == 404:
        return NoReleaseError("No published release was found on GitHub yet.")
    return UpdateError(f"GitHub answered with an error (HTTP {exc.code}). Try again later.")


def _open(url: str, opener: Opener, accept: str = "application/vnd.github+json") -> Any:
    try:
        return opener(_request(url, accept), timeout=TIMEOUT_S)
    except urllib.error.HTTPError as exc:
        raise _friendly_http_error(exc) from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
        raise NoInternetError("Could not reach GitHub. Check your internet connection and try again.") from exc


def fetch_latest_release(repo: str = GITHUB_REPO, opener: Opener = urllib.request.urlopen) -> ReleaseInfo:
    with _open(API_LATEST.format(repo=repo), opener) as resp:
        try:
            data = json.loads(resp.read().decode("utf-8"))
        except ValueError as exc:
            raise UpdateError("GitHub sent a response that could not be read.") from exc
    tag = str(data.get("tag_name", ""))
    assets = data.get("assets") or []
    zip_asset = next((a for a in assets if str(a.get("name", "")).lower().endswith(".zip")
                      and str(a.get("name", "")).startswith(APP_NAME)), None)
    if zip_asset is None:
        raise MissingAssetError(f"Release {tag} has no {APP_NAME} zip file attached yet. Try again in a few minutes.")
    sha_asset = next((a for a in assets if a.get("name") == f"{zip_asset['name']}.sha256"), None) or \
        next((a for a in assets if str(a.get("name", "")).lower().endswith(".sha256")), None)
    if sha_asset is None:
        raise MissingAssetError(f"Release {tag} has no checksum file, so the download cannot be verified.")
    return ReleaseInfo(
        version=tag.lstrip("v"), tag=tag, notes=str(data.get("body") or "").strip(),
        html_url=str(data.get("html_url", "")), zip_name=str(zip_asset["name"]),
        zip_url=str(zip_asset["browser_download_url"]), zip_size=int(zip_asset.get("size") or 0),
        sha_url=str(sha_asset["browser_download_url"]),
    )


# ----------------------------------------------------------------- download

def download(url: str, dest: Path, progress: ProgressFn | None = None, opener: Opener = urllib.request.urlopen,
             cancel: threading.Event | None = None) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with _open(url, opener, accept="application/octet-stream") as resp:
        total = int(resp.headers.get("Content-Length") or 0) if getattr(resp, "headers", None) else 0
        done = 0
        try:
            with open(part, "wb") as fh:
                while True:
                    if cancel is not None and cancel.is_set():
                        raise CancelledError("The download was cancelled.")
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
        except (socket.timeout, TimeoutError, ConnectionError) as exc:
            part.unlink(missing_ok=True)
            raise NoInternetError("The download was interrupted. Check your connection and try again.") from exc
        except UpdateError:
            part.unlink(missing_ok=True)
            raise
    if total and done != total:
        part.unlink(missing_ok=True)
        raise NoInternetError("The download ended early. Try again.")
    part.replace(dest)
    return dest


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_checksum(text: str, filename: str) -> str:
    """Accept 'HASH  name', 'HASH *name' or a bare hash."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for line in lines:
        parts = line.split()
        if len(parts) >= 2 and parts[-1].lstrip("*") == filename and re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            return parts[0].lower()
    if len(lines) == 1 and re.fullmatch(r"[0-9a-fA-F]{64}", lines[0].split()[0]):
        return lines[0].split()[0].lower()
    raise ChecksumError("The checksum file is not in the expected format.")


def verify_checksum(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ChecksumError("The downloaded file is damaged (SHA-256 checksum mismatch).")


def safe_extract(zip_path: Path, dest: Path) -> Path:
    """Extract without allowing paths outside dest. Returns the folder that holds main.py."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    base = dest.resolve()
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.infolist():
                target = (base / member.filename).resolve()
                if target != base and base not in target.parents:
                    raise UpdateError("The update package contains unsafe paths and was rejected.")
            zf.extractall(base)
    except zipfile.BadZipFile as exc:
        raise UpdateError("The downloaded file is not a valid zip archive.") from exc
    candidates = [base] + sorted(p for p in base.iterdir() if p.is_dir())
    for c in candidates:
        if (c / "main.py").is_file() and (c / "app" / "version.py").is_file():
            return c
    raise UpdateError(f"The package does not look like {APP_NAME} (main.py is missing).")


def read_package_version(root: Path) -> str | None:
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', (Path(root) / "app" / "version.py").read_text("utf-8"))
    return m[1] if m else None


def stage_update(info: ReleaseInfo, work_dir: Path, progress: ProgressFn | None = None,
                 opener: Opener = urllib.request.urlopen, cancel: threading.Event | None = None) -> Path:
    """Download + verify + extract into work_dir. Returns the extracted app root."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    with _open(info.sha_url, opener, accept="application/octet-stream") as resp:
        expected = parse_checksum(resp.read().decode("utf-8", "replace"), info.zip_name)
    zip_path = download(info.zip_url, work_dir / info.zip_name, progress, opener, cancel)
    verify_checksum(zip_path, expected)
    root = safe_extract(zip_path, work_dir / "extracted")
    pkg_version = read_package_version(root)
    if pkg_version and info.version and pkg_version != info.version:
        raise UpdateError(f"The package says version {pkg_version} but the release is {info.version}.")
    return root


def default_work_dir() -> Path:
    import tempfile
    return Path(tempfile.gettempdir()) / f"{WORK_DIR_PREFIX}{datetime.now():%Y%m%d_%H%M%S}"


def remove_work_dir(path: Path) -> bool:
    """Delete one staging folder. Refuses any folder whose name lacks the update prefix."""
    path = Path(path)
    if not path.name.startswith(WORK_DIR_PREFIX) or path.is_symlink() or not path.is_dir():
        return False
    shutil.rmtree(path, ignore_errors=True)
    return not path.exists()


def cleanup_work_dirs(root: Path | None = None, min_age_s: float = STALE_WORK_DIR_S,
                      now: float | None = None) -> list[Path]:
    """Delete leftover staging folders in the temp folder and return the ones removed.

    Folders changed within ``min_age_s`` stay: another LocalTrainer window may be staging an update,
    or the helper may still be applying one from there.
    """
    import tempfile
    import time
    root = Path(root) if root is not None else Path(tempfile.gettempdir())
    now = time.time() if now is None else now
    removed: list[Path] = []
    try:
        candidates = sorted(root.glob(WORK_DIR_PREFIX + "*"))
    except OSError:
        return removed
    for path in candidates:
        try:
            if now - path.stat().st_mtime < min_age_s:
                continue
        except OSError:
            continue
        if remove_work_dir(path):
            removed.append(path)
    return removed


def launch_helper(new_root: Path, app_root: Path, version: str, work_dir: Path,
                  python: str = sys.executable) -> subprocess.Popen:
    """Copy the stdlib-only helper next to the staged files and start it detached from this process.

    The helper runs in the app folder, not the work dir: a venv's python.exe is a launcher that keeps
    its starting folder open until the helper exits, so the helper could not delete the work dir.
    """
    helper_src = Path(app_root) / "utils" / "update_helper.py"
    helper = Path(work_dir) / "update_helper.py"
    shutil.copy2(helper_src, helper)
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    return subprocess.Popen(
        [python, "-I", str(helper), "--source", str(new_root), "--target", str(app_root),
         "--pid", str(os.getpid()), "--version", version, "--work-dir", str(work_dir)],
        cwd=str(app_root), creationflags=flags, close_fds=True,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
