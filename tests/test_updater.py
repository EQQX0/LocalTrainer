"""Updater: version comparison, GitHub API (mocked), download, checksum, replace and rollback."""
from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import zipfile
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from core import updater as up
from utils import update_helper as helper

REPO_API = up.API_LATEST.format(repo="me/LocalTrainer")
ZIP_URL = "https://github.com/me/LocalTrainer/releases/download/v1.0.1/LocalTrainer-1.0.1.zip"
SHA_URL = ZIP_URL + ".sha256"


# ------------------------------------------------------------ fakes

class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *a: Any) -> None:
        self.close()


def http_error(url: str, code: int, headers: dict[str, str] | None = None) -> urllib.error.HTTPError:
    msg = Message()
    for k, v in (headers or {}).items():
        msg[k] = v
    return urllib.error.HTTPError(url, code, "error", msg, None)


class FakeGitHub:
    """Callable with urlopen's signature that serves canned URLs."""

    def __init__(self, routes: dict[str, Any]) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def __call__(self, req: Any, timeout: float = 0) -> FakeResponse:
        url = req.full_url if hasattr(req, "full_url") else str(req)
        self.calls.append(url)
        assert req.get_header("User-agent", "").startswith("LocalTrainer")
        route = self.routes[url]
        if isinstance(route, Exception):
            raise route
        return FakeResponse(route)


def build_package(tmp_path: Path, version: str = "1.0.1", requirements: str = "PyQt6>=6.7\n") -> bytes:
    """A release zip shaped like the GitHub Actions build (top folder LocalTrainer-x.y.z/)."""
    buf = io.BytesIO()
    top = f"LocalTrainer-{version}/"
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(top + "main.py", "print('new')\n")
        zf.writestr(top + "app/version.py", f'__version__ = "{version}"\n')
        zf.writestr(top + "core/new_module.py", "X = 1\n")
        zf.writestr(top + "requirements.txt", requirements)
        zf.writestr(top + "run.bat", "@echo off\n")
    return buf.getvalue()


def release_json(assets: list[str]) -> bytes:
    return json.dumps({
        "tag_name": "v1.0.1", "body": "## Changes\n- faster", "html_url": "https://github.com/me/LocalTrainer/releases/v1.0.1",
        "assets": [{"name": n, "size": 123, "browser_download_url":
                    f"https://github.com/me/LocalTrainer/releases/download/v1.0.1/{n}"} for n in assets],
    }).encode()


# ------------------------------------------------------------ versions

@pytest.mark.parametrize("remote,local,newer", [
    ("1.0.1", "1.0.0", True), ("v1.1.0", "1.0.9", True), ("2.0.0", "1.99.99", True),
    ("1.0.0", "1.0.0", False), ("0.9.9", "1.0.0", False), ("1.0.10", "1.0.9", True),
    ("1.0.0", "1.0.0-beta.2", True), ("1.0.0-beta.2", "1.0.0-beta.1", True), ("1.0.0-rc.1", "1.0.0", False),
    ("1.0.0-alpha", "1.0.0-alpha.1", False), ("not-a-version", "1.0.0", False),
])
def test_is_newer(remote: str, local: str, newer: bool) -> None:
    assert up.is_newer(remote, local) is newer


def test_parse_version_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        up.parse_version("1.0")


# ------------------------------------------------------------ release lookup

def test_fetch_latest_release() -> None:
    gh = FakeGitHub({REPO_API: release_json(["LocalTrainer-1.0.1.zip", "LocalTrainer-1.0.1.zip.sha256"])})
    info = up.fetch_latest_release("me/LocalTrainer", opener=gh)
    assert info.version == "1.0.1" and info.tag == "v1.0.1"
    assert info.zip_url == ZIP_URL and info.sha_url == SHA_URL
    assert "faster" in info.notes


def test_missing_zip_asset() -> None:
    gh = FakeGitHub({REPO_API: release_json(["notes.txt"])})
    with pytest.raises(up.MissingAssetError, match="no LocalTrainer zip"):
        up.fetch_latest_release("me/LocalTrainer", opener=gh)


def test_missing_checksum_asset() -> None:
    gh = FakeGitHub({REPO_API: release_json(["LocalTrainer-1.0.1.zip"])})
    with pytest.raises(up.MissingAssetError, match="checksum"):
        up.fetch_latest_release("me/LocalTrainer", opener=gh)


def test_rate_limited() -> None:
    err = http_error(REPO_API, 403, {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1893456000"})
    with pytest.raises(up.RateLimitError, match="60 update checks per hour"):
        up.fetch_latest_release("me/LocalTrainer", opener=FakeGitHub({REPO_API: err}))


def test_no_release_yet() -> None:
    with pytest.raises(up.NoReleaseError):
        up.fetch_latest_release("me/LocalTrainer", opener=FakeGitHub({REPO_API: http_error(REPO_API, 404)}))


def test_no_internet() -> None:
    gh = FakeGitHub({REPO_API: urllib.error.URLError("getaddrinfo failed")})
    with pytest.raises(up.NoInternetError, match="internet connection"):
        up.fetch_latest_release("me/LocalTrainer", opener=gh)


# ------------------------------------------------------------ download + checksum

def test_download_reports_progress(tmp_path: Path) -> None:
    data = b"x" * (up.CHUNK * 2 + 10)
    seen: list[tuple[int, int]] = []
    out = up.download(ZIP_URL, tmp_path / "a.zip", lambda d, t: seen.append((d, t)), opener=FakeGitHub({ZIP_URL: data}))
    assert out.read_bytes() == data
    assert seen[-1] == (len(data), len(data))
    assert not (tmp_path / "a.zip.part").exists()


def test_download_truncated(tmp_path: Path) -> None:
    class Short(FakeResponse):
        def __init__(self) -> None:
            super().__init__(b"abc")
            self.headers = {"Content-Length": "999"}

    with pytest.raises(up.NoInternetError, match="ended early"):
        up.download(ZIP_URL, tmp_path / "a.zip", opener=lambda req, timeout=0: Short())
    assert not (tmp_path / "a.zip").exists()


@pytest.mark.parametrize("text", ["{h}  LocalTrainer-1.0.1.zip", "{h} *LocalTrainer-1.0.1.zip", "{h}"])
def test_parse_checksum_formats(text: str) -> None:
    h = "a" * 64
    assert up.parse_checksum(text.format(h=h), "LocalTrainer-1.0.1.zip") == h


def test_verify_checksum(tmp_path: Path) -> None:
    f = tmp_path / "f.zip"
    f.write_bytes(b"payload")
    up.verify_checksum(f, hashlib.sha256(b"payload").hexdigest())
    with pytest.raises(up.ChecksumError):
        up.verify_checksum(f, "0" * 64)


def test_safe_extract_blocks_zip_slip(tmp_path: Path) -> None:
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../outside.txt", "nope")
    with pytest.raises(up.UpdateError, match="unsafe"):
        up.safe_extract(z, tmp_path / "x")
    assert not (tmp_path / "outside.txt").exists()


def test_stage_update_end_to_end(tmp_path: Path) -> None:
    pkg = build_package(tmp_path)
    sha = hashlib.sha256(pkg).hexdigest()
    gh = FakeGitHub({REPO_API: release_json(["LocalTrainer-1.0.1.zip", "LocalTrainer-1.0.1.zip.sha256"]),
                     ZIP_URL: pkg, SHA_URL: f"{sha}  LocalTrainer-1.0.1.zip\n".encode()})
    info = up.fetch_latest_release("me/LocalTrainer", opener=gh)
    root = up.stage_update(info, tmp_path / "work", opener=gh)
    assert (root / "main.py").read_text() == "print('new')\n"
    assert up.read_package_version(root) == "1.0.1"


def test_stage_update_rejects_bad_checksum(tmp_path: Path) -> None:
    pkg = build_package(tmp_path)
    gh = FakeGitHub({REPO_API: release_json(["LocalTrainer-1.0.1.zip", "LocalTrainer-1.0.1.zip.sha256"]),
                     ZIP_URL: pkg, SHA_URL: ("f" * 64 + "  LocalTrainer-1.0.1.zip").encode()})
    info = up.fetch_latest_release("me/LocalTrainer", opener=gh)
    with pytest.raises(up.ChecksumError):
        up.stage_update(info, tmp_path / "work", opener=gh)
    assert not (tmp_path / "work" / "extracted").exists()


# ------------------------------------------------------------ replace + rollback (helper)

def make_installed_app(root: Path, requirements: str = "PyQt6>=6.7\n") -> None:
    (root / "app").mkdir(parents=True)
    (root / "app" / "version.py").write_text('__version__ = "1.0.0"\n')
    (root / "core").mkdir()
    (root / "core" / "old_module.py").write_text("OLD = 1\n")
    (root / "main.py").write_text("print('old')\n")
    (root / "requirements.txt").write_text(requirements)
    # user data that must survive every update
    (root / "projects" / "helmets" / "runs").mkdir(parents=True)
    (root / "projects" / "helmets" / "settings.json").write_text('{"epochs": 7}')
    (root / "settings.json").write_text('{"theme": "light"}')
    (root / ".venv" / "Scripts").mkdir(parents=True)
    (root / ".venv" / "Scripts" / "python.exe").write_text("fake")
    (root / "weights").mkdir()
    (root / "weights" / "yolo26s.pt").write_bytes(b"w")


def snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
            and "update_backups" not in p.parts and p.name not in ("update.log", "update_status.json")}


def extracted_package(tmp_path: Path, **kw: Any) -> Path:
    z = tmp_path / "pkg.zip"
    z.write_bytes(build_package(tmp_path, **kw))
    return up.safe_extract(z, tmp_path / "extracted")


USER_DATA = ["projects/helmets/settings.json", "settings.json", ".venv/Scripts/python.exe", "weights/yolo26s.pt"]


def test_apply_update_replaces_app_and_keeps_user_data(tmp_path: Path) -> None:
    target = tmp_path / "LocalTrainer"
    make_installed_app(target)
    before = snapshot(target)
    source = extracted_package(tmp_path)
    moved, added = helper.apply_update(source, target, tmp_path / "backup", log=lambda m: None)
    after = snapshot(target)
    assert (target / "main.py").read_text() == "print('new')\n"
    assert (target / "core" / "new_module.py").exists()
    assert not (target / "core" / "old_module.py").exists()  # stale app files go away
    assert (tmp_path / "backup" / "core" / "old_module.py").exists()  # but are backed up
    for key in USER_DATA:
        assert after[str(Path(key))] == before[str(Path(key))]
    assert "projects" not in moved + added and ".venv" not in moved + added


def test_apply_update_rolls_back_on_copy_failure(tmp_path: Path) -> None:
    target = tmp_path / "LocalTrainer"
    make_installed_app(target)
    before = snapshot(target)
    source = extracted_package(tmp_path)
    calls = {"n": 0}

    def flaky_copy(src: Path, dst: Path) -> None:
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("disk full")
        helper._copy(src, dst)

    with pytest.raises(OSError):
        helper.apply_update(source, target, tmp_path / "backup", log=lambda m: None, copy=flaky_copy)
    assert snapshot(target) == before


def test_run_installs_changed_requirements(tmp_path: Path) -> None:
    target = tmp_path / "LocalTrainer"
    make_installed_app(target)
    source = extracted_package(tmp_path, requirements="PyQt6>=6.8\nnewdep\n")
    pip_calls: list[Path] = []
    ok = helper.run(source, target, "1.0.1", do_relaunch=False,
                    pip=lambda py, req, log: pip_calls.append(req) or True)
    assert ok and pip_calls == [target / "requirements.txt"]
    status = json.loads((target / "update_status.json").read_text())
    assert status["ok"] and status["version"] == "1.0.1"


def test_run_skips_pip_when_requirements_unchanged(tmp_path: Path) -> None:
    target = tmp_path / "LocalTrainer"
    make_installed_app(target)
    source = extracted_package(tmp_path)  # same requirements text
    ok = helper.run(source, target, "1.0.1", do_relaunch=False,
                    pip=lambda *a: pytest.fail("pip must not run"))
    assert ok


def test_run_rolls_back_when_pip_fails(tmp_path: Path) -> None:
    target = tmp_path / "LocalTrainer"
    make_installed_app(target)
    before = snapshot(target)
    source = extracted_package(tmp_path, requirements="brokendep==0.0.0\n")
    ok = helper.run(source, target, "1.0.1", do_relaunch=False, pip=lambda *a: False)
    assert not ok
    assert snapshot(target) == before
    status = json.loads((target / "update_status.json").read_text())
    assert not status["ok"] and "restored" in status["message"]


def test_protected_names_cover_user_data() -> None:
    for name in ("projects", "settings.json", ".venv", "venv", "weights"):
        assert name in helper.PROTECTED
