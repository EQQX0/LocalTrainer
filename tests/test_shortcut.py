"""Desktop shortcut fields and the never-raise contract (PowerShell itself is not run here)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from utils import shortcut


def test_fields_prefer_pythonw_next_to_python(tmp_path):
    (tmp_path / "python.exe").write_bytes(b"")
    (tmp_path / "pythonw.exe").write_bytes(b"")
    root = tmp_path / "Local Trainer's"
    f = shortcut.shortcut_fields(root, str(tmp_path / "python.exe"))
    assert f["LT_LNK_TARGET"] == str(tmp_path / "pythonw.exe")
    assert f["LT_LNK_ARGS"] == f'"{root / "main.py"}"'
    assert f["LT_LNK_DIR"] == str(root)
    assert f["LT_LNK_NAME"] == "LocalTrainer.lnk"


def test_fields_fall_back_to_python_without_pythonw(tmp_path):
    (tmp_path / "python.exe").write_bytes(b"")
    f = shortcut.shortcut_fields(tmp_path, str(tmp_path / "python.exe"))
    assert f["LT_LNK_TARGET"] == str(tmp_path / "python.exe")


def test_icon_ships_with_the_app():
    data = shortcut.ICON_FILE.read_bytes()
    assert data[:4] == b"\x00\x00\x01\x00"  # ICO header
    assert shortcut.shortcut_fields()["LT_LNK_ICON"] == f"{shortcut.ICON_FILE},0"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
@pytest.mark.parametrize("replace", [False, True])
def test_powershell_gets_fields_and_replace_flag(monkeypatch, replace):
    seen = {}

    def fake_run(cmd, env, **kw):
        seen.update(cmd=cmd, env=env, kw=kw)
        return subprocess.CompletedProcess(cmd, 0, stdout="created C:\\Users\\x\\Desktop\\LocalTrainer.lnk\n",
                                           stderr="")

    monkeypatch.setattr(shortcut.subprocess, "run", fake_run)
    assert shortcut.create_desktop_shortcut(replace=replace) == \
        ("created", "C:\\Users\\x\\Desktop\\LocalTrainer.lnk")
    assert seen["cmd"][0] == "powershell.exe"
    assert seen["env"]["LT_LNK_REPLACE"] == ("1" if replace else "0")
    assert seen["env"]["LT_LNK_ARGS"].endswith('main.py"')
    assert seen["kw"]["creationflags"] == subprocess.CREATE_NO_WINDOW


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_failures_are_reported_not_raised(monkeypatch):
    def boom(*a, **kw):
        raise FileNotFoundError("powershell.exe")

    monkeypatch.setattr(shortcut.subprocess, "run", boom)
    assert shortcut.create_desktop_shortcut()[0] == "failed"

    monkeypatch.setattr(shortcut.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1, stdout="", stderr="Access denied"))
    assert shortcut.create_desktop_shortcut() == ("failed", "Access denied")


def test_other_systems_are_skipped(monkeypatch):
    monkeypatch.setattr(shortcut.os, "name", "posix")
    assert shortcut.create_desktop_shortcut()[0] == "unsupported"


def test_setup_bat_makes_the_shortcut():
    text = (Path(__file__).resolve().parent.parent / "setup.bat").read_text(encoding="utf-8")
    assert "-m utils.shortcut --replace" in text
