"""The installer exe builds and guards the folder it installs into (the full install is checked by hand)."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("build_installer", ROOT / "tools" / "build_installer.py")
build_installer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_installer)


def test_payload_name_matches_the_installer_source():
    source = build_installer.SOURCE.read_text(encoding="utf-8")
    assert f'const string Payload = "{build_installer.PAYLOAD}";' in source


@pytest.mark.skipif(sys.platform != "win32", reason="csc.exe exists only on Windows")
def test_installer_builds_and_refuses_a_foreign_folder(tmp_path):
    zip_path = tmp_path / "LocalTrainer-9.8.7.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("LocalTrainer-9.8.7/main.py", "print('hi')\n")
    exe = build_installer.build(zip_path, tmp_path / "dist")
    assert exe.name == "LocalTrainer-Setup-9.8.7.exe"
    assert exe.read_bytes()[:2] == b"MZ"
    assert (tmp_path / "dist" / f"{exe.name}.sha256").read_text().endswith(f"  {exe.name}\n")

    foreign = tmp_path / "My Documents"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("keep me")
    res = subprocess.run([str(exe), str(foreign)], capture_output=True, text=True, stdin=subprocess.DEVNULL,
                         timeout=60)
    assert res.returncode == 1
    assert "is not empty" in res.stdout
    assert sorted(p.name for p in foreign.iterdir()) == ["notes.txt"]
    assert not (tmp_path / "My Documents.partial").exists()
