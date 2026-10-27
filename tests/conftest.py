"""Shared fixtures: tiny synthetic YOLO datasets built in a temp folder."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# smallest valid PNG (1x1 px); content is irrelevant to validation
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
)


def make_pair(img_dir: Path, lbl_dir: Path, stem: str, lines: list[str] | None) -> None:
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / f"{stem}.png").write_bytes(PNG)
    if lines is not None:
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def flat_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "flat"
    for i in range(10):
        make_pair(root / "images", root / "labels", f"img{i:02d}", [f"{i % 2} 0.5 0.5 0.2 0.2"])
    (root / "classes.txt").write_text("helmet\nhead\n", encoding="utf-8")
    return root


@pytest.fixture
def split_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "split"
    for split, n in (("train", 6), ("val", 2)):
        for i in range(n):
            make_pair(root / "images" / split, root / "labels" / split, f"{split}{i}", ["0 0.4 0.4 0.1 0.3"])
    (root / "data.yaml").write_text("names:\n  0: car\n", encoding="utf-8")
    return root
