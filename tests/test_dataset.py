"""Dataset detection, validation and the flat-to-split conversion."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from core import dataset as ds
from tests.conftest import make_pair


def tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        h.update(str(p.relative_to(root)).encode())
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


def test_detect_flat_and_split(flat_dataset: Path, split_dataset: Path) -> None:
    assert ds.detect_layout(flat_dataset).layout is ds.Layout.FLAT
    info = ds.detect_layout(split_dataset)
    assert info.layout is ds.Layout.SPLIT
    assert set(info.splits) == {"train", "val"}


def test_detect_accepts_valid_alias(tmp_path: Path) -> None:
    root = tmp_path / "rf"
    make_pair(root / "images" / "train", root / "labels" / "train", "a", ["0 0.5 0.5 0.1 0.1"])
    make_pair(root / "images" / "valid", root / "labels" / "valid", "b", ["0 0.5 0.5 0.1 0.1"])
    info = ds.detect_layout(root)
    assert info.splits["val"][0].name == "valid"


def test_detect_rejects_missing_folders(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    with pytest.raises(ds.DatasetError):
        ds.detect_layout(tmp_path)


def test_read_class_names(flat_dataset: Path, split_dataset: Path) -> None:
    assert ds.read_class_names(flat_dataset) == (["helmet", "head"], "classes.txt")
    assert ds.read_class_names(split_dataset) == (["car"], "data.yaml")


def test_read_class_names_none(tmp_path: Path) -> None:
    assert ds.read_class_names(tmp_path) is None


def test_validate_clean_dataset(flat_dataset: Path) -> None:
    rep = ds.validate(ds.detect_layout(flat_dataset), 2)
    assert rep.ok, rep.errors
    assert rep.images == {"all": 10}
    assert rep.class_counts == {0: 5, 1: 5}


def test_validate_finds_every_problem(tmp_path: Path) -> None:
    root = tmp_path / "bad"
    make_pair(root / "images", root / "labels", "ok", ["0 0.5 0.5 0.2 0.2"])
    make_pair(root / "images", root / "labels", "nolabel", None)
    make_pair(root / "images", root / "labels", "range", ["3 0.5 0.5 0.2 0.2"])
    make_pair(root / "images", root / "labels", "norm", ["0 1.5 0.5 0.2 0.2"])
    make_pair(root / "images", root / "labels", "short", ["0 0.5 0.5"])
    make_pair(root / "images", root / "labels", "float", ["0.5 0.5 0.5 0.2 0.2"])
    make_pair(root / "images", root / "labels", "empty", [])
    (root / "labels" / "orphan.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    rep = ds.validate(ds.detect_layout(root), num_classes=2)
    assert not rep.ok
    assert [p.stem for p in rep.missing_labels] == ["nolabel"]
    reasons = {p.stem: why for p, why in rep.invalid_files}
    assert set(reasons) == {"range", "norm", "short", "float"}
    assert "out of range" in reasons["range"]
    assert "between 0 and 1" in reasons["norm"]
    assert "expected 5 values" in reasons["short"]
    assert "whole number" in reasons["float"]
    assert [p.stem for p in rep.orphan_labels] == ["orphan"]
    assert rep.background_images == 1
    text = ds.report_text(rep, ["a", "b"])
    assert "Missing labels" in text and "Invalid label files" in text


def test_validate_split_needs_val(tmp_path: Path) -> None:
    root = tmp_path / "s"
    make_pair(root / "images" / "train", root / "labels" / "train", "a", ["0 0.5 0.5 0.1 0.1"])
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    rep = ds.validate(ds.detect_layout(root), 1)
    assert any("val split" in e for e in rep.errors)


def test_infer_class_count(flat_dataset: Path) -> None:
    assert ds.infer_class_count(ds.detect_layout(flat_dataset)) == 2


def test_split_items_is_deterministic_and_complete() -> None:
    items = [Path(f"{i}.jpg") for i in range(100)]
    train, val = ds.split_items(items, 0.2)
    assert len(train) == 80 and len(val) == 20
    assert set(train) | set(val) == set(items) and not set(train) & set(val)
    assert ds.split_items(list(reversed(items)), 0.2) == (train, val)  # order of input does not matter


def test_split_items_keeps_one_in_each_split() -> None:
    train, val = ds.split_items([Path("a.jpg"), Path("b.jpg")], 0.05)
    assert len(train) == 1 and len(val) == 1


def test_prepare_flat_splits_and_leaves_source_untouched(flat_dataset: Path, tmp_path: Path) -> None:
    before = tree_hash(flat_dataset)
    dest = tmp_path / "project" / "dataset"
    counts = ds.prepare_dataset(ds.detect_layout(flat_dataset), dest, val_ratio=0.2)
    assert counts == {"train": 8, "val": 2}
    for split, n in counts.items():
        imgs = sorted(p.stem for p in (dest / "images" / split).iterdir())
        lbls = sorted(p.stem for p in (dest / "labels" / split).iterdir())
        assert len(imgs) == n and imgs == lbls
    assert tree_hash(flat_dataset) == before
    # same seed -> same split on a second run
    first_val = sorted(p.name for p in (dest / "images" / "val").iterdir())
    ds.prepare_dataset(ds.detect_layout(flat_dataset), dest, val_ratio=0.2)
    assert sorted(p.name for p in (dest / "images" / "val").iterdir()) == first_val


def test_prepare_split_copies(split_dataset: Path, tmp_path: Path) -> None:
    before = tree_hash(split_dataset)
    dest = tmp_path / "p" / "dataset"
    assert ds.prepare_dataset(ds.detect_layout(split_dataset), dest) == {"train": 6, "val": 2}
    assert tree_hash(split_dataset) == before


def test_prepare_refuses_destination_inside_source(flat_dataset: Path) -> None:
    with pytest.raises(ds.DatasetError):
        ds.prepare_dataset(ds.detect_layout(flat_dataset), flat_dataset / "copy")
