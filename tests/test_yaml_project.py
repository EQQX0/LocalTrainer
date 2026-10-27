"""data.yaml generation, project names and settings persistence."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.project import Project, ProjectState, TrainSettings, list_projects, validate_name
from core.trainer import TrainConfig, is_oom_text, make_run_name, resolve_device
from core.yaml_gen import build_data_config, write_data_yaml


def test_data_yaml_has_absolute_paths_and_names(tmp_path: Path) -> None:
    out = write_data_yaml(tmp_path / "data.yaml", tmp_path / "dataset", ["helmet", "head"])
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert Path(data["path"]).is_absolute()
    assert Path(data["train"]).is_absolute() and data["train"].endswith("images/train")
    assert data["val"].endswith("images/val")
    assert data["nc"] == 2
    assert data["names"] == {0: "helmet", 1: "head"}


def test_data_yaml_rejects_empty_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_data_config(tmp_path, ["ok", " "])


def test_data_yaml_unicode(tmp_path: Path) -> None:
    out = write_data_yaml(tmp_path / "d.yaml", tmp_path, ["Schutzhelm", "café"])
    assert yaml.safe_load(out.read_text(encoding="utf-8"))["names"][1] == "café"


@pytest.mark.parametrize("name", ["helmets", "my-model_2", "A1"])
def test_valid_names(name: str) -> None:
    assert validate_name(name) is None


@pytest.mark.parametrize("name", ["", "has space", "bad/slash", "dot.name", "-lead", "CON", "lpt1", "x" * 65, "ä"])
def test_invalid_names(name: str) -> None:
    assert validate_name(name) is not None


def test_settings_roundtrip(tmp_path: Path) -> None:
    p = Project("helmets", base=tmp_path)
    p.create()
    state = ProjectState(dataset_source="D:/data", layout="flat", val_ratio=0.25, class_names=["a", "b"],
                         prepared=True, train=TrainSettings(family="yolo11", size="m", epochs=5, batch_auto=True))
    p.save_state(state)
    loaded = Project("helmets", base=tmp_path).load_state()
    assert loaded == state
    assert loaded.train.weights == "yolo11m.pt"
    assert list_projects(tmp_path) == ["helmets"]


def test_settings_defaults_and_bad_file(tmp_path: Path) -> None:
    p = Project("x", base=tmp_path)
    p.create()
    p.settings_file.write_text("{not json", encoding="utf-8")
    s = p.load_state().train
    assert (s.family, s.size, s.epochs, s.imgsz, s.batch, s.workers, s.patience) == ("yolo26", "s", 100, 640, 16, 4, 50)
    assert s.weights == "yolo26s.pt"


def test_unknown_family_falls_back() -> None:
    assert TrainSettings.from_dict({"family": "yolo99", "size": "z", "extra": 1}).weights == "yolo26s.pt"


def test_runs_are_listed_newest_first(tmp_path: Path) -> None:
    p = Project("m", base=tmp_path)
    p.create()
    for name in ("run_20260101_000000", "run_20260301_000000", "run_20260201_000000"):
        (p.runs_dir / name).mkdir()
    (p.runs_dir / "run_20260201_000000" / "weights").mkdir()
    (p.runs_dir / "run_20260201_000000" / "weights" / "last.pt").write_bytes(b"x")
    assert [r.name for r in p.list_runs()][0] == "run_20260301_000000"
    assert p.latest_trained_run().name == "run_20260201_000000"


def test_resolve_device() -> None:
    assert resolve_device("auto", True) == "0"
    assert resolve_device("auto", False) == "cpu"
    assert resolve_device("0", False) == "cpu"
    assert resolve_device("cpu", True) == "cpu"


def test_train_kwargs_use_separate_run_folder(tmp_path: Path) -> None:
    cfg = TrainConfig(str(tmp_path / "data.yaml"), "yolo26s.pt", 100, 640, -1, 4, 50, "0", str(tmp_path), make_run_name())
    kw = cfg.train_kwargs()
    assert kw["batch"] == -1 and kw["exist_ok"] is False
    assert kw["name"].startswith("run_") and kw["project"] == str(tmp_path)


def test_oom_detection() -> None:
    assert is_oom_text("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")
    assert not is_oom_text("FileNotFoundError: missing")
