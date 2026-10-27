"""Auto-export to ONNX after training, with fake torch and ultralytics modules."""
from __future__ import annotations

import sys
import threading
import types
from pathlib import Path
from typing import Any

import pytest
from PyQt6.QtCore import QCoreApplication

from core.project import Project, ProjectState, TrainSettings
from core.trainer import TrainConfig, TrainingSession, run_training


class FakeYOLO:
    """Stands in for ultralytics.YOLO. Class attributes steer what train() and export() do."""

    train_error: Exception | None = None
    export_error: Exception | None = None
    write_best = True
    exports: list[dict[str, Any]] = []

    def __init__(self, weights: str) -> None:
        self.weights = weights
        self.callbacks: dict[str, Any] = {}
        self.trainer: Any = None

    def add_callback(self, name: str, fn: Any) -> None:
        self.callbacks[name] = fn

    def train(self, **kw: Any) -> None:
        if FakeYOLO.train_error is not None:
            raise FakeYOLO.train_error
        save_dir = Path(kw["project"]) / kw["name"]
        weights = save_dir / "weights"
        weights.mkdir(parents=True)
        (weights / "last.pt").write_bytes(b"last")
        if FakeYOLO.write_best:
            (weights / "best.pt").write_bytes(b"best")
        batch_trainer = types.SimpleNamespace(epoch=0, epochs=kw["epochs"], stop=False)
        self.callbacks["on_train_batch_end"](batch_trainer)  # sets stop when the stop flag is up
        self.trainer = types.SimpleNamespace(save_dir=str(save_dir), best=str(weights / "best.pt"),
                                             last=str(weights / "last.pt"), stop=batch_trainer.stop)

    def export(self, **kw: Any) -> str:
        FakeYOLO.exports.append({"weights": self.weights, **kw})
        if FakeYOLO.export_error is not None:
            raise FakeYOLO.export_error
        out = Path(self.weights).with_suffix(".onnx")
        out.write_bytes(b"onnx" * 100)
        return str(out)


@pytest.fixture
def fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> type[FakeYOLO]:
    FakeYOLO.train_error = FakeYOLO.export_error = None
    FakeYOLO.write_best = True
    FakeYOLO.exports = []
    ultra = types.ModuleType("ultralytics")
    ultra.YOLO = FakeYOLO  # type: ignore[attr-defined]
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", ultra)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return FakeYOLO


def make_cfg(tmp_path: Path, auto: bool = True) -> TrainConfig:
    return TrainConfig(data_yaml=str(tmp_path / "data.yaml"), weights="yolo26n.pt", epochs=3, imgsz=320, batch=4,
                       workers=0, patience=10, device="cpu", runs_dir=str(tmp_path / "runs"),
                       run_name="run_test", auto_export_onnx=auto)


def train(tmp_path: Path, auto: bool = True, stop: bool = False) -> tuple[int, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    flag = threading.Event()
    if stop:
        flag.set()
    code = run_training(make_cfg(tmp_path, auto).payload(), events.append, flag)
    return code, events


def types_of(events: list[dict[str, Any]]) -> list[str]:
    return [str(e["type"]) for e in events if e["type"] not in ("log", "batch")]


# ------------------------------------------------------------ setting

def test_setting_defaults_to_on() -> None:
    assert TrainSettings().auto_export_onnx is True
    assert TrainSettings.from_dict({}).auto_export_onnx is True
    # a 1.0.0 settings file has every other key but not this one
    old = {"family": "yolo11", "size": "m", "epochs": 5, "imgsz": 640, "batch": 8, "batch_auto": False,
           "workers": 2, "patience": 20, "device": "auto"}
    assert TrainSettings.from_dict(old).auto_export_onnx is True


def test_setting_persists_with_project(tmp_path: Path) -> None:
    p = Project("helmets", base=tmp_path)
    p.create()
    p.save_state(ProjectState(train=TrainSettings(auto_export_onnx=False)))
    assert Project("helmets", base=tmp_path).load_state().train.auto_export_onnx is False
    p.save_state(ProjectState(train=TrainSettings(auto_export_onnx=True)))
    assert Project("helmets", base=tmp_path).load_state().train.auto_export_onnx is True


# ------------------------------------------------------------ worker

def test_export_runs_after_completion(tmp_path: Path, fake_ultralytics: type[FakeYOLO]) -> None:
    code, events = train(tmp_path)
    assert code == 0
    assert types_of(events) == ["done", "export_start", "export_done"]
    best = tmp_path / "runs" / "run_test" / "weights" / "best.pt"
    assert fake_ultralytics.exports == [{"weights": str(best), "format": "onnx", "imgsz": 320, "device": "cpu"}]
    done = events[-1]
    assert Path(done["path"]) == best.with_suffix(".onnx")
    assert done["size"] == 400
    assert next(e for e in events if e["type"] == "done")["stopped"] is False


def test_export_runs_after_stop(tmp_path: Path, fake_ultralytics: type[FakeYOLO]) -> None:
    code, events = train(tmp_path, stop=True)
    assert code == 0
    assert types_of(events) == ["stopping", "done", "export_start", "export_done"]
    assert next(e for e in events if e["type"] == "done")["stopped"] is True
    assert len(fake_ultralytics.exports) == 1


def test_export_skipped_after_crash(tmp_path: Path, fake_ultralytics: type[FakeYOLO]) -> None:
    fake_ultralytics.train_error = RuntimeError("CUDA error: device-side assert triggered")
    code, events = train(tmp_path)
    assert code == 1
    assert types_of(events) == ["error"]
    assert fake_ultralytics.exports == []


def test_export_skipped_when_off_or_best_missing(tmp_path: Path, fake_ultralytics: type[FakeYOLO]) -> None:
    code, events = train(tmp_path / "off", auto=False)
    assert code == 0 and "export_start" not in types_of(events)
    fake_ultralytics.write_best = False
    code, events = train(tmp_path / "nobest")
    assert code == 0 and "export_start" not in types_of(events)
    assert fake_ultralytics.exports == []


def test_export_failure_does_not_fail_the_run(tmp_path: Path, fake_ultralytics: type[FakeYOLO]) -> None:
    fake_ultralytics.export_error = RuntimeError("opset 99 is not supported")
    code, events = train(tmp_path)
    assert code == 0
    assert types_of(events) == ["done", "export_start", "export_failed"]
    assert "opset 99" in events[-1]["message"]
    weights = tmp_path / "runs" / "run_test" / "weights"
    assert (weights / "best.pt").read_bytes() == b"best"
    assert (weights / "last.pt").read_bytes() == b"last"


# ------------------------------------------------------------ Qt session

try:  # a full QApplication when widgets load, so UI tests in the same run can share it
    from PyQt6.QtWidgets import QApplication as _App
except ImportError:
    _App = QCoreApplication
APP = QCoreApplication.instance() or _App([])  # kept alive for the whole module


@pytest.fixture
def session(tmp_path: Path) -> tuple[TrainingSession, dict[str, list[Any]]]:
    s = TrainingSession()
    s.cfg = make_cfg(tmp_path)
    got: dict[str, list[Any]] = {"finished": [], "failed": [], "log": [], "exporting": []}
    s.finished.connect(got["finished"].append)
    s.failed.connect(lambda kind, msg: got["failed"].append((kind, msg)))
    s.log.connect(lambda text, _replace: got["log"].append(text))
    s.exporting.connect(lambda: got["exporting"].append(True))
    return s, got


def done_event(tmp_path: Path) -> dict[str, Any]:
    weights = tmp_path / "runs" / "run_test" / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    (weights / "best.pt").write_bytes(b"best")
    return {"type": "done", "save_dir": str(weights.parent), "best": str(weights / "best.pt"), "last": "",
            "stopped": False}


def test_session_reports_export_path_and_size(tmp_path: Path, session: Any) -> None:
    s, got = session
    s._on_event(done_event(tmp_path))
    s._on_event({"type": "export_start", "weights": str(tmp_path / "best.pt")})
    s._on_event({"type": "export_done", "path": "C:/runs/best.onnx", "size": 12_300_000})
    s._on_finished(0, False)
    assert got["exporting"] == [True]
    assert any("Exported ONNX: C:/runs/best.onnx (12.3 MB)" in line for line in got["log"])
    assert got["failed"] == []
    res = got["finished"][0]
    assert (res["onnx"], res["onnx_size"], res["forced"]) == ("C:/runs/best.onnx", 12_300_000, False)


def test_session_export_failure_still_finishes(tmp_path: Path, session: Any) -> None:
    s, got = session
    s._on_event(done_event(tmp_path))
    s._on_event({"type": "export_start", "weights": str(tmp_path / "best.pt")})
    s._on_event({"type": "export_failed", "message": "RuntimeError: boom", "traceback": "Traceback..."})
    s._on_finished(0, False)
    assert got["failed"] == []
    assert got["finished"][0]["onnx_error"] == "RuntimeError: boom"
    assert any("ONNX export failed" in line for line in got["log"])


def test_session_cancelled_export_keeps_weights(tmp_path: Path, session: Any) -> None:
    s, got = session
    done = done_event(tmp_path)
    best = Path(done["best"])
    partial = best.with_suffix(".onnx")
    partial.write_bytes(b"half")
    s._on_event(done)
    s._on_event({"type": "export_start", "weights": str(best)})
    s._on_finished(-1, True)
    assert got["failed"] == []
    res = got["finished"][0]
    assert res["onnx_error"] == "The ONNX export was cancelled."
    assert res["forced"] is False
    assert best.read_bytes() == b"best"
    assert not partial.exists()
