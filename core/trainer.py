"""Training: configuration, the code that runs inside the worker process, and the Qt session.

The GUI never imports torch or Ultralytics. Training runs in a child process
(core.worker) so the window stays responsive and a stop can always be forced.
"""
from __future__ import annotations

import shutil
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from core.jobs import WorkerProcess
from utils.paths import WEIGHTS_DIR

EmitFn = Callable[[dict[str, Any]], None]

FIRST_BATCH_TIMEOUT_S = 240  # no batch this long after train start -> dataloader workers hung
OOM_HINT = (
    "The GPU ran out of memory. Try a lower batch size (for example half of the current one) "
    "or a smaller image size such as 512 or 416 px, or a smaller model size."
)


def make_run_name(now: datetime | None = None) -> str:
    return f"run_{(now or datetime.now()):%Y%m%d_%H%M%S}"


def resolve_device(setting: str, cuda_available: bool) -> str:
    """'auto' picks GPU 0 when CUDA works, otherwise CPU."""
    if setting == "auto":
        return "0" if cuda_available else "cpu"
    if setting != "cpu" and not cuda_available:
        return "cpu"
    return setting


def is_oom_text(text: str) -> bool:
    t = text.lower()
    return "out of memory" in t or "outofmemoryerror" in t or "cuda error: out of memory" in t


@dataclass
class TrainConfig:
    data_yaml: str
    weights: str          # e.g. yolo26s.pt
    epochs: int
    imgsz: int
    batch: int            # -1 means AutoBatch
    workers: int
    patience: int
    device: str           # "0" or "cpu"
    runs_dir: str
    run_name: str

    @property
    def run_dir(self) -> Path:
        return Path(self.runs_dir) / self.run_name

    def train_kwargs(self) -> dict[str, Any]:
        return {
            "data": self.data_yaml,
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "workers": self.workers,
            "patience": self.patience,
            "device": self.device,
            "project": self.runs_dir,
            "name": self.run_name,
            "exist_ok": False,
            "plots": True,
            "verbose": True,
        }

    def payload(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------- inside the worker

def run_training(payload: dict[str, Any], emit: EmitFn, stop_event: Any) -> int:
    """Train with Ultralytics, streaming events. Runs in the worker process."""
    cfg = TrainConfig(**payload)
    try:
        import torch
        from ultralytics import YOLO
    except Exception as exc:  # broken install
        emit({"type": "error", "kind": "import", "message": f"Could not import torch/ultralytics: {exc}"})
        return 1

    weights = WEIGHTS_DIR / cfg.weights  # absolute path, Ultralytics downloads the asset here if missing
    state: dict[str, Any] = {"final": False, "batch": 0, "last_beat": 0.0, "stopping": False}

    def on_train_start(trainer: Any) -> None:
        nb = len(trainer.train_loader) if trainer.train_loader is not None else 0
        state["nb"] = nb
        emit({"type": "train_start", "epochs": trainer.epochs, "save_dir": str(trainer.save_dir),
              "batch": trainer.batch_size, "batches_per_epoch": nb})
        original = trainer.final_eval

        def final_eval_flagged() -> Any:
            state["final"] = True
            return original()

        trainer.final_eval = final_eval_flagged

    def on_train_epoch_start(trainer: Any) -> None:
        state["batch"] = 0

    def on_train_batch_end(trainer: Any) -> None:
        state["batch"] += 1
        now = time.monotonic()
        if stop_event.is_set() and not state["stopping"]:
            state["stopping"] = True
            trainer.stop = True  # Ultralytics breaks after this batch, validates and saves last.pt
            emit({"type": "stopping"})
        if now - state["last_beat"] >= 0.5:
            state["last_beat"] = now
            emit({"type": "batch", "epoch": trainer.epoch + 1, "epochs": trainer.epochs,
                  "i": state["batch"], "nb": state.get("nb", 0)})

    def on_fit_epoch_end(trainer: Any) -> None:
        m = trainer.metrics or {}
        losses: dict[str, float] = {}
        if getattr(trainer, "tloss", None) is not None:
            losses = trainer.label_loss_items(trainer.tloss, prefix="train")

        def f(v: Any) -> float | None:
            try:
                return round(float(v), 5)
            except (TypeError, ValueError):
                return None

        emit({
            "type": "final_metrics" if state["final"] else "epoch",
            "epoch": min(trainer.epoch + 1, trainer.epochs),
            "epochs": trainer.epochs,
            "box_loss": f(losses.get("train/box_loss")),
            "cls_loss": f(losses.get("train/cls_loss")),
            "val_box_loss": f(m.get("val/box_loss")),
            "precision": f(m.get("metrics/precision(B)")),
            "recall": f(m.get("metrics/recall(B)")),
            "map50": f(m.get("metrics/mAP50(B)")),
            "map50_95": f(m.get("metrics/mAP50-95(B)")),
            "epoch_time": f(getattr(trainer, "epoch_time", None)),
        })

    try:
        if cfg.device != "cpu" and torch.cuda.is_available():
            emit({"type": "log", "message": f"Using GPU {torch.cuda.get_device_name(0)}"})
        model = YOLO(str(weights))
        model.add_callback("on_train_start", on_train_start)
        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        model.add_callback("on_train_batch_end", on_train_batch_end)
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        model.train(**cfg.train_kwargs())
        trainer = model.trainer
        save_dir = Path(trainer.save_dir)
        emit({"type": "done", "save_dir": str(save_dir),
              "best": str(trainer.best) if Path(trainer.best).exists() else "",
              "last": str(trainer.last) if Path(trainer.last).exists() else "",
              "stopped": bool(state["stopping"])})
        return 0
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        kind = "oom" if is_oom_text(text) else "other"
        emit({"type": "error", "kind": kind, "message": text, "traceback": traceback.format_exc()})
        return 1


# ------------------------------------------------------------ Qt session

class TrainingSession(QObject):
    """Owns one training job, including the Windows dataloader-hang fallback."""

    log = pyqtSignal(str, bool)
    started = pyqtSignal(dict)          # train_start event
    batch_progress = pyqtSignal(int, int, int, int)  # epoch, epochs, batch i, batches per epoch
    epoch_metrics = pyqtSignal(dict)
    stopping = pyqtSignal()
    finished = pyqtSignal(dict)         # {"save_dir", "best", "last", "stopped", "forced"}
    failed = pyqtSignal(str, str)       # kind ("oom", "other", "crash"), message

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.cfg: TrainConfig | None = None
        self._worker: WorkerProcess | None = None
        self._tail: deque[str] = deque(maxlen=40)
        self._done: dict[str, Any] | None = None
        self._error: dict[str, Any] | None = None
        self._stop_requested = False
        self._restarting = False
        self._retried_workers = False
        self._got_batch = False
        self._hang_timer = QTimer(self)
        self._hang_timer.setSingleShot(True)
        self._hang_timer.timeout.connect(self._on_hang)

    def is_running(self) -> bool:
        return self._worker is not None and self._worker.is_running()

    def start(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self._retried_workers = False
        self._launch()

    def _launch(self) -> None:
        assert self.cfg is not None
        self._done = self._error = None
        self._stop_requested = self._restarting = self._got_batch = False
        self._tail.clear()
        self._worker = WorkerProcess(self)
        self._worker.log.connect(self._on_log)
        self._worker.message.connect(self._on_event)
        self._worker.finished.connect(self._on_finished)
        self._worker.start("train", self.cfg.payload())

    def request_stop(self) -> bool:
        """First call stops gracefully after the current batch. Second call kills. Returns True if forced."""
        if not self.is_running() or self._worker is None:
            return False
        if not self._stop_requested:
            self._stop_requested = True
            self._worker.request_stop()
            if not self._got_batch:  # still setting up: nothing to keep, kill now
                self._worker.kill()
                return True
            return False
        self._worker.kill()
        return True

    def kill(self) -> None:
        if self._worker is not None:
            self._worker.kill()
            self._worker.wait(5000)

    # signal handlers
    def _on_log(self, text: str, replace: bool) -> None:
        self._tail.append(text)
        self.log.emit(text, replace)

    def _on_event(self, ev: dict[str, Any]) -> None:
        kind = ev.get("type")
        if kind == "train_start":
            if self.cfg and self.cfg.workers > 0 and not self._retried_workers:
                self._hang_timer.start(FIRST_BATCH_TIMEOUT_S * 1000)
            self.started.emit(ev)
        elif kind == "batch":
            self._got_batch = True
            self._hang_timer.stop()
            self.batch_progress.emit(int(ev["epoch"]), int(ev["epochs"]), int(ev["i"]), int(ev.get("nb") or 0))
        elif kind in ("epoch", "final_metrics"):
            self._got_batch = True
            self.epoch_metrics.emit(ev)
        elif kind == "stopping":
            self.stopping.emit()
        elif kind == "log":
            self.log.emit(str(ev.get("message", "")), False)
        elif kind == "done":
            self._done = ev
        elif kind == "error":
            self._error = ev

    def _on_hang(self) -> None:
        if self._got_batch or self._worker is None or self.cfg is None:
            return
        self.log.emit(
            f"No training batch after {FIRST_BATCH_TIMEOUT_S} s. The dataloader workers look stuck "
            "(a known Windows issue). Restarting with workers = 0.", False)
        self._restarting = True
        self._worker.kill()

    def _on_finished(self, code: int, killed: bool) -> None:
        self._hang_timer.stop()
        assert self.cfg is not None
        if self._restarting:
            self._cleanup_empty_run(self.cfg.run_dir)
            self._retried_workers = True
            self.cfg.workers = 0
            self.cfg.run_name = f"{self.cfg.run_name}_w0"
            self._launch()
            return
        run_dir = str(self.cfg.run_dir)
        if self._done is not None:
            self.finished.emit({**self._done, "forced": False})
        elif killed:
            weights = self.cfg.run_dir / "weights"
            last = weights / "last.pt"
            best = weights / "best.pt"
            if not last.exists() and not best.exists():
                self._cleanup_empty_run(self.cfg.run_dir)
            self.finished.emit({"save_dir": run_dir, "best": str(best) if best.exists() else "",
                                "last": str(last) if last.exists() else "", "stopped": True, "forced": True})
        elif self._error is not None:
            self.failed.emit(str(self._error.get("kind", "other")), str(self._error.get("message", "")))
        else:
            tail = "\n".join(self._tail)
            kind = "oom" if is_oom_text(tail) else "crash"
            self.failed.emit(kind, f"The training process exited with code {code}.\n\n{tail[-2000:]}")

    @staticmethod
    def _cleanup_empty_run(run_dir: Path) -> None:
        """Remove a run folder that holds no weights (aborted during setup)."""
        weights = run_dir / "weights"
        if run_dir.is_dir() and not (weights.is_dir() and any(weights.iterdir())):
            shutil.rmtree(run_dir, ignore_errors=True)
