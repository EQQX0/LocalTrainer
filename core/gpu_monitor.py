"""Poll nvidia-smi for live VRAM usage without blocking the UI (QProcess is asynchronous)."""
from __future__ import annotations

import shutil

from PyQt6.QtCore import QObject, QProcess, QTimer, pyqtSignal

QUERY = ["--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]


def parse_smi(text: str) -> tuple[str, int, int, int] | None:
    """First GPU line -> (name, used MiB, total MiB, utilisation %)."""
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            try:
                return parts[0], int(float(parts[1])), int(float(parts[2])), int(float(parts[3]))
            except ValueError:
                return None
    return None


class GpuMonitor(QObject):
    sample = pyqtSignal(str, int, int, int)  # name, used MiB, total MiB, util %
    unavailable = pyqtSignal()

    def __init__(self, interval_ms: int = 2000, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._exe = shutil.which("nvidia-smi")
        self._proc: QProcess | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._poll)

    def start(self) -> None:
        if not self._exe:
            self.unavailable.emit()
            return
        self._poll()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        if self._proc is not None and self._proc.state() != QProcess.ProcessState.NotRunning:
            self._proc.kill()
            self._proc.waitForFinished(1000)

    def _poll(self) -> None:
        if self._proc is None:  # one reusable process; never deleted while signals may still fire
            self._proc = QProcess(self)
            self._proc.finished.connect(self._done)
            self._proc.errorOccurred.connect(self._failed)
        if self._proc.state() != QProcess.ProcessState.NotRunning:
            return  # previous query still running
        self._proc.start(self._exe, QUERY)

    def _done(self, *_: object) -> None:
        if self._proc is None:
            return
        parsed = parse_smi(bytes(self._proc.readAllStandardOutput()).decode("utf-8", "replace"))
        if parsed:
            self.sample.emit(*parsed)

    def _failed(self, err: QProcess.ProcessError) -> None:
        if err == QProcess.ProcessError.FailedToStart:
            self._timer.stop()
            self.unavailable.emit()
