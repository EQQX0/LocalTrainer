"""Run core.worker in a separate Python process and stream its output as Qt signals.

Protocol: the worker prints normal log text, plus one JSON event per line
prefixed with EVENT_PREFIX. The JSON payload is the only stdin line; the parent
then closes stdin. A stop request is a flag file the worker polls.
"""
from __future__ import annotations

import codecs
import json
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, pyqtSignal

from utils.paths import APP_ROOT, JOBS_DIR, ULTRALYTICS_CONFIG_DIR, WEIGHTS_DIR

EVENT_PREFIX = "@@LT@@ "
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
LINE_RE = re.compile(r"(\r\n|\n|\r)")


@dataclass
class LogLine:
    text: str
    replace_previous: bool  # True when the previous line was a progress line (ended with \r)


class OutputParser:
    """Turns a byte stream into log lines and JSON events. Qt-free so it can be unit tested."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._buffer = ""
        self._prev_progress = False

    def feed(self, data: bytes) -> list[LogLine | dict[str, Any]]:
        self._buffer += self._decoder.decode(data)
        parts = LINE_RE.split(self._buffer)
        self._buffer = parts.pop()  # unfinished tail
        out: list[LogLine | dict[str, Any]] = []
        for text, sep in zip(parts[0::2], parts[1::2]):
            item = self._line(text, progress=(sep == "\r"))
            if item is not None:
                out.append(item)
        return out

    def flush(self) -> list[LogLine | dict[str, Any]]:
        tail, self._buffer = self._buffer + self._decoder.decode(b"", final=True), ""
        item = self._line(tail, progress=False) if tail else None
        return [item] if item is not None else []

    def _line(self, text: str, progress: bool) -> LogLine | dict[str, Any] | None:
        text = ANSI_RE.sub("", text)
        if text.startswith(EVENT_PREFIX):
            try:
                return json.loads(text[len(EVENT_PREFIX):])
            except ValueError:
                pass
        if not text.strip():
            if progress:
                return None
            # keep blank lines out of the log, but end any progress run
            self._prev_progress = False
            return None
        line = LogLine(text.rstrip(), self._prev_progress)
        self._prev_progress = progress
        return line


def worker_environment() -> QProcessEnvironment:
    env = QProcessEnvironment.systemEnvironment()
    env.insert("PYTHONUNBUFFERED", "1")
    env.insert("PYTHONIOENCODING", "utf-8")
    env.insert("PYTHONUTF8", "1")
    env.insert("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))  # keep Ultralytics files inside the app
    env.insert("PYTHONPATH", str(APP_ROOT))
    env.insert("NO_COLOR", "1")
    env.insert("TERM", "dumb")
    return env


class WorkerProcess(QObject):
    """One worker process. Emits log lines and events; finished(exit_code, was_killed)."""

    log = pyqtSignal(str, bool)
    message = pyqtSignal(dict)
    finished = pyqtSignal(int, bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._proc: QProcess | None = None
        self._parser = OutputParser()
        self._killed = False
        self._stop_file: Path | None = None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.state() != QProcess.ProcessState.NotRunning

    def start(self, mode: str, payload: dict[str, Any]) -> None:
        if self.is_running():
            raise RuntimeError("A job is already running.")
        self._parser = OutputParser()
        self._killed = False
        proc = QProcess(self)
        proc.setProcessEnvironment(worker_environment())
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        proc.setWorkingDirectory(str(WEIGHTS_DIR))  # Ultralytics drops downloaded weights in the cwd
        proc.readyReadStandardOutput.connect(self._read)
        proc.finished.connect(self._done)
        proc.errorOccurred.connect(self._error)
        self._proc = proc
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        self._stop_file = JOBS_DIR / f"{uuid.uuid4().hex}.stop"
        data = (json.dumps({**payload, "_stop_file": str(self._stop_file)}) + "\n").encode("utf-8")

        def send_payload() -> None:  # stdin is only writable once the process runs
            proc.write(data)
            proc.closeWriteChannel()

        proc.started.connect(send_payload)
        proc.start(sys.executable, ["-u", "-m", "core.worker", mode])

    def request_stop(self) -> None:
        """Ask the worker to stop at its next checkpoint (it polls for the stop file)."""
        if self.is_running() and self._stop_file is not None:
            try:
                self._stop_file.touch()
            except OSError:
                pass

    def kill(self) -> None:
        if self.is_running() and self._proc is not None:
            self._killed = True
            self._proc.kill()

    def wait(self, ms: int = 5000) -> bool:
        return self._proc.waitForFinished(ms) if self._proc is not None else True

    # internals
    def _emit(self, items: list[LogLine | dict[str, Any]]) -> None:
        for item in items:
            if isinstance(item, LogLine):
                self.log.emit(item.text, item.replace_previous)
            else:
                self.message.emit(item)

    def _read(self) -> None:
        if self._proc is not None:
            self._emit(self._parser.feed(bytes(self._proc.readAllStandardOutput())))

    def _done(self, code: int, _status: QProcess.ExitStatus) -> None:
        self._read()
        self._emit(self._parser.flush())
        if self._stop_file is not None:
            self._stop_file.unlink(missing_ok=True)
        self.finished.emit(code, self._killed)

    def _error(self, err: QProcess.ProcessError) -> None:
        if err == QProcess.ProcessError.FailedToStart:
            self.log.emit(f"Could not start the worker process ({sys.executable}).", False)
            self.finished.emit(-1, False)
