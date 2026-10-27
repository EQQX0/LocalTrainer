"""Entry point of the worker process: python -m core.worker <mode>.

Reads a JSON payload from the only stdin line. Stop requests arrive as a flag file.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

EVENT_PREFIX = "@@LT@@ "
_lock = threading.Lock()


def emit(event: dict[str, Any]) -> None:
    line = "\n" + EVENT_PREFIX + json.dumps(event, default=str) + "\n"
    with _lock:
        sys.stdout.write(line)
        sys.stdout.flush()


class StopFlag:
    """Set when the parent creates the stop file. A thread blocked on stdin would stall imports
    on Windows, so commands travel through the file system instead."""

    def __init__(self, path: str | None) -> None:
        self._path = Path(path) if path else None
        self._set = False
        self._last_check = 0.0

    def set(self) -> None:
        self._set = True

    def is_set(self) -> bool:
        now = time.monotonic()
        if not self._set and self._path is not None and now - self._last_check >= 0.25:
            self._last_check = now
            self._set = self._path.exists()
        return self._set


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except AttributeError:
            pass
    payload = json.loads(sys.stdin.readline() or "{}")  # the parent closes stdin right after this line
    stop_event = StopFlag(payload.pop("_stop_file", None))
    if mode == "train":
        from core.trainer import run_training
        return run_training(payload, emit, stop_event)
    if mode == "predict":
        from core.predictor import run_prediction
        return run_prediction(payload, emit, stop_event)
    if mode == "export":
        from core.predictor import run_export
        return run_export(payload, emit, stop_event)
    if mode == "envcheck":
        from core.env_check import probe
        emit({"type": "env", **probe()})
        return 0
    emit({"type": "error", "kind": "other", "message": f"Unknown worker mode '{mode}'"})
    return 2


if __name__ == "__main__":  # required on Windows: dataloader workers re-import the main module
    sys.exit(main())
