"""Prediction and ONNX export with trained weights (run inside the worker process)."""
from __future__ import annotations

import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from core.dataset import IMAGE_EXTS

EmitFn = Callable[[dict[str, Any]], None]
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v", ".mpeg", ".mpg", ".gif"}


def find_weights(run_dir: Path) -> Path | None:
    """best.pt if it exists, else last.pt."""
    for name in ("best.pt", "last.pt"):
        p = Path(run_dir) / "weights" / name
        if p.is_file():
            return p
    return None


def is_supported_source(path: Path) -> bool:
    path = Path(path)
    return path.is_dir() or path.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS


def output_target(save_dir: Path) -> Path:
    """The single annotated file if there is exactly one, otherwise the folder."""
    save_dir = Path(save_dir)
    files = [p for p in save_dir.iterdir() if p.is_file()] if save_dir.is_dir() else []
    return files[0] if len(files) == 1 else save_dir


def run_prediction(payload: dict[str, Any], emit: EmitFn, stop_event: Any) -> int:
    """payload: weights, source, run_dir, device, imgsz."""
    try:
        from ultralytics import YOLO

        source = Path(payload["source"])
        out_root = Path(payload["run_dir"]) / "predict"
        name = f"predict_{datetime.now():%Y%m%d_%H%M%S}"
        model = YOLO(payload["weights"])
        count = 0
        save_dir: Path | None = None
        results = model.predict(source=str(source), stream=True, save=True, project=str(out_root), name=name,
                                exist_ok=True, device=payload.get("device", "cpu"),
                                imgsz=int(payload.get("imgsz", 640)), verbose=False)
        for result in results:
            count += 1
            if getattr(result, "save_dir", None):
                save_dir = Path(result.save_dir)
            if count % 25 == 0:
                emit({"type": "log", "message": f"Processed {count} image(s) or frame(s)"})
            if stop_event.is_set():
                break
        save_dir = save_dir or out_root / name
        emit({"type": "done", "save_dir": str(save_dir), "open": str(output_target(save_dir)), "count": count})
        return 0
    except Exception as exc:
        emit({"type": "error", "kind": "other", "message": f"{type(exc).__name__}: {exc}",
              "traceback": traceback.format_exc()})
        return 1


def export_onnx(weights: Path | str, imgsz: int) -> Path:
    """Export weights to ONNX next to them (best.pt -> best.onnx). Raises on failure."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out = model.export(format="onnx", imgsz=int(imgsz), device="cpu")
    path = Path(str(out)) if out else Path(weights).with_suffix(".onnx")
    if not path.is_file():
        raise RuntimeError(f"Ultralytics reported success but {path} does not exist")
    return path


def run_export(payload: dict[str, Any], emit: EmitFn, stop_event: Any) -> int:
    """payload: weights, imgsz. Writes best.onnx next to the weights."""
    try:
        out = export_onnx(payload["weights"], int(payload.get("imgsz", 640)))
        emit({"type": "done", "path": str(out), "size": out.stat().st_size})
        return 0
    except Exception as exc:
        emit({"type": "error", "kind": "other", "message": f"{type(exc).__name__}: {exc}",
              "traceback": traceback.format_exc()})
        return 1
