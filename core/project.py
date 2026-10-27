"""Projects: one folder per model under projects/<name>/ with its settings.json."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from utils.paths import PROJECTS_DIR

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}

FAMILIES = ("yolo26", "yolo11", "yolov8")
SIZES = ("n", "s", "m", "l", "x")


def validate_name(name: str) -> str | None:
    """Return an error message, or None if the name is a valid project folder name."""
    name = name.strip()
    if not name:
        return "Type a model name, for example helmets."
    if name.lower() in RESERVED:
        return f"'{name}' is reserved by Windows. Pick another name."
    if not NAME_RE.match(name):
        return "Use letters, digits, - and _ only (up to 64 characters, starting with a letter or digit)."
    return None


@dataclass
class TrainSettings:
    family: str = "yolo26"
    size: str = "s"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    batch_auto: bool = False
    workers: int = 4
    patience: int = 50
    device: str = "auto"  # auto | 0 | cpu

    @property
    def weights(self) -> str:
        return f"{self.family}{self.size}.pt"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainSettings":
        known = {f.name for f in fields(cls)}
        s = cls(**{k: v for k, v in data.items() if k in known})
        if s.family not in FAMILIES:
            s.family = "yolo26"
        if s.size not in SIZES:
            s.size = "s"
        return s


@dataclass
class ProjectState:
    dataset_source: str = ""
    layout: str = ""
    val_ratio: float = 0.2
    class_names: list[str] = field(default_factory=list)
    prepared: bool = False  # dataset copied, validated and data.yaml written
    train: TrainSettings = field(default_factory=TrainSettings)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectState":
        return cls(
            dataset_source=str(data.get("dataset_source", "")),
            layout=str(data.get("layout", "")),
            val_ratio=float(data.get("val_ratio", 0.2)),
            class_names=[str(n) for n in data.get("class_names", [])],
            prepared=bool(data.get("prepared", False)),
            train=TrainSettings.from_dict(data.get("train", {})),
        )


class Project:
    def __init__(self, name: str, base: Path = PROJECTS_DIR) -> None:
        self.name = name
        self.root = Path(base) / name

    # folders and files
    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def data_yaml(self) -> Path:
        return self.root / "data.yaml"

    @property
    def settings_file(self) -> Path:
        return self.root / "settings.json"

    def exists(self) -> bool:
        return self.root.is_dir()

    def create(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        if not self.settings_file.exists():
            self.save_state(ProjectState())

    def load_state(self) -> ProjectState:
        try:
            return ProjectState.from_dict(json.loads(self.settings_file.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return ProjectState()

    def save_state(self, state: ProjectState) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.settings_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(self.settings_file)

    def list_runs(self) -> list[Path]:
        """Run folders, newest first."""
        if not self.runs_dir.is_dir():
            return []
        return sorted((p for p in self.runs_dir.iterdir() if p.is_dir() and p.name.startswith("run_")),
                      key=lambda p: p.name, reverse=True)

    def latest_trained_run(self) -> Path | None:
        for run in self.list_runs():
            if (run / "weights" / "best.pt").is_file() or (run / "weights" / "last.pt").is_file():
                return run
        return None


def list_projects(base: Path = PROJECTS_DIR) -> list[str]:
    if not Path(base).is_dir():
        return []
    return sorted(p.name for p in Path(base).iterdir() if p.is_dir() and validate_name(p.name) is None)
