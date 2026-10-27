"""Well-known folders, all relative to the app root."""
from __future__ import annotations

from pathlib import Path

APP_ROOT: Path = Path(__file__).resolve().parent.parent
PROJECTS_DIR: Path = APP_ROOT / "projects"
WEIGHTS_DIR: Path = APP_ROOT / "weights"            # downloaded pretrained .pt files
ULTRALYTICS_CONFIG_DIR: Path = APP_ROOT / ".ultralytics"  # keeps Ultralytics settings inside the app
APP_SETTINGS_FILE: Path = APP_ROOT / "settings.json"  # app-level preferences (theme, last project)
UPDATE_STATUS_FILE: Path = APP_ROOT / "update_status.json"
ASSETS_DIR: Path = APP_ROOT / "ui" / "assets"
JOBS_DIR: Path = APP_ROOT / ".cache" / "jobs"     # stop-flag files for worker processes


def ensure_dirs() -> None:
    """Create runtime folders that the app writes into."""
    for d in (PROJECTS_DIR, WEIGHTS_DIR, ULTRALYTICS_CONFIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
