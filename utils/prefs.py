"""App-level preferences in settings.json at the app root (never replaced by updates)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.paths import APP_SETTINGS_FILE

DEFAULTS: dict[str, Any] = {"theme": "dark", "last_project": "", "geometry": ""}


def load_prefs(path: Path = APP_SETTINGS_FILE) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return {**DEFAULTS, **(data if isinstance(data, dict) else {})}
    except (OSError, ValueError):
        return dict(DEFAULTS)


def save_prefs(prefs: dict[str, Any], path: Path = APP_SETTINGS_FILE) -> None:
    try:
        Path(path).write_text(json.dumps(prefs, indent=2), encoding="utf-8")
    except OSError:
        pass  # preferences are a convenience; never crash over them
