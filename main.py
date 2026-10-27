"""LocalTrainer: train YOLO object detection models without the command line."""
from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path

# make "app", "core", "ui" and "utils" importable however the script is started
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QGuiApplication, QIcon
    from PyQt6.QtWidgets import QApplication

    from app.version import APP_NAME, __version__
    from ui.main_window import MainWindow
    from ui.theme import resolve_fonts
    from utils.paths import ASSETS_DIR, ensure_dirs
    from utils.prefs import load_prefs

    # crisp rendering at 125 % / 150 % Windows scaling (Qt 6 scales by default; keep fractional factors)
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    if os.name == "nt":
        try:  # own taskbar icon instead of python.exe's
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(f"{APP_NAME}.{__version__}")
        except (AttributeError, OSError):
            pass

    ensure_dirs()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setWindowIcon(QIcon(str(ASSETS_DIR / "icon.svg")))
    app.setStyle("Fusion")  # predictable base for the stylesheet on every Windows version
    fonts = resolve_fonts()
    window = MainWindow(fonts, load_prefs())
    window.show()
    if os.environ.get("LOCALTRAINER_SMOKE"):  # headless smoke test: quit after startup work ran
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(int(os.environ.get("LOCALTRAINER_SMOKE_MS", "4000")), window.close)
    return app.exec()


if __name__ == "__main__":  # required on Windows so worker processes do not re-run the GUI
    multiprocessing.freeze_support()
    sys.exit(main())
