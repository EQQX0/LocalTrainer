"""Update dialog: release notes, download progress, checksum, hand-off to the helper."""
from __future__ import annotations

import threading
from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QApplication, QDialog, QHBoxLayout, QProgressBar, QTextBrowser, QVBoxLayout, QWidget

from app.version import __version__
from core.updater import ReleaseInfo, default_work_dir, launch_helper, stage_update
from ui.theme import SP1, SP2, SP3
from ui.widgets import Banner, Task, button, label
from utils.fmt import fmt_bytes
from utils.paths import APP_ROOT


class UpdateDialog(QDialog):
    def __init__(self, info: ReleaseInfo, can_update: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.info = info
        self._cancel = threading.Event()
        self._task: Task | None = None
        self.setWindowTitle("Update available")
        self.setMinimumSize(560, 460)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(SP3, SP3, SP3, SP3)
        lay.setSpacing(SP2)
        lay.addWidget(label(f"Version {info.version} is ready", "display"))
        size = f" · download {fmt_bytes(info.zip_size)}" if info.zip_size else ""
        lay.addWidget(label(f"You have version {__version__}{size}. Your projects, settings and "
                            "virtual environment stay as they are.", "muted", wrap=True))
        self.notes = QTextBrowser()
        self.notes.setOpenExternalLinks(True)
        self.notes.setMarkdown(info.notes or "No release notes were published for this version.")
        self.notes.setAccessibleName("Release notes")
        lay.addWidget(self.notes, 1)
        self.banner = Banner()
        lay.addWidget(self.banner)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.hide()
        self.status = label("", "muted")
        lay.addWidget(self.progress)
        lay.addWidget(self.status)
        row = QHBoxLayout()
        row.setSpacing(SP1)
        self.web_btn = button("View on GitHub", "ghost", "Open the release page in your browser")
        self.web_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(info.html_url)))
        row.addWidget(self.web_btn)
        row.addStretch(1)
        self.later_btn = button("Later", tooltip="Close and keep the current version")
        self.later_btn.clicked.connect(self._later)
        self.update_btn = button("Update now", "primary",
                                 "Download, verify the SHA-256 checksum, install and restart LocalTrainer")
        self.update_btn.clicked.connect(self._start)
        self.update_btn.setDefault(True)
        row.addWidget(self.later_btn)
        row.addWidget(self.update_btn)
        lay.addLayout(row)
        if not can_update:
            self.update_btn.setEnabled(False)
            self.banner.show_message("warning", "Stop training first",
                                     "The update restarts the app, so it waits until training has finished.")

    def _later(self) -> None:
        if self._task is not None and self._task.isRunning():
            self._cancel.set()
            self.status.setText("Cancelling…")
            return
        self.reject()

    def _start(self) -> None:
        self.banner.hide()
        self.update_btn.setEnabled(False)
        self.later_btn.setText("Cancel")
        self.progress.show()
        self.progress.setValue(0)
        self.status.setText("Downloading…")
        self._cancel.clear()
        work_dir = default_work_dir()
        self._work_dir = work_dir
        self._task = Task(lambda progress: stage_update(self.info, work_dir, progress, cancel=self._cancel), self)
        self._task.progress.connect(self._on_progress)
        self._task.succeeded.connect(self._on_staged)
        self._task.failed.connect(self._on_failed)
        self._task.start()

    def _on_progress(self, done: int, total: int) -> None:
        total = total or self.info.zip_size
        if total:
            self.progress.setValue(int(done / total * 1000))
            self.status.setText(f"Downloading {fmt_bytes(done)} of {fmt_bytes(total)}")
        else:
            self.progress.setRange(0, 0)
            self.status.setText(f"Downloading {fmt_bytes(done)}")

    def _on_staged(self, new_root: object) -> None:
        self.progress.setRange(0, 1000)
        self.progress.setValue(1000)
        self.status.setText("Checksum verified. Restarting to finish the update…")
        try:
            launch_helper(Path(str(new_root)), APP_ROOT, self.info.version, self._work_dir)
        except OSError as exc:
            self._on_failed(exc)
            return
        QApplication.processEvents()
        self.accept()
        app = QApplication.instance()
        if app is not None:
            app.setProperty("updating", True)
            for w in app.topLevelWidgets():
                w.close()
            app.quit()

    def _on_failed(self, exc: object) -> None:
        self.progress.hide()
        self.status.setText("")
        self.later_btn.setText("Later")
        self.update_btn.setEnabled(True)
        self.update_btn.setText("Try again")
        self.banner.show_message("error", "The update did not finish", f"{exc} Your current version is unchanged.")
        self.update_btn.setFocus(Qt.FocusReason.OtherFocusReason)
