"""Main window: project, dataset, settings on the left; training monitor on the right."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QByteArray, Qt, QTimer, QUrl
from PyQt6.QtGui import QAction, QCloseEvent, QDesktopServices, QGuiApplication, QKeySequence, QResizeEvent
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
    QMenu, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter, QStackedWidget,
    QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser, QVBoxLayout, QWidget,
)

from app.version import __version__
from core import dataset as ds
from core.env_check import FIX_COMMAND_VENV
from core.gpu_monitor import GpuMonitor
from core.jobs import WorkerProcess
from core.predictor import find_weights
from core.project import FAMILIES, SIZES, Project, ProjectState, TrainSettings, list_projects, validate_name
from core.trainer import OOM_HINT, TrainConfig, TrainingSession, make_run_name, resolve_device
from core.updater import ReleaseInfo, UpdateError, cleanup_work_dirs, fetch_latest_release, is_newer
from core.yaml_gen import write_data_yaml
from ui.theme import SP1, SP2, SP3, Fonts, build_qss
from ui.update_dialog import UpdateDialog
from ui.widgets import Banner, ClassEditor, EmptyState, LogView, MetricsChart, StatTile, Task, button, label, repolish
from utils.fmt import fmt_bytes, fmt_duration, fmt_metric
from utils.paths import PROJECTS_DIR, UPDATE_STATUS_FILE
from utils.prefs import save_prefs
from utils.shortcut import create_desktop_shortcut

SIDEBAR_WIDTH = 408


class BadgeButton(QPushButton):
    """Button with a small corner badge, used for 'Check for updates'."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.badge = QLabel("New", self)
        self.badge.setProperty("role", "badge")
        self.badge.hide()

    def set_badge(self, visible: bool) -> None:
        self.badge.setVisible(visible)
        self.badge.adjustSize()
        self._place()

    def resizeEvent(self, ev: QResizeEvent) -> None:
        super().resizeEvent(ev)
        self._place()

    def _place(self) -> None:
        self.badge.move(self.width() - self.badge.width() - 2, 0)


def section(title: str, caption: str = "") -> tuple[QWidget, QVBoxLayout]:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(SP1)
    lay.addWidget(label(title, "heading"))
    if caption:
        lay.addWidget(label(caption, "muted", wrap=True))
    return w, lay


def form() -> QFormLayout:
    f = QFormLayout()
    f.setContentsMargins(0, SP1, 0, 0)
    f.setHorizontalSpacing(SP2)
    f.setVerticalSpacing(SP1)
    f.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return f


def spin(lo: int, hi: int, value: int, tip: str, step: int = 1, suffix: str = "") -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    s.setSingleStep(step)
    s.setSuffix(suffix)
    s.setToolTip(tip)
    s.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)  # arrow keys and wheel still work
    s.setAccelerated(True)
    return s


class MainWindow(QMainWindow):
    def __init__(self, fonts: Fonts, prefs: dict[str, Any]) -> None:
        super().__init__()
        self.fonts = fonts
        self.prefs = prefs
        self.theme = prefs.get("theme", "dark") if prefs.get("theme") in ("dark", "light") else "dark"
        self.project: Project | None = None
        self.state: ProjectState | None = None
        self.info: ds.DatasetInfo | None = None
        self.env: dict[str, Any] | None = None
        self.release: ReleaseInfo | None = None
        self._tasks: set[Task] = set()
        self._loading = False
        self._run_started_at = 0.0
        self._current_run: Path | None = None

        self.session = TrainingSession(self)
        self.job = WorkerProcess(self)        # predict / export
        self.env_job = WorkerProcess(self)    # startup CUDA check
        self.gpu = GpuMonitor(parent=self)

        self.setWindowTitle(f"LocalTrainer {__version__}")
        self.resize(1440, 900)
        self.setMinimumSize(1100, 700)
        self._build()
        self._connect()
        self._shortcuts()
        self.apply_theme(self.theme)
        if prefs.get("geometry"):
            self.restoreGeometry(QByteArray.fromBase64(prefs["geometry"].encode()))
        self._refresh_project_list()
        self._set_project(None)
        self.name_edit.setFocus()
        QTimer.singleShot(0, self._startup)

    # ================================================================ layout
    def _build(self) -> None:
        root = QWidget()
        root.setObjectName("Ground")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_topbar())
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_sidebar())
        self.right = QStackedWidget()
        self.right.addWidget(EmptyState(
            "Name your model to begin",
            "Type a name such as helmets on the left and press Create. LocalTrainer makes a folder for it "
            "under projects/ and keeps every training run there, side by side.",
            ["Then pick your labelled dataset folder. It is checked and copied; the original is never changed.",
             "Choose a model and settings, press Start training, and watch mAP50 climb epoch by epoch.",
             "When it is done, test the best weights on an image, a folder or a video, or export them to ONNX."]))
        self.right.addWidget(self._build_monitor())
        body.addWidget(self.right, 1)
        outer.addLayout(body, 1)
        self.setCentralWidget(root)
        self._build_statusbar()

    def _build_topbar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("TopBar")
        bar.setFixedHeight(64)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(SP3, SP1, SP3, SP1)
        lay.setSpacing(SP2)
        mark = label("LocalTrainer", "wordmark")
        mark.setAccessibleName("LocalTrainer")
        lay.addWidget(mark)
        self.project_lbl = label("", "project")
        lay.addWidget(self.project_lbl)
        lay.addStretch(1)
        self.theme_btn = button("Light theme", "ghost", "Switch between dark and light theme (Ctrl+Shift+L)")
        lay.addWidget(self.theme_btn)
        self.update_btn = BadgeButton("Check for updates")
        self.update_btn.setProperty("kind", "update")
        self.update_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update_btn.setToolTip("Look for a newer LocalTrainer release on GitHub (Ctrl+U)")
        lay.addWidget(self.update_btn)
        return bar

    def _build_sidebar(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(SIDEBAR_WIDTH)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        inner.setObjectName("Sidebar")
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(SP3, SP3, SP3, SP3)
        lay.setSpacing(SP3 + SP1)

        # ---- project
        sec, s = section("Project", "Each model lives in its own folder under projects/.")
        row = QHBoxLayout()
        row.setSpacing(SP1)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Model name, for example helmets")
        self.name_edit.setToolTip("Letters, digits, - and _. This becomes the folder projects/<name>/ (Ctrl+N)")
        self.name_edit.setAccessibleName("Model name")
        self.name_edit.setMaxLength(64)
        self.create_btn = button("Create", "primary", "Create the project folder, or open it if it exists")
        row.addWidget(self.name_edit, 1)
        row.addWidget(self.create_btn)
        s.addLayout(row)
        self.name_error = label("", "error", wrap=True)
        self.name_error.hide()
        s.addWidget(self.name_error)
        self.open_combo = QComboBox()
        self.open_combo.setToolTip("Open a project you created before")
        self.open_combo.setAccessibleName("Open existing project")
        s.addWidget(self.open_combo)
        lay.addWidget(sec)

        # ---- dataset
        self.dataset_sec, s = section("Dataset", "A folder with images/ and labels/ in YOLO format. "
                                                 "Your original files are only read, never changed.")
        self.pick_btn = button("Choose dataset folder…", tooltip="Pick the folder that holds images/ and labels/ (Ctrl+O)")
        s.addWidget(self.pick_btn)
        self.path_lbl = label("No folder chosen yet.", "muted", wrap=True)
        self.path_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        s.addWidget(self.path_lbl)
        self.layout_lbl = label("", None, wrap=True)
        self.layout_lbl.hide()
        s.addWidget(self.layout_lbl)
        f = form()
        self.val_spin = spin(1, 90, 20, "Share of images used for validation when the folder is flat. "
                             "The split uses a fixed random seed, so it is the same every time.", suffix=" %")
        f.addRow("Val share", self.val_spin)
        s.addLayout(f)
        self.classes_src = label("Classes", None)
        s.addWidget(self.classes_src)
        self.classes_hint = label("", "caption", wrap=True)
        self.classes_hint.hide()
        s.addWidget(self.classes_hint)
        self.class_editor = ClassEditor()
        s.addWidget(self.class_editor)
        self.validate_btn = button("Validate and prepare", "primary",
                                   "Check every image and label, then copy the dataset into the project and write "
                                   "data.yaml (F5)")
        s.addWidget(self.validate_btn)
        self.validate_status = label("", None, wrap=True)
        self.validate_status.hide()
        s.addWidget(self.validate_status)
        lay.addWidget(self.dataset_sec)

        # ---- model
        self.model_sec, s = section("Model")
        f = form()
        self.family_combo = QComboBox()
        self.family_combo.addItems(FAMILIES)
        self.family_combo.setToolTip("YOLO generation. yolo26 is the newest; yolo11 and yolov8 are older and well tested.")
        self.size_combo = QComboBox()
        for code, word in zip(SIZES, ("nano", "small", "medium", "large", "extra large")):
            self.size_combo.addItem(f"{code}  {word}", code)
        self.size_combo.setToolTip("Bigger models are more accurate but slower and use more GPU memory. "
                                   "s is a good start on a 12 GB card.")
        f.addRow("Family", self.family_combo)
        f.addRow("Size", self.size_combo)
        s.addLayout(f)
        self.weights_lbl = label("", "caption", wrap=True)
        s.addWidget(self.weights_lbl)
        lay.addWidget(self.model_sec)

        # ---- training settings
        self.train_sec, s = section("Training")
        f = form()
        self.epochs_spin = spin(1, 3000, 100, "Number of full passes over the training images")
        self.imgsz_spin = spin(64, 2048, 640, "Images are resized to this size in pixels. Multiples of 32.",
                               step=32, suffix=" px")
        brow = QHBoxLayout()
        brow.setSpacing(SP2)
        self.batch_spin = spin(1, 512, 16, "Images per training step. Lower it if the GPU runs out of memory.")
        self.batch_auto = QCheckBox("Auto")
        self.batch_auto.setToolTip("Let Ultralytics pick the largest batch that fits in about 60 % of GPU memory")
        brow.addWidget(self.batch_spin, 1)
        brow.addWidget(self.batch_auto)
        self.workers_spin = spin(0, 32, 4, "Processes that load images in the background. "
                                 "Use 0 if training hangs before the first epoch.")
        self.patience_spin = spin(0, 1000, 50, "Stop early when validation has not improved for this many epochs. "
                                  "0 turns early stopping off.", suffix=" epochs")
        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto", "auto")
        self.device_combo.addItem("GPU 0", "0")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.setToolTip("Auto uses the NVIDIA GPU when CUDA works and falls back to the CPU otherwise")
        self.auto_onnx_check = QCheckBox("Auto-export ONNX after training")
        self.auto_onnx_check.setChecked(True)
        self.auto_onnx_check.setToolTip("When a run finishes, stops early or is stopped, export best.pt to best.onnx "
                                        "in the same weights folder. A failed export keeps the run and its weights.")
        f.addRow("Epochs", self.epochs_spin)
        f.addRow("Image size", self.imgsz_spin)
        f.addRow("Batch", brow)
        f.addRow("Workers", self.workers_spin)
        f.addRow("Patience", self.patience_spin)
        f.addRow("Device", self.device_combo)
        f.addRow("ONNX", self.auto_onnx_check)
        s.addLayout(f)
        lay.addWidget(self.train_sec)
        lay.addStretch(1)
        scroll.setWidget(inner)
        return scroll

    def _build_monitor(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(SP3 + SP1, SP3, SP3 + SP1, SP2)
        lay.setSpacing(SP2)
        self.env_banner = Banner()
        self.msg_banner = Banner()
        lay.addWidget(self.env_banner)
        lay.addWidget(self.msg_banner)

        # actions
        row = QHBoxLayout()
        row.setSpacing(SP1)
        self.start_btn = button("Start training", "primary", "Train with the settings on the left (Ctrl+Enter)")
        self.stop_btn = button("Stop training", "danger",
                               "Stop after the current batch and keep the last saved weights (Ctrl+.). "
                               "Press again to force stop.")
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        row.addStretch(1)
        self.run_combo = QComboBox()
        self.run_combo.setMinimumWidth(220)
        self.run_combo.setToolTip("Training run used by Open results, Test model and Export ONNX")
        self.run_combo.setAccessibleName("Training run")
        row.addWidget(self.run_combo)
        self.results_btn = button("Open results folder", tooltip="Open the selected run folder in Explorer (Ctrl+R)")
        self.test_btn = button("Test model", tooltip="Run the best weights on an image, a folder or a video (Ctrl+T)")
        test_menu = QMenu(self.test_btn)
        self.test_file_act = QAction("Image or video…", self)
        self.test_dir_act = QAction("Folder of images…", self)
        test_menu.addAction(self.test_file_act)
        test_menu.addAction(self.test_dir_act)
        self.test_btn.setMenu(test_menu)
        self.export_btn = button("Export ONNX", tooltip="Export the best weights to ONNX next to best.pt (Ctrl+E)")
        row.addWidget(self.results_btn)
        row.addWidget(self.test_btn)
        row.addWidget(self.export_btn)
        lay.addLayout(row)

        # progress
        prog = QVBoxLayout()
        prog.setSpacing(SP1)
        head = QHBoxLayout()
        self.phase_lbl = label("Ready to train", "display")
        self.detail_lbl = label("", "muted")
        self.detail_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        head.addWidget(self.phase_lbl)
        head.addStretch(1)
        head.addWidget(self.detail_lbl)
        prog.addLayout(head)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFixedHeight(10)
        self.progress.setAccessibleName("Training progress")
        prog.addWidget(self.progress)
        lay.addLayout(prog)

        # stats
        stats = QHBoxLayout()
        stats.setSpacing(SP1)
        self.tiles = {
            "box_loss": StatTile("Box loss", "Training box regression loss for the last epoch. Lower is better."),
            "map50": StatTile("mAP50", "Mean average precision at IoU 0.50 on the val split. Higher is better."),
            "map50_95": StatTile("mAP50-95", "mAP averaged over IoU 0.50 to 0.95. The stricter, standard score."),
            "precision": StatTile("Precision", "Share of predicted boxes that are correct"),
            "recall": StatTile("Recall", "Share of real objects that were found"),
        }
        for t in self.tiles.values():
            stats.addWidget(t)
        lay.addLayout(stats)

        # chart + tabs
        split = QSplitter(Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)
        self.chart = MetricsChart()
        split.addWidget(self.chart)
        self.tabs = QTabWidget()
        self.log = LogView()
        self.epoch_table = QTableWidget(0, 7)
        self.epoch_table.setHorizontalHeaderLabels(
            ["Epoch", "Box loss", "Cls loss", "Precision", "Recall", "mAP50", "mAP50-95"])
        self.epoch_table.verticalHeader().setVisible(False)
        self.epoch_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.epoch_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.epoch_table.setAlternatingRowColors(True)
        self.epoch_table.setAccessibleName("Metrics per epoch")
        self.report_view = QTextBrowser()
        self.report_view.setAccessibleName("Validation report")
        self.report_view.setPlaceholderText("Press Validate and prepare to check the dataset. "
                                            "The report lists images per split, boxes per class and any problems.")
        self.tabs.addTab(self.log, "Log")
        self.tabs.addTab(self.epoch_table, "Epochs")
        self.tabs.addTab(self.report_view, "Validation report")
        split.addWidget(self.tabs)
        split.setSizes([320, 360])
        lay.addWidget(split, 1)
        return w

    def _build_statusbar(self) -> None:
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        self.gpu_lbl = label("GPU: checking…")
        self.vram_lbl = label("VRAM: –")
        self.cuda_lbl = label("CUDA: checking…")
        self.gpu_lbl.setToolTip("Graphics card used for training")
        self.vram_lbl.setToolTip("GPU memory in use, from nvidia-smi (1 GB = 1024 MiB)")
        self.cuda_lbl.setToolTip("Whether PyTorch can train on the GPU")
        for w in (self.gpu_lbl, self.vram_lbl, self.cuda_lbl):
            sb.addWidget(w)
        ver = label(f"Version {__version__}")
        ver.setToolTip("LocalTrainer version from app/version.py")
        sb.addPermanentWidget(ver)

    # ============================================================ wiring
    def _connect(self) -> None:
        self.create_btn.clicked.connect(self._create_project)
        self.name_edit.returnPressed.connect(self._create_project)
        self.name_edit.textChanged.connect(self._name_typed)
        self.open_combo.activated.connect(self._open_from_combo)
        self.pick_btn.clicked.connect(self._choose_dataset)
        self.validate_btn.clicked.connect(self._validate_and_prepare)
        self.class_editor.changed.connect(self._classes_changed)
        self.val_spin.valueChanged.connect(self._dataset_setting_changed)
        for w in (self.family_combo, self.size_combo, self.device_combo):
            w.currentIndexChanged.connect(self._settings_changed)
        for w in (self.epochs_spin, self.imgsz_spin, self.batch_spin, self.workers_spin, self.patience_spin):
            w.valueChanged.connect(self._settings_changed)
        self.batch_auto.toggled.connect(self._settings_changed)
        self.auto_onnx_check.toggled.connect(self._settings_changed)
        self.start_btn.clicked.connect(self._start_training)
        self.stop_btn.clicked.connect(self._stop_training)
        self.results_btn.clicked.connect(self._open_results)
        self.test_file_act.triggered.connect(lambda: self._test_model(folder=False))
        self.test_dir_act.triggered.connect(lambda: self._test_model(folder=True))
        self.export_btn.clicked.connect(self._export_onnx)
        self.run_combo.currentIndexChanged.connect(lambda *_: self._update_actions())
        self.theme_btn.clicked.connect(lambda: self.apply_theme("light" if self.theme == "dark" else "dark"))
        self.update_btn.clicked.connect(lambda: self._check_updates(silent=False))

        self.session.log.connect(self.log.append_line)
        self.session.started.connect(self._on_train_started)
        self.session.batch_progress.connect(self._on_batch)
        self.session.epoch_metrics.connect(self._on_epoch)
        self.session.stopping.connect(lambda: self.phase_lbl.setText("Stopping after this batch…"))
        self.session.exporting.connect(self._on_train_exporting)
        self.session.finished.connect(self._on_train_finished)
        self.session.failed.connect(self._on_train_failed)

        self.job.log.connect(self.log.append_line)
        self.job.message.connect(self._on_job_event)
        self.job.finished.connect(self._on_job_finished)
        self.env_job.message.connect(self._on_env)
        self.env_job.finished.connect(lambda code, _k: self._on_env_finished(code))
        self.gpu.sample.connect(self._on_gpu_sample)
        self.gpu.unavailable.connect(lambda: self.vram_lbl.setText("VRAM: nvidia-smi not found"))

    def _shortcuts(self) -> None:
        def sc(keys: str, fn: Any) -> None:
            act = QAction(self)
            act.setShortcut(QKeySequence(keys))
            act.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            act.triggered.connect(fn)
            self.addAction(act)

        sc("Ctrl+N", lambda: (self.name_edit.setFocus(), self.name_edit.selectAll()))
        sc("Ctrl+O", lambda: self.pick_btn.isEnabled() and self._choose_dataset())
        sc("F5", lambda: self.validate_btn.isEnabled() and self._validate_and_prepare())
        sc("Ctrl+Return", lambda: self.start_btn.isEnabled() and self._start_training())
        sc("Ctrl+Enter", lambda: self.start_btn.isEnabled() and self._start_training())
        sc("Ctrl+.", lambda: self.stop_btn.isEnabled() and self._stop_training())
        sc("Ctrl+R", lambda: self.results_btn.isEnabled() and self._open_results())
        sc("Ctrl+T", lambda: self.test_btn.isEnabled() and self.test_btn.showMenu())
        sc("Ctrl+E", lambda: self.export_btn.isEnabled() and self._export_onnx())
        sc("Ctrl+U", lambda: self._check_updates(silent=False))
        sc("Ctrl+Shift+L", lambda: self.apply_theme("light" if self.theme == "dark" else "dark"))

    # ============================================================ theme
    def apply_theme(self, theme: str) -> None:
        self.theme = theme
        app = QApplication.instance()
        if isinstance(app, QApplication):
            app.setStyleSheet(build_qss(theme, self.fonts))
        self.chart.set_theme(theme, self.fonts.display)
        self.theme_btn.setText("Light theme" if theme == "dark" else "Dark theme")
        self.prefs["theme"] = theme
        save_prefs(self.prefs)

    # ============================================================ startup
    def _startup(self) -> None:
        self.gpu.start()
        self.env_job.start("envcheck", {})
        self._show_update_status()
        threading.Thread(target=cleanup_work_dirs, name="update-cleanup", daemon=True).start()
        if not self.prefs.get("desktop_shortcut") and not os.environ.get("LOCALTRAINER_SMOKE"):
            # once per install: a shortcut you deleted stays deleted (setup.bat makes it again)
            self.prefs["desktop_shortcut"] = True
            save_prefs(self.prefs)
            threading.Thread(target=create_desktop_shortcut, name="desktop-shortcut", daemon=True).start()
        last = self.prefs.get("last_project")
        if last and (PROJECTS_DIR / last).is_dir():
            self._open_project(last)
        QTimer.singleShot(1500, lambda: self._check_updates(silent=True))

    def _show_update_status(self) -> None:
        if not UPDATE_STATUS_FILE.is_file():
            return
        try:
            status = json.loads(UPDATE_STATUS_FILE.read_text(encoding="utf-8"))
            UPDATE_STATUS_FILE.unlink()
        except (OSError, ValueError):
            return
        if status.get("ok"):
            self.msg_banner.show_message("success", f"Updated to version {status.get('version')}",
                                         "Your projects and settings were kept.")
        else:
            self.msg_banner.show_message("error", "The last update was rolled back", str(status.get("message", "")))

    def _on_env(self, ev: dict[str, Any]) -> None:
        if ev.get("type") != "env":
            return
        self.env = ev
        if ev.get("cuda_available") and not ev.get("problem"):
            self.gpu_lbl.setText(f"GPU: {ev.get('device_name')}")
            self.cuda_lbl.setText(f"CUDA {ev.get('cuda_build')} ready · torch {ev.get('torch')}")
            self.cuda_lbl.setProperty("tone", "success")
            self.env_banner.hide()
        else:
            self.gpu_lbl.setText(f"GPU: {ev.get('device_name') or 'not available'}")
            self.cuda_lbl.setText("CUDA: not available, training uses the CPU")
            self.cuda_lbl.setProperty("tone", "warning")
            fix = ev.get("fix") or FIX_COMMAND_VENV
            self.env_banner.show_message(
                "warning", "Training will run on the CPU, which is much slower",
                f"{ev.get('problem') or 'CUDA is not available.'}\nTo fix it, run this in the LocalTrainer folder:\n{fix}",
                [("Copy command", lambda: QGuiApplication.clipboard().setText(fix))])
        repolish(self.cuda_lbl)

    def _on_env_finished(self, code: int) -> None:
        if self.env is None:
            self.env = {"cuda_available": False}
            self.cuda_lbl.setText("CUDA: check failed")
            self.env_banner.show_message("warning", "The CUDA check could not run",
                                         f"Training falls back to the CPU. To repair PyTorch run:\n{FIX_COMMAND_VENV}",
                                         [("Copy command", lambda: QGuiApplication.clipboard().setText(FIX_COMMAND_VENV))])

    def _on_gpu_sample(self, name: str, used: int, total: int, util: int) -> None:
        self.vram_lbl.setText(f"VRAM {used / 1024:.1f} of {total / 1024:.1f} GB · load {util} %")
        if self.env is None or not self.env.get("device_name"):
            self.gpu_lbl.setText(f"GPU: {name}")

    # ============================================================ project
    def _refresh_project_list(self) -> None:
        current = self.project.name if self.project else None
        self.open_combo.blockSignals(True)
        self.open_combo.clear()
        names = list_projects()
        self.open_combo.addItem("Open an existing project…" if names else "No projects yet", "")
        for n in names:
            self.open_combo.addItem(n, n)
        if current in names:
            self.open_combo.setCurrentIndex(names.index(current) + 1)
        self.open_combo.setEnabled(bool(names))
        self.open_combo.blockSignals(False)

    def _name_typed(self, text: str) -> None:
        err = validate_name(text) if text else None
        self.name_edit.setProperty("invalid", bool(err))
        repolish(self.name_edit)
        self.name_error.setText(err or "")
        self.name_error.setVisible(bool(err))

    def _create_project(self) -> None:
        if self.session.is_running():
            return
        name = self.name_edit.text().strip()
        err = validate_name(name)
        if err:
            self._name_typed(name or " ")
            self.name_error.setText(err)
            self.name_error.show()
            self.name_edit.setFocus()
            return
        project = Project(name)
        if project.exists():
            box = QMessageBox(QMessageBox.Icon.Question, "Project exists",
                              f"A project called {name} already exists.", parent=self)
            open_btn = box.addButton("Open it", QMessageBox.ButtonRole.AcceptRole)
            box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(open_btn)
            box.exec()
            if box.clickedButton() is not open_btn:
                return
        else:
            project.create()
        self._open_project(name)

    def _open_from_combo(self, index: int) -> None:
        name = self.open_combo.itemData(index)
        if name and not self.session.is_running():
            self._open_project(name)

    def _open_project(self, name: str) -> None:
        project = Project(name)
        project.create()
        self._set_project(project)
        self.prefs["last_project"] = name
        save_prefs(self.prefs)

    def _set_project(self, project: Project | None) -> None:
        self.project = project
        self.info = None
        self._loading = True
        has = project is not None
        for w in (self.dataset_sec, self.model_sec, self.train_sec):
            w.setVisible(has)  # progressive disclosure: settings appear once a project exists
        self.right.setCurrentIndex(1 if has else 0)
        if project is None:
            self.state = None
            self.project_lbl.setText("")
            self._loading = False
            return
        self.state = project.load_state()
        self.project_lbl.setText(project.name)
        self.name_edit.setText(project.name)
        self.setWindowTitle(f"{project.name} · LocalTrainer {__version__}")
        st = self.state
        self._load_settings(st.train)
        self.val_spin.setValue(int(round(st.val_ratio * 100)))
        self.class_editor.set_names(st.class_names)
        self.report_view.clear()
        self.validate_status.hide()
        self.classes_hint.hide()
        self.layout_lbl.hide()
        self.path_lbl.setText(st.dataset_source or "No folder chosen yet.")
        if st.dataset_source:
            try:
                self.info = ds.detect_layout(Path(st.dataset_source))
                self._show_layout()
            except ds.DatasetError:
                self.info = None
            if st.prepared and project.data_yaml.is_file():
                self._set_validate_status("success", "Prepared. data.yaml is ready, you can start training.")
            elif st.dataset_source and self.info is None:
                self._set_validate_status("warning", "The original dataset folder is no longer available. "
                                                     "Choose it again to re-validate.")
        self._reset_monitor()
        self._refresh_project_list()
        self._refresh_runs()
        self._loading = False
        self._update_actions()

    # ============================================================ settings
    def _load_settings(self, s: TrainSettings) -> None:
        self.family_combo.setCurrentText(s.family)
        self.size_combo.setCurrentIndex(max(0, SIZES.index(s.size) if s.size in SIZES else 1))
        self.epochs_spin.setValue(s.epochs)
        self.imgsz_spin.setValue(s.imgsz)
        self.batch_spin.setValue(s.batch)
        self.batch_auto.setChecked(s.batch_auto)
        self.workers_spin.setValue(s.workers)
        self.patience_spin.setValue(s.patience)
        idx = self.device_combo.findData(s.device)
        self.device_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.auto_onnx_check.setChecked(s.auto_export_onnx)
        self._update_weights_label()

    def _read_settings(self) -> TrainSettings:
        return TrainSettings(
            family=self.family_combo.currentText(), size=self.size_combo.currentData(),
            epochs=self.epochs_spin.value(), imgsz=self.imgsz_spin.value(), batch=self.batch_spin.value(),
            batch_auto=self.batch_auto.isChecked(), workers=self.workers_spin.value(),
            patience=self.patience_spin.value(), device=self.device_combo.currentData(),
            auto_export_onnx=self.auto_onnx_check.isChecked(),
        )

    def _update_weights_label(self) -> None:
        s = self._read_settings()
        self.batch_spin.setEnabled(not s.batch_auto)
        self.weights_lbl.setText(f"Starts from {s.weights}. Downloaded once into weights/ on first use.")

    def _settings_changed(self, *_: Any) -> None:
        self._update_weights_label()
        if self._loading or self.project is None or self.state is None:
            return
        self.state.train = self._read_settings()
        self.project.save_state(self.state)

    def _dataset_setting_changed(self, *_: Any) -> None:
        if self._loading or self.project is None or self.state is None:
            return
        self.state.val_ratio = self.val_spin.value() / 100
        if self.info is not None and self.info.layout is ds.Layout.FLAT:
            self._invalidate()
        self.project.save_state(self.state)

    def _classes_changed(self) -> None:
        if self._loading or self.project is None or self.state is None:
            return
        self.state.class_names = self.class_editor.names()
        self._invalidate()
        self.project.save_state(self.state)

    def _invalidate(self) -> None:
        if self.state is not None and self.state.prepared:
            self.state.prepared = False
            self._set_validate_status("warning", "Settings changed. Validate and prepare again before training.")
        self._update_actions()

    # ============================================================ dataset
    def _choose_dataset(self) -> None:
        if self.project is None or self.session.is_running():
            return
        start = self.state.dataset_source if self.state and self.state.dataset_source else str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Choose the dataset folder", start)
        if not folder:
            return
        try:
            info = ds.detect_layout(Path(folder))
        except ds.DatasetError as exc:
            self.path_lbl.setText(folder)
            self._set_validate_status("error", str(exc))
            return
        self.info = info
        assert self.state is not None and self.project is not None
        self.state.dataset_source = str(Path(folder))
        self.state.layout = info.layout.value
        self.state.prepared = False
        self.path_lbl.setText(self.state.dataset_source)
        self._show_layout()
        self.validate_status.hide()
        found = ds.read_class_names(info.root)
        if found:
            names, source = found
            self.classes_src.setText(f"Classes from {source}")
            self.classes_hint.hide()
            self.class_editor.set_names(names)
        else:
            self._run_task(lambda _p: ds.infer_class_count(info), self._fill_inferred_classes)
        self.state.class_names = self.class_editor.names()
        self.project.save_state(self.state)
        self._update_actions()

    def _fill_inferred_classes(self, count: object) -> None:
        n = max(1, int(count))  # type: ignore[arg-type]
        self.classes_src.setText("Classes")
        self.classes_hint.setText(f"No classes.txt or data.yaml found. The labels use {n} class id(s). "
                                  "Type a name for each id, in order.")
        self.classes_hint.show()
        self.class_editor.set_names([f"class_{i}" for i in range(n)])
        self.class_editor.table.setFocus()
        self.class_editor.table.setCurrentCell(0, 1)

    def _show_layout(self) -> None:
        if self.info is None:
            return
        self.layout_lbl.setText(self.info.describe())
        self.layout_lbl.show()
        self.val_spin.setEnabled(self.info.layout is ds.Layout.FLAT)

    def _set_validate_status(self, tone: str, text: str) -> None:
        self.validate_status.setProperty("tone", tone)
        self.validate_status.setText(text)
        self.validate_status.show()
        repolish(self.validate_status)

    def _validate_and_prepare(self) -> None:
        if self.project is None or self.state is None or self.session.is_running():
            return
        if self.info is None:
            self._set_validate_status("error", "Choose a dataset folder first.")
            return
        names = self.class_editor.names()
        if not names or any(not n for n in names):
            self._set_validate_status("error", "Every class needs a name.")
            return
        info, project, ratio = self.info, self.project, self.val_spin.value() / 100

        def work(progress: Any) -> tuple[ds.ValidationReport, dict[str, int] | None]:
            rep = ds.validate(info, len(names))
            if not rep.ok:
                return rep, None
            counts = ds.prepare_dataset(info, project.dataset_dir, ratio, progress=progress)
            write_data_yaml(project.data_yaml, project.dataset_dir, names)
            return rep, counts

        self._set_busy_dataset(True)
        self._set_validate_status("info", "Checking images and labels…")
        task = self._run_task(work, lambda res: self._on_validated(res, names),
                              lambda exc: (self._set_busy_dataset(False),
                                           self._set_validate_status("error", f"Preparing failed: {exc}")))
        task.progress.connect(lambda d, t: self._set_validate_status("info", f"Copying files {d} of {t}…"))

    def _on_validated(self, result: object, names: list[str]) -> None:
        rep, counts = result  # type: ignore[misc]
        self._set_busy_dataset(False)
        self.report_view.setMarkdown(ds.report_text(rep, names))
        self.tabs.setCurrentWidget(self.report_view)
        assert self.state is not None and self.project is not None
        if rep.ok and counts is not None:
            self.state.prepared = True
            self.state.class_names = names
            extra = f" {len(rep.warnings)} warning(s), see the report." if rep.warnings else ""
            self._set_validate_status("success", f"Ready: {counts['train']} train and {counts['val']} val images "
                                                 f"copied, data.yaml written.{extra}")
        else:
            self.state.prepared = False
            self._set_validate_status("error", f"{len(rep.errors)} problem(s) block training. See the validation report.")
        self.project.save_state(self.state)
        self._update_actions()

    def _set_busy_dataset(self, busy: bool) -> None:
        for w in (self.pick_btn, self.validate_btn, self.class_editor, self.val_spin, self.start_btn):
            w.setEnabled(not busy)
        if not busy:
            self._show_layout()
            self._update_actions()

    def _run_task(self, fn: Any, on_done: Any, on_error: Any = None) -> Task:
        task = Task(fn, self)
        self._tasks.add(task)
        task.succeeded.connect(on_done)
        task.failed.connect(on_error or (lambda exc: self.msg_banner.show_message("error", "Something went wrong", str(exc))))
        task.finished.connect(lambda: self._tasks.discard(task))
        task.start()
        return task

    # ============================================================ runs
    def _refresh_runs(self, select: Path | None = None) -> None:
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        runs = self.project.list_runs() if self.project else []
        for r in runs:
            weights = find_weights(r)
            suffix = "" if weights else "  (no weights)"
            self.run_combo.addItem(r.name + suffix, str(r))
        if not runs:
            self.run_combo.addItem("No runs yet", "")
        if select is not None:
            idx = self.run_combo.findData(str(select))
            if idx >= 0:
                self.run_combo.setCurrentIndex(idx)
        else:
            latest = self.project.latest_trained_run() if self.project else None
            idx = self.run_combo.findData(str(latest)) if latest else 0
            self.run_combo.setCurrentIndex(max(idx, 0))
        self.run_combo.setEnabled(bool(runs))
        self.run_combo.blockSignals(False)
        self._update_actions()

    def _selected_run(self) -> Path | None:
        data = self.run_combo.currentData()
        return Path(data) if data else None

    def _update_actions(self) -> None:
        training = self.session.is_running()
        job = self.job.is_running()
        prepared = bool(self.project and self.state and self.state.prepared and self.project.data_yaml.is_file())
        run = self._selected_run()
        has_weights = bool(run and find_weights(run))
        self.start_btn.setEnabled(prepared and not training)
        self.start_btn.setToolTip("Train with the settings on the left (Ctrl+Enter)" if prepared else
                                  "Validate and prepare the dataset first")
        self.stop_btn.setEnabled(training)
        self.results_btn.setEnabled(bool(self.project))
        self.test_btn.setEnabled(has_weights and not job)
        self.export_btn.setEnabled(has_weights and not job)
        for w in (self.model_sec, self.train_sec, self.pick_btn, self.validate_btn, self.class_editor,
                  self.create_btn, self.name_edit, self.open_combo):
            w.setEnabled(not training and (bool(self.project) or w in (self.create_btn, self.name_edit, self.open_combo)))
        if self.info is None or self.info.layout is not ds.Layout.FLAT:
            self.val_spin.setEnabled(False)
        elif not training:
            self.val_spin.setEnabled(True)

    # ============================================================ training
    def _reset_monitor(self, total_epochs: int = 0) -> None:
        self.chart.clear(total_epochs)
        self.epoch_table.setRowCount(0)
        for t in self.tiles.values():
            t.set_value("–")
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.phase_lbl.setText("Ready to train")
        self.detail_lbl.setText("")

    def _start_training(self) -> None:
        if self.project is None or self.state is None or self.session.is_running():
            return
        if not self.state.prepared or not self.project.data_yaml.is_file():
            self._set_validate_status("error", "Validate and prepare the dataset before training.")
            return
        s = self._read_settings()
        self.state.train = s
        self.project.save_state(self.state)
        cuda_ok = bool(self.env and self.env.get("cuda_available") and not self.env.get("problem"))
        device = resolve_device(s.device, cuda_ok)
        if device == "cpu" and s.device != "cpu":
            self.msg_banner.show_message("warning", "No usable CUDA GPU, training on the CPU",
                                         "This works but is much slower. See the message above for the fix.")
        else:
            self.msg_banner.hide()
        cfg = TrainConfig(
            data_yaml=str(self.project.data_yaml), weights=s.weights, epochs=s.epochs, imgsz=s.imgsz,
            batch=-1 if s.batch_auto else s.batch, workers=s.workers, patience=s.patience, device=device,
            runs_dir=str(self.project.runs_dir), run_name=make_run_name(), auto_export_onnx=s.auto_export_onnx,
        )
        self._current_run = cfg.run_dir
        self._reset_monitor(s.epochs)
        self.log.clear()
        self.log.append_line(f"Starting {cfg.run_name}: {s.weights}, {s.epochs} epochs, {s.imgsz} px, "
                             f"batch {'auto' if s.batch_auto else s.batch}, device {device}")
        self.tabs.setCurrentWidget(self.log)
        self.phase_lbl.setText("Preparing…")
        self.detail_lbl.setText("Loading the model and scanning labels")
        self.progress.setRange(0, 0)  # busy until the first batch
        self.stop_btn.setText("Stop training")
        self._run_started_at = time.monotonic()
        self.session.start(cfg)
        self._update_actions()

    def _stop_training(self) -> None:
        if not self.session.is_running():
            return
        exporting = self.session.is_exporting()
        forced = self.session.request_stop()
        if exporting:
            self.phase_lbl.setText("Cancelling the ONNX export…")
        elif forced:
            self.phase_lbl.setText("Force stopping…")
        else:
            self.phase_lbl.setText("Stopping after this batch…")
            self.detail_lbl.setText("Saving the last weights. Press again to force stop.")
            self.stop_btn.setText("Force stop")

    def _on_train_started(self, ev: dict[str, Any]) -> None:
        self.phase_lbl.setText("Starting the first epoch…")
        self.detail_lbl.setText(f"Batch size {ev.get('batch')} · {ev.get('batches_per_epoch')} batches per epoch")
        self._run_started_at = time.monotonic()

    def _on_batch(self, epoch: int, epochs: int, i: int, nb: int) -> None:
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 1000)
        if nb:
            frac = min(1.0, ((epoch - 1) + min(i, nb) / nb) / max(epochs, 1))
            self.progress.setValue(int(frac * 1000))
            elapsed = time.monotonic() - self._run_started_at
            left = elapsed / frac - elapsed if frac > 0.01 else None
            eta = f" · about {fmt_duration(left)} left" if left is not None else ""
            self.detail_lbl.setText(f"{fmt_duration(elapsed)} elapsed{eta}")
        if "Stopping" not in self.phase_lbl.text():
            self.phase_lbl.setText(f"Epoch {epoch} of {epochs}")

    def _on_epoch(self, ev: dict[str, Any]) -> None:
        final = ev.get("type") == "final_metrics"
        for key, tile in self.tiles.items():
            v = ev.get(key)
            if v is not None:
                tile.set_value(fmt_metric(v, 3))
        if final:
            self.phase_lbl.setText("Final validation with best.pt")
            return
        epoch = int(ev.get("epoch", 0))
        self.chart.add_point(epoch, ev.get("map50"), ev.get("map50_95"))
        r = self.epoch_table.rowCount()
        self.epoch_table.insertRow(r)
        values = [str(epoch)] + [fmt_metric(ev.get(k), 4) for k in
                                 ("box_loss", "cls_loss", "precision", "recall", "map50", "map50_95")]
        for c, v in enumerate(values):
            item = QTableWidgetItem(v)
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.epoch_table.setItem(r, c, item)
        self.epoch_table.scrollToBottom()
        epochs = int(ev.get("epochs", 0))
        if epochs:
            self.progress.setValue(int(epoch / epochs * 1000))

    def _on_train_exporting(self) -> None:
        self.progress.setRange(0, 0)
        self.phase_lbl.setText("Exporting to ONNX…")
        self.detail_lbl.setText("Training is saved. Stop cancels only the export.")
        self.stop_btn.setText("Cancel export")

    def _on_train_finished(self, res: dict[str, Any]) -> None:
        self.progress.setRange(0, 1000)
        save_dir = Path(res.get("save_dir") or (self._current_run or ""))
        best, last = res.get("best"), res.get("last")
        elapsed = fmt_duration(time.monotonic() - self._run_started_at)
        actions = [("Open results folder", lambda: self._open_path(save_dir))]
        onnx, onnx_error = res.get("onnx"), res.get("onnx_error")
        onnx_line = f"\nONNX: {onnx} ({fmt_bytes(float(res.get('onnx_size') or 0))})" if onnx else ""
        if onnx_error:
            onnx_line = (f"\nONNX export failed: {str(onnx_error).rstrip('.')}. "
                         "The weights are fine; press Export ONNX to try again.")
        if res.get("stopped"):
            self.phase_lbl.setText("Training stopped")
            kept = Path(best or last).name if (best or last) else None
            body = (f"Kept the last saved weights ({kept}) in {save_dir}." if kept else
                    "It stopped before the first epoch finished, so there are no weights to keep.")
            kind = "warning" if onnx_error else "info"
            self.msg_banner.show_message(kind, "Training stopped", body + onnx_line, actions if kept else None)
        else:
            self.progress.setValue(1000)
            self.phase_lbl.setText("Training finished")
            title = f"Training finished in {elapsed}"
            if onnx_error:
                title += ", ONNX export failed"
            self.msg_banner.show_message("warning" if onnx_error else "success", title,
                                         f"Best weights: {best or 'not found'}{onnx_line}", actions)
        self.detail_lbl.setText(save_dir.name)
        self.stop_btn.setText("Stop training")
        self._refresh_runs(select=save_dir if save_dir.is_dir() else None)

    def _on_train_failed(self, kind: str, message: str) -> None:
        self.progress.setRange(0, 1000)
        self.stop_btn.setText("Stop training")
        self.phase_lbl.setText("Training failed")
        self.detail_lbl.setText("")
        if kind == "oom":
            s = self._read_settings()
            tips = f"Current settings: batch {'auto' if s.batch_auto else s.batch}, image size {s.imgsz} px."
            self.msg_banner.show_message("error", "Out of GPU memory", f"{OOM_HINT}\n{tips}",
                                         [("Halve the batch", self._halve_batch)])
            QMessageBox.warning(self, "Out of GPU memory", f"{OOM_HINT}\n\n{tips}")
        else:
            self.msg_banner.show_message("error", "Training failed", message[:1500])
        self.log.append_line(message)
        self._refresh_runs()

    def _halve_batch(self) -> None:
        self.batch_auto.setChecked(False)
        self.batch_spin.setValue(max(1, self.batch_spin.value() // 2))
        self.msg_banner.hide()

    # ============================================================ results / test / export
    def _open_path(self, path: Path) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_results(self) -> None:
        if self.project is None:
            return
        run = self._selected_run()
        self._open_path(run if run and run.is_dir() else self.project.runs_dir)

    def _job_payload_base(self) -> dict[str, Any] | None:
        run = self._selected_run()
        weights = find_weights(run) if run else None
        if run is None or weights is None:
            self.msg_banner.show_message("warning", "No trained weights yet", "Train a model first, or pick another run.")
            return None
        cuda_ok = bool(self.env and self.env.get("cuda_available") and not self.env.get("problem"))
        imgsz = self.state.train.imgsz if self.state else 640
        return {"weights": str(weights), "run_dir": str(run), "device": "0" if cuda_ok else "cpu", "imgsz": imgsz}

    def _test_model(self, folder: bool) -> None:
        payload = self._job_payload_base()
        if payload is None or self.job.is_running():
            return
        if folder:
            src = QFileDialog.getExistingDirectory(self, "Choose a folder of images")
        else:
            src, _ = QFileDialog.getOpenFileName(
                self, "Choose an image or video", "",
                "Images and videos (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.mp4 *.avi *.mov *.mkv *.wmv *.webm);;"
                "All files (*)")
        if not src:
            return
        payload["source"] = src
        self._job_kind = "predict"
        self.log.append_line(f"Testing {Path(payload['weights']).name} on {src}")
        self.tabs.setCurrentWidget(self.log)
        self.msg_banner.show_message("info", "Testing the model…", "The annotated output opens when it is ready.")
        self.job.start("predict", payload)
        self._update_actions()

    def _export_onnx(self) -> None:
        payload = self._job_payload_base()
        if payload is None or self.job.is_running():
            return
        self._job_kind = "export"
        self.log.append_line(f"Exporting {payload['weights']} to ONNX")
        self.tabs.setCurrentWidget(self.log)
        self.msg_banner.show_message("info", "Exporting to ONNX…", "This takes a few seconds to a minute.")
        self.job.start("export", payload)
        self._update_actions()

    def _on_job_event(self, ev: dict[str, Any]) -> None:
        kind = ev.get("type")
        if kind == "log":
            self.log.append_line(str(ev.get("message", "")))
        elif kind == "done" and getattr(self, "_job_kind", "") == "predict":
            target = Path(ev.get("open") or ev.get("save_dir", ""))
            self.msg_banner.show_message("success", f"Tested on {ev.get('count', 0)} image(s) or frame(s)",
                                         f"Annotated output: {ev.get('save_dir')}",
                                         [("Open output", lambda: self._open_path(target))])
            self._open_path(target)
        elif kind == "done":
            out = Path(str(ev.get("path", "")))
            size = f" ({fmt_bytes(float(ev['size']))})" if ev.get("size") else ""
            self.log.append_line(f"Exported ONNX: {out}{size}")
            self.msg_banner.show_message("success", "Exported to ONNX", f"{out}{size}",
                                         [("Show folder", lambda: self._open_path(out.parent))])
        elif kind == "error":
            title = "Testing failed" if getattr(self, "_job_kind", "") == "predict" else "Export failed"
            self.msg_banner.show_message("error", title, str(ev.get("message", ""))[:1500])

    def _on_job_finished(self, code: int, killed: bool) -> None:
        if code != 0 and not killed and not self.msg_banner.isVisible():
            self.msg_banner.show_message("error", "The job stopped unexpectedly", "See the log for details.")
        self._update_actions()

    # ============================================================ updates
    def _check_updates(self, silent: bool) -> None:
        if not silent:
            self.update_btn.setEnabled(False)
            self.update_btn.setText("Checking…")

        def done(info: object) -> None:
            self.update_btn.setEnabled(True)
            assert isinstance(info, ReleaseInfo)
            newer = is_newer(info.version)
            self.release = info if newer else None
            self._set_update_badge(newer, info.version)
            if not silent:
                if newer:
                    UpdateDialog(info, can_update=not self.session.is_running(), parent=self).exec()
                else:
                    QMessageBox.information(self, "No update", f"You have the latest version ({__version__}).")

        def failed(exc: object) -> None:
            self.update_btn.setEnabled(True)
            self._set_update_badge(self.release is not None, self.release.version if self.release else "")
            if not silent:
                msg = str(exc) if isinstance(exc, UpdateError) else f"The update check failed: {exc}"
                QMessageBox.warning(self, "Update check", msg)

        if not silent and self.release is not None:  # already known: open straight away
            self.update_btn.setEnabled(True)
            self._set_update_badge(True, self.release.version)
            UpdateDialog(self.release, can_update=not self.session.is_running(), parent=self).exec()
            return
        self._run_task(lambda _p: fetch_latest_release(), done, failed)

    def _set_update_badge(self, available: bool, version: str) -> None:
        self.update_btn.setText(f"Update to {version}" if available else "Check for updates")
        self.update_btn.setProperty("available", "true" if available else "false")
        self.update_btn.set_badge(available)
        self.update_btn.setToolTip(f"Version {version} is available. Click to see what changed (Ctrl+U)"
                                   if available else "Look for a newer LocalTrainer release on GitHub (Ctrl+U)")
        repolish(self.update_btn)

    # ============================================================ close
    def closeEvent(self, ev: QCloseEvent) -> None:
        app = QApplication.instance()
        updating = bool(app and app.property("updating"))
        if self.session.is_running() and not updating:
            answer = QMessageBox.question(
                self, "Training is running",
                "Quit and stop training? Weights saved so far are kept in the run folder.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                ev.ignore()
                return
        self.session.kill()
        self.job.kill()
        self.env_job.kill()
        self.gpu.stop()
        for t in list(self._tasks):
            t.wait(2000)
        self.prefs["geometry"] = bytes(self.saveGeometry().toBase64()).decode()
        save_prefs(self.prefs)
        ev.accept()
