"""Beginner UI: tooltips and info labels, collapsible sections, presets, the help window.

Runs headless (QT_QPA_PLATFORM=offscreen). Nothing here touches the real projects/ folder or settings.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["LOCALTRAINER_REDUCED_MOTION"] = "1"  # sections open instantly, so tests need no event loop

QtWidgets = pytest.importorskip("PyQt6.QtWidgets", reason="Qt widgets cannot load on this machine")
from PyQt6.QtCore import QPoint, Qt, QUrl  # noqa: E402
from PyQt6.QtGui import QKeySequence  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QPushButton, QStyle  # noqa: E402

from core import dataset as ds  # noqa: E402
from core.project import Project, TrainSettings  # noqa: E402
from ui import main_window as mw  # noqa: E402
from ui.help_texts import BUTTONS, CUSTOM, HELP_SECTIONS, METRICS, PRESETS, SECTIONS, SETTINGS, fixes_for  # noqa: E402
from ui.theme import AppStyle, resolve_fonts  # noqa: E402
from ui.widgets import CollapsibleSection, ReportView, WrapRow  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    if not isinstance(app, QApplication):
        pytest.skip("another test created a QCoreApplication; widgets need a QApplication")
    app.setStyle(AppStyle())
    return app


@pytest.fixture
def saved(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Capture save_prefs calls instead of writing settings.json."""
    calls: list[dict] = []
    monkeypatch.setattr(mw, "save_prefs", lambda prefs: calls.append(json.loads(json.dumps(prefs))))
    monkeypatch.setattr(mw, "list_projects", lambda *a, **k: [])
    monkeypatch.setattr(mw.MainWindow, "_startup", lambda self: None)  # no GPU poll, env check or update check
    return calls


@pytest.fixture
def make_window(qapp: QApplication, saved: list[dict]):
    windows = []

    def make(prefs: dict | None = None) -> mw.MainWindow:
        w = mw.MainWindow(resolve_fonts(), prefs if prefs is not None else {})
        windows.append(w)
        return w

    yield make
    for w in windows:
        w.close()
        w.deleteLater()
    qapp.processEvents()


@pytest.fixture
def project(tmp_path: Path) -> Project:
    p = Project("demo", base=tmp_path)
    p.create()
    return p


def open_project(win: mw.MainWindow, project: Project) -> None:
    win._set_project(project)


# ------------------------------------------------------------------ tooltips and info labels
def test_every_setting_has_tooltip_and_info_label(make_window) -> None:
    win = make_window()
    assert set(win.setting_widgets) == set(SETTINGS)
    for key, widget in win.setting_widgets.items():
        assert widget.toolTip().strip(), key
        info = win.info_labels[key]
        assert info.buddy is widget
        assert info.label.text() == SETTINGS[key]["label"]
        assert info.info.toolTip().strip(), key
        assert info.label.toolTip().strip(), key
        assert widget.accessibleDescription().strip(), key


def test_setting_tips_are_short_and_beginner_sized() -> None:
    for key, entry in SETTINGS.items():
        words = len(entry["tip"].split())
        assert 10 <= words < 60, (key, words)


def test_metric_tiles_chart_and_buttons_have_tooltips(make_window, project: Project) -> None:
    win = make_window()
    open_project(win, project)
    for key, tile in win.tiles.items():
        assert tile.toolTip().strip(), key
        assert key in METRICS
    assert win.chart.toolTip().strip()
    for b in win.findChildren(QPushButton):
        if b.text() and b.isVisibleTo(win):
            assert b.toolTip().strip(), b.text()


def test_banner_actions_get_tooltips(make_window) -> None:
    win = make_window()
    win.msg_banner.show_message("error", "Training failed", "boom", [("Help", lambda: None)])
    helps = [b for b in win.msg_banner.findChildren(QPushButton) if b.text() == "Help"]
    assert helps and helps[0].toolTip().strip()


def test_tooltip_delay_is_400_ms(qapp: QApplication) -> None:
    assert qapp.style().styleHint(QStyle.StyleHint.SH_ToolTip_WakeUpDelay) == 400


# ------------------------------------------------------------------ collapsible sections
def test_sections_start_collapsed_on_fresh_install(make_window) -> None:
    win = make_window({})
    assert [sid for sid, _t, _k in SECTIONS] == list(win.sections)
    for sec in win.sections.values():
        assert not sec.is_expanded()
        assert sec.content.isHidden()
        assert "collapsed" in sec.header.accessibleName()
        assert sec.header.text().startswith("+")
    assert win.show_all_btn.text() == "Show all"


def test_section_state_persists_in_app_prefs(make_window, saved: list[dict]) -> None:
    prefs: dict = {}
    win = make_window(prefs)
    win.sections["training"].header.click()
    assert win.sections["training"].is_expanded()
    assert "expanded" in win.sections["training"].header.accessibleName()
    assert win.sections["training"].header.text().startswith("−")
    assert saved and saved[-1]["sections"]["training"] is True
    again = make_window(saved[-1])
    assert again.sections["training"].is_expanded()
    assert not again.sections["hardware"].is_expanded()


def test_section_state_never_goes_into_project_settings(make_window, project: Project) -> None:
    win = make_window()
    open_project(win, project)
    before = project.settings_file.read_bytes() if project.settings_file.exists() else b""
    win.sections["data"].header.click()
    after = project.settings_file.read_bytes() if project.settings_file.exists() else b""
    assert before == after
    assert "sections" not in after.decode("utf-8")


def test_show_all_and_hide_all(make_window) -> None:
    win = make_window()
    win.show_all_btn.click()
    assert all(s.is_expanded() for s in win.sections.values())
    assert win.show_all_btn.text() == "Hide all"
    win.show_all_btn.click()
    assert not any(s.is_expanded() for s in win.sections.values())
    assert win.show_all_btn.text() == "Show all"


def test_section_header_toggles_from_keyboard(qapp: QApplication) -> None:
    sec = CollapsibleSection("x", "Example settings")
    assert sec.header.focusPolicy() & Qt.FocusPolicy.TabFocus
    QTest.keyClick(sec.header, Qt.Key.Key_Space)
    assert sec.is_expanded() and not sec.content.isHidden()
    QTest.keyClick(sec.header, Qt.Key.Key_Return)
    assert not sec.is_expanded() and sec.content.isHidden()
    QTest.keyClick(sec.header, Qt.Key.Key_Enter)
    assert sec.is_expanded()


# ------------------------------------------------------------------ modified dot and reset
def test_modified_dot_appears_and_clears(make_window, project: Project) -> None:
    win = make_window()
    open_project(win, project)
    training = win.sections["training"]
    assert not training.is_modified()
    win.imgsz_spin.setValue(320)
    assert training.is_modified()
    assert not training.dot.isHidden()
    assert "changed from defaults" in training.header.accessibleName()
    assert project.load_state().train.imgsz == 320
    training.reset_btn.click()
    assert win.imgsz_spin.value() == TrainSettings().imgsz
    assert not training.is_modified()
    assert training.dot.isHidden()
    assert project.load_state().train.imgsz == TrainSettings().imgsz


def test_each_section_resets_its_own_values(make_window, project: Project) -> None:
    win = make_window()
    open_project(win, project)
    win.device_combo.setCurrentIndex(win.device_combo.findData("cpu"))
    win.val_spin.setValue(35)
    win.auto_onnx_check.setChecked(False)
    win.workers_spin.setValue(1)
    assert all(s.is_modified() for s in win.sections.values())
    win.sections["hardware"].reset_btn.click()
    assert win.device_combo.currentData() == "auto"
    assert win.val_spin.value() == 35  # other sections untouched
    win.sections["data"].reset_btn.click()
    win.sections["output"].reset_btn.click()
    win.sections["training"].reset_btn.click()
    assert win.val_spin.value() == mw.DEFAULT_VAL_PERCENT
    assert win.auto_onnx_check.isChecked()
    assert not any(s.is_modified() for s in win.sections.values())
    st = project.load_state()
    assert st.val_ratio == pytest.approx(0.2)
    assert st.train.device == "auto" and st.train.workers == 4 and st.train.auto_export_onnx


# ------------------------------------------------------------------ presets
def test_fresh_project_shows_easy_preset(make_window, project: Project) -> None:
    win = make_window()
    open_project(win, project)
    assert win.preset_combo.currentText() == next(iter(PRESETS))


def test_preset_fills_values_and_edits_switch_to_custom(make_window, project: Project) -> None:
    win = make_window()
    open_project(win, project)
    i = win.preset_combo.findText("Quick test")
    win.preset_combo.setCurrentIndex(i)
    win.preset_combo.activated.emit(i)
    quick = PRESETS["Quick test"]
    assert win.size_combo.currentData() == quick["size"]
    assert win.epochs_spin.value() == quick["epochs"]
    assert win.preset_combo.currentText() == "Quick test"
    assert project.load_state().train.epochs == quick["epochs"]

    i = win.preset_combo.findText("Best accuracy")
    win.preset_combo.activated.emit(i)
    best = PRESETS["Best accuracy"]
    assert (win.size_combo.currentData(), win.batch_spin.value(), win.patience_spin.value()) == (
        best["size"], best["batch"], best["patience"])
    assert win.sections["training"].is_modified()  # batch 8, patience 100 differ from the defaults

    win.epochs_spin.setValue(best["epochs"] + 1)
    assert win.preset_combo.currentText() == CUSTOM
    win.epochs_spin.setValue(best["epochs"])
    assert win.preset_combo.currentText() == "Best accuracy"


def test_presets_use_existing_settings_keys() -> None:
    fields = set(TrainSettings.__dataclass_fields__)
    for name, values in PRESETS.items():
        assert set(values) <= fields, name


# ------------------------------------------------------------------ hidden settings still apply
def test_hidden_settings_apply_at_training_start(make_window, project: Project, monkeypatch) -> None:
    win = make_window()
    open_project(win, project)
    st = project.load_state()
    st.prepared = True
    project.save_state(st)
    project.data_yaml.write_text("path: .\n", encoding="utf-8")
    open_project(win, project)

    win.imgsz_spin.setValue(320)
    win.workers_spin.setValue(2)
    win.patience_spin.setValue(7)
    win.device_combo.setCurrentIndex(win.device_combo.findData("cpu"))
    win.auto_onnx_check.setChecked(False)
    assert not any(s.is_expanded() for s in win.sections.values())  # all of them hidden

    started = []
    monkeypatch.setattr(win.session, "start", lambda cfg: started.append(cfg))
    win._start_training()
    assert started, "training did not start"
    cfg = started[0]
    assert (cfg.imgsz, cfg.workers, cfg.patience, cfg.device, cfg.auto_export_onnx) == (320, 2, 7, "cpu", False)


def test_old_project_settings_load_and_keep_their_format(make_window, project: Project) -> None:
    old = {"family": "yolo11", "size": "m", "epochs": 42, "imgsz": 512, "batch": 4, "batch_auto": False,
           "workers": 3, "patience": 9, "device": "0", "auto_export_onnx": False}
    st = project.load_state()
    st.train = TrainSettings.from_dict(old)
    st.val_ratio = 0.3
    project.save_state(st)
    keys_before = set(json.loads(project.settings_file.read_text(encoding="utf-8")))

    win = make_window()
    open_project(win, project)
    assert win._read_settings() == TrainSettings.from_dict(old)
    assert win.val_spin.value() == 30
    assert win.preset_combo.currentText() == CUSTOM
    win.epochs_spin.setValue(43)
    data = json.loads(project.settings_file.read_text(encoding="utf-8"))
    assert set(data) == keys_before
    assert data["train"]["epochs"] == 43 and data["train"]["imgsz"] == 512


# ------------------------------------------------------------------ help
def test_help_opens_every_section(make_window) -> None:
    win = make_window()
    win.open_help()
    dlg = win.help_dialog
    assert dlg is not None and dlg.isVisible()
    assert dlg.current_section() == "quickstart"
    for sid, title, _html in HELP_SECTIONS:
        dlg.open_section(sid)
        assert dlg.current_section() == sid
        assert len(dlg.view.toPlainText()) > 200, sid
    for target in ("troubleshooting#oom", "troubleshooting#cuda", "troubleshooting#validation",
                   "troubleshooting#update", "troubleshooting#stuck", "troubleshooting#training"):
        dlg.open_section(target)
        assert dlg.current_section() == "troubleshooting"


def test_help_has_the_required_pages() -> None:
    ids = [sid for sid, _t, _h in HELP_SECTIONS]
    assert ids == ["quickstart", "settings", "results", "models", "troubleshooting", "glossary"]
    html = dict((sid, h) for sid, _t, h in HELP_SECTIONS)
    for anchor in ("oom", "cuda", "validation", "stuck", "update"):
        assert f'name="{anchor}"' in html["troubleshooting"], anchor
    assert "12 GB" in html["models"]


def test_help_search_filters_sections(make_window) -> None:
    win = make_window()
    win.open_help()
    dlg = win.help_dialog
    dlg.search.setText("out of memory")
    visible = dlg.visible_sections()
    assert "troubleshooting" in visible and len(visible) < len(HELP_SECTIONS)
    dlg.search.setText("zzqqxx")
    assert dlg.visible_sections() == []
    assert dlg.no_match.isVisibleTo(dlg)
    dlg.search.clear()
    assert len(dlg.visible_sections()) == len(HELP_SECTIONS)
    assert not dlg.no_match.isVisibleTo(dlg)


def test_f1_and_help_button_open_help(make_window) -> None:
    win = make_window()
    f1 = [a for a in win.actions() if a.shortcut() == QKeySequence("F1")]
    assert len(f1) == 1
    f1[0].trigger()
    assert win.help_dialog is not None and win.help_dialog.isVisible()
    win.help_dialog.close()
    win.help_btn.click()
    assert win.help_dialog.isVisible()


def test_empty_state_learn_more_opens_quick_start(make_window) -> None:
    win = make_window()
    win.empty_state.learn_more.click()
    assert win.help_dialog is not None and win.help_dialog.current_section() == "quickstart"


def test_help_follows_theme(make_window) -> None:
    win = make_window({"theme": "dark"})
    win.open_help()
    dark_css = win.help_dialog.view.document().defaultStyleSheet()
    win.apply_theme("light")
    assert win.help_dialog.view.document().defaultStyleSheet() != dark_css


# ------------------------------------------------------------------ validation report fixes
def test_validation_report_lists_fixes_and_links_to_help(make_window) -> None:
    rep = ds.ValidationReport(errors=["The val split has no images.", "3 label file(s) are invalid."])
    text = mw.MainWindow._report_with_fixes(rep, ["helmet"])
    assert "## How to fix" in text
    assert "help:troubleshooting#validation" in text
    for _line, fix in fixes_for(rep.errors):
        assert fix in text

    win = make_window()
    asked: list[str] = []
    win.report_view.help_requested.disconnect()
    win.report_view.help_requested.connect(asked.append)
    win.report_view._on_link(QUrl("help:troubleshooting#validation"))
    assert asked == ["troubleshooting#validation"]


def test_report_view_finds_the_fix_under_the_pointer(qapp: QApplication) -> None:
    view = ReportView()
    view.resize(600, 300)
    view.setPlainText("The val split has no images.")
    assert view.fix_at(QPoint(10, 10)) == fixes_for(["The val split has no images."])[0][1]
    view.setPlainText("Everything looks fine.")
    assert view.fix_at(QPoint(10, 10)) is None


def test_every_button_tip_exists() -> None:
    for key in ("help", "theme", "update", "create", "pick", "validate", "start", "start_blocked", "stop",
                "results", "test", "export", "show_all", "reset", "learn_more"):
        assert BUTTONS[key].strip(), key


def test_action_row_stacks_when_narrow(qapp: QApplication) -> None:
    left, right = QPushButton("Start training"), QPushButton("Open results folder and more")
    row = WrapRow(left, right)
    need = left.sizeHint().width() + right.sizeHint().width() + 8
    row.resize(need + 40, 60)
    row.show()
    qapp.processEvents()
    assert not row.is_wrapped()
    row.resize(need - 40, 120)
    qapp.processEvents()
    assert row.is_wrapped()
    assert row.minimumSizeHint().width() < need
    row.close()
