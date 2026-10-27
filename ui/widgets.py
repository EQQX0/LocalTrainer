"""Reusable widgets: banners, stat tiles, empty states, info labels, collapsible sections, class editor,
metrics chart, log view, tasks."""
from __future__ import annotations

import os
import sys
from typing import Any, Callable

from PyQt6.QtCore import (
    QEasingCurve, QEvent, QObject, QPoint, QPointF, QPropertyAnimation, QRectF, Qt, QThread, QTimer, QUrl,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor, QDesktopServices, QFont, QHelpEvent, QKeyEvent, QMouseEvent, QPainter, QPainterPath, QPen, QTextCursor,
)
from PyQt6.QtWidgets import (
    QAbstractItemView, QBoxLayout, QFrame, QHBoxLayout, QLayout, QHeaderView, QLabel, QPlainTextEdit, QPushButton,
    QSizePolicy, QTableWidget, QTableWidgetItem, QTextBrowser, QToolButton, QToolTip, QVBoxLayout, QWidget,
)

from ui.help_texts import BUTTONS, fixes_for, tip as wrap_tip
from ui.theme import FONT_SIZE, SP1, SP2, SP3, THEMES

TOOLTIP_DELAY_MS = 400
SECTION_ANIM_MS = 180


def label(text: str = "", role: str | None = None, wrap: bool = False) -> QLabel:
    lbl = QLabel(text)
    if role:
        lbl.setProperty("role", role)
    lbl.setWordWrap(wrap)
    return lbl


def button(text: str, kind: str | None = None, tooltip: str = "") -> QPushButton:
    btn = QPushButton(text)
    if kind:
        btn.setProperty("kind", kind)
    if tooltip:
        btn.setToolTip(tooltip)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    return btn


def reduced_motion() -> bool:
    """True when the user asked for less animation (env override, or Windows "Animation effects" off)."""
    env = os.environ.get("LOCALTRAINER_REDUCED_MOTION")
    if env is not None:
        return env not in ("", "0", "false")
    if sys.platform == "win32":
        try:
            import ctypes
            enabled = ctypes.c_int(1)
            # SPI_GETCLIENTAREAANIMATION
            if ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(enabled), 0):
                return not enabled.value
        except (AttributeError, OSError):
            pass
    return False


def show_tip_for(widget: QWidget) -> None:
    """Show a widget's tooltip right under it, as hover would."""
    text = widget.toolTip()
    if text and widget.isVisible():
        QToolTip.showText(widget.mapToGlobal(QPoint(0, widget.height() + 2)), text, widget)


class FocusTip(QObject):
    """Event filter: when a watched widget gets keyboard focus, show its tooltip after the hover delay."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(TOOLTIP_DELAY_MS)
        self._timer.timeout.connect(self._show)
        self._target: QWidget | None = None

    def watch(self, *widgets: QWidget) -> None:
        for w in widgets:
            w.installEventFilter(self)

    def eventFilter(self, obj: QObject, ev: QEvent) -> bool:  # noqa: N802 - Qt override
        if ev.type() == QEvent.Type.FocusIn and isinstance(obj, QWidget):
            if ev.reason() in (Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason,
                               Qt.FocusReason.ShortcutFocusReason):
                self._target = obj
                self._timer.start()
        elif ev.type() == QEvent.Type.FocusOut and obj is self._target:
            self._timer.stop()
            self._target = None
            QToolTip.hideText()
        return False

    def _show(self) -> None:
        if self._target is not None and self._target.hasFocus():
            show_tip_for(self._target)


def repolish(widget: QWidget) -> None:
    """Re-apply the stylesheet after a dynamic property change."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


class WrapRow(QWidget):
    """Two widget groups side by side that stack into two lines when the row is too narrow for both."""

    def __init__(self, left: QWidget, right: QWidget, spacing: int = SP1, parent: QWidget | None = None):
        super().__init__(parent)
        self.left, self.right = left, right
        self._lay = QBoxLayout(QBoxLayout.Direction.LeftToRight, self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(spacing)
        self._lay.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)  # the row may be narrower than both groups
        self._lay.addWidget(left, 0, Qt.AlignmentFlag.AlignLeft)
        self._lay.addStretch(1)
        self._lay.addWidget(right, 0, Qt.AlignmentFlag.AlignLeft)

    def is_wrapped(self) -> bool:
        return self._lay.direction() == QBoxLayout.Direction.TopToBottom

    def minimumSizeHint(self):  # noqa: N802 (Qt override)
        hint = super().minimumSizeHint()
        hint.setWidth(max(self.left.minimumSizeHint().width(), self.right.minimumSizeHint().width()))
        return hint

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        need = self.left.sizeHint().width() + self._lay.spacing() + self.right.sizeHint().width()
        fits = event.size().width() >= need
        direction = QBoxLayout.Direction.LeftToRight if fits else QBoxLayout.Direction.TopToBottom
        if direction != self._lay.direction():
            self._lay.setDirection(direction)
        super().resizeEvent(event)


class Banner(QFrame):
    """Inline message with a tone (error, warning, info, success) and optional action buttons."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("banner", "info")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(SP2, SP1 + 4, SP2, SP1 + 4)
        lay.setSpacing(SP2)
        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        self.title = label("", None)
        self.body = label("", None, wrap=True)
        self.body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        text_col.addWidget(self.title)
        text_col.addWidget(self.body)
        lay.addLayout(text_col, 1)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(SP1)
        lay.addLayout(self.actions)
        self.hide()

    def show_message(self, tone: str, title: str, body: str = "",
                     actions: list[tuple[str, Callable[[], None]]] | None = None) -> None:
        self.setProperty("banner", tone)
        self.title.setProperty("tone", tone)
        self.title.setText(title)
        self.body.setText(body)
        self.body.setVisible(bool(body))
        while self.actions.count():
            item = self.actions.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        for text, cb in actions or []:
            b = button(text, tooltip=wrap_tip(BUTTONS.get(text, "")))
            b.clicked.connect(cb)
            self.actions.addWidget(b)
        repolish(self)
        repolish(self.title)
        self.show()


class StatTile(QFrame):
    """A large number with a caption under it."""

    def __init__(self, caption: str, tooltip: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("tile", "true")
        self.setToolTip(tooltip)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(SP2, SP2, SP2, SP2)
        lay.setSpacing(2)
        self.value = label("–", "stat")
        self.caption = label(caption, "muted")
        lay.addWidget(self.value)
        lay.addWidget(self.caption)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName(caption)

    def set_value(self, text: str) -> None:
        self.value.setText(text)
        self.setAccessibleDescription(text)


class EmptyState(QWidget):
    """Left-aligned message in a readable column, vertically centred, with optional numbered steps."""

    def __init__(self, title: str, body: str, steps: list[str] | None = None,
                 learn_more: Callable[[], None] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(SP3, SP3, SP3, SP3)
        outer.addStretch(1)
        col = QWidget()
        col.setMaximumWidth(560)
        lay = QVBoxLayout(col)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(SP2)
        lay.addStretch(1)
        self.title = label(title, "display", wrap=True)
        self.body = label(body, "emptyBody", wrap=True)
        lay.addWidget(self.title)
        lay.addWidget(self.body)
        if steps:
            lay.addSpacing(SP1)
            lay.addWidget(label("How it works", "heading"))
            for i, step in enumerate(steps, 1):
                row = QHBoxLayout()
                row.setSpacing(SP2)
                num = label(str(i), "step")
                num.setFixedSize(24, 24)
                num.setAlignment(Qt.AlignmentFlag.AlignCenter)
                num.setAccessibleName(f"Step {i}")
                row.addWidget(num, 0, Qt.AlignmentFlag.AlignTop)
                row.addWidget(label(step, None, wrap=True), 1)
                lay.addLayout(row)
        self.learn_more: QPushButton | None = None
        if learn_more:
            self.learn_more = button("Learn more in Help (F1)", "link", wrap_tip(BUTTONS["learn_more"]))
            self.learn_more.clicked.connect(learn_more)
            lay.addWidget(self.learn_more, 0, Qt.AlignmentFlag.AlignLeft)
        lay.addStretch(1)
        outer.addWidget(col, 3)
        outer.addStretch(1)


class InfoLabel(QWidget):
    """A form label followed by a focusable "?" button; the help text shows on hover, focus or click.

    The same text is set as the tooltip of the label and of the buddy input, so hovering anywhere on
    the row explains the setting."""

    def __init__(self, text: str, tip_text: str, buddy: QWidget | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tip_text = wrap_tip(tip_text)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.label = QLabel(text)
        self.label.setToolTip(self.tip_text)
        self.info = QToolButton()
        self.info.setText("?")
        self.info.setProperty("kind", "info")
        self.info.setFixedSize(20, 20)
        self.info.setToolTip(self.tip_text)
        self.info.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.info.setCursor(Qt.CursorShape.WhatsThisCursor)
        self.info.setAccessibleName(f"Help: {text}")
        self.info.setAccessibleDescription(tip_text)
        self.info.clicked.connect(lambda: show_tip_for(self.info))
        lay.addWidget(self.label)
        lay.addWidget(self.info)
        lay.addStretch(1)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(TOOLTIP_DELAY_MS)
        self._timer.timeout.connect(self._show_if_focused)
        self.info.installEventFilter(self)
        self.buddy = buddy
        if buddy is not None:
            self.label.setBuddy(buddy)
            buddy.setToolTip(self.tip_text)
            if not buddy.accessibleName():
                buddy.setAccessibleName(text)
            buddy.setAccessibleDescription(tip_text)

    def _show_if_focused(self) -> None:
        if self.info.hasFocus():
            show_tip_for(self.info)

    def eventFilter(self, obj: QObject, ev: QEvent) -> bool:  # noqa: N802 - Qt override
        if obj is self.info:
            if ev.type() == QEvent.Type.FocusIn:
                self._timer.start()
            elif ev.type() == QEvent.Type.FocusOut:
                self._timer.stop()
                QToolTip.hideText()
        return False


class CollapsibleSection(QWidget):
    """A header button with + / \u2212 that shows or hides its content with a short height animation.

    Values inside stay live while collapsed. A dot after the title marks values that differ from
    their defaults, and a "Reset to defaults" link sits at the end of the content."""

    toggled_state = pyqtSignal(bool)
    reset_requested = pyqtSignal()

    def __init__(self, sid: str, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.sid, self.title = sid, title
        self._expanded = False
        self._modified = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(SP1)
        self.header = button("", "section")
        self.header.setCheckable(True)
        self.header.setToolTip(wrap_tip(f"Show or hide the {title.lower()}. Hidden values still apply."))
        self.header.toggled.connect(self.set_expanded)
        self.header.installEventFilter(self)  # Enter toggles too, not only Space
        self.dot = label("", "dot")
        self.dot.setFixedSize(8, 8)
        self.dot.setToolTip("Changed from the defaults")
        self.dot.hide()
        head.addWidget(self.header, 1)
        head.addWidget(self.dot, 0, Qt.AlignmentFlag.AlignVCenter)
        head.addSpacing(SP1)
        lay.addLayout(head)

        self.content = QWidget()
        self.body = QVBoxLayout(self.content)
        self.body.setContentsMargins(SP1, SP1, 0, SP2)
        self.body.setSpacing(SP1)
        self.reset_btn = button("Reset to defaults", "link", wrap_tip(BUTTONS["reset"]))
        self.reset_btn.setAccessibleName(f"Reset {title.lower()} to defaults")
        self.reset_btn.setEnabled(False)
        self.reset_btn.clicked.connect(self.reset_requested.emit)
        reset_row = QHBoxLayout()
        reset_row.addWidget(self.reset_btn)
        reset_row.addStretch(1)
        self.body.addLayout(reset_row)
        lay.addWidget(self.content)
        self.content.hide()

        self._anim = QPropertyAnimation(self.content, b"maximumHeight", self)
        self._anim.setDuration(SECTION_ANIM_MS)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.finished.connect(self._anim_done)
        self._sync_header()

    def eventFilter(self, obj: QObject, ev: QEvent) -> bool:  # noqa: N802 - Qt override
        if (obj is self.header and ev.type() == QEvent.Type.KeyPress and isinstance(ev, QKeyEvent)
                and ev.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)):
            self.header.toggle()
            return True
        return super().eventFilter(obj, ev)

    def add_widget(self, widget: QWidget) -> None:
        self.body.insertWidget(self.body.count() - 1, widget)

    def add_layout(self, layout: Any) -> None:
        self.body.insertLayout(self.body.count() - 1, layout)

    def is_expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool, animate: bool = True) -> None:
        expanded = bool(expanded)
        if expanded == self._expanded:
            return
        self._expanded = expanded
        self.header.blockSignals(True)
        self.header.setChecked(expanded)
        self.header.blockSignals(False)
        self._sync_header()
        self._anim.stop()
        if not animate or reduced_motion() or not self.isVisible():
            self.content.setMaximumHeight(16777215)
            self.content.setVisible(expanded)
        elif expanded:
            self.content.setMaximumHeight(0)
            self.content.show()
            self._anim.setStartValue(0)
            self._anim.setEndValue(max(1, self.content.sizeHint().height()))
            self._anim.start()
        else:
            self._anim.setStartValue(self.content.height())
            self._anim.setEndValue(0)
            self._anim.start()
        self.toggled_state.emit(expanded)

    def _anim_done(self) -> None:
        if not self._expanded:
            self.content.hide()
        self.content.setMaximumHeight(16777215)  # free again, so the content can grow or relayout

    def set_modified(self, modified: bool) -> None:
        self._modified = bool(modified)
        self.dot.setVisible(self._modified)
        self.reset_btn.setEnabled(self._modified)
        self._sync_header()

    def is_modified(self) -> bool:
        return self._modified

    def _sync_header(self) -> None:
        # U+2212 minus has the same advance as "+", so the title does not shift when toggled
        self.header.setText(("\u2212 " if self._expanded else "+ ") + self.title)
        state = "expanded" if self._expanded else "collapsed"
        changed = ", changed from defaults" if self._modified else ""
        self.header.setAccessibleName(f"{self.title}, {state}{changed}")


class ReportView(QTextBrowser):
    """Validation report. Hovering a problem line shows how to fix it; help: links ask for a Help page."""

    help_requested = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setOpenLinks(False)
        self.anchorClicked.connect(self._on_link)

    def fix_at(self, pos: QPoint) -> str | None:
        line = self.cursorForPosition(pos).block().text()
        found = fixes_for([line]) if line else []
        return found[0][1] if found else None

    def viewportEvent(self, ev: QEvent) -> bool:  # noqa: N802 - Qt override
        if ev.type() == QEvent.Type.ToolTip and isinstance(ev, QHelpEvent):
            fix = self.fix_at(ev.pos())
            if fix:
                QToolTip.showText(ev.globalPos(), wrap_tip(f"How to fix: {fix}"), self.viewport())
            else:
                QToolTip.hideText()
            return True
        return super().viewportEvent(ev)

    def _on_link(self, url: QUrl) -> None:
        if url.scheme() == "help":
            self.help_requested.emit(url.path() + (f"#{url.fragment()}" if url.fragment() else ""))
        elif url.scheme() in ("http", "https"):
            QDesktopServices.openUrl(url)


class ClassEditor(QWidget):
    """Editable class-name table in class-id order."""

    changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(SP1)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["ID", "Class name"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                                   | QAbstractItemView.EditTrigger.EditKeyPressed
                                   | QAbstractItemView.EditTrigger.AnyKeyPressed)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setMinimumHeight(140)
        self.table.setToolTip("Class names in class-id order. Double-click or press F2 to rename.")
        self.table.setAccessibleName("Class names")
        self.table.itemChanged.connect(lambda *_: self.changed.emit())
        self.empty = label("No classes yet. Choose a dataset folder, or add classes by hand.", "muted", wrap=True)
        lay.addWidget(self.empty)
        lay.addWidget(self.table)
        self.changed.connect(self._sync_empty)
        row = QHBoxLayout()
        row.setSpacing(SP1)
        self.add_btn = button("Add class", tooltip="Add a class at the next id")
        self.remove_btn = button("Remove last", tooltip="Remove the class with the highest id")
        self.add_btn.clicked.connect(lambda: self._add(f"class_{self.table.rowCount()}", edit=True))
        self.remove_btn.clicked.connect(self._remove_last)
        row.addWidget(self.add_btn)
        row.addWidget(self.remove_btn)
        row.addStretch(1)
        lay.addLayout(row)
        self._sync_empty()

    def _sync_empty(self) -> None:
        empty = self.table.rowCount() == 0
        self.empty.setVisible(empty)
        self.table.setVisible(not empty)
        self.remove_btn.setEnabled(not empty)

    def names(self) -> list[str]:
        out = []
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 1)
            out.append(item.text().strip() if item else "")
        return out

    def set_names(self, names: list[str]) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for n in names:
            self._add(n)
        self.table.blockSignals(False)
        self.changed.emit()

    def _add(self, name: str, edit: bool = False) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        id_item = QTableWidgetItem(str(r))
        id_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(r, 0, id_item)
        self.table.setItem(r, 1, QTableWidgetItem(name))
        if edit:
            self.table.setCurrentCell(r, 1)
            self.table.editItem(self.table.item(r, 1))
            self.changed.emit()

    def _remove_last(self) -> None:
        if self.table.rowCount():
            self.table.removeRow(self.table.rowCount() - 1)
            self.changed.emit()


class MetricsChart(QWidget):
    """Line chart of mAP50 (solid) and mAP50-95 (dashed) per epoch. Hover or arrow keys inspect values."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.theme = "dark"
        self.display_font = "Bahnschrift"
        self.map50: list[tuple[int, float]] = []
        self.map5095: list[tuple[int, float]] = []
        self.total_epochs = 0
        self.cursor_index: int | None = None
        self.setMinimumHeight(150)  # leaves room for the tabs at 1100x700 with the action bar wrapped
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip("mAP50 per epoch. Hover, or focus and use the arrow keys, to read values.")
        self.setAccessibleName("mAP50 chart")

    def set_theme(self, theme: str, display_font: str) -> None:
        self.theme, self.display_font = theme, display_font
        self.update()

    def clear(self, total_epochs: int = 0) -> None:
        self.map50.clear()
        self.map5095.clear()
        self.total_epochs = total_epochs
        self.cursor_index = None
        self.update()

    def add_point(self, epoch: int, map50: float | None, map5095: float | None) -> None:
        if map50 is not None:
            self.map50 = [p for p in self.map50 if p[0] != epoch] + [(epoch, map50)]
        if map5095 is not None:
            self.map5095 = [p for p in self.map5095 if p[0] != epoch] + [(epoch, map5095)]
        self.setAccessibleDescription(f"Epoch {epoch}: mAP50 {map50 if map50 is not None else 'n/a'}")
        self.update()

    # geometry
    def _plot_rect(self) -> QRectF:
        return QRectF(56, 40, max(10, self.width() - 56 - SP3), max(10, self.height() - 40 - 40))

    def _x_max(self) -> int:
        last = max([e for e, _ in self.map50] + [1])
        return max(self.total_epochs, last, 2)

    def _to_point(self, rect: QRectF, epoch: int, value: float) -> QPointF:
        x = rect.left() + (epoch - 1) / (self._x_max() - 1) * rect.width()
        y = rect.bottom() - max(0.0, min(1.0, value)) * rect.height()
        return QPointF(x, y)

    def paintEvent(self, _event: Any) -> None:
        t = THEMES[self.theme]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(t["surface"]))
        rect = self._plot_rect()
        small = QFont(self.font())  # the stylesheet body font; the app default can differ
        small.setPixelSize(FONT_SIZE["small"])
        p.setFont(small)

        if not self.map50:
            p.setPen(QColor(t["ink"]))
            big = QFont(self.display_font)
            big.setPixelSize(FONT_SIZE["heading"])
            big.setWeight(QFont.Weight.DemiBold)
            p.setFont(big)
            p.drawText(self.rect().adjusted(0, -24, 0, -24), Qt.AlignmentFlag.AlignCenter, "No metrics yet")
            p.setFont(small)
            p.setPen(QColor(t["ink_muted"]))
            p.drawText(self.rect().adjusted(SP3, 24, -SP3, 24), Qt.AlignmentFlag.AlignCenter,
                       "mAP50 appears here after the first epoch is validated.")
            return

        # y labels (no grid lines: values sit at the left edge)
        p.setPen(QColor(t["ink_muted"]))
        for v in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = rect.bottom() - v * rect.height()
            p.drawText(QRectF(0, y - 8, 48, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{v:.2f}")
        # baseline band
        p.fillRect(QRectF(rect.left(), rect.bottom(), rect.width(), 2), QColor(t["raised"]))
        # x labels
        xmax = self._x_max()
        step = max(1, round(xmax / 6))
        for e in list(range(1, xmax + 1, step)) + [xmax]:
            pt = self._to_point(rect, e, 0)
            p.drawText(QRectF(pt.x() - 24, rect.bottom() + 6, 48, 18), Qt.AlignmentFlag.AlignCenter, str(e))
        p.drawText(QRectF(rect.left(), rect.bottom() + 22, rect.width(), 18), Qt.AlignmentFlag.AlignCenter, "epoch")

        def draw_series(points: list[tuple[int, float]], col: str, dashed: bool, width: float) -> None:
            pts = sorted(points)
            if not pts:
                return
            path = QPainterPath(self._to_point(rect, *pts[0]))
            for e, v in pts[1:]:
                path.lineTo(self._to_point(rect, e, v))
            pen = QPen(QColor(col), width)
            pen.setStyle(Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
            if len(pts) <= 60:  # markers help when color is not enough
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(col))
                for e, v in pts:
                    p.drawRect(QRectF(self._to_point(rect, e, v) - QPointF(2.5, 2.5), QRectF(0, 0, 5, 5).size()))

        draw_series(self.map5095, t["ink_muted"], True, 2.0)
        draw_series(self.map50, t["accent_ink"], False, 2.5)

        # legend (top right): line sample + words, so colour is never the only cue
        lx = rect.right() - 220
        for i, (name, col, dashed) in enumerate((("mAP50", t["accent_ink"], False), ("mAP50-95", t["ink_muted"], True))):
            x = lx + i * 110
            pen = QPen(QColor(col), 2.5)
            pen.setStyle(Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
            p.setPen(pen)
            p.drawLine(QPointF(x, 20), QPointF(x + 24, 20))
            p.setPen(QColor(t["ink"]))
            p.drawText(QRectF(x + 30, 10, 80, 20), Qt.AlignmentFlag.AlignVCenter, name)

        # readout for the inspected epoch
        pts = sorted(self.map50)
        idx = self.cursor_index if self.cursor_index is not None else len(pts) - 1
        idx = max(0, min(idx, len(pts) - 1))
        e, v = pts[idx]
        other = dict(self.map5095).get(e)
        pt = self._to_point(rect, e, v)
        if self.cursor_index is not None:
            p.fillRect(QRectF(pt.x() - 1, rect.top(), 2, rect.height()), QColor(t["hover"]))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(t["accent"]))
        p.drawRect(QRectF(pt.x() - 4, pt.y() - 4, 8, 8))
        p.setPen(QColor(t["ink"]))
        readout = f"Epoch {e}    mAP50 {v:.3f}" + (f"    mAP50-95 {other:.3f}" if other is not None else "")
        p.drawText(QRectF(rect.left(), 10, rect.width() - 240, 20), Qt.AlignmentFlag.AlignVCenter, readout)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if not self.map50:
            return
        rect = self._plot_rect()
        pts = sorted(self.map50)
        x = ev.position().x()
        self.cursor_index = min(range(len(pts)), key=lambda i: abs(self._to_point(rect, pts[i][0], 0).x() - x))
        self.update()

    def leaveEvent(self, _ev: Any) -> None:
        if not self.hasFocus():
            self.cursor_index = None
            self.update()

    def keyPressEvent(self, ev: QKeyEvent) -> None:
        if self.map50 and ev.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Home, Qt.Key.Key_End):
            n = len(self.map50)
            i = self.cursor_index if self.cursor_index is not None else n - 1
            i = {Qt.Key.Key_Left: i - 1, Qt.Key.Key_Right: i + 1, Qt.Key.Key_Home: 0, Qt.Key.Key_End: n - 1}[ev.key()]
            self.cursor_index = max(0, min(n - 1, i))
            self.update()
        else:
            super().keyPressEvent(ev)

    def focusOutEvent(self, ev: Any) -> None:
        self.cursor_index = None
        self.update()
        super().focusOutEvent(ev)


class LogView(QPlainTextEdit):
    """Read-only log that keeps progress bars on one line and follows the end unless scrolled up."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Log")
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setPlaceholderText("The Ultralytics output appears here when training starts.")
        self.setAccessibleName("Training log")

    def append_line(self, text: str, replace_previous: bool = False) -> None:
        bar = self.verticalScrollBar()
        at_end = bar.value() >= bar.maximum() - 4
        if replace_previous and not self.document().isEmpty():
            # overwrite the last line in place (progress bar update)
            cur = QTextCursor(self.document().lastBlock())
            cur.movePosition(QTextCursor.MoveOperation.StartOfBlock)
            cur.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
            cur.insertText(text)
        else:
            self.appendPlainText(text)
        if at_end:
            bar.setValue(bar.maximum())


class Task(QThread):
    """Run fn(progress) in a background thread; results come back as signals on the UI thread."""

    succeeded = pyqtSignal(object)
    failed = pyqtSignal(object)
    progress = pyqtSignal(int, int)

    def __init__(self, fn: Callable[[Callable[[int, int], None]], Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            self.succeeded.emit(self._fn(self.progress.emit))
        except Exception as exc:  # delivered to the UI as a friendly message
            self.failed.emit(exc)
