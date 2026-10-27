"""Reusable widgets: banners, stat tiles, empty states, class editor, metrics chart, log view, tasks."""
from __future__ import annotations

from typing import Any, Callable

from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QKeyEvent, QMouseEvent, QPainter, QPainterPath, QPen, QTextCursor
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QFrame, QHBoxLayout, QHeaderView, QLabel, QPlainTextEdit, QPushButton,
    QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ui.theme import FONT_SIZE, SP1, SP2, SP3, THEMES


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


def repolish(widget: QWidget) -> None:
    """Re-apply the stylesheet after a dynamic property change."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


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
            b = button(text)
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
    """Left-aligned message in a readable column, vertically centred."""

    def __init__(self, title: str, body: str, steps: list[str] | None = None, parent: QWidget | None = None) -> None:
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
        for step in steps or []:
            lay.addWidget(label(step, None, wrap=True))
        lay.addStretch(1)
        outer.addWidget(col, 3)
        outer.addStretch(1)


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
        self.setMinimumHeight(220)
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
        small = QFont(QApplication.font())
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
