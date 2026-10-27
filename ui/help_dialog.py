"""Built-in help: a searchable list of pages on the left, the selected page on the right."""
from __future__ import annotations

import re

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QTextBrowser, QVBoxLayout,
)

from ui.help_texts import HELP_SECTIONS
from ui.theme import SP1, SP2, SP3, Fonts, help_css
from ui.widgets import button, label

_TAG = re.compile(r"<[^>]+>")


def plain(html: str) -> str:
    return re.sub(r"\s+", " ", _TAG.sub(" ", html)).replace("&lt;", "<").replace("&gt;", ">").strip()


class HelpDialog(QDialog):
    """Non-modal help window. open_section("troubleshooting#oom") jumps to a page and an anchor."""

    def __init__(self, theme: str, fonts: Fonts, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("LocalTrainer help")
        self.setModal(False)
        self.resize(940, 640)
        self.setMinimumSize(640, 420)
        self.fonts = fonts
        self._pages = {sid: (title, html) for sid, title, html in HELP_SECTIONS}
        self._text = {sid: plain(title + " " + html).lower() for sid, title, html in HELP_SECTIONS}

        root = QVBoxLayout(self)
        root.setContentsMargins(SP3, SP3, SP3, SP2)
        root.setSpacing(SP2)
        root.addWidget(label("Help", "display"))

        body = QHBoxLayout()
        body.setSpacing(SP2)
        nav = QVBoxLayout()
        nav.setSpacing(SP1)
        self.search = QLineEdit()
        self.search.setObjectName("HelpSearch")
        self.search.setPlaceholderText("Search help")
        self.search.setClearButtonEnabled(True)
        self.search.setAccessibleName("Search help")
        self.search.setToolTip("Type a word such as batch, memory or mAP to show only the pages that mention it")
        self.search.textChanged.connect(self.filter)
        self.list = QListWidget()
        self.list.setObjectName("HelpNav")
        self.list.setAccessibleName("Help pages")
        self.list.setFixedWidth(240)
        for sid, title, _html in HELP_SECTIONS:
            item = QListWidgetItem(title)
            item.setData(Qt.ItemDataRole.UserRole, sid)
            self.list.addItem(item)
        self.list.currentItemChanged.connect(self._on_current)
        self.no_match = label("No pages match. Try a shorter word.", "muted", wrap=True)
        self.no_match.hide()
        nav.addWidget(self.search)
        nav.addWidget(self.list, 1)
        nav.addWidget(self.no_match)
        body.addLayout(nav)

        self.view = QTextBrowser()
        self.view.setObjectName("HelpBody")
        self.view.setAccessibleName("Help page")
        self.view.setOpenLinks(False)
        self.view.anchorClicked.connect(self._on_link)
        body.addWidget(self.view, 1)
        root.addLayout(body, 1)

        foot = QHBoxLayout()
        foot.addStretch(1)
        self.close_btn: QPushButton = button("Close", tooltip="Close the help window (Esc)")
        self.close_btn.clicked.connect(self.close)
        foot.addWidget(self.close_btn)
        root.addLayout(foot)

        QShortcut(QKeySequence("Ctrl+F"), self, activated=self.search.setFocus)
        self.set_theme(theme)
        self.list.setCurrentRow(0)

    # ------------------------------------------------------------------ API
    def set_theme(self, theme: str) -> None:
        self.view.document().setDefaultStyleSheet(help_css(theme, self.fonts))
        self._render(self.current_section())

    def current_section(self) -> str:
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else HELP_SECTIONS[0][0]

    def section_ids(self) -> list[str]:
        return [sid for sid, _t, _h in HELP_SECTIONS]

    def open_section(self, target: str = "quickstart") -> None:
        sid, _, anchor = target.partition("#")
        if sid not in self._pages:
            sid = HELP_SECTIONS[0][0]
        if self.search.text():
            self.search.clear()
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.ItemDataRole.UserRole) == sid:
                self.list.setCurrentRow(i)
                break
        self._render(sid)
        if anchor:
            self.view.scrollToAnchor(anchor)
        self.show()
        self.raise_()
        self.activateWindow()

    def visible_sections(self) -> list[str]:
        return [self.list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.list.count()) if not self.list.item(i).isHidden()]

    def filter(self, text: str) -> None:
        words = text.lower().split()
        first_visible = None
        for i in range(self.list.count()):
            item = self.list.item(i)
            sid = item.data(Qt.ItemDataRole.UserRole)
            hit = all(w in self._text[sid] for w in words)
            item.setHidden(not hit)
            if hit and first_visible is None:
                first_visible = i
        self.no_match.setVisible(first_visible is None)
        cur = self.list.currentItem()
        if first_visible is not None and (cur is None or cur.isHidden()):
            self.list.setCurrentRow(first_visible)
        elif first_visible is not None:
            self._render(self.current_section())

    # ------------------------------------------------------------------ internals
    def _on_current(self, cur: QListWidgetItem | None, _prev: QListWidgetItem | None) -> None:
        if cur is not None:
            self._render(cur.data(Qt.ItemDataRole.UserRole))

    def _render(self, sid: str) -> None:
        _title, html = self._pages[sid]
        words = [w for w in self.search.text().split() if len(w) > 1]
        for w in words:  # mark search hits outside tags so the reader can find them
            html = re.sub(f"(?i)(?<![<\\w/&])({re.escape(w)})(?![^<]*>)", r"<b><u>\1</u></b>", html)
        self.view.setHtml(html)

    def _on_link(self, url: QUrl) -> None:
        if url.scheme() == "help":
            self.open_section(url.path() + (f"#{url.fragment()}" if url.fragment() else ""))
        elif url.scheme() in ("http", "https"):
            QDesktopServices.openUrl(url)
        elif url.fragment():
            self.view.scrollToAnchor(url.fragment())
