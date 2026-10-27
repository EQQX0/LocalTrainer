"""Design tokens and the single Qt stylesheet built from them.

Art direction: a graphite workbench with one signal colour, the chartreuse of an
annotation box. Surfaces are separated by tone steps, not lines or shadows.
Corners are square (2 px), spacing sits on an 8 px grid, and the display face
is Bahnschrift, the DIN-style face that ships with Windows.
"""
from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtGui import QColor, QFontDatabase
from PyQt6.QtWidgets import QProxyStyle, QStyle, QStyleFactory

from utils.paths import ASSETS_DIR

# 8 px spacing grid
SP1, SP2, SP3, SP4, SP6 = 8, 16, 24, 32, 48

FONT_SIZE = {"small": 12, "body": 14, "title": 16, "heading": 20, "display": 28, "stat": 30}
RADIUS = {"sm": 2, "md": 4}

DARK: dict[str, str] = {
    "ground": "#111311",      # window
    "surface": "#191C19",     # side panel, top bar
    "raised": "#222621",      # cards, buttons
    "field": "#2A2F29",       # inputs
    "hover": "#323830",
    "pressed": "#3B4239",
    "ink": "#ECEFE7",
    "ink_muted": "#A6AD9F",
    "ink_faint": "#7D8577",
    "accent": "#C8F135",      # annotation-box chartreuse
    "accent_hover": "#D6F95C",
    "accent_pressed": "#B4DB2A",
    "on_accent": "#121410",
    "accent_ink": "#C8F135",  # accent used as text / plot line
    "accent_soft": "#2E3A12",
    "danger": "#FF7262",
    "danger_soft": "#3A1D19",
    "warning": "#F2B33D",
    "warning_soft": "#3A2E14",
    "info_soft": "#1E2A2E",
    "selection": "#3E4D17",
}

LIGHT: dict[str, str] = {
    "ground": "#F2F3EE",
    "surface": "#E7E9E2",
    "raised": "#DCDFD5",
    "field": "#FFFFFF",
    "hover": "#D2D6CA",
    "pressed": "#C6CBBD",
    "ink": "#141713",
    "ink_muted": "#4B5247",
    "ink_faint": "#6A7265",
    "accent": "#C8F135",
    "accent_hover": "#D6F95C",
    "accent_pressed": "#B4DB2A",
    "on_accent": "#121410",
    "accent_ink": "#4A6600",
    "accent_soft": "#E5F5B5",
    "danger": "#B3261E",
    "danger_soft": "#F7DEDA",
    "warning": "#8A5A00",
    "warning_soft": "#F7EBCB",
    "info_soft": "#DCE8EA",
    "selection": "#D5EC8A",
}

THEMES = {"dark": DARK, "light": LIGHT}


@dataclass
class Fonts:
    display: str
    body: str
    mono: str


def _first_available(candidates: list[str]) -> str:
    families = set(QFontDatabase.families())
    return next((c for c in candidates if c in families), candidates[-1])


def resolve_fonts() -> Fonts:
    return Fonts(
        display=_first_available(["Bahnschrift", "Segoe UI Variable Display", "Segoe UI", "Arial"]),
        body=_first_available(["Segoe UI Variable Text", "Segoe UI", "Arial"]),
        mono=_first_available(["Cascadia Mono", "Cascadia Code", "Consolas", "Courier New"]),
    )


def color(theme: str, key: str) -> QColor:
    return QColor(THEMES[theme][key])


def build_qss(theme: str, fonts: Fonts) -> str:
    t = THEMES[theme]
    fs, r = FONT_SIZE, RADIUS
    chevron = (ASSETS_DIR / f"chevron-{theme}.svg").as_posix()
    check = (ASSETS_DIR / "check.svg").as_posix()
    return f"""
* {{
    font-family: "{fonts.body}";
    font-size: {fs['body']}px;
    color: {t['ink']};
    outline: none;
}}
QMainWindow, QDialog, QWidget#Ground {{ background: {t['ground']}; }}
QWidget#Sidebar, QWidget#TopBar {{ background: {t['surface']}; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}

/* ---------- type roles ---------- */
QLabel {{ background: transparent; }}
QLabel[role="wordmark"] {{ font-family: "{fonts.display}"; font-size: {fs['heading']}px; font-weight: 600; }}
QLabel[role="display"] {{ font-family: "{fonts.display}"; font-size: {fs['display']}px; font-weight: 600; }}
QLabel[role="heading"] {{ font-family: "{fonts.display}"; font-size: {fs['title']}px; font-weight: 600; }}
QLabel[role="stat"] {{ font-family: "{fonts.display}"; font-size: {fs['stat']}px; font-weight: 500; }}
QLabel[role="muted"], QLabel[role="caption"] {{ color: {t['ink_muted']}; }}
QLabel[role="caption"] {{ font-size: {fs['small']}px; }}
QLabel[role="error"] {{ color: {t['danger']}; font-size: {fs['small']}px; }}
QLabel[role="project"] {{ color: {t['ink_muted']}; font-size: {fs['title']}px; }}
QLabel[role="emptyTitle"] {{ font-family: "{fonts.display}"; font-size: {fs['heading']}px; font-weight: 600; }}
QLabel[role="emptyBody"] {{ color: {t['ink_muted']}; }}
QLabel[role="badge"] {{
    background: {t['accent']}; color: {t['on_accent']}; font-size: 11px; font-weight: 700;
    border-radius: {r['sm']}px; padding: 0px 4px;
}}

/* ---------- cards ---------- */
QFrame[card="true"] {{ background: {t['raised']}; border-radius: {r['md']}px; }}
QFrame[tile="true"] {{ background: {t['surface']}; border-radius: {r['md']}px; }}
QFrame[banner="error"] {{ background: {t['danger_soft']}; border-radius: {r['md']}px; }}
QFrame[banner="warning"] {{ background: {t['warning_soft']}; border-radius: {r['md']}px; }}
QFrame[banner="info"] {{ background: {t['info_soft']}; border-radius: {r['md']}px; }}
QFrame[banner="success"] {{ background: {t['accent_soft']}; border-radius: {r['md']}px; }}
QLabel[tone="error"] {{ color: {t['danger']}; font-weight: 600; }}
QLabel[tone="warning"] {{ color: {t['warning']}; font-weight: 600; }}
QLabel[tone="success"] {{ color: {t['accent_ink']}; font-weight: 600; }}
QLabel[tone="info"] {{ color: {t['ink']}; font-weight: 600; }}

/* ---------- buttons ---------- */
QPushButton, QToolButton {{
    background: {t['raised']}; color: {t['ink']};
    border: 2px solid transparent; border-radius: {r['sm']}px;
    padding: 6px 14px; min-height: 24px;
}}
QPushButton:hover, QToolButton:hover {{ background: {t['hover']}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {t['pressed']}; }}
QPushButton:focus, QToolButton:focus {{ border-color: {t['accent_ink']}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {t['ink_faint']}; background: {t['surface']}; }}
QPushButton[kind="primary"] {{ background: {t['accent']}; color: {t['on_accent']}; font-weight: 600; }}
QPushButton[kind="primary"]:hover {{ background: {t['accent_hover']}; }}
QPushButton[kind="primary"]:pressed {{ background: {t['accent_pressed']}; }}
QPushButton[kind="primary"]:focus {{ border-color: {t['ink']}; }}
QPushButton[kind="primary"]:disabled {{ background: {t['raised']}; color: {t['ink_faint']}; }}
QPushButton[kind="danger"] {{ color: {t['danger']}; font-weight: 600; }}
QPushButton[kind="danger"]:disabled {{ color: {t['ink_faint']}; }}
QPushButton[kind="ghost"] {{ background: transparent; color: {t['ink_muted']}; padding: 6px 8px; }}
QPushButton[kind="ghost"]:hover {{ background: {t['hover']}; color: {t['ink']}; }}
QPushButton[kind="update"] {{ background: {t['raised']}; }}
QPushButton[kind="update"][available="true"] {{ background: {t['accent_soft']}; color: {t['accent_ink']}; font-weight: 600; }}
QPushButton::menu-indicator {{ image: none; width: 0px; }}

/* ---------- inputs ---------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {t['field']}; color: {t['ink']};
    border: 2px solid transparent; border-radius: {r['sm']}px;
    padding: 4px 8px; min-height: 24px;
    selection-background-color: {t['selection']}; selection-color: {t['ink']};
}}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{ background: {t['hover']}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {t['accent_ink']}; background: {t['field']}; }}
QLineEdit[invalid="true"] {{ border-color: {t['danger']}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{ color: {t['ink_faint']}; background: {t['surface']}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{ image: url("{chevron}"); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{
    background: {t['raised']}; color: {t['ink']}; border: none; padding: 4px;
    selection-background-color: {t['selection']}; selection-color: {t['ink']};
}}
QCheckBox {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: {r['sm']}px; background: {t['field']}; border: 2px solid {t['ink_faint']}; }}
QCheckBox::indicator:hover {{ border-color: {t['ink_muted']}; }}
QCheckBox::indicator:focus {{ border-color: {t['accent_ink']}; }}
QCheckBox::indicator:checked {{ background: {t['accent']}; border-color: {t['accent']}; image: url("{check}"); }}
QCheckBox:focus {{ color: {t['accent_ink']}; }}

/* ---------- lists, tables, log ---------- */
QListWidget, QTableWidget, QTextBrowser, QPlainTextEdit {{
    background: {t['surface']}; color: {t['ink']}; border: 2px solid transparent; border-radius: {r['md']}px;
    selection-background-color: {t['selection']}; selection-color: {t['ink']};
}}
QListWidget:focus, QTableWidget:focus, QTextBrowser:focus, QPlainTextEdit:focus {{ border-color: {t['accent_ink']}; }}
QListWidget::item {{ padding: 4px 8px; }}
QListWidget::item:hover {{ background: {t['hover']}; }}
QListWidget::item:selected {{ background: {t['selection']}; color: {t['ink']}; }}
QListWidget QLineEdit {{ padding: 0px 4px; min-height: 18px; }}
QPlainTextEdit#Log {{ font-family: "{fonts.mono}"; font-size: 12px; background: {t['ground']}; padding: 8px; }}
QHeaderView::section {{
    background: {t['raised']}; color: {t['ink_muted']}; border: none; padding: 6px 8px; font-weight: 600;
}}
QTableWidget {{ gridline-color: {t['raised']}; alternate-background-color: {t['ground']}; }}
QTableCornerButton::section {{ background: {t['raised']}; border: none; }}

/* ---------- tabs ---------- */
QTabWidget::pane {{ border: none; background: transparent; top: 0px; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{
    background: transparent; color: {t['ink_muted']}; padding: 8px 16px; margin-right: 4px;
    border: none; border-bottom: 3px solid transparent;
}}
QTabBar::tab:hover {{ color: {t['ink']}; background: {t['hover']}; }}
QTabBar::tab:selected {{ color: {t['ink']}; border-bottom-color: {t['accent']}; font-weight: 600; }}
QTabBar::tab:focus {{ color: {t['accent_ink']}; }}

/* ---------- progress ---------- */
QProgressBar {{
    background: {t['field']}; border: none; border-radius: {r['sm']}px; height: 10px;
    text-align: center; color: transparent;
}}
QProgressBar::chunk {{ background: {t['accent']}; border-radius: {r['sm']}px; }}

/* ---------- scrollbars ---------- */
QScrollBar:vertical {{ background: transparent; width: 12px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 2px; }}
QScrollBar::handle {{ background: {t['pressed']}; border-radius: 4px; min-height: 32px; min-width: 32px; }}
QScrollBar::handle:hover {{ background: {t['ink_faint']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0px; height: 0px; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ---------- chrome ---------- */
QStatusBar {{ background: {t['surface']}; color: {t['ink_muted']}; }}
QStatusBar QLabel {{ color: {t['ink_muted']}; padding: 0px 8px; }}
QStatusBar QLabel[tone="error"] {{ color: {t['danger']}; }}
QStatusBar QLabel[tone="warning"] {{ color: {t['warning']}; }}
QStatusBar QLabel[tone="success"] {{ color: {t['accent_ink']}; }}
QStatusBar::item {{ border: none; }}
QSplitter::handle {{ background: {t['ground']}; }}
QSplitter::handle:hover {{ background: {t['hover']}; }}
QToolTip {{
    background: {t['ink']}; color: {t['ground']}; border: none; padding: 6px 8px;
    font-size: {fs['small']}px;
}}
QMenu {{ background: {t['raised']}; border: none; padding: 4px; }}
QMenu::item {{ padding: 6px 16px; border-radius: {r['sm']}px; }}
QMenu::item:selected {{ background: {t['selection']}; color: {t['ink']}; }}
QMessageBox QLabel {{ min-width: 320px; }}

/* ---------- progressive disclosure and help ---------- */
QPushButton[kind="section"] {{
    background: transparent; text-align: left; padding: 6px 4px;
    font-family: "{fonts.display}"; font-size: {fs['title']}px; font-weight: 600;
}}
QPushButton[kind="section"]:hover {{ background: {t['hover']}; }}
QPushButton[kind="link"] {{
    background: transparent; color: {t['accent_ink']}; padding: 2px 0px; min-height: 0px; text-align: left;
}}
QPushButton[kind="link"]:hover {{ background: transparent; text-decoration: underline; }}
QPushButton[kind="link"]:focus {{ border-color: {t['accent_ink']}; }}
QToolButton[kind="info"] {{
    background: {t['raised']}; color: {t['ink_muted']}; font-size: {fs['small']}px; font-weight: 700;
    padding: 0px; min-height: 0px; min-width: 0px; width: 16px; height: 16px; border-radius: 9px;
}}
QToolButton[kind="info"]:hover {{ background: {t['hover']}; color: {t['ink']}; }}
QLabel[role="dot"] {{ background: {t['accent_ink']}; border-radius: 4px; }}
QLabel[role="step"] {{
    font-family: "{fonts.display}"; font-weight: 600; color: {t['on_accent']}; background: {t['accent']};
    border-radius: {r['sm']}px;
}}
QLineEdit#HelpSearch {{ min-height: 28px; }}
QListWidget#HelpNav {{ background: {t['surface']}; }}
QListWidget#HelpNav::item {{ padding: 8px 12px; }}
QTextBrowser#HelpBody {{ background: {t['surface']}; padding: 8px 16px; }}
"""


def help_css(theme: str, fonts: Fonts) -> str:
    """Rich-text stylesheet for help pages and the validation report (QTextDocument CSS subset)."""
    t = THEMES[theme]
    return f"""
body {{ color: {t['ink']}; font-family: "{fonts.body}"; font-size: {FONT_SIZE['body']}px; }}
h2 {{ font-family: "{fonts.display}"; font-size: {FONT_SIZE['heading']}px; font-weight: 600; margin-top: 4px; }}
h3 {{ font-family: "{fonts.display}"; font-size: {FONT_SIZE['title']}px; font-weight: 600; margin-top: 16px; }}
p, li {{ line-height: 140%; }}
a {{ color: {t['accent_ink']}; }}
th {{ background: {t['raised']}; color: {t['ink_muted']}; }}
td {{ border-bottom: 1px solid {t['raised']}; }}
"""


class AppStyle(QProxyStyle):
    """Fusion with a 400 ms tooltip delay, short enough to feel responsive and long enough not to flicker."""

    TOOLTIP_DELAY_MS = 400

    def __init__(self) -> None:
        super().__init__(QStyleFactory.create("Fusion"))

    def styleHint(self, hint, option=None, widget=None, returnData=None):  # noqa: N802 - Qt override
        if hint == QStyle.StyleHint.SH_ToolTip_WakeUpDelay:
            return self.TOOLTIP_DELAY_MS
        return super().styleHint(hint, option, widget, returnData)
