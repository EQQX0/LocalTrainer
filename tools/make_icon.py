"""Render ui/assets/icon.svg into ui/assets/icon.ico for the desktop shortcut.

    python tools/make_icon.py

Each size is stored as a PNG inside the ICO (Windows Vista and later read these).
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, Qt
from PyQt6.QtGui import QGuiApplication, QImage, QPainter
from PyQt6.QtSvg import QSvgRenderer

ROOT = Path(__file__).resolve().parent.parent
SVG = ROOT / "ui" / "assets" / "icon.svg"
ICO = ROOT / "ui" / "assets" / "icon.ico"
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def render_png(renderer: QSvgRenderer, size: int) -> bytes:
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()
    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    return bytes(data)


def main() -> int:
    app = QGuiApplication(sys.argv)  # noqa: F841 (QPainter needs an application)
    renderer = QSvgRenderer(str(SVG))
    pngs = [render_png(renderer, s) for s in SIZES]
    header = struct.pack("<HHH", 0, 1, len(SIZES))
    offset = len(header) + 16 * len(SIZES)
    entries, blobs = b"", b""
    for size, png in zip(SIZES, pngs):
        dim = 0 if size >= 256 else size  # 0 means 256 in the ICO directory
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(png), offset + len(blobs))
        blobs += png
    ICO.write_bytes(header + entries + blobs)
    print(f"Wrote {ICO} ({ICO.stat().st_size} bytes, sizes {', '.join(map(str, SIZES))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
