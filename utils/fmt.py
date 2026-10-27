"""Formatting helpers (metric units everywhere)."""
from __future__ import annotations


def fmt_duration(seconds: float) -> str:
    """Human-readable duration: '45 s', '12 min', '1 h 20 min'."""
    s = max(0, int(round(seconds)))
    if s < 60:
        return f"{s} s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m} min {s:02d} s" if m < 10 else f"{m} min"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"


def fmt_bytes(n: float) -> str:
    """Bytes in SI units (kB, MB, GB)."""
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"


def fmt_metric(value: float | None, digits: int = 3) -> str:
    return "–" if value is None else f"{value:.{digits}f}"
