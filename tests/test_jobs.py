"""Worker output parsing: log lines, progress bars and JSON events."""
from __future__ import annotations

from core.gpu_monitor import parse_smi
from core.jobs import EVENT_PREFIX, LogLine, OutputParser


def test_parser_splits_lines_events_and_progress() -> None:
    p = OutputParser()
    out = p.feed(b"hello\n 10%|##\r 50%|####\r100%|########\n")
    out += p.feed(("\n" + EVENT_PREFIX + '{"type": "epoch", "epoch": 1}\n').encode())
    out += p.feed(b"\x1b[34mcolored\x1b[0m tail")
    out += p.flush()
    assert out[0] == LogLine("hello", False)
    assert out[1] == LogLine(" 10%|##", False)
    assert out[2] == LogLine(" 50%|####", True)       # replaces the 10 % line
    assert out[3] == LogLine("100%|########", True)  # final state replaces the 50 % line
    assert out[4] == {"type": "epoch", "epoch": 1}
    assert out[5] == LogLine("colored tail", False)


def test_parser_handles_split_utf8_and_chunks() -> None:
    p = OutputParser()
    data = "mAP ━━━ 100%\n".encode()
    out = []
    for i in range(len(data)):  # one byte at a time
        out += p.feed(data[i:i + 1])
    assert out == [LogLine("mAP ━━━ 100%", False)]


def test_parse_smi() -> None:
    assert parse_smi("NVIDIA GeForce RTX 5070, 3120, 12227, 87\n") == ("NVIDIA GeForce RTX 5070", 3120, 12227, 87)
    assert parse_smi("garbage") is None
