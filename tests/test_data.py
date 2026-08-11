# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Corpus preparation behavior. No network, no Tenstorrent dependencies."""

from pathlib import Path

from train.data import CorpusStats, DOCUMENT_SEPARATOR, prepare_corpus


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_prepare_normalizes_crlf(tmp_path: Path):
    src = _write(tmp_path, "src.txt", "alpha\r\nbeta\r\n")
    dest = tmp_path / "out.txt"
    prepare_corpus(src, dest)
    assert "\r" not in dest.read_text(encoding="utf-8")


def test_prepare_drops_blank_lines(tmp_path: Path):
    src = _write(tmp_path, "src.txt", "alpha\n\n\nbeta\n")
    dest = tmp_path / "out.txt"
    stats = prepare_corpus(src, dest)
    assert dest.read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert stats.line_count == 2


def test_prepare_reports_stats(tmp_path: Path):
    src = _write(tmp_path, "src.txt", "alpha\nbeta\n")
    dest = tmp_path / "out.txt"
    stats = prepare_corpus(src, dest)
    assert isinstance(stats, CorpusStats)
    assert stats.bytes_written == dest.stat().st_size
    assert stats.truncated is False


def test_prepare_truncates_at_max_bytes(tmp_path: Path):
    # Ten lines of 10 bytes each ("xxxxxxxxx\n"); cap at 25 bytes -> 2 whole lines.
    src = _write(tmp_path, "src.txt", "".join("x" * 9 + "\n" for _ in range(10)))
    dest = tmp_path / "out.txt"
    stats = prepare_corpus(src, dest, max_bytes=25)
    assert stats.truncated is True
    assert stats.line_count == 2
    assert stats.bytes_written <= 25


def test_prepare_never_splits_a_line(tmp_path: Path):
    src = _write(tmp_path, "src.txt", "short\n" + "y" * 100 + "\n")
    dest = tmp_path / "out.txt"
    prepare_corpus(src, dest, max_bytes=20)
    # Every written line must be complete.
    for line in dest.read_text(encoding="utf-8").splitlines():
        assert line in ("short", "y" * 100)


def test_prepare_rewrites_separator_line_to_eos(tmp_path: Path):
    """A line that is exactly the TinyStories separator becomes the literal `</s>`."""
    src = _write(tmp_path, "src.txt", f"alpha\n{DOCUMENT_SEPARATOR}\nbeta\n")
    dest = tmp_path / "out.txt"
    prepare_corpus(src, dest)
    assert dest.read_text(encoding="utf-8") == "alpha\n</s>\nbeta\n"


def test_prepare_leaves_embedded_separator_text_alone(tmp_path: Path):
    """A line merely containing the separator inside other text is untouched.

    Substring-replacing would corrupt prose that happens to mention the delimiter; only
    an exact-match line (the real story boundary) is rewritten.
    """
    line = f"She said {DOCUMENT_SEPARATOR} out loud, which confused everyone."
    src = _write(tmp_path, "src.txt", line + "\n")
    dest = tmp_path / "out.txt"
    prepare_corpus(src, dest)
    assert dest.read_text(encoding="utf-8") == line + "\n"


def test_prepare_counts_rewritten_separator_line(tmp_path: Path):
    """The rewritten `</s>` line counts toward line_count like any other line."""
    src = _write(tmp_path, "src.txt", f"alpha\n{DOCUMENT_SEPARATOR}\nbeta\n")
    dest = tmp_path / "out.txt"
    stats = prepare_corpus(src, dest)
    assert stats.line_count == 3
