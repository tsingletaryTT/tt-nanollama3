# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Normalisation tests.

The Gutenberg boilerplate test matters for licensing, not tidiness: PG applies a trademark
licence to its headers and footers while the underlying pre-1929 text is public domain.
Stripping them is what keeps the "public domain texts" claim accurate.
"""
from scripts.prepare_corpus import normalise, strip_gutenberg_boilerplate

HEADED = """*** START OF THE PROJECT GUTENBERG EBOOK THE LIFE OF THE SPIDER ***

The real text begins here.
And continues.

*** END OF THE PROJECT GUTENBERG EBOOK THE LIFE OF THE SPIDER ***
Produced by Some Volunteer."""

ETEXT_HEADED = """*** START OF THE PROJECT GUTENBERG ETEXT THE ODYSSEY ***

The real epic text.
More verses.

*** END OF THE PROJECT GUTENBERG ETEXT THE ODYSSEY ***
Produced by John Volunteer."""

START_ONLY = """*** START OF THE PROJECT GUTENBERG EBOOK MY BOOK ***

Just the beginning.
No ending marker here."""

END_ONLY = """Some preliminary text.

*** END OF THE PROJECT GUTENBERG EBOOK MY BOOK ***
Produced by Someone."""


def test_strips_gutenberg_start_and_end_markers():
    result = strip_gutenberg_boilerplate(HEADED)
    assert "The real text begins here." in result.text
    assert "PROJECT GUTENBERG EBOOK" not in result.text
    assert "Produced by Some Volunteer." not in result.text
    assert result.marker_status == "both"


def test_strips_etext_markers():
    result = strip_gutenberg_boilerplate(ETEXT_HEADED)
    assert "The real epic text." in result.text
    assert "PROJECT GUTENBERG ETEXT" not in result.text
    assert "Produced by John Volunteer." not in result.text
    assert result.marker_status == "both"


def test_detects_start_only_marker():
    result = strip_gutenberg_boilerplate(START_ONLY)
    assert "Just the beginning." in result.text
    assert "START OF THE PROJECT GUTENBERG EBOOK" not in result.text
    assert result.marker_status == "start-only"


def test_detects_end_only_marker():
    result = strip_gutenberg_boilerplate(END_ONLY)
    assert "Some preliminary text." in result.text
    assert "END OF THE PROJECT GUTENBERG EBOOK" not in result.text
    assert result.marker_status == "end-only"


def test_leaves_text_without_markers_untouched():
    plain = "No markers here.\nJust prose."
    result = strip_gutenberg_boilerplate(plain)
    assert result.text == plain
    assert result.marker_status == "none"


def test_normalise_collapses_carriage_returns():
    assert "\r" not in normalise("line one\r\nline two\r\n")


def test_normalise_collapses_runs_of_blank_lines():
    assert normalise("a\n\n\n\n\nb") == "a\n\nb"


def test_normalise_strips_trailing_whitespace_per_line():
    assert normalise("a   \nb\t\n") == "a\nb"


def test_normalise_is_idempotent():
    once = normalise("a\r\n\n\n\nb   \n")
    assert normalise(once) == once


def test_prepare_source_handles_json_array():
    """Test that valid JSON that is not an object is skipped."""
    import json
    import tempfile
    from pathlib import Path
    from scripts.prepare_corpus import prepare_source

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "input.jsonl"
        dest_path = Path(tmpdir) / "output.txt"

        # Write a JSON array (not an object)
        src_path.write_text("[1, 2, 3]\n")

        counts = prepare_source("test", src_path, dest_path)
        assert counts["skipped"] == 1
        assert counts["kept"] == 0


def test_prepare_source_handles_non_string_text_field():
    """Test that documents with non-string text field are skipped."""
    import json
    import tempfile
    from pathlib import Path
    from scripts.prepare_corpus import prepare_source

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "input.jsonl"
        dest_path = Path(tmpdir) / "output.txt"

        # Write JSON with numeric text field
        src_path.write_text('{"text": 123}\n')

        counts = prepare_source("test", src_path, dest_path)
        assert counts["skipped"] == 1
        assert counts["kept"] == 0
