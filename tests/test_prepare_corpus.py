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


def test_strips_gutenberg_start_and_end_markers():
    out = strip_gutenberg_boilerplate(HEADED)
    assert "The real text begins here." in out
    assert "PROJECT GUTENBERG EBOOK" not in out
    assert "Produced by Some Volunteer." not in out


def test_leaves_text_without_markers_untouched():
    plain = "No markers here.\nJust prose."
    assert strip_gutenberg_boilerplate(plain) == plain


def test_normalise_collapses_carriage_returns():
    assert "\r" not in normalise("line one\r\nline two\r\n")


def test_normalise_collapses_runs_of_blank_lines():
    assert normalise("a\n\n\n\n\nb") == "a\n\nb"


def test_normalise_strips_trailing_whitespace_per_line():
    assert normalise("a   \nb\t\n") == "a\nb"


def test_normalise_is_idempotent():
    once = normalise("a\r\n\n\n\nb   \n")
    assert normalise(once) == once
