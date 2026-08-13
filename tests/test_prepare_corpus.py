# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Normalisation tests.

The Gutenberg boilerplate test matters for licensing, not tidiness: PG applies a trademark
licence to its headers and footers while the underlying pre-1929 text is public domain.
Stripping them is what keeps the "public domain texts" claim accurate.
"""
from scripts.prepare_corpus import (
    MARKER_BOTH,
    MARKER_NONE,
    MARKER_START_ONLY,
    MARKER_END_ONLY,
    _FRONT_MATTER,
    normalise,
    strip_front_matter,
    strip_gutenberg_boilerplate,
)

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
    assert result.marker_status == MARKER_BOTH


def test_strips_etext_markers():
    result = strip_gutenberg_boilerplate(ETEXT_HEADED)
    assert "The real epic text." in result.text
    assert "PROJECT GUTENBERG ETEXT" not in result.text
    assert "Produced by John Volunteer." not in result.text
    assert result.marker_status == MARKER_BOTH


def test_detects_start_only_marker():
    result = strip_gutenberg_boilerplate(START_ONLY)
    assert "Just the beginning." in result.text
    assert "START OF THE PROJECT GUTENBERG EBOOK" not in result.text
    assert result.marker_status == MARKER_START_ONLY


def test_detects_end_only_marker():
    result = strip_gutenberg_boilerplate(END_ONLY)
    assert "Some preliminary text." in result.text
    assert "END OF THE PROJECT GUTENBERG EBOOK" not in result.text
    assert result.marker_status == MARKER_END_ONLY


def test_leaves_text_without_markers_untouched():
    plain = "No markers here.\nJust prose."
    result = strip_gutenberg_boilerplate(plain)
    assert result.text == plain
    assert result.marker_status == MARKER_NONE


def test_normalise_collapses_carriage_returns():
    """Not just that \\r is gone -- that the surrounding content survives the collapse."""
    result = normalise("line one\r\nline two\r\n")
    assert "\r" not in result
    assert result == "line one\nline two"


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


def test_prepare_source_counts_marker_status_correctly():
    """Test that marker status counts are correct through the full pipeline.

    This test verifies the critical fix for CRITICAL A: marker statuses are
    counted correctly when documents flow through prepare_source.
    """
    import tempfile
    from pathlib import Path
    from scripts.prepare_corpus import prepare_source

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "input.jsonl"
        dest_path = Path(tmpdir) / "output.txt"

        # Create one document of each marker type
        lines = [
            # MARKER_BOTH: has both START and END markers
            '{"text": "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\\n\\nReal text.\\n\\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***"}\n',
            # MARKER_START_ONLY: has START but no END
            '{"text": "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\\n\\nText with no end marker"}\n',
            # MARKER_END_ONLY: has END but no START
            '{"text": "Some preamble text\\n\\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***"}\n',
            # MARKER_NONE: no markers at all
            '{"text": "Just plain text with no markers at all"}\n',
        ]
        src_path.write_text("".join(lines))

        counts = prepare_source("test", src_path, dest_path)

        # Verify each marker type was counted correctly
        assert counts["both"] == 1, f"Expected both=1, got {counts['both']}"
        assert counts["start_only"] == 1, f"Expected start_only=1, got {counts['start_only']}"
        assert counts["end_only"] == 1, f"Expected end_only=1, got {counts['end_only']}"
        assert counts["none"] == 1, f"Expected none=1, got {counts['none']}"
        assert counts["kept"] == 4, f"Expected kept=4, got {counts['kept']}"
        assert counts["skipped"] == 0, f"Expected skipped=0, got {counts['skipped']}"


REAL_RESIDUE = """This eBook was produced by Les Bowler.

There are several editions of this ebook in the Project Gutenberg collection.
Various characteristics of each ebook are listed to aid in selection.

Even the sandy kitten was neglected, and the story truly begins here."""


def test_strips_producer_credit_and_edition_note():
    out, removed = strip_front_matter(REAL_RESIDUE)
    assert "Les Bowler" not in out
    assert "Project Gutenberg collection" not in out
    assert "Even the sandy kitten was neglected" in out
    assert removed >= 2


def test_leaves_ordinary_prose_completely_alone():
    prose = ("The printer's trademark appeared on the flyleaf.\n"
             "These drawings would have been reproduced by modern processes.\n"
             "The story begins.")
    out, removed = strip_front_matter(prose)
    assert out == prose
    assert removed == 0


def test_stops_at_the_first_non_matching_line():
    """Once real text starts, nothing after it is ever removed."""
    doc = ("Produced by A. Volunteer\n"
           "The real story begins here.\n"
           "This eBook was produced by someone else.\n")
    out, _ = strip_front_matter(doc)
    assert "The real story begins here." in out
    assert "someone else" in out, "stripping must not resume after real text starts"


def test_does_not_scan_beyond_the_document_head():
    body = "\n".join(f"line {i}" for i in range(60))
    doc = body + "\nThis eBook was produced by Someone.\n"
    out, removed = strip_front_matter(doc)
    assert "This eBook was produced by Someone." in out
    assert removed == 0


def test_empty_and_whitespace_documents_do_not_raise():
    assert strip_front_matter("") == ("", 0)
    assert strip_front_matter("   \n\n  ")[1] == 0


def test_strips_wrapped_transcription_credit_and_email_continuation():
    """"Transcribed from the..." wraps onto a second line carrying the transcriber's email.

    Both lines are packaging (a real PG credit format found in this corpus); neither is
    part of the work, and both must go while the following prose is kept untouched.
    """
    doc = ("Transcribed from the 1905 Chapman and Hall edition by David\n"
           "Price, email ccx074@pglaf.org\n"
           "\n"
           "The real story begins here.")
    out, removed = strip_front_matter(doc)
    assert "Transcribed from the" not in out
    assert "ccx074@pglaf.org" not in out
    assert "The real story begins here." in out
    assert removed == 2


def test_strips_transcription_credit_with_missing_space():
    """A real variant in this corpus drops the space in "edition by": "editionby".

    Matching on "transcribed from the" alone (rather than requiring "edition by" as two
    words) means this typo doesn't need its own special case.
    """
    doc = "Transcribed from the 1910 Chapman and Hall editionby David\nThe real story begins here."
    out, removed = strip_front_matter(doc)
    assert "editionby" not in out
    assert "The real story begins here." in out
    assert removed == 1


def test_email_after_real_text_has_started_is_not_removed():
    """The stop-at-first-non-match guarantee must hold even for the email pattern."""
    doc = ("The real story begins here.\n"
           "Contact me at someone@example.com for details.\n")
    out, removed = strip_front_matter(doc)
    assert "someone@example.com" in out
    assert removed == 0


def test_leaves_midsentence_produced_by_prose_alone():
    """A real line from this corpus: "produced by" appears mid-sentence, not as a credit."""
    prose = "The sound was produced by the friction of a rope round the beams of a door."
    out, removed = strip_front_matter(prose)
    assert out == prose
    assert removed == 0


# --- Regression tests for a real content-loss incident -----------------------------------
#
# A blanket `re.IGNORECASE` over `produced\s+by\s+[A-Z]` made "[A-Z]" match lowercase
# letters too, so any word-wrapped line in the corpus that happened to *start* with
# "produced by " (lowercase, mid-sentence) was misclassified as a producer credit and
# deleted. This actually happened: 12 real prose lines were stripped out of poetry.txt in
# production before the fix. These tests pin the fix and must fail loudly if the scoped
# `(?-i:...)` flag is ever "simplified" away.

def test_wordwrapped_prose_lines_are_kept_verbatim():
    """Three real lines that were wrongly deleted from poetry.txt before this fix.

    Each starts with lowercase "produced by" only because of where the source text
    happened to wrap — none of them is a producer credit.
    """
    for prose in (
        "produced by charity, or charity by faith, but the inducements to",
        "produced by the friction of a rope round the beams of a door; and",
        "produced by whipping them with nettles. The object of this ceremony",
    ):
        out, removed = strip_front_matter(prose)
        assert out == prose, f"real prose was altered: {prose!r} -> {out!r}"
        assert removed == 0, f"real prose was wrongly counted as front matter: {prose!r}"


def test_genuine_producer_credits_are_still_stripped():
    """The fix must not regress detection of real credits with a capitalized name."""
    for credit in (
        "Produced by Jeroen Hellingman and the Distributed Proofreading Team",
        "Produced by David Price",
    ):
        out, removed = strip_front_matter(credit)
        assert out == ""
        assert removed == 1, f"genuine credit was not stripped: {credit!r}"


def test_front_matter_regex_does_not_match_lowercase_wrapped_produced_by():
    """Regex-level pin: a blanket IGNORECASE here silently deletes real prose.

    This asserts directly against `_FRONT_MATTER`, not just `strip_front_matter`, so a
    future edit that removes the scoped `(?-i:...)` flag fails here immediately instead of
    only showing up as missing text somewhere in a 2GB corpus.
    """
    assert _FRONT_MATTER.match("produced by charity, or charity by faith") is None


# --- Regression tests for the SECOND version of the same bug ------------------------------
#
# The first fix, `(?-i:[Pp]roduced\s+by\s+[A-Z])`, scoped IGNORECASE off but kept "[Pp]"
# case-insensitive. That still matched a lowercase "produced by" whenever the *next* word
# happened to be capitalised — exactly what 19th-century prose produces whenever "by" is
# followed by a proper noun or personification ("produced by Nature herself...", "produced
# by God's providence alone."). Every existing test above happens to use a lowercase word
# after "by", which is why this slipped through the first fix's regression tests. The real
# fix requires the literal capital "P" too: `(?-i:Produced\s+by\s+[A-Z])`.

def test_wordwrapped_prose_with_capitalised_next_word_is_kept_verbatim():
    """Prose lines that slipped past the first (incomplete) fix.

    All four start with lowercase "produced by" — not a credit — but the word right after
    "by" is capitalised (a proper noun or personification), which is exactly the shape the
    `[Pp]`-scoped version of the fix still misclassified as packaging.
    """
    for prose in (
        "produced by American manufacturers using modern machines.",
        "produced by Nature herself, without the aid of man.",
        "produced by England's finest craftsmen in that era.",
        "produced by God's providence alone.",
    ):
        out, removed = strip_front_matter(prose)
        assert out == prose, f"real prose was altered: {prose!r} -> {out!r}"
        assert removed == 0, f"real prose was wrongly counted as front matter: {prose!r}"


def test_front_matter_regex_does_not_match_lowercase_produced_by_capitalised_next_word():
    """Regex-level pin for the capitalised-next-word case, mirroring the test above it.

    This is the specific shape that made `(?-i:[Pp]roduced\\s+by\\s+[A-Z])` insufficient:
    a lowercase "produced by" followed by a capitalised word. Asserted directly against
    `_FRONT_MATTER` so a regression here fails immediately, at the regex level.
    """
    for prose in (
        "produced by American manufacturers using modern machines.",
        "produced by Nature herself, without the aid of man.",
        "produced by England's finest craftsmen in that era.",
        "produced by God's providence alone.",
    ):
        assert _FRONT_MATTER.match(prose) is None, f"should not match: {prose!r}"


def test_genuine_uppercase_credit_still_stripped_after_capital_p_fix():
    """Requiring a literal capital "P" must not regress detection of real credits."""
    for credit in (
        "Produced by Jeroen Hellingman and the Distributed Proofreading Team",
        "Produced by David Price",
        "This eBook was produced by Les Bowler.",
        "This ebook was produced by: David Edwards, Ross Cooling",
    ):
        out, removed = strip_front_matter(credit)
        assert removed == 1, f"genuine credit was not stripped: {credit!r}"
