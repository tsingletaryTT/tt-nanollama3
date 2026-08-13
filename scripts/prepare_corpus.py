#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Normalise fetched documents into one plain-text file per source.

Boilerplate stripping is a LICENSING step, not a cosmetic one: Project Gutenberg applies a
trademark licence to its headers and footers, while the underlying pre-1929 texts are public
domain. Removing them is what makes "public domain texts" an accurate claim.

NOTE: Pre-1997 SMALL PRINT-era boilerplate is NOT stripped. Its legal block is front matter
(marking where the book begins), not a footer, so it requires different handling than the
START/END marker model. No such document has been observed in currently pinned sources.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.corpus import SOURCES, get_source  # noqa: E402
from train.paths import shared_dir  # noqa: E402

# Marker status constants to prevent typos and ensure consistency
MARKER_BOTH = "both"
MARKER_START_ONLY = "start-only"
MARKER_END_ONLY = "end-only"
MARKER_NONE = "none"

_START_EBOOK = re.compile(r"^\*\*\*\s*START OF TH(?:E|IS) PROJECT GUTENBERG EBOOK.*$",
                          re.IGNORECASE | re.MULTILINE)
_END_EBOOK = re.compile(r"^\*\*\*\s*END OF TH(?:E|IS) PROJECT GUTENBERG EBOOK.*$",
                        re.IGNORECASE | re.MULTILINE)
_START_ETEXT = re.compile(r"^\*\*\*\s*START OF TH(?:E|IS) PROJECT GUTENBERG ETEXT.*$",
                          re.IGNORECASE | re.MULTILINE)
_END_ETEXT = re.compile(r"^\*\*\*\s*END OF TH(?:E|IS) PROJECT GUTENBERG ETEXT.*$",
                        re.IGNORECASE | re.MULTILINE)
_BLANKS = re.compile(r"\n{3,}")
_TRAILING = re.compile(r"[ \t]+$", re.MULTILINE)


class BoilerplateResult(NamedTuple):
    """Result of boilerplate stripping with status information."""
    text: str
    marker_status: str  # One of MARKER_* constants


def strip_gutenberg_boilerplate(text: str) -> BoilerplateResult:
    """Keep only what lies between the PG start and end markers, when present.

    Returns a BoilerplateResult with the stripped text and marker status.
    Marker status is one of the MARKER_* constants to make asymmetric cases visible.
    """
    found_start = False
    found_end = False

    # Try EBOOK markers first (newer format)
    start = _START_EBOOK.search(text)
    if start:
        text = text[start.end():]
        found_start = True
    else:
        # Try ETEXT markers (older format)
        start = _START_ETEXT.search(text)
        if start:
            text = text[start.end():]
            found_start = True

    # Try EBOOK markers first
    end = _END_EBOOK.search(text)
    if end:
        text = text[: end.start()]
        found_end = True
    else:
        # Try ETEXT markers
        end = _END_ETEXT.search(text)
        if end:
            text = text[: end.start()]
            found_end = True

    text = text.strip("\n")

    if found_start and found_end:
        status = MARKER_BOTH
    elif found_start and not found_end:
        status = MARKER_START_ONLY
    elif not found_start and found_end:
        status = MARKER_END_ONLY
    else:
        status = MARKER_NONE

    return BoilerplateResult(text, status)


#: Lines that are unambiguously Project Gutenberg packaging rather than the work itself.
#:
#: Deliberately narrow. These run only over the head of a document and stop at the first
#: line that does not match, because a pattern that eats real prose is far worse than one
#: that leaves a producer credit behind. "reproduced by" and "the printer's trademark" are
#: real sentences in this corpus and must not match.
#:
#: Two alternatives were added after the initial pass, once the raw text showed a credit
#: format the confirmed examples didn't cover:
#:   - ``transcribed\s+from\s+the\b`` — "Transcribed from the 1905 Chapman and Hall edition
#:     by David Price" is genuine PG packaging. Anchored on "the" immediately after
#:     "transcribed from" so it doesn't depend on "edition by" being two words (one real
#:     instance in this corpus reads "...editionby David" with the space dropped).
#:   - the email alternative — that same credit wraps onto a second physical line
#:     ("Price, email ccx074@pglaf.org"), which is not "produced by ..." or "transcribed
#:     from ..." on its own. A head-window line carrying an email address is unambiguously
#:     packaging. Written as ``.*?[\w.+-]+@[\w.-]+\.\w+`` rather than anchoring the address
#:     to the start of the line: the enclosing ``^\s*(?:...)`` wrapper only skips leading
#:     whitespace before the alternation, and the address sits mid-line after "Price,
#:     email ", so the alternative needs its own leading ``.*?`` to reach it.
#:
#:     This alternative is deliberately context-free: it matches an address anywhere in the
#:     head window without requiring a credit line before it. It is kept that way on
#:     MEASUREMENT, not on an argument about what the corpus contains. Every match of this
#:     alternative across every raw source was inspected and every one is the genuine credit
#:     continuation line (e.g. "...email ccx074@pglaf.org"), and the two modern sources were
#:     checked exhaustively rather than reasoned about: 2,119,489 tinystories documents and
#:     241,787 wikipedia_simple documents, zero touched by this alternative.
#:
#:     An earlier version of this comment justified it instead by claiming "every source
#:     here is a pre-1929 public-domain text, and email addresses did not exist when they
#:     were written". That is false, and it is the kind of false that invites someone to
#:     extend the rule: `tinystories` is 2023 GPT-generated text and `wikipedia_simple` is a
#:     live modern encyclopedia (which certainly does contain email addresses in article
#:     bodies), and both pass through `strip_front_matter`. The rule survives because it
#:     only ever runs over a document's first 40 lines, stops at the first non-matching
#:     line, and no document in these sources opens with an address — not because addresses
#:     could not appear.
#:
#:     Do not add scoping here (e.g. requiring a preceding "produced by"/"transcribed from"
#:     line) on the strength of the pre-1929 argument, which does not hold. Do re-run the
#:     count above if a new source is added: the empirical claim is what this rests on, and
#:     a source whose documents can open with an email address would break it.
#:
#: CRITICAL: ``(?-i:Produced\s+by\s+[A-Z])`` turns IGNORECASE OFF for this entire
#: alternative — not just for the ``[A-Z]`` character class. ``(?-i:...)`` is a scoped-flag
#: group; every literal and class inside it becomes case-sensitive, so "Produced", "by",
#: and "[A-Z]" are ALL matched exactly as written within this one alternative, while the
#: rest of `_FRONT_MATTER` stays case-insensitive as before. (An earlier version of this
#: comment claimed the scope "covers only [A-Z]" and left "by" written as literal
#: lowercase, expecting IGNORECASE to still apply to it from outside the group — that is
#: not how ``(?-i:...)`` works, and the comment was wrong, not the regex.)
#:
#: Two real bugs already lived in this one alternative, in order:
#:   1. The original, fully case-insensitive ``produced\s+by\s+[A-Z]`` matched ANY
#:      word-wrapped line starting with "produced by " regardless of the next word's case
#:      (`[A-Z]` under IGNORECASE matches lowercase too). This silently deleted real prose
#:      from poetry.txt in production — 12 documents lost outright, because for each the
#:      stripped line was the document's entire text. Confirmed victims included "produced
#:      by charity, or charity by faith, but the inducements to".
#:   2. The first fix, ``(?-i:[Pp]roduced\s+by\s+[A-Z])``, scoped the flag but kept "[Pp]"
#:      case-insensitive, so it *still* matched a lowercase "produced by" whenever the
#:      following word happened to be capitalised — exactly what 19th-century prose
#:      produces whenever "by" is followed by a proper noun or personification: "produced
#:      by Nature herself, without the aid of man.", "produced by God's providence alone."
#:      No live instance of this had reached the shipped corpus, but it was the same
#:      failure mode gated on the next word's case rather than closed.
#: The fix: require the literal capital "P" too. A genuine PG credit line is always
#: line-initial and capitalised ("Produced by David Price"); a wrapped prose line
#: beginning "produced by" never is. This loses no genuine credits: the lowercase form
#: "this ebook was produced by: David Edwards, Ross Cooling" still strips via the separate,
#: still-case-insensitive ``(?:this\s+)?e-?(?:book|text)\s+was\s+produced\s+by\b``
#: alternative above — it was never covered by this one.
#:
#: If either the ``P`` or the scoped flag is ever "simplified" away, one of these two bugs
#: comes back. Re-verify against real word-wrapped 19th-century prose — not just the
#: confirmed packaging examples — before touching this alternative again.
_FRONT_MATTER = re.compile(
    r"^\s*(?:"
    r"(?:this\s+)?e-?(?:book|text)\s+was\s+produced\s+by\b"
    r"|(?-i:Produced\s+by\s+[A-Z])"
    r"|there\s+are\s+several\s+editions\s+of\s+this\s+ebook\b"
    r"|various\s+characteristics\s+of\s+each\s+ebook\b"
    r"|transcriber'?s?\s+note\b"
    r"|updated\s+editions\s+will\s+replace\b"
    r"|this\s+file\s+was\s+produced\s+from\b"
    r"|transcribed\s+from\s+the\b"
    r"|.*?[\w.+-]+@[\w.-]+\.\w+"
    r")",
    re.IGNORECASE,
)

#: How far into a document front matter may appear. Beyond this it is the work, not packaging.
_FRONT_MATTER_WINDOW = 40


def strip_front_matter(text: str) -> tuple:
    """Remove Project Gutenberg packaging lines from a document's head.

    Returns ``(cleaned_text, lines_removed)``. Scans at most the first
    ``_FRONT_MATTER_WINDOW`` lines and stops permanently at the first line that is neither
    blank nor packaging — once the work has started, nothing later is removed even if it
    resembles a credit.
    """
    if not text.strip():
        return text, 0
    lines = text.split("\n")
    keep_from = 0
    removed = 0
    for i, line in enumerate(lines[:_FRONT_MATTER_WINDOW]):
        if not line.strip():
            keep_from = i + 1
            continue
        if _FRONT_MATTER.match(line):
            keep_from = i + 1
            removed += 1
            continue
        break
    if removed == 0:
        return text, 0
    return "\n".join(lines[keep_from:]).lstrip("\n"), removed


def normalise(text: str) -> str:
    """CRLF -> LF, strip trailing whitespace, collapse blank-line runs to one."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING.sub("", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip("\n")


def prepare_source(name: str, src: Path, dest: Path) -> dict:
    """Normalise one source's raw jsonl into a plain-text file.

    Returns a dict with:
    - kept: documents successfully written
    - both: documents with both START and END markers
    - start_only: documents with START marker but no END
    - end_only: documents with END marker but no START
    - none: documents with no markers
    - skipped: malformed JSON or missing text field
    - front_matter_lines: total lines of residual PG front matter stripped across all docs
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    counts = {
        "kept": 0,
        "both": 0,
        "start_only": 0,
        "end_only": 0,
        "none": 0,
        "skipped": 0,
        "front_matter_lines": 0,
    }

    with src.open("r", encoding="utf-8") as fin, dest.open("w", encoding="utf-8") as fout:
        for line in fin:
            try:
                data = json.loads(line)
                text = data.get("text", "")
            except (json.JSONDecodeError, TypeError, AttributeError):
                counts["skipped"] += 1
                continue

            if not isinstance(text, str):
                counts["skipped"] += 1
                continue

            result = strip_gutenberg_boilerplate(text)
            stripped_text, front_matter_removed = strip_front_matter(result.text)
            counts["front_matter_lines"] += front_matter_removed
            text = normalise(stripped_text)

            # Track marker status using constants
            if result.marker_status == MARKER_BOTH:
                counts["both"] += 1
            elif result.marker_status == MARKER_START_ONLY:
                counts["start_only"] += 1
            elif result.marker_status == MARKER_END_ONLY:
                counts["end_only"] += 1
            elif result.marker_status == MARKER_NONE:
                counts["none"] += 1

            if not text:
                continue

            fout.write(text)
            fout.write("\n\n")
            counts["kept"] += 1

    return counts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", default=None)
    args = p.parse_args()

    names = args.source or sorted(SOURCES)
    for name in names:
        try:
            get_source(name)
        except KeyError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        src = shared_dir("raw") / name / "text.jsonl"
        if not src.is_file():
            print(f"skipping {name}: {src} not found (run fetch_corpus.py first)")
            continue
        dest = shared_dir("corpus") / f"{name}.txt"
        counts = prepare_source(name, src, dest)
        size_mb = dest.stat().st_size / 1e6
        marker_info = (f"both: {counts['both']}, start-only: {counts['start_only']}, "
                       f"end-only: {counts['end_only']}, none: {counts['none']}")
        if counts['skipped'] > 0:
            marker_info += f", skipped: {counts['skipped']}"
        if counts['front_matter_lines'] > 0:
            marker_info += f", front-matter-lines: {counts['front_matter_lines']}"
        print(f"{name:22} {counts['kept']:>7,} docs -> {dest.name} ({size_mb:,.1f} MB) "
              f"({marker_info})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
