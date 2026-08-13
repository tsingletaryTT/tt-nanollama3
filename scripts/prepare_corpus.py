#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Normalise fetched documents into one plain-text file per source.

Boilerplate stripping is a LICENSING step, not a cosmetic one: Project Gutenberg applies a
trademark licence to its headers and footers, while the underlying pre-1929 texts are public
domain. Removing them is what makes "public domain texts" an accurate claim.
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

_START_EBOOK = re.compile(r"^\*\*\*\s*START OF TH(?:E|IS) PROJECT GUTENBERG EBOOK.*$",
                          re.IGNORECASE | re.MULTILINE)
_END_EBOOK = re.compile(r"^\*\*\*\s*END OF TH(?:E|IS) PROJECT GUTENBERG EBOOK.*$",
                        re.IGNORECASE | re.MULTILINE)
_START_ETEXT = re.compile(r"^\*\*\*\s*START OF TH(?:E|IS) PROJECT GUTENBERG ETEXT.*$",
                          re.IGNORECASE | re.MULTILINE)
_END_ETEXT = re.compile(r"^\*\*\*\s*END OF TH(?:E|IS) PROJECT GUTENBERG ETEXT.*$",
                        re.IGNORECASE | re.MULTILINE)
_START_SMALL_PRINT = re.compile(r"^\*+\s*START OF THE SMALL PRINT.*$",
                                re.IGNORECASE | re.MULTILINE)
_END_SMALL_PRINT = re.compile(r"^\*+.*END\*.*THE SMALL PRINT.*$",
                              re.IGNORECASE | re.MULTILINE)
_BLANKS = re.compile(r"\n{3,}")
_TRAILING = re.compile(r"[ \t]+$", re.MULTILINE)


class BoilerplateResult(NamedTuple):
    """Result of boilerplate stripping with status information."""
    text: str
    marker_status: str  # "both", "start-only", "end-only", "none"


def strip_gutenberg_boilerplate(text: str) -> BoilerplateResult:
    """Keep only what lies between the PG start and end markers, when present.

    Returns a BoilerplateResult with the stripped text and marker status.
    Marker status is "both", "start-only", "end-only", or "none" to make
    asymmetric cases visible.
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
        else:
            # Try SMALL PRINT block
            end = _END_SMALL_PRINT.search(text)
            if end:
                text = text[: end.start()]
                found_end = True

    text = text.strip("\n")

    if found_start and found_end:
        status = "both"
    elif found_start and not found_end:
        status = "start-only"
    elif not found_start and found_end:
        status = "end-only"
    else:
        status = "none"

    return BoilerplateResult(text, status)


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
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    counts = {
        "kept": 0,
        "both": 0,
        "start_only": 0,
        "end_only": 0,
        "none": 0,
        "skipped": 0,
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
            text = normalise(result.text)

            # Track marker status
            if result.marker_status == "both":
                counts["both"] += 1
            elif result.marker_status == "start_only":
                counts["start_only"] += 1
            elif result.marker_status == "end_only":
                counts["end_only"] += 1
            else:  # "none"
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
        print(f"{name:22} {counts['kept']:>7,} docs -> {dest.name} ({size_mb:,.1f} MB) "
              f"({marker_info})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
