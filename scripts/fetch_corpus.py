#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Fetch each registered source's text at its pinned revision.

Writes one JSON object per document to artifacts/raw/<source>/text.jsonl. Nothing here is
committed: the project ships a recipe, not a corpus, because CC-BY-SA-3.0 and
CDLA-Sharing-1.0 are not obviously compatible terms on one redistributed work.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_gutenberg_catalogue import GUTENBERG_REPO, matches_source, gutenberg_sources  # noqa: E402
from train.corpus import SOURCES, CorpusSource, get_source  # noqa: E402
from train.paths import shared_dir  # noqa: E402

#: Column holding the document body, per dataset.
TEXT_COLUMN = {
    "sedthh/gutenberg_english": "TEXT",
    "roneneldan/TinyStories": "text",
    "wikimedia/wikipedia": "text",
    "biglam/gutenberg-poetry-corpus": "line",
}


def write_documents(rows: Iterable[Dict[str, object]], dest: Path) -> int:
    """Write ``{"text": ...}`` per line, skipping empties. Returns documents written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with dest.open("w", encoding="utf-8") as fh:
        for row in rows:
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1
    return written


def fetch_gutenberg_batch(sources: list[CorpusSource], limit_rows: int = 0) -> Dict[str, int]:
    """Fetch multiple Gutenberg sources in one streaming pass. Returns {source_name: count}.

    Stop after reading N rows from the source (0 = no limit). This is a COST bound, not a
    document count: rows that are filtered out — non-matching books, blank text, malformed
    metadata — still consume the budget, so fewer than N documents may be written. The budget
    is shared across all sources in the single pass.
    """
    from datasets import load_dataset

    # Verify all sources are from the same Gutenberg repo at the same revision
    if not sources or not all(s.hf_repo == GUTENBERG_REPO for s in sources):
        raise ValueError("fetch_gutenberg_batch requires all sources to be from gutenberg_english")

    revision = sources[0].hf_revision
    if not all(s.hf_revision == revision for s in sources):
        raise ValueError("All Gutenberg sources must use the same revision")

    # Open destination files for each source
    dest_files = {}
    file_handles = {}
    for src in sources:
        dest = shared_dir("raw") / src.name / "text.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest_files[src.name] = dest
        file_handles[src.name] = dest.open("w", encoding="utf-8")

    try:
        # Stream the dataset once, routing each row to matching sources
        kwargs = {"split": sources[0].hf_split, "revision": revision, "streaming": True}
        ds = load_dataset(GUTENBERG_REPO, **kwargs)

        counts = {src.name: 0 for src in sources}
        seen = 0

        for row in ds:
            seen += 1
            if limit_rows and seen > limit_rows:
                break

            md = row.get("METADATA")
            if isinstance(md, str):
                try:
                    md = json.loads(md)
                except json.JSONDecodeError:
                    continue
            if not isinstance(md, dict):
                continue

            text = row.get("TEXT")
            if not isinstance(text, str) or not text.strip():
                continue

            # Check which sources match this row
            for src in sources:
                if matches_source(md, src):
                    file_handles[src.name].write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    counts[src.name] += 1

        return counts

    finally:
        # Close all file handles
        for fh in file_handles.values():
            if not fh.closed:
                fh.close()


def iter_source_rows(source: CorpusSource, limit_rows: int = 0) -> Iterator[Dict[str, object]]:
    """Stream a source's rows, normalised to ``{"text": str}`` and filtered if Gutenberg.

    Stop after reading N rows from the source (0 = no limit). This is a cost bound, not a
    document count: rows that are filtered out — non-matching books, blank text, malformed
    metadata — still consume the budget, so fewer than N documents may be yielded.
    """
    from datasets import load_dataset

    column = TEXT_COLUMN.get(source.hf_repo)
    if column is None:
        raise ValueError(
            f"no text column registered for {source.hf_repo}; add it to TEXT_COLUMN"
        )

    kwargs = {"split": source.hf_split, "revision": source.hf_revision, "streaming": True}
    if source.hf_config:
        kwargs["name"] = source.hf_config
    ds = load_dataset(source.hf_repo, **kwargs)

    seen = 0
    for row in ds:
        seen += 1
        if limit_rows and seen > limit_rows:
            return

        if source.hf_repo == GUTENBERG_REPO:
            md = row.get("METADATA")
            if isinstance(md, str):
                try:
                    md = json.loads(md)
                except json.JSONDecodeError:
                    continue
            if not isinstance(md, dict) or not matches_source(md, source):
                continue
        text = row.get(column)
        if not isinstance(text, str) or not text.strip():
            continue
        yield {"text": text}


def fetch_source(source: CorpusSource, dest: Optional[Path] = None,
                 limit_rows: int = 0) -> int:
    """Fetch one source to ``artifacts/raw/<name>/text.jsonl``. Returns documents written."""
    target = dest or (shared_dir("raw") / source.name / "text.jsonl")
    return write_documents(iter_source_rows(source, limit_rows), target)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", default=None,
                   help="Source name (repeatable). Default: all registered sources.")
    p.add_argument("--limit-rows", type=int, default=0,
                   help="Stop after reading N rows from the source (0 = no limit). "
                        "This is a cost bound, not a document count: rows that are filtered out "
                        "— non-matching books, blank text, malformed metadata — still consume the "
                        "budget, so fewer than N documents may be written. For smoke tests.")
    args = p.parse_args()

    names = args.source or sorted(SOURCES)

    # Separate Gutenberg sources from others
    gutenberg_sources_all = gutenberg_sources()
    requested_gutenberg = [n for n in names if n in gutenberg_sources_all]
    requested_other = [n for n in names if n not in gutenberg_sources_all]

    # Fetch multiple Gutenberg sources in one pass if more than one is requested
    if len(requested_gutenberg) > 1:
        try:
            sources_objs = [get_source(name) for name in requested_gutenberg]
            print(f"fetching {len(sources_objs)} Gutenberg sources in one pass ...", flush=True)
            counts = fetch_gutenberg_batch(sources_objs, limit_rows=args.limit_rows)
            for name, count in counts.items():
                print(f"  {name}: {count:,} documents")
                if count == 0:
                    print(f"  WARNING: {name} produced no documents", file=sys.stderr)
        except KeyError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    else:
        # Fetch single Gutenberg or non-Gutenberg sources individually
        for name in requested_gutenberg:
            try:
                src = get_source(name)
            except KeyError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1
            print(f"fetching {name} from {src.hf_repo}@{src.hf_revision} ...", flush=True)
            n = fetch_source(src, limit_rows=args.limit_rows)
            print(f"  {n:,} documents")
            if n == 0:
                print(f"  WARNING: {name} produced no documents", file=sys.stderr)

    # Fetch non-Gutenberg sources individually
    for name in requested_other:
        try:
            src = get_source(name)
        except KeyError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"fetching {name} from {src.hf_repo}@{src.hf_revision} ...", flush=True)
        n = fetch_source(src, limit_rows=args.limit_rows)
        print(f"  {n:,} documents")
        if n == 0:
            print(f"  WARNING: {name} produced no documents", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
