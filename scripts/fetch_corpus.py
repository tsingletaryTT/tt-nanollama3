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

from scripts.build_gutenberg_catalogue import GUTENBERG_REPO, matches_source  # noqa: E402
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


def iter_source_rows(source: CorpusSource, limit_rows: int = 0) -> Iterator[Dict[str, object]]:
    """Stream a source's rows, normalised to ``{"text": str}`` and filtered if Gutenberg."""
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
        if not isinstance(text, str):
            continue
        yield {"text": text}
        seen += 1
        if limit_rows and seen >= limit_rows:
            return


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
                   help="Cap documents per source (0 = all). For smoke tests.")
    args = p.parse_args()

    names = args.source or sorted(SOURCES)
    for name in names:
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
