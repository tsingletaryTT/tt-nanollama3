#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Build a metadata-only catalogue of sedthh/gutenberg_english, and verify our authors exist.

WHY METADATA ONLY. The dataset is 48,284 books and 10.75 GB. Everything this plan needs to
decide -- do the named authors exist, how many books does each slice select, is any slice
short -- lives in the METADATA column. Reading one column instead of the whole dataset turns
a multi-gigabyte download into a small one, and it happens BEFORE any text is fetched, so a
composition that cannot be satisfied fails early and cheaply.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.corpus import SOURCES, CorpusSource  # noqa: E402
from train.paths import shared_dir  # noqa: E402

GUTENBERG_REPO = "sedthh/gutenberg_english"
CATALOGUE_NAME = "gutenberg_catalogue.jsonl"


def _norm(value: Optional[str]) -> str:
    return (value or "").casefold()


def matches_source(metadata: Dict[str, object], source: CorpusSource) -> bool:
    """True when a catalogue row belongs to ``source``.

    Author and bookshelf selectors are a UNION: a source listing both takes rows matching
    either. Matching is case-insensitive substring, because Gutenberg records names as
    "Dunsany, Lord" and "Fabre, Jean-Henri" and a registry entry should not have to
    reproduce the punctuation exactly.

    A source with no selectors matches nothing. That is deliberate: an empty selector on a
    48,000-book dataset would silently pull everything.
    """
    if not source.authors and not source.bookshelves:
        return False
    authors = _norm(metadata.get("authors"))
    shelves = _norm(metadata.get("bookshelves"))
    for wanted in source.authors:
        if _norm(wanted) in authors:
            return True
    for wanted in source.bookshelves:
        if _norm(wanted) in shelves:
            return True
    return False


def gutenberg_sources() -> Dict[str, CorpusSource]:
    """Registry sources served by the Gutenberg dataset."""
    return {n: s for n, s in SOURCES.items() if s.hf_repo == GUTENBERG_REPO}


def default_gutenberg_revision() -> str:
    """The pinned revision shared by every Gutenberg-backed registry source.

    Used as the CLI default so a plain ``build_gutenberg_catalogue.py`` run catalogues the
    same revision ``fetch_corpus.py`` actually fetches, instead of drifting to whatever
    "main" currently points at. Raises if the registry's Gutenberg sources ever disagree on
    revision -- that would mean there is no single honest default to fall back to.
    """
    revisions = {s.hf_revision for s in gutenberg_sources().values()}
    if len(revisions) != 1:
        raise ValueError(f"Gutenberg sources disagree on pinned revision: {sorted(revisions)}")
    return next(iter(revisions))


def iter_metadata(revision: str) -> Iterable[Dict[str, object]]:
    """Yield one metadata dict per book, reading only the METADATA column.

    NOTE: We project the METADATA column directly via PyArrow, not load_dataset().
    The obvious approach of load_dataset(..., columns=["METADATA"], streaming=True)
    raises ValueError on datasets 2.14.6: a casting bug when projecting columns in
    streaming mode. PyArrow projection works and costs ~6-8 MB for all 48,284 rows,
    vs. ~15 GB if full TEXT rows crossed the network. This approach recovers the
    stated goal: reading one column instead of the whole dataset.
    """
    import pyarrow.parquet as pq
    import fsspec
    from huggingface_hub import HfApi, hf_hub_url

    # List all parquet shards in the dataset at this revision.
    api = HfApi()
    repo_files = api.list_repo_files(repo_id=GUTENBERG_REPO, revision=revision,
                                      repo_type="dataset")
    parquet_files = [
        fname
        for fname in repo_files
        if isinstance(fname, str) and fname.startswith("data/train-") and fname.endswith(".parquet")
    ]
    parquet_files.sort()

    # Fail loudly if no shards were found — this is a loader problem, not an empty result.
    if not parquet_files:
        raise RuntimeError(
            f"No parquet shards found in {GUTENBERG_REPO} (revision={revision}). "
            f"Expected files matching 'data/train-*.parquet'. "
            f"Dataset layout may have changed, or revision is invalid."
        )

    # Read each parquet shard with column projection. Get HTTPS URLs for each file.
    for fname in parquet_files:
        url = hf_hub_url(repo_id=GUTENBERG_REPO, filename=fname, revision=revision,
                         repo_type="dataset")
        # Use fsspec to open the remote parquet file with no cache (cache_type="none"),
        # then read with PyArrow column projection. This eliminates ~194 MB of wasted
        # TEXT column reads from fsspec's default 5 MiB read-ahead, reducing network
        # traffic from ~213 MB to ~6-8 MB for all 48,284 rows.
        with fsspec.open(url, "rb", cache_type="none") as f:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=1024, columns=["METADATA"]):
                for row in batch.to_pylist():
                    raw = row.get("METADATA")
                    if isinstance(raw, str):
                        try:
                            raw = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                    if isinstance(raw, dict):
                        yield raw


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--revision", default=default_gutenberg_revision(),
                   help="Dataset revision to catalogue (default: the registry's pinned "
                        "Gutenberg revision, not 'main').")
    p.add_argument("--out", type=Path, default=None,
                   help="Catalogue path (default: artifacts/raw/gutenberg_catalogue.jsonl)")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after N rows (0 = all). For smoke-testing the pipeline.")
    args = p.parse_args()

    out = args.out or (shared_dir("raw") / CATALOGUE_NAME)
    out.parent.mkdir(parents=True, exist_ok=True)

    sources = gutenberg_sources()
    counts = {name: 0 for name in sources}
    total = 0

    with out.open("w", encoding="utf-8") as fh:
        for md in iter_metadata(args.revision):
            total += 1
            record = {
                "text_id": md.get("text_id"),
                "title": md.get("title"),
                "authors": md.get("authors"),
                "bookshelves": md.get("bookshelves"),
                "subjects": md.get("subjects"),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            for name, src in sources.items():
                if matches_source(md, src):
                    counts[name] += 1
            if args.limit and total >= args.limit:
                break

    print(f"catalogued {total:,} books -> {out}")
    print()
    print(f"{'source':22} {'books':>7}  selectors")
    print("-" * 60)
    for name, src in sorted(sources.items()):
        sel = ", ".join(src.authors + src.bookshelves)[:34]
        print(f"{name:22} {counts[name]:>7}  {sel}")

    missing = [n for n, c in counts.items() if c == 0]
    if missing:
        print()
        print(f"WARNING: these sources selected ZERO books: {missing}")
        print("The composition assumes they exist. Investigate before fetching text.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
