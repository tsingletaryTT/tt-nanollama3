#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Build the corpus and tokenizer artifacts NanoLlama3 trains against.

    python scripts/build_tokenizer.py
    python scripts/build_tokenizer.py --corpus artifacts/corpus/blend.txt

By default this trains directly on the pre-built, licence-audited corpus blend at
``artifacts/corpus/blend.txt`` (Task 3/4). When ``--corpus`` points at a file that
already exists, the legacy TinyStories-only fetch/prepare pipeline is skipped entirely
and the tokenizer is trained on that file as-is -- no download, no rewrite of
``artifacts/corpus/corpus.txt``, and no truncation via ``--corpus-mb`` (a head-truncating
byte cap would amputate the blend, since it is written one source at a time in sorted
order -- see Task 5 notes). ``--corpus-mb`` only still applies to the legacy path, taken
when the given ``--corpus`` file does not exist: TinyStories is fetched and prepared
into that path, capped as before.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from convert.tokenizer import VOCAB_SIZE, load_exported, train_bpe  # noqa: E402
from train.data import fetch_corpus, prepare_corpus  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ARTIFACTS / "corpus" / "blend.txt",
                        help="Corpus file to train on (default: %(default)s). If this "
                             "file already exists it is used as-is -- no fetch, no "
                             "prepare, no --corpus-mb cap. If it does not exist, the "
                             "legacy path fetches TinyStories and prepares it into this "
                             "path, capped by --corpus-mb.")
    parser.add_argument("--corpus-mb", type=int, default=512,
                        help="Megabytes of corpus to keep when the legacy fetch-and-"
                             "prepare path runs, i.e. only when --corpus does not "
                             "already exist (default: 512). Whole lines only. Never "
                             "applied to a pre-built corpus such as the blend.")
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE,
                        help=f"Total vocabulary including specials (default: {VOCAB_SIZE}).")
    args = parser.parse_args()

    raw_dir = ARTIFACTS / "raw"
    corpus_out = args.corpus
    tok_out = ARTIFACTS / "tokenizer"

    if corpus_out.is_file():
        # A pre-built corpus (e.g. the Task 3/4 blend) already exists at this path.
        # Train directly on it: no network re-fetch, no rewrite of this file, and
        # critically no --corpus-mb truncation -- a head-truncating byte cap would
        # amputate a sorted, per-source-concatenated blend rather than sample it.
        print(f">> Using existing corpus at {corpus_out} ({corpus_out.stat().st_size:,} "
              f"bytes) -- skipping fetch/prepare and ignoring --corpus-mb")
    else:
        print(f">> Fetching TinyStories into {raw_dir}")
        raw = fetch_corpus(raw_dir, split="train")

        print(f">> Preparing corpus (cap {args.corpus_mb} MB) -> {corpus_out}")
        stats = prepare_corpus(raw, corpus_out, max_bytes=args.corpus_mb * 1024 * 1024)
        print(f"   {stats.line_count:,} lines, {stats.bytes_written:,} bytes, "
              f"truncated={stats.truncated}")

    print(f">> Training {args.vocab_size}-token BPE -> {tok_out}")
    train_bpe(corpus_out, tok_out, vocab_size=args.vocab_size, show_progress=True)

    # vocab_size given to train_bpe is a ceiling, not a promise: BPE stops early if the
    # corpus runs out of pairs worth merging. Since --corpus-mb is user-tunable, a small
    # value can silently under-shoot the target and produce an artifact whose vocabulary
    # mismatches the model config's vocab_size (train/configs/model/, selected via
    # train/sizes.py) — a failure that otherwise doesn't surface
    # until an embedding-shape mismatch much later. Check the achieved size here instead.
    achieved = load_exported(tok_out)
    achieved_size = len(achieved.get_vocab())
    print(f">> Achieved vocabulary: {achieved_size:,} (target: {args.vocab_size:,})")
    if achieved_size != args.vocab_size:
        print(
            f"\nERROR: tokenizer vocabulary is {achieved_size:,}, not the requested "
            f"{args.vocab_size:,}. This almost always means the corpus was too small "
            f"to exhaust the vocabulary cap -- BPE stops merging once it runs out of "
            f"pairs worth learning. Re-run with a larger --corpus-mb.",
            file=sys.stderr,
        )
        return 1

    print("\nDone. Artifacts:")
    print(f"  corpus:    {corpus_out}")
    print(f"  tokenizer: {tok_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
