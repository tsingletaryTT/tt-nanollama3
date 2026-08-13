#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Build the corpus and tokenizer artifacts NanoLlama3 trains against.

    python scripts/build_tokenizer.py                                  # the nine-source blend
    python scripts/build_tokenizer.py --corpus artifacts/corpus/corpus.txt  # TinyStories only

Two paths, and this script will not silently swap one for the other.

**The blend** (default). ``artifacts/corpus/blend.txt`` is the licence-audited nine-source
blend built by ``scripts/blend_corpus.py``. When ``--corpus`` names a file that already
exists, it is trained on as-is: no download, no rewrite, and no truncation via
``--corpus-mb`` (a head-truncating byte cap would amputate a blend written one source at a
time in sorted order, not sample it).

**The legacy TinyStories-only path**, taken when ``--corpus`` names a file that does NOT
exist: TinyStories is fetched and prepared into that path, capped by ``--corpus-mb``.

It refuses to take the legacy path when the target is named ``blend.txt``. That
combination -- default ``--corpus``, no blend built yet -- used to fetch TinyStories and
write it INTO ``artifacts/corpus/blend.txt``. Every later run then found the file, printed
"Using existing corpus ... skipping fetch/prepare", and trained on TinyStories forever
while the operator believed it was training on the nine-source blend. Nothing downstream
could tell the difference: a corpus is just a text file, and the name was the only claim
being made about its contents. So the name is now defended.
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

#: The blend's filename, reserved for output of ``scripts/blend_corpus.py``. Compared by
#: name rather than by full path so that a copy of the pipeline pointed at another
#: directory keeps the same guarantee.
BLEND_NAME = "blend.txt"

#: Where the legacy TinyStories-only path is meant to write. Named here so the error
#: message can offer it rather than leaving the operator to guess.
LEGACY_CORPUS = ARTIFACTS / "corpus" / "corpus.txt"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path,
                        default=ARTIFACTS / "corpus" / BLEND_NAME,
                        help="Corpus file to train on (default: %(default)s). If this "
                             "file already exists it is used as-is -- no fetch, no "
                             "prepare, no --corpus-mb cap. If it does not exist, the "
                             "legacy TinyStories-only path fetches and prepares into this "
                             f"path, capped by --corpus-mb -- unless it is named "
                             f"{BLEND_NAME}, which only scripts/blend_corpus.py may write.")
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
    elif corpus_out.name == BLEND_NAME:
        # The legacy path may never write this name. See the module docstring: a
        # TinyStories-only file called blend.txt is indistinguishable from the real blend
        # to every later run, including this script's own "already exists" branch.
        print(
            f"\nERROR: {corpus_out} does not exist, and the legacy TinyStories-only path "
            f"is not allowed to create a file named {BLEND_NAME} -- it would be a "
            f"TinyStories corpus wearing the blend's name, and every later run would "
            f"train on it believing it was the nine-source blend.\n\n"
            f"Build the blend first:\n"
            f"    python scripts/fetch_corpus.py\n"
            f"    python scripts/prepare_corpus.py\n"
            f"    python scripts/measure_corpus.py\n"
            f"    python scripts/blend_corpus.py\n\n"
            f"Or train on TinyStories alone, named honestly:\n"
            f"    python scripts/build_tokenizer.py --corpus {LEGACY_CORPUS} "
            f"--corpus-mb {args.corpus_mb}\n",
            file=sys.stderr,
        )
        return 1
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
