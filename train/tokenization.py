# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Turn a prepared corpus into token-id arrays on disk.

tt-train's own ``prepare_data`` (``ttml/common/data.py:82``) encodes the whole corpus in
one ``encode(text)`` call and splits 90/10 in memory. Against our 536 MB corpus that is a
single enormous tokenizer invocation and a multi-gigabyte intermediate Python list, so we
encode in line batches instead and grow a numpy array.

Chunk size is a **memory knob, not a correctness knob** — tokenizing the same corpus with
any two chunk sizes must produce byte-identical output, which the tests assert.

No ttnn/ttml imports: this runs on any machine.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

#: What ``ttml.common.trainer.get_batch_ttml`` expects to receive.
TOKEN_DTYPE = np.uint32


@dataclass
class TokenStats:
    """What ``tokenize_corpus`` produced, for logging and the model card."""

    total_tokens: int
    train_tokens: int
    val_tokens: int
    vocab_size: int


def tokenize_corpus(
    corpus: Path,
    tokenizer_dir: Path,
    out_dir: Path,
    val_fraction: float = 0.1,
    chunk_lines: int = 50_000,
) -> TokenStats:
    """Encode ``corpus`` with the tokenizer in ``tokenizer_dir``; write train/val ``.npy``.

    The split is taken at the end of the token stream (the last ``val_fraction`` of tokens
    become validation), matching tt-train's 90/10 tail split so our numbers stay comparable
    to its runs.
    """
    from transformers import AutoTokenizer

    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError(f"val_fraction must be in [0.0, 1.0], got {val_fraction}")
    if chunk_lines <= 0:
        raise ValueError(f"chunk_lines must be > 0, got {chunk_lines}")

    corpus, tokenizer_dir, out_dir = Path(corpus), Path(tokenizer_dir), Path(out_dir)
    if not corpus.is_file():
        raise FileNotFoundError(f"corpus not found: {corpus}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)

    pieces: List[np.ndarray] = []
    batch: List[str] = []

    def _flush() -> None:
        if not batch:
            return
        # add_special_tokens=False: the corpus already carries its own `</s>` separators;
        # letting the tokenizer inject more would double them at every story boundary.
        encoded = tok(batch, add_special_tokens=False)["input_ids"]
        flat = [i for seq in encoded for i in seq]
        if flat:
            pieces.append(np.asarray(flat, dtype=TOKEN_DTYPE))
        batch.clear()

    with corpus.open("r", encoding="utf-8") as fin:
        for line in fin:
            batch.append(line.rstrip("\n"))
            if len(batch) >= chunk_lines:
                _flush()
    _flush()

    ids = np.concatenate(pieces) if pieces else np.zeros(0, dtype=TOKEN_DTYPE)
    n_val = int(len(ids) * val_fraction)
    split = len(ids) - n_val
    train_ids, val_ids = ids[:split], ids[split:]

    np.save(out_dir / "train_ids.npy", train_ids)
    np.save(out_dir / "val_ids.npy", val_ids)

    return TokenStats(
        total_tokens=int(len(ids)),
        train_tokens=int(len(train_ids)),
        val_tokens=int(len(val_ids)),
        vocab_size=int(tok.vocab_size),
    )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default="artifacts/corpus/blend.txt",
                    help="Path to the prepared corpus text file (default: %(default)s). "
                         "Matches scripts/build_tokenizer.py's default on purpose: the "
                         "documented sequence trains the tokenizer and then tokenizes "
                         "with it, and two different defaults meant step 2 looked for a "
                         "file step 1 never wrote. Pass artifacts/corpus/corpus.txt to "
                         "tokenize the legacy TinyStories-only corpus instead.")
    p.add_argument("--tokenizer", default="artifacts/tokenizer",
                    help="Directory holding the trained tokenizer.")
    p.add_argument("--out", default="artifacts/tokens",
                    help="Directory to write train_ids.npy / val_ids.npy into.")
    p.add_argument("--val-fraction", type=float, default=0.1,
                    help="Fraction of the token stream (tail) held out for validation.")
    p.add_argument("--chunk-lines", type=int, default=50_000,
                    help="Corpus lines encoded per tokenizer call (a memory knob only).")
    return p.parse_args(argv)


def main() -> int:
    args = _parse_args()
    stats = tokenize_corpus(
        Path(args.corpus),
        Path(args.tokenizer),
        Path(args.out),
        val_fraction=args.val_fraction,
        chunk_lines=args.chunk_lines,
    )
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
