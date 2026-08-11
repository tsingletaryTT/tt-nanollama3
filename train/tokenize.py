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

from dataclasses import dataclass
from pathlib import Path
from typing import List

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
