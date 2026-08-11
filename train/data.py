# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Fetch and prepare the NanoLlama3 training corpus.

The corpus is TinyStories (``roneneldan/TinyStories``) — small, clean, and known to
produce coherent output at ~22M parameters, which is the scale this model targets.
We use the V2/GPT-4 variant, the higher-quality regeneration of the original.

This module deliberately imports nothing from Tenstorrent: corpus prep must run on any
machine, including one with no hardware and no tt-metal checkout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

#: The Hub dataset holding the corpus.
CORPUS_REPO = "roneneldan/TinyStories"

#: Filenames per split, verified present in the dataset repo.
CORPUS_FILES = {
    "train": "TinyStoriesV2-GPT4-train.txt",
    "valid": "TinyStoriesV2-GPT4-valid.txt",
}


@dataclass
class CorpusStats:
    """What ``prepare_corpus`` produced, for logging and for the model card."""

    bytes_written: int
    line_count: int
    truncated: bool


def fetch_corpus(dest_dir: Path, split: str = "train") -> Path:
    """Download the TinyStories file for ``split`` into ``dest_dir``.

    Returns the local path. If the file is already present it is returned unchanged —
    the corpus is ~2 GB and re-downloading it is never what the caller wants.
    """
    if split not in CORPUS_FILES:
        raise ValueError(f"split must be one of {sorted(CORPUS_FILES)}, not {split!r}")

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    local = dest_dir / CORPUS_FILES[split]
    if local.exists():
        return local

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=CORPUS_REPO,
        filename=CORPUS_FILES[split],
        repo_type="dataset",
        local_dir=str(dest_dir),
    )
    return Path(downloaded)


def prepare_corpus(src: Path, dest: Path, max_bytes: Optional[int] = None) -> CorpusStats:
    """Normalize ``src`` into ``dest`` and report what was written.

    Normalization is deliberately minimal — the tokenizer should see text close to what
    the model will be served: CRLF collapsed to LF, blank lines dropped (they carry no
    signal and inflate the corpus), trailing whitespace stripped.

    ``max_bytes`` caps the output. The cap is applied on **whole lines only**: a partial
    final line would introduce a token boundary that never occurs in real text.
    """
    src, dest = Path(src), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    lines = 0
    truncated = False

    with src.open("r", encoding="utf-8", errors="replace") as fin, \
            dest.open("w", encoding="utf-8", newline="\n") as fout:
        for raw in fin:
            line = raw.replace("\r\n", "\n").rstrip()
            if not line:
                continue
            encoded_len = len(line.encode("utf-8")) + 1  # +1 for the newline
            if max_bytes is not None and written + encoded_len > max_bytes:
                truncated = True
                break
            fout.write(line + "\n")
            written += encoded_len
            lines += 1

    return CorpusStats(bytes_written=written, line_count=lines, truncated=truncated)
