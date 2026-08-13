# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Fetch and prepare the tt-tnt training corpus.

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

#: Pinned commit SHA of the dataset repo. This tokenizer defines the vocabulary of a
#: model we intend to publish, so the corpus it is trained on must be reproducible —
#: without a pin, a future Hub-side change to the file would silently retrain a
#: different vocabulary from the one shipped. Verified current as of 2026-08-11.
CORPUS_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"

#: TinyStories' own story delimiter: a line containing exactly this text separates one
#: story from the next in the raw corpus. Left as ordinary text it wastes a vocabulary
#: slot on the subword "endoftext" and the model never sees an actual stop token,
#: so `prepare_corpus` rewrites it to the tokenizer's `</s>` special token instead.
DOCUMENT_SEPARATOR = "<|endoftext|>"

#: The literal replacement text for `DOCUMENT_SEPARATOR` lines. Matches
#: ``convert.tokenizer.SPECIAL_TOKENS[2]`` — kept as a plain string here (rather than
#: importing convert/) so this module stays free of any dependency beyond the stdlib.
EOS_TOKEN_TEXT = "</s>"


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
        revision=CORPUS_REVISION,
        local_dir=str(dest_dir),
    )
    return Path(downloaded)


def prepare_corpus(src: Path, dest: Path, max_bytes: Optional[int] = None) -> CorpusStats:
    """Normalize ``src`` into ``dest`` and report what was written.

    Normalization is deliberately minimal — the tokenizer should see text close to what
    the model will be served: CRLF collapsed to LF, blank lines dropped (they carry no
    signal and inflate the corpus), trailing whitespace stripped.

    A line that is *exactly* ``DOCUMENT_SEPARATOR`` (after stripping) is TinyStories'
    story delimiter, not prose, and is rewritten to the literal text ``</s>`` so the
    tokenizer maps it to the eos special token. Only an exact match is rewritten — a
    line that merely *contains* the separator inside other text is left untouched, since
    a substring replace would corrupt prose that happens to mention it. The rewritten
    line counts toward ``line_count`` and the byte budget exactly like any other line.

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
            if line == DOCUMENT_SEPARATOR:
                line = EOS_TOKEN_TEXT
            encoded_len = len(line.encode("utf-8")) + 1  # +1 for the newline
            if max_bytes is not None and written + encoded_len > max_bytes:
                truncated = True
                break
            fout.write(line + "\n")
            written += encoded_len
            lines += 1

    return CorpusStats(bytes_written=written, line_count=lines, truncated=truncated)
