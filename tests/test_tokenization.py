# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Corpus tokenization. Uses a real tokenizer; no hardware, no ttml."""

from pathlib import Path

import numpy as np
import pytest

from convert.tokenizer import train_bpe
from train.tokenize import TOKEN_DTYPE, TokenStats, tokenize_corpus


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("tok")
    corpus = d / "train.txt"
    corpus.write_text("\n".join(["the cat sat on the mat", "a dog ran fast"] * 200) + "\n",
                      encoding="utf-8")
    return train_bpe(corpus, d / "tokenizer", vocab_size=400)


@pytest.fixture(scope="module")
def tiny_corpus(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("corpus") / "corpus.txt"
    p.write_text("\n".join([f"line number {i} the cat sat" for i in range(500)]) + "\n",
                 encoding="utf-8")
    return p


def test_writes_both_splits(tiny_corpus, tiny_tokenizer, tmp_path):
    out = tmp_path / "tokens"
    tokenize_corpus(tiny_corpus, tiny_tokenizer, out)
    assert (out / "train_ids.npy").is_file()
    assert (out / "val_ids.npy").is_file()


def test_dtype_is_uint32(tiny_corpus, tiny_tokenizer, tmp_path):
    out = tmp_path / "tokens"
    tokenize_corpus(tiny_corpus, tiny_tokenizer, out)
    assert np.load(out / "train_ids.npy").dtype == TOKEN_DTYPE


def test_split_fraction_and_totals(tiny_corpus, tiny_tokenizer, tmp_path):
    out = tmp_path / "tokens"
    stats = tokenize_corpus(tiny_corpus, tiny_tokenizer, out, val_fraction=0.1)
    train = np.load(out / "train_ids.npy")
    val = np.load(out / "val_ids.npy")
    assert isinstance(stats, TokenStats)
    assert stats.train_tokens == len(train)
    assert stats.val_tokens == len(val)
    assert stats.total_tokens == len(train) + len(val)
    # 10% val, within one chunk's rounding
    assert abs(stats.val_tokens / stats.total_tokens - 0.1) < 0.02


def test_chunking_does_not_change_output(tiny_corpus, tiny_tokenizer, tmp_path):
    """Chunk size is a memory knob, never a correctness knob."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    tokenize_corpus(tiny_corpus, tiny_tokenizer, a, chunk_lines=7)
    tokenize_corpus(tiny_corpus, tiny_tokenizer, b, chunk_lines=5000)
    assert np.array_equal(np.load(a / "train_ids.npy"), np.load(b / "train_ids.npy"))
    assert np.array_equal(np.load(a / "val_ids.npy"), np.load(b / "val_ids.npy"))


def test_vocab_size_reported_from_tokenizer(tiny_corpus, tiny_tokenizer, tmp_path):
    """`stats.vocab_size` is the tokenizer's ACHIEVED vocabulary, not the target.

    The fixture corpus exhausts BPE merges far below the 400 cap — 283 in practice —
    so this asserts agreement with the tokenizer itself rather than a hardcoded
    number. `vocab_size` is a ceiling, not a promise (see Plan 1).
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tiny_tokenizer), local_files_only=True)
    stats = tokenize_corpus(tiny_corpus, tiny_tokenizer, tmp_path / "tokens")
    assert stats.vocab_size == tok.vocab_size
    assert 260 < stats.vocab_size <= 400  # above the byte alphabet, under the cap


def test_eos_survives_tokenization(tiny_tokenizer, tmp_path):
    """A `</s>` line must become the single eos id, not several ordinary tokens."""
    corpus = tmp_path / "c.txt"
    corpus.write_text("the cat sat\n</s>\nthe dog ran\n", encoding="utf-8")
    tokenize_corpus(corpus, tiny_tokenizer, tmp_path / "tokens", val_fraction=0.0)
    ids = np.load(tmp_path / "tokens" / "train_ids.npy")
    assert 2 in ids.tolist(), "eos id 2 absent — separators are not reaching the token stream"
