# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Corpus tokenization. Uses a real tokenizer; no hardware, no ttml."""

from pathlib import Path

import numpy as np
import pytest

from convert.tokenizer import train_bpe
from train.tokenization import (
    TOKEN_DTYPE,
    TokenArtifactExistsError,
    TokenStats,
    _parse_args,
    tokenize_corpus,
)


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
    # 10% val, within int() truncation's rounding
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


def test_rejects_val_fraction_above_one(tiny_corpus, tiny_tokenizer, tmp_path):
    """val_fraction > 1.0 makes `split` negative and `ids[:negative]` silently drops data."""
    with pytest.raises(ValueError, match="val_fraction"):
        tokenize_corpus(tiny_corpus, tiny_tokenizer, tmp_path / "tokens", val_fraction=1.5)


def test_rejects_non_positive_chunk_lines(tiny_corpus, tiny_tokenizer, tmp_path):
    with pytest.raises(ValueError, match="chunk_lines"):
        tokenize_corpus(tiny_corpus, tiny_tokenizer, tmp_path / "tokens", chunk_lines=0)


# ---------------------------------------------------------------------------
# Regression coverage for the overwrite guard: a model's parity gate is only meaningful
# against the exact tokens it was trained and validated on, so silently regenerating
# train_ids.npy/val_ids.npy in place (e.g. from a retrained tokenizer or a different
# corpus) must never happen without the caller explicitly saying so. This is exactly the
# bug that broke tests/test_hf_parity.py and tests/test_ttml_forward.py: the tokenizer was
# retrained and artifacts/tokens re-tokenized with it, silently invalidating the v2 model's
# parity gate. See CLAUDE.md's "parity-gate-restore" entry.
# ---------------------------------------------------------------------------


def test_refuses_to_overwrite_existing_train_and_val_ids(tiny_corpus, tiny_tokenizer, tmp_path):
    out = tmp_path / "tokens"
    tokenize_corpus(tiny_corpus, tiny_tokenizer, out)
    train_before = np.load(out / "train_ids.npy").copy()
    val_before = np.load(out / "val_ids.npy").copy()

    with pytest.raises(TokenArtifactExistsError, match="train_ids.npy") as excinfo:
        tokenize_corpus(tiny_corpus, tiny_tokenizer, out)
    # Both existing artifacts are named, not just the first one found.
    assert "val_ids.npy" in str(excinfo.value)

    # The refused call must not have touched what was already there.
    assert np.array_equal(train_before, np.load(out / "train_ids.npy"))
    assert np.array_equal(val_before, np.load(out / "val_ids.npy"))


def test_refuses_to_overwrite_when_only_one_file_exists(tiny_corpus, tiny_tokenizer, tmp_path):
    """Partial state (e.g. an interrupted prior run) is refused too, not just a full pair."""
    out = tmp_path / "tokens"
    out.mkdir()
    sentinel = np.zeros(3, dtype=TOKEN_DTYPE)
    np.save(out / "val_ids.npy", sentinel)

    with pytest.raises(TokenArtifactExistsError, match="val_ids.npy"):
        tokenize_corpus(tiny_corpus, tiny_tokenizer, out)
    assert np.array_equal(sentinel, np.load(out / "val_ids.npy"))


def test_overwrite_true_permits_regeneration(tiny_corpus, tiny_tokenizer, tmp_path):
    out = tmp_path / "tokens"
    tokenize_corpus(tiny_corpus, tiny_tokenizer, out)
    # overwrite=True is an explicit, informed choice — it must succeed and produce a normal
    # TokenStats, not merely swallow the guard.
    stats = tokenize_corpus(tiny_corpus, tiny_tokenizer, out, overwrite=True)
    assert isinstance(stats, TokenStats)
    assert (out / "train_ids.npy").is_file()
    assert (out / "val_ids.npy").is_file()


def test_cli_force_flag_defaults_to_false():
    assert _parse_args([]).force is False


def test_cli_force_flag_can_be_set():
    assert _parse_args(["--force"]).force is True


def test_cli_overwrite_alias_sets_the_same_flag():
    """--overwrite is documented as an alias for --force; it must map to the same dest."""
    assert _parse_args(["--overwrite"]).force is True
