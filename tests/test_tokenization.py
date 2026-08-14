# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Corpus tokenization. Uses a real tokenizer; no hardware, no ttml."""

import json
from pathlib import Path

import numpy as np
import pytest

from convert.tokenizer import train_bpe
from train.tokenization import (
    TOKEN_DTYPE,
    TokenArtifactExistsError,
    TokenStats,
    _parse_args,
    _tokenize_stratified,
    encode_batch,
    source_word_counts_from_manifest,
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


# ---------------------------------------------------------------------------
# Stratified split: proportional per-source holdout, not one tail of the whole stream.
#
# THE BUG THIS COVERS. The default tail-of-the-whole-stream split assumes a homogeneous
# corpus. scripts/blend_corpus.py writes sources concatenated in sorted-name order, so on
# the real nine-source blend the tail is entirely wikipedia_simple (it sorts last) --
# val_ids.npy ended up 100% Simple English Wikipedia despite the corpus being ~94.5%
# something else, and every validation loss reported for that model was measuring domain
# transfer, not learning progress. These tests use a synthetic two-source corpus small
# enough to hand-verify the expected split.
# ---------------------------------------------------------------------------


def _two_source_lines():
    """20 lines of "alpha" text, then 20 lines of "zebra" text -- sorted-name order.

    Each line has a fixed, known word count (4) so the exact word offset of every line
    boundary is easy to state and check by hand.
    """
    alpha = [f"alpha word number {i}" for i in range(20)]
    zebra = [f"zebra word number {i}" for i in range(20)]
    return alpha, zebra


@pytest.fixture(scope="module")
def two_source_corpus(tmp_path_factory) -> Path:
    alpha, zebra = _two_source_lines()
    p = tmp_path_factory.mktemp("stratified_corpus") / "blend.txt"
    p.write_text("\n".join(alpha + zebra) + "\n", encoding="utf-8")
    return p


#: Word counts matching _two_source_lines() exactly: 20 lines * 4 words/line each.
TWO_SOURCE_WORD_COUNTS = {"alpha": 80, "zebra": 80}


def test_stratified_split_holds_out_from_every_source(two_source_corpus, tiny_tokenizer,
                                                        tmp_path):
    """val_ids must contain tokens from BOTH sources, not just whichever sorts last.

    This is the direct regression check for the bug: with the default split on a corpus
    shaped like this (two unequal-looking sources concatenated), the tail would be 100%
    "zebra". Stratified must not do that.
    """
    from transformers import AutoTokenizer

    out = tmp_path / "tokens"
    stats = tokenize_corpus(two_source_corpus, tiny_tokenizer, out, val_fraction=0.5,
                            source_word_counts=TWO_SOURCE_WORD_COUNTS)
    assert stats.source_splits is not None
    assert set(stats.source_splits) == {"alpha", "zebra"}
    for name in ("alpha", "zebra"):
        assert stats.source_splits[name]["val_tokens"] > 0, (
            f"{name} contributed zero val tokens -- the split is not stratified"
        )

    # Cross-check against directly tokenizing each source's own text: the stratified val
    # array is exactly each source's own tail-half of tokens, concatenated in sorted order.
    tok = AutoTokenizer.from_pretrained(str(tiny_tokenizer), local_files_only=True)
    alpha_lines, zebra_lines = _two_source_lines()
    alpha_ids = encode_batch(alpha_lines, tok)
    zebra_ids = encode_batch(zebra_lines, tok)
    expected_val = np.concatenate([
        alpha_ids[len(alpha_ids) - int(len(alpha_ids) * 0.5):],
        zebra_ids[len(zebra_ids) - int(len(zebra_ids) * 0.5):],
    ])
    assert np.array_equal(np.load(out / "val_ids.npy"), expected_val)


def test_stratified_split_train_ids_exclude_the_held_out_tails(two_source_corpus,
                                                                 tiny_tokenizer, tmp_path):
    from transformers import AutoTokenizer

    out = tmp_path / "tokens"
    tokenize_corpus(two_source_corpus, tiny_tokenizer, out, val_fraction=0.5,
                    source_word_counts=TWO_SOURCE_WORD_COUNTS)

    tok = AutoTokenizer.from_pretrained(str(tiny_tokenizer), local_files_only=True)
    alpha_lines, zebra_lines = _two_source_lines()
    alpha_ids = encode_batch(alpha_lines, tok)
    zebra_ids = encode_batch(zebra_lines, tok)
    expected_train = np.concatenate([
        alpha_ids[:len(alpha_ids) - int(len(alpha_ids) * 0.5)],
        zebra_ids[:len(zebra_ids) - int(len(zebra_ids) * 0.5)],
    ])
    assert np.array_equal(np.load(out / "train_ids.npy"), expected_train)


def test_stratified_split_reports_per_source_totals(two_source_corpus, tiny_tokenizer,
                                                      tmp_path):
    out = tmp_path / "tokens"
    stats = tokenize_corpus(two_source_corpus, tiny_tokenizer, out, val_fraction=0.5,
                            source_word_counts=TWO_SOURCE_WORD_COUNTS)
    for name, rec in stats.source_splits.items():
        assert rec["train_tokens"] + rec["val_tokens"] == rec["total_tokens"]
    assert (sum(r["train_tokens"] for r in stats.source_splits.values())
            == stats.train_tokens)
    assert (sum(r["val_tokens"] for r in stats.source_splits.values())
            == stats.val_tokens)


def test_non_stratified_split_leaves_source_splits_none(tiny_corpus, tiny_tokenizer,
                                                          tmp_path):
    """The pre-existing default path must not start reporting source_splits either."""
    stats = tokenize_corpus(tiny_corpus, tiny_tokenizer, tmp_path / "tokens")
    assert stats.source_splits is None


def test_stratified_split_preserves_each_sources_relative_share_of_train(tiny_tokenizer,
                                                                          tmp_path):
    """THE SECOND CONSEQUENCE OF THE BUG, not just validation representativeness.

    The default tail-of-the-whole-stream split does not just make validation
    unrepresentative -- on the real nine-source blend it also skewed the TRAINING
    mixture itself: wikipedia_simple (target 15% of the blend) supplied only 5.02% of
    actual train_ids tokens, because the entire held-out tail happened to be carved out
    of that one source while every other source kept its full allocation, inflating
    everyone else's relative share (tinystories rose from a 31% target to 35.07% of
    train). Holding out the SAME val_fraction from every source (this test's whole
    point) cannot do that: every source shrinks by the identical factor (1 -
    val_fraction), so relative shares among the survivors are mathematically unchanged.
    This test proves that with sources of deliberately very different sizes -- shrinking
    all of them by the same fraction must leave a 6:3:1 ratio a 6:3:1 ratio, not skew it.
    """
    lines_by_source = {
        "big": [f"big word {i} filler" for i in range(60)],     # 60 lines * 3 words = 180
        "mid": [f"mid word {i} filler" for i in range(30)],     # 30 lines * 3 words = 90
        "small": [f"small word {i} filler" for i in range(10)],  # 10 lines * 3 words = 30
    }
    corpus = tmp_path / "blend.txt"
    all_lines = lines_by_source["big"] + lines_by_source["mid"] + lines_by_source["small"]
    corpus.write_text("\n".join(all_lines) + "\n", encoding="utf-8")
    word_counts = {name: sum(len(line.split()) for line in lines)
                  for name, lines in lines_by_source.items()}

    out = tmp_path / "tokens"
    stats = tokenize_corpus(corpus, tiny_tokenizer, out, val_fraction=0.2,
                            source_word_counts=word_counts)

    total_train = sum(s["train_tokens"] for s in stats.source_splits.values())
    total_all = sum(s["total_tokens"] for s in stats.source_splits.values())
    for name in lines_by_source:
        share_in_train = stats.source_splits[name]["train_tokens"] / total_train
        share_of_whole_blend = stats.source_splits[name]["total_tokens"] / total_all
        # Proportional per-source holdout preserves share to well within a percent --
        # the only slack is np.int() truncation of val_tokens for each source.
        assert share_in_train == pytest.approx(share_of_whole_blend, abs=0.01), (
            f"{name}'s share of train ({share_in_train:.4f}) drifted from its share of "
            f"the whole blend ({share_of_whole_blend:.4f}) -- stratified splitting "
            f"should never change a source's relative weight in the training mixture"
        )


def test_stratified_mode_still_refuses_to_overwrite_existing_arrays(two_source_corpus,
                                                                     tiny_tokenizer,
                                                                     tmp_path):
    """The overwrite guard (TokenArtifactExistsError) is checked before the split
    strategy is even chosen, so passing source_word_counts must not be a way around it."""
    out = tmp_path / "tokens"
    tokenize_corpus(two_source_corpus, tiny_tokenizer, out, val_fraction=0.5,
                    source_word_counts=TWO_SOURCE_WORD_COUNTS)
    train_before = np.load(out / "train_ids.npy").copy()
    val_before = np.load(out / "val_ids.npy").copy()

    with pytest.raises(TokenArtifactExistsError):
        tokenize_corpus(two_source_corpus, tiny_tokenizer, out, val_fraction=0.5,
                        source_word_counts=TWO_SOURCE_WORD_COUNTS)

    assert np.array_equal(train_before, np.load(out / "train_ids.npy"))
    assert np.array_equal(val_before, np.load(out / "val_ids.npy"))


def test_stratified_split_rejects_word_count_overshoot(two_source_corpus, tiny_tokenizer,
                                                         tmp_path):
    """A declared word count that doesn't land on a line boundary is a caller error, not
    something to silently round through."""
    bad_counts = {"alpha": 81, "zebra": 79}  # off by one in each direction
    with pytest.raises(ValueError, match="overshot"):
        tokenize_corpus(two_source_corpus, tiny_tokenizer, tmp_path / "tokens",
                        val_fraction=0.5, source_word_counts=bad_counts)


def test_stratified_split_rejects_leftover_corpus_lines(two_source_corpus, tiny_tokenizer,
                                                          tmp_path):
    """source_word_counts that undershoot the real corpus must not silently drop the
    remainder on the floor."""
    short_counts = {"alpha": 80, "zebra": 40}
    with pytest.raises(ValueError, match="more lines than source_word_counts"):
        tokenize_corpus(two_source_corpus, tiny_tokenizer, tmp_path / "tokens",
                        val_fraction=0.5, source_word_counts=short_counts)


def test_stratified_split_rejects_corpus_shorter_than_declared(two_source_corpus,
                                                                 tiny_tokenizer, tmp_path):
    """source_word_counts that overshoot the real corpus must not silently under-fill."""
    long_counts = {"alpha": 80, "zebra": 120}
    with pytest.raises(ValueError, match="ended before source"):
        tokenize_corpus(two_source_corpus, tiny_tokenizer, tmp_path / "tokens",
                        val_fraction=0.5, source_word_counts=long_counts)


def test_encode_batch_matches_manual_tokenizer_call(tiny_tokenizer):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tiny_tokenizer), local_files_only=True)
    batch = ["the cat sat", "a dog ran"]
    expected = [i for seq in tok(batch, add_special_tokens=False)["input_ids"] for i in seq]
    assert encode_batch(batch, tok).tolist() == expected


def test_encode_batch_of_empty_list_is_empty_array(tiny_tokenizer):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tiny_tokenizer), local_files_only=True)
    result = encode_batch([], tok)
    assert len(result) == 0
    assert result.dtype == TOKEN_DTYPE


def test_source_word_counts_from_manifest_reads_emitted_words(tmp_path):
    manifest_path = tmp_path / "blend_manifest.json"
    manifest_path.write_text(json.dumps({
        "sources": {
            "alpha": {"emitted_words": 80, "emitted_tokens": 999},
            "zebra": {"emitted_words": 80, "emitted_tokens": 111},
        }
    }))
    assert source_word_counts_from_manifest(manifest_path) == {"alpha": 80, "zebra": 80}


def test_cli_blend_manifest_defaults_to_none():
    assert _parse_args([]).blend_manifest is None


def test_cli_blend_manifest_accepts_a_path():
    args = _parse_args(["--blend-manifest", "docs/measurements/blend_manifest.json"])
    assert args.blend_manifest == Path("docs/measurements/blend_manifest.json")


def test_stratified_helper_returns_arrays_and_splits_directly(two_source_corpus,
                                                                tiny_tokenizer):
    """_tokenize_stratified is exercised directly too, not only through tokenize_corpus,
    since it is where the actual per-source boundary logic lives."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tiny_tokenizer), local_files_only=True)
    train_ids, val_ids, splits = _tokenize_stratified(
        two_source_corpus, tok, TWO_SOURCE_WORD_COUNTS, val_fraction=0.25, chunk_lines=3)
    assert isinstance(train_ids, np.ndarray) and isinstance(val_ids, np.ndarray)
    assert set(splits) == {"alpha", "zebra"}
    assert len(train_ids) + len(val_ids) == sum(s["total_tokens"] for s in splits.values())
