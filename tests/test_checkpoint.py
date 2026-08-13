# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Checkpoint header schema. Pure dict work — no hardware, no ttml import."""

from pathlib import Path

import pytest

from train.checkpoint import (
    CHECKPOINT_FORMAT,
    build_header,
    checkpoint_path,
    latest_checkpoint,
    validate_header,
)


def _header(**kw):
    base = dict(
        step=100,
        model_config_path="/models/nanollama3.yaml",
        tokenizer_dir="artifacts/tokenizer",
        corpus_tokens=127_635_889,
        batch_size=64,
    )
    base.update(kw)
    return build_header(**base)


def test_header_carries_format_version():
    assert _header()["format"] == CHECKPOINT_FORMAT


def test_header_carries_resume_and_conversion_fields():
    h = _header()
    for field in ("step", "vocab_size", "seq_len", "model_config_path", "tokenizer_dir",
                  "corpus_tokens", "batch_size", "tokens_seen", "created_at"):
        assert field in h, f"header missing {field}"


def test_header_records_vocab_and_seq_len_from_config():
    from train.config import SEQ_LEN, VOCAB_SIZE

    h = _header()
    assert h["vocab_size"] == VOCAB_SIZE
    assert h["seq_len"] == SEQ_LEN


def test_header_computes_tokens_seen_from_step_batch_and_seq_len():
    """tokens_seen must be derived, not guessed -- batch_size isn't recorded anywhere
    else a model card could read it from, so this is the only source of truth."""
    from train.config import SEQ_LEN

    h = _header(step=3000, batch_size=64)
    assert h["tokens_seen"] == 3000 * 64 * SEQ_LEN
    assert h["tokens_seen"] == 49_152_000


def test_corpus_tokens_and_tokens_seen_are_independent_fields():
    """corpus_tokens (whole corpus) must not be confused with tokens_seen (this run's
    actual training volume) -- they differ by design for a partial-epoch run."""
    h = _header(step=3000, batch_size=64, corpus_tokens=127_635_889)
    assert h["corpus_tokens"] == 127_635_889
    assert h["tokens_seen"] == 49_152_000
    assert h["corpus_tokens"] != h["tokens_seen"]


def test_extra_carries_ttml_cpp_defaults_absent_from_any_yaml():
    """intermediate_dim, weight_tying, rms_norm_eps, and weights_dtype exist only as ttml
    C++ defaults/manifest facts -- they must survive build_header via extra and pass
    validate_header, or a converter has nowhere to read them from but a guess."""
    h = _header(extra={
        "transformer_config": {"embedding_dim": 384, "num_heads": 6, "num_groups": 3},
        "intermediate_dim": 1024,
        "weight_tying": True,
        "rms_norm_eps": 1e-5,
        "weights_dtype": "bfloat16",
    })
    validate_header(h)  # must not raise
    assert h["intermediate_dim"] == 1024
    assert h["weight_tying"] is True
    assert h["rms_norm_eps"] == 1e-5
    assert h["weights_dtype"] == "bfloat16"
    assert h["transformer_config"]["embedding_dim"] == 384


def test_extra_is_merged_without_clobbering_required_fields():
    h = _header(extra={"note": "smoke run"})
    assert h["note"] == "smoke run"
    assert h["step"] == 100  # extra must not overwrite schema fields


def test_extra_cannot_override_a_schema_field():
    with pytest.raises(ValueError, match="may not override"):
        _header(extra={"vocab_size": 999})


def test_validate_accepts_a_built_header():
    validate_header(_header())  # must not raise


def test_validate_rejects_missing_field():
    h = _header()
    del h["vocab_size"]
    with pytest.raises(ValueError, match="vocab_size"):
        validate_header(h)


def test_validate_rejects_future_format():
    h = _header()
    h["format"] = CHECKPOINT_FORMAT + 1
    with pytest.raises(ValueError, match="format"):
        validate_header(h)


def test_checkpoint_path_is_step_numbered():
    p = checkpoint_path(Path("/ckpt"), 2500)
    assert p == Path("/ckpt/tt_tnt_step00002500.pkl")


def test_checkpoint_paths_sort_lexicographically_by_step():
    """Zero-padding matters: without it, step10 sorts before step9."""
    paths = sorted(str(checkpoint_path(Path("/c"), s)) for s in (9, 10, 100))
    assert paths == [str(checkpoint_path(Path("/c"), s)) for s in (9, 10, 100)]


def test_latest_checkpoint_returns_none_for_empty_dir(tmp_path):
    assert latest_checkpoint(tmp_path) is None


def test_latest_checkpoint_picks_highest_step_not_newest_file(tmp_path):
    """The docstring used to say "newest"; with one directory shared across runs,
    highest-step and most-recently-written are not the same file. Create the
    higher-step file *first* and the lower-step one *second* -- so the lower-step
    file has the newer mtime -- to prove the selection is by step, not by mtime."""
    high_step_path = checkpoint_path(tmp_path, 5000)
    low_step_path = checkpoint_path(tmp_path, 100)
    high_step_path.touch()
    low_step_path.touch()  # written after, so it has the newer mtime
    assert low_step_path.stat().st_mtime_ns >= high_step_path.stat().st_mtime_ns
    assert latest_checkpoint(tmp_path) == high_step_path


def test_latest_checkpoint_finds_pre_rename_nanollama3_files(tmp_path):
    """Checkpoints written before the tt-nanollama3 -> tt-tnt rename are never renamed on
    disk (they are evidence of a real run under the old name) -- a directory holding only
    those files, e.g. the real ``artifacts/checkpoints/``, must still resolve via
    ``--resume latest``."""
    legacy = tmp_path / "nanollama3_step00003000.pkl"
    legacy.touch()
    assert latest_checkpoint(tmp_path) == legacy


def test_latest_checkpoint_picks_the_higher_step_across_both_naming_schemes(tmp_path):
    """A directory that mixes pre-rename and post-rename checkpoints (e.g. an old baseline
    directory a new run resumed into) must pick the highest **step**, regardless of which
    naming scheme it happens to be written under."""
    old_low = tmp_path / "nanollama3_step00000500.pkl"
    old_low.touch()
    new_high = checkpoint_path(tmp_path, 3000)
    new_high.touch()
    assert latest_checkpoint(tmp_path) == new_high

    # And the reverse: an old-prefixed file can still be the higher step.
    tmp_path2 = tmp_path / "mixed2"
    tmp_path2.mkdir()
    old_high = tmp_path2 / "nanollama3_step00021034.pkl"
    old_high.touch()
    new_low = checkpoint_path(tmp_path2, 100)
    new_low.touch()
    assert latest_checkpoint(tmp_path2) == old_high
