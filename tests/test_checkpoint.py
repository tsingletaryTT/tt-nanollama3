# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Checkpoint header schema. Pure dict work — no hardware, no ttml import."""

from pathlib import Path

import pytest

from train.checkpoint import (
    CHECKPOINT_FORMAT,
    build_header,
    checkpoint_path,
    validate_header,
)


def _header(**kw):
    base = dict(
        step=100,
        model_config_path="/models/nanollama3.yaml",
        tokenizer_dir="artifacts/tokenizer",
        total_tokens=127_635_889,
    )
    base.update(kw)
    return build_header(**base)


def test_header_carries_format_version():
    assert _header()["format"] == CHECKPOINT_FORMAT


def test_header_carries_resume_and_conversion_fields():
    h = _header()
    for field in ("step", "vocab_size", "seq_len", "model_config_path",
                  "tokenizer_dir", "total_tokens", "created_at"):
        assert field in h, f"header missing {field}"


def test_header_records_vocab_and_seq_len_from_config():
    from train.config import SEQ_LEN, VOCAB_SIZE

    h = _header()
    assert h["vocab_size"] == VOCAB_SIZE
    assert h["seq_len"] == SEQ_LEN


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
    assert p == Path("/ckpt/nanollama3_step00002500.pkl")


def test_checkpoint_paths_sort_lexicographically_by_step():
    """Zero-padding matters: without it, step10 sorts before step9."""
    paths = sorted(str(checkpoint_path(Path("/c"), s)) for s in (9, 10, 100))
    assert paths == [str(checkpoint_path(Path("/c"), s)) for s in (9, 10, 100)]
