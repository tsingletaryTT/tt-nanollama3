# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Training config assembly. Pure dict/attribute work — no hardware, no ttml import."""

import pytest

from train.config import SEQ_LEN, VOCAB_SIZE, RunConfig, build_yaml_config, run_config_from_yaml


def _yaml(**kw):
    return build_yaml_config("artifacts/tokenizer", "/models/nanollama3.yaml", **kw)


def test_declares_bpe_and_tokenizer_path():
    cfg = _yaml()
    tc = cfg["training_config"]
    assert tc["tokenizer_type"] == "bpe"
    assert tc["tokenizer_path"] == "artifacts/tokenizer"


def test_carries_model_config_path():
    assert _yaml()["training_config"]["model_config"] == "/models/nanollama3.yaml"


def test_overrides_are_applied():
    tc = _yaml(batch_size=8, max_steps=1234)["training_config"]
    assert tc["batch_size"] == 8
    assert tc["max_steps"] == 1234


def test_run_config_carries_seq_len():
    """The whole point: ttml's TrainingConfig never sets seq_len, and train() needs it."""
    rc = run_config_from_yaml(_yaml())
    assert rc.seq_len == SEQ_LEN == 256


def test_run_config_has_every_field_train_reads():
    rc = run_config_from_yaml(_yaml(batch_size=4, max_steps=20))
    for field in ("seq_len", "steps", "batch_size", "gradient_accumulation_steps", "eval_every"):
        assert hasattr(rc, field), f"train() reads cfg.{field}; RunConfig lacks it"
    assert rc.steps == 20
    assert rc.batch_size == 4


def test_vocab_size_matches_the_tokenizer_contract():
    assert VOCAB_SIZE == 32000


def test_rejects_seq_len_beyond_model_capacity():
    with pytest.raises(ValueError, match="max_sequence_length"):
        _yaml(seq_len=512)


def test_emits_optimizer_section():
    """ttml.common.utils.create_optimizer raises ValueError without this section."""
    opt = _yaml()["training_config"]["optimizer"]
    assert opt["type"] == "AdamW"
    assert opt["lr"] == 0.0003
    assert opt["weight_decay"] == 0.01
