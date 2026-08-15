# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Training config assembly. Pure dict/attribute work — no hardware, no ttml import."""

from pathlib import Path

import pytest
import yaml

from train.config import (
    DEFAULT_SEED,
    SEQ_LEN,
    VOCAB_SIZE,
    RunConfig,
    apply_optimizer_override,
    build_yaml_config,
    run_config_from_yaml,
)


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
    assert rc.seq_len == SEQ_LEN == 512


def test_run_config_has_every_field_train_reads():
    rc = run_config_from_yaml(_yaml(batch_size=4, max_steps=20))
    for field in ("seq_len", "steps", "batch_size", "gradient_accumulation_steps", "eval_every"):
        assert hasattr(rc, field), f"train() reads cfg.{field}; RunConfig lacks it"
    assert rc.steps == 20
    assert rc.batch_size == 4


def test_vocab_size_matches_the_tokenizer_contract():
    assert VOCAB_SIZE == 32000


def test_rejects_seq_len_not_equal_to_max_sequence_length():
    """The critical invariant: cfg.seq_len must equal the model's max_sequence_length.

    ttml never cross-checks the two itself (rotary_embedding_llama's prefill validator
    checks head_dim but not the sequence dimension), so a mismatch would silently produce
    wrong rotary embeddings on-device rather than raise -- this repo rejects it up front
    instead, before a device is ever opened. Both directions of mismatch must raise: a
    seq_len smaller than the model's max_sequence_length is just as wrong as a larger one,
    since it is the RoPE cache's shape that stops matching the batch window's, not a
    capacity ceiling.
    """
    with pytest.raises(ValueError, match="max_sequence_length"):
        _yaml(seq_len=256, max_sequence_length=512)
    with pytest.raises(ValueError, match="max_sequence_length"):
        _yaml(seq_len=1024, max_sequence_length=512)


def test_accepts_seq_len_matching_max_sequence_length():
    tc = _yaml(seq_len=512, max_sequence_length=512)["training_config"]
    assert tc["seq_len"] == 512


def test_rejects_seq_len_not_a_multiple_of_32():
    """32 is the Tenstorrent tile dimension; ttml's Llama constructor hard-rejects
    max_sequence_length values that aren't multiples of it (models/llama.cpp:124-128).
    Reject early with the actual value named, rather than letting ttml raise deep inside
    model construction after the device is already open."""
    with pytest.raises(ValueError, match="500"):
        _yaml(seq_len=500, max_sequence_length=500)


def test_emits_optimizer_section():
    """ttml.common.utils.create_optimizer raises ValueError without this section."""
    opt = _yaml()["training_config"]["optimizer"]
    assert opt["type"] == "AdamW"
    assert opt["lr"] == 0.0003
    assert opt["weight_decay"] == 0.01


def test_seed_defaults_to_the_value_every_committed_run_used():
    """v1-v4 all ran at 5489, hardcoded. Changing this default would not fail anything
    loudly -- it would silently make every number in docs/measurements/ irreproducible,
    so the default is pinned here rather than left to whatever the constant happens to be."""
    assert DEFAULT_SEED == 5489
    assert _yaml()["training_config"]["seed"] == 5489


def test_seed_can_be_overridden():
    """The whole point of the flag: the same recipe at a different seed, which is how
    run-to-run variance (the noise floor for every between-run comparison) gets measured."""
    assert _yaml(seed=20260815)["training_config"]["seed"] == 20260815


def test_seed_override_moves_nothing_else():
    """A seed-replicate run must differ from its twin in exactly one number."""
    baseline = _yaml(batch_size=16, max_steps=10764)["training_config"]
    varied = _yaml(batch_size=16, max_steps=10764, seed=20260815)["training_config"]
    differing = {k for k in baseline if baseline[k] != varied.get(k)}
    assert differing == {"seed"}, f"seed override also changed {differing - {'seed'}}"


def test_config_file_seed_is_documentation_and_never_overrides_the_real_one():
    """The trap. ``train/configs/*.yaml`` carries a ``seed:`` key, but
    ``apply_optimizer_override`` copies the ``optimizer`` block and nothing else, so that
    key has never had any effect. Anyone who "changes the seed" by editing the yaml is
    changing a comment. Pinned so the illusion cannot quietly become real either."""
    cfg = _yaml(seed=20260815)
    override_path = Path(__file__).resolve().parent.parent / "train" / "configs"
    override_path = override_path / "nanollama3_bpe_v2.yaml"
    with override_path.open("r", encoding="utf-8") as f:
        on_disk = yaml.safe_load(f)
    # The file really does carry a seed, and it differs from what we passed --
    # otherwise this test would pass for the wrong reason.
    assert on_disk["training_config"]["seed"] == 5489
    apply_optimizer_override(cfg, override_path)
    assert cfg["training_config"]["seed"] == 20260815


def test_stochastic_rounding_defaults_off():
    """The v1 behaviour (and the bug): every existing call site is unaffected unless it
    opts in explicitly. See tests/test_training_config.py for what leaving this off cost."""
    assert _yaml()["training_config"]["optimizer"]["stochastic_rounding"] is False


def test_stochastic_rounding_can_be_enabled():
    opt = _yaml(stochastic_rounding=True)["training_config"]["optimizer"]
    assert opt["stochastic_rounding"] is True
    # Nothing else in the optimizer block should move.
    assert opt["type"] == "AdamW"
    assert opt["lr"] == 0.0003


def test_apply_optimizer_override_replaces_the_optimizer_block(tmp_path):
    override_path = tmp_path / "override.yaml"
    override_path.write_text(
        yaml.dump({"training_config": {"optimizer": {"type": "AdamW", "lr": 0.001,
                                                        "stochastic_rounding": True}}})
    )
    cfg = _yaml()
    original_lr = cfg["training_config"]["optimizer"]["lr"]

    result = apply_optimizer_override(cfg, override_path)

    assert result is cfg  # mutates and returns the same dict for chaining
    opt = cfg["training_config"]["optimizer"]
    assert opt["stochastic_rounding"] is True
    assert opt["lr"] == 0.001
    assert opt["lr"] != original_lr


def test_apply_optimizer_override_leaves_non_optimizer_fields_untouched(tmp_path):
    """Only the "which optimizer recipe" question is answered by an override file —
    steps/batch_size/checkpoint_dir keep coming from the CLI, per the function's docstring."""
    override_path = tmp_path / "override.yaml"
    override_path.write_text(
        yaml.dump({"training_config": {"optimizer": {"stochastic_rounding": True}}})
    )
    cfg = _yaml(batch_size=8, max_steps=1234)

    apply_optimizer_override(cfg, override_path)

    tc = cfg["training_config"]
    assert tc["batch_size"] == 8
    assert tc["max_steps"] == 1234


def test_apply_optimizer_override_rejects_a_file_without_an_optimizer_block(tmp_path):
    override_path = tmp_path / "bad.yaml"
    override_path.write_text(yaml.dump({"training_config": {"seed": 1}}))
    with pytest.raises(ValueError, match="optimizer"):
        apply_optimizer_override(_yaml(), override_path)


def test_v2_config_file_has_stochastic_rounding_enabled():
    """The actual shipped recipe file: train/run.py --config points here to get the fix."""
    v2_path = (
        Path(__file__).resolve().parent.parent / "train" / "configs" / "nanollama3_bpe_v2.yaml"
    )
    with v2_path.open("r", encoding="utf-8") as f:
        v2 = yaml.safe_load(f)
    assert v2["training_config"]["optimizer"]["stochastic_rounding"] is True


def test_v2_config_file_is_loadable_via_apply_optimizer_override():
    v2_path = (
        Path(__file__).resolve().parent.parent / "train" / "configs" / "nanollama3_bpe_v2.yaml"
    )
    cfg = _yaml()
    apply_optimizer_override(cfg, v2_path)
    assert cfg["training_config"]["optimizer"]["stochastic_rounding"] is True
