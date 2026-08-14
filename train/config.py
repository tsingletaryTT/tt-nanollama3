# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Assemble the tt-train config for a tt-tnt run.

Two things this module exists to get right:

1. **``seq_len``.** ``ttml.common.trainer.train()`` reads ``cfg.seq_len`` for
   ``build_causal_mask()`` and ``get_batch_ttml()``, but ``ttml.common.config.TrainingConfig``
   never defines it — the value lives on ``TransformerConfig`` as ``max_sequence_length``.
   Handing ``train()`` a bare ``TrainingConfig`` raises ``AttributeError`` before it trains.
   ``RunConfig`` copies it across explicitly.
2. **The tokenizer path.** ttml resolves ``tokenizer_path`` relative to
   ``$TT_METAL_HOME/tt-train`` (``ttml/common/data.py:91``), which is *not* where our
   tokenizer lives. We bypass ttml's data loading entirely (see ``train/run.py``), so this
   path is recorded for provenance rather than consumed by ttml.

No ttnn/ttml imports here — this is dict and attribute work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml

#: FALLBACK training sequence length, not the value real runs use.
#:
#: ``train/run.py`` derives its ``--seq-len`` from the selected ``ModelSize``'s own
#: ``max_sequence_length``, because ``build_yaml_config`` below rejects anything else. This
#: constant is only what ``RunConfig`` falls back to when a config dict omits ``seq_len``
#: and what ``build_yaml_config``'s own keyword defaults are, which is a test convenience.
#:
#: It was 256 (tt-train's original), then 512 for the first nine-source run. The vendored
#: configs no longer agree with each other -- ``tt-tnt-384.yaml`` declares 2048 as of
#: 2026-08-14 and ``tt-tnt-1024.yaml`` still declares 512 -- so there is no single number
#: this constant could hold that would be right for both, which is precisely why it must not
#: be the source of a real run's window. See ``.superpowers/seqlen-ddp-investigation.md``
#: for why lengthening the window is safe (no C++ ceiling, flash-attention SDPA memory is
#: linear in sequence length, the tokenized data path needs no changes) and
#: ``train/sizes.py`` for why 384 moved to 2048 and 1024 did not.
SEQ_LEN = 512

#: Must equal the tokenizer's vocabulary (Plan 1 pins it at exactly this).
VOCAB_SIZE = 32000


class RunConfig:
    """Every attribute ``ttml.common.trainer.train()`` reads, plus what our loop needs.

    Deliberately a plain object rather than a subclass of ttml's ``TrainingConfig``: the
    fields ``train()`` requires are not the fields that class provides, and inheriting
    would hide exactly the mismatch this exists to fix.
    """

    def __init__(self, tc: Dict[str, Any]):
        self.seq_len = int(tc.get("seq_len", SEQ_LEN))
        self.steps = int(tc.get("max_steps", 20))
        self.batch_size = int(tc.get("batch_size", 64))
        self.validation_batch_size = int(
            tc.get("validation_batch_size", max(self.batch_size // 2, 1))
        )
        self.gradient_accumulation_steps = int(tc.get("gradient_accumulation_steps", 1))
        self.eval_every = int(tc.get("eval_every", 200))
        self.save_every = int(tc.get("model_save_interval", 0))
        self.checkpoint_dir = tc.get("checkpoint_dir", "artifacts/checkpoints")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"RunConfig(seq_len={self.seq_len}, steps={self.steps}, "
                f"batch_size={self.batch_size}, eval_every={self.eval_every})")


def build_yaml_config(
    tokenizer_dir: str,
    model_config_path: str,
    *,
    seq_len: int = SEQ_LEN,
    max_sequence_length: int = SEQ_LEN,
    batch_size: int = 64,
    max_steps: int = 20,
    eval_every: int = 200,
    gradient_accumulation_steps: int = 1,
    checkpoint_dir: str = "artifacts/checkpoints",
    stochastic_rounding: bool = False,
) -> Dict[str, Any]:
    """Build the config dict ``TransformerModelFactory`` and ``RunConfig`` consume.

    ``stochastic_rounding`` defaults to ``False`` (ttml's own default,
    ``optimizers/optimizer_registry.cpp:37``) to keep every existing call site's behaviour
    unchanged. **This is the flag whose absence produced 13 permanently-frozen RMSNorm
    gammas** in the original 3000-step run: bfloat16 parameters at 1.0 have a step size (ulp)
    of 0.0039, an order of magnitude larger than the ~3e-4 Adam updates those gammas
    received, so every update rounded deterministically back to 1.0 and was discarded. With
    stochastic rounding on, a rounding direction is chosen probabilistically weighted by how
    close the true update lands to each representable bfloat16 value, so a stream of
    sub-ulp updates accumulates real drift instead of vanishing every single time. See
    ``tests/test_training_config.py`` and
    ``docs/superpowers/specs/2026-08-11-followups.md`` item 1. Callers that want the fix
    pass ``stochastic_rounding=True`` directly, or point ``train/run.py --config`` at
    ``train/configs/nanollama3_bpe_v2.yaml`` (see ``apply_optimizer_override`` below).

    ``seq_len`` vs. ``max_sequence_length``: these are two independent numbers upstream
    (``cfg.seq_len`` is the window drawn per training batch; ``max_sequence_length`` is
    the model's declared context, which sizes the RoPE cos/sin tables built once at model
    construction) and **must be identical**. ``rotary_embedding_llama``'s prefill-mode
    validator checks the head dimension but never checks the sequence dimension against
    the input's (see ``.superpowers/seqlen-ddp-investigation.md``, §1.3), so a mismatch
    would not raise on-device -- it would silently hand a shorter batch window a
    differently-shaped rotary cache and produce wrong rotary embeddings with no error at
    all. That failure mode is worse than a crash, so it is rejected here, at config-build
    time, before a device is ever opened.
    """
    if seq_len % 32 != 0:
        raise ValueError(
            f"seq_len must be a multiple of 32 (the Tenstorrent tile dimension); got "
            f"seq_len={seq_len}."
        )
    if seq_len != max_sequence_length:
        raise ValueError(
            f"seq_len ({seq_len}) must equal the model's max_sequence_length "
            f"({max_sequence_length}) -- they are independent numbers upstream (the batch "
            f"window vs. the RoPE cache size) that ttml never cross-checks. A mismatch "
            f"would not raise inside ttml; it would silently produce wrong rotary "
            f"embeddings (rotary_embedding_llama's prefill validator does not check the "
            f"sequence dimension). Pass a --seq-len equal to the selected --size's "
            f"max_sequence_length, or pick a --size whose max_sequence_length is "
            f"{seq_len}."
        )
    return {
        "training_config": {
            "seed": 5489,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "max_steps": max_steps,
            "eval_every": eval_every,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            # Checkpointing is deferred to Stage 3 (needs ttml's checkpoint format read
            # first), so this is always 0 for now — `checkpoint_dir`/`save_every` below
            # are threaded through RunConfig but currently unused.
            "model_save_interval": 0,
            "checkpoint_dir": checkpoint_dir,
            "tokenizer_type": "bpe",
            "tokenizer_path": tokenizer_dir,
            "model_config": model_config_path,
            # REQUIRED: ttml.common.utils.create_optimizer raises
            # ValueError("training_config must contain an 'optimizer' section") without
            # this, and passes the dict straight to the C++ optimizer factory. Values
            # match tt-train's own training_shakespeare_nanollama3.yaml.
            "optimizer": {
                "type": "AdamW",
                "lr": 0.0003,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1.0e-8,
                "weight_decay": 0.01,
                "amsgrad": False,
                "stochastic_rounding": stochastic_rounding,
            },
        },
        "device_config": {"mesh_shape": [1, 1], "enable_ddp": False, "enable_tp": False},
    }


def run_config_from_yaml(yaml_config: Dict[str, Any]) -> RunConfig:
    """Extract the run config from an assembled YAML dict."""
    return RunConfig(yaml_config.get("training_config", {}))


def apply_optimizer_override(
    yaml_config: Dict[str, Any], override_path: Union[str, "Path"]
) -> Dict[str, Any]:
    """Replace ``yaml_config["training_config"]["optimizer"]`` with the block from a file.

    ``train/run.py`` assembles its config entirely from CLI flags via ``build_yaml_config``
    — there is no on-disk "the" training config to load. This function is the bridge to a
    real, on-disk *recipe* file such as ``train/configs/nanollama3_bpe_v2.yaml`` (a copy of
    the nanollama3 BPE recipe with ``stochastic_rounding: true`` added), for a caller that
    wants to opt into a config-file-defined optimizer without a dedicated CLI flag for every
    future tweak.

    Deliberately narrow: only the ``optimizer`` sub-block is taken from ``override_path``.
    Per-invocation operational knobs (``--steps``, ``--batch-size``, ``--checkpoint-dir``,
    ...) keep coming from the CLI, exactly as before — this does not turn ``train/run.py``
    into a general YAML-config loader, it only answers "which optimizer recipe" question.

    Mutates and returns ``yaml_config`` for convenient chaining; raises ``ValueError`` if
    the override file doesn't have the expected ``training_config.optimizer`` shape, so a
    typo'd path produces a clear failure instead of a silent no-op.
    """
    override_path = Path(override_path)
    with override_path.open("r", encoding="utf-8") as f:
        override = yaml.safe_load(f)
    try:
        optimizer = override["training_config"]["optimizer"]
    except (KeyError, TypeError) as e:
        raise ValueError(
            f"{override_path} does not have a training_config.optimizer block to apply"
        ) from e
    yaml_config.setdefault("training_config", {})["optimizer"] = optimizer
    return yaml_config
