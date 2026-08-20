#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Feasibility probe: does a sparse MoE construct and forward inside tt-tnt's Llama?

The whole Mixture of Enthusiasts plan rests on one claim -- that ttml's
``SparseMoEEP`` is separable from the DeepSeek model that ships it, because
``LlamaBlock.mlp`` is a slot with a matching ``forward(Tensor) -> Tensor``
signature. This is the cheapest possible test of that claim, and it runs before
any training is scheduled.

Needs a gozer lease.

    gozer run --chips 1 --who claude:moe-subclass --reason "feasibility" -- \\
      python scripts/probe_moe_subclass.py
"""
import sys, os
sys.path.insert(0, "/home/ttuser/code/tt-tnt")
for k, v in (("TT_METAL_HOME","/home/ttuser/tt-metal"),
             ("TT_METAL_RUNTIME_ROOT","/home/ttuser/tt-metal"),
             ("TT_METAL_ARCH_NAME","blackhole"), ("TT_LOGGER_LEVEL","FATAL")):
    os.environ.setdefault(k, v)
sys.path.append("/home/ttuser/tt-metal/tt-train/sources/ttml")

from train.enthusiasts import MoEHyperparams, install_enthusiasts
from train.config import build_yaml_config
from train.sizes import get_size
import train.model as tt_tnt_model
from ttml.common.utils import initialize_device, set_seed

size = get_size("1024")
cfg = build_yaml_config(
    tokenizer_dir="artifacts/tokenizer",
    model_config_path=str(size.model_config_path) if hasattr(size,"model_config_path") else "x.yaml",
    seq_len=512, max_sequence_length=512, batch_size=2, max_steps=2)
initialize_device(cfg)
set_seed(cfg["training_config"]["seed"])

tc = {"model_type":"llama","num_heads":16,"num_groups":4,"embedding_dim":1024,
      "dropout_prob":0.0,"num_blocks":8,"vocab_size":32000,
      "max_sequence_length":512,"runner_type":"default","theta":500000.0}
model = tt_tnt_model.create_model(cfg, tc)
print(f"dense model built: {len(list(model.blocks))} blocks")

hp = MoEHyperparams(dim=1024, moe_inter_dim=928, n_routed_experts=10,
                    n_activated_experts=2, n_shared_experts=1)
summary = install_enthusiasts(model, hp, gate_policy="learned", first_moe_block=2)
print(f"\nswapped: blocks {summary['moe_blocks']}")
print(f"mlp types now: {[type(b.mlp).__name__ for b in model.blocks]}")

# The real gate: does it forward and backward?
import numpy as np, ttml
B, T = 2, 512
ids = np.random.default_rng(0).integers(0, 32000, size=(B, 1, 1, T)).astype(np.uint32)
tgt = np.random.default_rng(1).integers(0, 32000, size=(B, 1, 1, T)).astype(np.uint32)
try:
    x = ttml.autograd.Tensor.from_numpy(ids)
    y = ttml.autograd.Tensor.from_numpy(tgt)
    print("\nattempting forward...")
    out = model(x, None) if True else None
    print(f"  forward OK: {type(out).__name__}")
except Exception as e:
    print(f"  forward raised: {type(e).__name__}: {str(e)[:220]}")
