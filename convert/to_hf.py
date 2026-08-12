# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Assemble a Hugging Face model directory from a NanoLlama3 checkpoint.

Everything about the architecture comes from the **checkpoint header**. Plan 3 enriched
that header precisely so this step never guesses: ``intermediate_dim``, ``weight_tying``,
and ``rms_norm_eps`` exist only as ttml C++ defaults and are recoverable from nothing else.
A missing field raises rather than defaulting — a quiet default is how a converted model
silently mismatches the weights it ships with.

No ttnn, no ttml: the checkpoint is plain pickle + numpy.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict

import numpy as np

from convert.checkpoint_reader import read_checkpoint_meta, read_tensors
from convert.hf_mapping import map_name, split_kv, squeeze_leading

_REQUIRED = ("vocab_size", "seq_len", "intermediate_dim", "weight_tying",
             "rms_norm_eps", "weights_dtype", "transformer_config")


def build_config(header: Dict[str, Any]) -> Dict[str, Any]:
    """HF ``LlamaConfig`` fields, entirely from ``header``."""
    missing = [f for f in _REQUIRED if f not in header]
    if missing:
        raise ValueError(
            f"checkpoint header missing field(s) required for conversion: "
            f"{', '.join(missing)}. Re-run scripts/backfill_checkpoint_headers.py."
        )
    tc = header["transformer_config"]
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": int(tc["embedding_dim"]),
        "num_hidden_layers": int(tc["num_blocks"]),
        "num_attention_heads": int(tc["num_heads"]),
        "num_key_value_heads": int(tc["num_groups"]),
        "intermediate_size": int(header["intermediate_dim"]),
        "vocab_size": int(header["vocab_size"]),
        "max_position_embeddings": int(header["seq_len"]),
        "rope_theta": float(tc["theta"]),
        "rms_norm_eps": float(header["rms_norm_eps"]),
        "tie_word_embeddings": bool(header["weight_tying"]),
        "torch_dtype": str(header["weights_dtype"]),
        "hidden_act": "silu",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 3,
    }


def convert_checkpoint(ckpt: Path, tokenizer_dir: Path, out_dir: Path) -> Dict[str, Any]:
    """Write a loadable HF model directory. Returns the config that was written."""
    from safetensors.numpy import save_file

    ckpt, tokenizer_dir, out_dir = Path(ckpt), Path(tokenizer_dir), Path(out_dir)
    header, _manifest = read_checkpoint_meta(ckpt)
    config = build_config(header)
    tc = header["transformer_config"]
    head_dim = config["hidden_size"] // config["num_attention_heads"]

    out: Dict[str, np.ndarray] = {}
    for name, tensor in read_tensors(ckpt):
        target = map_name(name)
        if target is None:
            continue
        if name.endswith("attention/kv_linear/weight"):
            k, v = split_kv(tensor, num_groups=int(tc["num_groups"]), head_dim=head_dim)
            out[target[0]], out[target[1]] = k, v
        elif isinstance(target, tuple):
            # Tied embedding: one tensor, both destinations.
            arr = squeeze_leading(tensor)
            out[target[0]] = arr
            out[target[1]] = arr
        else:
            out[target] = squeeze_leading(tensor)

    down = out.get("model.layers.0.mlp.down_proj.weight")
    gate = out.get("model.layers.0.mlp.gate_proj.weight")
    if down is not None and gate is not None and down.shape != gate.shape[::-1]:
        raise ValueError(
            f"MLP role assignment looks wrong: down_proj {down.shape} is not the "
            f"transpose-shape of gate_proj {gate.shape}. Check MLP_ROLES in hf_mapping."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_dir / "model.safetensors"))
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        src = tokenizer_dir / f
        if src.is_file():
            shutil.copy2(src, out_dir / f)
    return config
