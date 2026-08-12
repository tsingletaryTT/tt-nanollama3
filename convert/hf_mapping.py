# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Map ttml tensor names and layouts onto Hugging Face Llama conventions.

Three things here are not guessable from a tensor alone, and each is a way to produce a
model that loads cleanly and is silently wrong:

1. **Weight tying.** The checkpoint has ``llama/fc/weight`` and *no* embedding tensor, so
   that one array must be written to both ``model.embed_tokens.weight`` and
   ``lm_head.weight``. A converter expecting an embedding emits random weights, no error.
2. **Fused K+V.** ``kv_linear`` packs both projections into one tensor; the split point
   comes from ``num_groups × head_dim``, which lives in the header, not the array.
3. **Leading unit dims.** ttml stores 2-D weights as ``(1, 1, out, in)``.

Pure numpy. No ttnn, no ttml.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple, Union

import numpy as np

#: SwiGLU convention: w1 gates, w3 lifts, w2 projects back down. Verified by shape in
#: ``convert.to_hf`` — down_proj is the one whose dims are transposed relative to the others.
MLP_ROLES = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}

_BLOCK = re.compile(r"^llama/llama_block_(\d+)/(.+)$")


def map_name(ttml_name: str) -> Optional[Union[str, Tuple[str, str]]]:
    """HF parameter name for a ttml name.

    Returns a single name, a **pair** (tied embedding, or the fused K/V split), or ``None``
    when the tensor has no HF counterpart.
    """
    if ttml_name == "llama/fc/weight":
        # Tied: one tensor, two destinations.
        return ("model.embed_tokens.weight", "lm_head.weight")
    if ttml_name == "llama/ln_fc/gamma":
        return "model.norm.weight"

    m = _BLOCK.match(ttml_name)
    if not m:
        return None
    idx, rest = m.group(1), m.group(2)
    prefix = f"model.layers.{idx}"

    if rest == "attention_norm/gamma":
        return f"{prefix}.input_layernorm.weight"
    if rest == "mlp_norm/gamma":
        return f"{prefix}.post_attention_layernorm.weight"
    if rest == "attention/q_linear/weight":
        return f"{prefix}.self_attn.q_proj.weight"
    if rest == "attention/out_linear/weight":
        return f"{prefix}.self_attn.o_proj.weight"
    if rest == "attention/kv_linear/weight":
        return (f"{prefix}.self_attn.k_proj.weight", f"{prefix}.self_attn.v_proj.weight")

    mlp = re.match(r"^mlp/(w[123])/weight$", rest)
    if mlp:
        return f"{prefix}.mlp.{MLP_ROLES[mlp.group(1)]}.weight"
    return None


def squeeze_leading(tensor: np.ndarray) -> np.ndarray:
    """Drop ttml's leading unit dimensions: ``(1, 1, out, in)`` -> ``(out, in)``."""
    arr = np.asarray(tensor)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def split_kv(tensor: np.ndarray, *, num_groups: int, head_dim: int):
    """Split the fused K+V projection into ``(k, v)``.

    ``kv_linear`` stacks K then V along the output dimension, so the row count must be
    ``num_groups * head_dim * 2``. We check that rather than trusting it — a silent
    mis-split produces a model that runs and generates nonsense.
    """
    arr = squeeze_leading(tensor)
    expected = num_groups * head_dim * 2
    if arr.shape[0] != expected:
        raise ValueError(
            f"kv_linear: expected {expected} rows "
            f"(num_groups={num_groups} x head_dim={head_dim} x 2), got {arr.shape[0]}"
        )
    half = expected // 2
    return arr[:half], arr[half:]
