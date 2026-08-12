# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Map ttml tensor names and layouts onto Hugging Face Llama conventions.

Four things here are not guessable from a tensor alone, and each is a way to produce a
model that loads cleanly and is silently wrong:

1. **Weight tying.** The checkpoint has ``llama/fc/weight`` and *no* embedding tensor, so
   that one array must be written to both ``model.embed_tokens.weight`` and
   ``lm_head.weight``. A converter expecting an embedding emits random weights, no error.
2. **Fused K+V.** ``kv_linear`` packs both projections into one tensor; the split point
   comes from ``num_groups × head_dim``, which lives in the header, not the array.
3. **Leading unit dims.** ttml stores 2-D weights as ``(1, 1, out, in)``.
4. **RoPE row layout.** ``q_linear``/``k_proj`` rows are ordered for ttml's *interleaved*
   RoPE pairing; HF Llama's ``rotate_half`` expects *split-halves* pairing. Copying rows
   straight through (as an earlier version of this converter did) is invisible to every
   shape/name check -- the tensor is the right shape, in the right place, with the right
   name -- and produces a model that loads, ties weights, and generates plausible fluent
   text while being numerically wrong. See ``permute_rope_qk`` below.

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
    """Drop ttml's leading unit dimensions.

    ttml represents every tensor at a fixed rank of 4, padding with leading size-1 axes:
    a weight matrix is stored as ``(1, 1, out, in)`` and a norm gamma -- a genuine 1-D
    vector -- is stored as ``(1, 1, 1, hidden)``. The two cases need different amounts of
    squeezing to reach their real shape, and nothing here is told in advance which one a
    given tensor is, so the only safe rule is to keep dropping leading unit axes for as
    long as one remains: ``(out, in)`` stops on its own once ``out`` (never ``1`` for any
    real weight in this model) becomes the leading dim, while ``(hidden,)`` keeps going
    all the way down to a true 1-D array. Stopping early at ``ndim == 2`` (as an earlier
    version of this function did) left gammas one squeeze short, at ``(1, hidden)``
    instead of ``(hidden,)`` -- a shape `transformers` rejects outright when loading
    ``LlamaRMSNorm``, rather than silently accepting a wrong tensor.
    """
    arr = np.asarray(tensor)
    while arr.ndim > 1 and arr.shape[0] == 1:
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


def permute_rope_qk(tensor: np.ndarray, *, num_heads: int, head_dim: int) -> np.ndarray:
    """Reorder q/k projection rows from ttml's RoPE convention to HF Llama's.

    RoPE rotates a head's activations in 2-D planes, pairing up rows of the projection
    weight two at a time. Two incompatible conventions exist for *which* two rows form a
    pair, and nothing about a weight matrix's shape, name, or values reveals which one its
    author assumed:

    - **Interleaved** (ttml's convention here, and original Meta Llama's): row ``2i`` pairs
      with row ``2i + 1`` -- adjacent rows.
    - **Split-halves** (HF Llama's ``rotate_half``/``apply_rotary_pos_emb``): row ``i``
      pairs with row ``i + head_dim // 2`` -- a row and its counterpart in the second half
      of the head.

    Copying rows straight through mismatches these pairings, which degrades attention
    quality without producing an error or obviously-garbage output: the resulting model
    still loads, ties its embedding correctly, and generates locally-fluent text, so this
    is invisible to every check *except* an actual loss comparison against a known-good
    target. This was measured directly (numerical verification, hf-conversion plan Task 3):
    on the real checkpoint, HF-side validation loss with rows copied straight through was
    3.20 nats against a training-time target of 1.8781 nats; applying exactly this
    permutation to ``q_proj``/``k_proj`` and nothing else brought it to 1.927 nats. See
    CLAUDE.md's "numerical verification finds a real bug" section for the full writeup.

    This is the same row permutation Meta's own ``convert_llama_weights_to_hf.py`` applies
    when converting original-format (interleaved) Llama checkpoints to HF's format --
    not something invented for this project.

    Applies to ``q_proj`` and ``k_proj`` only, never ``v_proj``: RoPE rotates queries and
    keys before the attention dot product, but values pass through unrotated.

    ``num_heads`` is the tensor's own head count -- ``num_attention_heads`` for q_proj,
    ``num_key_value_heads`` (ttml's ``num_groups``) for k_proj -- never hardcoded, since
    grouped-query attention gives k_proj fewer heads than q_proj over the same head_dim.
    """
    out_features, in_features = tensor.shape
    return (
        tensor.reshape(num_heads, head_dim // 2, 2, in_features)
        .transpose(0, 2, 1, 3)
        .reshape(out_features, in_features)
    )
