# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Pure-NumPy reimplementation of ttml's Llama forward pass.

**Why this exists.** `convert/hf_mapping.py` and `convert/to_hf.py` encode our own
*interpretation* of ttml's conventions (RoPE pairing, K/V split order, GQA broadcast, ...).
This module is an independently-derived instrument to check that interpretation: it is
written from `docs/ttml-forward-reference.md` alone, which was itself derived from ttml's
C++ source (`ttml/...`) and the ttnn kernels it calls, never from our converter. If this
module agreed with the converter because it copied the converter's assumptions, it would
prove nothing -- see the reference doc's "Independence" section and the plan-4 postmortem
it cites (a wrong RoPE layout that survived four checks because nothing here was written
down and cross-checked against source independently).

Every convention implemented below is cited to the reference doc section that derives it,
which in turn cites the ttml file:line it was read from. Where the doc says ttml does
something -- even something that looks unnatural -- this module does that, not a "more
sensible" alternative; deviations belong in the report as findings, not silent corrections.

**Purity.** No `ttnn`/`ttml` import, ever -- checked by
`tests/test_ttml_forward.py::test_ttml_forward_module_imports_no_tenstorrent`, a subprocess
import-and-inspect-`sys.modules` probe in the same style as
`test_checkpoint_reader.py::test_convert_checkpoint_reader_imports_no_tenstorrent`,
`test_tokenizer.py::test_convert_module_imports_no_tenstorrent`, and
`test_backfill_checkpoint_headers.py::test_backfill_script_imports_no_tenstorrent`: this
module must run on a machine with no tt-metal checkout and no Tenstorrent hardware. Only
`numpy` (plus whatever dtype the checkpoint's bfloat16 tensors arrive as, handled by a plain
`.astype(np.float32)`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np

from convert.checkpoint_reader import read_checkpoint_meta, read_tensors

#: RMSNorm epsilon. docs/ttml-forward-reference.md §0 and §2: the checkpoint header's
#: `rms_norm_eps` field is *descriptive, not authoritative* -- `LlamaBlock` constructs
#: `RMSNormLayer(embedding_size)` with the C++ default argument
#: (ttml/modules/llama_block.cpp:47-48, ttml/modules/rms_norm_module.hpp:17,23) and nothing
#: plumbs a YAML/header value through. So this is hardcoded to the C++ default rather than
#: read from the checkpoint header, even though the two happen to agree for this checkpoint.
#:
#: **This hardcoding is also load-bearing for parity-gate coverage, not just fidelity to
#: ttml.** `RMS_NORM_EPS` being a plain module constant (rather than something threaded
#: through from the checkpoint header on every call) is what makes
#: `tests/test_numpy_parity.py::test_parity_gate_is_not_hollow_it_catches_epsilon_moved_outside_the_sqrt`
#: able to construct its probe at all -- that test proves the parity gate catches epsilon
#: moved outside the sqrt, the one perturbation docs/ttml-forward-reference.md's §9 Q6 found
#: invisible to the cross-entropy check. Someone "cleaning this up" to read `rms_norm_eps`
#: from the header instead would be a reasonable-looking refactor that silently removes this
#: coverage. Read that test before making this change, not just this comment.

RMS_NORM_EPS = 1e-5


def squeeze_leading(x: np.ndarray) -> np.ndarray:
    """Drop leading singleton dimensions, stopping at the first real one (or at 1-D).

    ttml checkpoint tensors are rank-4 with two (linears) or three (gammas) leading size-1
    dims: `[1, 1, out, in]` for a `LinearLayer` weight
    (ttml/modules/linear_module.cpp:19-25) and `[1, 1, 1, C]` for an `RMSNormLayer` gamma
    (docs/ttml-forward-reference.md §0's tensor-shape table). Squeezing "the first two dims"
    unconditionally would leave gammas at rank 2 (`[1, C]`) instead of the rank-1 vector the
    reference formulas in §2/§8 expect, so this squeezes *every* leading dim that happens to
    be 1, not a fixed count of two.

    **Deliberately duplicated, not imported.** `convert/hf_mapping.py` has its own version
    of this squeeze. Importing it here would wire this independent NumPy path back into the
    converter under test, defeating the point of a from-scratch reference (see the module
    docstring). The logic is a three-line loop; the duplication costs little and buys real
    independence.
    """
    while x.ndim > 1 and x.shape[0] == 1:
        x = x[0]
    return x


def rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = RMS_NORM_EPS) -> np.ndarray:
    """ttml's RMSNorm. docs/ttml-forward-reference.md §2, citing the fused kernel
    (ttml/metal/ops/rmsnorm_fw/device/kernels/compute/rmsnorm_fw_kernel.cpp:288-372) and the
    composite reference (ttml/ops/rmsnorm_op.cpp:69-125), which agree:

        rms(x) = sqrt(mean_c(x_c^2) + eps)     # eps INSIDE the sqrt, added to mean(x^2)
        y_c    = gamma_c * x_c / rms(x)        # plain gamma, not (1 + gamma)

    `mean` is over the last axis; `gamma` broadcasts over every leading (batch/sequence)
    axis. Both of the "traps" this formula avoids -- eps added after the sqrt, or gamma
    read as `1 + gamma` -- are called out explicitly in §2 because either one produces a
    plausible-looking model that is quietly wrong (§9's Q6 table: `1+gamma` costs 15.9 SE
    end to end; misplaced eps is the one perturbation that is invisible at the CE level, so
    getting this formula right cannot rely on the Step 3 check catching a mistake here).
    """
    mean_sq = np.mean(np.square(x), axis=-1, keepdims=True)
    rms = np.sqrt(mean_sq + eps)
    return gamma * x / rms


def apply_rope(
    x: np.ndarray, positions: np.ndarray, theta: float, *, head_dim: int
) -> np.ndarray:
    """ttml's RoPE: interleaved pairing `(x[2i], x[2i+1])`, not split-halves.

    docs/ttml-forward-reference.md §3, verdict at §3.5 ("very high confidence", four
    mutually-consistent lines of evidence -- the frequency table, the 2x2 block-diagonal
    rotation matrix, the per-tile matmul that makes split-halves structurally impossible at
    Dh=64, and ttml's own HF importer un-permuting split-halves into this layout on import).

    Frequency table (§3.1, `ttml/ops/rope_op.cpp:198-205`, `gen_freqs`):

        inv_freq[i] = theta ** (-2 * floor(i/2) / head_dim)     for i in [0, head_dim)

    so adjacent channels 2j, 2j+1 share a frequency. Rotation (§3.2-3.3, combining
    `gen_trans_mat` at `ttml/ops/rope_op.cpp:237-248` with the ttnn kernel's
    `out = x*cos + (x @ trans_mat)*sin` at
    `.../rotary_embedding_llama/device/kernels/compute/rotary_embedding_llama.cpp:90-138`):

        out[2j]   = x[2j]  * cos_j - x[2j+1] * sin_j
        out[2j+1] = x[2j+1] * cos_j + x[2j]  * sin_j
        cos_j = cos(p * inv_freq[2j]),  sin_j = sin(p * inv_freq[2j]),  p = position

    Position offset is always `0 .. S-1` here (§3.7: the no-KV-cache path passes token
    position 0, and this reference never runs decode with a cache), but `positions` is
    taken as an explicit argument rather than hardcoded to `arange` -- both so a caller can
    apply the position-offset convention if it ever needs to, and so tests can assert the
    rotation-by-p-then-by-minus-p round trip.

    `x`'s last axis must be `head_dim`; any leading axes (heads, batch, ...) broadcast
    through the `[len(positions), head_dim]` cos/sin tables unchanged. `head_dim` is a
    required keyword rather than inferred from `x.shape[-1]` alone so a caller applying
    RoPE to a *fused* multi-head tensor (width = num_heads * head_dim) gets a validation
    error instead of a silently-wrong pairing period; see the ValueError below.

    RoPE scaling (§3.8, `apply_rope_scaling` at `ttml/ops/rope_op.cpp:67-108`) is not
    implemented: it is only invoked when `scaling_factor != 1.0`
    (`ttml/ops/rope_op.cpp:208`), and this checkpoint's config has scaling off (§0).
    """
    if x.shape[-1] != head_dim:
        raise ValueError(
            f"apply_rope: x's last axis ({x.shape[-1]}) must equal head_dim ({head_dim}); "
            "reshape to per-head tensors before calling, ttml never applies RoPE across a "
            "fused multi-head width."
        )
    if head_dim % 2 != 0:
        raise ValueError(f"apply_rope: head_dim must be even for interleaved pairing, got {head_dim}")
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape[0] != x.shape[-2]:
        raise ValueError(
            f"apply_rope: len(positions) ({positions.shape[0]}) must match x's sequence "
            f"axis ({x.shape[-2]})"
        )

    channel = np.arange(head_dim)
    inv_freq = np.power(float(theta), -2.0 * (channel // 2) / head_dim)  # §3.1
    ang = positions[:, None] * inv_freq[None, :]  # [S, head_dim]
    cos = np.cos(ang).astype(x.dtype)
    sin = np.sin(ang).astype(x.dtype)

    rot = np.empty_like(x)
    rot[..., 0::2] = -x[..., 1::2]  # x @ trans_mat, §3.2
    rot[..., 1::2] = x[..., 0::2]
    return x * cos + rot * sin


def swiglu(x: np.ndarray, w1: np.ndarray, w2: np.ndarray, w3: np.ndarray) -> np.ndarray:
    """ttml's SwiGLU MLP. docs/ttml-forward-reference.md §6, citing the call site
    (`ttml/modules/llama_block.cpp:36`) and both the composite
    (`ttml/ops/swiglu_op.cpp:41-54`) and fused (`:56-125`) implementations, which agree:

        mlp(x) = ( silu(x @ w1.T) * (x @ w3.T) ) @ w2.T
        silu(z) = z / (1 + exp(-z))

    **The trap this exists to avoid:** the call site passes arguments in the order
    `w1, w2, w3` while the module *registers* them as `w1, w3, w2`
    (`ttml/modules/llama_block.cpp:29-31` vs `:36`) -- position is not meaning here. SiLU is
    on the `w1` branch; `w3` is the plain (unactivated) gate. In HF `LlamaMLP` terms
    `w1 = gate_proj`, `w3 = up_proj`, `w2 = down_proj`, corroborated by ttml's own HF
    importer's mapping (`ttml/models/llama.cpp:602-638`) -- cited for cross-check only, not
    as the source this was derived from.

    All three weights are `[out, in]` (§0's "row-major [out,in]" convention), so every
    projection here is `x @ W.T`, matching `y = x @ W.T` for `LinearLayer` generally.
    """
    gate = x @ w1.T
    silu = gate / (1.0 + np.exp(-gate))
    up = x @ w3.T
    return (silu * up) @ w2.T


def attention(
    x: np.ndarray,
    q_w: np.ndarray,
    kv_w: np.ndarray,
    out_w: np.ndarray,
    *,
    num_heads: int,
    num_groups: int,
    theta: float,
) -> np.ndarray:
    """ttml's grouped-query attention sub-layer (one block's worth).

    docs/ttml-forward-reference.md §1 (`ttml/modules/grouped_query_attention.cpp:36-57`),
    §4 (head splitting / K-before-V), §3 (RoPE on Q/K only, never V), and §5 (SDPA scale,
    additive causal mask, GQA head->group mapping). `x` is `[S, embedding_dim]`
    (batch already dropped, per the reference doc's convention); returns the sub-layer
    output `[S, embedding_dim]` *before* the residual add, which is the caller's job
    (§1's block structure adds this to the pre-norm residual, not this function).

    Pipeline, in order:

    1. `q = x @ q_w.T`, `kv = x @ kv_w.T` -- plain linears, no bias anywhere in this model
       (§0: `GQAConfig.bias_linears = false`).
    2. Head split (§4, `grouped_heads_creation` -> `nlp_create_qkv_heads`): Q's heads are
       contiguous `head_dim`-wide slices of `q`, head-major
       (`ttml/ops/multi_head_utils.cpp:88-130`, writer kernel comment confirming
       `[B, num_q_heads, s, head_dim]` shuffled from `[B, 1, s, num_q_heads*head_dim]`).
       The fused `kv` tensor's **first** `num_groups*head_dim` columns are K, the **last**
       `num_groups*head_dim` are V -- K precedes V (§4.2, the reader kernel's
       Q-tiles-then-K-tiles-then-V-tiles order, corroborated by the backward pass's
       `concat({grad_k, grad_v})` and by ttml's own HF importer's K-then-V row concat).
    3. RoPE (§3) on Q and K only -- V is never rotated (§1's explicit note, confirmed by the
       HF importer permuting q_proj/k_proj rows on import but leaving v_proj alone). Applied
       per-head, using absolute positions `0 .. S-1` (§3.7: no KV cache in this forward).
    4. GQA broadcast (§5.3): query head `h` attends to KV group `h // (num_heads //
       num_groups)` -- contiguous blocks of query heads share a group, i.e.
       `np.repeat(..., axis=0)` (`repeat_interleave` semantics), matching HF's `repeat_kv`.
       **Not** `np.tile` (round-robin) -- getting this backwards still produces a
       correctly-shaped tensor and a model that still runs, which is exactly what makes it
       dangerous (§8's comment on this line).
    5. Scaled dot-product attention (§5): `scale = 1/sqrt(head_dim)` (the head dimension,
       not the model dimension -- `ttml/metal/ops/sdpa_fw/device/sdpa_fw_program_factory.cpp
       :271,293-294` reads `query.padded_shape()[3]`), with an **additive** causal mask
       (`0` keep / `-inf` masked, added to the raw scores before the softmax -- §5.2's two
       mask-application code paths, `Arbitrary` (the one this checkpoint's training driver
       actually took, `train/run.py:150` -> `trainer.py:73-76,102`) and the built-in
       `Causal` path, agree numerically).
    6. `heads_fusion` (§4.3): the inverse of step 2's Q split -- concatenate heads back into
       a contiguous `[S, num_heads*head_dim]` row, head-major.
    7. `out = fused @ out_w.T`.
    """
    seq_len = x.shape[0]
    # q_w's out-features, i.e. H * Dh -- numerically equal to the model's embedding_dim on
    # this architecture (num_heads * head_dim == hidden_size), but that is a fact about this
    # specific config, not a general one: a model where head_dim != hidden/num_heads would
    # have q_linear's out-features differ from the embedding dim, and naming this
    # `embedding_dim` would silently misdescribe what it actually is.
    q_out_features = q_w.shape[0]
    head_dim = q_out_features // num_heads
    heads_per_group = num_heads // num_groups

    q = x @ q_w.T  # [S, H*Dh]
    kv = x @ kv_w.T  # [S, 2*G*Dh]

    # §4.1: head-major contiguous Dh-wide slices, i.e. reshape(S, H, Dh) then move heads
    # to the front.
    q_heads = q.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)  # [H, S, Dh]
    kv_group_width = num_groups * head_dim
    k_heads = (
        kv[:, :kv_group_width].reshape(seq_len, num_groups, head_dim).transpose(1, 0, 2)
    )  # [G, S, Dh] -- §4.2: K occupies the first G*Dh columns
    v_heads = (
        kv[:, kv_group_width:].reshape(seq_len, num_groups, head_dim).transpose(1, 0, 2)
    )  # [G, S, Dh] -- V occupies the last G*Dh columns, after K

    positions = np.arange(seq_len, dtype=np.float64)  # §3.7: no KV cache -> 0 .. S-1
    q_heads = apply_rope(q_heads, positions, theta, head_dim=head_dim)  # §3: Q ...
    k_heads = apply_rope(k_heads, positions, theta, head_dim=head_dim)  # ... and K only

    # §5.3: repeat_interleave, not tile -- query head h attends to group h // heads_per_group.
    k_heads = np.repeat(k_heads, heads_per_group, axis=0)  # [H, S, Dh]
    v_heads = np.repeat(v_heads, heads_per_group, axis=0)  # [H, S, Dh]

    scale = 1.0 / np.sqrt(head_dim)  # §5.1: head dim, not model dim
    causal = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)  # §5.2: additive, 0/-inf
    scores = q_heads @ k_heads.transpose(0, 2, 1) * scale + causal  # [H, S, S]
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    attn = weights @ v_heads  # [H, S, Dh]

    fused = attn.transpose(1, 0, 2).reshape(seq_len, num_heads * head_dim)  # §4.3
    return fused @ out_w.T


def forward(checkpoint_path: Union[str, Path], token_ids: np.ndarray) -> np.ndarray:
    """Full ttml Llama forward pass over a single sequence, from raw checkpoint tensors.

    docs/ttml-forward-reference.md §1 (block structure, pre-norm/two-residual) and §7
    (tied embedding/output projection, unscaled at both ends). Returns raw logits
    `[len(token_ids), vocab_size]` -- no softmax, no temperature (§7.2: neither end of the
    embedding/output path applies any scaling).

    Weights are read via `read_tensors` (declaration order, one tensor at a time off the
    pickle stream -- see `convert/checkpoint_reader.py`) rather than any name-sorted
    listing, matching the same "don't assume alphabetical order" discipline
    `checkpoint_reader` itself is tested against. Every tensor is squeezed of its leading
    singleton batch dims (`squeeze_leading`, above) and cast from the checkpoint's
    bfloat16 storage dtype to float32 for arithmetic.
    """
    checkpoint_path = Path(checkpoint_path)
    header, _manifest = read_checkpoint_meta(checkpoint_path)
    cfg = header["transformer_config"]
    num_heads = int(cfg["num_heads"])
    num_groups = int(cfg["num_groups"])
    num_blocks = int(cfg["num_blocks"])
    theta = float(cfg["theta"])

    weights: Dict[str, np.ndarray] = {}
    for name, array in read_tensors(checkpoint_path, group="model"):
        weights[name] = squeeze_leading(array).astype(np.float32)

    tokens = np.asarray(token_ids, dtype=np.int64)

    # §7.1: one tied [vocab, embedding_dim] tensor serves both the embedding table (row
    # gather here) and the output projection (transpose, at the very end). §7.2: a plain
    # gather, no sqrt(d_model) scaling, no positional term added here (positions live
    # inside attention's RoPE, not in the embedding).
    tok_emb = weights["llama/fc/weight"]  # [vocab, embedding_dim]
    x = tok_emb[tokens]  # [S, embedding_dim]

    for block in range(num_blocks):
        p = f"llama/llama_block_{block}"

        # §1: pre-norm attention sub-layer, residual added after.
        h = rms_norm(x, weights[f"{p}/attention_norm/gamma"])
        a = attention(
            h,
            weights[f"{p}/attention/q_linear/weight"],
            weights[f"{p}/attention/kv_linear/weight"],
            weights[f"{p}/attention/out_linear/weight"],
            num_heads=num_heads,
            num_groups=num_groups,
            theta=theta,
        )
        x = x + a

        # §1: pre-norm MLP sub-layer, residual added after.
        h = rms_norm(x, weights[f"{p}/mlp_norm/gamma"])
        m = swiglu(
            h,
            weights[f"{p}/mlp/w1/weight"],
            weights[f"{p}/mlp/w2/weight"],
            weights[f"{p}/mlp/w3/weight"],
        )
        x = x + m

    x = rms_norm(x, weights["llama/ln_fc/gamma"])
    logits = x @ tok_emb.T  # §7.2: bare linear, no bias, no logit scaling
    return logits
