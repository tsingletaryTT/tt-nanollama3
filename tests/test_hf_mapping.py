# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""ttml -> HF name and layout mapping. Pure numpy; no hardware, no ttml."""

import numpy as np
import pytest

from convert.hf_mapping import MLP_ROLES, map_name, permute_rope_qk, split_kv, squeeze_leading


def test_tied_embedding_maps_to_both_targets():
    assert map_name("llama/fc/weight") == ("model.embed_tokens.weight", "lm_head.weight")


def test_final_norm_maps():
    assert map_name("llama/ln_fc/gamma") == "model.norm.weight"


@pytest.mark.parametrize("ttml,hf", [
    ("llama/llama_block_0/attention/q_linear/weight", "model.layers.0.self_attn.q_proj.weight"),
    ("llama/llama_block_5/attention/out_linear/weight", "model.layers.5.self_attn.o_proj.weight"),
    ("llama/llama_block_3/attention_norm/gamma", "model.layers.3.input_layernorm.weight"),
    ("llama/llama_block_2/mlp_norm/gamma", "model.layers.2.post_attention_layernorm.weight"),
    ("llama/llama_block_4/mlp/w1/weight", "model.layers.4.mlp.gate_proj.weight"),
    ("llama/llama_block_4/mlp/w2/weight", "model.layers.4.mlp.down_proj.weight"),
    ("llama/llama_block_4/mlp/w3/weight", "model.layers.4.mlp.up_proj.weight"),
])
def test_block_tensors_map(ttml, hf):
    assert map_name(ttml) == hf


def test_kv_linear_maps_to_a_pair():
    got = map_name("llama/llama_block_1/attention/kv_linear/weight")
    assert got == ("model.layers.1.self_attn.k_proj.weight",
                   "model.layers.1.self_attn.v_proj.weight")


def test_unknown_name_returns_none():
    assert map_name("llama/mystery/weight") is None


def test_squeeze_drops_leading_unit_dims():
    t = np.zeros((1, 1, 384, 384), dtype=np.float32)
    assert squeeze_leading(t).shape == (384, 384)


def test_squeeze_leaves_two_d_alone():
    t = np.zeros((384, 384), dtype=np.float32)
    assert squeeze_leading(t).shape == (384, 384)


def test_squeeze_drops_all_leading_units_down_to_a_1d_gamma():
    """ttml stores a norm gamma as (1, 1, 1, hidden) -- a real 1-D vector, not a 2-D
    row. squeeze_leading must not stop early just because ndim happens to hit 2."""
    t = np.zeros((1, 1, 1, 384), dtype=np.float32)
    assert squeeze_leading(t).shape == (384,)


def test_squeeze_drops_a_single_leading_unit_from_two_d():
    t = np.zeros((1, 384), dtype=np.float32)
    assert squeeze_leading(t).shape == (384,)


def test_squeeze_leaves_a_true_1d_array_alone():
    t = np.zeros((384,), dtype=np.float32)
    assert squeeze_leading(t).shape == (384,)


def test_squeeze_does_not_over_squeeze_a_legitimate_2d_weight():
    """A weight matrix with no leading unit dims at all must come out unchanged --
    squeezing must only ever remove dims it can prove are ttml padding, not real ones."""
    t = np.zeros((192, 384), dtype=np.float32)
    assert squeeze_leading(t).shape == (192, 384)


def test_split_kv_halves_the_output_dim():
    # 3 groups x 64 head_dim x 2 (K and V) = 384 rows, hidden 384
    t = np.arange(384 * 384, dtype=np.float32).reshape(384, 384)
    k, v = split_kv(t, num_groups=3, head_dim=64)
    assert k.shape == (192, 384)
    assert v.shape == (192, 384)


def test_split_kv_is_a_partition_not_a_copy():
    """Every row must appear in exactly one of K or V, in order."""
    t = np.arange(384 * 4, dtype=np.float32).reshape(384, 4)
    k, v = split_kv(t, num_groups=3, head_dim=64)
    assert np.array_equal(np.concatenate([k, v], axis=0), t)


def test_split_kv_rejects_wrong_row_count():
    t = np.zeros((100, 384), dtype=np.float32)
    with pytest.raises(ValueError, match="expected 384 rows"):
        split_kv(t, num_groups=3, head_dim=64)


def test_mlp_roles_follow_swiglu_convention():
    assert MLP_ROLES == {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}


def test_permute_rope_qk_preserves_shape():
    # 2 heads x head_dim 4 = 8 output rows, hidden 3
    t = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)
    out = permute_rope_qk(t, num_heads=2, head_dim=4)
    assert out.shape == (8, 3)


def test_permute_rope_qk_is_a_permutation_not_a_copy_or_a_loss():
    """Every row of the input must appear exactly once in the output -- this is a
    reordering of rows, not a projection that could quietly drop or duplicate data."""
    t = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)
    out = permute_rope_qk(t, num_heads=2, head_dim=4)
    in_rows = {tuple(row) for row in t}
    out_rows = {tuple(row) for row in out}
    assert in_rows == out_rows
    assert len(out) == len(t)


def test_permute_rope_qk_swaps_the_inner_two_row_blocks_per_head():
    """Interleaved -> split-halves: within each head's head_dim rows, the permutation
    groups rows by their position mod 2 (even rows, then odd rows) -- e.g. for head_dim=4,
    rows [0,1,2,3] of a head become [0,2,1,3]. Pinned against a hand-computed example so a
    future refactor can't silently invert or scramble the permutation while keeping the
    'it's some permutation of the same rows' test above green."""
    t = np.arange(8 * 1, dtype=np.float32).reshape(8, 1)
    out = permute_rope_qk(t, num_heads=2, head_dim=4)
    # Head 0 (rows 0-3): interleaved order [0,1,2,3] -> split-halves order [0,2,1,3].
    # Head 1 (rows 4-7): interleaved order [4,5,6,7] -> split-halves order [4,6,5,7].
    expected = np.array([[0], [2], [1], [3], [4], [6], [5], [7]], dtype=np.float32)
    assert np.array_equal(out, expected)


def test_permute_rope_qk_handles_grouped_query_attention_head_counts():
    """k_proj has fewer heads than q_proj under GQA (ttml's num_groups); num_heads must be
    the tensor's own head count, not hardcoded, or a GQA model gets the wrong block size."""
    # 3 groups x head_dim 64 x hidden 384, matching this project's real k_proj shape.
    t = np.arange(192 * 384, dtype=np.float32).reshape(192, 384)
    out = permute_rope_qk(t, num_heads=3, head_dim=64)
    assert out.shape == (192, 384)
