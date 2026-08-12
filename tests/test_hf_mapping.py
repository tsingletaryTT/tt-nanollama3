# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""ttml -> HF name and layout mapping. Pure numpy; no hardware, no ttml."""

import numpy as np
import pytest

from convert.hf_mapping import MLP_ROLES, map_name, split_kv, squeeze_leading


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
