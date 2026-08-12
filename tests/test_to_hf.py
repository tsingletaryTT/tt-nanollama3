# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""HF config assembly. Pure dict work plus one guarded end-to-end test."""

from pathlib import Path

import pytest

from convert.to_hf import build_config

CKPT = Path("artifacts/checkpoints/nanollama3_step00003000.pkl")


def _header(**kw):
    h = {
        "format": 1, "step": 3000, "vocab_size": 32000, "seq_len": 256,
        "intermediate_dim": 1024, "weight_tying": True, "rms_norm_eps": 1e-05,
        "weights_dtype": "bfloat16", "batch_size": 64, "tokens_seen": 49152000,
        "transformer_config": {
            "embedding_dim": 384, "num_blocks": 6, "num_heads": 6,
            "num_groups": 3, "theta": 500000.0,
        },
    }
    h.update(kw)
    return h


def test_config_is_llama():
    c = build_config(_header())
    assert c["model_type"] == "llama"
    assert c["architectures"] == ["LlamaForCausalLM"]


def test_dimensions_come_from_the_header():
    c = build_config(_header())
    assert c["hidden_size"] == 384
    assert c["num_hidden_layers"] == 6
    assert c["num_attention_heads"] == 6
    assert c["num_key_value_heads"] == 3
    assert c["intermediate_size"] == 1024
    assert c["vocab_size"] == 32000
    assert c["max_position_embeddings"] == 256
    assert c["rope_theta"] == 500000.0
    assert c["rms_norm_eps"] == 1e-05


def test_tie_word_embeddings_reflects_the_header():
    assert build_config(_header())["tie_word_embeddings"] is True
    assert build_config(_header(weight_tying=False))["tie_word_embeddings"] is False


def test_dtype_reflects_the_header():
    assert build_config(_header())["torch_dtype"] == "bfloat16"


def test_missing_header_field_raises_rather_than_defaulting():
    """A converter that quietly defaults is how a model silently mismatches."""
    h = _header()
    del h["intermediate_dim"]
    with pytest.raises(ValueError, match="intermediate_dim"):
        build_config(h)


@pytest.mark.skipif(not CKPT.is_file(), reason="no trained checkpoint on this machine")
def test_end_to_end_against_the_real_checkpoint(tmp_path):
    from convert.to_hf import convert_checkpoint

    out = tmp_path / "hf"
    cfg = convert_checkpoint(CKPT, Path("artifacts/tokenizer"), out)
    assert (out / "config.json").is_file()
    assert (out / "model.safetensors").is_file()
    assert (out / "tokenizer.json").is_file()
    assert cfg["vocab_size"] == 32000

    from safetensors.numpy import load_file

    tensors = load_file(str(out / "model.safetensors"))
    # 6 layers x 9 tensors + embed + lm_head + final norm
    assert "model.embed_tokens.weight" in tensors
    assert "lm_head.weight" in tensors
    assert tensors["model.embed_tokens.weight"].shape == (32000, 384)
    assert tensors["model.layers.0.self_attn.k_proj.weight"].shape == (192, 384)
    assert tensors["model.layers.0.mlp.down_proj.weight"].shape == (384, 1024)

    # Norm gammas must land as genuine 1-D vectors, not (1, hidden) -- HF's
    # LlamaRMSNorm.weight is nn.Parameter(torch.ones(hidden_size)), and a size mismatch
    # here is exactly the defect that made a previously-emitted model directory fail to
    # load via transformers.AutoModelForCausalLM.from_pretrained().
    assert tensors["model.norm.weight"].shape == (384,)
    assert tensors["model.layers.0.input_layernorm.weight"].shape == (384,)
    assert tensors["model.layers.0.post_attention_layernorm.weight"].shape == (384,)
