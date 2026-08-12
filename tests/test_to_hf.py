# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""HF config assembly. Pure dict work plus one guarded end-to-end test."""

import json
import pickle
from pathlib import Path

import numpy as np
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


def test_missing_nested_transformer_config_field_raises_the_same_way():
    """embedding_dim/num_blocks/etc. live inside transformer_config, not the top-level
    header -- a missing one there must raise the same informative ValueError as a missing
    top-level field, not a bare KeyError."""
    h = _header()
    del h["transformer_config"]["embedding_dim"]
    with pytest.raises(ValueError, match="embedding_dim"):
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
    # 6 layers x 9 tensors + embed + final norm. The real checkpoint has weight_tying=True,
    # so lm_head.weight is deliberately omitted (see the tied-embedding branch in
    # convert_checkpoint) -- transformers reconstructs it from the tied embedding at load.
    assert "model.embed_tokens.weight" in tensors
    assert "lm_head.weight" not in tensors
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


# --- Synthetic checkpoints -----------------------------------------------------------
#
# The untied-embedding path (Fix 3), the completeness post-condition (Fix 4), and the
# unmapped-tensor guard (Fix 3) all need checkpoints this repo doesn't actually have on
# disk -- every real checkpoint produced so far has weight_tying=True and a complete
# manifest. Rather than attempt to produce a real untied checkpoint (which would need an
# actual ttml training run with weight_tying disabled), these tests write minimal but
# structurally faithful fake checkpoints -- same on-disk shape ttml.checkpointing produces
# (one pickle record of {format, header, manifest}, then one pickle per tensor in
# named_parameters declaration order), same leading-unit-dim convention squeeze_leading
# expects, and real shape relationships (kv_linear's row count, q/k head-splittable row
# counts, down_proj's transpose-shape relative to gate_proj).
#
# A tiny 1-block, 2-head/1-group, embedding_dim=8 model keeps every tensor small while
# still exercising every branch in convert_checkpoint's tensor-assembly loop.

_SYNTH_TC = {"embedding_dim": 8, "num_blocks": 1, "num_heads": 2, "num_groups": 1, "theta": 10000.0}


def _synth_header(**kw):
    h = {
        "format": 1, "step": 1, "vocab_size": 10, "seq_len": 8,
        "intermediate_dim": 16, "weight_tying": False, "rms_norm_eps": 1e-05,
        "weights_dtype": "float32", "batch_size": 1, "tokens_seen": 8,
        "transformer_config": dict(_SYNTH_TC),
    }
    h.update(kw)
    return h


def _w(out_dim, in_dim, offset=0):
    """A ttml-shaped weight matrix: (1, 1, out, in). ``offset`` lets two same-shaped
    tensors (e.g. the untied tok_emb/fc pair) hold genuinely different values, so a test
    comparing them for equality is actually exercising something."""
    return (np.arange(out_dim * in_dim, dtype=np.float32) + offset).reshape(1, 1, out_dim, in_dim)


def _g(dim):
    """A ttml-shaped norm gamma: (1, 1, 1, dim)."""
    return np.ones((1, 1, 1, dim), dtype=np.float32)


def _synth_tensors(*, weight_tying):
    """One block's worth of tensors matching ``_synth_header``'s shapes.

    embedding_dim=8, num_heads=2, num_groups=1 -> head_dim=4, q_linear rows = 2*4=8,
    kv_linear rows = 1*4*2=8 (K then V, 4 rows each), intermediate_dim=16.
    """
    tensors = {}
    if weight_tying:
        tensors["llama/fc/weight"] = _w(10, 8)
    else:
        # Distinct offsets: an untied embedding table and output projection are two real,
        # independent tensors, and a test asserting they land in different HF slots needs
        # them to actually hold different values to mean anything.
        tensors["llama/tok_emb/weight"] = _w(10, 8, offset=0)
        tensors["llama/fc/weight"] = _w(10, 8, offset=10_000)
    tensors["llama/ln_fc/gamma"] = _g(8)
    tensors["llama/llama_block_0/attention_norm/gamma"] = _g(8)
    tensors["llama/llama_block_0/mlp_norm/gamma"] = _g(8)
    tensors["llama/llama_block_0/attention/q_linear/weight"] = _w(8, 8)
    tensors["llama/llama_block_0/attention/kv_linear/weight"] = _w(8, 8)
    tensors["llama/llama_block_0/attention/out_linear/weight"] = _w(8, 8)
    tensors["llama/llama_block_0/mlp/w1/weight"] = _w(16, 8)
    tensors["llama/llama_block_0/mlp/w2/weight"] = _w(8, 16)
    tensors["llama/llama_block_0/mlp/w3/weight"] = _w(16, 8)
    return tensors


def _write_fake_checkpoint(path: Path, header: dict, tensors: dict) -> Path:
    """Write a checkpoint matching ttml.checkpointing's on-disk shape (see
    convert/checkpoint_reader.py's module docstring), without importing ttml."""
    manifest = {"model": {"named_parameters": {name: {} for name in tensors}}}
    record = {"format": 1, "header": header, "manifest": manifest}
    with open(path, "wb") as f:
        pickle.dump(record, f)
        for arr in tensors.values():
            pickle.dump(arr, f)
    return path


def _write_fake_tokenizer_dir(tokenizer_dir: Path, *, bos=1, eos=2, pad=3, unk=0) -> Path:
    """Minimal special_tokens_map.json + tokenizer_config.json, matching the shape read by
    convert.to_hf._tokenizer_special_ids -- enough to drive Fix 5's cross-check, not a real
    tokenizer export."""
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    (tokenizer_dir / "special_tokens_map.json").write_text(json.dumps({
        "bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>", "unk_token": "<unk>",
    }), encoding="utf-8")
    (tokenizer_dir / "tokenizer_config.json").write_text(json.dumps({
        "added_tokens_decoder": {
            str(unk): {"content": "<unk>"},
            str(bos): {"content": "<s>"},
            str(eos): {"content": "</s>"},
            str(pad): {"content": "<pad>"},
        },
    }), encoding="utf-8")
    return tokenizer_dir


def test_convert_checkpoint_raises_on_unmapped_ttml_tensor(tmp_path):
    """A ttml tensor hf_mapping.map_name doesn't recognize must raise, naming it -- not be
    silently dropped via `continue`, which is exactly how an untied model's real embedding
    table could vanish from the output with no error at all."""
    from convert.to_hf import convert_checkpoint

    tensors = _synth_tensors(weight_tying=True)
    tensors["llama/mystery_module/weight"] = _w(4, 4)
    ckpt = _write_fake_checkpoint(tmp_path / "fake.pkl", _synth_header(weight_tying=True), tensors)

    with pytest.raises(ValueError, match="llama/mystery_module/weight"):
        convert_checkpoint(ckpt, tmp_path / "no_tokenizer", tmp_path / "hf_out")


def test_convert_checkpoint_untied_writes_distinct_embedding_and_lm_head(tmp_path):
    """The fix this test exists for: before it, llama/fc/weight fanned out to both
    destinations unconditionally, so an untied checkpoint's real embedding table
    (llama/tok_emb/weight) would be reported as unmapped and dropped while fc/weight was
    duplicated into the embedding slot -- a model that loads, claims
    tie_word_embeddings=False, and is numerically wrong with no error anywhere."""
    from convert.to_hf import convert_checkpoint
    from safetensors.numpy import load_file

    header = _synth_header(weight_tying=False)
    tensors = _synth_tensors(weight_tying=False)
    ckpt = _write_fake_checkpoint(tmp_path / "fake.pkl", header, tensors)

    out_dir = tmp_path / "hf_out"
    cfg = convert_checkpoint(ckpt, tmp_path / "no_tokenizer", out_dir)
    assert cfg["tie_word_embeddings"] is False

    out = load_file(str(out_dir / "model.safetensors"))
    # tok_emb/weight and fc/weight were written as different arrays (np.arange, so every
    # element differs) -- if the old unconditional fan-out shipped, both HF keys would hold
    # fc/weight's values and this equality check would incorrectly pass.
    assert not np.array_equal(out["model.embed_tokens.weight"], out["lm_head.weight"])
    assert np.array_equal(out["model.embed_tokens.weight"], tensors["llama/tok_emb/weight"][0, 0])
    assert np.array_equal(out["lm_head.weight"], tensors["llama/fc/weight"][0, 0])


def test_convert_checkpoint_tied_writes_the_embedding_and_omits_lm_head(tmp_path):
    """Regression guard: the weight_tying=False fix above must not break the existing
    weight_tying=True path, and (per the duplicate-embedding decision recorded in
    .superpowers/sdd/2026-08-12-packaging/progress.md) the tied path now writes only
    model.embed_tokens.weight -- lm_head.weight is a deliberate omission, not a bug, so this
    checks its absence rather than checking the two arrays are equal."""
    from convert.to_hf import convert_checkpoint
    from safetensors.numpy import load_file

    header = _synth_header(weight_tying=True)
    tensors = _synth_tensors(weight_tying=True)
    ckpt = _write_fake_checkpoint(tmp_path / "fake.pkl", header, tensors)

    out_dir = tmp_path / "hf_out"
    cfg = convert_checkpoint(ckpt, tmp_path / "no_tokenizer", out_dir)
    assert cfg["tie_word_embeddings"] is True

    out = load_file(str(out_dir / "model.safetensors"))
    assert "lm_head.weight" not in out
    assert np.array_equal(out["model.embed_tokens.weight"], tensors["llama/fc/weight"][0, 0])


def test_convert_checkpoint_raises_on_missing_hf_key(tmp_path):
    """A truncated manifest (here: mlp/w3/weight, the up_proj source, missing entirely)
    must raise naming the missing key, not silently emit a safetensors file that
    transformers would later fill in with a randomly-initialized tensor and only a
    warning."""
    from convert.to_hf import convert_checkpoint

    header = _synth_header(weight_tying=True)
    tensors = _synth_tensors(weight_tying=True)
    del tensors["llama/llama_block_0/mlp/w3/weight"]
    ckpt = _write_fake_checkpoint(tmp_path / "fake.pkl", header, tensors)

    with pytest.raises(ValueError, match="model.layers.0.mlp.up_proj.weight"):
        convert_checkpoint(ckpt, tmp_path / "no_tokenizer", tmp_path / "hf_out")


def test_build_config_accepts_a_tokenizer_dir_whose_ids_match(tmp_path):
    tok_dir = _write_fake_tokenizer_dir(tmp_path / "tokenizer")
    c = build_config(_header(), tokenizer_dir=tok_dir)  # should not raise
    assert c["bos_token_id"] == 1
    assert c["eos_token_id"] == 2
    assert c["pad_token_id"] == 3


def test_build_config_raises_when_tokenizer_ids_disagree(tmp_path):
    """build_config hardcodes bos/eos/pad = 1/2/3; if the tokenizer this model actually ships
    with disagrees, that must be a hard failure, not a silently-wrong config.json."""
    tok_dir = _write_fake_tokenizer_dir(tmp_path / "tokenizer", eos=5)
    with pytest.raises(ValueError, match="eos_token_id"):
        build_config(_header(), tokenizer_dir=tok_dir)


def test_build_config_skips_the_check_when_tokenizer_dir_has_no_files(tmp_path):
    """No tokenizer files present -> nothing to cross-check against -> no raise. Matches
    convert_checkpoint's existing tolerance for a partially-populated tokenizer_dir."""
    c = build_config(_header(), tokenizer_dir=tmp_path)
    assert c["bos_token_id"] == 1


@pytest.mark.skipif(
    not Path("artifacts/tokenizer/special_tokens_map.json").is_file(),
    reason="no tokenizer artifacts on this machine",
)
def test_generation_config_is_written(tmp_path):
    """A published model needs eos_token_id and default sampling recorded --
    transformers logs a warning and falls back to config.json's (differently-scoped)
    defaults when generation_config.json is absent."""
    from convert.to_hf import convert_checkpoint

    header = _synth_header(weight_tying=True)
    tensors = _synth_tensors(weight_tying=True)
    ckpt = _write_fake_checkpoint(tmp_path / "fake.pkl", header, tensors)
    tok_dir = _write_fake_tokenizer_dir(tmp_path / "tokenizer")

    out = tmp_path / "hf_out"
    convert_checkpoint(ckpt, tok_dir, out)

    gen = json.loads((out / "generation_config.json").read_text(encoding="utf-8"))
    assert gen["eos_token_id"] == 2
    assert gen["bos_token_id"] == 1
    assert gen["pad_token_id"] == 3


def test_generation_config_ids_come_from_the_same_source_as_build_config():
    """The brief for this fix: generation_config.json's ids must be sourced from the same
    place build_config gets bos/eos/pad_token_id, so the two documents cannot disagree."""
    from convert.to_hf import _BOS_TOKEN_ID, _EOS_TOKEN_ID, _PAD_TOKEN_ID, build_config

    config = build_config(_header())
    assert config["bos_token_id"] == _BOS_TOKEN_ID
    assert config["eos_token_id"] == _EOS_TOKEN_ID
    assert config["pad_token_id"] == _PAD_TOKEN_ID


def test_tokenizer_class_matches_what_actually_loads(tmp_path):
    """tokenizer_config.json, as exported by convert/tokenizer.py's
    PreTrainedTokenizerFast.save_pretrained(), declares tokenizer_class: PreTrainedTokenizer
    (transformers strips the Fast suffix on save) while the tokenizer actually loads as
    PreTrainedTokenizerFast. convert_checkpoint must correct this when copying the tokenizer
    into out_dir -- artifacts/tokenizer/ itself is a separate, later-published artifact and
    must not be touched (see the source tokenizer_config.json still saying
    PreTrainedTokenizer, asserted below)."""
    from convert.to_hf import convert_checkpoint

    header = _synth_header(weight_tying=True)
    tensors = _synth_tensors(weight_tying=True)
    ckpt = _write_fake_checkpoint(tmp_path / "fake.pkl", header, tensors)
    tok_dir = _write_fake_tokenizer_dir(tmp_path / "tokenizer")
    # The fake tokenizer fixture doesn't set tokenizer_class; make it faithfully reproduce
    # the real upstream defect this test guards against.
    cfg = json.loads((tok_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    cfg["tokenizer_class"] = "PreTrainedTokenizer"
    (tok_dir / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")

    out = tmp_path / "hf_out"
    convert_checkpoint(ckpt, tok_dir, out)

    written = json.loads((out / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert written["tokenizer_class"] == "PreTrainedTokenizerFast"
    # The source tokenizer directory must be untouched -- it is a separate artifact,
    # published later, and changing it would invalidate the tokenizer's own tests.
    source = json.loads((tok_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert source["tokenizer_class"] == "PreTrainedTokenizer"


def test_max_position_embeddings_equals_the_trained_sequence_length(tmp_path):
    """Guards the serving trap: tokenizer_config.json advertises model_max_length
    1000000000000000019884624838656 (a sentinel for "no limit"). A caller deriving a serving
    max_model_len from the tokenizer instead of config.json would silently accept 4k-token
    contexts from a model trained to a 256-token window -- degraded output, no error."""
    from convert.to_hf import convert_checkpoint

    header = _synth_header(weight_tying=True)
    tensors = _synth_tensors(weight_tying=True)
    ckpt = _write_fake_checkpoint(tmp_path / "fake.pkl", header, tensors)
    tok_dir = _write_fake_tokenizer_dir(tmp_path / "tokenizer")

    out = tmp_path / "hf_out"
    convert_checkpoint(ckpt, tok_dir, out)

    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["max_position_embeddings"] == header["seq_len"] == 8


@pytest.mark.skipif(not CKPT.is_file(), reason="no trained checkpoint on this machine")
def test_max_position_embeddings_matches_the_real_checkpoint_header(tmp_path):
    """Same guard as above, against the real checkpoint rather than a synthetic one."""
    from convert.checkpoint_reader import read_checkpoint_meta
    from convert.to_hf import convert_checkpoint

    out = tmp_path / "hf_out"
    cfg = convert_checkpoint(CKPT, Path("artifacts/tokenizer"), out)
    header, _manifest = read_checkpoint_meta(CKPT)
    assert cfg["max_position_embeddings"] == header["seq_len"] == 256


def test_convert_checkpoint_raises_if_max_position_embeddings_disagrees_with_header(
    tmp_path,
):
    """The guard this fix exists to add: config.json's max_position_embeddings is derived
    from header["seq_len"] by build_config, so under normal operation they can never
    disagree -- this test forces the disagreement (by handing convert_checkpoint an
    already-built, tampered config) to prove the assertion actually fires rather than being
    unreachable dead code."""
    import convert.to_hf as to_hf

    header = _synth_header(weight_tying=True)
    tensors = _synth_tensors(weight_tying=True)
    ckpt = _write_fake_checkpoint(tmp_path / "fake.pkl", header, tensors)
    tok_dir = _write_fake_tokenizer_dir(tmp_path / "tokenizer")

    real_build_config = to_hf.build_config

    def _tampered_build_config(hdr, tokenizer_dir=None):
        cfg = real_build_config(hdr, tokenizer_dir=tokenizer_dir)
        cfg["max_position_embeddings"] += 1
        return cfg

    original = to_hf.build_config
    to_hf.build_config = _tampered_build_config
    try:
        with pytest.raises(ValueError, match="max_position_embeddings"):
            to_hf.convert_checkpoint(ckpt, tok_dir, tmp_path / "hf_out")
    finally:
        to_hf.build_config = original


def test_lm_head_weight_is_omitted_from_safetensors_under_tying(tmp_path):
    """Conventional practice with tie_word_embeddings: true is to omit lm_head.weight --
    transformers reconstructs it from the tied embedding at load time. Verified empirically
    (see .superpowers/sdd/2026-08-12-packaging/progress.md): loads with no warnings, ties
    correctly, bit-identical logits, 36% smaller on disk."""
    from convert.to_hf import convert_checkpoint
    from safetensors.numpy import load_file

    header = _synth_header(weight_tying=True)
    tensors = _synth_tensors(weight_tying=True)
    ckpt = _write_fake_checkpoint(tmp_path / "fake.pkl", header, tensors)
    tok_dir = _write_fake_tokenizer_dir(tmp_path / "tokenizer")

    out_dir = tmp_path / "hf_out"
    convert_checkpoint(ckpt, tok_dir, out_dir)

    out = load_file(str(out_dir / "model.safetensors"))
    assert "lm_head.weight" not in out
    assert "model.embed_tokens.weight" in out


def test_lm_head_weight_is_present_when_untied(tmp_path):
    """Regression guard: the tying-only omission above must not touch the untied path,
    where embed_tokens and lm_head are two genuinely distinct tensors."""
    from convert.to_hf import convert_checkpoint
    from safetensors.numpy import load_file

    header = _synth_header(weight_tying=False)
    tensors = _synth_tensors(weight_tying=False)
    ckpt = _write_fake_checkpoint(tmp_path / "fake.pkl", header, tensors)
    tok_dir = _write_fake_tokenizer_dir(tmp_path / "tokenizer")

    out_dir = tmp_path / "hf_out"
    convert_checkpoint(ckpt, tok_dir, out_dir)

    out = load_file(str(out_dir / "model.safetensors"))
    assert "lm_head.weight" in out


def test_loaded_model_still_ties_embedding_and_lm_head_without_the_duplicate(tmp_path):
    """The load-time proof, not just the file-contents proof: AutoModelForCausalLM must
    still tie model.embed_tokens.weight and lm_head.weight (torch.equal, not just same
    shape) when lm_head.weight was never in the safetensors file to begin with."""
    from transformers import AutoModelForCausalLM
    import torch

    from convert.to_hf import convert_checkpoint

    header = _synth_header(weight_tying=True)
    tensors = _synth_tensors(weight_tying=True)
    ckpt = _write_fake_checkpoint(tmp_path / "fake.pkl", header, tensors)
    tok_dir = _write_fake_tokenizer_dir(tmp_path / "tokenizer")

    out_dir = tmp_path / "hf_out"
    convert_checkpoint(ckpt, tok_dir, out_dir)

    m = AutoModelForCausalLM.from_pretrained(str(out_dir))
    assert torch.equal(m.model.embed_tokens.weight, m.lm_head.weight)


def test_completeness_check_expects_lm_head_only_when_untied(tmp_path):
    """The completeness post-condition's expected-key count changes shape under tying: with
    lm_head.weight omitted, a tied checkpoint's expected set is 9*num_hidden_layers + 2
    (embed_tokens, norm) rather than +3 -- and this must not silently accept a genuinely
    missing key by undercounting, nor reject a correct tied conversion by still demanding
    lm_head.weight."""
    from convert.to_hf import convert_checkpoint
    from safetensors.numpy import load_file

    tied_header = _synth_header(weight_tying=True)
    tied_tensors = _synth_tensors(weight_tying=True)
    tied_ckpt = _write_fake_checkpoint(tmp_path / "tied.pkl", tied_header, tied_tensors)
    tok_dir = _write_fake_tokenizer_dir(tmp_path / "tokenizer")
    tied_out = tmp_path / "tied_out"
    convert_checkpoint(tied_ckpt, tok_dir, tied_out)
    tied_keys = set(load_file(str(tied_out / "model.safetensors")))
    # 9 per-layer suffixes * 1 layer + embed_tokens + norm (no lm_head under tying).
    assert len(tied_keys) == 9 * 1 + 2

    untied_header = _synth_header(weight_tying=False)
    untied_tensors = _synth_tensors(weight_tying=False)
    untied_ckpt = _write_fake_checkpoint(tmp_path / "untied.pkl", untied_header, untied_tensors)
    untied_out = tmp_path / "untied_out"
    convert_checkpoint(untied_ckpt, tok_dir, untied_out)
    untied_keys = set(load_file(str(untied_out / "model.safetensors")))
    # 9 per-layer suffixes * 1 layer + embed_tokens + norm + lm_head.
    assert len(untied_keys) == 9 * 1 + 3


def test_build_config_matches_the_real_tokenizer():
    """The guard this repo actually relies on: artifacts/tokenizer/'s real special-token
    ids must agree with the hardcoded values, today and after any future tokenizer rebuild."""
    c = build_config(_header(), tokenizer_dir=Path("artifacts/tokenizer"))
    assert c["bos_token_id"] == 1
    assert c["eos_token_id"] == 2
    assert c["pad_token_id"] == 3


def test_convert_to_hf_module_imports_no_tenstorrent():
    """convert/ must run on a machine with no hardware and no tt-metal checkout, and
    convert.to_hf is the module that actually gets run at conversion time -- a stray ttml/ttnn
    import here would defeat the whole point of a CPU-only conversion path. torch is
    deliberately not banned -- transformers imports it transitively and CPU torch runs
    anywhere.

    Checked in a subprocess: this test session has already imported plenty, so inspecting
    our own sys.modules would prove nothing.
    """
    import subprocess
    import sys

    probe = (
        "import sys; import convert.to_hf; "
        "bad=[m for m in ('ttnn','ttml') if m in sys.modules]; "
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, check=True, cwd=str(Path(__file__).parent.parent),
    )
    assert out.stdout.strip() == "", f"convert.to_hf pulled in: {out.stdout.strip()}"
