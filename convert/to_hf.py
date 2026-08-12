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
from typing import Any, Dict, List, Optional

import numpy as np

from convert.checkpoint_reader import read_checkpoint_meta, read_tensors
from convert.hf_mapping import map_name, permute_rope_qk, split_kv, squeeze_leading

_REQUIRED = ("vocab_size", "seq_len", "intermediate_dim", "weight_tying",
             "rms_norm_eps", "weights_dtype", "transformer_config")

#: Nested under header["transformer_config"] -- validated separately from _REQUIRED
#: because a missing key there is otherwise a bare KeyError instead of the same
#: informative ValueError the top-level fields get.
_REQUIRED_TRANSFORMER_CONFIG = ("embedding_dim", "num_blocks", "num_heads", "num_groups", "theta")

#: Hardcoded rather than read from the tokenizer, because the checkpoint header carries no
#: tokenizer information at all -- these are believed correct for this project's own
#: tokenizer (verified against artifacts/tokenizer/), not discovered generically. build_config
#: cross-checks them against the tokenizer's own files (see _tokenizer_special_ids) as a
#: guard against silent drift, not as a replacement for keeping this source of truth right.
_BOS_TOKEN_ID = 1
_EOS_TOKEN_ID = 2
_PAD_TOKEN_ID = 3


def _tokenizer_special_ids(tokenizer_dir: Path) -> Optional[Dict[str, int]]:
    """Resolve (bos, eos, pad) token ids from the tokenizer's own on-disk files.

    Returns ``None`` if either file is missing, matching ``convert_checkpoint``'s existing
    tolerance for optional tokenizer files (it copies whatever of the three tokenizer files
    is present rather than requiring all three) -- and so a caller building a config from a
    bare header with no tokenizer directory handy (e.g. the unit tests in ``test_to_hf.py``)
    doesn't have this check forced on it.

    Raises if the files exist but a special token's id can't be resolved at all -- that's a
    malformed tokenizer export, not a value this function has an opinion about.
    """
    special_map_path = tokenizer_dir / "special_tokens_map.json"
    tok_config_path = tokenizer_dir / "tokenizer_config.json"
    if not special_map_path.is_file() or not tok_config_path.is_file():
        return None

    special_map = json.loads(special_map_path.read_text(encoding="utf-8"))
    tok_config = json.loads(tok_config_path.read_text(encoding="utf-8"))
    # tokenizer_config.json's added_tokens_decoder maps id (as a string key) -> token info,
    # including the literal token text -- the reverse of what we need, so invert it.
    content_to_id = {
        info["content"]: int(id_str)
        for id_str, info in tok_config.get("added_tokens_decoder", {}).items()
    }

    ids: Dict[str, int] = {}
    for role, key in (("bos", "bos_token"), ("eos", "eos_token"), ("pad", "pad_token")):
        content = special_map.get(key)
        if content is None or content not in content_to_id:
            raise ValueError(
                f"tokenizer at {tokenizer_dir} has no resolvable id for {key!r}: "
                f"special_tokens_map.json says {content!r}, which does not appear in "
                f"tokenizer_config.json's added_tokens_decoder"
            )
        ids[role] = content_to_id[content]
    return ids


def build_config(
    header: Dict[str, Any], tokenizer_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """HF ``LlamaConfig`` fields, entirely from ``header``.

    ``bos_token_id``/``eos_token_id``/``pad_token_id`` are hardcoded (see
    ``_BOS_TOKEN_ID`` et al.) rather than read from anywhere, since the checkpoint header
    carries no tokenizer information. When ``tokenizer_dir`` is given, those hardcoded values
    are cross-checked against the tokenizer's own ``special_tokens_map.json`` /
    ``tokenizer_config.json`` and this raises on disagreement -- a guard against the
    hardcoded values silently going stale if the tokenizer is ever regenerated with different
    special-token ids, not a way of deriving them fresh each time.
    """
    missing = [f for f in _REQUIRED if f not in header]
    if missing:
        raise ValueError(
            f"checkpoint header missing field(s) required for conversion: "
            f"{', '.join(missing)}. Re-run scripts/backfill_checkpoint_headers.py."
        )
    tc = header["transformer_config"]
    missing_tc = [f for f in _REQUIRED_TRANSFORMER_CONFIG if f not in tc]
    if missing_tc:
        raise ValueError(
            f"checkpoint header's transformer_config missing field(s) required for "
            f"conversion: {', '.join(missing_tc)}. Re-run "
            f"scripts/backfill_checkpoint_headers.py."
        )
    config = {
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
        "bos_token_id": _BOS_TOKEN_ID,
        "eos_token_id": _EOS_TOKEN_ID,
        "pad_token_id": _PAD_TOKEN_ID,
    }

    if tokenizer_dir is not None:
        tok_ids = _tokenizer_special_ids(Path(tokenizer_dir))
        if tok_ids is not None:
            hardcoded = {"bos": config["bos_token_id"], "eos": config["eos_token_id"],
                         "pad": config["pad_token_id"]}
            mismatches = {role: (hardcoded[role], tok_ids[role])
                          for role in hardcoded if hardcoded[role] != tok_ids[role]}
            if mismatches:
                detail = ", ".join(
                    f"{role}_token_id: hardcoded {h} != tokenizer's {t}"
                    for role, (h, t) in mismatches.items()
                )
                raise ValueError(
                    f"build_config's hardcoded special-token ids disagree with "
                    f"{tokenizer_dir}'s tokenizer files: {detail}. Update "
                    f"_BOS_TOKEN_ID/_EOS_TOKEN_ID/_PAD_TOKEN_ID in convert/to_hf.py to match."
                )

    return config


def convert_checkpoint(ckpt: Path, tokenizer_dir: Path, out_dir: Path) -> Dict[str, Any]:
    """Write a loadable HF model directory. Returns the config that was written."""
    from safetensors.numpy import save_file

    ckpt, tokenizer_dir, out_dir = Path(ckpt), Path(tokenizer_dir), Path(out_dir)
    header, _manifest = read_checkpoint_meta(ckpt)
    config = build_config(header, tokenizer_dir=tokenizer_dir)
    tc = header["transformer_config"]
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    weight_tying = bool(header["weight_tying"])

    out: Dict[str, np.ndarray] = {}
    unmapped: List[str] = []
    for name, tensor in read_tensors(ckpt):
        target = map_name(name, weight_tying=weight_tying)
        if target is None:
            # Not a bare "nothing to do here" -- a renamed ttml module, a manifest entry
            # this converter doesn't know about, or (the case that matters most) an untied
            # model's real embedding table would all land here silently if this only
            # `continue`d. Collected and raised after the loop so one run reports every
            # unmapped name at once, rather than stopping at the first.
            unmapped.append(name)
            continue
        if name.endswith("attention/kv_linear/weight"):
            # split_kv squeezes leading unit dims internally before splitting.
            k, v = split_kv(tensor, num_groups=int(tc["num_groups"]), head_dim=head_dim)
            # RoPE applies to keys, never to values (see permute_rope_qk's docstring for
            # why the straight-copy version of this line shipped a numerically wrong
            # model that still loaded and generated plausible text).
            out[target[0]] = permute_rope_qk(
                k, num_heads=int(tc["num_groups"]), head_dim=head_dim
            )
            out[target[1]] = v
        elif name.endswith("attention/q_linear/weight"):
            arr = squeeze_leading(tensor)
            out[target] = permute_rope_qk(
                arr, num_heads=config["num_attention_heads"], head_dim=head_dim
            )
        elif isinstance(target, tuple):
            # Tied embedding: one tensor, both destinations.
            arr = squeeze_leading(tensor)
            out[target[0]] = arr
            out[target[1]] = arr
        else:
            out[target] = squeeze_leading(tensor)

    if unmapped:
        raise ValueError(
            f"{len(unmapped)} ttml tensor(s) in {ckpt} had no HF mapping and would have "
            f"been silently dropped: {', '.join(sorted(unmapped))}. This can mean a renamed "
            f"ttml module, a manifest entry hf_mapping.map_name doesn't know about, or -- "
            f"most importantly for weight_tying=False checkpoints -- a real embedding table "
            f"(llama/tok_emb/weight) that must not be discarded. Fix hf_mapping.map_name "
            f"rather than ignoring this."
        )

    # Cheap guard, not a full proof: this catches a down<->up mislabeling (down_proj is
    # defined as the transpose-shaped one of the pair), but gate_proj and up_proj share
    # the same shape, so a down<->gate label swap would pass this check too. It's still
    # worth keeping -- shape mismatches are exactly the class of MLP_ROLES typo this can
    # catch for free -- just don't read a pass here as proof the whole mapping is right.
    down = out.get("model.layers.0.mlp.down_proj.weight")
    gate = out.get("model.layers.0.mlp.gate_proj.weight")
    if down is not None and gate is not None and down.shape != gate.shape[::-1]:
        raise ValueError(
            f"MLP role assignment looks wrong: down_proj {down.shape} is not the "
            f"transpose-shape of gate_proj {gate.shape}. Check MLP_ROLES in hf_mapping."
        )

    # Completeness post-condition: the config implies an exact HF key set (9 tensors per
    # transformer layer, plus the embedding, final norm, and lm_head), and nothing upstream
    # of this point actually confirms the emitted safetensors file has all of them. A
    # truncated manifest -- a checkpoint saved mid-write, or a future ttml change that drops
    # a tensor -- would otherwise produce a safetensors file silently missing keys, and
    # `transformers.AutoModelForCausalLM.from_pretrained` fills gaps with a random
    # initialization and only a warning, not an error: exactly the "loads cleanly, silently
    # wrong" failure mode this whole conversion path exists to guard against.
    _PER_LAYER_SUFFIXES = (
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    )
    expected_keys = {
        f"model.layers.{i}.{suffix}"
        for i in range(config["num_hidden_layers"])
        for suffix in _PER_LAYER_SUFFIXES
    }
    expected_keys |= {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    # len(expected_keys) == 9 * num_hidden_layers + 3 by construction: 9 per-layer suffixes
    # above, times num_hidden_layers layers, plus the 3 top-level keys unioned in just above.

    actual_keys = set(out)
    missing_keys = expected_keys - actual_keys
    unexpected_keys = actual_keys - expected_keys
    if missing_keys or unexpected_keys:
        raise ValueError(
            "convert_checkpoint produced an incomplete/mismatched key set for "
            f"num_hidden_layers={config['num_hidden_layers']} "
            f"(expected {len(expected_keys)} = 9*layers + 3 keys, got {len(actual_keys)}). "
            f"Missing: {sorted(missing_keys) or 'none'}. "
            f"Unexpected: {sorted(unexpected_keys) or 'none'}."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_dir / "model.safetensors"))
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        src = tokenizer_dir / f
        if src.is_file():
            shutil.copy2(src, out_dir / f)
    return config
