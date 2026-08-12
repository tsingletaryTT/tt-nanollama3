<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Hugging Face Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert a NanoLlama3 checkpoint into a Hugging Face model directory that loads with `AutoModelForCausalLM` and generates text.

**Architecture:** Entirely CPU-side, in `convert/`. `convert/checkpoint_reader.py` (from Plan 3) already reads a checkpoint's header and manifest with stdlib `pickle`; this plan adds tensor streaming to it. This plan adds a name/layout mapper and a writer that emits `config.json` + `model.safetensors` beside the existing tokenizer.

**Tech Stack:** Python 3.10+, numpy, `ml_dtypes` (bfloat16), `safetensors`, `transformers`, pytest

## The finding that shapes this plan: conversion needs no hardware

Plan 3 assumed conversion would need a device, because `ttml.checkpointing.load_checkpoint` restores *in place* into live `NamedParameters`, which requires a constructed model. That assumption is **wrong**, and it was verified wrong against the real artifact:

```
record 1: type=numpy.ndarray shape=(1, 1, 384, 384) dtype=bfloat16
record 2: type=numpy.ndarray shape=(1, 1, 1024, 384) dtype=bfloat16
```

`save_checkpoint` writes `pickle.dump(Sharding.from_tensor(t).gather(t), f)` per tensor (`ttml/checkpointing.py:169`), and what lands on disk is a **plain numpy array**. Reading the whole checkpoint — header, manifest, and every tensor — requires only `pickle`, `numpy`, and `ml_dtypes` for the bfloat16 dtype. Confirmed: `torch` was never imported during the read.

So there is **no on-device export step**. The entire conversion runs anywhere, including CI. Do not reintroduce a hardware dependency.

## Global Constraints

- Every new Python file starts with:
  `# SPDX-License-Identifier: Apache-2.0`
  `# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC`
- Python 3.10+
- **`convert/` must NOT import `ttnn` or `ttml`** — this is the whole point. `torch` is permitted (safetensors and transformers pull it), but no Tenstorrent stack.
- **No bare `assert` for guards** — `if ...: raise ValueError(...)`. `python -O` strips asserts.
- Architecture values come from the **checkpoint header**, never hardcoded. Plan 3 enriched it precisely so this plan need not guess. Reading them from anywhere else is a defect.
- The tokenizer is **not regenerated** — `artifacts/tokenizer/` already holds the 32000-token export, and its files are copied into the output directory as-is.

## What the checkpoint contains (verified, not assumed)

Header fields available (15 total): `format`, `step`, `vocab_size` (32000), `seq_len` (256), `intermediate_dim` (1024), `weight_tying` (True), `rms_norm_eps` (1e-05), `weights_dtype` ("bfloat16"), `batch_size` (64), `tokens_seen` (49152000), `corpus_tokens`, `transformer_config`, `model_config_path`, `tokenizer_dir`, `created_at`.

50 model tensors:

| Count | ttml name | HF target |
|---|---|---|
| 1 | `llama/fc/weight` | `model.embed_tokens.weight` **and** `lm_head.weight` |
| 6 | `llama/llama_block_N/attention/q_linear/weight` | `model.layers.N.self_attn.q_proj.weight` |
| 6 | `llama/llama_block_N/attention/kv_linear/weight` | **split** → `k_proj.weight`, `v_proj.weight` |
| 6 | `llama/llama_block_N/attention/out_linear/weight` | `model.layers.N.self_attn.o_proj.weight` |
| 6 | `llama/llama_block_N/attention_norm/gamma` | `model.layers.N.input_layernorm.weight` |
| 6 | `llama/llama_block_N/mlp/w1/weight` | `model.layers.N.mlp.gate_proj.weight` |
| 6 | `llama/llama_block_N/mlp/w3/weight` | `model.layers.N.mlp.up_proj.weight` |
| 6 | `llama/llama_block_N/mlp/w2/weight` | `model.layers.N.mlp.down_proj.weight` |
| 6 | `llama/llama_block_N/mlp_norm/gamma` | `model.layers.N.post_attention_layernorm.weight` |
| 1 | `llama/ln_fc/gamma` | `model.norm.weight` |

**The `w1`/`w2`/`w3` → gate/up/down assignment above is a hypothesis to verify, not a fact.** SwiGLU convention makes `w2` the down-projection, which is checkable from shape alone: gate and up are `(intermediate=1024, hidden=384)`, down is `(hidden=384, intermediate=1024)`. Task 1 asserts this rather than trusting it. Distinguishing gate from up requires a numerical check (Task 3).

## Four traps this plan exists to avoid

1. **Weight tying means there is no `tok_emb` tensor.** Verified: the manifest has `llama/fc/weight` and no `tok_emb` entry. A converter that expects an embedding tensor produces a model with a randomly-initialised embedding table **and no error**.
2. **`kv_linear` is a fused K+V projection.** Output dim is `num_groups(3) × head_dim(64) × 2 = 384`, which must be split into separate `k_proj` and `v_proj`. The split point is not guessable from the tensor alone.
3. **RoPE layout convention.** ttml and HF Llama may differ on whether q/k rows are stored interleaved or as split halves. Getting this wrong yields a model that **loads cleanly and generates garbage** — no exception, no warning. Task 3 exists to catch it numerically.
4. **Stream order is declaration order, not sorted order.** ttml writes tensor records in the order `_walk` emitted them, and `load_checkpoint` says so explicitly: *"file order owns the stream order"*. But `convert.checkpoint_reader.tensor_names()` returns a **sorted** list. Zipping that sorted list against the record stream mis-assigns **every tensor** — and the result still loads, because the shapes largely match. `read_tensors` (Task 1) must walk the manifest in declaration order, and must never use the sorted helper for this purpose.

---

## File Structure

| File | Responsibility |
|---|---|
| `convert/hf_mapping.py` | ttml name → HF name mapping, tensor reshaping, `kv_linear` split |
| `convert/to_hf.py` | Assemble `config.json`, write `model.safetensors`, copy tokenizer |
| `scripts/convert_checkpoint.py` | CLI over the above |
| `tests/test_hf_mapping.py` | Mapping and split logic against synthetic tensors |
| `tests/test_to_hf.py` | Config assembly; end-to-end against the real checkpoint when present |

---

## Task 1: Name mapping and tensor layout

**Files:**
- Create: `convert/hf_mapping.py`
- Modify: `convert/checkpoint_reader.py` — add `read_tensors`
- Test: `tests/test_hf_mapping.py`

**Interfaces:**
- Consumes: `convert.checkpoint_reader.read_record0` (Plan 3, existing)
- Produces, added to `convert/checkpoint_reader.py`:
  - `read_tensors(path, group="model") -> Iterator[tuple[str, np.ndarray]]` — streams `(name, array)` pairs **in manifest declaration order**. Plan 3's reader has no tensor-reading function at all; this adds it. It must walk the manifest's `named_parameters` dict in insertion order and must **not** use `tensor_names()`, which sorts (see trap 4). Yield one tensor at a time so peak memory stays at roughly one tensor, matching how ttml wrote them.
- Produces, in `convert/hf_mapping.py`:
  - `map_name(ttml_name: str) -> str | tuple[str, str] | None` — HF name, a pair for `kv_linear`, or `None` for tensors with no direct target
  - `split_kv(tensor, *, num_groups: int, head_dim: int) -> tuple[np.ndarray, np.ndarray]`
  - `squeeze_leading(tensor) -> np.ndarray` — drops ttml's leading `(1, 1, ...)` dims
  - `MLP_ROLES: dict[str, str]` — `{"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/code/tt-nanollama3 && python -m pytest tests/test_hf_mapping.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'convert.hf_mapping'`

- [ ] **Step 3: Implement `convert/hf_mapping.py`**

```python
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
```

- [ ] **Step 4: Run tests, then the suite, then commit**

Run: `python -m pytest tests/test_hf_mapping.py -v` — expected 12 passed.
Run: `python -m pytest -q` — expected 79 passed (67 prior + 12).

```bash
git add convert/hf_mapping.py tests/test_hf_mapping.py
git commit -m "feat(convert): ttml -> HF name and layout mapping

Three things a converter cannot infer from tensors alone, each a silent-wrong-model
risk: weight tying (fc/weight goes to both embed_tokens and lm_head, and no
embedding tensor exists), the fused kv_linear split, and ttml's leading unit dims."
```

---

## Task 2: Config assembly and safetensors output

**Files:**
- Create: `convert/to_hf.py`
- Create: `scripts/convert_checkpoint.py`
- Test: `tests/test_to_hf.py`

**Interfaces:**
- Consumes: `convert.checkpoint_reader`, `convert.hf_mapping`
- Produces:
  - `build_config(header: dict) -> dict` — HF `LlamaConfig` fields, **entirely from the header**
  - `convert_checkpoint(ckpt: Path, tokenizer_dir: Path, out_dir: Path) -> dict` — writes the directory, returns the config

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run to verify failure, then implement `convert/to_hf.py`**

Run: `python -m pytest tests/test_to_hf.py -v` → `ModuleNotFoundError`.

```python
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
```

`scripts/convert_checkpoint.py` is an argparse CLI over `convert_checkpoint`, defaulting to the newest checkpoint, `artifacts/tokenizer`, and `artifacts/hf/`.

- [ ] **Step 3: Run the tests and the suite, then commit**

Expected: `tests/test_to_hf.py` 6 passed (5 + 1 end-to-end); full suite 85 passed.

---

## Task 3: Numerical verification — the step that catches a silently-wrong model

**Files:**
- Create: `tests/test_hf_parity.py`

**Interfaces:** none — this task proves the conversion is correct.

**Why this task is the point of the plan.** Tasks 1–2 produce a directory that will load without error even if the RoPE layout convention is wrong, the K/V split is backwards, or gate and up are swapped. Every one of those produces fluent-looking garbage. Only numerical comparison catches them.

- [ ] **Step 1: Load the converted model and check it is structurally sane**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Numerical verification of the converted model. CPU only."""

from pathlib import Path

import pytest

HF = Path("artifacts/hf")

pytestmark = pytest.mark.skipif(
    not (HF / "config.json").is_file(),
    reason="no converted model; run scripts/convert_checkpoint.py first",
)


def test_loads_with_automodel():
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(str(HF))
    assert m.config.vocab_size == 32000
    assert m.config.num_key_value_heads == 3


def test_embedding_and_lm_head_are_tied():
    from transformers import AutoModelForCausalLM
    import torch

    m = AutoModelForCausalLM.from_pretrained(str(HF))
    assert torch.equal(m.model.embed_tokens.weight, m.lm_head.weight)


def test_forward_pass_produces_finite_logits():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(str(HF))
    m = AutoModelForCausalLM.from_pretrained(str(HF)).eval()
    ids = tok("Once upon a time", return_tensors="pt").input_ids
    with torch.no_grad():
        out = m(ids).logits
    assert out.shape[-1] == 32000
    assert torch.isfinite(out).all()


def test_next_token_distribution_is_not_uniform():
    """A mis-mapped model often yields near-uniform logits — entropy catches it.

    ln(32000) = 10.37 nats is the uniform ceiling. A model trained to ~1.88 loss must
    be far below that on in-distribution text.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(str(HF))
    m = AutoModelForCausalLM.from_pretrained(str(HF)).eval()
    ids = tok("Once upon a time there was a little girl named", return_tensors="pt").input_ids
    with torch.no_grad():
        logits = m(ids).logits[0, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum().item()
    assert entropy < 7.0, f"next-token entropy {entropy:.2f} nats is near-uniform (10.37)"
```

- [ ] **Step 2: Generate text and read it**

Run:

```bash
cd ~/code/tt-nanollama3 && python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
tok = AutoTokenizer.from_pretrained('artifacts/hf')
m = AutoModelForCausalLM.from_pretrained('artifacts/hf').eval()
ids = tok('Once upon a time', return_tensors='pt').input_ids
with torch.no_grad():
    out = m.generate(ids, max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.95)
print(tok.decode(out[0], skip_special_tokens=True))
"
```

**This is the first time anyone reads this model's output.** Report the sample verbatim, without cleanup.

Expected, honestly: TinyStories-flavoured English with simple sentence structure — plausibly coherent for a clause or two, likely drifting. At 0.43 of an epoch, do not expect narrative consistency. If it emits repeated tokens, pure punctuation, or obviously random words, **that is a finding**: it means one of the three layout traps is live, and the conversion is wrong even though every test above passed.

- [ ] **Step 3: Cross-check perplexity against the training run**

Compute HF-side loss on held-out validation tokens and compare with the training run's **1.8781**. They should be close. A large gap points at a layout error that entropy alone did not catch.

```bash
python -c "
import numpy as np, torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('artifacts/hf').eval()
val = np.load('artifacts/tokens/val_ids.npy')[:256*20].astype('int64').reshape(20, 256)
x = torch.from_numpy(val)
with torch.no_grad():
    out = m(x[:, :-1], labels=x[:, 1:])
print('HF-side val loss:', float(out.loss))
"
```

Report the number. Within ~0.2 nats of 1.8781 is a pass; a gap of 1+ nats means the conversion is wrong.

- [ ] **Step 4: Record results in CLAUDE.md and commit**

Include the generated sample verbatim, the entropy, and the HF-side loss against 1.8781.

---

## Self-Review

**Every capability named above resolves to code or a test.** Task 1's mapping → `test_hf_mapping.py` (12 named tests). Task 2's config assembly → `test_to_hf.py::test_dimensions_come_from_the_header`, and the safetensors output → `test_end_to_end_against_the_real_checkpoint`. Task 3's numerical verification → `test_next_token_distribution_is_not_uniform` plus the two reported measurements. No table above claims coverage no task delivers.

**Does the artifact satisfy the rationale?** The stated reason for this plan is producing a model that loads *and is correct*. Tasks 1–2 alone would satisfy only the first half — which is why Task 3 exists and why the plan is not done without it.

**Placeholders.** None.

**Verified before writing.** Tensor payloads unpickle as plain numpy arrays (checked against the real checkpoint); the 50-tensor inventory and its naming pattern; `weight_tying` with no `tok_emb`; `ml_dtypes` supplying bfloat16 without torch. The `w1`/`w2`/`w3` role assignment is explicitly marked a hypothesis, guarded by a shape check in `convert_checkpoint` and settled numerically in Task 3.

## Known risks

- **Gate/up swap is shape-invisible.** `gate_proj` and `up_proj` have identical shapes, so only Task 3's numerical checks can distinguish them. If loss is high but entropy is reasonable, swap `w1`/`w3` in `MLP_ROLES` and re-measure — that is the first thing to try.
- **RoPE layout.** If HF-side loss is far above 1.8781 while structure checks pass, suspect interleaved-vs-split-halves in q/k. Fixing it means permuting rows within each head, not changing the mapping.
- **`safetensors.numpy` and bfloat16 — verified working**, so this is no longer a risk: a round-trip of an `ml_dtypes.bfloat16` array through `save_file`/`load_file` preserves the dtype. Noted because the earlier draft of this plan listed it as a hazard.
- **Plan 3's reader had no tensor-reading function**, and this plan's first draft imported two functions (`read_header`, `read_tensors`) that did not exist. Corrected above: `read_checkpoint_meta` is the real header accessor, and `read_tensors` is new work in Task 1. Check any other API a plan names before dispatching it.
