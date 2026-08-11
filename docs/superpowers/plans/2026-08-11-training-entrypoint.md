<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Training Entrypoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train NanoLlama3 on Blackhole from our own corpus and tokenizer, using a training entrypoint this repo owns.

**Architecture:** Three layers with clean boundaries. `train/tokenization.py` turns the prepared corpus into token-id arrays on disk (pure Python, chunked, no Tenstorrent imports). `train/config.py` assembles the tt-train YAML config and a run-config object that carries every field `ttml.common.trainer.train()` actually reads. `train/run.py` is the hardware entrypoint: it reuses ttml's `TransformerModelFactory`, `create_optimizer`, and `train()` while supplying our own data and a real validation loop.

**Tech Stack:** Python 3.10+, `ttml` (tt-train, built at `~/tt-metal/build_Release`), `ttnn`, numpy, `transformers`/`tokenizers`, pytest

## Why this repo owns the entrypoint

There is **no working stock Python trainer** in any tt-metal checkout on this machine. Verified:

1. `tt-train/sources/examples/nano_gpt/` in `~/tt-metal` is **C++** (`main.cpp`), and it hardcodes `CharTokenizer` at line 507. `sources/ttml/tokenizers/` contains only `char_tokenizer` — there is no C++ BPE.
2. The Python successor, `sources/examples/python/transformers/training.py`, is stale in **three independent ways**:
   - it does `from trainer import train`, but no top-level `trainer` module is on its `sys.path` (the real one is `ttml.common.trainer`);
   - it calls `train(...)` with **7 positional args including `val_ids`**, while `ttml/common/trainer.py:49` accepts `(cfg, model, optim, train_ids, use_ddp=False, use_tp=False)` — six, with no `val_ids`;
   - `TrainingConfig` (`ttml/common/config.py:47-82`) **never defines `seq_len`**, which `train()` requires for `build_causal_mask(cfg.seq_len)` and `get_batch_ttml(...)`. `max_sequence_length` lives on `TransformerConfig` instead.
3. `train_nanogpt.py`, which the tt-vscode-toolkit lesson still invokes, exists only in `~/code/tt-metal` at `v0.66.0-dev` — older than `~/tt-metal` and older than the lesson's own verified v0.73.

Upgrading tt-metal does not fix this: `~/tt-metal-v0.75.0` ships the same stale `training.py`.

Two further behaviors we must work around rather than inherit:

- **`prepare_data` hardcodes the corpus.** `ttml/common/data.py:82-83` calls `load_shakespeare_text()`, which reads `$TT_METAL_HOME/tt-train/data/shakespeare.txt` with no parameter and no config key (`data.py:27-33`). It ignores `data_path` entirely.
- **Validation is a placeholder.** `ttml/common/trainer.py` (in the `eval_every` branch) does `val_losses.append(train_losses[-1])` under the comment "keep existing placeholder behavior". Any val_loss it reports **is the train loss**. We compute real validation ourselves.

## Global Constraints

- Every new Python file starts with:
  `# SPDX-License-Identifier: Apache-2.0`
  `# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC`
  (a `#!` shebang may precede them in an executable script)
- Python 3.10+
- **Purity boundary — read carefully, it differs from Plan 1.** `convert/`, `train/data.py`, and `train/tokenization.py` must NOT import `ttnn` or `ttml`. `train/run.py` and `train/config.py`'s device section are the hardware entrypoint and **are expected to import them** — do not apply the Plan 1 purity rule to those two files.
- Model architecture comes from tt-train's `nanollama3.yaml` and is not redefined here: `num_heads: 6`, `num_groups: 3`, `embedding_dim: 384`, `num_blocks: 6`, `max_sequence_length: 256`, `theta: 500000.0`, `model_type: llama`, `vocab_size: 32000`.
- **Sequence length is 256** everywhere. It is `max_sequence_length` on `TransformerConfig` and must be copied onto the run config as `seq_len`, because `train()` reads `cfg.seq_len` and `TrainingConfig` does not provide it.
- Inputs from Plan 1, already on disk: `artifacts/corpus/corpus.txt` (3,548,279 lines / 536,870,821 bytes) and `artifacts/tokenizer/` (vocabulary exactly 32000, `</s>` = id 2).
- `ttml` requires `TT_METAL_HOME` **and** `TT_METAL_RUNTIME_ROOT` set, plus `TT_METAL_ARCH_NAME=blackhole` on this hardware. Let `ttml` close the device — bypassing its shutdown triggers a teardown abort in `MetalContext::destroy_all_instances`. If the board times out on device open, `tt-smi -r` first.

---

## File Structure

| File | Responsibility |
|---|---|
| `train/tokenization.py` | Corpus text → token-id `.npy` arrays, chunked; train/val split |
| `train/config.py` | Assemble the tt-train YAML dict + a `RunConfig` carrying every field `train()` reads |
| `train/run.py` | Hardware entrypoint: build model/optimizer, run training, real validation, checkpoint |
| `tests/test_tokenization.py` | Chunked encoding, split boundaries, dtype, resume-on-existing |
| `tests/test_trainconfig.py` | Config assembly, `seq_len` propagation, vocab agreement |

`tokenization.py` is separate from `data.py` because they have different lifetimes and different failure modes: `data.py` is a one-time corpus fetch, `tokenization.py` is re-run whenever the tokenizer changes. (Originally named `tokenize.py`; renamed in the whole-branch fix wave because it shadowed the stdlib `tokenize` module — see Task 1.)

---

## Task 1: Corpus tokenization to disk

**Files:**
- Create: `train/tokenization.py`
- Test: `tests/test_tokenization.py`

**Interfaces:**
- Consumes: `artifacts/tokenizer/` (Plan 1), `artifacts/corpus/corpus.txt` (Plan 1)
- Produces:
  - `TOKEN_DTYPE = np.uint32` — module constant; matches what `get_batch_ttml` expects
  - `tokenize_corpus(corpus: Path, tokenizer_dir: Path, out_dir: Path, val_fraction: float = 0.1, chunk_lines: int = 50_000) -> TokenStats`
  - `TokenStats` — dataclass with `total_tokens: int`, `train_tokens: int`, `val_tokens: int`, `vocab_size: int`
  - Writes `out_dir/train_ids.npy` and `out_dir/val_ids.npy`

**Why chunked:** `ttml.common.data.prepare_data` encodes the entire corpus in one `encode(text)` call. Against 536 MB that is a single enormous tokenizer invocation and a multi-gigabyte intermediate list. We encode in line batches and append to a growing array instead.

- [ ] **Step 1: Write the failing tests**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Corpus tokenization. Uses a real tokenizer; no hardware, no ttml."""

from pathlib import Path

import numpy as np
import pytest

from convert.tokenizer import train_bpe
from train.tokenization import TOKEN_DTYPE, TokenStats, tokenize_corpus


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("tok")
    corpus = d / "train.txt"
    corpus.write_text("\n".join(["the cat sat on the mat", "a dog ran fast"] * 200) + "\n",
                      encoding="utf-8")
    return train_bpe(corpus, d / "tokenizer", vocab_size=400)


@pytest.fixture(scope="module")
def tiny_corpus(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("corpus") / "corpus.txt"
    p.write_text("\n".join([f"line number {i} the cat sat" for i in range(500)]) + "\n",
                 encoding="utf-8")
    return p


def test_writes_both_splits(tiny_corpus, tiny_tokenizer, tmp_path):
    out = tmp_path / "tokens"
    tokenize_corpus(tiny_corpus, tiny_tokenizer, out)
    assert (out / "train_ids.npy").is_file()
    assert (out / "val_ids.npy").is_file()


def test_dtype_is_uint32(tiny_corpus, tiny_tokenizer, tmp_path):
    out = tmp_path / "tokens"
    tokenize_corpus(tiny_corpus, tiny_tokenizer, out)
    assert np.load(out / "train_ids.npy").dtype == TOKEN_DTYPE


def test_split_fraction_and_totals(tiny_corpus, tiny_tokenizer, tmp_path):
    out = tmp_path / "tokens"
    stats = tokenize_corpus(tiny_corpus, tiny_tokenizer, out, val_fraction=0.1)
    train = np.load(out / "train_ids.npy")
    val = np.load(out / "val_ids.npy")
    assert isinstance(stats, TokenStats)
    assert stats.train_tokens == len(train)
    assert stats.val_tokens == len(val)
    assert stats.total_tokens == len(train) + len(val)
    # 10% val, within int() truncation's rounding
    assert abs(stats.val_tokens / stats.total_tokens - 0.1) < 0.02


def test_chunking_does_not_change_output(tiny_corpus, tiny_tokenizer, tmp_path):
    """Chunk size is a memory knob, never a correctness knob."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    tokenize_corpus(tiny_corpus, tiny_tokenizer, a, chunk_lines=7)
    tokenize_corpus(tiny_corpus, tiny_tokenizer, b, chunk_lines=5000)
    assert np.array_equal(np.load(a / "train_ids.npy"), np.load(b / "train_ids.npy"))
    assert np.array_equal(np.load(a / "val_ids.npy"), np.load(b / "val_ids.npy"))


def test_vocab_size_reported_from_tokenizer(tiny_corpus, tiny_tokenizer, tmp_path):
    """`stats.vocab_size` is the tokenizer's ACHIEVED vocabulary, not the target.

    The fixture corpus exhausts BPE merges far below the 400 cap — 283 in practice —
    so this asserts agreement with the tokenizer itself rather than a hardcoded
    number. `vocab_size` is a ceiling, not a promise (see Plan 1).
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tiny_tokenizer), local_files_only=True)
    stats = tokenize_corpus(tiny_corpus, tiny_tokenizer, tmp_path / "tokens")
    assert stats.vocab_size == tok.vocab_size
    assert 260 < stats.vocab_size <= 400  # above the byte alphabet, under the cap


def test_eos_survives_tokenization(tiny_tokenizer, tmp_path):
    """A `</s>` line must become the single eos id, not several ordinary tokens."""
    corpus = tmp_path / "c.txt"
    corpus.write_text("the cat sat\n</s>\nthe dog ran\n", encoding="utf-8")
    tokenize_corpus(corpus, tiny_tokenizer, tmp_path / "tokens", val_fraction=0.0)
    ids = np.load(tmp_path / "tokens" / "train_ids.npy")
    assert 2 in ids.tolist(), "eos id 2 absent — separators are not reaching the token stream"


def test_rejects_val_fraction_above_one(tiny_corpus, tiny_tokenizer, tmp_path):
    """val_fraction > 1.0 makes `split` negative and `ids[:negative]` silently drops data."""
    with pytest.raises(ValueError, match="val_fraction"):
        tokenize_corpus(tiny_corpus, tiny_tokenizer, tmp_path / "tokens", val_fraction=1.5)


def test_rejects_non_positive_chunk_lines(tiny_corpus, tiny_tokenizer, tmp_path):
    with pytest.raises(ValueError, match="chunk_lines"):
        tokenize_corpus(tiny_corpus, tiny_tokenizer, tmp_path / "tokens", chunk_lines=0)
```

(The last two tests — the `val_fraction`/`chunk_lines` guards — were added in the later
whole-branch fix wave, alongside the `train/tokenize.py` → `train/tokenization.py` rename.
They are shown here so the plan matches the file as it exists on disk.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/code/tt-nanollama3 && python -m pytest tests/test_tokenization.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'train.tokenization'`

- [ ] **Step 3: Implement `train/tokenization.py`**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Turn a prepared corpus into token-id arrays on disk.

tt-train's own ``prepare_data`` (``ttml/common/data.py:82``) encodes the whole corpus in
one ``encode(text)`` call and splits 90/10 in memory. Against our 536 MB corpus that is a
single enormous tokenizer invocation and a multi-gigabyte intermediate Python list, so we
encode in line batches instead and grow a numpy array.

Chunk size is a **memory knob, not a correctness knob** — tokenizing the same corpus with
any two chunk sizes must produce byte-identical output, which the tests assert.

No ttnn/ttml imports: this runs on any machine.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

#: What ``ttml.common.trainer.get_batch_ttml`` expects to receive.
TOKEN_DTYPE = np.uint32


@dataclass
class TokenStats:
    """What ``tokenize_corpus`` produced, for logging and the model card."""

    total_tokens: int
    train_tokens: int
    val_tokens: int
    vocab_size: int


def tokenize_corpus(
    corpus: Path,
    tokenizer_dir: Path,
    out_dir: Path,
    val_fraction: float = 0.1,
    chunk_lines: int = 50_000,
) -> TokenStats:
    """Encode ``corpus`` with the tokenizer in ``tokenizer_dir``; write train/val ``.npy``.

    The split is taken at the end of the token stream (the last ``val_fraction`` of tokens
    become validation), matching tt-train's 90/10 tail split so our numbers stay comparable
    to its runs.
    """
    from transformers import AutoTokenizer

    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError(f"val_fraction must be in [0.0, 1.0], got {val_fraction}")
    if chunk_lines <= 0:
        raise ValueError(f"chunk_lines must be > 0, got {chunk_lines}")

    corpus, tokenizer_dir, out_dir = Path(corpus), Path(tokenizer_dir), Path(out_dir)
    if not corpus.is_file():
        raise FileNotFoundError(f"corpus not found: {corpus}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)

    pieces: List[np.ndarray] = []
    batch: List[str] = []

    def _flush() -> None:
        if not batch:
            return
        # add_special_tokens=False: the corpus already carries its own `</s>` separators;
        # letting the tokenizer inject more would double them at every story boundary.
        encoded = tok(batch, add_special_tokens=False)["input_ids"]
        flat = [i for seq in encoded for i in seq]
        if flat:
            pieces.append(np.asarray(flat, dtype=TOKEN_DTYPE))
        batch.clear()

    with corpus.open("r", encoding="utf-8") as fin:
        for line in fin:
            batch.append(line.rstrip("\n"))
            if len(batch) >= chunk_lines:
                _flush()
    _flush()

    ids = np.concatenate(pieces) if pieces else np.zeros(0, dtype=TOKEN_DTYPE)
    n_val = int(len(ids) * val_fraction)
    split = len(ids) - n_val
    train_ids, val_ids = ids[:split], ids[split:]

    np.save(out_dir / "train_ids.npy", train_ids)
    np.save(out_dir / "val_ids.npy", val_ids)

    return TokenStats(
        total_tokens=int(len(ids)),
        train_tokens=int(len(train_ids)),
        val_tokens=int(len(val_ids)),
        vocab_size=int(tok.vocab_size),
    )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default="artifacts/corpus/corpus.txt",
                    help="Path to the prepared corpus text file.")
    p.add_argument("--tokenizer", default="artifacts/tokenizer",
                    help="Directory holding the trained tokenizer.")
    p.add_argument("--out", default="artifacts/tokens",
                    help="Directory to write train_ids.npy / val_ids.npy into.")
    p.add_argument("--val-fraction", type=float, default=0.1,
                    help="Fraction of the token stream (tail) held out for validation.")
    p.add_argument("--chunk-lines", type=int, default=50_000,
                    help="Corpus lines encoded per tokenizer call (a memory knob only).")
    return p.parse_args(argv)


def main() -> int:
    args = _parse_args()
    stats = tokenize_corpus(
        Path(args.corpus),
        Path(args.tokenizer),
        Path(args.out),
        val_fraction=args.val_fraction,
        chunk_lines=args.chunk_lines,
    )
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

(The `argparse` CLI and the two validation guards above the tokenizer call were added in the
later whole-branch fix wave, alongside the `train/tokenize.py` → `train/tokenization.py`
rename — the rename is what makes `python train/tokenization.py` runnable as a script in
the first place, since `train/tokenize.py` shadowed the stdlib `tokenize` module that numpy
imports transitively.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/code/tt-nanollama3 && python -m pytest tests/test_tokenization.py -v`
Expected: 8 passed (includes the two `val_fraction`/`chunk_lines` guard tests added in the
later whole-branch fix wave; originally 6)

- [ ] **Step 5: Run the whole suite**

Run: `cd ~/code/tt-nanollama3 && python -m pytest -q`
Expected: 23 passed (15 from Plan 1 + 8 new)

- [ ] **Step 6: Commit**

```bash
cd ~/code/tt-nanollama3
git add train/tokenization.py tests/test_tokenization.py
git commit -m "feat(tokenize): chunked corpus tokenization to .npy token arrays

tt-train's prepare_data encodes the whole corpus in one call; against 536 MB
that is a multi-gigabyte intermediate. Encode in line batches instead, with
chunk size proven to be a memory knob rather than a correctness knob."
```

---

## Task 2: Training config assembly

**Files:**
- Create: `train/config.py`
- Test: `tests/test_trainconfig.py`

**Interfaces:**
- Consumes: nothing — this is pure config assembly. (`TokenStats` is consumed by Task 3,
  not here.)
- Produces:
  - `SEQ_LEN = 256`, `VOCAB_SIZE = 32000` — module constants
  - `build_yaml_config(tokenizer_dir, model_config_path, *, batch_size, max_steps, ...) -> dict`
  - `RunConfig` — a plain object carrying **every** attribute `ttml.common.trainer.train()` reads: `seq_len`, `steps`, `batch_size`, `gradient_accumulation_steps`, `eval_every`, plus `validation_batch_size` and `save_every` for our own loop
  - `run_config_from_yaml(yaml_config: dict) -> RunConfig`

**The `seq_len` trap this task exists to close:** `train()` reads `cfg.seq_len`, but `ttml.common.config.TrainingConfig` never sets it — `grep seq_len ttml/common/config.py` returns nothing. `max_sequence_length` lives on `TransformerConfig`. Passing a bare `TrainingConfig` to `train()` raises `AttributeError` before a single step runs. `RunConfig` copies it across explicitly.

- [ ] **Step 1: Write the failing tests**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Training config assembly. Pure dict/attribute work — no hardware, no ttml import."""

import pytest

from train.config import SEQ_LEN, VOCAB_SIZE, RunConfig, build_yaml_config, run_config_from_yaml


def _yaml(**kw):
    return build_yaml_config("artifacts/tokenizer", "/models/nanollama3.yaml", **kw)


def test_declares_bpe_and_tokenizer_path():
    cfg = _yaml()
    tc = cfg["training_config"]
    assert tc["tokenizer_type"] == "bpe"
    assert tc["tokenizer_path"] == "artifacts/tokenizer"


def test_carries_model_config_path():
    assert _yaml()["training_config"]["model_config"] == "/models/nanollama3.yaml"


def test_overrides_are_applied():
    tc = _yaml(batch_size=8, max_steps=1234)["training_config"]
    assert tc["batch_size"] == 8
    assert tc["max_steps"] == 1234


def test_run_config_carries_seq_len():
    """The whole point: ttml's TrainingConfig never sets seq_len, and train() needs it."""
    rc = run_config_from_yaml(_yaml())
    assert rc.seq_len == SEQ_LEN == 256


def test_run_config_has_every_field_train_reads():
    rc = run_config_from_yaml(_yaml(batch_size=4, max_steps=20))
    for field in ("seq_len", "steps", "batch_size", "gradient_accumulation_steps", "eval_every"):
        assert hasattr(rc, field), f"train() reads cfg.{field}; RunConfig lacks it"
    assert rc.steps == 20
    assert rc.batch_size == 4


def test_vocab_size_matches_the_tokenizer_contract():
    assert VOCAB_SIZE == 32000


def test_rejects_seq_len_beyond_model_capacity():
    with pytest.raises(ValueError, match="max_sequence_length"):
        _yaml(seq_len=512)


def test_emits_optimizer_section():
    """ttml.common.utils.create_optimizer raises ValueError without this section."""
    opt = _yaml()["training_config"]["optimizer"]
    assert opt["type"] == "AdamW"
    assert opt["lr"] == 0.0003
    assert opt["weight_decay"] == 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/code/tt-nanollama3 && python -m pytest tests/test_trainconfig.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'train.config'`

- [ ] **Step 3: Implement `train/config.py`**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Assemble the tt-train config for a NanoLlama3 run.

Two things this module exists to get right:

1. **``seq_len``.** ``ttml.common.trainer.train()`` reads ``cfg.seq_len`` for
   ``build_causal_mask()`` and ``get_batch_ttml()``, but ``ttml.common.config.TrainingConfig``
   never defines it — the value lives on ``TransformerConfig`` as ``max_sequence_length``.
   Handing ``train()`` a bare ``TrainingConfig`` raises ``AttributeError`` before it trains.
   ``RunConfig`` copies it across explicitly.
2. **The tokenizer path.** ttml resolves ``tokenizer_path`` relative to
   ``$TT_METAL_HOME/tt-train`` (``ttml/common/data.py:91``), which is *not* where our
   tokenizer lives. We bypass ttml's data loading entirely (see ``train/run.py``), so this
   path is recorded for provenance rather than consumed by ttml.

No ttnn/ttml imports here — this is dict and attribute work.
"""

from __future__ import annotations

from typing import Any, Dict

#: tt-train's ``nanollama3.yaml`` declares ``max_sequence_length: 256``.
SEQ_LEN = 256

#: Must equal the tokenizer's vocabulary (Plan 1 pins it at exactly this).
VOCAB_SIZE = 32000


class RunConfig:
    """Every attribute ``ttml.common.trainer.train()`` reads, plus what our loop needs.

    Deliberately a plain object rather than a subclass of ttml's ``TrainingConfig``: the
    fields ``train()`` requires are not the fields that class provides, and inheriting
    would hide exactly the mismatch this exists to fix.
    """

    def __init__(self, tc: Dict[str, Any]):
        self.seq_len = int(tc.get("seq_len", SEQ_LEN))
        self.steps = int(tc.get("max_steps", 20))
        self.batch_size = int(tc.get("batch_size", 64))
        self.validation_batch_size = int(
            tc.get("validation_batch_size", max(self.batch_size // 2, 1))
        )
        self.gradient_accumulation_steps = int(tc.get("gradient_accumulation_steps", 1))
        self.eval_every = int(tc.get("eval_every", 200))
        self.save_every = int(tc.get("model_save_interval", 0))
        self.checkpoint_dir = tc.get("checkpoint_dir", "artifacts/checkpoints")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"RunConfig(seq_len={self.seq_len}, steps={self.steps}, "
                f"batch_size={self.batch_size}, eval_every={self.eval_every})")


def build_yaml_config(
    tokenizer_dir: str,
    model_config_path: str,
    *,
    seq_len: int = SEQ_LEN,
    batch_size: int = 64,
    max_steps: int = 20,
    eval_every: int = 200,
    gradient_accumulation_steps: int = 1,
    checkpoint_dir: str = "artifacts/checkpoints",
) -> Dict[str, Any]:
    """Build the config dict ``TransformerModelFactory`` and ``RunConfig`` consume."""
    if seq_len > SEQ_LEN:
        raise ValueError(
            f"seq_len {seq_len} exceeds the model's max_sequence_length ({SEQ_LEN}); "
            "the RoPE tables and causal mask are built for that length."
        )
    return {
        "training_config": {
            "seed": 5489,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "max_steps": max_steps,
            "eval_every": eval_every,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            # Checkpointing is deferred to Stage 3 (needs ttml's checkpoint format read
            # first), so this is always 0 for now — `checkpoint_dir`/`save_every` below
            # are threaded through RunConfig but currently unused.
            "model_save_interval": 0,
            "checkpoint_dir": checkpoint_dir,
            "tokenizer_type": "bpe",
            "tokenizer_path": tokenizer_dir,
            "model_config": model_config_path,
            # REQUIRED: ttml.common.utils.create_optimizer raises
            # ValueError("training_config must contain an 'optimizer' section") without
            # this, and passes the dict straight to the C++ optimizer factory. Values
            # match tt-train's own training_shakespeare_nanollama3.yaml.
            "optimizer": {
                "type": "AdamW",
                "lr": 0.0003,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1.0e-8,
                "weight_decay": 0.01,
                "amsgrad": False,
                "stochastic_rounding": False,
            },
        },
        "device_config": {"mesh_shape": [1, 1], "enable_ddp": False, "enable_tp": False},
    }


def run_config_from_yaml(yaml_config: Dict[str, Any]) -> RunConfig:
    """Extract the run config from an assembled YAML dict."""
    return RunConfig(yaml_config.get("training_config", {}))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/code/tt-nanollama3 && python -m pytest tests/test_trainconfig.py -v`
Expected: 8 passed

- [ ] **Step 5: Run the whole suite and commit**

Run: `cd ~/code/tt-nanollama3 && python -m pytest -q` — expected 31 passed (29 originally;
+2 from the guard tests the later fix wave added to `tests/test_tokenization.py`).

```bash
git add train/config.py tests/test_trainconfig.py
git commit -m "feat(config): assemble tt-train config with a RunConfig that carries seq_len

ttml.common.config.TrainingConfig never defines seq_len, but trainer.train()
reads cfg.seq_len for the causal mask and batching -- passing a bare
TrainingConfig raises AttributeError before training starts. RunConfig copies
max_sequence_length across explicitly."
```

---

## Task 3: The hardware entrypoint

**Files:**
- Create: `train/run.py`
- Modify: `CLAUDE.md` (log the run and its numbers)

**Interfaces:**
- Consumes: `train.tokenization` outputs, `train.config.{build_yaml_config, run_config_from_yaml, VOCAB_SIZE}`
- Produces: a trained checkpoint under `artifacts/checkpoints/`, and a printed loss curve

**This task imports `ttml` and `ttnn` and requires hardware.** It has no unit tests; its verification is a real short run with a monotonically decreasing loss.

- [ ] **Step 1: Implement `train/run.py`**

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Train NanoLlama3 on Tenstorrent hardware.

We own this entrypoint because tt-train's own Python trainer does not work against the
current tree: ``examples/python/transformers/training.py`` imports a ``trainer`` module
that is not on its path, calls ``train()`` with an extra ``val_ids`` argument the signature
does not accept, and relies on a ``TrainingConfig`` that lacks the ``seq_len`` ``train()``
requires. Its data loader also hardcodes ``$TT_METAL_HOME/tt-train/data/shakespeare.txt``.

What we reuse from ttml (never reimplemented): ``TransformerModelFactory``,
``create_optimizer``, ``initialize_device``, ``set_seed``, and the ``train()`` loop itself.
What we supply: our corpus, our tokenizer, ``seq_len``, and a **real** validation loss —
ttml's ``train()`` fills ``val_losses`` with a copy of the training loss under a comment
calling it placeholder behavior, so a val number from it means nothing.

    python train/run.py --steps 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.config import VOCAB_SIZE, build_yaml_config, run_config_from_yaml  # noqa: E402


def _default_tt_metal_home() -> str:
    return os.environ.get("TT_METAL_HOME", os.path.expanduser("~/tt-metal"))


def _prepare_env(tt_metal_home: str, arch: str) -> None:
    """ttml needs all three of these before import; it aborts without RUNTIME_ROOT."""
    os.environ.setdefault("TT_METAL_HOME", tt_metal_home)
    os.environ.setdefault("TT_METAL_RUNTIME_ROOT", tt_metal_home)
    os.environ.setdefault("TT_METAL_ARCH_NAME", arch)
    os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
    sys.path.append(f"{tt_metal_home}/tt-train/sources/ttml")


def evaluate(model, val_ids: np.ndarray, cfg, batches: int = 10) -> float:
    """Real validation loss over ``batches`` sampled windows.

    ttml's train() does not compute this — it appends the last training loss and labels it
    val_loss. We run the model in eval mode over held-out tokens and average properly.

    ``model.eval()`` only toggles dropout (0.0 in this config) — it does not disable
    gradient tracking. Without ``no_grad()``, every forward pass here would still build a
    full autograd graph that gets thrown away, wasting memory and compute and OOMing first
    at larger ``validation_batch_size``. ``no_grad`` also lives in ``ttml.common.utils``,
    alongside ``build_causal_mask``, so one import covers both.
    """
    import ttml
    import ttnn
    from ttml.common.trainer import get_batch_ttml
    from ttml.common.utils import build_causal_mask, no_grad

    mask = ttml.autograd.Tensor.from_numpy(
        build_causal_mask(cfg.seq_len), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
    )
    model.eval()
    total = 0.0
    with no_grad():
        for _ in range(batches):
            x, y = get_batch_ttml(val_ids, cfg.seq_len, cfg.validation_batch_size, False)
            logits = model(x, mask)
            loss = ttml.ops.loss.cross_entropy_loss(logits, y, ttml.ops.ReduceType.MEAN)
            total += float(loss.to_numpy().mean())
            ttml.autograd.AutoContext.get_instance().reset_graph()
    model.train()
    return total / batches


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens-dir", default=str(ROOT / "artifacts" / "tokens"))
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--arch", default="blackhole", choices=["blackhole", "wormhole_b0"])
    p.add_argument("--tt-metal-home", default=_default_tt_metal_home())
    p.add_argument("--dry-run", action="store_true",
                   help="Print the resolved config and exit without opening a device.")
    args = p.parse_args()

    tokens = Path(args.tokens_dir)
    train_path, val_path = tokens / "train_ids.npy", tokens / "val_ids.npy"
    if not train_path.is_file():
        print(f"ERROR: {train_path} not found. Run train/tokenization.py first.",
              file=sys.stderr)
        return 1
    if not val_path.is_file():
        print(f"ERROR: {val_path} not found. Run train/tokenization.py first.", file=sys.stderr)
        return 1

    model_config = Path(args.tt_metal_home) / "tt-train/configs/model_configs/nanollama3.yaml"
    if not model_config.is_file():
        print(f"ERROR: model config not found at {model_config}", file=sys.stderr)
        return 1

    yaml_config = build_yaml_config(
        str(ROOT / "artifacts" / "tokenizer"), str(model_config),
        batch_size=args.batch_size, max_steps=args.steps, eval_every=args.eval_every,
    )
    cfg = run_config_from_yaml(yaml_config)

    print(f"NanoLlama3 training — steps={cfg.steps} batch={cfg.batch_size} "
          f"seq_len={cfg.seq_len} arch={args.arch}")
    if args.dry_run:
        print("--dry-run set: not opening a device.")
        return 0

    _prepare_env(args.tt_metal_home, args.arch)

    import ttml  # noqa: E402
    from ttml.common.model_factory import TransformerModelFactory  # noqa: E402
    from ttml.common.trainer import train  # noqa: E402
    from ttml.common.utils import create_optimizer, initialize_device, set_seed  # noqa: E402

    train_ids = np.load(train_path)
    val_ids = np.load(val_path)
    print(f"  train tokens={len(train_ids):,}  val tokens={len(val_ids):,}")

    # Nothing else checks that the token stream fits the model's vocabulary. The model's
    # embedding table is sized from the model config yaml (transformer_config.vocab_size),
    # not from train.config.VOCAB_SIZE — config.py never reads the yaml, it only asserts
    # its own constant against itself. If the two disagree, or if a token id from a
    # different tokenizer slipped in, an out-of-range embedding lookup produces silent
    # garbage or an on-device fault with no diagnostic. Catch it here, before the device
    # is even open.
    with model_config.open("r", encoding="utf-8") as f:
        model_yaml = yaml.safe_load(f)
    model_vocab_size = model_yaml["transformer_config"]["vocab_size"]
    assert model_vocab_size == VOCAB_SIZE, (
        f"model config vocab_size ({model_vocab_size}) at {model_config} does not match "
        f"train.config.VOCAB_SIZE ({VOCAB_SIZE}); the tokenizer and model disagree on "
        "vocabulary size."
    )
    max_train_id = int(train_ids.max())
    assert max_train_id < VOCAB_SIZE, (
        f"max token id in {train_path} is {max_train_id}, which is >= VOCAB_SIZE "
        f"({VOCAB_SIZE}); these tokens were produced by a different tokenizer than the "
        "one the model config expects — re-tokenize with the matching tokenizer."
    )

    set_seed(yaml_config["training_config"]["seed"])
    try:
        initialize_device(yaml_config)
    except Exception:
        print(
            "ERROR: initialize_device failed to open the device. If the board timed out, "
            "run `tt-smi -r` to reset it and retry.",
            file=sys.stderr,
        )
        raise

    # Everything from here to the end of the function runs with the device open, so it
    # all belongs inside this try — model/optimizer construction included. If either
    # raises (bad config, on-device OOM) before train()/evaluate() even start, the device
    # must still be closed in the finally below, or teardown aborts in
    # MetalContext::destroy_all_instances.
    try:
        model = TransformerModelFactory(yaml_config).create_model()
        optimizer = create_optimizer(model, yaml_config)

        # ttml's train() sets the progress bar's val_loss to a copy of train_loss whenever
        # step % eval_every == 0 or step == 1 — it is not a real validation number. Tell the
        # operator before the bar starts printing it, not after they've already trusted it.
        print(
            "note: the progress bar's val_loss is ttml's placeholder (a copy of "
            "train_loss); the real validation loss is computed after training and "
            "printed below."
        )
        # train() takes exactly (cfg, model, optim, train_ids, use_ddp, use_tp) — no val_ids.
        train_losses, _ = train(cfg, model, optimizer, train_ids, False, False)
        val_loss = evaluate(model, val_ids, cfg)
        if train_losses:
            print(f"\nfirst train loss : {train_losses[0]:.4f}")
            print(f"last  train loss : {train_losses[-1]:.4f}")
        else:
            print("\nno training steps ran (--steps 0); no train loss to report.")
        print(f"real  val   loss : {val_loss:.4f}")
    finally:
        # Let ttml close the device — bypassing this triggers a teardown abort in
        # MetalContext::destroy_all_instances.
        ttml.autograd.AutoContext.get_instance().close_device()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

(The vocabulary assertions, the `initialize_device` `tt-smi -r` hint, the `no_grad()` guard
in `evaluate()`, the ttml placeholder banner, the `val_path` existence check, and the
`train_losses[0]` empty-list guard were all added in the later whole-branch fix wave. The
original `_SCRIPT_DIR` `sys.path` workaround that once preceded the `numpy` import — needed
only because `train/tokenize.py` shadowed the stdlib `tokenize` module — was removed in that
same fix wave, once the rename to `train/tokenization.py` made it unnecessary.)

- [ ] **Step 2: Verify the dry run needs no hardware**

Run: `cd ~/code/tt-nanollama3 && python train/run.py --dry-run --steps 20`
Expected: prints the resolved config and exits 0 without opening a device.

- [ ] **Step 3: Tokenize the real corpus**

Run:
```bash
cd ~/code/tt-nanollama3 && python -c "
from pathlib import Path
from train.tokenization import tokenize_corpus
s = tokenize_corpus(Path('artifacts/corpus/corpus.txt'), Path('artifacts/tokenizer'), Path('artifacts/tokens'))
print(s)
"
```
Expected: a `TokenStats` with `vocab_size=32000` and roughly 1.3–1.6×10^8 total tokens. Record the real numbers.

- [ ] **Step 4: Run training for real**

Run: `cd ~/code/tt-nanollama3 && python train/run.py --steps 20 --batch-size 64`

Expected: a monotonically decreasing training loss beginning near `ln(32000) ≈ 10.4` (a freshly initialized model's loss sits at the log of the vocabulary size), and a real validation loss at the end. If the board times out opening the device, run `tt-smi -r` and retry once.

Report the actual first and last losses, the val loss, and seconds per step. **Never quote the first step's timing as model performance** — it is timing the compiler.

- [ ] **Step 5: Log the run in CLAUDE.md and commit**

Add a section recording: tokens produced, steps run, first/last train loss, real val loss, and s/step. Then:

```bash
git add train/run.py CLAUDE.md
git commit -m "feat(train): hardware entrypoint with real validation

Owns the entrypoint because tt-train's Python trainer is broken three ways
against the current tree (stale trainer import, extra val_ids argument, and a
TrainingConfig with no seq_len), and its data loader hardcodes shakespeare.txt.
Reuses TransformerModelFactory, create_optimizer, and train(); supplies our
corpus, seq_len, and a real validation loss -- ttml's val_losses is a
documented placeholder that copies the training loss."
```

---

## Self-Review

**Spec coverage.** Implements the design spec's Stage 2 (training). Stage 3 (checkpoint → HF conversion), Stage 4 (kernels + parity), Stage 5 (vLLM adapter), and Stages 6–7 (bundle + publish) remain for later plans.

**Placeholders.** None — every step carries runnable code or an exact command with expected output.

**Type consistency.** `TOKEN_DTYPE`/`TokenStats` are defined in Task 1 and consumed by name in Task 3. `SEQ_LEN`, `RunConfig`, `build_yaml_config`, `run_config_from_yaml` are defined in Task 2 and used with the same signatures in Task 3. `train()` is called with exactly the six parameters `ttml/common/trainer.py:49` declares.

**Verified before writing, not assumed.** `train()`'s signature, `TrainingConfig`'s missing `seq_len`, the placeholder validation, `prepare_data`'s hardcoded corpus path, and the three independent breaks in `training.py` were all read from source in `~/tt-metal` rather than inferred. Plan 1's lesson was that unverified plan assertions ship verbatim.

## Known risks

- **`create_optimizer` and `initialize_device` signatures are taken from `training.py`'s usage**, which is otherwise stale code. If either differs, Task 3 Step 4 will fail fast at startup — check `ttml/common/utils.py` before assuming the model is at fault.
- **`get_batch_ttml` and `build_causal_mask` are imported from `ttml.common.trainer`** for the validation loop. They are module-level functions there today; if they move, `evaluate()` breaks while `train()` still works.
- **Token count is an estimate.** ~1.3–1.6×10^8 assumes roughly 3.5–4 bytes per token on English prose. The real number goes in CLAUDE.md.
- **No checkpoint is written yet.** `model_save_interval` is 0. Saving weights is Stage 3's problem, and Stage 3 needs to know ttml's checkpoint format (`ttml/checkpointing.py`) before this plan pretends to produce one.
