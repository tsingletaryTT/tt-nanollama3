<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Checkpointing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save and resume NanoLlama3 training checkpoints, then run a real training job that produces weights worth converting.

**Architecture:** `train/checkpoint.py` wraps `ttml.checkpointing` with our header schema and nothing more — ttml already streams tensors atomically, so we add metadata and path conventions, not storage. `train/run.py` gains periodic checkpointing by calling ttml's `train()` in chunks and saving between them, because ttml's `train()` has no checkpoint hook of its own.

**Tech Stack:** Python 3.10+, `ttml` (tt-train, built at `~/tt-metal/build_Release`), `ttnn`, numpy, pytest

## Why chunked training rather than a checkpoint callback

`ttml.common.trainer.train()` runs its entire step loop internally and **never checkpoints** — `grep -n "save\|checkpoint" ttml/common/trainer.py` returns nothing, even though `TrainingConfig` exposes a `save_every` field (`ttml/common/config.py:69`). That field is dead: nothing in ttml reads it.

There is no callback parameter, so the only ways to checkpoint periodically are to reimplement the loop (forbidden by the repo's reuse policy, and it would duplicate gradient accumulation, DDP synchronization, and the bf16 loss handling) or to call `train()` repeatedly with a smaller `cfg.steps` and save between calls. We do the latter.

This is safe because `train()` holds no cross-call state: it takes the model and optimizer as arguments and mutates them in place. The AdamW moments live in the optimizer object, which persists across calls. What does **not** carry across calls is the progress bar (restarts per chunk) and the returned `train_losses` list (per-call), so we accumulate those ourselves.

## What ttml already provides — do not reimplement

Verified in `~/tt-metal/tt-train/sources/ttml/ttml/checkpointing.py`:

- `save_checkpoint(path, *, header=None, model_params=None, optimizer=None, display_progress=False)` — writes an opaque `header` dict plus model and/or optimizer tensors. **Atomic**: writes `path + ".tmp"` then renames, so a crash mid-write leaves any previous checkpoint intact (`checkpointing.py:165-169`). Tensors are gathered to host **one at a time**, so peak host memory is roughly one tensor.
- `load_checkpoint(path, *, model_params=None, optimizer=None, display_progress=False) -> dict` — restores in place, resharded onto the current mesh, and returns the header. A group present in the file but not requested is skipped; requesting a group the file lacks is an error.
- `read_header(path) -> dict` — reads the header **without reading any tensor data**.

The canonical usage example is `~/tt-metal/tt-train/sources/examples/train/checkpointing.py`; our header schema follows its shape.

## Global Constraints

- Every new Python file starts with:
  `# SPDX-License-Identifier: Apache-2.0`
  `# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC`
  (a `#!` shebang may precede them in an executable script)
- Python 3.10+
- **Purity boundary.** `convert/`, `train/data.py`, `train/tokenization.py`, and `train/config.py` must NOT import `ttnn` or `ttml`. `train/run.py` and `train/checkpoint.py` are hardware-side and may.
- **No bare `assert` for guards.** Plan 2 left vocab guards as bare `assert`, which `python -O` strips — a guard against silent corruption that can itself be silently removed. All new guards use `if ...: raise ValueError(...)`. Do not add bare-`assert` guards, and convert the two existing ones in `train/run.py` while you are there.
- Model architecture is unchanged and comes from tt-train's `nanollama3.yaml`: `embedding_dim: 384`, `num_blocks: 6`, `num_heads: 6`, `num_groups: 3`, `max_sequence_length: 256`, `theta: 500000.0`, `vocab_size: 32000`.
- Inputs already on disk from Plans 1–2: `artifacts/tokenizer/` (vocab exactly 32000) and `artifacts/tokens/{train_ids,val_ids}.npy` (127,635,889 tokens: 114,872,301 train / 12,763,588 val).
- `ttml` needs `TT_METAL_HOME`, `TT_METAL_RUNTIME_ROOT`, and `TT_METAL_ARCH_NAME=blackhole`. Let `ttml` close the device. If the board times out on device open, `tt-smi -r` first.
- Checkpoints are written under `artifacts/checkpoints/` and are **gitignored** — they are hundreds of MB.

---

## File Structure

| File | Responsibility |
|---|---|
| `train/checkpoint.py` | Header schema + save/load/peek wrappers over `ttml.checkpointing` |
| `train/run.py` (modify) | Chunked training loop, periodic saves, `--resume`, non-`assert` guards |
| `tests/test_checkpoint.py` | Header schema construction and validation, no hardware |

`train/checkpoint.py` is separate from `run.py` because the conversion work in the next plan needs to read checkpoint headers without importing the training entrypoint.

---

## Task 1: Header schema and checkpoint wrappers

**Files:**
- Create: `train/checkpoint.py`
- Test: `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: `train.config.VOCAB_SIZE`, `train.config.SEQ_LEN`
- Produces:
  - `CHECKPOINT_FORMAT = 1` — our header schema version, distinct from ttml's own `FORMAT_VERSION`
  - `build_header(step: int, *, model_config_path: str, tokenizer_dir: str, total_tokens: int, extra: dict | None = None) -> dict`
  - `validate_header(header: dict) -> None` — raises `ValueError` naming the missing or mismatched field
  - `checkpoint_path(checkpoint_dir: Path, step: int) -> Path` — returns `<dir>/nanollama3_step<step>.pkl`
  - `save(path, *, header, model_params, optimizer, display_progress=False) -> None` — thin pass-through to ttml
  - `load(path, *, model_params=None, optimizer=None, display_progress=False) -> dict`
  - `peek(path) -> dict` — header only, no tensor reads

**Why a header schema at all:** ttml's header is opaque, so nothing validates it. A checkpoint whose header omits `vocab_size` is unconvertible later without guessing, and guessing is how a converted model silently mismatches its tokenizer. The schema is what makes the next plan's conversion step possible from the checkpoint alone.

- [ ] **Step 1: Write the failing tests**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Checkpoint header schema. Pure dict work — no hardware, no ttml import."""

from pathlib import Path

import pytest

from train.checkpoint import (
    CHECKPOINT_FORMAT,
    build_header,
    checkpoint_path,
    validate_header,
)


def _header(**kw):
    base = dict(
        step=100,
        model_config_path="/models/nanollama3.yaml",
        tokenizer_dir="artifacts/tokenizer",
        total_tokens=127_635_889,
    )
    base.update(kw)
    return build_header(**base)


def test_header_carries_format_version():
    assert _header()["format"] == CHECKPOINT_FORMAT


def test_header_carries_resume_and_conversion_fields():
    h = _header()
    for field in ("step", "vocab_size", "seq_len", "model_config_path",
                  "tokenizer_dir", "total_tokens", "created_at"):
        assert field in h, f"header missing {field}"


def test_header_records_vocab_and_seq_len_from_config():
    from train.config import SEQ_LEN, VOCAB_SIZE

    h = _header()
    assert h["vocab_size"] == VOCAB_SIZE
    assert h["seq_len"] == SEQ_LEN


def test_extra_is_merged_without_clobbering_required_fields():
    h = _header(extra={"note": "smoke run"})
    assert h["note"] == "smoke run"
    assert h["step"] == 100  # extra must not overwrite schema fields


def test_extra_cannot_override_a_schema_field():
    with pytest.raises(ValueError, match="may not override"):
        _header(extra={"vocab_size": 999})


def test_validate_accepts_a_built_header():
    validate_header(_header())  # must not raise


def test_validate_rejects_missing_field():
    h = _header()
    del h["vocab_size"]
    with pytest.raises(ValueError, match="vocab_size"):
        validate_header(h)


def test_validate_rejects_future_format():
    h = _header()
    h["format"] = CHECKPOINT_FORMAT + 1
    with pytest.raises(ValueError, match="format"):
        validate_header(h)


def test_checkpoint_path_is_step_numbered():
    p = checkpoint_path(Path("/ckpt"), 2500)
    assert p == Path("/ckpt/nanollama3_step00002500.pkl")  # zero-padded; see the sort test


def test_checkpoint_paths_sort_lexicographically_by_step():
    """Zero-padding matters: without it, step10 sorts before step9."""
    paths = sorted(str(checkpoint_path(Path("/c"), s)) for s in (9, 10, 100))
    assert paths == [str(checkpoint_path(Path("/c"), s)) for s in (9, 10, 100)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/code/tt-nanollama3 && python -m pytest tests/test_checkpoint.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'train.checkpoint'`

- [ ] **Step 3: Implement `train/checkpoint.py`**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Checkpoint header schema and thin wrappers over ``ttml.checkpointing``.

ttml already does the hard part — it streams tensors to disk one at a time and writes
atomically (temp file then rename), so a crash mid-write leaves the previous checkpoint
intact. We add two things it deliberately leaves open:

1. **A validated header schema.** ttml's header is an opaque dict, so nothing checks it. A
   checkpoint whose header omits ``vocab_size`` cannot be converted later without guessing,
   and guessing is how a converted model silently mismatches its tokenizer.
2. **Path conventions**, so checkpoints sort by step and a resume can find the newest.

Everything else is a pass-through. Do not reimplement ttml's storage.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Dict, Optional

from train.config import SEQ_LEN, VOCAB_SIZE

#: Our header schema version, independent of ttml's own on-disk FORMAT_VERSION.
#: Bump when a field's meaning changes, not when one is added.
CHECKPOINT_FORMAT = 1

#: Fields every checkpoint header must carry. `extra` may not shadow any of these.
_REQUIRED = (
    "format", "step", "vocab_size", "seq_len",
    "model_config_path", "tokenizer_dir", "total_tokens", "created_at",
)


def build_header(
    step: int,
    *,
    model_config_path: str,
    tokenizer_dir: str,
    total_tokens: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the header stored alongside a checkpoint's tensors.

    ``vocab_size`` and ``seq_len`` are recorded from ``train.config`` rather than passed in:
    they must describe the model that produced these weights, and taking them from the
    single source of truth removes the chance of a caller recording something else.
    """
    header: Dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "step": int(step),
        "vocab_size": VOCAB_SIZE,
        "seq_len": SEQ_LEN,
        "model_config_path": str(model_config_path),
        "tokenizer_dir": str(tokenizer_dir),
        "total_tokens": int(total_tokens),
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    if extra:
        clashes = sorted(set(extra) & set(_REQUIRED))
        if clashes:
            raise ValueError(f"extra may not override schema field(s): {', '.join(clashes)}")
        header.update(extra)
    return header


def validate_header(header: Dict[str, Any]) -> None:
    """Raise ``ValueError`` if ``header`` is not a checkpoint header this code can read."""
    missing = [f for f in _REQUIRED if f not in header]
    if missing:
        raise ValueError(f"checkpoint header missing required field(s): {', '.join(missing)}")
    fmt = header["format"]
    if fmt > CHECKPOINT_FORMAT:
        raise ValueError(
            f"checkpoint header format {fmt} is newer than this code understands "
            f"({CHECKPOINT_FORMAT}); upgrade tt-nanollama3 to read it"
        )


def checkpoint_path(checkpoint_dir: Path, step: int) -> Path:
    """``<dir>/nanollama3_step<step>.pkl``, zero-padded so paths sort by step.

    Without padding, ``step10`` sorts before ``step9`` and "newest checkpoint" becomes wrong.
    """
    return Path(checkpoint_dir) / f"nanollama3_step{int(step):08d}.pkl"


def latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Newest checkpoint in ``checkpoint_dir``, or ``None`` if there are none."""
    paths = sorted(Path(checkpoint_dir).glob("nanollama3_step*.pkl"))
    return paths[-1] if paths else None


def save(path: Path, *, header: Dict[str, Any], model_params, optimizer,
         display_progress: bool = False) -> None:
    """Write a checkpoint. Pass-through to ttml, which handles atomicity and streaming."""
    from ttml.checkpointing import save_checkpoint

    validate_header(header)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(str(path), header=header, model_params=model_params,
                    optimizer=optimizer, display_progress=display_progress)


def load(path: Path, *, model_params=None, optimizer=None,
         display_progress: bool = False) -> Dict[str, Any]:
    """Restore a checkpoint in place and return its validated header."""
    from ttml.checkpointing import load_checkpoint

    header = load_checkpoint(str(path), model_params=model_params, optimizer=optimizer,
                             display_progress=display_progress)
    validate_header(header)
    return header


def peek(path: Path) -> Dict[str, Any]:
    """Read a checkpoint's header without touching its tensors."""
    from ttml.checkpointing import read_header

    header = read_header(str(path))
    validate_header(header)
    return header
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/code/tt-nanollama3 && python -m pytest tests/test_checkpoint.py -v`
Expected: 10 passed

Note the zero-padding test will fail against a naive `f"step{step}"` implementation — that is the point of it.

- [ ] **Step 5: Run the whole suite and commit**

Run: `cd ~/code/tt-nanollama3 && python -m pytest -q` — expected 41 passed (31 prior + 10 new).

```bash
git add train/checkpoint.py tests/test_checkpoint.py
git commit -m "feat(checkpoint): validated header schema over ttml.checkpointing

ttml streams tensors atomically already; what it leaves open is the header,
which is opaque and unvalidated. A checkpoint missing vocab_size cannot be
converted later without guessing. Adds the schema plus zero-padded step paths
so 'newest checkpoint' is not lexicographically wrong at step 10."
```

---

## Task 2: Chunked training with periodic checkpoints and resume

**Files:**
- Modify: `train/run.py`

**Interfaces:**
- Consumes: `train.checkpoint.{build_header, checkpoint_path, latest_checkpoint, save, load, peek}`
- Produces: no new Python API — new CLI flags `--save-every`, `--resume`, `--checkpoint-dir`

**The chunking contract:** with `--steps N --save-every K`, call ttml's `train()` `ceil(N/K)` times with `cfg.steps = K` (the final chunk may be shorter), saving after each. The optimizer object persists across calls so AdamW moments carry over; `train_losses` is per-call and must be accumulated by us.

- [ ] **Step 1: Add the CLI flags and chunked loop**

In `main()`, add arguments:

```python
    p.add_argument("--save-every", type=int, default=0,
                   help="Checkpoint every N steps (0 disables checkpointing).")
    p.add_argument("--checkpoint-dir", default=str(ROOT / "artifacts" / "checkpoints"))
    p.add_argument("--resume", default=None,
                   help="Checkpoint path to resume from, or 'latest' to pick the newest "
                        "in --checkpoint-dir.")
```

Replace the single `train(...)` call inside the device-open `try` with:

```python
        start_step = 0
        if args.resume:
            resume_path = (checkpoint.latest_checkpoint(Path(args.checkpoint_dir))
                           if args.resume == "latest" else Path(args.resume))
            if resume_path is None or not resume_path.is_file():
                raise FileNotFoundError(f"no checkpoint to resume from: {args.resume}")
            header = checkpoint.load(resume_path, model_params=model.parameters(),
                                     optimizer=optimizer)
            start_step = int(header["step"])
            print(f"  resumed from {resume_path} at step {start_step}")

        remaining = cfg.steps
        step = start_step
        all_losses = []
        chunk = args.save_every if args.save_every > 0 else remaining
        while remaining > 0:
            cfg.steps = min(chunk, remaining)
            losses, _ = train(cfg, model, optimizer, train_ids, False, False)
            all_losses.extend(losses)
            remaining -= cfg.steps
            step += cfg.steps
            if args.save_every > 0:
                path = checkpoint.checkpoint_path(Path(args.checkpoint_dir), step)
                checkpoint.save(
                    path,
                    header=checkpoint.build_header(
                        step, model_config_path=str(model_config),
                        tokenizer_dir=str(ROOT / "artifacts" / "tokenizer"),
                        total_tokens=int(len(train_ids) + len(val_ids)),
                    ),
                    model_params=model.parameters(), optimizer=optimizer,
                )
                print(f"  checkpoint saved: {path}")
        train_losses = all_losses
```

- [ ] **Step 2: Convert the bare-`assert` guards**

Plan 2's vocab guards use bare `assert`, which `python -O` strips — a guard against silent corruption that can itself vanish. Replace both with explicit raises:

```python
    if model_vocab_size != VOCAB_SIZE:
        raise ValueError(
            f"model config declares vocab_size={model_vocab_size} but train.config.VOCAB_SIZE "
            f"is {VOCAB_SIZE}; the tokenizer and the model disagree"
        )
    if int(train_ids.max()) >= VOCAB_SIZE:
        raise ValueError(
            f"token id {int(train_ids.max())} exceeds vocab_size {VOCAB_SIZE}; these tokens "
            "were produced by a different tokenizer than the model config expects"
        )
```

- [ ] **Step 3: Verify the dry run still needs no device**

Run: `cd ~/code/tt-nanollama3 && python train/run.py --dry-run --steps 20 --save-every 10`
Expected: prints the resolved config and exits 0 without opening a device.

- [ ] **Step 4: Verify checkpointing and resume on hardware**

Run a short job that saves twice, then resume from it:

```bash
cd ~/code/tt-nanollama3
python train/run.py --steps 4 --save-every 2 --batch-size 8
ls -la artifacts/checkpoints/
python train/run.py --steps 2 --resume latest --batch-size 8
```

Expected: two checkpoint files from the first run; the second run prints `resumed from ... at step 4` and its **first loss is close to the first run's last loss**, not back near 10.4. A resumed run that restarts near 10.4 means the weights did not load, and that is a failure regardless of what the file contains.

Record the actual losses from both runs.

- [ ] **Step 5: Confirm the header round-trips**

```bash
cd ~/code/tt-nanollama3 && python -c "
from pathlib import Path
from train.checkpoint import latest_checkpoint, peek
p = latest_checkpoint(Path('artifacts/checkpoints'))
print(p)
print(peek(p))
"
```
Expected: prints the newest checkpoint path and a header with `format`, `step`, `vocab_size=32000`, `seq_len=256`, `total_tokens=127635889`.

- [ ] **Step 6: Commit**

```bash
git add train/run.py
git commit -m "feat(train): periodic checkpoints and resume via chunked train() calls

ttml's train() has no checkpoint hook -- grep for save/checkpoint in
trainer.py returns nothing, and TrainingConfig.save_every is dead. Rather
than reimplement the loop, call train() in chunks and save between; the
optimizer persists across calls so AdamW moments carry over.

Also converts Plan 2's bare-assert vocab guards to explicit raises: python -O
strips asserts, and these guard against silent on-device corruption."
```

---

## Task 3: The real training run

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:** none — this task produces a trained checkpoint on disk.

Plan 2 proved the mechanism over 20 steps. This produces weights worth converting.

- [ ] **Step 1: Choose the step count from measured throughput**

Plan 2 measured ~0.12–0.14 s/step at `batch_size=64, seq_len=256`. Run:

```bash
cd ~/code/tt-nanollama3 && python train/run.py --steps 3000 --save-every 500 --batch-size 64
```

At the measured rate, 3000 steps is roughly 6–7 minutes of compute plus checkpoint-write time, and it covers `3000 × 64 × 256 ≈ 49.2M` tokens — about 0.43 epochs over the 114.9M-token training split. Report the wall-clock actually observed.

- [ ] **Step 2: Report the curve honestly**

Record: first loss, last loss, the real validation loss, s/step steady state, and the total wall clock. State plainly how many tokens were seen and what fraction of an epoch that is. Do **not** describe the result as a capable model — at ~22M parameters over a fraction of an epoch it is a demonstration, and the model card must say so.

- [ ] **Step 3: Log the run in CLAUDE.md and commit**

Add a section to `CLAUDE.md` recording the numbers above and the checkpoint path, matching the file's existing tone.

```bash
git add CLAUDE.md
git commit -m "docs(claude): log the first real NanoLlama3 training run"
```

---

## Self-Review

**Spec coverage.** Implements the first half of the design spec's Stage 3 (checkpointing) plus the real training run the spec's Stage 2 always implied. HF-format conversion is deliberately **not** here — see Known risks.

**Every capability named above resolves to code or a test.** Applying the rule Plan 2 added after the fact:
- "Header schema construction and validation" → `build_header`/`validate_header`, tests 1–8
- "Path conventions" → `checkpoint_path` only, via `test_checkpoint_path_is_step_numbered` and
  `test_checkpoint_paths_sort_lexicographically_by_step`. **Correction (final whole-branch
  review):** this line originally claimed `latest_checkpoint` too, but neither of those two
  tests exercises it — `latest_checkpoint` had no direct test at all at this point in the
  plan. A later review found and fixed the resulting gap (its docstring said "newest" when
  the code actually picks highest-step, which silently differ once a directory is shared
  across runs) and added `test_latest_checkpoint_returns_none_for_empty_dir` and
  `test_latest_checkpoint_picks_highest_step_not_newest_file` to `tests/test_checkpoint.py`.
- "Chunked training, periodic saves, resume" → Task 2 Steps 1 and 4
- "No bare `assert` guards" → Task 2 Step 2, an explicit conversion of the two existing ones

Nothing in the tables above claims coverage that no task delivers.

**Placeholders.** None.

**Type consistency.** `build_header` returns the dict `validate_header` accepts and `save`/`load`/`peek` pass through. `checkpoint_path` and `latest_checkpoint` both return `Path`. Task 2 calls each with the signature Task 1 defines.

**Verified before writing.** `save_checkpoint`/`load_checkpoint`/`read_header` signatures, their atomicity and one-tensor-at-a-time streaming, the absence of any checkpoint logic in `train()`, and the dead `TrainingConfig.save_every` field were all read from `~/tt-metal` source. The header schema follows the canonical example at `sources/examples/train/checkpointing.py`.

## Known risks

- **Conversion needs a device, and that shapes the next plan.** `load_checkpoint` restores *in place* into live `NamedParameters`, so reading weights requires constructing the model first — which requires hardware. The canonical `load_for_inference` does exactly this. The next plan should therefore split conversion in two: an on-device export dumping plain tensors, then a pure-CPU step assembling HF `config.json` + safetensors. Do not assume conversion can run anywhere.
- **Resume restores weights and optimizer, not the data cursor.** ttml's `get_batch` samples random windows via `np.random.randint` each call, so there is no cursor to restore — but it also means a resumed run does not continue the same data order. The canonical example captures RNG state in its header for this reason; we do not, because random-window sampling makes it cosmetic here. If sequential batching is ever adopted, this becomes real and the header must grow an RNG field.
- **`cfg.steps` is mutated in the chunk loop.** `RunConfig` is ours and nothing else reads it mid-run, so this is safe today, but it means `cfg.steps` no longer reflects the user's requested total after the first chunk. Read `args.steps` for that.
