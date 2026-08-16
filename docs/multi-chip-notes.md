<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Multi-chip training notes

> **STATUS CHANGED 2026-08-16: four-chip DDP works and is measured.** `train/run.py --ddp 4`
> runs the 1024 size on all four chips at **193.4 s/1000 steps against 770.2 single-chip — a
> 3.98x wall-clock win** — with gradients verifiably synchronised (all four replicas
> bit-identical, plus a loss trajectory matching the single-chip run to within a quarter of
> the seed-noise floor). **The measured results and the whole bring-up live in
> [`.superpowers/ddp-bringup.md`](../.superpowers/ddp-bringup.md); read that first.**
>
> This file is kept because its *analysis* is still the best statement of why DDP is the
> right parallelism here and what the arithmetic is. Two things in it are now known to be
> wrong, and are corrected inline below: **catch #1** (setting `TT_MESH_GRAPH_DESC_PATH` is
> necessary but not sufficient — the descriptor's declared *shape* is the load-bearing part,
> and a shape mismatch hangs rather than fails) and the closing section's claim that this
> "should be its own plan" (it was; this is it).

**Original status, retained for context: not done here.** Every run in this repo is
single-chip. This file records what a multi-chip run would require, so that work starts from
known ground instead of rediscovering it. Nothing below has been measured *by this repo* — the
measurements cited are from the `ct5`/`ct8` lessons in
[tt-vscode-toolkit](https://github.com/tenstorrent/tt-vscode-toolkit), which verified
four-chip DDP on the same class of hardware. Treat them as well-sourced expectations, not as
our results.

(The `ct5` expectation of "~3.98x on 4 chips" turned out to be almost exactly right: we
measured **3.98x**.)

## The hardware, stated correctly

The development host is a **TT-QuietBox 2**. `tt-smi -s` reports four devices of
`board_type: p300c`, which reads as four independent boards but is not:

```
dev0  bus 0000:01:00.0  board_id 0000046131924062
dev1  bus 0000:02:00.0  board_id 0000046131924062   <- same board as dev0
dev2  bus 0000:03:00.0  board_id 0000046131924055
dev3  bus 0000:04:00.0  board_id 0000046131924055   <- same board as dev2
```

`board_id` pairs the devices up: **two dual-chip p300 cards, four Blackhole chips total**,
wired as a `P300_X2` 2×2 ring mesh. The chips are mesh-connected, not independent
accelerators. Anything written as "4× p300c boards" is wrong — a mistake this repo's README
made until 2026-08-12.

## What we do today, and where it is set

`train/config.py:123` hardcodes:

```python
"device_config": {"mesh_shape": [1, 1], "enable_ddp": False, "enable_tp": False}
```

**This is the only device config that takes effect.** `train/configs/nanollama3_bpe_v2.yaml`
also contains a `device_config` block, but `apply_optimizer_override` reads *only*
`training_config.optimizer` from that file — editing `mesh_shape` there changes nothing. That
trap is called out in the YAML itself.

Single-chip operation is directly visible in telemetry. During the v2 training run, with
`tt-smi -s` sampled mid-run:

| | working chip | idle chips |
|---|---|---|
| Power | **82 W** | 61–73 W |
| Temperature | **73 °C** | 63–68 °C |
| `aiclk` | 1350 MHz | 1350 MHz |

Idle Blackhole holds its clock rather than dropping it, so idle chips stay warm and the
**power** gap is the more reliable indicator of which chip is working.

## Data parallel vs tensor parallel

**DDP is the applicable one.** It replicates the model on each chip and splits the batch, so
per-chip memory is unchanged — appropriate here, because this model has no capacity problem
(22M parameters, well inside one chip's DRAM). The motivation is speed and demonstrating the
mesh, not fitting the model.

**TP is not, and would likely break conversion.** Tensor parallelism shards weights across
chips; `convert/checkpoint_reader.py` and `convert/hf_mapping.py` assume whole tensors. Do not
enable it without reworking the converter.

## The three known catches

All three come from `ct8`'s troubleshooting section, recorded there from real four-chip runs.

**1. The fabric router times out on 2/4-chip Blackhole.** The run dies at
`Fabric Router Sync: Timeout` unless `TT_MESH_GRAPH_DESC_PATH` is set. This is the
single most likely first failure.

> **CORRECTED 2026-08-16 — right that the variable must be set, wrong about what to set it
> to, and the failure is worse than a timeout.** Setting it to *this box's own* descriptor
> (`p300_x2_mesh_graph_descriptor.textproto`) is **not** sufficient: that file declares
> `device_topology { dims: [2, 2] }`, and a DDP-only run must open `[1, 4]` (a 2-D mesh is a
> hard `TT_FATAL` unless two parallelisms are enabled). With that mismatch the mesh **opens
> successfully**, step 1 trains, and step 2 hangs forever in the first gradient all-reduce —
> no timeout, no error. `device_topology.dims` must equal the `MeshShape` opened. This repo
> now ships matching descriptors under `train/configs/mesh/`, selected automatically by
> `--ddp`. A genuine `Fabric Router Sync: Timeout` does occur, but for the *other* case —
> a descriptor declaring more devices than you open (e.g. `--ddp 2` against the 4-device
> `p300_x2`), which fails cleanly in 10 s. Full evidence in `.superpowers/ddp-bringup.md`.

**2. Checkpoint saving throws under DDP.** The stock saver fails with
`Can't get a single buffer from host storage distributed over mesh`. Weights are *replicated*
across chips, so the fix is to pull them through
`ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)` and keep the first replica.

This is also the reason to expect **the converter to keep working**: DDP replicates rather
than shards, so once the first replica is extracted the checkpoint contains the same whole
tensors it does today. That is a strong expectation, **not a verified fact** — the parity gate
(`tests/test_numpy_parity.py`) should be run against a DDP checkpoint before any claim is
made, since it is exactly the instrument that would catch a layout change.

> **CORRECTED 2026-08-16 — the caution above was right, and the expectation was wrong.** This
> is the one part of DDP that does **not** work. The replicas really are bit-identical (max
> `|replica0 - replica_i| = 0.0`), so the *data* is exactly as this section predicts — but the
> optimizer step re-marks each parameter's topology from `Replicate` to `Shard(0)`, and ttml's
> saver faithfully honours that metadata and writes every replica concatenated on dim 0. A
> `--ddp 4` checkpoint is 1,475,602,288 bytes against 737,824,624 for the same `--ddp 1` run,
> with every tensor `(4, 1, out, in)` instead of `(1, 1, out, in)`.
>
> `train/checkpoint.py:assert_saveable_on_mesh` now **refuses to write** such a checkpoint
> rather than let a plausible-looking but unreadable file reach `convert/`. Filed as
> `docs/upstream-tt-metal-asks.md` entry 3. Produce publishable weights from a `--ddp 1` run;
> that costs no fidelity, because DDP is a wall-clock optimisation over the same trajectory.
> The parity-gate-against-a-DDP-checkpoint check this paragraph asks for therefore remains
> **outstanding** — there is currently no valid DDP checkpoint to run it against.

**3. Auto-resume is broken.** Any run without `--fresh` triggers auto-resume, which injects an
empty `--resume` and dies in argparse. Use `--fresh` and checkpoint frequently.

## The arithmetic that is easy to get wrong

**Correction (2026-08-13): the claim that used to be here was backwards.** `batch_size` under
ttml's DDP is the **total** batch, sharded across devices on dim 0 — not multiplied by the
device count. `ttml/common/trainer.py:30-38` builds `batch_size` host-side samples and shards
them across the mesh with `shard_tensor_to_mesh_mapper(device, 0)`; each of the 4 chips sees
`batch_size / 4` samples, not `batch_size` samples. Gradients are then all-reduced and
explicitly **divided by the axis size** (`core/distributed/distributed.cpp:36`,
`ttnn::multiply(result, 1.0F / scaler)`), i.e. a proper mean — so the optimizer sees the
gradient for a batch of exactly `batch_size`, identical to single-chip. This is corroborated
by tt-train's own shipped config comment in `training_llama8b_tp_ddp_galaxy.yaml`:
`batch_size: 8  # Total batch size across all DP groups`.

DDP across 4 chips at `batch_size: 64` therefore means **64 sequences per step, not 256** —
the opposite of what the previous version of this section claimed (that version was also
internally inconsistent: its own "`batch_size` must be divisible by the device count... 64 /
4 = 16" bullet only makes sense under sharding, and directly contradicts the "256 sequences"
headline above it). Consequences, corrected:

- **The step budget does not move.** Effective batch stays 64 whether the run is single-chip
  or `[1,4]` DDP at `batch_size: 64`. The same `--steps` value trains the same amount of data
  either way.
- **The learning rate does not need to move**, for the same reason — DDP at unchanged
  `batch_size` is not a larger-batch optimization problem, just a faster wall-clock version of
  the same one. (A learning rate change would become necessary only if `batch_size` is *also*
  raised, e.g. to 256, to make full use of 4 chips' worth of compute per step — that is a
  separate, deliberate decision, not a side effect of turning DDP on.)
- **`batch_size` must still be divisible by the device count** (`ct3`) so it shards evenly:
  64 / 4 = 16 per chip.

Expected speedup, from `ct5`'s measurements: **~1.95× on 2 chips, ~3.98× on 4** — near-linear.
Whether that holds for a model this small is untested; at 0.134 s/step the per-step
gradient-synchronization overhead may claim a meaningful share.

## Why DDP is not enabled on this run — a silent-failure trap, not an oversight

> **RESOLVED 2026-08-16.** This section's analysis was correct in every particular, and the
> three coordinated changes it names are exactly the three that were made: `use_ddp` is now
> passed to both `train()` and `evaluate()` from a single `--ddp`-derived value,
> `_init_parallelism_context` initialises the parallelism context after `initialize_device`
> **and verifies it took**, and `evaluate()` shards its batch and composes its loss. The trap
> it warns about was also reproduced deliberately as a negative control: with the context left
> uninitialised, four chips train happily at full speed over a perfectly ordinary descending
> loss curve while the replicas' weights drift apart (max 2.44e-3 after four steps). It is
> exactly as invisible as this section says. See `.superpowers/ddp-bringup.md`.

Turning on `enable_ddp: True` in `train/config.py`'s `device_config` today would change
**nothing** — `ttml.common.utils.initialize_device` only ever reads `mesh_shape`
(`ttml/common/utils.py:108-119`); `enable_ddp` is read by `ttml.common.model_factory` for
vocab padding but never by the device-init path, and `train/run.py` calls `train()` with
`use_ddp` **hardcoded to `False`** (`train/run.py`'s `train_fn(cfg, model, optimizer,
train_ids, False, False)` call). So a config-only edit is inert.

The dangerous part is what happens if someone fixes *only* that — passes `use_ddp=True` to
`train()` without also initialising the parallelism context. `train()`'s gradient
synchronization goes through `core/distributed/distributed.cpp:56-59`:

```cpp
void synchronize_gradients(const serialization::NamedParameters& parameters) {
    if (!autograd::ctx().is_parallelism_context_initialized()) {
        return;                                  // <-- silent early return
    }
```

`is_parallelism_context_initialized()` only becomes true after an explicit call to
`AutoContext::initialize_parallelism_context(DistributedConfig)`
(`autograd/auto_context.cpp:233-238`). Nothing in this repo calls it (grepped: zero hits for
`initialize_parallelism_context` / `DistributedConfig` / `ttml.Mesh`). So the batch would be
correctly sharded across 4 chips, but the gradients would **never be reduced** — each replica
computes its own gradient from its own quarter of the batch and takes its own step. The four
replicas diverge from step 1. The reported loss is the mean over four increasingly different
models, nothing crashes, and the checkpoint silently keeps only replica 0's weights. This is
the same failure class this repo keeps finding: no error, no warning, just a wrong number
that looks fine.

That is why this run stays single-chip, `mesh_shape: [1, 1]`: enabling DDP correctly needs at
least three coordinated changes (pass `use_ddp=True` to both `train()` and `evaluate()`,
initialise the parallelism context after `initialize_device`, and give `evaluate()` a
composer for the multi-device case) — it is not a one-line flip, and doing it partially is
worse than not doing it at all. See `.superpowers/seqlen-ddp-investigation.md` §2 for the
full gap analysis, including why `[1,4]` is the only viable mesh shape and `[2,2]` is a hard
`TT_FATAL`.

## If we do this

The honest reason is demonstration: a Tenstorrent-first reference model that never uses more
than one chip of a four-chip mesh is leaving the most Tenstorrent-specific thing about the
hardware unshown. It should be its own plan with its own measured baseline — a fresh run
compared against the single-chip result on the same held-out windows, at `batch_size`
unchanged (per the correction above, keeping `batch_size` fixed under `[1,4]` DDP needs no
step-count or LR retuning — the effective batch is identical) — and not a config edit bolted
onto an existing run, given the three coordinated code changes §"Why DDP is not enabled"
above lists as still outstanding.
