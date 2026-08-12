<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Multi-chip training notes

**Status: not done here.** Every run in this repo is single-chip. This file records what a
multi-chip run would require, so that work starts from known ground instead of rediscovering
it. Nothing below has been measured *by this repo* — the measurements cited are from the
`ct5`/`ct8` lessons in
[tt-vscode-toolkit](https://github.com/tenstorrent/tt-vscode-toolkit), which verified
four-chip DDP on the same class of hardware. Treat them as well-sourced expectations, not as
our results.

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

**2. Checkpoint saving throws under DDP.** The stock saver fails with
`Can't get a single buffer from host storage distributed over mesh`. Weights are *replicated*
across chips, so the fix is to pull them through
`ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)` and keep the first replica.

This is also the reason to expect **the converter to keep working**: DDP replicates rather
than shards, so once the first replica is extracted the checkpoint contains the same whole
tensors it does today. That is a strong expectation, **not a verified fact** — the parity gate
(`tests/test_numpy_parity.py`) should be run against a DDP checkpoint before any claim is
made, since it is exactly the instrument that would catch a layout change.

**3. Auto-resume is broken.** Any run without `--fresh` triggers auto-resume, which injects an
empty `--resume` and dies in argparse. Use `--fresh` and checkpoint frequently.

## The arithmetic that is easy to get wrong

DDP across 4 chips at `batch_size: 64` means **256 sequences per step**, not 64. Consequences:

- **The step budget moves.** Three epochs over the 114.9M-token training split is ~21,033
  steps at batch 64. At an effective batch of 256 the same token count is **~5,258 steps**.
  Reusing `--steps 21034` under DDP would train roughly *four times* the intended data.
- **The learning rate probably moves too.** `lr: 3e-4` was chosen for batch 64. A 4× larger
  batch conventionally wants a higher LR; leaving it unchanged is a different optimization
  problem, not merely a faster version of the same one.
- **`batch_size` must be divisible by the device count** (`ct3`). 64 / 4 = 16, so this is fine.

Expected speedup, from `ct5`'s measurements: **~1.95× on 2 chips, ~3.98× on 4** — near-linear.
Whether that holds for a model this small is untested; at 0.134 s/step the per-step
gradient-synchronization overhead may claim a meaningful share.

## If we do this

The honest reason is demonstration: a Tenstorrent-first reference model that never uses more
than one chip of a four-chip mesh is leaving the most Tenstorrent-specific thing about the
hardware unshown. It should be its own plan with its own measured baseline — a fresh run at a
corrected step count and LR, compared against the v2 single-chip result on the same held-out
windows — and not a config edit bolted onto an existing run.
