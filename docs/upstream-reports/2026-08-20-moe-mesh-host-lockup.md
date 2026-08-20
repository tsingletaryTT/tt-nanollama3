# Host hard-lockup after ttml sparse-MoE opens a 4-chip P300 mesh

**Status: NOT FILED.** Drafted for Taylor to post (or discard). An earlier version was
filed as tt-metal#53842 in error and retracted; that issue is closed with its body
replaced, though GitHub's edit history retains the original text.

## Summary

A `ttml` sparse-MoE arm (`SparseMoEEP`, tt-train Python `train()`) **hard-froze the host**
about 20 seconds after opening a 4-chip mesh on 2x p300c. Not a Python exception and not a
device error — the machine stopped. The journal ends mid-stream: no OOM record, no kernel
panic, nothing in `/sys/fs/pstore` or `/var/crash`, no shutdown sequence. Unusable for
~40 minutes until a hard reset.

No minimal repro yet. The reason it still seems worth reporting: **a dense arm ran 3000
steps to a clean exit on the identical mesh minutes earlier**, and a MoE arm on *one*
board is clean.

## Environment

- tt-metal **v0.77.0** (`ttnn` v0.77.0), source build
- 2x **p300c** (Blackhole), 4 chips, `ClusterType::P300_X2`
- Host 249 GB RAM, kernel 7.0.0-28-generic
- `ttml` Python `train()`, `--model-impl python`, `SparseMoEEP` in `LlamaBlock.mlp`

## Timeline (pre-crash journal)

| time | event |
|---|---|
| 12:08 | earlier attempt died: `Timed out while waiting for active ethernet core 29-25 to become active again. Try resetting the board.` (`llrt.cpp:594`, via `RiscFirmwareInitializer::assert_active_ethernet_cores_to_reset`) |
| ~12:09 | board reset; 20-step dense smoke passes |
| 12:11:35 | **dense** arm starts, `--ddp 4` |
| 12:22:17 | dense arm exits **0**, 3000 steps |
| 12:22:17 | first **MoE** arm starts, same mesh/seed/corpus/optimizer |
| **12:22:37** | **last journal line. Host wedged.** |

Telemetry just before: 75-81 degC, power peaking ~249 W. Unremarkable.

## What differs between the two arms

Identical `--size 1024 --seed 5489 --ddp 4 --seq-len 512`, same corpus, same optimizer
(AdamW, beta2 0.999, `stochastic_rounding: true`), same warm-start checkpoint. The only
difference is `--moe --gate-policy learned` — `SparseMoEEP` replacing `LlamaMLP` in the
later blocks (top-2 of 10 experts + 1 shared, expert width 928). So the trigger tracks
MoE mesh/collective init, not load, memory, or thermals.

## Possibly relevant

The prior ethernet-core timeout suggests the eth/fabric layer was already fragile, and MoE
exercises collectives the dense path does not. A *host*-visible lockup rather than a device
hang the driver can surface may point at that path being able to wedge the PCIe/driver
interface.

`ttml` ships mesh-graph descriptors for 8 and 32 chips only, so this run used a vendored
`[1, 4]` descriptor. A mismatch there is known to hang (not error) in the first gradient
all-reduce — a plausible neighbour, though that would normally wedge the process, not the
host.

## Current status

MoE at `--ddp 2` on a single board is clean (200 steps, loss 3.578 -> 3.297), so the
experiment continues there and avoids the two-board mesh. Instrumented attempts on the
4-chip config are possible on request — the hardware and exact setup are available.
