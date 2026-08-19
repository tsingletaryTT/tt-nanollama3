<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Kernel APIs we use that upstream has deprecated

Our custom kernels in `kernels/` are compiled by tt-metal's JIT at run time, not by
our build. That means a removed API is not a build failure here — it is a run-time
compile failure inside a gate, discovered whenever we next happen to run it.

Found while upgrading tt-metal to **v0.77.0** (2026-08-18). Both still compile and
both still gate green; the warnings below are the only notice we get.

| API we call | replacement | deadline | our call sites |
|---|---|---|---|
| `sub_tiles_bcast_scalar_init_short` | `sub_bcast_scalar_init` | **removed after 2026-09-15** | `core_argmax_compute.cpp:97` |
| `noc_async_read_tile` | `noc_async_read_page<AddrGen>` | none stated | `core_reduce_reader.cpp:39,45` |

The first has a date on it and that date is about a month out. When it passes, the
per-core argmax kernel stops compiling — and it will fail at the point where a gate
runs, with a JIT error, not at build time.

The second carries no date, so it is the lower risk, but it is a signature change
(`<typename AddrGen>`) rather than a rename.

## Why this file exists rather than a fix

The migration is small but it is not free: `sub_bcast_scalar_init` is a rename with
a possible argument-order change, and the argmax kernel is gated at 110/110 exact.
Changing it means re-running that gate on hardware. Worth doing deliberately rather
than folding into an upgrade commit.

## How to see the warnings again

They are emitted by the JIT during any run that builds these kernels:

```bash
gozer run --chips 1 --who claude:kernels --reason "deprecation check" -- \
  python scripts/probe_core_argmax_device.py 2>&1 | grep -i deprecat
```
