<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Kernel APIs we use that upstream has deprecated

Our custom kernels in `kernels/` are compiled by tt-metal's JIT at run time, not by
our build. That means a removed API is not a build failure here — it is a run-time
compile failure inside a gate, discovered whenever we next happen to run it.

Found while upgrading tt-metal to **v0.77.0** (2026-08-18). **Both are now migrated**
(2026-08-18) and both gates are green with no deprecation warnings left. This file
stays as the record of what was moved and why, and as the place to log the next one.

| API we call | replacement | deadline | our call sites |
|---|---|---|---|
| ~~`sub_tiles_bcast_scalar_init_short`~~ | `sub_bcast_scalar_init` | removed after 2026-09-15 | **migrated** — `core_argmax_compute.cpp:97` |
| ~~`noc_async_read_tile`~~ | `noc_async_read_page<AddrGen>` | none stated | **migrated** — `core_reduce_reader.cpp:39,45`, `core_argmax_reader.cpp:33,35,37` |

The first has a date on it and that date is about a month out. When it passes, the
per-core argmax kernel stops compiling — and it will fail at the point where a gate
runs, with a JIT error, not at build time.

The second carries no date, so it is the lower risk, but it is a signature change
(`<typename AddrGen>`) rather than a rename.

## How the migration went

Both turned out to be drop-in. `sub_bcast_scalar_init(icb0, icb1)` takes the same
two arguments as the name it replaces, and `noc_async_read_page` deduces its
`AddrGen` template parameter from the `TensorAccessor` we already pass, so the call
sites are unchanged apart from the name.

Re-gated on hardware after the change, because "drop-in" is a claim about the
signature and the gate is a claim about the answer:

    per-core argmax   110/110 exact on max AND index, 0 ties   PASS
    Gumbel sampler    PASS across five seeds (see the sweep)

## How to see the warnings again

They are emitted by the JIT during any run that builds these kernels:

```bash
gozer run --chips 1 --who claude:kernels --reason "deprecation check" -- \
  python scripts/probe_core_argmax_device.py 2>&1 | grep -i deprecat
```
