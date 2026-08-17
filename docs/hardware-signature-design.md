<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Hardware signature: making TT the reference implementation

## The question this answers

Not *"can we make TT match CPU?"* — that is the decode-defect problem and it is
still open (`docs/upstream-tt-metal-asks.md`, and the nine refuted hypotheses in
the session logs). This is the inverse question:

> How do we make it so that running on CPU you can't possibly get "the rightness"
> you would on TT?

The reframe matters because of who holds authority. Today CPU is the reference
and TT deviates, so every measurement we own — `scripts/free_running_check.py`
most of all — scores TT by its agreement with CPU. Under that framing TT can
only ever be *wrong*. If instead the generative process is **defined in terms of
facts the silicon has**, TT becomes the reference and CPU becomes the simulation.

## What the silicon actually provides

Verified on this box, against tt-metal at `~/tt-metal`:

| Fact | Where it lives | Reachable how |
|---|---|---|
| Per-core PRNG | `PRNG_SEED_Seed_Val_ADDR32`, a Tensix **config register**; SFPU draws via `TTI_SFPMOV(0, 9, …)` | `rand_tile_init(seed)` / `rand_tile()` in `api/compute/eltwise_unary/rand.h` |
| Core identity | Kernel knows its own `(x, y)` | per-`CoreCoord` runtime args; `ttl.node(dims=2)` in tt-lang |
| Core-local L1 | Per-Tensix SRAM | circular buffers pinned to a `CoreRangeSet` |
| Register-coherent token bags | measured, not assumed | `scripts/probe_grid_layout.py` — a core's bag is 86% as coherent as a cosine neighbourhood, and past ~200 cells it beats one |

## The correction that shapes everything downstream

An earlier framing of this work leaned on per-core stochastic-rounding noise as
the ingredient a CPU "could not possibly" reach. **That was overstated.** What
the hardware has is a *seeded LFSR sitting in a config register* — not entropy.
Hand a CPU the seed and the algorithm and it reproduces the sequence exactly.
The hardware gives per-core **distinctness**, not per-core **unreproducibility**.

This is the better outcome, and not as a consolation prize. Everything this
project measures by — the digest-pinned prompt sets, the seed-noise floor, the
`episod-log.md` discipline of asking one question over and over — depends on
being able to run the same thing twice. A process seeded from genuine entropy
would be unmeasurable and unloggable: we could never say a change moved
anything, because nothing would ever repeat.

So the achievable claim is not *"impossible to fake"*. It is:

> **The hardware is the reference implementation. Everything else is a
> simulation of it.**

That survives contact with our own measurement discipline. The stronger claim
does not.

## What is still unmeasured

The design rests on an assumption nobody has tested here: that a Tensix core is
a *distinct source*. If two cores handed the same seed emit byte-identical
streams, then "per-core randomness" is something the host manufactures by
handing out different seeds — real and usable, but carrying no signature of the
silicon at all.

`scripts/probe_core_prng.py` settles it. Three conditions, each run twice:

* **distinct** — every core seeded differently
* **identical** — every core seeded the *same* (the decisive one)
* **unseeded** — `rand_tile_init()` never called

Running each twice is what separates *different* from *unreproducible*. A stream
that changes between runs is not a signature; it is noise, and noise cannot be
logged or replayed for a reader.

Outcomes and what each licenses:

| Result | What we may claim |
|---|---|
| `identical` cores agree bitwise | No intrinsic core identity. Per-core behaviour is host-manufactured and CPU-reproducible. Hardware is a reference implementation. **Expected.** |
| `identical` cores differ, reproducibly | Something is bound to the physical core. Find out what before building on it. |
| `unseeded` differs run-to-run | Genuinely irreproducible — and therefore useless as a medium, even though it is the one thing CPU truly cannot mimic. |

## Why not tt-lang

`ttl.call_extern_func` was the obvious route and is a dead end on this box:

* tt-lang main needs an LLVM rebuild at its pinned commit — `mlir::InnerTileAlignment`
  exists in pinned `37aca9d38` but not in the checked-out `e568ab3b0` that
  `build/llvm-install` was compiled from.
* PyPI `tt-lang` caps at **1.1.6**, which does not contain `call_extern_func`
  (it landed in tags `v1.1.8`/`v1.1.9`; tt-lang publishes no release wheels).
* `tt-lang` 1.1.6 hard-pins `ttnn==0.74.0`, which would fight the editable ttnn
  our vLLM serving path depends on.

`ttnn.generic_op` + `ttnn.KernelDescriptor` reaches the same hardware with the
stack already installed: inline-or-file kernel source, explicit `core_ranges`,
`runtime_args` keyed per `CoreCoord`, and `compiler_include_paths`. It needs no
build, and it respects the standing constraint that **we do not edit tt-metal** —
the kernel source lives in this repo and is compiled against tt-metal, not
inside it.

## What this is not

This does **not** fix the decode defect, and must not be used to excuse it. The
measured on-device output is degenerate, not merely different: malformed
non-words (*Invisers*, *o'Splains*, *megathering*), local-repeat 0.161 against
CPU's 0.104, termination 2/15 against 5/15. Until the decode path is bisected
with `~/tt-metal/models/common/validation_tools.py`, that remains damage. Its
one genuine asset is that it is *deterministic* — the minimum bar for being a
medium rather than noise.

A hardware-bound sampler and a bisected decode path are independent work. This
document covers only the first.
