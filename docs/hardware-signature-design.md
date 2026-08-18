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

## What the per-core PRNG is, and is not

The Tensix PRNG is a seeded LFSR held in a config register, not an entropy
source. Given the seed and the algorithm, a CPU reproduces the sequence exactly.
The hardware provides per-core *distinctness*, not per-core *unreproducibility*.

That suits this project. Everything here is measured by comparison — digest-pinned
prompt sets, seed-noise floors, the same question asked of every checkpoint — and
all of it requires running the same thing twice. A process seeded from genuine
entropy could not be measured or logged, because nothing would repeat.

So the claim this design supports is:

> The hardware is the reference implementation. Everything else is a simulation
> of it.

## The trilemma

Pushing on "CPU can't possibly get it" leads somewhere uncomfortable that is
better stated up front than discovered late:

Anything deterministic can be simulated. If a process replays identically on
TT, then it is an algorithm plus a state, and a sufficiently determined person
can reimplement both on a CPU. Reproducibility *is* simulability. So "impossible
to reach on CPU" and "reproducible" cannot both hold. Pick one:

1. **Unfakeable** — seed from something genuinely physical (unseeded PRNG state,
   NoC arrival order under contention). No CPU can reproduce it. Neither can the
   next TT run, which makes it unmeasurable and unloggable. It fails this
   project's own standards.
2. **Reproducible** — seeded, deterministic, replayable, loggable in
   `episod-log.md`. A CPU can match it bit-for-bit given the algorithm and the
   layout. TT is then the *reference implementation*, not an oracle.
3. **Expensive to fake** — reproducible in principle, but a CPU reproducing it
   must simulate 110 core-local memories and their PRNGs per token. Feasible,
   and roughly two orders of magnitude slower. TT is necessary *in practice*
   rather than in principle.

The recommendation is **2 with 3's character**: a deterministic process whose
*definition* is a hardware measurement. The token-to-core mapping would come
from the QAP layout onto the real harvested 11×10 grid — and that layout is not
an algorithm anyone can derive, it is a measurement of this silicon
(`scripts/probe_grid_layout.py`). A CPU can replay it only by importing our
measured layout, which is precisely what "the hardware is the reference
implementation" means in practice.

Option 1 stays available as a deliberate, labelled mode — an *unrepeatable
edition*, valuable exactly because it cannot be reproduced, and never used for
anything we intend to measure.

## What is still unmeasured

The design rests on an assumption nobody has tested here: that a Tensix core is
a *distinct source*. If two cores handed the same seed emit byte-identical
streams, then "per-core randomness" is something the host manufactures by
handing out different seeds — real and usable, but carrying no signature of the
silicon at all.

`scripts/probe_core_prng.py` settles it. Three conditions, each run twice:

* `distinct` — every core seeded differently
* `identical` — every core seeded the same
* `unseeded` — `rand_tile_init()` never called

Running each twice is what separates *different* from *unreproducible*. A stream
that changes between runs is not a signature; it is noise, and noise cannot be
logged or replayed for a reader.

Outcomes and what each licenses:

| Result | What we may claim |
|---|---|
| `identical` cores agree bitwise | No intrinsic core identity. Per-core behaviour is host-manufactured and CPU-reproducible. Hardware is a reference implementation. **Expected.** |
| `identical` cores differ, reproducibly | Something is bound to the physical core. Find out what before building on it. |
| `unseeded` differs run-to-run | Genuinely irreproducible — and therefore useless as a medium, even though it is the one thing CPU truly cannot mimic. |

## Measured result (2026-08-17, p300c Blackhole, 16 cores)

`docs/measurements/core-prng-probe.json` and `…-unseeded.json`.

| Condition | Cores differing from core 0 | Replays across runs |
|---|---|---|
| `distinct` (seed = base + i·7919) | 15 / 16 | **yes** |
| `identical` (same seed everywhere) | **0 / 16** | yes |
| `unseeded`, fresh process | 7 / 16 | no |

The middle row is the one that matters. Sixteen cores handed the same seed
produce byte-identical streams, so the seeded PRNG carries no intrinsic core
identity: the seed fully determines the sequence and nothing about the physical
Tensix enters it. Per-core behaviour is assigned by the host and is reproducible
on CPU by anyone who reimplements the LFSR.

`distinct` confirms the mechanism replays exactly, which is what makes it usable.

Under `unseeded`, the cores are not uniform at rest — 7 of 16
differ from core 0. So some per-core state does exist in that register before
anyone writes it. Two limits on what that licenses, both important:

* It does not replay. The register advances as draws are taken, so a second
  dispatch never sees what the first saw.
* This probe **cannot** test boot-state reproducibility at all. Doing that needs
  a device reset between observations, which these runs did not do. What is
  measured is "state at first dispatch after `open_device`", not "state at boot".

So the unseeded register is exactly trilemma option 1: the one thing a CPU
genuinely cannot mimic, and unusable for anything we intend to measure.

Consequence for the design: proceed on option 2 with option 3's character,
as planned. Structure comes from the measured grid layout; selection comes from
host-assigned per-core seeds. The claim is "the hardware is the reference
implementation", and that claim is now backed by measurement rather than hope.

## On-device per-core scoring (2026-08-17)

`docs/measurements/core-scores-device-gate.json` — **110/110 cores exact, zero delta.**

The layout is now physically real. The vocabulary is permuted so each core's ~291
tokens are contiguous, padded to one 32x32 tile, and handed to one Tensix. Each
core reduces its own bag out of its own L1 and writes one number. No core reads
another core's memory, and the token→core assignment is the measured layout.

Reduction is MAX rather than log-sum-exp: a stable log-sum-exp needs max,
broadcast-subtract, exp, sum and log — four more stages and a second pass — while
MAX is one reduce and is a legitimate scoring rule on its own. The log-sum-exp
version is the follow-up.

Requirements this path depends on, each of which fails quietly if missed:

1. Compute kernels define `kernel_main`. The older
   `namespace NAMESPACE { void MAIN }` form does not link against this tt-metal.
2. `compute_kernel_hw_startup()` must be called before the reduce. It programs
   the unpacker tile descriptors and math ALU format registers; without it the
   reduce packs zeros and, at full grid width, the program stalls.
3. The gate's reference must be at least as precise as the device. Tensix source
   registers narrow fp32 entering the math pipeline, so with arbitrary fp32
   inputs the comparison measures float narrowing rather than the reduction.
   The gate feeds bf16-exact inputs, and reports agreement under both
   round-to-nearest and truncation so the ambiguity cannot return unnoticed.

## On-device sampling (2026-08-17)

`docs/measurements/core-sampling-device-gate.json` — **PASS**: TV distance
0.1064 against a sampling-noise floor of mean 0.1284 / p95 0.1512 over 400
draws, 54 distinct cores, deterministic replay.

Sampling is done by **Gumbel-max**, which turns a categorical draw into the max
reduce already proven exact:

    argmax_i ( logit_i / T + g_i ),   g_i = -log(-log u_i)

is distributed exactly as a draw from `softmax(logits / T)` — no normalisation,
no cross-core sum. And it **composes hierarchically**: if every core perturbs its
own tokens and reports its own max, the argmax across cores is a draw from the
softmax over the entire vocabulary, provably. The sampler therefore decomposes
into the shape of the hardware — 110 cores, each answering about its
own region, from its own L1, using its own random stream, needing to know nothing
about any other core.

The oracle changes here, and that is the point. Scoring is deterministic
arithmetic and is gated bit-for-bit against NumPy. Sampling cannot be: the device
draws from the Tensix PRNG, a hardware LFSR NumPy cannot reproduce. There is no
bit-parity oracle for this stage, and inventing one would mean reimplementing the
silicon. What is gated instead is distribution (against a bootstrap noise floor,
not an arbitrary threshold) and determinism. From this stage on, the device
*defines* the sample and the CPU can only confirm it is correctly distributed.

The SFPU `log` and `negative` ops each require their init call immediately
before use, and interleaved ops must re-init at every switch. Omitting them
produces a kernel that runs, replays deterministically and yields plausible
spread while sampling from the wrong distribution, which is why this stage is
gated against a noise floor rather than by inspection.

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

## What this is and is not

This is a sampling design. It does not change the model's weights and does not
account for the decode defect, which was a separate problem in the vLLM layer and
is resolved: on-device generation now produces coherent text at a local-repeat
rate of 0.031 against a CPU reference of 0.000
(`docs/measurements/decode-defect-resolved.json`).

A hardware-bound sampler and a correct decode path are independent work. This
document covers only the first.

