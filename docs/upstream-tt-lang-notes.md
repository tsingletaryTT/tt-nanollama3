<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# tt-lang: what blocked a port, and what actually fixed it

Notes from porting `kernels/core_argmax_compute.cpp` — a kernel already gated at
110/110 exact through `ttnn.generic_op` — to tt-lang. The port was attempted
first against the PyPI wheel and then against tt-lang built from `main`, and the
two attempts give different answers.

Reproduction: `kernels/ttlang/argmax_ttlang.py`, driven by
`scripts/probe_argmax_ttlang.py`.

## The headline: this was a publishing gap, not a language limitation

Against **tt-lang 1.1.6** (the newest wheel on PyPI) the port failed to compile:

```
error: operand #1 does not dominate this use
  --> hit = ttl.eq(f_blk, m)
  note: operand defined here (op in the same block)
  --> m = ttl.math.reduce_max(f_blk, dims=[-1])
```

Against **tt-lang 1.1.9.dev12**, built from `main`, that error does not occur.

The relevant work is `711fcb38b` (*Defer intermediate DFB materialization*),
which added `lib/Dialect/TTL/Transforms/ComputeOpCreationPlanning.h` — a pass
that models this case explicitly:

> *The compute insertion anchor does not dominate these consumers in the original
> IR. Kernel planning may still select the creation when each consumer is an
> earlier fused source that erases the recorded operand use.*

That commit's first release tag is **v1.1.8**. The tags `v1.1.7`, `v1.1.8` and
`v1.1.9` all exist in git; PyPI stops at **1.1.6**, and the project publishes no
GitHub release artifacts. So the fix has been released for three versions and is
installable by nobody.

`ttl.call_extern_func` — the escape hatch to arbitrary tt-metal C++, and the only
route to `rand_tile` for the Tensix PRNG — has the same shape: present in
`v1.1.8`/`v1.1.9` tags, absent from every published wheel. The source build has
it (86 ops against the wheel's 77, plus `Kernel` and `KernelKind`).

## Where the port stands now

A different, narrower error:

```
error: dataflow buffer 1 requires incompatible unpack modes in one kernel
  --> sel = ttl.mul(hit, i_blk)
  note: operand 0 establishes the conflicting unpack mode
  --> m = ttl.math.reduce_max(f_blk, dims=[-1])
```

`hit` descends from the reduce and carries its unpack mode; `i_blk` is a plain
user dataflow buffer. Multiplying them asks one buffer to be unpacked two ways
inside a single kernel. Untried candidates: `CompilerOptions.reuse_user_dfbs` /
`compiler_dfbs` to stop buffer sharing, or staging the index through an
intermediate rather than multiplying straight from its DFB.

The shipping kernel remains the C++ one. This is a portability experiment on work
whose right answer is already known, not a dependency.

## Two model facts worth knowing before porting anything

`reduce_max`/`reduce_sum` reduce **across tiles**, not within one: `dims` indexes
the tile dimensions of a block. `examples/eltwise_broadcast_reduce.py` makes every
tile hold a uniform value specifically so a tile-level reduce matches an
element-level reference. Any kernel laid out as one tile per core — reducing that
core's 1024 elements to a scalar — is a *within*-tile reduction and needs a
different layout here.

The op set was never the obstacle. `sub`, `mul`, `eq` and `reduce_max` are all in
1.1.6, and `eq` removes the need for a `1 - abs(sign(d))` equality trick.

## Building from source, since the wheel is three releases behind

What worked, without disturbing anything pre-existing:

1. `git worktree add --detach <path> 37aca9d38` from `third-party/llvm-project` —
   the pinned commit — so the existing checkout stays where it is. The stale
   `build/llvm-install` was compiled from `e568ab3b0`, which lacks
   `mlir::InnerTileAlignment` and is why an in-place build fails.
2. Build LLVM/MLIR with tt-lang's own flag set (`cmake/modules/BuildLLVM.cmake`:
   MLIR only, host target, assertions on, Python bindings on) to a **new** prefix.
   ~3.6 GB installed.
3. `pip install "nanobind>=2.9,<3.0"` — MLIR's Python bindings require it and it
   was absent. Verify `ttnn` still imports afterwards.
4. Configure tt-lang with `-DMLIR_PREFIX=<new install>` and
   `-DTTLANG_EXTERNAL_TT_METAL_DIR=/home/ttuser/tt-metal`. External mode returns
   before tt-lang's patch step, so the serving tt-metal tree is read-only and no
   second multi-hour tt-metal build is needed.
5. Run against it with `PYTHONPATH=<repo>/build-main/python_packages`.

Note for anyone upgrading the wheel instead: `tt-lang` 1.1.6 hard-pins
`ttnn==0.74.0`, so install it with `--no-deps` or it will replace an editable
ttnn built from a local tt-metal.

## Correction

An earlier version of this file said tt-lang was unusable without an LLVM rebuild,
and a later one said the argmax port revealed a language limitation. Both are
withdrawn. The wheel installs and runs with no build at all; the argmax blocker
was a released-but-unpublished fix. The only claim that survives unchanged is
that the *Gumbel* kernel needs `call_extern_func`, which is likewise released and
unpublished.
