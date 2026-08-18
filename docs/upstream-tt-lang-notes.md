<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# tt-lang notes from porting a working kernel

Observations from porting `kernels/core_argmax_compute.cpp` — a kernel already
gated at 110/110 exact through `ttnn.generic_op` — to tt-lang. Recorded because
the port failed for a specific, reproducible reason rather than a vague one.

Version: tt-lang **1.1.6** from PyPI. Reproduction:
`kernels/ttlang/argmax_ttlang.py`, driven by `scripts/probe_argmax_ttlang.py`.

## 1. A reduce result cannot be consumed in the same compute block

```
error: operand #1 does not dominate this use
  --> hit = ttl.eq(f_blk, m)
  note: operand defined here (op in the same block)
  --> m = ttl.math.reduce_max(f_blk, dims=[-1])
```

`reduce_max` lowers into a stage whose SSA value is not visible to elementwise
ops that follow it in the same `@ttl.compute` block. Any algorithm shaped as
*reduce, then compare against the reduced value* therefore cannot be written as
one block. Argmax is exactly that shape: `M = max(x)`, `m = (x == M)`,
`s = m * index`, `i = max(s)`.

Splitting into two operations — one emitting the max, a second consuming it —
should work, at the cost of a second dispatch. Untried here.

## 2. `reduce_max`/`reduce_sum` reduce across tiles, not within one

`dims` indexes the tile dimensions of a block. `examples/eltwise_broadcast_reduce.py`
makes every tile hold a uniform value specifically so that a tile-level reduce
matches an element-level reference, and is marked `xfail-compiler` besides.

This matters for any kernel whose natural layout is one tile per core: reducing a
core's 1024 elements to a scalar is a *within*-tile reduction, and the data has to
be laid out differently to express it here. That is a modelling difference rather
than a defect, but it is not obvious from the op signature.

## 3. The op set was not the problem

`sub`, `mul`, `eq` and `reduce_max` are all present in 1.1.6. `eq` in particular
removes the need for a `1 - abs(sign(d))` equality trick. The obstacle was
scheduling, not expressiveness.

## 4. What is genuinely out of reach on PyPI

`ttl.call_extern_func` — the escape hatch to arbitrary tt-metal C++, and the only
route to the Tensix PRNG (`rand_tile`), which tt-lang has no op for — landed in
tags `v1.1.8`/`v1.1.9`. PyPI caps at 1.1.6 and the project publishes no GitHub
release wheels, so the sampler's Gumbel kernel cannot be written in tt-lang from
an installable version.

Building from source instead requires an LLVM rebuild at the pinned commit:
`mlir::InnerTileAlignment` exists in pinned `37aca9d38` but not in the
checked-out `e568ab3b0` that `build/llvm-install` was compiled from.

## Correction to an earlier claim in this repo

An earlier note here said tt-lang was unusable without that rebuild. That was
wrong and is withdrawn: the PyPI wheel installs and runs with no build at all,
bundles its own MLIR, and compiles kernels against the installed ttnn. The
accurate statement is per kernel — the argmax port is blocked by (1) above, and
the Gumbel kernel by (4).

Note for anyone upgrading: `tt-lang` 1.1.6 hard-pins `ttnn==0.74.0`, so install
it with `--no-deps` or it will replace an editable ttnn built from a local
tt-metal.
