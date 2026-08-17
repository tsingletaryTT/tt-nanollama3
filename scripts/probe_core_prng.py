#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Measure whether the Tensix PRNG carries intrinsic per-core identity.

Why this probe exists
---------------------
The plan for a hardware-bound sampler rests on a claim that has never been
measured on this box: that each Tensix core is a distinct source of randomness.
The silicon does have a per-core PRNG — ``init_prng_seed()`` writes a *config
register* (``PRNG_SEED_Seed_Val_ADDR32``) local to one Tensix, and the SFPU draws
from it. But a seeded LFSR in a register is not entropy. Handed the same seed,
two cores may well produce byte-identical streams, in which case "per-core
randomness" is something the *host* manufactures by handing out different seeds,
and a CPU can reproduce every bit of it.

This probe decides that question before anything is built on top of it.

Four conditions, each run twice
-------------------------------
1. ``distinct``   — every core seeded differently (seed = base + core index)
2. ``identical``  — every core seeded with the *same* value
3. ``unseeded``   — ``rand_tile_init()`` never called; the PRNG is read from
                    whatever state the core was left in

Running each condition twice is what separates "different" from "unreproducible".
A stream that differs across runs is not a signature — it is noise, and noise
cannot be logged, measured, or reproduced for a reader.

What the answers mean
---------------------
* ``identical`` cores agree bitwise
      → the PRNG has NO intrinsic core identity. Per-core behaviour must be
        constructed by the host (distinct seeds), and is fully reproducible on
        CPU by anyone who reimplements the LFSR. The hardware is then a
        *reference implementation*, not an unreachable oracle. This is the
        expected outcome, and it is still a usable foundation — but the honest
        claim shrinks accordingly.
* ``identical`` cores differ, reproducibly
      → there IS state bound to the physical core. This is the interesting
        result and would need a follow-up to find what varies (harvesting,
        boot order, per-core config).
* ``unseeded`` differs run-to-run
      → genuinely irreproducible, and therefore useless as a medium even though
        it is the only condition a CPU truly cannot mimic.

Requires hardware. Take a lease first:

    gozer run --chips 1 --who "claude:tt-tnt" --reason "per-core PRNG probe" -- \
        python scripts/probe_core_prng.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import ttnn

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPUTE_KERNEL = REPO_ROOT / "kernels" / "prng_probe_compute.cpp"
WRITER_KERNEL = REPO_ROOT / "kernels" / "tile_writer.cpp"

TILE = 32
FLOAT32_BYTES = 4
TILE_BYTES = FLOAT32_BYTES * TILE * TILE

INTERMED_CB = 0


def select_cores(device, num_cores: int) -> list:
    """Pick the first ``num_cores`` cores in row-major order off the worker grid.

    Row-major order matters later: adjacency in this list is adjacency on the
    NoC, so a difference that tracks the index is a difference that tracks
    physical position.
    """
    grid = device.compute_with_storage_grid_size()
    cores = [
        ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)
    ]
    if num_cores > len(cores):
        raise SystemExit(
            f"asked for {num_cores} cores but the worker grid is "
            f"{grid.x}x{grid.y} = {len(cores)} (harvesting reduces this)"
        )
    return cores[:num_cores]


def run_condition(
    device,
    cores: list,
    seeds: list[int],
    tiles_per_core: int,
    skip_seed: bool,
) -> np.ndarray:
    """Dispatch the probe once and return draws shaped (num_cores, tiles, 32, 32)."""
    num_cores = len(cores)
    total_tiles = num_cores * tiles_per_core

    # One tile per page; page_id is the core's slot, so output attribution is
    # positional and needs no side-channel.
    output = ttnn.empty(
        (total_tiles * TILE, TILE),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # generic_op requires at least one input and one output tensor (it asserts
    # io_tensors.size() >= 2). This probe genuinely has no input -- the draws are
    # generated on-core from the PRNG -- so we hand it a single unused tile to
    # satisfy the contract. No kernel reads it.
    unused_input = ttnn.zeros(
        (TILE, TILE),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

    cb_format = ttnn.CBFormatDescriptor(
        buffer_index=INTERMED_CB,
        data_format=ttnn.float32,
        page_size=TILE_BYTES,
    )
    cb_descriptor = ttnn.CBDescriptor(
        # Double-buffered so the writer can drain tile i while compute fills i+1.
        total_size=2 * TILE_BYTES,
        core_ranges=core_set,
        format_descriptors=[cb_format],
    )

    compute_rt_args = []
    writer_rt_args = []
    for idx, core in enumerate(cores):
        compute_rt_args.append((core, [seeds[idx], tiles_per_core]))
        writer_rt_args.append(
            (core, [output.buffer_address(), idx * tiles_per_core, tiles_per_core])
        )

    compute_config = ttnn.ComputeConfigDescriptor()
    compute_config.math_approx_mode = False
    compute_config.fp32_dest_acc_en = True

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(COMPUTE_KERNEL),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[INTERMED_CB],
        defines=[("SKIP_PRNG_SEED", "1")] if skip_seed else [],
        runtime_args=compute_rt_args,
        config=compute_config,
    )

    writer_compile_args = [INTERMED_CB]
    writer_compile_args.extend(ttnn.TensorAccessorArgs(output).get_compile_time_args())
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(WRITER_KERNEL),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=writer_compile_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    program = ttnn.ProgramDescriptor(
        kernels=[compute_kernel, writer_kernel], semaphores=[], cbs=[cb_descriptor]
    )

    result = ttnn.generic_op([unused_input, output], program)
    draws = ttnn.to_torch(result).float().numpy()
    return draws.reshape(num_cores, tiles_per_core, TILE, TILE)


def summarise(name: str, run_a: np.ndarray, run_b: np.ndarray) -> dict:
    """Reduce two runs of one condition to the facts that decide the design."""
    num_cores = run_a.shape[0]

    reproducible = bool(np.array_equal(run_a, run_b))

    # Compare every core against core 0 within a single run. Bitwise equality is
    # the right test: these are raw draws, so "close" is meaningless — either the
    # same LFSR sequence came out or it did not.
    matches_core0 = [bool(np.array_equal(run_a[0], run_a[i])) for i in range(num_cores)]
    distinct_cores = num_cores - sum(matches_core0)

    flat = run_a.reshape(num_cores, -1)
    in_unit_interval = bool(flat.min() >= 0.0 and flat.max() < 1.0)

    return {
        "condition": name,
        "num_cores": num_cores,
        "reproducible_across_runs": reproducible,
        "cores_distinct_from_core0": distinct_cores,
        "cores_identical_to_core0": int(sum(matches_core0)),
        "draws_in_unit_interval": in_unit_interval,
        "mean": float(flat.mean()),
        "std": float(flat.std()),
    }


def verdict(distinct: dict, identical: dict, unseeded: dict) -> str:
    """State plainly what the numbers license us to claim. No hedging."""
    if not distinct["reproducible_across_runs"]:
        return (
            "NOT REPRODUCIBLE — even with fixed seeds the draws changed between "
            "runs. Nothing can be built on this until that is explained; a "
            "sampler seeded from it could not be logged or replayed."
        )
    if identical["cores_identical_to_core0"] == identical["num_cores"]:
        return (
            "NO INTRINSIC CORE IDENTITY — same seed gives byte-identical streams "
            "on every core. Per-core behaviour is manufactured by the host, and "
            "is reproducible on CPU by anyone who reimplements the LFSR. The "
            "hardware is a reference implementation, not an unreachable oracle."
        )
    if identical["reproducible_across_runs"]:
        return (
            "INTRINSIC CORE IDENTITY, REPRODUCIBLE — cores given the same seed "
            f"diverge ({identical['cores_distinct_from_core0']} of "
            f"{identical['num_cores']} differ from core 0) and do so identically "
            "across runs. Something is bound to the physical core; find out what "
            "before building on it."
        )
    return (
        "INTRINSIC BUT UNSTABLE — cores differ under the same seed, but not the "
        "same way twice. Suggestive, unusable as-is."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=16, help="cores to probe")
    parser.add_argument("--tiles-per-core", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--only",
        choices=["distinct", "identical", "unseeded"],
        help=(
            "Run a single condition. Required to read 'unseeded' honestly: the "
            "PRNG config register survives across dispatches, so running it "
            "after a seeded condition measures that condition's leftover state, "
            "not the state the core booted into."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    # A single-condition run must not overwrite the full three-condition
    # measurement: they answer different questions and are not interchangeable.
    if args.out is None:
        stem = f"core-prng-probe-{args.only}" if args.only else "core-prng-probe"
        args.out = REPO_ROOT / "docs" / "measurements" / f"{stem}.json"

    device = ttnn.open_device(device_id=args.device_id)
    try:
        cores = select_cores(device, args.cores)
        n = len(cores)

        conditions = {
            # Seeds spaced far apart: adjacent seeds in a weak LFSR can produce
            # correlated streams, which would masquerade as "cores are similar".
            "distinct": ([args.seed + i * 7919 for i in range(n)], False),
            # Same seed on every core: the decisive condition.
            "identical": ([args.seed] * n, False),
            "unseeded": ([0] * n, True),
        }

        if args.only:
            conditions = {args.only: conditions[args.only]}

        results = {}
        for name, (seeds, skip) in conditions.items():
            run_a = run_condition(device, cores, seeds, args.tiles_per_core, skip)
            run_b = run_condition(device, cores, seeds, args.tiles_per_core, skip)
            results[name] = summarise(name, run_a, run_b)
            print(json.dumps(results[name], indent=2))

        if args.only:
            # A single condition cannot support the cross-condition verdict; say
            # so rather than inventing one from partial evidence.
            final = f"SINGLE CONDITION ({args.only}) — no cross-condition verdict."
        else:
            final = verdict(
                results["distinct"], results["identical"], results["unseeded"]
            )
        print("\nVERDICT:", final)

        payload = {
            "cores_probed": n,
            "tiles_per_core": args.tiles_per_core,
            "base_seed": args.seed,
            "core_coords": [[c.x, c.y] for c in cores],
            "conditions": results,
            "verdict": final,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
