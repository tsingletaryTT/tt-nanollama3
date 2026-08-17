#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Gate: on-device Gumbel-max sampling draws from the right distribution.

What changes at this stage
--------------------------
Scoring (``probe_core_scores_device.py``) is deterministic arithmetic, so it can
be gated against NumPy bit-for-bit, and is: 110/110 exact.

Sampling cannot be. Each core draws from the Tensix PRNG -- a hardware LFSR whose
sequence we cannot reproduce in NumPy -- so there is no bit-parity oracle for this
stage and pretending otherwise would mean inventing one. What *can* be gated is
the thing that actually matters:

  1. **Distribution.** Does the device draw cores with the probabilities the
     model implies? Gumbel-max is exact, not approximate, so the target is known
     in closed form.
  2. **Determinism.** Same seeds, same draw. This is the bar that separates a
     medium from noise, and the whole logging discipline depends on it.

This is the point where "the hardware is the reference implementation" stops
being a slogan: from here on the device defines the sample, and the CPU can only
check that the sample is *correctly distributed*, not that it is the same one.

The target distribution
-----------------------
Under Gumbel-max, if every core perturbs its own tokens and reports its max, the
probability that core ``c`` wins is exactly

    P(c) = softmax_c( logsumexp( logits_c / T ) )

which is the correct hierarchical marginal -- so agreement here also confirms
that the per-core decomposition composes the way the theory says.

The bar
-------
Total-variation distance against a bootstrap floor: draw the same number of
samples from the true distribution in NumPy, many times, and see how far those
land from the truth by sampling noise alone. The device passes if its TV distance
sits inside that null band. A fixed threshold would be arbitrary; this one is
calibrated to the number of draws actually taken.

Requires hardware:

    gozer run --chips 1 --who "claude:tt-tnt" --reason "device sampling gate" -- \
        python scripts/probe_core_sampling_device.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import ttnn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.probe_core_scores_device import (  # noqa: E402
    CB_LOGITS,
    CB_SCALER,
    TILE,
    TILE_BYTES,
    pack_by_core,
)
from scripts.sample_topological import TokenCoreMap  # noqa: E402

GUMBEL_KERNEL = ROOT / "kernels" / "core_gumbel_compute.cpp"
READER_KERNEL = ROOT / "kernels" / "core_reduce_reader.cpp"
WRITER_KERNEL = ROOT / "kernels" / "tile_writer.cpp"

CB_PERTURBED, CB_OUT = 2, 16

#: Same spacing as the PRNG probe: adjacent seeds in a weak LFSR can produce
#: correlated streams, which here would couple neighbouring cores' draws.
SEED_STRIDE = 7919


def build_program(logits_tensor, scaler_tensor, output, cores, seeds):
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

    def cb(index):
        return ttnn.CBDescriptor(
            total_size=2 * TILE_BYTES,
            core_ranges=core_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=index, data_format=ttnn.float32, page_size=TILE_BYTES
                )
            ],
        )

    reader_compile = [CB_LOGITS, CB_SCALER]
    reader_compile.extend(ttnn.TensorAccessorArgs(logits_tensor).get_compile_time_args())
    reader_compile.extend(ttnn.TensorAccessorArgs(scaler_tensor).get_compile_time_args())
    writer_compile = [CB_OUT]
    writer_compile.extend(ttnn.TensorAccessorArgs(output).get_compile_time_args())

    reader_rt, writer_rt, compute_rt = [], [], []
    for idx, core in enumerate(cores):
        reader_rt.append(
            (core, [logits_tensor.buffer_address(), scaler_tensor.buffer_address(), idx])
        )
        writer_rt.append((core, [output.buffer_address(), idx, 1]))
        compute_rt.append((core, [int(seeds[idx])]))

    compute_config = ttnn.ComputeConfigDescriptor()
    compute_config.math_approx_mode = False
    compute_config.fp32_dest_acc_en = True

    return ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=str(READER_KERNEL),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_set, compile_time_args=reader_compile,
                runtime_args=reader_rt, config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(GUMBEL_KERNEL),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_set,
                compile_time_args=[CB_LOGITS, CB_SCALER, CB_PERTURBED, CB_OUT],
                runtime_args=compute_rt, config=compute_config,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(WRITER_KERNEL),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_set, compile_time_args=writer_compile,
                runtime_args=writer_rt, config=ttnn.WriterConfigDescriptor(),
            ),
        ],
        semaphores=[],
        cbs=[cb(CB_LOGITS), cb(CB_SCALER), cb(CB_PERTURBED), cb(CB_OUT)],
    )


def target_distribution(packed: np.ndarray, membership, scaled: np.ndarray) -> np.ndarray:
    """P(core wins) under Gumbel-max = softmax over cores of their log-sum-exp."""
    lse = np.array(
        [np.logaddexp.reduce(scaled[m].astype(np.float64)) for m in membership]
    )
    lse -= lse.max()
    weights = np.exp(lse)
    return weights / weights.sum()


def bootstrap_floor(probabilities: np.ndarray, n_draws: int, trials: int, seed: int):
    """TV distance from truth achievable by sampling noise alone."""
    rng = np.random.default_rng(seed)
    distances = []
    for _ in range(trials):
        counts = rng.multinomial(n_draws, probabilities)
        distances.append(0.5 * np.abs(counts / n_draws - probabilities).sum())
    return np.array(distances)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=ROOT / "artifacts" / "token_core_map.npz")
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "docs" / "measurements" / "core-sampling-device-gate.json",
    )
    args = parser.parse_args()

    layout = TokenCoreMap.load(args.map)
    rng = np.random.default_rng(args.seed)
    import torch

    logits = rng.normal(0.0, 4.0, size=layout.token_cell.shape[0]).astype(np.float32)
    logits = torch.from_numpy(logits).bfloat16().float().numpy()
    # Temperature is folded in here so the kernel does no scaling -- one less
    # place for host and device to disagree.
    scaled = (logits / args.temperature).astype(np.float32)

    packed, membership = pack_by_core(scaled, layout)
    probabilities = target_distribution(packed, membership, scaled)

    device = ttnn.open_device(device_id=args.device_id)
    try:
        grid = device.compute_with_storage_grid_size()
        cores = [ttnn.CoreCoord(c % grid.x, c // grid.x) for c in range(layout.n_cells)]

        logits_tensor = ttnn.from_torch(
            torch.from_numpy(packed.reshape(layout.n_cells * TILE, TILE)),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        scaler_tensor = ttnn.full(
            (TILE, TILE), 1.0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        output = ttnn.empty(
            (layout.n_cells * TILE, TILE), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        def one_draw(draw_index: int) -> int:
            seeds = [
                args.seed + draw_index * 1_000_003 + c * SEED_STRIDE
                for c in range(layout.n_cells)
            ]
            program = build_program(logits_tensor, scaler_tensor, output, cores, seeds)
            result = ttnn.generic_op([logits_tensor, scaler_tensor, output], program)
            tiles = ttnn.to_torch(result).float().numpy().reshape(layout.n_cells, TILE, TILE)
            return int(np.argmax(tiles[:, 0, 0]))

        print(f"drawing {args.draws} samples across {layout.n_cells} cores…")
        winners = np.array([one_draw(i) for i in range(args.draws)])

        # Determinism: the same seeds must give the same winner.
        repeat_first = one_draw(0)
    finally:
        ttnn.close_device(device)

    counts = np.bincount(winners, minlength=layout.n_cells)
    empirical = counts / args.draws
    tv = 0.5 * np.abs(empirical - probabilities).sum()

    null = bootstrap_floor(probabilities, args.draws, 400, args.seed)
    null_p95 = float(np.percentile(null, 95))
    deterministic = bool(repeat_first == winners[0])

    print(f"\ndistinct cores drawn   {len(np.unique(winners))}/{layout.n_cells}")
    print(f"TV distance to target  {tv:.4f}")
    print(f"sampling-noise floor   mean {null.mean():.4f}, p95 {null_p95:.4f}")
    print(f"deterministic replay   {deterministic}")

    passed = bool(tv <= null_p95 and deterministic)
    if passed:
        verdict = "PASS — draws are correctly distributed and replay exactly"
    elif not deterministic:
        verdict = "FAIL — same seeds gave a different winner; not a usable medium"
    else:
        verdict = f"FAIL — TV {tv:.4f} exceeds the sampling-noise p95 {null_p95:.4f}"
    print(f"\nVERDICT: {verdict}")

    payload = {
        "cores": int(layout.n_cells),
        "draws": args.draws,
        "temperature": args.temperature,
        "tv_distance": float(tv),
        "noise_floor_mean": float(null.mean()),
        "noise_floor_p95": null_p95,
        "distinct_cores_drawn": int(len(np.unique(winners))),
        "deterministic": deterministic,
        "pass": passed,
        "verdict": verdict,
        "layout_digest": layout.provenance["embedding_sha256"],
        "note": (
            "No bit-parity oracle exists for this stage: the device draws from the "
            "Tensix PRNG, which NumPy cannot reproduce. Gated on distribution and "
            "determinism instead."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
