#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Gate: per-core argmax WITH INDEX on 110 Tensix, against NumPy.

The value half of this was already proven exact. What is new is the index, and an
index is unforgiving: it is either the right slot or it is not, so this gates on
exact equality for both numbers on every core.

Inputs are made bf16-exact before they go near the device, for the reason the
scoring gate documents: Tensix source registers narrow fp32 entering the math
pipeline, so with arbitrary fp32 inputs the comparison measures float narrowing
rather than the reduction. Here it matters twice over -- a value that shifts by
one ULP can move which slot compares equal to the maximum, turning a numerics
question into a wrong index.

Ties: NumPy's argmax returns the FIRST maximal index, this kernel returns the
LAST. Random fp32 data has no ties, but the check reports any it finds rather
than letting them read as failures.

    gozer run --chips 1 --who "claude:tt-tnt" --reason "argmax gate" -- \
        python scripts/probe_core_argmax_device.py
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

from scripts.probe_core_scores_device import TILE, TILE_BYTES, TILE_ELEMS  # noqa: E402

KERNEL = ROOT / "kernels" / "core_argmax_compute.cpp"
READER = ROOT / "kernels" / "core_argmax_reader.cpp"
WRITER = ROOT / "kernels" / "tile_writer.cpp"

CB_IN, CB_SCALER, CB_INDEX = 0, 1, 2
CB_OUT, CB_MAX, CB_SCRATCH = 16, 17, 3


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cores", type=int, default=110)
    p.add_argument("--seed", type=int, default=20260818)
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=ROOT / "docs" / "measurements" / "core-argmax-device-gate.json")
    args = p.parse_args()

    import torch

    rng = np.random.default_rng(args.seed)
    n = args.cores
    field = rng.normal(0.0, 4.0, size=(n, TILE_ELEMS)).astype(np.float32)
    field = torch.from_numpy(field).bfloat16().float().numpy()

    expected_max = field.max(axis=1)
    expected_idx = field.argmax(axis=1)
    ties = int(sum((field[i] == expected_max[i]).sum() > 1 for i in range(n)))

    index_tile = np.arange(TILE_ELEMS, dtype=np.float32)

    device = ttnn.open_device(device_id=args.device_id)
    try:
        grid = device.compute_with_storage_grid_size()
        cores = [ttnn.CoreCoord(c % grid.x, c // grid.x) for c in range(n)]
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

        def dram(arr, shape):
            return ttnn.from_torch(torch.from_numpy(arr.reshape(shape)),
                                   dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                   device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        t_in = dram(field, (n * TILE, TILE))
        t_index = dram(index_tile, (TILE, TILE))
        t_scaler = ttnn.full((TILE, TILE), 1.0, dtype=ttnn.float32,
                             layout=ttnn.TILE_LAYOUT, device=device,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Two tiles out per core: the max, then its index.
        t_out = ttnn.empty((2 * n * TILE, TILE), dtype=ttnn.float32,
                           layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)

        def cb(i, tiles=2):
            return ttnn.CBDescriptor(
                total_size=tiles * TILE_BYTES, core_ranges=core_set,
                format_descriptors=[ttnn.CBFormatDescriptor(
                    buffer_index=i, data_format=ttnn.float32, page_size=TILE_BYTES)])

        reader_ct = [CB_IN, CB_SCALER, CB_INDEX]
        for t in (t_in, t_scaler, t_index):
            reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        # One writer, two tiles per core: max at page 2i, index at page 2i+1.
        writer_ct = [CB_OUT]
        writer_ct.extend(ttnn.TensorAccessorArgs(t_out).get_compile_time_args())

        reader_rt, w1_rt = [], []
        for i, c in enumerate(cores):
            reader_rt.append((c, [t_in.buffer_address(), t_scaler.buffer_address(),
                                  t_index.buffer_address(), i]))
            w1_rt.append((c, [t_out.buffer_address(), 2 * i, 2]))

        cfg = ttnn.ComputeConfigDescriptor()
        cfg.math_approx_mode = False
        cfg.fp32_dest_acc_en = True

        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(kernel_source=str(READER),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core_set, compile_time_args=reader_ct,
                    runtime_args=reader_rt, config=ttnn.ReaderConfigDescriptor()),
                ttnn.KernelDescriptor(kernel_source=str(KERNEL),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core_set,
                    compile_time_args=[CB_IN, CB_SCALER, CB_INDEX, CB_OUT, CB_MAX, CB_SCRATCH],
                    runtime_args=[], config=cfg),
                ttnn.KernelDescriptor(kernel_source=str(WRITER),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core_set, compile_time_args=writer_ct,
                    runtime_args=w1_rt, config=ttnn.WriterConfigDescriptor()),
            ],
            semaphores=[],
            cbs=[cb(CB_IN), cb(CB_SCALER), cb(CB_INDEX), cb(CB_OUT, tiles=3),
                 cb(CB_MAX), cb(CB_SCRATCH)])

        res = ttnn.generic_op([t_in, t_scaler, t_index, t_out], program)
        tiles = ttnn.to_torch(res).float().numpy().reshape(2 * n, TILE, TILE)
    finally:
        ttnn.close_device(device)

    got_max = tiles[0::2, 0, 0]
    got_idx = tiles[1::2, 0, 0].astype(np.int64)

    max_ok = int((got_max == expected_max).sum())
    idx_ok = int((got_idx == expected_idx).sum())
    passed = bool(max_ok == n and idx_ok == n)

    print(f"cores                {n}")
    print(f"max  exact           {max_ok}/{n}")
    print(f"index exact          {idx_ok}/{n}")
    print(f"ties in the data     {ties}")
    if idx_ok != n:
        bad = int(np.argmax(got_idx != expected_idx))
        print(f"  first mismatch core {bad}: expected {expected_idx[bad]} got {got_idx[bad]}")
    print(f"\nVERDICT: {'PASS' if passed else 'FAIL'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "cores": n, "max_exact": max_ok, "index_exact": idx_ok,
        "ties_in_data": ties, "pass": passed, "seed": args.seed,
        "note": ("Index is gated on exact equality. NumPy argmax returns the first "
                 "maximal index and this kernel returns the last; ties are counted "
                 "so a tie can never be mistaken for a failure."),
    }, indent=2) + "\n")
    print(f"wrote {args.out}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
