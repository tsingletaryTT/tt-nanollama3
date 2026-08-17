#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Gate: per-core scoring on 110 Tensix cores against the NumPy reference.

What runs where
---------------
The token->core layout is made *physically real* here. The vocabulary is permuted
so that each core's ~291 tokens are contiguous, padded to one 32x32 tile, and one
tile is handed to one Tensix. Each core then reduces its own bag out of its own
L1 and writes a single number. No core reads another core's memory, and the
assignment of tokens to cores is the measured layout, not an arbitrary split.

That is the sampler's first stage running on the silicon it was designed around.

Why MAX and not log-sum-exp
---------------------------
A stable log-sum-exp needs max, broadcast-subtract, exp, sum, log — four more
stages and a second pass. MAX is one reduce and is a legitimate scoring rule on
its own ("which region holds the single most probable token?"). The CPU sampler
carries both so they can be compared; this gate covers the MAX path.

The padding trap
----------------
A core owns ~291 tokens but a tile holds 1024 elements. Padding must be a value
that can never win the reduction — under MAX that is a large negative number.
Under SUM it would have to be zero. Getting it wrong does not crash: it silently
returns the padding as the score, which is exactly the class of bug this gate
exists to catch.

Requires hardware:

    gozer run --chips 1 --who "claude:tt-tnt" --reason "per-core score gate" -- \
        python scripts/probe_core_scores_device.py
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

from scripts.sample_topological import TokenCoreMap  # noqa: E402

READER_KERNEL = ROOT / "kernels" / "core_reduce_reader.cpp"
COMPUTE_KERNEL = ROOT / "kernels" / "core_reduce_compute.cpp"
WRITER_KERNEL = ROOT / "kernels" / "tile_writer.cpp"

TILE = 32
TILE_ELEMS = TILE * TILE
TILE_BYTES = 4 * TILE_ELEMS

CB_LOGITS, CB_SCALER, CB_OUT = 0, 1, 16

#: Padding for slots a core does not own. Must lose every MAX comparison, and
#: must stay finite: -inf through the reduce hardware is not worth trusting
#: without a reason to.
PAD = -1.0e30


def pack_by_core(logits: np.ndarray, layout: TokenCoreMap) -> tuple[np.ndarray, np.ndarray]:
    """Permute the vocabulary into one padded tile per core.

    Returns ``(packed, membership)`` where ``packed`` is ``(n_cells, 1024)`` and
    ``membership[c]`` is the token ids core ``c`` owns, in ascending order.
    """
    packed = np.full((layout.n_cells, TILE_ELEMS), PAD, dtype=np.float32)
    membership = []
    for cell in range(layout.n_cells):
        tokens = np.flatnonzero(layout.token_cell == cell)
        if tokens.size > TILE_ELEMS:
            raise SystemExit(
                f"core {cell} owns {tokens.size} tokens but a tile holds "
                f"{TILE_ELEMS}; use a larger grid or more tiles per core"
            )
        packed[cell, : tokens.size] = logits[tokens]
        membership.append(tokens)
    return packed, membership


def run_device(device, packed: np.ndarray, cores: list) -> np.ndarray:
    """Dispatch the reduction and return one score per core."""
    n_cells = packed.shape[0]

    logits_tensor = ttnn.from_torch(
        __import__("torch").from_numpy(packed.reshape(n_cells * TILE, TILE)),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scaler_tensor = ttnn.full(
        (TILE, TILE), 1.0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = ttnn.empty(
        (n_cells * TILE, TILE), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

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

    reader_rt, writer_rt = [], []
    for idx, core in enumerate(cores):
        reader_rt.append(
            (core, [logits_tensor.buffer_address(), scaler_tensor.buffer_address(), idx])
        )
        writer_rt.append((core, [output.buffer_address(), idx, 1]))

    compute_config = ttnn.ComputeConfigDescriptor()
    compute_config.math_approx_mode = False
    compute_config.fp32_dest_acc_en = True

    program = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=str(READER_KERNEL),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_set,
                compile_time_args=reader_compile,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(COMPUTE_KERNEL),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_set,
                compile_time_args=[CB_LOGITS, CB_SCALER, CB_OUT],
                runtime_args=[],
                config=compute_config,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(WRITER_KERNEL),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_set,
                compile_time_args=writer_compile,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
        ],
        semaphores=[],
        cbs=[cb(CB_LOGITS), cb(CB_SCALER), cb(CB_OUT)],
    )

    result = ttnn.generic_op([logits_tensor, scaler_tensor, output], program)
    tiles = ttnn.to_torch(result).float().numpy().reshape(n_cells, TILE, TILE)
    # REDUCE_SCALAR puts the answer in element [0][0] of each tile.
    return tiles[:, 0, 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=ROOT / "artifacts" / "token_core_map.npz")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help=(
            "Use only the first N cells. Bisection handle: a stall across 110 "
            "cores says nothing about which kernel is at fault; the same stall on "
            "one core is a single-kernel question."
        ),
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "docs" / "measurements" / "core-scores-device-gate.json",
    )
    args = parser.parse_args()

    layout = TokenCoreMap.load(args.map)
    rng = np.random.default_rng(args.seed)
    # Synthetic logits rather than a model forward: the gate is about the
    # reduction and the layout, and synthetic values are exactly reproducible
    # without loading 123M parameters.
    logits = rng.normal(0.0, 4.0, size=layout.token_cell.shape[0]).astype(np.float32)

    # Make every input exactly bf16-representable before it goes near the device.
    # The Tensix source registers narrow fp32 on the way into the math pipeline,
    # so with arbitrary fp32 inputs the gate is really testing float narrowing
    # rather than the reduction, and a 1-ULP disagreement is unattributable.
    # With bf16-exact inputs the true max is bf16-exact too, and the device must
    # return it bit-for-bit -- any mismatch is then a real bug in the reduce, the
    # padding, or the permutation, which is what this gate is for.
    import torch

    logits = torch.from_numpy(logits).bfloat16().float().numpy()

    packed, membership = pack_by_core(logits, layout)
    n_cells = min(args.cores, layout.n_cells) if args.cores else layout.n_cells
    packed = packed[:n_cells]
    membership = membership[:n_cells]
    exact_max = np.array([logits[m].max() for m in membership], dtype=np.float32)

    # The Tensix source registers narrow fp32 on the way into the math pipeline,
    # so a MAX reduce returns the winning value rounded to bfloat16 even with
    # fp32 CBs and fp32 DST accumulation. Measured: true max 11.938511 came back
    # as 11.937500, which is exactly bf16(11.938511).
    #
    # The bar is therefore bf16(reference), compared EXACTLY -- not a loose
    # epsilon. That is a sharper test than the fp32 one it replaces: it asserts
    # the device found the true maximum and represented it in bf16, and it still
    # fails if the padding leaks, the permutation disagrees, or the tile layout
    # is wrong. A tolerance band would have let all three through.
    # Two candidate narrowings, because which one the hardware uses is a
    # measurement, not an assumption:
    #   rne   - round-to-nearest-even, what torch.bfloat16() does
    #   trunc - drop the low 16 mantissa bits, what truncating hardware does
    # They differ by at most 1 ULP, so comparing against the wrong one produces
    # exactly the "mostly right, a few cores off by a hair" pattern that is easy
    # to misread as a real numerical bug.
    # Inputs are bf16-exact, so both narrowings are identity here; they are still
    # computed so a future change to the input distribution cannot silently
    # reintroduce the ambiguity.
    rne = torch.from_numpy(exact_max).bfloat16().float().numpy()
    trunc = (exact_max.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)
    expected = exact_max

    device = ttnn.open_device(device_id=args.device_id)
    try:
        grid = device.compute_with_storage_grid_size()
        available = grid.x * grid.y
        if available < n_cells:
            raise SystemExit(
                f"need {n_cells} cores but the grid exposes "
                f"{grid.x}x{grid.y} = {available}"
            )
        cores = [ttnn.CoreCoord(c % grid.x, c // grid.x) for c in range(n_cells)]
        print(f"grid {grid.x}x{grid.y}; dispatching {n_cells} cores, one tile each")
        observed = run_device(device, packed, cores)
    finally:
        ttnn.close_device(device)

    matches_rne = int((observed == rne).sum())
    matches_trunc = int((observed == trunc).sum())
    print(f"narrowing: matches round-to-nearest {matches_rne}/{n_cells}, "
          f"matches truncation {matches_trunc}/{n_cells}")
    if matches_trunc > matches_rne:
        expected = trunc
        print("  -> hardware truncates; gating against truncated reference")

    delta = np.abs(observed - expected)
    worst = int(np.argmax(delta))
    exact = int((observed == expected).sum())

    print(f"\ncores            {n_cells}")
    print(f"exact matches    {exact}/{n_cells}")
    print(f"max |delta|      {delta.max():.6g}  (core {worst})")
    print(f"  expected {expected[worst]:.6f}   observed {observed[worst]:.6f}")

    passed = bool(delta.max() == 0.0)
    print(f"\nVERDICT: {'PASS — device matches bf16(NumPy) exactly' if passed else 'FAIL'}")

    payload = {
        "cores": int(n_cells),
        "exact_matches": exact,
        "max_abs_delta": float(delta.max()),
        "reference": "bfloat16(numpy max) compared exactly",
        "matches_round_to_nearest": matches_rne,
        "matches_truncation": matches_trunc,
        "max_abs_delta_vs_fp32": float(np.abs(observed - exact_max).max()),
        "pass": passed,
        "layout_digest": layout.provenance["embedding_sha256"],
        "seed": args.seed,
        "pool": "MAX",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
