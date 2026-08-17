#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Generate text with the sampling step running on 110 Tensix cores.

What is on the device
---------------------
Each step, every core receives its own region of the vocabulary -- its ~291
tokens, contiguous in one tile, from the measured layout -- perturbs them with
Gumbel noise drawn from *its own* Tensix PRNG, and writes the perturbed tile
back. That is the hardware-bound part, and it is exact: Gumbel-max means the
argmax of those perturbed values is a draw from softmax(logits / T), and the
decomposition across cores is provably equivalent to sampling over the whole
vocabulary at once.

What is on the host
-------------------
The model forward (this is a 123M model; the transformer runs in PyTorch), and
the final argmax. The argmax is host-side for a concrete reason rather than
convenience: ``reduce`` returns the winning *value*, not its index, and
generation needs to know which token won. The neighbourhood mask has to be
applied at that same point anyway, so the two are done together.

The neighbourhood
-----------------
The winning core defines a region; ``--hops`` then admits every core within that
many NoC hops on the torus, and the token is the argmax over the admitted cores.
``--direction`` shifts the region one step first, which is how the fan asks the
same question from a different proximity.

Requires hardware:

    gozer run --chips 1 --who "claude:tt-tnt" --reason "device topological sampling" -- \
        python scripts/sample_topological_device.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import ttnn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.probe_core_scores_device import (  # noqa: E402
    CB_LOGITS, CB_SCALER, TILE, TILE_BYTES, TILE_ELEMS, pack_by_core,
)
from scripts.sample_topological import (  # noqa: E402
    SIX_DIRECTIONS, TokenCoreMap, load_model, neighbourhood,
)

GUMBEL_KERNEL = ROOT / "kernels" / "core_gumbel_compute.cpp"
READER_KERNEL = ROOT / "kernels" / "core_reduce_reader.cpp"
WRITER_KERNEL = ROOT / "kernels" / "tile_writer.cpp"

CB_PERTURBED, CB_OUT = 2, 16
SEED_STRIDE = 7919


class DeviceSampler:
    """Holds the device-side state so tensors are allocated once, not per token."""

    def __init__(self, device, layout: TokenCoreMap, membership, temperature: float,
                 seed: int):
        self.device = device
        self.layout = layout
        self.membership = membership
        self.temperature = temperature
        self.seed = seed

        grid = device.compute_with_storage_grid_size()
        if grid.x * grid.y < layout.n_cells:
            raise SystemExit(
                f"layout needs {layout.n_cells} cores, grid exposes {grid.x}x{grid.y}"
            )
        self.cores = [
            ttnn.CoreCoord(c % grid.x, c // grid.x) for c in range(layout.n_cells)
        ]
        self.core_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(c, c) for c in self.cores]
        )
        self.scaler = ttnn.full(
            (TILE, TILE), 1.0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.output = ttnn.empty(
            (layout.n_cells * TILE, TILE), dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT, device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _cb(self, index):
        return ttnn.CBDescriptor(
            total_size=2 * TILE_BYTES,
            core_ranges=self.core_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=index, data_format=ttnn.float32, page_size=TILE_BYTES
                )
            ],
        )

    def perturb(self, packed: np.ndarray, step: int) -> np.ndarray:
        """Upload logits, run per-core Gumbel perturbation, return (cells, 1024)."""
        import torch

        n = self.layout.n_cells
        logits_tensor = ttnn.from_torch(
            torch.from_numpy(packed.reshape(n * TILE, TILE)),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        reader_compile = [CB_LOGITS, CB_SCALER]
        reader_compile.extend(
            ttnn.TensorAccessorArgs(logits_tensor).get_compile_time_args()
        )
        reader_compile.extend(ttnn.TensorAccessorArgs(self.scaler).get_compile_time_args())
        # The writer drains the PERTURBED buffer here, not the reduced one.
        writer_compile = [CB_PERTURBED]
        writer_compile.extend(ttnn.TensorAccessorArgs(self.output).get_compile_time_args())

        reader_rt, writer_rt, compute_rt = [], [], []
        for idx, core in enumerate(self.cores):
            reader_rt.append(
                (core, [logits_tensor.buffer_address(), self.scaler.buffer_address(), idx])
            )
            writer_rt.append((core, [self.output.buffer_address(), idx, 1]))
            # Seeds must advance with the step, or every token would be drawn
            # from the same noise and generation would lock immediately.
            compute_rt.append(
                (core, [self.seed + step * 1_000_003 + idx * SEED_STRIDE])
            )

        compute_config = ttnn.ComputeConfigDescriptor()
        compute_config.math_approx_mode = False
        compute_config.fp32_dest_acc_en = True

        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=str(READER_KERNEL),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=self.core_set, compile_time_args=reader_compile,
                    runtime_args=reader_rt, config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(GUMBEL_KERNEL),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=self.core_set,
                    compile_time_args=[CB_LOGITS, CB_SCALER, CB_PERTURBED, CB_OUT],
                    defines=[("PERTURB_ONLY", "1")],
                    runtime_args=compute_rt, config=compute_config,
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(WRITER_KERNEL),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=self.core_set, compile_time_args=writer_compile,
                    runtime_args=writer_rt, config=ttnn.WriterConfigDescriptor(),
                ),
            ],
            semaphores=[], cbs=[self._cb(CB_LOGITS), self._cb(CB_SCALER),
                                self._cb(CB_PERTURBED), self._cb(CB_OUT)],
        )

        result = ttnn.generic_op([logits_tensor, self.scaler, self.output], program)
        return ttnn.to_torch(result).float().numpy().reshape(n, TILE_ELEMS)

    def sample(self, logits: np.ndarray, step: int, *, hops: int,
               direction: tuple[int, int] | None) -> tuple[int, int]:
        """One token. Returns ``(token_id, winning_cell)``."""
        scaled = (logits / max(self.temperature, 1e-6)).astype(np.float32)
        packed, _ = pack_by_core(scaled, self.layout)
        perturbed = self.perturb(packed, step)

        # Slots a core does not own arrived as a huge negative and stay that way
        # after perturbation, so they cannot win; mask them anyway so a layout
        # change can never let padding through silently.
        owned = np.zeros_like(perturbed, dtype=bool)
        for cell, tokens in enumerate(self.membership):
            owned[cell, : len(tokens)] = True
        perturbed = np.where(owned, perturbed, -np.inf)

        origin = int(np.argmax(perturbed.max(axis=1)))
        winner = origin
        if direction is not None:
            x, y = self.layout.cell_xy(origin)
            winner = self.layout.xy_cell(x + direction[0], y + direction[1])

        admitted = neighbourhood(self.layout, winner, hops)
        mask = np.full(self.layout.n_cells, False)
        mask[admitted] = True
        if direction is not None:
            # Exclude the cell that actually won. Without this the fan is
            # degenerate: a one-hop shift still leaves the origin inside a
            # one-hop neighbourhood, so the global argmax is re-admitted and
            # every orthogonal direction returns the SAME token. Measured --
            # all four of (+1,0), (-1,0), (0,+1), (0,-1) produced byte-identical
            # text; only the two-hop diagonals escaped. The fan is meant to ask
            # from a *different* proximity, so the origin must be off the table.
            mask[origin] = False
        restricted = np.where(mask[:, None], perturbed, -np.inf)

        flat = int(np.argmax(restricted))
        cell, slot = divmod(flat, TILE_ELEMS)
        return int(self.membership[cell][slot]), cell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Tell me a way to go faster than light that will not work.")
    parser.add_argument("--hf-model", type=Path, default=None)
    parser.add_argument("--map", type=Path, default=ROOT / "artifacts" / "token_core_map.npz")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--fan", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    if args.hf_model is None:
        from scripts.probe_grid_layout import _default_model

        args.hf_model = _default_model()

    layout = TokenCoreMap.load(args.map)
    membership = [
        np.flatnonzero(layout.token_cell == c) for c in range(layout.n_cells)
    ]
    tokenizer, model = load_model(args.hf_model)

    import torch

    device = ttnn.open_device(device_id=args.device_id)
    try:
        sampler = DeviceSampler(device, layout, membership, args.temperature, args.seed)

        def run(direction):
            ids = tokenizer(args.prompt, return_tensors="pt").input_ids
            visited = []
            for step in range(args.max_new_tokens):
                with torch.no_grad():
                    logits = model(ids).logits[0, -1].float().numpy()
                # The direction is a BRANCH, not a standing constraint. Applying
                # it at every step forces the sampler out of its best region on
                # every token, and the penalty compounds: measured, all six
                # directions produced word salad by ~20 tokens. Diverging once at
                # the branch point and then generating normally is what "ask the
                # same question from six proximities" actually means.
                token, cell = sampler.sample(
                    logits, step, hops=args.hops,
                    direction=direction if step == 0 else None,
                )
                visited.append(cell)
                ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
                if tokenizer.eos_token_id is not None and token == tokenizer.eos_token_id:
                    break
            text = tokenizer.decode(ids[0], skip_special_tokens=True)
            return text[len(args.prompt):].strip(), visited

        print(f"layout {layout.spec.name} {layout.spec.label}, sampling on "
              f"{layout.n_cells} Tensix cores\n")
        print(f"PROMPT: {args.prompt}\n")

        if args.fan:
            for dx, dy in SIX_DIRECTIONS:
                text, visited = run((dx, dy))
                print(f"  ({dx:+d},{dy:+d}) cells {visited[:6]}…")
                print(f"          {text}\n")
        else:
            text, visited = run(None)
            print(f"OUTPUT: {text}")
            print(f"\ncells visited: {visited}")
            print(f"distinct cores: {len(set(visited))}/{len(visited)}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
