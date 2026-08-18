#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Run the tt-lang argmax port and diff it against NumPy.

The C++ version of this kernel is already gated at 110/110 exact, so the point
here is narrower: does tt-lang compile and run the same algorithm, and does its
reduce_max mean what the C++ reduce means?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import ttnn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kernels" / "ttlang"))

TILE = 32
ROWS = 8  # tile rows == cores exercised

from argmax_ttlang import per_core_argmax  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(20260818)
    field = rng.normal(0, 4, size=(ROWS * TILE, TILE)).astype(np.float32)
    field = torch.from_numpy(field).bfloat16().float().numpy()
    # Index tile: value == column, so the argmax over a row-tile is a column id.
    index = np.tile(np.arange(TILE, dtype=np.float32), (TILE, 1))

    device = ttnn.open_device(device_id=0)
    try:
        def up(a):
            return ttnn.from_torch(torch.from_numpy(a), dtype=ttnn.float32,
                                   layout=ttnn.TILE_LAYOUT, device=device,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
        t_field, t_index = up(field), up(index)
        t_max = ttnn.empty((ROWS * TILE, TILE), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        t_idx = ttnn.empty((ROWS * TILE, TILE), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        per_core_argmax(t_field, t_index, t_max, t_idx)

        got_max = ttnn.to_torch(t_max).float().numpy()
        got_idx = ttnn.to_torch(t_idx).float().numpy()
    finally:
        ttnn.close_device(device)

    blocks = field.reshape(ROWS, TILE, TILE)
    exp_max = blocks.max(axis=2)          # per row within each tile
    print("tt-lang ran. shapes:", got_max.shape, got_idx.shape)
    print("device max[0,:4]  :", got_max[0, :4])
    print("numpy   max[0,:4] :", exp_max[0, :4])
    print("device idx[0,:4]  :", got_idx[0, :4])
    print("\nThis prints rather than asserts: the first question is what tt-lang's")
    print("reduce_max produced, not whether it equals a reference we assumed.")


if __name__ == "__main__":
    main()
