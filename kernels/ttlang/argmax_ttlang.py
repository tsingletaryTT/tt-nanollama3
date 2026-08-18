#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The per-core argmax in tt-lang. Compiles further on main than on the wheel.

Kept as a reproduction, not as dead code pretending to work. The shipping kernel
is kernels/core_argmax_compute.cpp, gated at 110/110 exact.

    error: operand #1 does not dominate this use
      --> hit = ttl.eq(f_blk, m)
      note: operand defined here (op in the same block)
      --> m = ttl.math.reduce_max(f_blk, dims=[-1])

That is the failure on tt-lang 1.1.6, the newest PyPI wheel. It does NOT occur on
1.1.9.dev12 built from main: the handling landed in v1.1.8 via 711fcb38b, which
added a compute-planning pass that models this exact case. PyPI stops at 1.1.6,
so the fix has been released for three versions and is installable by nobody.

On main the port reaches a different, narrower error:

    error: dataflow buffer 1 requires incompatible unpack modes in one kernel
      --> sel = ttl.mul(hit, i_blk)
      note: operand 0 establishes the conflicting unpack mode
      --> m = ttl.math.reduce_max(f_blk, dims=[-1])

`hit` descends from the reduce and carries its unpack mode; `i_blk` is a plain
user DFB. Untried: CompilerOptions.reuse_user_dfbs / compiler_dfbs, or staging
the index through an intermediate.

Everything else about the port was fine: sub, mul, eq and reduce_max all exist in
1.1.6, and eq removes the need for the (1 - abs(sign(d))) trick a C++ version
would want. The obstacle is scheduling, not the op set.

Why port at all
---------------
`kernels/core_argmax_compute.cpp` is gated at 110/110 exact via ttnn.generic_op,
so this is not needed for correctness -- it is a test of whether tt-lang can
express the same kernel, on work whose right answer is already known. That makes
it a comparison rather than a leap: whatever this produces can be diffed against
a version that is already proven.

What tt-lang 1.1.6 gives us
---------------------------
`sub`, `mul`, `eq` and `reduce_max` are all present, so the algorithm maps
directly and the `1 - abs(sign(d))` trick the C++ version would have needed is
unnecessary -- `eq` is a real op here.

The open question, which this file exists to answer
---------------------------------------------------
`reduce_max(x, dims=[...])` reduces across the TILE dimensions of a block. Our
argmax needs a reduction WITHIN one tile: each core owns ~291 tokens padded into
a single 32x32 tile, and the max is over those 1024 elements, not across tiles.
`examples/eltwise_broadcast_reduce.py` makes tiles uniform specifically so that a
tile-level reduce matches an element-level reference, which suggests within-tile
reduction is not what this op does.

So the layout here is deliberately different from the C++ kernel: instead of one
1024-element tile per core, a core's region is laid out as ROWS of a block, and
the reduction runs across that block's tile dimension. If that compiles and
matches, the port is real but the data layout has to change with it -- which is a
finding about tt-lang's model, not a detail.
"""

import ttl
import ttnn

TILE_SIZE = 32


@ttl.operation(grid="full")
def per_core_argmax(field: ttnn.Tensor, index: ttnn.Tensor,
                    out_max: ttnn.Tensor, out_idx: ttnn.Tensor) -> None:
    """max and argmax over each node's slice, one tile row at a time."""
    row_tiles = field.shape[0] // TILE_SIZE
    grid_cols, grid_rows = ttl.grid_size(dims=2)
    rows_per_node = -(-row_tiles // (grid_rows * grid_cols))

    f_dfb = ttl.make_dataflow_buffer_like(field, shape=(1, 1), block_count=2)
    i_dfb = ttl.make_dataflow_buffer_like(index, shape=(1, 1), block_count=2)
    m_dfb = ttl.make_dataflow_buffer_like(out_max, shape=(1, 1), block_count=2)
    a_dfb = ttl.make_dataflow_buffer_like(out_idx, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        base = (node_row * grid_cols + node_col) * rows_per_node
        for local in range(rows_per_node):
            if base + local < row_tiles:
                with (
                    f_dfb.wait() as f_blk,
                    i_dfb.wait() as i_blk,
                    m_dfb.reserve() as m_blk,
                    a_dfb.reserve() as a_blk,
                ):
                    # 1. the maximum of this core's slice
                    m = ttl.math.reduce_max(f_blk, dims=[-1])
                    # 2-4. mask the index by "is this the max", then reduce again.
                    #      eq is exact here: both sides are the same bf16 values.
                    hit = ttl.eq(f_blk, m)
                    sel = ttl.mul(hit, i_blk)
                    a = ttl.math.reduce_max(sel, dims=[-1])
                    m_blk.store(m)
                    a_blk.store(a)

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        base = (node_row * grid_cols + node_col) * rows_per_node
        for local in range(rows_per_node):
            r = base + local
            if r < row_tiles:
                with f_dfb.reserve() as f_blk, i_dfb.reserve() as i_blk:
                    tx_f = ttl.copy(field[r : r + 1, 0:1], f_blk)
                    tx_i = ttl.copy(index[0:1, 0:1], i_blk)
                    tx_f.wait()
                    tx_i.wait()

    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        base = (node_row * grid_cols + node_col) * rows_per_node
        for local in range(rows_per_node):
            r = base + local
            if r < row_tiles:
                with m_dfb.wait() as m_blk, a_dfb.wait() as a_blk:
                    tx_m = ttl.copy(m_blk, out_max[r : r + 1, 0:1])
                    tx_a = ttl.copy(a_blk, out_idx[r : r + 1, 0:1])
                    tx_m.wait()
                    tx_a.wait()
