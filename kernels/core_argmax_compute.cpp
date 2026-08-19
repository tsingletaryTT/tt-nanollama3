// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// Per-core argmax WITH INDEX: each Tensix reports both the largest perturbed
// value in its own region and WHICH token produced it.
//
// Why this kernel exists
// ----------------------
// `reduce` returns a winning value and not its index, so every earlier version of
// this sampler shipped all 110 full tiles back to the host and did the argmax
// there -- 110 x 4 KB per token. This kernel returns two numbers per core
// instead, which is 880 bytes per token rather than 450 KB, and leaves the host
// with a 110-element comparison instead of a 112,640-element one.
//
// How the index is recovered
// --------------------------
// There is no argmax primitive here, but there is a max, and that is enough:
//
//   1. M = max(tile)                            -- the reduce already proven exact
//   2. d = tile - M                             -- broadcast scalar subtract; 0 at the max
//   3. m = (d == 0)                             -- 1 at the max, 0 elsewhere
//   4. s = m * index_tile                       -- the winning index, 0 elsewhere
//   5. i = max(s)                               -- the index itself
//
// Step 5 works because indices are non-negative, so the masked-out zeros can
// never beat a real index -- EXCEPT when the winner is index 0, where max(s) is
// 0, which is also the correct answer. Ties resolve to the LARGEST index, which
// is arbitrary but deterministic, and determinism is the property this project
// gates on.
//
// The index tile is uploaded by the host as a constant 0..1023 rather than
// synthesised here: it is the same for every core and every step, so generating
// it on-device each dispatch would be work with no purpose.
//
// Both results go to ONE output CB, max first then index, because a Tensix has
// two NOCs and therefore room for two dataflow kernels. A reader plus two writers
// is three, and the runtime rejects it with `local_noc0_in_use and
// local_noc1_in_use`.
//
// Compile-time args: 0 perturbed CB, 1 scaler CB, 2 index CB,
//                    3 output CB (2 tiles), 4 max-scalar CB, 5 scratch CB

#include "api/compute/compute_kernel_api.h"
#include "api/compute/reduce.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_in     = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);
    constexpr uint32_t cb_index  = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out    = get_compile_time_arg_val(3);
    constexpr uint32_t cb_max    = get_compile_time_arg_val(4);
    constexpr uint32_t cb_scratch= get_compile_time_arg_val(5);

    cb_wait_front(cb_in, 1);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_index, 1);

    // ---- 1. M = max(tile) -----------------------------------------------
    cb_reserve_back(cb_max, 1);
    compute_kernel_hw_startup(cb_in, cb_scaler, cb_max);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_SCALAR>(cb_in, cb_scaler, cb_max);
    tile_regs_acquire();
    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_SCALAR>(cb_in, cb_scaler, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_max);
    tile_regs_release();
    cb_push_back(cb_max, 1);
    cb_wait_front(cb_max, 1);

    // Emit M as output tile 0. cb_max stays as the intermediate the broadcast
    // subtract below reads its scalar from.
    cb_reserve_back(cb_out, 1);
    compute_kernel_hw_startup(cb_max, cb_out);
    copy_tile_init(cb_max);
    tile_regs_acquire();
    copy_tile(cb_max, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);

    // ---- 2-4. mask the index tile by "is this the max" -------------------
    cb_reserve_back(cb_scratch, 1);
    compute_kernel_hw_startup(cb_in, cb_max, cb_scratch);
    tile_regs_acquire();

    // d = tile - M, broadcasting M from element [0][0] of cb_max.
    sub_bcast_scalar_init(cb_in, cb_max);
    sub_tiles_bcast_scalar(cb_in, cb_max, 0, 0, 0);

    // m = (d == 0): exactly 1 where the maximum sits.
    eqz_tile_init();
    eqz_tile(0);

    // s = m * index_tile
    copy_tile_init(cb_index);
    copy_tile(cb_index, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);

    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_scratch);
    tile_regs_release();
    cb_push_back(cb_scratch, 1);
    cb_wait_front(cb_scratch, 1);

    // ---- 5. i = max(s) ---------------------------------------------------
    cb_reserve_back(cb_out, 1);
    compute_kernel_hw_startup(cb_scratch, cb_scaler, cb_out);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_SCALAR>(cb_scratch, cb_scaler, cb_out);
    tile_regs_acquire();
    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_SCALAR>(cb_scratch, cb_scaler, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);

    cb_pop_front(cb_scratch, 1);
    cb_pop_front(cb_max, 1);
    cb_pop_front(cb_index, 1);
    cb_pop_front(cb_scaler, 1);
    cb_pop_front(cb_in, 1);
}
