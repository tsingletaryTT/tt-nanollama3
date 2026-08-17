// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// Per-core score reduction: each Tensix reduces its own bag of vocabulary to one
// number, entirely out of its own L1.
//
// This is the heart of the topological sampler moved onto the silicon. On CPU,
// scripts/sample_topological.py reduces each core's member logits with a
// log-sum-exp. Here the reduction is MAX, for a reason worth stating plainly
// rather than hiding: a numerically stable log-sum-exp needs max, then a
// broadcast subtract, then exp, then sum, then log -- four more stages and a
// second pass over the tile. MAX is one reduce, and it is a legitimate scoring
// rule in its own right ("which region holds the single most probable token?").
// The log-sum-exp version is the follow-up, and the CPU reference carries both
// so the two can be gated against each other.
//
// The padding matters. A core owns ~291 tokens but a tile is 1024 elements, so
// the host pads with a large negative value. Under MAX that padding can never
// win; under SUM it would have to be zero instead. Getting this wrong would not
// crash -- it would quietly return the padding value as the score.
//
// Compile-time args:  0: logits CB id
//                     1: scaler CB id
//                     2: output CB id

#include "api/compute/compute_kernel_api.h"
#include "api/compute/reduce.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

// Entry point is `kernel_main`, matching this tt-metal's chlkc_list.h. The older
// `namespace NAMESPACE { void MAIN }` convention does not link here.
void kernel_main() {
    constexpr uint32_t cb_logits = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);

    // Programs the unpacker tile descriptors and the math ALU format registers
    // for this operand trio. Without it the math pipeline is unconfigured and the
    // reduce packs zeros -- silently, with no error. Measured: expected 11.938511,
    // observed 0.000000 on a single core before this call was added.
    compute_kernel_hw_startup(cb_logits, cb_scaler, cb_out);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_SCALAR>(cb_logits, cb_scaler, cb_out);

    cb_wait_front(cb_logits, 1);
    cb_wait_front(cb_scaler, 1);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();
    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_SCALAR>(cb_logits, cb_scaler, 0, 0, 0);
    tile_regs_commit();

    tile_regs_wait();
    // REDUCE_SCALAR leaves the result in element [0][0]; the rest of the packed
    // tile is not meaningful and the host reads only that element.
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, 1);
    cb_pop_front(cb_logits, 1);
    cb_pop_front(cb_scaler, 1);
}
