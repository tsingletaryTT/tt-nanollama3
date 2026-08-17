// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// Sampling by Gumbel-max, one core per region, each drawing from its own PRNG.
//
// Why Gumbel-max
// --------------
// Sampling a token from softmax(logits) normally needs exp, a full sum, a
// cumulative sum and a search -- awkward on a tile engine and requiring a
// cross-core total before anything can be drawn. Gumbel-max needs none of it:
//
//     argmax_i ( logit_i / T + g_i ),   g_i = -log(-log(u_i)),  u_i ~ U(0,1)
//
// is distributed exactly as a draw from softmax(logits / T). No normalisation,
// no communication -- and the operation it reduces to is a MAX, which is the
// reduction already proven exact on all 110 cores
// (docs/measurements/core-scores-device-gate.json).
//
// It also composes hierarchically, which is what makes it right for this design.
// If every core perturbs its OWN tokens and takes its own max, then the argmax
// across cores of those maxima is a draw from the softmax over the ENTIRE
// vocabulary -- provably, not approximately. So the sampler decomposes into
// exactly the shape of the hardware: 110 cores each answering about their own
// region out of their own L1, from their own random stream, with no core needing
// to know anything about another.
//
// The temperature is folded into the logits on the host, so this kernel does no
// scaling: one less place for the two implementations to disagree.
//
// Padding: unowned slots arrive as a large negative value. Adding a Gumbel
// variate (order 1) leaves them hugely negative, so they still cannot win.
//
// Compile-time args:  0: logits CB   1: scaler CB   2: intermediate CB   3: out CB
// Runtime args:       0: this core's PRNG seed

#include "api/compute/compute_kernel_api.h"
#include "api/compute/reduce.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/rand.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_logits = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);
    constexpr uint32_t cb_perturbed = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);

    const uint32_t seed = get_arg_val<uint32_t>(0);

    union {
        float f;
        uint32_t u;
    } from, scale;
    from.f = 0.0f;
    scale.f = 1.0f;

    cb_wait_front(cb_logits, 1);

    // ---- pass 1: perturb this core's tokens with its own Gumbel noise -------
    compute_kernel_hw_startup(cb_logits, cb_perturbed);
    copy_tile_init(cb_logits);
    init_sfpu(cb_logits, cb_perturbed);
    rand_tile_init(seed);

    cb_reserve_back(cb_perturbed, 1);
    tile_regs_acquire();

    copy_tile(cb_logits, 0, 0);

    // dst1 <- u ~ U(0,1), then -log(-log(u)) in place. Two logs and two negations
    // rather than a table: the SFPU has log, and this is the exact transform, so
    // there is nothing to approximate.
    // Every SFPU op needs its own init immediately before use -- they program
    // different LUT/config state, so interleaving them means re-initialising at
    // each switch. tt-metal's own unary codegen emits exactly this pairing.
    // Omitting them does not fail loudly: the draws come out mis-distributed
    // (measured TV 0.9324 against a 0.5008 noise floor) while still replaying
    // deterministically, which reads like a modelling error rather than a bug.
    rand_tile(1, from.u, scale.u);
    log_tile_init();
    log_tile(1);       // log(u)          -- negative
    negative_tile_init();
    negative_tile(1);  // -log(u)         -- positive
    log_tile_init();
    log_tile(1);       // log(-log(u))
    negative_tile_init();
    negative_tile(1);  // -log(-log(u))   -- Gumbel(0,1)

    add_binary_tile_init();
    add_binary_tile(0, 1, 0);

    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_perturbed);
    tile_regs_release();
    cb_push_back(cb_perturbed, 1);

#ifdef PERTURB_ONLY
    // Generation path: the host wants the whole perturbed tile, not its maximum.
    // The reduce gives the winning VALUE but not its index, and generation needs
    // to know *which token* won -- so the argmax moves to the host, where the
    // neighbourhood mask has to be applied anyway. The device keeps the part
    // that is actually hardware-bound: each core perturbing its own region from
    // its own PRNG. The writer is pointed at cb_perturbed in this mode.
    cb_pop_front(cb_logits, 1);
#else
    // ---- pass 2: this core's Gumbel-max statistic ---------------------------
    cb_wait_front(cb_perturbed, 1);
    cb_wait_front(cb_scaler, 1);
    cb_reserve_back(cb_out, 1);

    compute_kernel_hw_startup(cb_perturbed, cb_scaler, cb_out);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_SCALAR>(cb_perturbed, cb_scaler, cb_out);

    tile_regs_acquire();
    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_SCALAR>(cb_perturbed, cb_scaler, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, 1);
    cb_pop_front(cb_perturbed, 1);
    cb_pop_front(cb_scaler, 1);
    cb_pop_front(cb_logits, 1);
#endif
}
