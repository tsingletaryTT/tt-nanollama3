// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// Compute kernel for the per-core PRNG probe.
//
// Purpose
// -------
// Each Tensix core has a PRNG whose seed lives in a *config register*
// (PRNG_SEED_Seed_Val_ADDR32), written by init_prng_seed() and drawn from by the
// SFPU via TTI_SFPMOV(0, 9, ...). This kernel makes that hardware observable: it
// seeds the local PRNG (optionally not at all) and fills whole tiles with draws,
// so the host can compare streams across cores and across runs.
//
// The question this exists to answer is NOT "does rand work". It is:
//
//     Is a core's random stream a property of the *silicon it runs on*, or only
//     of the seed we handed it?
//
// If two different cores given the SAME seed produce the SAME stream, then the
// PRNG carries no intrinsic per-core identity — the per-core character comes
// entirely from the host handing out different seeds, and a CPU can reproduce
// all of it. If they differ, some part of the stream is bound to the physical
// core. Either answer is useful; the design downstream depends on which it is.
//
// Build-time switch
// -----------------
//   SKIP_PRNG_SEED  If defined, rand_tile_init() is never called, so the PRNG is
//                   read from whatever state the core booted into (or was left
//                   in by a previous program). This is the "unseeded" condition
//                   — the only one that could plausibly be irreproducible on a
//                   CPU, and equally the one that may not be reproducible on
//                   hardware either. Measured, not assumed.
//
// Compile-time args:  0: intermediate CB id
// Runtime args:       0: seed          (ignored when SKIP_PRNG_SEED is defined)
//                     1: tiles to emit on this core

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rand.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t intermed_cb_id = get_compile_time_arg_val(0);

    const uint32_t seed = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    // rand_tile() maps its draws onto [from, from + scale). We want the raw
    // [0, 1) distribution so the host sees the PRNG as directly as possible:
    // any affine rescaling here would only obscure a difference between cores.
    // The API takes both as *bit patterns* of floats, not as floats.
    union {
        float f;
        uint32_t u;
    } from, scale;
    from.f = 0.0f;
    scale.f = 1.0f;

    CircularBuffer cb_intermed(intermed_cb_id);

    init_sfpu(intermed_cb_id, intermed_cb_id);

#ifndef SKIP_PRNG_SEED
    // Writes the seed to this core's PRNG config register. Note this is a
    // per-core write: nothing about it is shared across the grid.
    rand_tile_init(seed);
#endif

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_intermed.reserve_back(1);

        tile_regs_acquire();
        rand_tile(0, from.u, scale.u);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, intermed_cb_id, 0);
        tile_regs_release();

        cb_intermed.push_back(1);
    }
}
