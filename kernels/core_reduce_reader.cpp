// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// Reader for the per-core score reduction.
//
// Each Tensix reads exactly ONE tile of logits -- its own bag of vocabulary --
// plus the shared scaler tile the reduce API requires. That one-tile-per-core
// arrangement is the whole point: the token->core layout
// (artifacts/token_core_map.npz) is made physically real by permuting the
// vocabulary so that a core's ~291 tokens are contiguous in DRAM, padded out to
// a 32x32 tile. A core then scores its own region out of its own L1, touching no
// other core's memory.
//
// Compile-time args:  0: logits CB id
//                     1: scaler CB id
//                     2+: TensorAccessorArgs for logits, then for the scaler
// Runtime args:       0: logits DRAM base address
//                     1: scaler DRAM base address
//                     2: this core's tile index (== its cell id in the layout)

#include "api/dataflow/dataflow_api.h"
#include <stdint.h>

void kernel_main() {
    constexpr uint32_t cb_logits = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);

    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t scaler_addr = get_arg_val<uint32_t>(1);
    const uint32_t tile_index = get_arg_val<uint32_t>(2);

    constexpr auto logits_args = TensorAccessorArgs<2>();
    const auto logits = TensorAccessor(logits_args, logits_addr, get_tile_size(cb_logits));
    constexpr auto scaler_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
    const auto scaler = TensorAccessor(scaler_args, scaler_addr, get_tile_size(cb_scaler));

    // This core's slice of the vocabulary.
    cb_reserve_back(cb_logits, 1);
    noc_async_read_page(tile_index, logits, get_write_ptr(cb_logits));

    // The scaler is one shared tile of 1.0 and every core reads page 0 of it.
    // reduce_tile takes it for all pool types; for MAX it does not affect the
    // result, but the API still requires the operand.
    cb_reserve_back(cb_scaler, 1);
    noc_async_read_page(0, scaler, get_write_ptr(cb_scaler));

    noc_async_read_barrier();
    cb_push_back(cb_logits, 1);
    cb_push_back(cb_scaler, 1);
}
