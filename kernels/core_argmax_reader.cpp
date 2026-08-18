// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// Reader for the per-core argmax kernel: this core's field tile, the shared
// scaler tile, and the shared constant index tile (0..1023). The last two are
// page 0 for every core -- identical for all of them, so they are uploaded once
// rather than regenerated per core.
//
// Compile-time: 0 field CB, 1 scaler CB, 2 index CB, 3+ accessors for each
// Runtime:      0 field addr, 1 scaler addr, 2 index addr, 3 this core's tile

#include "api/dataflow/dataflow_api.h"
#include <stdint.h>

void kernel_main() {
    constexpr uint32_t cb_in     = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);
    constexpr uint32_t cb_index  = get_compile_time_arg_val(2);

    const uint32_t in_addr     = get_arg_val<uint32_t>(0);
    const uint32_t scaler_addr = get_arg_val<uint32_t>(1);
    const uint32_t index_addr  = get_arg_val<uint32_t>(2);
    const uint32_t tile_index  = get_arg_val<uint32_t>(3);

    constexpr auto in_args = TensorAccessorArgs<3>();
    const auto in = TensorAccessor(in_args, in_addr, get_tile_size(cb_in));
    constexpr auto scaler_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const auto scaler = TensorAccessor(scaler_args, scaler_addr, get_tile_size(cb_scaler));
    constexpr auto index_args = TensorAccessorArgs<scaler_args.next_compile_time_args_offset()>();
    const auto index = TensorAccessor(index_args, index_addr, get_tile_size(cb_index));

    cb_reserve_back(cb_in, 1);
    noc_async_read_tile(tile_index, in, get_write_ptr(cb_in));
    cb_reserve_back(cb_scaler, 1);
    noc_async_read_tile(0, scaler, get_write_ptr(cb_scaler));
    cb_reserve_back(cb_index, 1);
    noc_async_read_tile(0, index, get_write_ptr(cb_index));

    noc_async_read_barrier();
    cb_push_back(cb_in, 1);
    cb_push_back(cb_scaler, 1);
    cb_push_back(cb_index, 1);
}
