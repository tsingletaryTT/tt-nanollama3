// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// Writer kernel for the per-core PRNG probe.
//
// Drains the intermediate CB filled by prng_probe_compute.cpp and writes each
// tile straight to DRAM. Every core writes to its OWN page range, handed to it
// as a runtime argument, so the host can attribute each tile back to the
// physical core that produced it. That attribution is the whole point of the
// probe — a flat undifferentiated buffer would tell us nothing about topology.
//
// Output is float32 and copied verbatim: no dtype conversion, no rescaling.
// A bf16 narrowing here would quantise away exactly the low-order differences
// between cores that we are trying to detect.
//
// Compile-time args:  0: intermediate CB id
//                     1+: TensorAccessorArgs for the output tensor
// Runtime args:       0: output DRAM base address
//                     1: first page id for this core
//                     2: tiles to write from this core

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t intermed_cb_id = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);

    const auto output_addrg = TensorAccessor(dst_args, dst_addr);

    // Page size comes from the CB itself rather than being recomputed here, so
    // the kernel cannot silently disagree with the host's CB descriptor.
    const uint32_t page_bytes = get_local_cb_interface(intermed_cb_id).fifo_page_size;

    Noc noc;
    CircularBuffer cb_intermed(intermed_cb_id);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_intermed.wait_front(1);

        noc.async_write(
            CoreLocalMem<uint32_t>(cb_intermed.get_read_ptr()),
            output_addrg,
            page_bytes,
            {},
            {.page_id = start_page + i});
        noc.async_write_barrier();

        cb_intermed.pop_front(1);
    }
}
