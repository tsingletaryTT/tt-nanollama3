# ttml SDPA backward mismatches raw sequence length against tile-padded length

**Status: NOT FILED.** Drafted for Taylor to post upstream (or discard) — this repo's
standing convention for `docs/upstream-reports/` (see
`2026-08-20-moe-mesh-host-lockup.md` in this same directory). This note was promised in
`scripts/derive_traces.py`'s tile-alignment comments and `task-2-report.md` but never
actually written until now (task-6-report.md, FIX 5(a)).

## Summary

`ttml`'s SDPA backward kernel (`metal/ops/sdpa_bw/device/sdpa_bw_kv_device_operation.cpp`)
computes its *expected* input shape from the **raw, un-padded** sequence length `T`, but
is handed a tensor that some earlier stage in the backward pass built from the
**tile-padded** sequence length (`T` rounded up to the next multiple of 32, ttml's tile
width). Whenever a training batch's collated sequence length is not itself already a
multiple of 32, the two disagree and the kernel aborts with a `TT_FATAL`. This is not a
model or data bug — it reproduces on `SFTTrainer` with `pad_token_id` padding, real
`ttml` model weights, and a completely ordinary 8-example batch; the only thing that
matters is whether `T % 32 == 0`.

## Repro

`ttml.trainers.SFTTrainer.train()`, one optimizer step, on a batch whose collated
sequence length `T = 22` (8 identical hand-built examples: an 12-token prompt + a
10-token completion, `sft_collate_fn`'s dynamic per-batch padding — see below for why
this is not a corner case). Fails in `loss.backward()`:

```
TT_FATAL: u_scaler shape mismatch: expected (*, 1408, 32), got Shape([1, 1, 2048, 32])
  (metal/ops/sdpa_bw/device/sdpa_bw_kv_device_operation.cpp:77)
info:
u_shape[-2] == expected_rows && u_shape[-1] == tt::constants::TILE_WIDTH
u_scaler shape mismatch: expected (*, 1408, 32), got Shape([1, 1, 2048, 32])
backtrace:
 --- ttml::metal::ops::sdpa_bw::device::SDPABackwardKVDeviceOperation::validate_on_program_cache_miss(...)
 --- ttnn::device_operation::detail::launch_operation_with_adapter<...SDPABackwardKVDeviceOperation...>(...)
 --- ttnn::prim::ttml_sdpa_kv_bw(...)
 --- ttml::metal::sdpa_bw(...)
 --- ttml::autograd::Tensor::backward(bool)
```

## Root cause, worked out from the numbers

- `expected_rows = qB * qH * qS = 4 * 16 * 22 = 1408` — batch 4, 16 heads, and the
  **raw** (un-padded) sequence length `qS = 22`. (Equivalently `22 * 64 = 1408`, folding
  `qB * qH = 64` together.) The validator computed its expectation from the raw `T`.
- The tensor it was actually compared against has shape `[1, 1, 2048, 32]`, and
  `2048 = 4 * 16 * 32` (equivalently `32 * 64 = 2048`) — the same batch/head geometry,
  but with `T` rounded UP to `32`, the next tile boundary above 22. Some upstream buffer
  in the backward pass was built from the **tile-padded** sequence length.
- The two disagree in exactly the case where `T` is not itself already a multiple of 32
  (`ceil(22 / 32) * 32 = 32 != 22`). For any `T` that is already tile-aligned, raw and
  padded agree and the mismatch cannot occur — which is exactly why this bug is invisible
  until an odd-length batch is fed through.
- Also visible earlier in the same run, non-fatal but corroborating the same root cause:
  `rotary_embedding_llama sequence tile coverage mismatch: input_Ht=1, cos_Ht=16,
  sin_Ht=16, rotary_Ht=1` — the forward pass already treats the sequence as 1 tile
  (`ceil(22/32)=1`) in one place while other internal tensors are still shaped for the
  full 16-tile (512-token) rotary cache. Same family of raw-vs-tile-rounded shape
  disagreement, different op.

## Workaround (what this repo actually does)

Confirmed by direct diagnostic (`task-2-report.md`, section 6): re-running the identical
pipeline (same prompt/completion text, same model, same `SFTConfig`) but padding the
completion with `pad_token_id` so the collated sequence lands exactly on a tile boundary
(`T=32` instead of `22`) completes all optimizer steps cleanly, with the loss decreasing
across the run — i.e. the mask + SFTTrainer + model combination itself is fine; the
blocker is specifically the tile alignment of `T`.

The fix applied here is **pad every training example to a multiple of 32 tokens before
it ever reaches the dataloader**, not in a custom collate function:
`scripts/derive_traces.py`'s `build_sft_examples` pads each example's `input_ids` (with
`pad_token_id`) and `labels` (with `-100`, so the padding never enters the loss) up to
its own next multiple of 32, *per example*, before batching.

**Important caveat this workaround does NOT fully close**: `ttml.datasets.sft_collate_fn`
itself still pads dynamically, per batch, to `min(longest example in the batch,
max_seq_len)`. If every example fed to it is individually tile-aligned (as
`build_sft_examples` guarantees here), the batch max is guaranteed aligned too, since the
max of several multiples of 32 is itself a multiple of 32. But `sft_collate_fn`'s own
dynamic padding provides **no such guarantee on its own** — a caller that feeds it
examples of arbitrary raw length (not pre-aligned) can still produce a non-tile-aligned
batch and hit this exact crash. The alignment has to happen upstream of the collate
function, at the example-construction stage, for every caller of `SFTTrainer` in this
codebase or any other.

## Environment

- `ttml` (tt-train), installed from `/home/ttuser/tt-metal/tt-train/sources/ttml`
- `ttml.trainers.SFTTrainer`, `ttml.datasets.{InMemoryDataloader,sft_collate_fn}`
- Dense Llama-family model via `train.model.create_model`
- Single-chip `(1,1)` device mesh

## Suggested fix (for whoever files this upstream)

Either (a) have the SDPA backward validator compute `expected_rows` from the
tile-padded sequence length (matching whatever upstream buffer already uses it), or
(b) have `sft_collate_fn` itself round its per-batch padding up to the next multiple of
32 so every caller gets this guarantee for free, rather than requiring every caller to
independently pre-align its own examples (as this repo now does).
