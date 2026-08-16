<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Upstream asks for tt-metal / tt-train

Things this project needs from tt-metal that it cannot fix in its own tree, written up so
that someone with commit rights can act on them without rediscovering the analysis. Each entry
states the defect, the fix, the measurement that justifies it, and what we did instead in the
meantime.

Against tt-metal `620793d898` (`rollback-pre-qwen36-1576-g620793d898`).

---

## 1. Python cannot pass a null attention mask to `CppLlama` / `GPT2Transformer`

**Status:** open. Worked around in `train/model.py`; see
`.superpowers/attention-mask-fix.md` for the full report and every measurement.

### The defect

`ttml::ops::scaled_dot_product_attention` selects the fused SDPA kernel's mask mode from
*whether a mask object was passed*, not from what is in it
(`tt-train/sources/ttml/ops/scaled_dot_product_attention.cpp:249-255`):

```cpp
ttml::metal::AttentionMaskType mask_type = ttml::metal::AttentionMaskType::Causal;
if (mask.has_value() && mask.value()) {
    mask_tensor = mask.value()->get_value();
    mask_type = ttml::metal::AttentionMaskType::Arbitrary;
}
```

`Arbitrary` does the full S×S attention instead of the triangular half, and additionally loses
the forward program factory's load-balancing path, which is gated on `Causal`
(`tt-train/sources/ttml/metal/ops/sdpa_fw/device/sdpa_fw_program_factory.cpp:305-316`):

```cpp
const bool use_balanced_parallelism =
    (mask_type == AttentionMaskType::Causal) && (St % 2 == 0) && ...
```

A lower-triangular all-ones mask — which is what `ttml.common.utils.build_causal_mask` returns
and what `ttml.common.trainer.train()` passes on every step
(`ttml/common/trainer.py:73-76, 102`) — is exactly what `Causal` already computes. Passing it
buys nothing and roughly doubles the attention work.

tt-train's own example already knows this
(`tt-train/sources/examples/train/train.py:461-467`):

> DeepSeek's composite SDPA needs the mask passed explicitly; gpt2/llama/qwen3 use a fused
> SDPA that materializes its own causal mask internally, so only build it for DeepSeek.

But a caller who follows that advice against `CppLlama` gets a `TypeError`, because the
nanobind binding declares the mask non-optional
(`tt-train/sources/ttml/nanobind/nb_models.cpp:330-337`, and the same at `:237-246` for
GPT-2) — even though the C++ `Llama::operator()` behind it takes
`std::optional<TensorPtr>` (`models/llama.hpp:60-68`) and handles `std::nullopt` correctly all
the way down. There is no alternative Python entry point: all three
`ModuleBase::operator()` overloads `throw std::logic_error` in the base
(`modules/module_base.cpp:114-128`) and only the two non-optional ones are bound
(`nanobind/nb_modules.cpp:78-86`).

### The measurement

One Blackhole p300c, full training step (forward + cross-entropy + backward + AdamW),
s/1000 steps, measured through this project's training entry point over 300 steps at seed 5489:

| shape | explicit causal mask (`Arbitrary`) | `mask=None` (`Causal`) | speedup |
|---|---|---|---|
| 22M params, batch 16, seq 2048 | 503.3 | 356.7 | **1.41x** |
| 123M params, batch 64, seq 512 | 890.0 | 776.7 | **1.15x** |

Correctness, on trained weights: held-out cross-entropy moves by 4.1e-4 nats (4.122742 →
4.122333); a perturbation probe shows both paths are strictly causal (perturbing token *t*
leaves every logit before *t* bit-identical); and against an independent fp32 NumPy reference
the unmasked path's mean absolute error is fractionally *lower* than the masked path's
(0.015445 vs 0.015610, correlation 0.99996 for both).

### The fix

`<nanobind/stl/optional.h>` is already included at `nb_models.cpp:8`, so this needs no new
include. The KV-cache overload (`nb_models.cpp:339-352`) must keep its mask non-optional: there
the mask is not square and `models/llama.cpp` reads
`mask->get_value().logical_shape()[-1]` to size the cache slice.

```diff
--- a/tt-train/sources/ttml/nanobind/nb_models.cpp
+++ b/tt-train/sources/ttml/nanobind/nb_models.cpp
@@ GPT-2, around line 237
         py_gpt2.def(
             "__call__",
             [](models::gpt2::Transformer& self,
                const ttml::autograd::TensorPtr& tensor,
-               const ttml::autograd::TensorPtr& mask) {
-                return self(tensor, std::optional<ttml::autograd::TensorPtr>(mask));
+               const std::optional<ttml::autograd::TensorPtr>& mask) {
+                return self(tensor, mask);
             },
             nb::arg("tensor"),
-            nb::arg("mask"),
+            nb::arg("mask") = std::nullopt,
             "Model forward pass with causal mask.");

@@ Llama, no-KV-cache overload, around line 330
         py_llama.def(
             "__call__",
             [](models::llama::Llama& self,
                const ttml::autograd::TensorPtr& tensor,
-               const ttml::autograd::TensorPtr& mask) { return self(tensor, mask); },
+               const std::optional<ttml::autograd::TensorPtr>& mask) { return self(tensor, mask); },
             nb::arg("tensor"),
-            nb::arg("mask"),
+            nb::arg("mask") = std::nullopt,
             "Model forward pass without KV cache.");
```

A second, separable change belongs in `ttml/common/trainer.py:73-76`: stop building a causal
mask for models whose SDPA materializes its own, matching what `examples/train/train.py`
already does. That one needs the model type plumbed into `train()`'s config, which is a design
call for the maintainer rather than a mechanical edit, so it is described rather than patched
here. Without it, every caller of `ttml.common.trainer.train()` keeps paying the
arbitrary-mask cost even after the binding is fixed.

### What we did instead

`train/model.py` trains ttml's *Python* `Llama` (`ttml.models.llama.Llama`) rather than
`CppLlama`. It is the same architecture over the same fused ops, costs the same per step
(521.7 vs 521.9 s/1000 with the mask still passed), and its `forward` reaches
`ttml.ops.attention.scaled_dot_product_attention`, whose binding is already declared
`nb::arg("mask") = std::nullopt` (`nanobind/nb_ops.cpp:280-293`). The wrapper renames its
parameters to the C++ scheme so checkpoints, HF conversion and `--resume` are unaffected.

**Once the binding is fixed upstream, `train/model.py`'s renaming layer can go away and
`--model-impl cpp` becomes as fast as `python`.**
