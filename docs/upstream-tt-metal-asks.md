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

---

## 2. A mesh graph descriptor whose declared dims disagree with the opened `MeshShape` hangs instead of failing

**Status:** open. Worked around in this repo by shipping matching descriptors
(`train/configs/mesh/mesh-1x2.textproto`, `mesh-1x4.textproto`); see
`.superpowers/ddp-bringup.md` for the full experiment and every measurement.

**Nothing we need is blocked on this** — the workaround is complete and costs nothing at
runtime. This is a diagnosability ask, filed because the failure mode is expensive to debug and
will be hit again by anyone bringing up multi-chip training on a box whose physical topology is
not a line.

### The defect

`tt::tt_fabric` accepts a mesh graph descriptor whose `device_topology.dims` differ from the
`MeshShape` the process actually opens, and gives no indication that anything is wrong. Device
open succeeds, the parallelism context initialises and self-reports correctly, the model builds,
the batch shards, and forward/backward/optimizer all run at full speed. The run then **hangs
forever** the first time a CCL collective traverses the mesh axis.

Concretely, on a TT-QuietBox 2 (four Blackhole p300c, physically a 2x2 ring), with tt-metal's
own `tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto` — which
declares `device_topology { dims: [ 2, 2 ] }` and is the correct descriptor for the hardware —
and a process opening `MeshShape([1, 4])`:

```
step 1: batch / forward / backward / loss=10.60938 / SYNC OK / optim.step  (0.3s)
step 2: batch / forward / backward
                                     <- never returns
```

`[1, 4]` is not an exotic request here: `ttml::autograd::ParallelismContext`'s constructor
*requires* a line topology for a DDP-only run — a 2-D mesh `TT_FATAL`s unless the number of
enabled parallelisms equals the number of mesh dimensions
(`tt-train/sources/ttml/autograd/auto_context.cpp:198-204`) — so any DDP-only training job on
this class of hardware must open `[1, N]` while the shipped descriptor declares `[2, 2]`.

Two things make this particularly costly to diagnose:

1. **The host does not block in the collective.** tt-metal enqueues asynchronously, so
   `synchronize_gradients` returns having only queued work and the host stalls at its next
   blocking read — `loss.to_numpy()` in the *following* step. The stack points at the loss read,
   one phase and one iteration away from the actual fault.
2. **Everything that could have reported the mismatch reports success.** `open_device` returns a
   `MeshDevice` of shape `[1, 4]`; `ParallelismContext` inspects that mesh and correctly reports
   a DDP axis of 4 devices. There is no point at which the two shapes are compared.

### The measurement

Four Blackhole p300c, `--size 1024` (123M params), batch 64, seq 512, four training steps,
identical in every respect except the descriptor's declared dims:

| descriptor `device_topology.dims` | opened `MeshShape` | result |
|---|---|---|
| *(none — `enable_fabric` has no default for 4 devices)* | `[1, 4]` | hang before device open; killed at 600 s, no output |
| `[ 2, 2 ]` (tt-metal's `p300_x2`) | `[1, 4]` | opens, trains step 1, **hangs in step 2 forever** |
| `[ 1, 4 ]` (ours) | `[1, 4]` | **works**; 300 steps at 193.4 s/1000, all four replicas bit-identical |
| `[ 1, 2 ]` (tt-metal's `p300`) | `[1, 2]` | works; replicas bit-identical |
| `[ 2, 2 ]` (tt-metal's `p300_x2`) | `[1, 2]` | fails **cleanly** in 10 s: `Fabric Router Sync: Timeout ... on Device 2` |

The last row is the useful contrast: when the descriptor declares *more devices* than are
opened, the failure is caught and the message is accurate. It is only the *shape* disagreement,
at equal device count, that goes undetected.

### The fix

A single equality check where the mesh device is created against the fabric's active mesh
descriptor: if the descriptor's `device_topology.dims` do not match the requested `MeshShape`,
`TT_FATAL` with both shapes named. The error text should say which file is in force (the
descriptor path is already known — `get_mgd_path` sets `TT_MESH_GRAPH_DESC_PATH` when it picks a
default) and that the descriptor must declare the logical mesh being opened, not the physical
cabling.

A second, smaller ask in the same area: `ttml::ttnn_fixed::distributed::enable_fabric` has no
default descriptor for 4 devices (`tt-train/sources/ttml/ttnn_fixed/distributed/tt_metal.cpp:80-88`
handles 8 and 32 only) and silently falls back to a bare `FABRIC_2D` that hangs. Either ship a
4-device default or make the `std::nullopt` path refuse rather than proceed — a fallback that
reliably hangs is worse than an error.

### What we did instead

`train/configs/mesh/mesh-1x2.textproto` and `mesh-1x4.textproto` are vendored in this repo and
selected by device count in `train/run.py`'s `_mesh_graph_descriptor_path`, which exports
`TT_MESH_GRAPH_DESC_PATH` before ttml is imported. `tests/test_run_validation.py` asserts each
file declares the `[1, N]` shape its device count opens, and that an unsupported device count
raises rather than falling back to a mismatched descriptor — because a wrong descriptor hangs
rather than failing, a fallback would be the worst possible default.

---

## 3. A DDP training step re-marks replicated parameters as `Shard(0)`, so checkpoints save every replica

**Status:** open upstream, but **no longer blocking here** — corrected 2026-08-16 from this
repo, in `train/checkpoint.py:replicated_for_save`. See `.superpowers/ddp-checkpoint-fix.md`.

**This entry previously claimed this could not be fixed in our own tree. That was wrong**, and
the correction is the useful part of this update: `ttnn.Tensor.update_tensor_topology` is bound
in Python (`ttnn/cpp/ttnn-nanobind/pytensor.cpp:1611`) and `ttnn.TensorTopology` is constructible
from Python (`ttnn/core/distributed/distributed_nanobind.cpp:734`). The false placement can
therefore be corrected by *any* holder of the tensor, not only where it is written. This repo now
re-marks each parameter `Replicate` immediately before a save and restores the original topology
immediately after, which moves no data at all — a `--ddp 4` save costs exactly what a `--ddp 1`
save costs, and produces a byte-identical-sized file (737,824,624 bytes, verified equal to the
`--ddp 1` figure below).

The defect below is still real and still worth fixing at source: every consumer of a DDP-trained
parameter's topology is currently told something false, and each one has to know to disbelieve
it. What has changed is only that we are no longer waiting on it.

### The defect

Under DDP the weights are replicated and **stay** replicated — verified directly, not assumed:
after training steps on a `[1, 4]` mesh, every chip's copy of every one of the 66 parameter
tensors is bit-identical (`max |replica0 - replica_i| = 0.000000e+00`). The *data* is correct.

The tensor's **topology metadata** is not. Probed on the same parameter
(`llama/llama_block_0/attention/q_linear/weight`, logical shape `[1, 1, 1024, 1024]`) before
and after two DDP training steps:

| when | `Sharding.placements` | `dist_shape` | `is_fully_replicated` | `gather()` returns |
|---|---|---|---|---|
| freshly built model | `[PlacementReplicate()]` | `[4]` | `True` | `(1, 1, 1024, 1024)` |
| after 2 DDP steps | `[PlacementShard(0)]` | `[4]` | `False` | `(4, 1, 1024, 1024)` |

Something in the step — the gradient all-reduce in
`core/distributed/distributed.cpp`, or the output tensor of the fused AdamW kernel — writes back
a parameter whose recorded placement is `Shard(0)` on the DDP axis, even though the value it
wrote is identical on every device.

`ttml.checkpointing.save_checkpoint` then does exactly what the metadata says
(`ttml/checkpointing.py:169`, `Sharding.from_tensor(tensor).gather(tensor)`): a `Shard` axis is
concatenated along its sharded dim. Every saved tensor gains a leading dimension of 4 holding
four identical copies. Measured on this project's 1024 size (123M params), same run, same step
count, differing only in `--ddp`:

| run | checkpoint size |
|---|---|
| `--ddp 1` | 737,824,624 bytes |
| `--ddp 4` | 1,475,602,288 bytes |

`Sharding.gather` is **not** the bug — given `Replicate` it correctly takes a single copy, which
is what the "freshly built model" row shows. It is faithfully honouring wrong metadata.

### Why this matters beyond file size

The resulting checkpoint is wrong in a way that reads as plausible. Every parameter name is
correct and every value is correct; only the shape has an extra leading axis. This project's
`convert/checkpoint_reader.py`, `convert/hf_mapping.py` and `convert/ttml_forward.py` all match
on literal parameter names and assume whole `[1, 1, out, in]` tensors, so the error surfaces (if
at all) far from its cause, during HF conversion or parity checking.

### The fix

Preserve the placement when writing an updated parameter value back. A parameter that was
`Replicate` on a mesh axis before the optimizer step is still `Replicate` after it — the
all-reduce exists precisely to guarantee that. Wherever the post-step tensor is constructed, it
should inherit the parameter's existing `tensor_topology()` rather than defaulting to a sharded
placement.

Failing that, `synchronize_gradients` (which already knows each parameter's placement — it calls
`is_sharded_on_axis` to decide which axes to reduce over,
`core/distributed/distributed.cpp:43-52`) is a natural place to restore it.

### What we did instead

`train/checkpoint.py:replicated_for_save` rewrites each offending parameter's `TensorTopology`
to an otherwise-identical one whose placements are all `Replicate`, runs ttml's saver, and
restores the original topology in a `finally`. ttml's `Sharding.gather` then takes a single copy,
which is what `--ddp 1` writes and what `convert/` expects. Verified on hardware: after 50 DDP
steps, every tensor in the written file is **bitwise equal (max abs difference 0.000000e+00)** to
replica 0 read independently through a `concat_mesh_to_tensor_composer`, which does not consult
the placements being corrected. The restore is what makes this safe to run mid-run at a
`--save-every` boundary: afterwards all 66 parameters carry the topology they carried before, so
a run that checkpoints is the same computation as one that does not.

`assert_saveable_on_mesh` remains a real gate, narrowed rather than removed. It refuses unless
**both** (a) ttml's live parallelism context is DDP and only DDP — under TP a `Shard` placement
may be the truth, and re-marking it would write a quarter of a model — and (b) the tensor is
distributed over exactly the DDP axis. Both conditions fail closed; any failure to read the
parallelism context is a reason to refuse, never permission.

---

## 4. Stochastic rounding under DDP breaks the replica-identity invariant

**Status:** open. Not blocking — a `--ddp N` checkpoint records replica 0, which is a complete
and coherent model — but it means "the model" a multi-chip run produces is one of N
non-identical models rather than the single model DDP is supposed to maintain. Measured
2026-08-16; see `.superpowers/ddp-checkpoint-fix.md` §4.

### The defect

Data parallelism's defining invariant is that every replica holds the *same* parameters: the
gradient all-reduce exists precisely to guarantee it. `stochastic_rounding: true` breaks it. Each
device's AdamW chooses its rounding direction from its own RNG, so four replicas that receive a
bit-identical all-reduced gradient still write four different bfloat16 values, and thereafter
perform independent random walks about a common trajectory.

Two `--size 1024` DDP runs on a `[1, 4]` mesh, four steps, identical in every respect except one
optimizer flag:

| `stochastic_rounding` | parameters whose replicas differ | `max |replica0 - replica_i|` |
|---|---|---|
| `false` | **0 / 66** | **0.000000e+00** |
| `true` | **66 / 66** | **2.343750e-02** (`llama/llama_block_5/mlp_norm/gamma`) |

At 50 steps the divergence is 3.125e-02. The RMSNorm gammas are the loudest because they sit at
1.0, where one bfloat16 ulp (0.0078) is an order of magnitude larger than the ~3e-4 update they
receive — which is the same arithmetic that made `stochastic_rounding` necessary in the first
place (see `train/configs/nanollama3_bpe_v2.yaml`). Stochastic rounding is the right fix for
that; it simply was not made DDP-aware.

### Why it matters

1. **It is silent.** Nothing reports it, the loss curve is unaffected, and the natural
   verification — "check the replicas are bit-identical" — is the one this breaks. A checkpoint
   must therefore *choose* a replica, and nothing records which.
2. **It defeats the obvious form of a save-time guard.** This repo wanted to gate its topology
   correction on "the replicas agree, so taking one copy is safe". That gate passes under the
   default config and refuses under the project's recommended one, so the guard had to be built
   on structural facts about the parallelism context instead.

### The fix

Draw the rounding decisions identically across the DDP axis: seed the stochastic-rounding RNG
from a value shared along that axis (the ParallelismContext already knows the axis and the
device's index on it), rather than per-device. A parameter that starts replicated then stays
replicated exactly, and DDP's invariant survives a feature that is otherwise strictly good.

### What we did instead

Nothing that changes the training: this is reported, not worked around. `replicated_for_save`
writes replica 0, and `.superpowers/ddp-checkpoint-fix.md` records that this is a choice among N
rather than a distinction without a difference.
