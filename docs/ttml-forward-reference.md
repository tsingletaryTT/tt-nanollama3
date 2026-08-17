# ttml's Llama forward pass — derived from source

Purpose. This is the specification a pure-NumPy reimplementation of ttml's Llama forward
pass is written against, so that the NumPy model can be run on the raw checkpoint tensors and
its logits compared against the Hugging Face conversion at a tight tolerance.

Independence. Everything below is derived from ttml's C++ (and the ttnn kernels it calls),
plus the checkpoint's own manifest. Nothing here was taken from `convert/hf_mapping.py` or
`convert/to_hf.py` — those files encode our converter's *interpretation* of these same
conventions, and deriving from them would make the eventual parity check circular. Plan 4
shipped a wrong RoPE row layout that survived four independent checks precisely because the
convention was never written down and cross-checked against the source; this document is the
remedy.

`convert/checkpoint_reader.py` *was* used, but only to stream raw tensor names, shapes and
dtypes out of the checkpoint pickle — it contains no model semantics.

Source root. All `ttml/...` citations below are relative to
`$TT_METAL_HOME/tt-train/sources/ttml/` (here: `/home/ttuser/tt-metal/tt-train/sources/ttml/`).
`ttnn/...` citations are relative to `$TT_METAL_HOME/`.

---

## 0. The model under test

Confirmed from the checkpoint header (`artifacts/checkpoints/nanollama3_step00003000.pkl`,
record 0) and from `ttml/models/llama.hpp:24-40`:

| field | value | note |
|---|---|---|
| `embedding_dim` | 384 | |
| `num_blocks` | 6 | |
| `num_heads` (H) | 6 | |
| `num_groups` (G) | 3 | GQA; `heads_per_group = H/G = 2` |
| `head_dim` (Dh) | 64 | `embedding_dim / num_heads`, `ttml/models/llama.cpp:160` |
| `max_sequence_length` | 256 | |
| `theta` | 500000.0 | |
| `vocab_size` | 32000 | not padded — see §7 |
| `intermediate_dim` | 1024 | explicit in config, so the `4d·2/3` rounding at `ttml/modules/llama_block.cpp:22-23` is **not** used |
| `dropout_prob` | 0.0 | so dropout is identity in both train and eval |
| `rms_norm_eps` | 1e-5 | C++ default, `ttml/modules/rms_norm_module.hpp:17,23`. **The header field is descriptive, not authoritative** — `LlamaBlock` constructs `RMSNormLayer(embedding_size)` with the default argument (`ttml/modules/llama_block.cpp:47-48`) and nothing plumbs a YAML value through, so the value is fixed by the C++ regardless of what the header says. |
| `weight_tying` | Enabled | |
| weights dtype | bfloat16 | |

The model is the **C++** `ttml::models::llama::Llama`, not a Python re-implementation:
`ttml/ttml/common/model_factory.py:153` calls `ttml.models.llama.create_cpp_llama_model(lcfg)`,
which is `ttml/nanobind/nb_models.cpp:142-143` → `ttml::models::llama::create`. The checkpoint's
tensor names (`llama/llama_block_0/attention/kv_linear/weight`, …) match the C++ module tree
exactly, which corroborates this.

RoPE scaling is **off**: `LlamaConfig::scaling_factor` defaults to `0.0F`
(`ttml/models/llama.hpp:36`), the scaling block at `ttml/models/llama.cpp:145-156` is therefore
skipped, `RopeScalingParams::scaling_factor` keeps its `1.0F` default
(`ttml/ops/rope_op.hpp:19`), and `gen_freqs` skips `apply_rope_scaling` at
`ttml/ops/rope_op.cpp:208`. The checkpoint header carries no `rope_scaling` key, consistent
with this.

### Tensor / weight layout convention

`LinearLayer` stores its weight as `[1, 1, out_features, in_features]`
(`ttml/modules/linear_module.cpp:19-25`) — i.e. **row-major `[out, in]`, the same orientation as
PyTorch `nn.Linear.weight`**. So every linear in this model is `y = x @ W.T`. The checkpoint
shapes confirm it:

```
llama/fc/weight                                  (1, 1, 32000, 384)
llama/llama_block_0/attention/q_linear/weight    (1, 1,   384, 384)
llama/llama_block_0/attention/kv_linear/weight   (1, 1,   384, 384)   # 2·G·Dh = 2·3·64 = 384
llama/llama_block_0/attention/out_linear/weight  (1, 1,   384, 384)
llama/llama_block_0/mlp/w1/weight                (1, 1,  1024, 384)
llama/llama_block_0/mlp/w3/weight                (1, 1,  1024, 384)
llama/llama_block_0/mlp/w2/weight                (1, 1,   384, 1024)
llama/llama_block_0/attention_norm/gamma         (1, 1,     1, 384)
llama/llama_block_0/mlp_norm/gamma               (1, 1,     1, 384)
llama/ln_fc/gamma                                (1, 1,     1, 384)
```

Activations are rank-4 `[B, 1, S, C]` throughout the block stack, and `[B, H, S, Dh]` inside
attention. There are no biases anywhere: `GQAConfig.bias_linears = false`
(`ttml/modules/llama_block.cpp:55`) and the three MLP linears are constructed with
`has_bias=false` (`ttml/modules/llama_block.cpp:25-27`).

---

## 1. Block structure — confirmed

`ttml/modules/llama_block.cpp:65-78`. Pre-norm, two residuals, residual taken **before** the
norm and added to the sub-layer output:

```cpp
auto residual = input;
auto h = (*m_attention_norm)(input);
h = (*m_attention)(h, mask);
h = ops::add(h, residual);

residual = h;
auto x = (*m_mlp_norm)(h);
x = (*m_mlp)(x);
x = ops::add(x, residual);
```

Attention internals, `ttml/modules/grouped_query_attention.cpp:36-57` — also confirmed, and
note that RoPE is applied to Q and K only, never V:

```cpp
auto q  = (*m_q_linear)(x);
auto kv = (*m_kv_linear)(x);
auto [query_with_heads, key_with_heads, value_with_heads] =
    ops::grouped_heads_creation(q, kv, m_num_heads, m_num_groups);
if (m_embedding) {
    query_with_heads = (*m_embedding)(query_with_heads);   // RoPE
    key_with_heads   = (*m_embedding)(key_with_heads);     // RoPE
}
auto attention = ttml::ops::scaled_dot_product_attention(query_with_heads, key_with_heads,
                                                         value_with_heads, mask);
attention = ops::heads_fusion(attention);
auto out = (*m_out_linear)(attention);
out = (*m_dropout)(out);
```

Top level, `ttml/models/llama.cpp:207-252`: embed → `num_blocks` × `LlamaBlock` → `ln_fc`
(RMSNorm) → `fc` (linear to vocab). No scaling anywhere in between (§7).

`ttml/models/llama.cpp:188-223` pads the token sequence up to a multiple of 32 before the
embedding lookup and slices back afterwards. For S = 256 this is a no-op.

---

## 2. RMSNorm

Citations. `ttml/modules/rms_norm_module.cpp:24-30`, `ttml/ops/rmsnorm_op.cpp:26-67`
(fused, the default) and `ttml/ops/rmsnorm_op.cpp:69-125` (composite reference), plus the
device kernel `ttml/metal/ops/rmsnorm_fw/device/kernels/compute/rmsnorm_fw_kernel.cpp:288-372`
and its program factory `.../rmsnorm_fw_program_factory.cpp:154-157`.

`RMSNormLayer` uses the **fused** path (`m_use_composite = false`,
`ttml/modules/rms_norm_module.hpp:18`; the dispatch is at `rms_norm_module.cpp:25-29`), so the
kernel is the authority. Reading it:

- `calculate_sum_x_squared()` (kernel lines 52-93) accumulates `x²` across the C axis.
- The row reduction (kernel lines 296-306) uses `cb_scaler`, which the program factory packs as
  `1/num_inner` where `num_inner = input.logical_shape()[-1] = C`
  (`rmsnorm_fw_program_factory.cpp:154,157`). So the reduction yields **mean(x²)**, not sum.
- `add_binary_tile(reduction_register, eps_register, ...)` then `sqrt_tile(...)`
  (kernel lines 308-318): **epsilon is added to the mean of squares, inside the sqrt.**
- `calculate_input_multiplied_by_gamma_and_divided_by_rms()` (kernel lines 150-203) computes
  `x * gamma` broadcast over rows, then multiplies by `1/rms` broadcast over columns.

The composite path agrees exactly (`rmsnorm_op.cpp:85-123`: `square → mean(-1) → +eps → sqrt`,
then `gamma * x / rms`).

### Formula

```
rms(x) = sqrt( mean_c(x_c²) + eps )
y_c    = gamma_c * x_c / rms(x)
```

with `eps = 1e-5`, mean over the last axis (C = 384), `gamma` of shape `[C]`.

Where gamma sits: the kernel multiplies `gamma` into `x` *before* dividing by `rms`, but
both are elementwise scalar operations on the same element, so `gamma·x/rms == gamma·(x/rms)`
— mathematically it does not matter. What *does* matter and is unambiguous: the weight is
plain `gamma`, **not** `1 + gamma` (no unit offset anywhere;
`ttml/modules/rms_norm_module.cpp:13` initialises gamma to ones, and nothing adds one at use
time).

Confidence: high. Two independent implementations in the tree (kernel + composite) agree.

---

## 3. RoPE — **interleaved**, `(x[2i], x[2i+1])`

This is the item that broke Plan 4, so it is established four ways below. They are not four
independent derivations — §3.1 and §3.2/§3.3 read the same op from different angles — but they
are four *mutually consistent lines of evidence* of different kinds: a numerical one (the
frequency table), a structural one (the rotation matrix and the kernel that applies it), an
architectural one (a tiling constraint that makes the alternative impossible), and an
intentional one (ttml's own importer converting the other convention into this one).

Citations. `ttml/ops/rope_op.cpp:191-235` (`gen_freqs`), `:237-248` (`gen_trans_mat`),
`:250-299` (`build_rope_params`), `:112-189` (`rope`), and the ttnn kernel that actually does
the arithmetic,
`ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/compute/rotary_embedding_llama.cpp:90-138`.

### 3.1 The frequency table

`gen_freqs`, `ttml/ops/rope_op.cpp:198-205`:

```cpp
xt::xarray<uint32_t> pair_idx_u = xt::arange<uint32_t>(head_dim) / 2u;  // 0,0,1,1,2,2,...
xt::xarray<float>    pair_idx   = xt::cast<float>(pair_idx_u);
pair_idx *= 2.0F / static_cast<float>(head_dim);        // 2·floor(i/2)/Dh
xt::xarray<float>    inv_freq   = 1.0f / xt::pow(theta, pair_idx);
```

So for channel index `i ∈ [0, Dh)`:

```
inv_freq[i] = theta ^ ( -2·floor(i/2) / Dh )
```

which means `inv_freq[2j] == inv_freq[2j+1] == theta^(-2j/Dh)`. **Adjacent channels share a
frequency**, whereas a split-halves layout would give `inv_freq[j] == inv_freq[j + Dh/2]`.

This is a *necessary* condition for interleaved pairing and is flatly inconsistent with
split-halves, but on its own it is not sufficient — it says which channels share a frequency,
not which channels get rotated into each other. The pairing itself is fixed by §3.2 + §3.3.

Then, `ttml/ops/rope_op.cpp:214-229`:

```cpp
pos        = arange(sequence_length)                 // [L, 1]
theta_mat  = pos * inv_freq                          // [L, Dh], outer product
theta_mat  = fmod(theta_mat, 2π)                     // principal range
sin_freqs  = sin(theta_mat);  cos_freqs = cos(theta_mat)
```

Caches are `[1, 1, L, Dh]` (validated at `ttml/ops/rope_op.cpp:46`) — full head-dim width, with
each frequency duplicated across its adjacent channel pair.

### 3.2 The rotation matrix

`gen_trans_mat`, `ttml/ops/rope_op.cpp:237-248`, builds a 32×32 (`ttnn::TILE_SIZE`) matrix:

```cpp
for (int i = 0; i < TILE_SIZE; i += 2)  trans_mat(0,0,i,   i+1) =  1.0F;
for (int j = 1; j < TILE_SIZE; j += 2)  trans_mat(0,0,j,   j-1) = -1.0F;
```

This is block-diagonal in 2×2 blocks of `[[0, 1], [-1, 0]]`. Applied as `rot = x @ T`
(row-vector convention, which is what `matmul_tiles(in_cb, trans_mat_cb, ...)` does):

```
rot[2j]   = Σ_k x[k]·T[k, 2j]   = x[2j+1]·(-1) = -x[2j+1]
rot[2j+1] = Σ_k x[k]·T[k, 2j+1] = x[2j]·(+1)   =  x[2j]
```

The 2×2 blocks land on *adjacent* channel pairs. Because `TILE_SIZE = 32` is even and
`Dh = 64 = 2 tiles`, the same 32×32 matrix tiles across the head dim without disturbing any
pair. This fixes the pairing as `(2j, 2j+1)`.

#### The tiling makes split-halves structurally impossible

`rotary_embedding_llama.cpp:93-96` applies that one 32×32 tile to each head-dim tile
independently:

```cpp
for (uint32_t j = 0; j < Wt; ++j) {
    matmul_tiles(in_cb, trans_mat_cb, j, in1_index, j);   // in1_index == 0, never reassigned (:65)
    pack_tile(j, rotated_in_interm_cb, j);
}
```

`in1_index` is initialised to `0` at `:65` and never modified, so every one of the `Wt`
head-dim tiles is multiplied by the *same* single `trans_mat` tile, and tile `j`'s output
depends only on tile `j`'s input. With `Dh = 64`, `Wt = 2`.

Split-halves pairing would require mixing column `i` with column `i + Dh/2 = i + 32` — i.e.
tile 0 with tile 1. **A per-tile 32×32 matmul cannot express that**, whatever the matrix
contains: there is no data path between the two tiles in this op. So the op is *architecturally
incapable* of split-halves RoPE at this head dim.

This is the strongest of the four arguments, because it does not depend on reading the
frequency table correctly, on my sign conventions for `x @ T`, or on anyone's intent — it is a
constraint on what the kernel can compute at all.

### 3.3 The kernel that combines them

`rotary_embedding_llama.cpp:90-138` — the only arithmetic in the op:

```
rotated    = x @ trans_mat        (line 94)
sin_interm = rotated * sin        (line 105)
cos_interm = x * cos              (line 115)
out        = cos_interm + sin_interm   (line 132)
```

i.e. `out = x·cos + (x @ T)·sin`. Substituting §3.2:

```
out[2j]   = x[2j]·cos_j   − x[2j+1]·sin_j
out[2j+1] = x[2j+1]·cos_j + x[2j]·sin_j
```

with `cos_j = cos(p · theta^(−2j/Dh))`, `sin_j = sin(p · theta^(−2j/Dh))`, `p` = absolute
position. That is exactly the standard 2-D rotation applied to the pair `(x[2j], x[2j+1])`.

### 3.4 Confirmation of intent — ttml's own HF importer

ttml's safetensors loader permutes HF `q_proj`/`k_proj` rows on the way in
(`ttml/models/llama.cpp:64-91`, called at `:551` for Q with `n_heads`, at `:569` for K with
`num_groups`):

```cpp
// Reorder rows within each head: [0..D/2-1, D/2..D-1] → interleave → [0, D/2, 1, D/2+1, ...]
const int64_t src_even = head_row0 + i;          const int64_t dst_even = head_row0 + (2 * i);
const int64_t src_odd  = head_row0 + half + i;   const int64_t dst_odd  = head_row0 + (2 * i + 1);
```

So ttml row `2i` is HF row `i`, and ttml row `2i+1` is HF row `i + Dh/2` — the two rows HF pairs
under *its* split-halves convention at frequency index `i`. ttml deliberately converts
split-halves → interleaved on import. `v_proj` gets **no** permutation
(`ttml/models/llama.cpp:577-585`), consistent with RoPE never touching V.

### 3.5 Verdict

> **ttml pairs `(x[2i], x[2i+1])` — interleaved (GPT-NeoX-style adjacent pairs, i.e. the
> "Meta/original Llama" layout), *not* split-halves `(x[i], x[i+Dh/2])` (the HF
> `LlamaAttention` layout).**
>
> **Confidence: very high.** Four mutually consistent lines of evidence: the frequency table's
> `floor(i/2)` (§3.1, necessary-not-sufficient); the 2×2 block-diagonal `trans_mat` combined
> with the ttnn kernel's `x·cos + (x@T)·sin` (§3.2-3.3, which fixes the pairing); the per-tile
> `matmul_tiles` loop that makes split-halves structurally impossible at `Dh = 64` (§3.3, the
> strongest); and ttml's own HF importer explicitly un-permuting split-halves into interleaved
> (§3.4, confirming intent). §8.1 then confirms it end-to-end: the split-halves variant costs
> 1.28 nats (≈12 SE).

Consequently, the pairing acts on the rows of `q_linear`/`kv_linear`'s weight **within each
head**: for query head `h`, RoPE pairs weight rows `h·Dh + 2j` and `h·Dh + 2j + 1`.

### 3.6 Why four caches, not two

`build_rope_params`, `ttml/ops/rope_op.cpp:287-292`:

```cpp
.cos_cache     = cos_freqs,
.sin_cache     = sin_freqs,
.neg_cos_cache = cos_freqs,             // cos(θ) = cos(−θ): symmetry over x-axis
.neg_sin_cache = ttnn::neg(sin_freqs),  // sin(−θ) = −sin(θ)
```

The `neg_*` pair is **backward-pass only**. RoPE is orthogonal, so its vector-Jacobian product
is the inverse rotation, i.e. a rotation by `−θ`; the grad function at
`ttml/ops/rope_op.cpp:170-184` re-runs the *same* `rotary_embedding_llama` op on the incoming
gradient with `neg_cos_cache`/`neg_sin_cache` substituted, reusing `trans_mat` unchanged (the
comment at `:166-169` says exactly this). The four caches therefore imply nothing about the
forward pairing — they are a precompute-instead-of-negate-at-runtime optimisation for training.

A pure forward reference needs only `cos_cache` and `sin_cache`.

### 3.7 Position offset

`rope(input, params, token_position)` slices the caches starting at `token_position`
(`ttml/ops/rope_op.cpp:141-151`) for decode with a KV cache. In the no-cache path
`RotaryEmbedding::operator()` passes `0` (`ttml/modules/rotary_embedding.cpp:15-17`), so
positions are simply `0 … S−1`. A full-sequence reference uses `p = 0 … S−1`.

### 3.8 `apply_rope_scaling`

Present at `ttml/ops/rope_op.cpp:67-108`, invoked only when
`RopeScalingParams::scaling_factor != 1.0f` (`:208`). As established in §0 it is **not** active
for this checkpoint. A reference implementation should omit it, and should assert that the
checkpoint records no scaling params rather than silently ignoring them.

---

## 4. `grouped_heads_creation` — K precedes V, head-major, plain reshape

Citations. `ttml/ops/multi_head_utils.cpp:88-130`, which calls
`ttnn::experimental::nlp_create_qkv_heads(qs, kvs, num_q_heads=H, num_kv_heads=G, transpose_k_heads=false, …)`.
The semantics come from
`ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads.cpp:20-32`
(shape contract) and the dataflow kernels
`.../device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads.cpp:49-94` and
`.../writer_tm_tile_layout_nlp_create_qkv_heads.cpp:61-138`, plus the program factory's own
comment at `.../device/nlp_create_qkv_heads_program_factory.cpp:58-60`:

> Output shape for Q is: `[B, num_q_heads, s, head_dim]`, shuffled from `[B, 1, s, num_q_heads * head_dim]`

### 4.1 Q

The writer walks `c_dim = 0 … num_q_heads−1` (outer) × `w_dim` over the head-dim tiles (inner),
consuming the input row's tiles strictly in order (`writer:64-82`). So query head `h` is the
`h`-th contiguous `Dh`-wide slice of the input row:

```
q_heads[h, s, d] = q[s, h·Dh + d]
```

which is exactly `q.reshape(S, H, Dh).transpose(→ H, S, Dh)`.

### 4.2 The fused KV split

The reader (`reader:49-94`) processes each row-block in the order **Q tiles → K tiles → V
tiles**, and when a separate `kv` tensor is supplied (`READ_FROM_INPUT_TENSOR_KV`, which is our
case) it reads K and V from consecutive page ids of that one tensor: first `kv_num_tiles`
(= `G · Dh/32` tiles = `G·Dh` columns), then the next `kv_num_tiles`.

> **K occupies the first `G·Dh` columns of the fused `kv` tensor; V occupies the last `G·Dh`.
> K precedes V.**

Within each half, groups are head-major and contiguous, identically to Q. For this model
(`G = 3`, `Dh = 64`, fused width 384):

```
k_heads[g, s, d] = kv[s,          g·64 + d]    # columns   0 … 191
v_heads[g, s, d] = kv[s, 192 +    g·64 + d]    # columns 192 … 383
```

Because `kv_linear.weight` is `[out, in]` (§0), the same split applies to its **rows**: rows
0–191 are the K projection, rows 192–383 the V projection.

Corroborated by the backward pass, which reassembles the gradient as
`ttnn::concat({grad_k, grad_v}, dim=3)` — K first (`ttml/ops/multi_head_utils.cpp:116`) — and by
ttml's HF importer, which builds `kv_linear` by row-concatenating K then V
(`ttml/models/llama.cpp:419-423`).

### 4.3 `heads_fusion` — the inverse

`ttml/ops/multi_head_utils.cpp:60-86` calls `nlp_concat_heads`, whose reader
(`ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads.cpp:43-62`)
walks `c_dim` outer / `w_dim` inner and writes contiguously, giving
`out[s, h·Dh + d] = x[h, s, d]` — head-major concat, the exact inverse of §4.1. Independently
confirmed by the gradient at `ttml/ops/multi_head_utils.cpp:75-79`, which is
`transpose(-2,-1) → reshape(B,H,E/H,S) → transpose(-2,-1)`, the transpose-of-reshape inverse.

Confidence: high for the K-before-V ordering and the head-major layout.

---

## 5. `scaled_dot_product_attention`

Citations. `ttml/ops/scaled_dot_product_attention.cpp:241-300` (the production entry point,
which dispatches to the fused metal kernel) and `:140-239` (`_composite`, a second
implementation kept in-tree). The fused kernel's scale is set in
`ttml/metal/ops/sdpa_fw/device/sdpa_fw_program_factory.cpp:271,293-294`; the softmax and
masking live in
`ttml/metal/ops/sdpa_fw/device/kernels/compute/sdpa_fw_compute_kernel.cpp:116-238` and
`.../sdpa_compute_utils.hpp:160-200`; the causal mask tiles are generated in
`.../kernels/dataflow/sdpa_fw_writer_kernel.cpp:49-61`.

### 5.1 Scale factor

`sdpa_fw_program_factory.cpp:293-294`:

```cpp
const uint32_t scaler = std::bit_cast<uint32_t>(1.0F / std::sqrt(static_cast<float>(qEmbd)));
```

where `qEmbd = query.padded_shape()[3]` (`:271`) — the **head dimension**, not the model
dimension. The composite path uses the same value from the logical shape
(`scaled_dot_product_attention.cpp:150`). Here `Dh = 64` is tile-aligned so padded == logical == 64.

```
scale = 1 / sqrt(head_dim) = 1/8
```

### 5.2 Causal mask — additive, large negative, applied to the raw scores

This checkpoint took the `Arbitrary` path, not the built-in `Causal` one. The driver is
`train/run.py:150` → `ttml.common.trainer.train()`, and that function builds an explicit mask:

```python
causal_mask = build_causal_mask(cfg.seq_len)                    # trainer.py:73
tt_mask = ttml.autograd.Tensor.from_numpy(
    causal_mask, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)      # trainer.py:74-76  [1,1,T,T] bf16
...
logits = model(tt_x, tt_mask)                                   # trainer.py:102
```

with `build_causal_mask` = `np.tril(np.ones((1, 1, T, T), dtype=np.float32))`, **1 = attend**
(`ttml/ttml/common/utils.py:160-169`). tt-tnt's own validation loop does the same
(`train/run.py:79-87`). Because a mask tensor is present, `scaled_dot_product_attention` sets
`mask_type = AttentionMaskType::Arbitrary` (`scaled_dot_product_attention.cpp:252-255`), so the
kernel's `USE_ATTN_MASK` branch runs and its built-in `CAUSAL_MASK` branch does **not**.

*(An earlier draft of this document justified the same conclusion from
`nano_gpt/main.cpp:538`, the C++ example driver, which does leave `masks_tensor` unset. That is
not our entry point. The arithmetic below is unchanged, but the justification was wrong — and
"confident claim about which code path ran, derived from the wrong entry point" is precisely
the failure mode this document exists to prevent, so it is recorded rather than quietly
overwritten.)*

The two branches coincide numerically, which is what makes this a cross-check rather than a
correction. Both produce an **additive** `0 / −1e9`:

*Arbitrary* (`sdpa_compute_utils.hpp:67-98`), given mask `m ∈ {0, 1}` and score `s`:

```
mask_tile(s, m)            -> s·m                       (masked positions zeroed)
add_unary_tile(m, −1.0)    -> m − 1  ∈ {−1, 0}
mul_unary_tile(m, 1e9)     -> (m−1)·1e9 ∈ {−1e9, 0}
add_binary_tile(s, m)      -> s·m + (m−1)·1e9
```

so attend (`m=1`) → `s`, masked (`m=0`) → `−1e9`.

*Causal* (`sdpa_fw_writer_kernel.cpp:52-61`) pre-bakes two reusable tiles — `tile[0]` = `0.0`
on/below the diagonal and `−1e9` above, `tile[1]` = all `−1e9` for K tiles strictly past the
diagonal — which the compute kernel adds in FP32 via packer L1 accumulation
(`sdpa_fw_compute_kernel.cpp:174-190`).

On a `tril` pattern the two are the same function. The constant is the same in both:
`BF16_NEG_LARGE_BITS = 0xCE6E`, "upper 16 bits of `-1e9F` (bfloat16)"
(`ttml/metal/common/dataflow_utils.hpp:46`).

So: **additive mask, `0` for keep and `−1e9` for masked, applied to the score before the
softmax.** No multiplicative masking, no post-softmax renormalisation. A NumPy reference should
use `softmax(QK^T/sqrt(Dh) + causal_neg_inf)`.

One implementation detail worth recording: the scale is folded into the softmax exponent rather
than applied to the scores up front, and the `Arbitrary` mask path says so explicitly
(`sdpa_compute_utils.hpp:86-88`: "Scaling is NOT applied here — it is deferred to after
max-subtraction for better numerical precision"). `sdpa_compute_utils.hpp:167-174` then computes
`exp(scale · (score − rowmax))` where `score = QK^T + mask`. The kernel therefore evaluates
`softmax(scale·(QK^T + mask))`, whereas the composite path evaluates
`softmax(scale·QK^T + mask)` (`scaled_dot_product_attention.cpp:152-179`). The two differ only
in whether the effective mask constant is `−1e9` or `−1.25e8`; both are `−inf` for every
practical purpose after `exp`.

Trap worth naming: under `Arbitrary` the built-in causal pattern is *not* additionally
applied — the supplied `[1,1,S,S]` mask must itself be causal, and here it is. Anyone who
passes a non-causal mask (or an all-ones one) gets a non-causal model with no complaint from
the kernel.

### 5.3 GQA head→group mapping

The fused reader computes, at
`ttml/metal/ops/sdpa_fw/device/kernels/dataflow/sdpa_fw_reader_kernel.cpp:75-77` and `:125-130`:

```cpp
const uint32_t q_head_idx   = (global_row_idx / Ht) % q_heads;
const uint32_t kv_group_idx = q_head_idx / heads_per_group;   // heads_per_group = qNH / kNH
```

The composite path agrees: it reshapes `(B, H, S, D) → (B·G, H/G, S, D)` and broadcasts one KV
head across each `H/G` block (`scaled_dot_product_attention.cpp:59-65`).

> **Query head `h` attends to KV group `h // (H/G)`** — contiguous blocks, i.e.
> `repeat_interleave` semantics, matching HF's `repeat_kv`. For `H=6, G=3`: heads {0,1}→group 0,
> {2,3}→group 1, {4,5}→group 2. *Not* `tile`/round-robin.

### 5.4 Formula

```
scores[h, i, j] = (q[h, i, :] · k[h//2, j, :]) / sqrt(64)     for j <= i
                = -inf                                        for j >  i
attn[h, i, :]   = softmax_j(scores[h, i, :]) @ v[h//2, :, :]
```

Confidence: high for the scale and the additive causal mask; **high** for the head→group
mapping (fused and composite paths independently agree).

---

## 6. SwiGLU

Citations. `ttml/modules/llama_block.cpp:34-37` (the call site) and
`ttml/ops/swiglu_op.cpp:56-125` (fused, default) / `:41-54` (composite reference).

Call site, `llama_block.cpp:36`:

```cpp
return ops::swiglu(input, m_w1->get_weight(), m_w2->get_weight(), m_w3->get_weight(), dropout_prob);
```

— note the **argument order is `w1, w2, w3`** while the module registration order is
`w1, w3, w2` (`llama_block.cpp:29-31`). The parameter names, not the positions, are what matter.

The composite spells the expression out unambiguously (`swiglu_op.cpp:50-53`):

```cpp
const auto swished = ops::silu(ops::linear_op(tensor, w1, nullptr));
const auto gate    = ops::linear_op(tensor, w3, nullptr);
const auto x       = ops::linear_op(ops::mul(swished, gate), w2, nullptr);
```

The fused path is identical — `swiglu_op.cpp:98-102` computes `x @ w1.T`, `x @ w3.T`, then
multiplies with SiLU fused onto the *left-hand* operand (`silu_lhs` is bound to the `linear1`
side at `:96,101`), then `@ w2.T`. The shape assertions at `:81-91` pin the orientations:
`w1, w3` are `[H, D]` and `w2` is `[D, H]`, matching the checkpoint (`w1`, `w3` are
`(1024, 384)`, `w2` is `(384, 1024)`).

### Formula

```
mlp(x) = ( silu(x @ w1.T) * (x @ w3.T) ) @ w2.T
silu(z) = z * sigmoid(z) = z / (1 + exp(-z))
```

SiLU is on the `w1` branch; `w3` is the un-activated gate. In HF `LlamaMLP` terms,
`w1 = gate_proj`, `w3 = up_proj`, `w2 = down_proj` — corroborated independently by ttml's HF
importer, which maps `mlp.gate_proj → mlp/w1`, `mlp.up_proj → mlp/w3`, `mlp.down_proj → mlp/w2`
(`ttml/models/llama.cpp:602-638`).

Dropout is `0.0` for this model and is applied only to the *output* of the `w2` projection
(`swiglu_op.cpp:113-124`), so it is identity here.

Confidence: high. Two implementations agree and the checkpoint shapes are consistent with
only this assignment.

---

## 7. Embedding, output projection, weight tying

Citations. `ttml/models/llama.cpp:136-141` (construction), `:169,177` (registration),
`:207,250-252` (forward), `ttml/modules/embedding_module.cpp:38-55`,
`ttml/ops/embedding_op.cpp:16-28`, `ttml/modules/module_base.cpp:56-82` (the dedup that explains
the checkpoint's tensor set), and `ttml/models/llama.cpp:461-475` (the importer's own choice of
name under tying).

### 7.1 One tensor, two roles

```cpp
auto last_fc = std::make_shared<ttml::modules::LinearLayer>(embedding_dim, vocab_size, false);
if (config.weight_tying == WeightTyingType::Enabled) {
    tok_emb = std::make_shared<ttml::modules::Embedding>(last_fc->get_weight());
}
```

`Embedding` is constructed from the *same* `TensorPtr` as the output linear
(`embedding_module.cpp:51-55` just stores the pointer). The two layouts happen to coincide:
`LinearLayer` stores `[out, in] = [vocab_size, embedding_dim]`
(`linear_module.cpp:20`), and `Embedding` stores `[num_embeddings, embedding_dim]`
(`embedding_module.cpp:20`). **No transpose is involved in either direction.**

`ModuleBase::parameters()` dedups by tensor address specifically to handle this
(`module_base.cpp:65-81`, whose comment names weight tying as the use case), which is why the
checkpoint contains **50** tensors — 6 blocks × 8 + `llama/ln_fc/gamma` + `llama/fc/weight` —
and **no `llama/tok_emb/weight` at all**. Verified directly against the checkpoint manifest.

*Which* of the two names survives is not luck: `m_named_modules` is a `std::map`, not an
`unordered_map` (`ttml/modules/module_base.hpp:31`), and the comment immediately above it
(`:27-30`) says this is deliberate — "we need to keep order of iteration for serialization …
special case for weight tying in transformers … stored/loaded name is the same between
different runs". The BFS in `parameters()` therefore visits modules in lexicographic order,
`"fc"` sorts before `"tok_emb"`, and the pointer-dedup keeps whichever name it reaches first.
So `llama/fc/weight` is the stable, reproducible name for the tied tensor — which is also the
name ttml's own importer targets under tying (`ttml/models/llama.cpp:464-466`).

### 7.2 Both ends are unscaled

- **Embedding lookup**: `embedding_op` (`ttml/ops/embedding_op.cpp:16-28`) is `untilize` →
  `ttnn::embedding` → `reshape`. A row gather and nothing else — **no `sqrt(d_model)`
  multiplier**, no normalisation, no positional term (Llama does positions inside attention,
  `llama.cpp:225`).
- **Output projection**: `auto logits = (*fc)(out);` (`llama.cpp:251`) is a bare
  `ops::linear_op` with `bias = nullptr`. **No logit scaling, no soft-cap, no temperature.**

```
h[s, :]      = W_fc[ token_id[s], : ]
logits[s, :] = h_final[s, :] @ W_fc.T
```

### 7.3 Vocab is not padded

`vocab_size_divisible_by_32` (`llama.cpp:123`) is used **only** on the untied branch
(`llama.cpp:140`). Under tying, `fc` is built with the raw `vocab_size`. 32000 is already a
multiple of 32, and the checkpoint's `(1, 1, 32000, 384)` confirms it. Logits are `[S, 32000]`
with no padded tail to slice off.

Confidence: high.

---

## 8. Reference forward pass

Putting §§1-7 together. Shapes are for a single sequence, batch dropped; all weights are the
checkpoint tensors with their two leading singleton dims squeezed.

```python
Dh, H, G, EPS, THETA = 64, 6, 3, 1e-5, 500_000.0
REP = H // G  # 2

def rmsnorm(x, gamma):                                   # §2
    return gamma * x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)

def silu(z):                                             # §6
    return z / (1.0 + np.exp(-z))

def rope(t, S):                                          # §3  t: [nH, S, Dh]
    i        = np.arange(Dh)
    inv_freq = THETA ** (-2.0 * (i // 2) / Dh)           # inv_freq[2j] == inv_freq[2j+1]
    ang      = np.arange(S)[:, None] * inv_freq[None, :] # [S, Dh]
    cos, sin = np.cos(ang), np.sin(ang)
    rot            = np.empty_like(t)
    rot[..., 0::2] = -t[..., 1::2]                       # x @ trans_mat
    rot[..., 1::2] =  t[..., 0::2]
    return t * cos + rot * sin

def forward(W, tokens):
    S = len(tokens)
    x = W["llama/fc/weight"][tokens]                     # §7: plain gather, no scaling

    causal = np.triu(np.full((S, S), -np.inf), k=1)      # §5.2: additive, 0 / -inf

    for b in range(6):
        p = f"llama/llama_block_{b}"
        h  = rmsnorm(x, W[f"{p}/attention_norm/gamma"])

        q  = h @ W[f"{p}/attention/q_linear/weight"].T   # [S, 384]
        kv = h @ W[f"{p}/attention/kv_linear/weight"].T  # [S, 384]

        q = q.reshape(S, H, Dh).transpose(1, 0, 2)                 # §4.1 [H, S, Dh]
        k = kv[:, : G * Dh].reshape(S, G, Dh).transpose(1, 0, 2)   # §4.2 K first
        v = kv[:, G * Dh :].reshape(S, G, Dh).transpose(1, 0, 2)   # §4.2 then V

        q = rope(q, S)                                   # §3 — Q and K only
        k = rope(k, S)

        k = np.repeat(k, REP, axis=0)                    # §5.3 head h -> group h // REP
        v = np.repeat(v, REP, axis=0)

        scores = q @ k.transpose(0, 2, 1) / np.sqrt(Dh) + causal   # §5.1, §5.2
        scores -= scores.max(-1, keepdims=True)
        w = np.exp(scores); w /= w.sum(-1, keepdims=True)
        a = w @ v                                        # [H, S, Dh]

        a = a.transpose(1, 0, 2).reshape(S, H * Dh)      # §4.3 heads_fusion
        x = x + a @ W[f"{p}/attention/out_linear/weight"].T

        h = rmsnorm(x, W[f"{p}/mlp_norm/gamma"])         # §1: second residual
        gated = silu(h @ W[f"{p}/mlp/w1/weight"].T) * (h @ W[f"{p}/mlp/w3/weight"].T)
        x = x + gated @ W[f"{p}/mlp/w2/weight"].T        # §6

    x = rmsnorm(x, W["llama/ln_fc/gamma"])
    return x @ W["llama/fc/weight"].T                    # §7: tied, no scaling
```

`np.repeat(k, REP, axis=0)` (not `np.tile`) is the §5.3 mapping. Getting this backwards is a
close cousin of the RoPE bug: with `H=6, G=3` both produce a correctly-shaped tensor and a
model that still generates text.

### 8.1 Empirical check of this derivation

The code block above was run verbatim against
`artifacts/checkpoints/nanollama3_step00003000.pkl`, on eight random 256-token windows of
`artifacts/tokens/val_ids.npy`, in float32. This is *not* the parity harness (that is the next
task's job, and it compares against the HF conversion); it is a cheap self-check that the
document's own pseudocode is internally consistent and reproduces the model the checkpoint
actually encodes.

Mean next-token cross-entropy, and the same run with one convention deliberately broken:

| variant | mean CE (nats) | Δ vs derived | in SE |
|---|---|---|---|
| **as derived above** | **1.847** (sd 0.315, **SE 0.112**) | — | — |
| RoPE changed to split-halves `(x[i], x[i+Dh/2])` | 3.131 | +1.284 | 11.5 |
| GQA broadcast changed to `tile` instead of `repeat_interleave` | 3.717 | +1.870 | 16.8 |
| fused `kv` split changed to V-before-K | 7.603 | +5.756 | 51.6 |

Dispersion matters here, so it is reported rather than left implicit. Per-window spread is
large (sd ≈ 0.315 nats across the eight windows), giving SE of the mean ≈ **0.112 nats** and a
2σ detection floor of ≈ **0.22 nats**. This instrument is *coarse*: it cannot see anything
smaller than roughly a fifth of a nat.

The training run's own reported loss at step 3000 is **1.8781**; the derived reference lands at
1.847, a gap of 0.031 — **0.3 SE**, comfortably inside noise. (The two are also not the same
measurement: 1.8781 is a running loss over training batches, 1.847 a point estimate over eight
validation windows. Agreement this close is not evidence of anything beyond "no gross error".)

Two things are worth drawing out:

1. **Coarse, but decisive for its actual purpose.** The three ablations sit at 12–52 SE. Layout
   errors of the kind this document exists to prevent are enormous on this scale; there is no
   risk of one hiding in the noise.
2. **The split-halves variant lands at 3.13, and Plan 4's broken model measured 3.20.** That is
   a close reproduction of the historical failure — independent evidence that the bug Plan 4
   shipped was exactly this pairing, and that §3's verdict is the correction for it.

What this does **not** establish: that the NumPy reference matches ttml's device output to
within 1e-3 logits. A 0.22-nat floor is four orders of magnitude coarser than the tolerance the
next task wants to assert, and a genuinely subtle error hides inside it without difficulty —
moving the epsilon outside the sqrt costs 0.0002 nats (0.0 SE), i.e. is completely invisible
here. That is the whole reason the next task exists. See Q1 and Q6 in §9.

---

## 9. Open questions

Recorded rather than guessed, per the plan's rule. None of these blocks writing the reference,
but each is a real gap. **Q2 and Q4 have since been closed** and are kept here, marked, rather
than deleted — a closed question with its evidence is more useful to the next reader than a
silent absence. Q1 is the one that carries a consequence for the next task.

**Q1 — Device numerical precision, and therefore the achievable tolerance. This is the one that
matters.**

There is a concrete, identified instance, not just a general worry: **ttml's RMSNorm computes
its mean in bfloat16.** `rmsnorm_fw_program_factory.cpp:157-158` packs both the mean divisor and
epsilon as bf16 constants:

```cpp
uint32_t packed_scaler = pack_two_bfloat16_to_uint32(1.F / static_cast<float>(num_inner));
uint32_t packed_eps    = pack_two_bfloat16_to_uint32(args.epsilon);
```

`1/384 = 0.0026041667` is not representable in bfloat16 (8-bit significand); the nearest bf16
value is off by roughly 0.1–0.2% relative. That error enters `mean(x²)` systematically, and so
enters every one of the 13 RMSNorm calls in the forward pass. **No float32 NumPy reference can
reproduce it**, because it is not a rounding difference the reference could match — it is a
different divisor. This argues concretely that a **1e-3 NumPy-vs-device logit tolerance is not
achievable**, independent of anything else.

Still untraced: the accumulation and output dtype of `ttnn_fixed::matmul`
(`ttml/ttnn_fixed/matmuls.cpp`), which governs every projection and the QK^T/AV products, and
what `core::ComputeKernelConfig::precise()` resolves to per-op. Those are the remaining terms.

Consequence for the next task — the important part: the 1e-3 tolerance is a
NumPy-vs-HF-conversion budget, and both sides of *that* comparison are float32 host
arithmetic, so it may well be fine. What it must **not** be read as is a NumPy-vs-device budget.
And a NumPy-vs-NumPy control should be run before the gate is asserted, otherwise a failure is
ambiguous between "the conversion is wrong" and "the tolerance was never achievable".

Q2 — `fmod(theta_mat, 2π)` — CLOSED, immaterial.
`ttml/ops/rope_op.cpp:221` reduces the angle to the principal range before sin/cos; a naive
NumPy `rope` does not. Measured: running the §8 reference in float32 with and without
`np.fmod(ang, 2π)` gives mean CE **1.8470 both ways** — identical to four decimal places
(Δ = −0.00002, 0.0 SE). At `S = 256` the largest angle is ~255 rad, well inside float32's exact
range for this reduction. **No action needed**, though applying the `fmod` costs nothing if a
reader wants to remove the question entirely.

Q3 — `RotaryEmbeddingParams::theta` is never populated.
`build_rope_params` (`ttml/ops/rope_op.cpp:287-298`) sets every field *except* `.theta`, so the
struct member keeps its `10000.0F` default (`ttml/ops/rope_op.hpp:32`) while the frequencies are
built from the real `theta = 500000` passed as an argument to `gen_freqs` (`:285`). A grep found
no reader of `params.theta` anywhere in `ops/`, `modules/`, or `models/`, so it appears inert —
but I cannot prove no consumer exists outside those directories. **To resolve:** grep the whole
tt-train tree including tests and the distributed model variants. **Flagged because** anyone who
later reads `params.theta` expecting the configured value will get `10000` and produce a
plausible-but-wrong model — the same failure shape as the Plan 4 bug.

Q4 — Value of the causal mask constant — CLOSED.
`constexpr uint16_t BF16_NEG_LARGE_BITS = 0xCE6E;  // upper 16 bits of -1e9F (bfloat16)`
(`ttml/metal/common/dataflow_utils.hpp:46`, used at `:275` by `fill_causal_mask_tile`). So the
constant is exactly `−1e9` rounded to bf16, matching the `1e9` multiplier the `Arbitrary` path
uses (§5.2) — the two mask branches agree on the value as well as the form. Both `−1e9` and
`−1e9/8 ≈ −1.25e8` underflow to exactly 0 after `exp`, identically to a true `−inf`, so a
reference using `−np.inf` is correct. **No action needed.**

Q5 — Whether ttml's `softmax` differs from a textbook softmax.
The fused kernel uses an online/flash formulation with running max and log-sum-exp
(`sdpa_compute_utils.hpp:160-274`); the composite path calls `ttml::metal::softmax`
(`scaled_dot_product_attention.cpp:181`). I read enough of the flash path to confirm it is a
standard max-subtracted softmax, but did not audit `metal/ops/softmax` itself. Since the
production path is the fused one, this is a gap only for anyone reasoning about the composite
path. **To resolve:** read `ttml/metal/ops/softmax/`.

Q6 — No ttml activation was ever captured; §8.1 is an end-to-end proxy, not an op-level check.
Every convention above is read off source. §8.1 then runs the derived reference and shows it
reproduces the checkpoint's training loss (1.847 vs 1.8781, 0.3 SE) while three
deliberately-broken variants do not — real evidence, and much stronger than the four checks Plan
4 passed, but still a single scalar at the end of a six-block stack.

**Its detection floor is ≈0.22 nats (2σ), not the "~0.05" an earlier draft of this section
claimed** — the per-window sd is 0.315 across eight windows, so SE ≈ 0.112. That is about 4×
coarser than stated, and the corrected figure is the one to reason with.

Measured examples of what does and does not hide inside it:

| perturbation | mean CE | Δ | visible? |
|---|---|---|---|
| eps moved *outside* the sqrt | 1.8467 | −0.0002 (0.0 SE) | **no — invisible** |
| `1 + gamma` instead of `gamma` | 3.6238 | +1.777 (15.9 SE) | yes, loud |
| embedding scaled by `sqrt(384)` | 10.7208 | +8.874 (79.6 SE) | yes, very loud |

So the genuinely §8.1-invisible set is essentially **just epsilon placement** — the other
"quiet-looking" conventions turn out to be loud. Useful to know when deciding where to look
first if the harness misbehaves.

To resolve: capture per-layer activations from an actual ttml forward pass and diff them
against the reference block by block, which localises a convention error to one op instead of
leaving a whole-model mismatch to bisect. **Consequence:** if the next task's HF-vs-NumPy
comparison fails at 1e-3, §8.1 does *not* license concluding "the NumPy side is right, so the
converter is wrong" — both sides remain suspects until an op-level trace says otherwise.

---

## 10. Confidence summary

| # | Item | Verdict | Confidence | Basis |
|---|---|---|---|---|
| 1 | RMSNorm | `gamma · x / sqrt(mean(x²) + eps)`, eps **inside** the sqrt, added to the mean of squares; plain `gamma`, not `1+gamma`; eps = 1e-5 | High | Fused kernel + composite agree |
| 2 | **RoPE pairing** | **Interleaved `(x[2i], x[2i+1])`** — *not* split-halves | **Very high** | Four consistent lines of evidence (§3.1-3.4), incl. the per-tile matmul impossibility proof; ablation costs 1.28 nats (12 SE) |
| 2b | RoPE freq / caches | `inv_freq[i] = θ^(−2·floor(i/2)/Dh)`, angle = `pos · inv_freq`; `neg_*` caches are backward-only | High | `gen_freqs`, `build_rope_params` comments |
| 3 | `grouped_heads_creation` | **K first**, then V, each `G·Dh` wide; head-major contiguous `Dh` slices; plain reshape+transpose | High | ttnn reader/writer kernels + backward `concat({k,v})` + HF importer |
| 4 | SDPA scale | `1/sqrt(head_dim)` = 1/8 | High | Program factory + composite agree |
| 4b | Causal mask | **Additive**: 0 keep / `−1e9` masked, added pre-softmax. This checkpoint used the **`Arbitrary`** path (`trainer.py` passes an explicit `tril` mask), which coincides numerically with the built-in `Causal` path | High | `trainer.py:73-76,102` + `utils.py:160-169` + both kernel branches (§5.2) |
| 4c | GQA mapping | `kv_group = q_head // (H/G)` — `repeat_interleave`, not `tile` | High | Fused reader + composite reshape agree |
| 5 | SwiGLU | `(silu(x @ w1ᵀ) * (x @ w3ᵀ)) @ w2ᵀ`; SiLU on the **w1** branch | High | Fused + composite agree; shape asserts; HF importer mapping |
| 6 | Embedding / output | One tied `[32000, 384]` tensor; gather in, `@ Wᵀ` out; **no scaling at either end**; no `tok_emb/weight` in the checkpoint; vocab unpadded | High | Construction + `parameters()` dedup + checkpoint manifest |
| — | End-to-end behaviour | Reference reproduces the training loss (1.847 vs 1.8781, 0.3 SE); ablations cost 12–52 SE | Decisive for layout errors; **floor ≈0.22 nats** | §8.1 |
| — | Numerical tolerance | **Resolved for NumPy-vs-HF** — Task 3's parity gate (`tests/test_numpy_parity.py`) measures ~5e-6 to ~1.6e-5 absolute agreement on a 256-token window, ~120x inside a 1e-3 tolerance. **Still unresolved for NumPy-vs-device** — RMSNorm's mean divisor is bf16 on the actual hardware, and no float32 host comparison bounds that gap | — | Q1 |
| — | `fmod` rounding | **Closed** — identical to 4 d.p. | — | Q2 |
| — | Mask constant | **Closed** — `0xCE6E` = `−1e9` in bf16 | — | Q4 |
| — | `params.theta` inert | **Believed inert, not proven** | — | Q3 |

Six of six requested items resolved with high or very-high confidence. Of the original six open
questions, **two are now closed** (Q2, Q4). The four that remain are not convention questions:
Q1 is about achievable numerical precision (and has hardened from a worry into an identified
obstacle), Q3 is a latent trap in ttml rather than a fact about this forward pass, Q5 is a gap
only for the non-production composite path, and Q6 is about the coarseness of the end-to-end
check.

If you read only one open question, read Q1 — though note its body scopes the remaining
obstacle to the *NumPy-vs-device* comparison specifically. The *NumPy-vs-HF-conversion* gate
Q1 originally worried might be unreachable was resolved by Task 3 at ~5e-6 to ~1.6e-5
(`tests/test_numpy_parity.py`), two orders of magnitude inside the 1e-3 tolerance that
concerned this document when it was written.
