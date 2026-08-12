<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# NumPy Parity Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the Hugging Face conversion against an independently-derived NumPy reimplementation of ttml's forward pass, at a logit tolerance materially tighter than the loss gate's ~0.2 nats. The exact tolerance is **measured, not assumed** — see "On the achievable tolerance" below.

**Architecture:** `convert/ttml_forward.py` implements ttml's Llama forward in pure NumPy, reading raw checkpoint tensors and using **ttml's own conventions** — interleaved RoPE, ttml's tensor layouts, no HF permutation. `tests/test_numpy_parity.py` compares its logits against the converted HF model's on a fixed token window.

**Tech Stack:** Python 3.10+, numpy, `ml_dtypes`, transformers (for the comparison side only), pytest

## Why this exists, and why it must not be circular

Plan 4's acceptance gate is HF-side validation loss within 0.2 nats of the training run. Measured sampling sd is 0.024 nats, so **any defect costing less than ~0.2 nats is invisible**. Worse, 13 of 57 tensors — all the RMSNorm gammas — are validated by *nothing*: they are all exactly 1.0 in this checkpoint (see `docs/superpowers/specs/2026-08-11-followups.md`), so swapping two of them changes loss by exactly 0.0000.

Plan 4 also demonstrated that agreement among checks proves little when the checks share a blind spot. The first conversion loaded, tied correctly, showed 4.75-nat entropy, and generated fluent prose — while computing the wrong function, because every one of those checks was insensitive to RoPE row layout.

**The independence requirement follows directly.** This NumPy implementation must be derived from **ttml's C++ source**, not from `convert/`. If it is written by reading our own converter, it inherits the converter's misunderstandings, agrees with it, and proves nothing. The two paths must reach the same logits by different routes:

- **NumPy path:** raw checkpoint tensors → ttml conventions (interleaved RoPE, fused KV, ttml's norm order) → logits
- **HF path:** raw checkpoint tensors → `convert/` (split KV, permuted RoPE rows, HF layout) → `transformers` → logits

Agreement is then real evidence. **Anyone implementing Task 1 must not read `convert/hf_mapping.py` or `convert/to_hf.py`** beyond the tensor-name inventory, and must derive every numerical convention from `~/tt-metal/tt-train/sources/ttml/`.

## What has been verified about ttml's forward pass

Read from source; treat as starting points to confirm, not as settled facts.

**Block structure** (`modules/llama_block.cpp:65-78`) — pre-norm, two residuals:

```
residual = x
h = attention_norm(x)          # RMSNormLayer
h = attention(h, mask)
h = h + residual
residual = h
h = mlp_norm(h)                # RMSNormLayer
h = mlp(h)
h = h + residual
```

**Attention** (`modules/grouped_query_attention.cpp:36-57`):

```
q  = q_linear(x)
kv = kv_linear(x)                                    # fused
q_heads, k_heads, v_heads = grouped_heads_creation(q, kv, num_heads, num_groups)
q_heads = rope(q_heads);  k_heads = rope(k_heads)    # RoPE on q and k only
attn = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)
out  = out_linear(heads_fusion(attn))
```

**MLP** (`modules/llama_block.cpp:34-37`) — `swiglu(x, w1, w2, w3)`, confirming `w2` is the down-projection independently of shape.

**RMSNorm** (`modules/rms_norm_module.hpp`) — `epsilon = 1e-5` default, matching the checkpoint header's `rms_norm_eps`.

**Still to derive in Task 1** (do not guess these):
- `ops::rope_op` builds `cos_cache`, `sin_cache`, **and** `neg_cos_cache`, `neg_sin_cache` (`ops/rope_op.cpp:41-57`). Four caches implies a specific rotation form — derive it.
- `ops::grouped_heads_creation` — exactly how the fused `kv` splits into K and V, and how heads are laid out.
- `ops::scaled_dot_product_attention` — the scaling factor and mask convention.
- `ops::swiglu` — which of `w1`/`w3` is gated and which is lifted.

## On the achievable tolerance — corrected after Task 1

This plan originally promised ~1e-3. Task 1's review found concrete evidence that is not
reachable against the device, and the reason is worth stating so nobody re-asserts it:

ttml packs the RMSNorm mean divisor as **bfloat16** — `pack_two_bfloat16_to_uint32(1.F / num_inner)`
(`rmsnorm_fw_program_factory.cpp:157`), and `packed_eps` likewise. bf16's 8-bit significand gives
ttml's `mean(x²)` a systematic ~0.1–0.2% relative error that **no float32 NumPy reference can
reproduce**. `ttnn_fixed::matmul`'s accumulation and output dtype remain untraced and may add more.

Two consequences bind Task 3:

- **Measure before asserting.** Run a NumPy-vs-NumPy control (fp32 vs bf16-rounded activations)
  to establish what agreement is *possible*, then set the gate from that. Asserting an
  unreachable tolerance makes every failure ambiguous between "the converter is wrong" and
  "the tolerance was never achievable" — the worst possible property for a diagnostic.
- **Disagreement does not automatically indict the converter.** Task 1's loss check has a
  measured standard error of ~0.102 nats (per-window sd ≈ 0.29 over 8 windows), so it cannot
  license trusting the NumPy path over the converter. On disagreement, both paths are suspect.

## Global Constraints

- SPDX header pair on new files.
- Python 3.10+.
- **`convert/` must NOT import `ttnn` or `ttml`.** This harness is pure NumPy and must run on any machine — it is the CPU-side instrument the project's own thesis calls for.
- **No bare `assert` for guards** in production code.
- **`convert/ttml_forward.py` must be derived from ttml C++ source, not from `convert/hf_mapping.py` or `convert/to_hf.py`.** See the independence requirement above. Cite the ttml source file and line for each convention in comments.
- `pyproject.toml` must NOT be modified.
- **Never write anything under `artifacts/checkpoints/` or `artifacts/hf/`.** Read only.
- **If a step produces a number that decides pass/fail, it must be a test**, not a shell command in prose. (Plan 4's lesson: its RoPE gate lived in a bash block, so the regression test had to be added reactively after the bug shipped.)

---

## File Structure

| File | Responsibility |
|---|---|
| `docs/ttml-forward-reference.md` | Derived, cited description of ttml's forward pass |
| `convert/ttml_forward.py` | Pure-NumPy ttml forward from raw checkpoint tensors |
| `tests/test_ttml_forward.py` | Component tests: RoPE, RMSNorm, SwiGLU, attention |
| `tests/test_numpy_parity.py` | Logit comparison NumPy vs HF; the actual gate |

---

## Task 1: Derive and document ttml's forward pass

**Files:**
- Create: `docs/ttml-forward-reference.md`

**Interfaces:** none — this task produces a written reference the next task implements against.

**Why a separate task:** the derivation is the part most likely to be wrong, and a written artifact can be reviewed on its own before any code depends on it. Plan 4's RoPE bug was a derivation error that survived because nothing wrote the convention down.

- [ ] **Step 1: Derive each convention from source, with citations**

For each item below, read the ttml C++ and record: the file and line, the mathematical form, and any layout assumption. Where a formula is ambiguous from the code, say so explicitly rather than guessing.

1. **RMSNorm** — the exact formula and where epsilon sits (inside or outside the sqrt).
2. **RoPE** — how `cos_cache`/`sin_cache`/`neg_cos_cache`/`neg_sin_cache` are built from `theta` and position, and precisely which element pairs are rotated together. This is the convention that broke Plan 4; it is the most important item here. State plainly whether ttml pairs `(x[2i], x[2i+1])` (interleaved) or `(x[i], x[i+d/2])` (split-halves).
3. **`grouped_heads_creation`** — how the fused `kv` tensor splits into K and V, whether K precedes V, and the resulting head layout.
4. **`scaled_dot_product_attention`** — the scale factor and how the causal mask is applied.
5. **`swiglu`** — the exact expression in terms of `w1`, `w2`, `w3`.
6. **Embedding and output** — how `fc/weight` serves as both embedding lookup and output projection under weight tying, and whether any scaling is applied at either end.

- [ ] **Step 2: Record open questions explicitly**

Anything you could not determine from source goes in a "Open questions" section with what you would need to resolve it. Do **not** fill gaps with plausible assumptions — an unmarked guess here defeats the independence the whole plan rests on.

- [ ] **Step 3: Commit**

```bash
git add docs/ttml-forward-reference.md
git commit -m "docs: derive ttml's Llama forward pass from source

Written reference for the NumPy parity harness, with file:line citations for
each convention. Derived from ttml C++ only -- deliberately not from our own
converter, so the parity check is not circular."
```

---

## Task 2: NumPy implementation

**Files:**
- Create: `convert/ttml_forward.py`
- Test: `tests/test_ttml_forward.py`

**Interfaces:**
- Consumes: `convert.checkpoint_reader.read_tensors`, `read_checkpoint_meta`
- Produces:
  - `rms_norm(x, gamma, eps) -> np.ndarray`
  - `apply_rope(x, positions, theta, *, head_dim) -> np.ndarray` — ttml's convention
  - `swiglu(x, w1, w2, w3) -> np.ndarray`
  - `attention(x, q_w, kv_w, out_w, *, num_heads, num_groups, theta) -> np.ndarray`
  - `forward(checkpoint_path, token_ids) -> np.ndarray` — logits, shape `(seq, vocab)`

- [ ] **Step 1: Write component tests first**

Each component gets a test against a hand-computed expectation, not against the HF model — testing against HF here would reintroduce circularity. For example, `rms_norm` on a small known vector with a known gamma, computed by hand in the test.

For RoPE specifically, assert the **rotation property** rather than a magic array: rotating a vector by position `p` then by `-p` must return the original, and the inner product between two vectors rotated by the same position must be preserved. Those hold for any correct RoPE and fail for a wrong pairing.

- [ ] **Step 2: Implement**

Read tensors with `read_tensors` in declaration order, `squeeze_leading` equivalents applied locally (do not import from `hf_mapping` — reimplement the trivial squeeze inline, with a comment noting the deliberate duplication and why).

- [ ] **Step 3: Validate the implementation independently of HF**

Compute cross-entropy loss over held-out tokens from `artifacts/tokens/val_ids.npy` using **only** the NumPy path. It should land near **1.8781** — the training run's own number.

This is the check that proves the NumPy path is itself correct before it is used to judge anything else. If it lands far off, the NumPy implementation is wrong and must be fixed before Task 3; do not proceed with a broken reference.

Make this a test with an explicit tolerance, not a shell command.

- [ ] **Step 4: Commit**

---

## Task 3: The parity gate

**Files:**
- Create: `tests/test_numpy_parity.py`

**Interfaces:** none — this task produces the gate.

- [ ] **Step 1: Compare logits on a fixed window**

Run both paths on the same fixed token window (a seeded slice of `val_ids.npy`, or a short fixed prompt) and compare logits:

- max absolute difference
- max relative difference
- correlation

Pick the tolerance from what you measure, not in advance — but state the reasoning. bf16 weights on both sides with fp32 accumulation should agree far better than 1e-2; if the observed agreement is worse than ~1e-2 relative, something is genuinely different between the paths and that is a finding.

- [ ] **Step 2: Prove the gate is not hollow**

Demonstrate the gate fails when the conversion is wrong, by the same method Plan 4's reviewer used: monkeypatch `permute_rope_qk` to identity in a scratch copy, reconvert into a throwaway directory, and confirm the parity test fails. Report the measured divergence.

**Do not modify `artifacts/hf/`.** Use a temporary directory.

- [ ] **Step 3: Close the norm-mapping blind spot**

This is the gap the loss cannot see. With all gammas at 1.0, swapping two norms is loss-invisible — verify the parity gate is *also* currently blind to it, then close it directly: construct a synthetic checkpoint with **distinct, non-unit gamma values**, run it through `convert_checkpoint`, and assert each gamma lands at its correct HF destination.

That test does not depend on the trained checkpoint's degenerate gammas, so it keeps working when the upstream `stochastic_rounding` issue is fixed and gammas become real.

- [ ] **Step 4: Record results and commit**

Update CLAUDE.md with what the parity gate measures, its tolerance, and — explicitly — what it still cannot see.

---

## Self-Review

**Every capability named above resolves to a test.** Task 1 produces a document (reviewable on its own). Task 2's components → `tests/test_ttml_forward.py`, and its independent loss validation is a test with a tolerance, not a shell step. Task 3's parity comparison, its not-hollow proof, and the norm-mapping assertion are all tests. **No load-bearing number in this plan lives in a bash block** — that is Plan 4's lesson applied.

**Does the artifact satisfy the rationale?** The stated reason is an instrument sharper than the loss gate that can see the norm mapping. Task 3 Step 3 addresses the norm mapping specifically; Steps 1–2 address the sharpness and prove it. If Task 3 Step 3 were dropped, the plan would not satisfy its own justification.

**Independence is a constraint, not an aspiration** — stated in Global Constraints, repeated in Task 1 and Task 2 Step 2, and the reason Task 1 exists as a separate reviewable artifact.

## Known risks

- **The NumPy implementation could be wrong in the same way the converter is.** Task 2 Step 3 guards this: the NumPy path is validated against the training run's own loss before it is used to judge the converter.
- **Derivation gaps.** If Task 1 cannot determine a convention from source, that is an open question to escalate, not to paper over. A plausible-looking guess that happens to match the converter would silently defeat the whole plan.
- **The gate cannot see what neither path implements.** If both drop RoPE scaling (which the header does not record — see followups item 3), they will agree and both be wrong. This harness validates the *conversion*, not the *checkpoint's completeness*.
