<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Follow-ups after Plans 1–4

Findings that outlived the branch that produced them. Recorded here so the next plan starts
from what we learned rather than rediscovering it.

## 1. The model trained with its norm layers frozen — report upstream

**This is a real training bug, not a conversion artifact.** All 13 RMSNorm gammas in
`nanollama3_step00003000.pkl` are exactly `1.0` (`mean=1.0, sd=0.0`).

The cause is arithmetic, not a missing gradient. The optimizer state carries genuine gradients
for them — `exp_avg` absmax ≈ 3.6e-4 for block-0 gammas, 2.5e-3 for `ln_fc` — but the
parameters are bf16 at 1.0, where one ulp is 0.0039. With `stochastic_rounding: False` in the
training config, a ~3e-4 Adam step rounds back to 1.0 on **every** step. The updates are
computed and then discarded.

Consequences:

- The model is training with 13 of its layers effectively frozen. Whatever it has learned, it
  learned without them.
- **23% of the conversion's mapping decisions are validated by nothing.** Swapping layer 3's
  `input_layernorm` and `post_attention_layernorm` changes HF-side loss by *exactly* 0.0000.
  The loss gate cannot see the norm mapping at all, and will stay blind until this is fixed —
  at which point a norm mis-mapping becomes a live defect no current test would catch.

Worth raising with the tt-train side independently of anything here. `stochastic_rounding` is
an existing config flag, currently off; fp32 master weights would also resolve it.

## 2. The converter needs a check the loss cannot provide

The acceptance gate is HF-side validation loss within 0.2 nats of the training run. Measured
sampling sd is 0.024 nats, so **any defect costing less than ~0.2 nats is invisible**. For
calibration, from deliberate defect injection: a single layer's `q_proj` left un-permuted costs
0.685 nats, a gate/up swap 0.431, a K/V swap 5.753. Per-tensor attention defects are caught;
subtler ones are not — and the norm mapping (item 1) is not caught at all.

**The missing instrument:** a ~40-line pure-numpy reimplementation of ttml's forward pass
(interleaved RoPE, ttml's norm order) run on the raw checkpoint tensors and compared against HF
logits on a fixed token window. That is a ~1e-3 relative-error gate instead of a ~0.2-nat one,
it validates the norm mapping the loss cannot, and it needs no hardware — consistent with the
finding that conversion is entirely CPU-side.

This is the single highest-value item here.

## 3. RoPE scaling would be silently dropped

ttml supports `scaling_factor`, `original_context_length`, `high_freq_factor`, and
`low_freq_factor` (`ttml/models/llama.cpp:141-155`). The checkpoint header's
`transformer_config` records **none** of them, and `build_config` never emits `rope_scaling`.

A scaled-RoPE checkpoint would convert cleanly, load cleanly, and be wrong at long context —
and unlike every other trap we found, the information needed to detect it is not in the
checkpoint at all. The header schema would need extending *before* such a checkpoint exists,
since backfill cannot recover what was never recorded.

## 4. Packaging traps, verified against the tt-kernel packager

A v4 bundle does not ingest a model directory. It packages an authored
`tt_kernel_manifest.json` plus an optional adapter folder, and references weights only as an HF
repo id consumed by `snapshot_download`. **It never reads `config.json`, never hashes weights,
and never validates that `entrypoint.arch_name` matches the weights repo.** All correctness
responsibility sits upstream of it.

- **`max_model_len` must be pinned to 256.** `tokenizer_config.json` advertises
  `model_max_length: 1000000000000000019884624838656`, so anyone deriving it from the tokenizer
  gets a serving stack accepting 4k contexts from a model trained to 256 — degraded output, no
  error.
- **The model card and license must exist on the HF repo *before* the first `tt-kernel push`.**
  `tag_repo`/`set_catalog_listing` replace the entire model-card front matter with
  `ModelCardData(tags=...)`, discarding anything added afterwards.
- **Weights must be pushed to an HF repo first.** `weights.repo` will not accept a local path,
  and `artifacts/hf/` is gitignored and local-only.
- `generation_config.json` is absent; it is where `eos_token_id` and default sampling belong for
  a published model.
- `tokenizer_config.json` declares `"tokenizer_class": "PreTrainedTokenizer"` while the
  tokenizer loads as `PreTrainedTokenizerFast`. Tolerated by transformers 4.52; worth fixing
  before publishing.

## 5. Process: the gap between predicting a bug and catching one

Plan 4 named the RoPE layout trap in advance and was right. But the check that actually caught
it lived in a bash block, not a test, which is why the regression test had to be added
reactively after the failure.

**Rule for future plans:** if a step produces a number that decides pass/fail, it must be a test
in the plan text, not a shell command in prose. Plan 4's Self-Review asserted "every capability
named above resolves to code or a test" — for its central capability, at plan-writing time, it
did not.

This is the fourth instance of the same shape: a plan's justification written more confidently
than the artifact it justifies. The existing gate — every capability in a table must name a
test — catches table claims. It does not catch a *task step* that is load-bearing but untested.

## 6. Smaller items

- No purity regression test for `convert/hf_mapping.py` specifically.
- `read_tensors`' nested/optimizer-group branch works (verified against the real checkpoint:
  100 records, 50 names × 2 moments) but is unpinned by tests.
- The tied embedding is stored twice in `model.safetensors` — 68.6 MB for a 44 MB model. With
  `tie_word_embeddings: true`, omitting `lm_head.weight` is conventional.
- `convert_checkpoint` unpickles and discards ~88 MB of optimizer records after the model group
  completes; stopping early would roughly halve conversion I/O.
- `safetensors.numpy.load_file` on this artifact needs `ml_dtypes` imported first. Not a defect
  (torch/transformers/vLLM read bf16 natively), but the plan's "verified working" was
  conditional in a way nobody wrote down.
- `transformer_config` duplicates `vocab_size`/`max_sequence_length`, which `build_config`
  ignores in favour of the top-level fields. A one-line cross-check would close the ambiguity.
- The design spec still names `convert/ckpt_to_hf.py`; the file is `convert/to_hf.py`.
