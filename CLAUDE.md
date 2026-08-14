# tt-tnt — project notes for Claude

## What this project is

The reference example of a **Tenstorrent-first model**: tt-tnt (originally built, and
published, as **tt-nanollama3** — see README.md's "Lineage" section for what changed and
why), trained from random init on Blackhole with `ttml` (tt-train), packaged as a
**tt-kernel v4 bundle**, and served through the **Tenstorrent vLLM plugin**. Small model,
complete story — the point is to show end to end what a model built for TT from line one
looks like across train → package → publish → serve.

The log below is kept in the order it happened, under the names used at the time — entries
before the 2026-08-13 rename say "NanoLlama3"/"tt-nanollama3" because that was this project's
name when they were written, not because they refer to something else.

Design: [`docs/superpowers/specs/2026-08-11-tt-nanollama3-design.md`](docs/superpowers/specs/2026-08-11-tt-nanollama3-design.md)

## How we got here (2026-08-11)

The original prompt was a changelog request against `tt-kernel-package-manager`, which turned
into: *"what would it take to make tt-animatediff part of tt-kernel-cache?"* Exploring that
surfaced a blocker, and the plan pivoted twice.

**Pivot 1 — animatediff doesn't fit tt-kernel.** Both tt-kernel serving backends terminate in
`/v1/chat/completions`; a text-to-video model has no chat-shaped output. Then the repo was
updated to `00dba42`, which made it worse: v4 is explicitly "vLLM only, kernels-less" and v3
is "legacy, read-only supported." A model needing a kernel cache has only the deprecating
path; a non-vLLM model has none. Findings recorded in Appendix A of the spec — worth taking
to the tt-kernel maintainers.

**Pivot 2 — use a model we actually own.** The `lfs-00`…`lfs-05` lesson arc in
`tt-vscode-toolkit` builds a Llama-3-style model from scratch, and it had already been
trained: `~/tt-metal/tt-train/checkpoints/nanollama3_char_3k.pkl_final.pkl`. Being an LLM, it
fits v4 perfectly, and the weights are ours outright — no redistribution question.

## Licensing — a relationship to maintain, not a one-time file

This repo is **Apache-2.0**, matching tt-metal and tt-vscode-toolkit. Every source file carries
an SPDX header. That much is settled. What needs *maintaining* is the honesty of the
provenance section in `README.md`, because two upstreams are not simply Apache-2.0:

- **TinyStories is CDLA-Sharing-1.0**, a share-alike *data* license — not permissive. We do not
  redistribute the corpus (it downloads from the Hub at a pinned revision), and we do **not**
  assert that weights trained on it fall outside "Data Derivative". Do not quietly upgrade that
  hedge into a claim.
- **Mini-LLM declares no license** (verified via the GitHub API — `license: None`), so it grants
  no rights. It is credited for architectural *choices*, which come from published papers; our
  implementation derives from tt-train's `nanollama3` config and `ttml`, not its source. Keep
  that distinction sharp — a credit is not a license inheritance.

**Rules for future work:**
- New source files get the SPDX header pair. No exceptions.
- Adding a dependency, corpus, or checkpoint source means adding its license to the README's
  provenance section *in the same change* — not later.
- **When weights are finally published**, the model card must name the corpus and its license
  explicitly, and describe the model as a demonstration rather than a capable general model.
  At ~22M parameters over a fraction of an epoch, any other framing would be false.
- If a future corpus is more permissive (or more restrictive) than TinyStories, the provenance
  section changes with it. It is not boilerplate.

## Key decisions

- **~22M params, 32K BPE, small real corpus.** Uses the existing `nanollama3.yaml` unchanged.
  Deliberately *not* Mini-LLM's ~80M/361M-token/~5h-A100 run, which would need a new model
  config plus a data pipeline and would block packaging behind a multi-day job.
- **Ship both altitudes.** ttnn path as the portable default, the lesson's TT-Lang kernels as
  the tuned path, bound by parity tests against `reference_gpt.py`. This is what makes it an
  exemplar rather than a packaging demo.
- **Real adapter, not a disguise.** Validate conversion via a throwaway `LlamaForCausalLM`
  spike, then write `TTNanoLlama3` registering its own `arch_name`.
- **Reuse policy:** never reimplement what ttnn or ttml already provide.

## Gotchas learned during design

- `nanollama3.yaml` vs `nanollama3_char.yaml` differ **only** by `vocab_size: 32000` — same
  6 heads / 3 groups / 384 dim / 6 blocks / seq 256 / θ=500000. Switching to BPE does not
  make the model meaningfully larger; it adds the embedding table.
- Only `shakespeare.txt` (1.1 MB) is present in `tt-train/data/`. Far too small for a 32K
  vocab — a real corpus is required.
- Upstream tt-metal runs **no CI for training ops on Blackhole** (`tt-train` `GTEST_SKIP`s
  softmax, cross-entropy, rmsnorm, SDPA on p100/p150). The lesson arc's run is our own
  verification, at v0.73. `~/tt-metal` is currently on `rollback-pre-qwen36-1576-g620793d898`.
- Let `ttml` close the device — bypassing its `finally` triggers a teardown abort in
  `MetalContext::destroy_all_instances`.
- If the board times out on device open, `tt-smi -r` first. Common on p300c/QB2.
- `max_sequence_length` is 256, so the manifest's `max_model_len` must be 256. Don't promise
  more than the model was trained for.

## `feat/tokenizer-and-corpus` (2026-08-11)

Built the corpus-prep and tokenizer pipeline the model needs before any training run:
`train/data.py` (fetch + normalize TinyStories), `convert/tokenizer.py` (32K byte-level
BPE, exported for both ttml load paths), and `scripts/build_tokenizer.py` (the script that
actually produces the shipped artifacts). Real numbers from the production run: the 2.2 GB
raw download (2,227,753,162 bytes) prepares down to 536,870,821 bytes / 3,548,279 lines at
the 512 MB cap, and the tokenizer trained on that reaches exactly 32000 tokens — no
shortfall. (Line count is higher than a naive pre-fix estimate would suggest: `</s>` is 9
bytes shorter than `<|endoftext|>`, so once separator lines are rewritten, more whole
lines fit under the same 512 MB cap before truncation kicks in.)

A whole-branch review caught three things worth not rediscovering:

- **`vocab_size` is a ceiling, not a promise.** `BpeTrainer` stops merging once the corpus
  runs out of pairs worth learning — a small corpus (or a small `--corpus-mb`) silently
  under-shoots the target. `scripts/build_tokenizer.py` now reloads the export via
  `load_exported()` and hard-fails (non-zero exit) if the achieved vocabulary doesn't
  match the requested one, rather than printing "Done" over a mismatch that would only
  surface later as an embedding-shape failure in tt-train.
- **`PreTrainedTokenizerFast` overrides `add_prefix_space` on wrapping.** Setting
  `pre_tokenizers.ByteLevel(add_prefix_space=True)` on the backend tokenizer is not
  enough — `PreTrainedTokenizerFast.__init__` (transformers 4.52.4) applies its own
  `add_prefix_space=False` default onto the wrapped tokenizer, silently discarding it.
  Merges were being *learned* with prefix-space on but *applied* with it off. Fix: pass
  `add_prefix_space=True` to the `PreTrainedTokenizerFast(...)` constructor itself, and
  verify by reading the exported `tokenizer.json` back — don't assume the flag survived
  wrapping. One consequence: decode() now legitimately produces a leading space on text
  that doesn't already start with whitespace — that's the injected prefix space coming
  back out, not data loss.
- **TinyStories' `<|endoftext|>` separator must be mapped to `</s>`, not left as prose.**
  Passed through unmodified, it's four ordinary subword tokens per occurrence (662,878
  occurrences in the production corpus — 18.7% of all lines), a wasted vocabulary slot on
  the subword `endoftext`, and — critically — zero real appearances of `</s>` in training
  data, so the model never learns a stop token. `prepare_corpus` now rewrites lines that
  are *exactly* the separator (after stripping) to the literal text `</s>`, which the
  tokenizer maps to the eos special id; lines that merely mention the separator inside
  other text are left alone, since a substring replace would corrupt real prose.

## `feat/training-entrypoint` (2026-08-11)

Wrote `train/run.py`, the hardware entrypoint that actually trains NanoLlama3 on Blackhole.
tt-train's own Python trainer (`examples/python/transformers/training.py`) doesn't run
against the current tree — three independent breaks (stale `trainer` module import, an
extra `val_ids` argument `train()` doesn't accept, a `TrainingConfig` missing the `seq_len`
`train()` reads) plus a hardcoded `shakespeare.txt` data path — so this entrypoint reuses
`TransformerModelFactory`, `create_optimizer`, `initialize_device`, `set_seed`, and `train()`
itself, and supplies our corpus, our tokenizer, `seq_len`, and a real `evaluate()`. That last
part matters: ttml's `train()` fills `val_losses` by copying the last training loss under a
comment calling it placeholder behavior, so a val number straight out of `train()` means
nothing — `evaluate()` runs the model in eval mode over held-out tokens with its own
`get_batch_ttml`/`build_causal_mask` calls and averages properly.

**Tokenizing the real corpus** (Task 1's pipeline, run against the actual TinyStories
corpus): `TokenStats(total_tokens=127635889, train_tokens=114872301, val_tokens=12763588,
vocab_size=32000)` — inside the 1.3–1.6×10^8 estimate, vocab exactly on target.

**A real 20-step run on p300c/Blackhole**, batch size 64, seq_len 256:

- First train loss: `10.6875` — matches the freshly-initialized-model expectation of
  `ln(32000) ≈ 10.4` closely enough to confirm the vocab/model setup is right.
- Last train loss: `7.4688`, monotonically non-increasing across all 20 steps (one flat
  step at 8.0625→8.0625, otherwise strictly down).
- Real validation loss (our `evaluate()`, 10 sampled batches, not `train()`'s placeholder):
  `7.0281` — distinct from the last train loss, so the loop was not bypassed.
- Step 1 took 18.73s (kernel compilation, not model performance); steps 2–20 averaged
  ~0.12–0.14 s/step (climbing to ~7.15 it/s by the end) — that second number is the one
  that reflects the model.

No checkpoint is written (`model_save_interval` is 0 in `RunConfig`/`build_yaml_config`) —
that's Stage 3's job, once we've read `ttml/checkpointing.py`'s format.

## `feat/checkpointing` (2026-08-11)

Added a validated checkpoint header schema over `ttml.checkpointing` (`train/checkpoint.py`)
and periodic save/resume to the training entrypoint (`train/run.py`, calling `train()` in
`--save-every`-sized chunks so the same `optimizer` object persists across calls — AdamW's
moments carry over rather than resetting each chunk). Then ran the first real training run
worth keeping.

**The real run**, p300c/Blackhole, `--steps 3000 --save-every 500 --batch-size 64`:

- First train loss `10.6875` — consistent with a near-uniform initial distribution (uniform
  over a 32000-token vocabulary would be `ln(32000) ≈ 10.37`; the measured value is 0.31 nats
  above that, still the expected ballpark for a freshly-initialized model, not an exact
  match), last train loss `1.9219`, real validation loss (our own `evaluate()`, 10 sampled
  batches, not `train()`'s placeholder) `1.8781` — perplexity `≈ e^1.8781 ≈ 6.5`. Validation
  coming in *below* that last train figure is expected, not a labeling bug: at 0.43 of an
  epoch there's no repeated exposure to the data to overfit on, dropout is 0.0 so there's no
  train-time regularization noise either, and the train figure is a single noisy mini-batch
  while the validation figure averages ten — comparing a lone sample to a ten-batch average
  will show sampling noise in either direction. That the two numbers differ at all (rather
  than being identical) is itself the evidence that `evaluate()` genuinely ran, instead of
  silently falling back to `train()`'s placeholder copy.
- **Windowed-mean loss curve** (300-step windows, reconstructed from all 3000 per-step
  readings): a clean log shape — steep for the first ~300 steps, then a steadily decelerating
  decline into a noisy 1.9–2.1 band for the back half — not a plateau, not divergence.

  | Steps       | Mean   | Min    | Max     |
  |-------------|--------|--------|---------|
  | 1–300       | 4.4747 | 3.0625 | 10.6875 |
  | 301–600     | 2.8236 | 2.5469 | 3.1875  |
  | 601–900     | 2.4894 | 2.2656 | 2.7344  |
  | 901–1200    | 2.3149 | 2.1250 | 2.4844  |
  | 1201–1500   | 2.1941 | 2.0469 | 2.3438  |
  | 1501–1800   | 2.1187 | 1.9688 | 2.2969  |
  | 1801–2100   | 2.0543 | 1.9141 | 2.2188  |
  | 2101–2400   | 2.0060 | 1.8750 | 2.2031  |
  | 2401–2700   | 1.9621 | 1.8594 | 2.1250  |
  | 2701–3000   | 1.9345 | 1.8125 | 2.0938  |

  As expected for a raw per-batch metric, the curve is *not* step-to-step monotonic: of 2999
  step-to-step transitions, 1359 went up and 1417 went down (223 unchanged), while every
  windowed average above kept falling.
- Steady state `~0.134 s/step` (7.44–7.50 it/s), matching Plan 2's 0.12–0.14 s/step. Unlike
  Plan 2, this run's first step showed no visible compiler-warmup stall — most likely because
  Task 2's same-shapes hardware runs earlier today had already warmed the on-disk kernel
  cache; not confirmed by inspecting the cache directly, so treat as a plausible explanation,
  not a verified one.
- Total wall clock `~6 min 47 s` (process start to final printed loss), including all six
  checkpoint writes — the writes are bounded below ~0.5 s per write. Measurement resolution
  here (~±0.25 s per 500-step chunk, from checkpoint-file mtime deltas) exceeds the actual
  effect size (~0.16 s implied by the chunk-to-chunk variance), so "no visible cost" would
  overstate what was measured; every 500-step chunk took the same ~67.1 s regardless, which is
  the tighter, honest claim.
- Six checkpoints, `nanollama3_step00000500.pkl` through `nanollama3_step00003000.pkl`
  (`artifacts/checkpoints/`, gitignored), 132,185,963 bytes each at the time of this run,
  final one at step 3000 as requested. (A later header back-fill — see the checkpoint-header
  fix wave below — added 339 bytes to each file's header record; tensor data is untouched.)

**This is a demonstration, not a capable model.** `3000 × 64 × 256 ≈ 49.2M tokens` is about
**0.43 of one epoch** over the 114.9M-token training split — this run never saw even half the
corpus once. TinyStories is also a synthetic, deliberately simple corpus (short children's
stories, small effective vocabulary, regular grammar, built so small models can fit it), so a
low loss here is expected and does not indicate general language competence — no text was
decoded from this checkpoint to check.

## `feat/checkpointing` — final whole-branch review fix wave (2026-08-11)

A final review before merge found the header schema above missing exactly what it existed
to prevent: architecture facts a converter can't recover without guessing. Fixed, no
training re-run (the box is shared; the six checkpoints above stand as the run's evidence):

- **Header now carries `intermediate_dim=1024`, `weight_tying=True`, `rms_norm_eps=1e-5`,
  `weights_dtype="bfloat16"`, and the full `transformer_config`.** The first three exist only
  as ttml C++ defaults (`modules/llama_block.cpp`, `models/llama.hpp`,
  `modules/rms_norm_module.hpp`) — nanollama3.yaml never sets them. `weight_tying` is the one
  that actually matters: because it's on, these checkpoints have no `llama/tok_emb/weight`
  tensor at all (confirmed against the manifest — 50 model tensors, none named `tok_emb`); a
  converter that didn't know would produce a model with a randomly-initialized embedding
  table and raise no error.
- **All six existing checkpoints were back-filled in place** with
  `scripts/backfill_checkpoint_headers.py` — a pure-CPU, stdlib-`pickle`-only rewrite of each
  file's header record, tensor bytes copied through unchanged. Verified byte-for-byte against
  a pre-backfill backup: the tail (everything after record 0) is bit-identical; only the
  header record grew, by exactly 339 bytes per file (132,185,963 → 132,186,302).
- **`total_tokens` (the whole corpus, 127,635,889) renamed to `corpus_tokens`**, with a new
  `batch_size` field and a derived `tokens_seen = step * batch_size * seq_len` — the number
  that actually describes training volume (49,152,000 at step 3000), which `total_tokens`
  was silently overstating by ~2.6x for anyone reading the header as a model-card source.
- **`latest_checkpoint()`'s docstring corrected** from "newest" to "highest-step" (it sorts
  by zero-padded step, not mtime) and `--resume` now prints the loaded header's `created_at`
  alongside the step, so an operator sharing `artifacts/checkpoints/` across runs can see
  which run's weights they actually got.
- **`checkpoint.load()` now validates the header before restoring any tensor**, not after —
  a bad or future-format header fails fast instead of first mutating the live model.
- **Added `convert/checkpoint_reader.py`**, a pure-CPU (`pickle`-only, no ttml/ttnn) reader
  for a checkpoint's header and tensor manifest — what the back-fill script needed, and the
  first piece of the CPU-side conversion path referenced in the design spec's Known Risks.

(Per-fix commands and the full post-back-fill verification log for this wave live in the
review's own working notes, not linked here — see this section for the numbers that matter
to anyone reading this file without that tree checked out.)

## `feat/hf-conversion` Task 3 — numerical verification finds a real bug (2026-08-11)

Task 3 added `tests/test_hf_parity.py` (4 tests, skip-guarded on `artifacts/hf/config.json`
existing) and ran the brief's Step 2/3 checks by hand. Structural checks and generation look
fine; the perplexity cross-check does not, and the gap traces to exactly the failure mode the
plan's Known Risks called out in advance.

**Structural re-derivation (independent of the controller's numbers, matched exactly):**
22.025088M params, embed/lm_head tied, `model.norm.weight` shape `(384,)`, next-token entropy
on "...a little girl named" = **4.7509 nats** (uniform ceiling 10.37), top-5
`[' Lily',' Lucy',' Jane',' Sue',' Sarah']`.

**Generated sample** (brief's exact command, `do_sample=True, temperature=0.8, top_p=0.95`,
no seed set, reported verbatim, not cherry-picked):

> Once upon a time, there was a little girl named Lily. Lily had a pretty flower. She loved
> to dance. She loved to dance. One day, she danced every day, she found a big, blue flower.
> The flower was very pretty flower in the sun, her dress, and it had a big,

Locally fluent TinyStories-flavoured English, globally drifting and repetitive ("She loved to
dance. She loved to dance.") — the expected shape for 0.43 of an epoch on a 22M model, not a
sign of a layout trap on its own.

**Perplexity cross-check — the brief's own Step 3 command has a bug.** Its literal code
(`m(x[:, :-1], labels=x[:, 1:])`) pre-shifts both `input_ids` and `labels` by hand, but
`LlamaForCausalLM`'s internal loss function (`ForCausalLMLoss` in
`transformers.loss.loss_utils`) *also* shifts `labels` internally before computing
cross-entropy. Passing already-shifted tensors through `labels=` double-shifts, comparing
each prediction against the token two positions ahead instead of one. Run literally, it
reports **8.53 nats** — worse than doing nothing. Verified against two independent correct
formulations (`model(x, labels=x)`, which lets HF do its one intended shift, and manual
`cross_entropy` on `logits` vs. `x[:, 1:]` with no `labels=` kwarg at all): both agree at
~3.19–3.20 nats on the same data, confirming the double shift as the reason 8.53 differs from
the corrected number, not a second bug.

With the shift bug fixed, and sampling matched to how the training run's own `evaluate()`
computes 1.8781 (`ttml.common.data.get_batch`: 10 batches of 32 random 256-token windows
drawn uniformly across the *full* 12.76M-token validation set, not one contiguous block) —
**HF-side val loss = 3.20 nats** (range 3.13–3.27 across the 10 batches). That is **1.32 nats
above 1.8781** — a fail by the brief's own "1+ nats means the conversion is wrong somewhere
the entropy check didn't catch" threshold. The entropy and generation checks above did not
catch this; that is exactly why Task 3's Step 3 exists.

**Root-caused via the brief's own Known Risks, without touching `artifacts/hf/` or any
tracked file** (all work done against scratch copies under
`/tmp/.../scratchpad`, using `convert.to_hf.convert_checkpoint` called directly):
- *Gate/up swap, ruled out.* Swapping `w1`/`w3` in `MLP_ROLES` in-process and reconverting
  made loss **worse** (3.63 nats), not better — the current `w1=gate_proj`/`w3=up_proj`
  assignment in `convert/hf_mapping.py` is correct.
- *RoPE interleaved-vs-split-halves, confirmed as the cause.* `convert/to_hf.py` copies
  `q_proj`/`k_proj` weight rows straight through with no permutation. Applying the classic
  Meta-Llama interleaved→split-halves permutation (reshape each head's rows as
  `(head_dim/2, 2)`, transpose, flatten — the same operation Meta's own
  `convert_llama_weights_to_hf.py` applies) to `q_proj` and `k_proj` in a scratch copy of the
  converted weights, with everything else unchanged, brought the same random-batch loss
  measurement down to **1.927 nats** (range 1.83–2.00) — within 0.05 nats of 1.8781, a clean
  pass. No source file or artifact was modified to get this number; it is a diagnostic
  reconversion in `/tmp` only.

**Conclusion at this point in the investigation: `artifacts/hf/` was measurably wrong.** It
loaded without error, tied weights correctly, produced finite non-uniform logits, and
generated plausible-looking text — every check Tasks 1–2 could have run would pass — but its
RoPE layout did not match ttml's convention, which silently degrades attention quality without
producing garbage output. Task 1/2's test suites (`test_hf_mapping.py`, `test_to_hf.py`) had
no test that would catch this; `rope_theta` is checked, the row layout within each head was
not. This was reported to the controller rather than patched immediately, since fixing it
meant writing to `artifacts/hf/`, which was off-limits under Task 3's constraints. The
controller authorized the fix; see the next section for what shipped.

## `feat/hf-conversion` Task 3 fix — RoPE row permutation, `artifacts/hf/` regenerated (2026-08-11)

Fixed for real, with the controller's authorization to write `artifacts/hf/` (the earlier
prohibition was to protect a directory believed validated; Task 3 showed it wasn't).

**The fix: `convert/hf_mapping.permute_rope_qk`.** ttml's `q_linear`/`k_linear` rows are
ordered for RoPE's *interleaved* pairing (row `2i` pairs with row `2i+1`); HF Llama's
`rotate_half` expects *split-halves* pairing (row `i` pairs with row `i + head_dim // 2`). A
weight matrix carries no signal of which convention its author assumed, so this was invisible
to every shape/name check — the tensor was the right shape, in the right place, with the
right name. The permutation (`reshape(num_heads, head_dim//2, 2, in_features).transpose(0, 2,
1, 3).reshape(out_features, in_features)`) is the same row reordering Meta's own
`convert_llama_weights_to_hf.py` applies when converting original-format (interleaved) Llama
checkpoints — not invented for this project. `num_heads` and `head_dim` come from the
checkpoint header's `transformer_config` (`num_heads` for `q_proj`, `num_groups` for
`k_proj`, both times through `config["num_attention_heads"]`/`int(tc["num_groups"])` in
`convert/to_hf.py`) — never hardcoded — so a differently-shaped future model gets the right
block size automatically. `v_proj` is untouched: RoPE rotates queries and keys before the
attention dot product, values pass through unrotated. 5 new tests in `test_hf_mapping.py`
pin the permutation's shape, that it's a true row permutation (no row dropped or duplicated —
checked by comparing row sets, not just `.shape`), a hand-verified example of which rows move
where, and that it uses whatever head count it's given rather than an assumed 6/3/64.

**`artifacts/hf/` regenerated** via `python scripts/convert_checkpoint.py` from the same
`nanollama3_step00003000.pkl` checkpoint (untouched — `artifacts/checkpoints/` remained
off-limits throughout). `model.safetensors` is a fresh file; everything else about the
pipeline (tokenizer files, config assembly) is unchanged.

**Re-verified end to end, same methodology as before, same checkpoint:**

| Check | Before fix | After fix | Target |
|---|---|---|---|
| Entropy, "...a little girl named" | 4.7509 nats | 4.9765 nats | < 7.0 (uniform 10.37) |
| Top-5 | `Lily,Lucy,Jane,Sue,Sarah` | `Lily,Lucy,Jane,Sue,Mia` | — |
| HF-side val loss (10×32×256, matched sampling) | 3.20 nats | **1.927 nats** | 1.8781 ± 0.2 |

The loss lands 0.049 nats from target — a clean pass, and it exactly reproduces the 1.927
measured in the earlier scratch-copy diagnostic (same checkpoint, same fix, same code path,
so this is confirmation the regeneration applied the fix correctly, not a new independent
result). Entropy moved a little (4.75 → 4.98 nats) but stayed far below the 7.0 test threshold
and the 10.37 uniform ceiling — a properly-rotated attention mechanism sharpens the
prediction slightly further, as expected, though this single number was never going to be
the thing that caught the bug.

**New sample, same command, no seed, verbatim, not cherry-picked:**

> Once upon a time, there was a little dog named Max. Max loved to play with his ball. One
> day, Max saw a big ball in the park. Max wanted to play with the ball, but he was very
> dirty. Max had an idea. He would push the ball with his paws to clean it.

**Does it look better, or just different?** Read honestly: this sample keeps one character
and one throughline for its whole length (dog wants to play with a dirty ball, forms a plan
to clean it) with no verbatim-repeated sentences, where the earlier sample looped ("She loved
to dance. She loved to dance.") and lost its thread in the last clause. On this single
comparison it reads as more coherent, not merely different — but it's one temperature-0.8
sample against one other temperature-0.8 sample, and generation is stochastic, so this is a
data point, not proof that every sample from the fixed model beats every sample from the
broken one. The loss number (3.20 → 1.927 nats, a real and repeatable difference on 320
held-out windows) is the reliable evidence; the prose is corroborating, not dispositive. This
matches the general shape of the lesson regardless of which single sample happened to land
better: structural checks and even a read of the generated text are not enough on their own
to confirm a conversion is right, which is the entire reason Task 3's numerical comparison
exists.

**Regression test added:** `test_hf_parity.py::test_validation_loss_matches_the_training_run`
computes the same 10×32×256 random-window loss and asserts it's within 0.2 nats of 1.8781,
skip-guarded (separately from the module's `artifacts/hf/`-existence guard) on
`artifacts/tokens/val_ids.npy` existing. This is the test that would have caught the RoPE bug
before it ever reached a report — nothing in the suite pinned this number before now.

**The brief's own Step 3 example command was also fixed**, in
`docs/superpowers/plans/2026-08-11-hf-conversion.md`, to remove the double-shift bug found
while executing this task (see the section above) and to match the training run's random
sampling, so the next person to read the plan doesn't inherit either defect.

Test suite: **108 passed** (103 from the numerical-verification commit + 4 new in
`test_hf_mapping.py` for `permute_rope_qk` + 1 new regression test in `test_hf_parity.py`),
0 skipped (converted model and validation tokens both present on this machine), 0 failed.

## `feat/hf-conversion` — pre-merge whole-branch review fix wave (2026-08-11)

A final review before merge (verdict: ready to merge) found five cheap, high-value items.
None required a training re-run; `artifacts/checkpoints/` was never touched.

**The 0.049-nat residual gap's documented cause was wrong.** Both `test_hf_parity.py`'s
`LOSS_TOLERANCE` comment and this file's Task 3 write-up (above) left the residual gap
between 1.9271 (converted model) and 1.8781 (training run) attributed to, or open to being
read as, fp32-CPU-vs-bf16-device precision. Measured directly instead: same seed, same
windows, bf16 gives 1.9315 and fp32 gives 1.9314 — dtype accounts for roughly 1e-4 nats, not
0.049. The real driver is sampling: seeds 0, 1, and 2 against the same model give 1.9314,
1.9208, and 1.8856 nats respectively — a seed-to-seed standard deviation of 0.024 nats, which
puts the 0.049-nat gap at roughly z ≈ 1.2. That's unremarkable noise, not a signal, and
correcting the *reason* matters even though the pass/fail verdict doesn't change: attributing
a real, measured 0.024-nat seed-to-seed spread to a nonexistent precision effect would send
the next person chasing bf16/fp32 numerics instead of understanding that the regression
test's fixed seed (`np.random.default_rng(0)`) is deliberately pinned for exactly this
reason — an unpinned seed would make the 0.2-nat gate flakier for no benefit. `LOSS_TOLERANCE`
stays at 0.2; only the explanation changed, in `tests/test_hf_parity.py` and in
`.superpowers/sdd/2026-08-11-hf-conversion/task-3-report.md`'s concern #4.

**Four other fixes, in brief:**
- **README's Status section was publicly stale.** It said conversion and packaging were both
  still pending and "no text has been decoded" — both false since the Task 3 fix wave above.
  Updated to state the conversion is numerically verified (1.9271 vs. 1.8781) and weights
  remain unpublished with tt-kernel packaging as the sole remaining stage, keeping the
  existing honest capability framing and citing the Max-the-dog sample as one data point.
- **`convert_checkpoint` now raises on unmapped ttml tensors** instead of silently
  `continue`-ing past them, and **`llama/fc/weight`'s fan-out to both HF embedding slots is
  now conditional on `header["weight_tying"]`** rather than unconditional. Untied models
  (`llama/tok_emb/weight` present alongside `llama/fc/weight`, per
  `ttml/models/llama.cpp:466`) were the real risk this closes: before this fix, the real
  embedding table would be silently dropped as "unmapped" while `fc/weight` was written to
  *both* `model.embed_tokens.weight` and `lm_head.weight`, producing a model that loads
  cleanly, reports `tie_word_embeddings: false`, and is numerically wrong with no error at
  any stage. Every real checkpoint produced so far has `weight_tying=True`, so this path was
  untested until now — new tests use a synthetic untied manifest rather than attempting to
  produce a real untied checkpoint.
- **`convert_checkpoint` now verifies the emitted HF key set is exactly what the config
  implies** (`9 × num_hidden_layers + 3`) before writing, raising with the missing/unexpected
  names. Previously a truncated manifest would silently produce a safetensors file missing
  keys, and `transformers` would randomly initialize them with only a warning.
- **`build_config`'s hardcoded `bos/eos/pad = 1/2/3` is now checked, not just trusted.**
  `convert_checkpoint` cross-references `tokenizer_dir`'s `special_tokens_map.json` /
  `tokenizer_config.json` and raises if the resolved ids disagree with the hardcoded values.
  Verified correct against `artifacts/tokenizer/` today; this is a guard against silent drift
  if the tokenizer is ever regenerated with different special-token ids, not a refactor.

Re-ran `scripts/convert_checkpoint.py` against the same `nanollama3_step00003000.pkl`
checkpoint (Fixes 3-5 touch the write path); `AutoModelForCausalLM.from_pretrained` still
loads `artifacts/hf/` and `test_validation_loss_matches_the_training_run` still passes.

## `feat/numpy-parity` — an independent NumPy reference, and a sharper gate (2026-08-12)

**Why this plan exists.** The HF-conversion loss gate (`test_hf_parity.py`, 0.2-nat
tolerance) caught the RoPE bug above, but two things about it are uncomfortable: its 2σ
floor is ~0.22 nats (sampling sd 0.024 × ~9), so anything cheaper than that is invisible; and
**all 13 RMSNorm gammas in the trained checkpoint are exactly 1.0** (an upstream
`stochastic_rounding` issue — see `docs/superpowers/specs/2026-08-11-followups.md`), so
swapping two norms' destinations changes loss by exactly `0.0000`. 23% of the conversion's
mapping decisions were validated by nothing.

Three tasks: **Task 1** derived ttml's forward pass straight from its C++ source into
`docs/ttml-forward-reference.md`, deliberately never reading `convert/hf_mapping.py` or
`convert/to_hf.py` — a NumPy path built from that converter would just agree with its own
misunderstandings. **Task 2** implemented `convert/ttml_forward.py` from that doc and
validated it independently by reproducing the training run's own held-out cross-entropy
(1.8488 measured, vs. training's 1.8781 — see the Task 2 report). **Task 3** (this section)
built the actual instrument: `tests/test_numpy_parity.py`, comparing the NumPy path's logits
against `artifacts/hf/`'s logits directly, at a tolerance measured from data rather than
picked in advance.

### What the parity gate measures, and the tolerance

Both paths run on the host in **float32** (`AutoModelForCausalLM.from_pretrained(...,
torch_dtype=torch.float32)`), from the same bfloat16-stored checkpoint weights, on a fixed
seeded window of `val_ids.npy`. This is a **NumPy-vs-HF** comparison, not NumPy-vs-device —
the earlier (and wrong) worry that a bf16 RMSNorm divisor makes ~1e-3 unreachable bounds a
device comparison, not this host-vs-host one.

Measured across six seeds/windows before picking a number: max absolute logit difference
**5.2e-6 to 8.5e-6**; max relative difference (restricted to `|logit| > 0.01` — unrestricted
relative error is dominated by meaningless blowups near zero-crossings, e.g. two logits of
-1.05e-5 and -1.15e-5 differ by "8%" while being numerically indistinguishable) **1.4e-4 to
4.7e-4**; correlation indistinguishable from 1.0 (`1 - corr ≈ 1e-13`). A NumPy-vs-NumPy
control (float32 throughout vs. bf16-rounded activations at every sub-layer boundary) showed
this precision effect alone would cost ~0.03 absolute and ~3-4 orders of magnitude more than
the actual NumPy-vs-HF gap — confirming the tight agreement is real, not an artifact of both
sides sharing rounding.

Tolerances set from that (all in `tests/test_numpy_parity.py`): `MAX_ABS_TOLERANCE = 1e-3`
(~100-200x the measured worst case), `MAX_REL_TOLERANCE = 5e-3` floored at `|logit| > 0.01`
(~10x the measured worst case, and tighter than the plan's own ~1e-2 "something is wrong"
ballpark by 2x), `MIN_CORRELATION = 0.9999`. Wide margin above measurement noise, and — per
the not-hollow proof below — many orders of magnitude below what an actual bug produces.

**Proof the gate is not hollow.** Monkeypatching `permute_rope_qk` to the identity function
(Plan 4's exact historical bug — straight-copied RoPE rows) and reconverting into a scratch
directory (never touching `artifacts/hf/`) produced max_abs = **4.60**, correlation =
**0.972** — ~4600x over the abs tolerance's budget. Plan 4's reviewer measured the same bug
at the loss level as 3.2015 nats against a 1.8781 target; this gate catches the identical
defect at the logit level, by a much larger margin relative to its own tolerance than the
loss gate had relative to its.

### What this gate still cannot see

1. **The norm-mapping blind spot this plan exists to close.** On the real checkpoint (all
   gammas exactly 1.0), swapping two RMSNorm gammas' HF destinations is a no-op — verified
   directly: swapping block 0's and block 1's `input_layernorm.weight` mapping and
   reconverting gives max_abs = 1.23e-5, identical (to the precision measured) to the no-swap
   baseline on this test's seed/window, because the swap moves nothing between two gammas
   that are both exactly 1.0. **The parity gate is exactly as blind to this as the loss gate
   is, on this checkpoint, for the same reason** (`test_parity_gate_is_blind_to_a_norm_swap_on_the_real_checkpoint`
   measures and confirms this rather than assuming it). Closed *structurally* instead:
   `test_convert_checkpoint_places_each_rmsnorm_gamma_at_its_correct_destination` builds a
   synthetic checkpoint with distinct non-unit gammas and asserts each lands at its correct
   HF tensor name — a test that runs unconditionally (no `artifacts/` dependency) and stays
   meaningful regardless of whether the real checkpoint's gammas ever stop being degenerate.
2. **What neither path implements is invisible to both.** If both the NumPy reference and
   the converter drop RoPE scaling (the checkpoint header records no `scaling_factor`), they
   agree with each other and are both wrong relative to ttml's actual runtime behaviour. This
   harness validates the *conversion* — does `convert/` correctly translate ttml's checkpoint
   into HF's format — not the checkpoint's *completeness*.
3. **ttnn's own accumulation/output dtype on real hardware is untraced.** Both paths compared
   here run entirely on the host; neither touches a Tenstorrent device. This says nothing
   about NumPy-vs-device agreement, which needs its own (looser) tolerance — Task 1's finding
   that ttml's RMSNorm kernel packs its mean divisor as bfloat16 bounds *that* comparison, not
   this one.
4. **This compares two implementations of the same architecture, not the checkpoint against
   ground truth.** If ttml's own forward pass itself has a bug relative to what the training
   run's loss curve implies, nothing here catches it — that anchor is Task 2 Step 3's
   independent cross-entropy check against the training run's own held-out figure, not this
   gate.
5. **`convert/checkpoint_reader.py` is a shared dependency of both paths.** The NumPy path
   (`convert.ttml_forward.forward`) and the HF path (`convert.to_hf.convert_checkpoint`) both
   call `convert.checkpoint_reader.read_tensors`, which owns the name↔tensor association and
   the declaration-order stream walk over the checkpoint's pickle records. A misassignment or
   a stream-order error there is **common-mode**, not independent: both paths would read the
   same wrong tensor under the same name and agree perfectly while both being wrong. This is
   the one piece of "independence" this plan does not actually have — the two paths are
   independently *derived* downstream of the reader, not independently *reading the
   checkpoint*. It is anchored only by the coarser CE test (Task 2's held-out cross-entropy
   check, floor ≈0.22 nats) and by `test_checkpoint_reader.py`'s own ordering tests, not by
   the parity gate, which cannot see it by construction.

### The standing skip-guard gap, partially addressed

Task 2's review noted the decisive tests are all `skipif`-guarded on `artifacts/` (gitignored),
so a CI run can report "N passed" while nothing load-bearing executed. The synthetic
gamma-mapping test above is deliberately **not** guarded — it needs no real artifacts and
runs every time — but every other test in `test_numpy_parity.py` still needs the real
checkpoint, tokenizer, converted `artifacts/hf/`, and `val_ids.npy`, and is still skip-guarded
on them. The gap is not closed, just narrowed by one genuinely-unconditional test.

Test suite: **151 passed** (146 pre-existing + 5 new in `test_numpy_parity.py`), 0 skipped, 0
failed on this machine (all `artifacts/` fixtures present); the synthetic gamma test alone
would still pass, unconditionally, on a machine with none of them.

## `feat/numpy-parity` — pre-merge whole-branch review fix wave (2026-08-12)

A final review before merge (verdict: ready to merge, with fixes) found seven small items —
none changing a measured number, several requiring one. No training re-run;
`artifacts/checkpoints/` and `artifacts/hf/` were never touched.

- **The parity window was widened from 64 to 256 tokens** — the model's full
  `max_position_embeddings`. At 64, positions 64-255 were covered only by the CE check's
  0.22-nat floor; a position-dependent defect (e.g. a RoPE angle that drifts with sequence
  length) could have hidden there. Re-measured at 256 tokens across seven seeds: max_abs
  ranges ~8.3e-6 to ~1.6e-5, max_rel ~3.0e-4 to ~5.6e-4, correlation indistinguishable from 1
  (1 - corr ~1.2 to 1.3e-13) — still ~60-120x inside the gate's tolerances. The whole file
  still runs in ~5.8s.
- **Every measured figure in `test_numpy_parity.py`'s docstrings now matches what the
  committed test configuration actually produces** — the previous headline table quoted a
  256-token measurement (8.46e-6) that the committed 64-token test could not itself produce
  (its own number was 5.26e-6/2.13e-4). In a branch whose thesis is "measured, not assumed,"
  the number a future debugger compares a failure against has to be reproducible by running
  the test, not a number from a related but different configuration.
- **The independence claim was one dependency too strong.** Both the NumPy path and the HF
  path call `convert.checkpoint_reader.read_tensors`, which owns the name↔tensor association
  and the declaration-order stream walk — a bug there would be common-mode, producing two
  identically-wrong paths that agree perfectly. `docs/ttml-forward-reference.md` was already
  honest about this; CLAUDE.md's "What this gate still cannot see" list and the plan's
  "different routes" framing were not, so both now name `checkpoint_reader` as a shared
  dependency explicitly (see item 5 in the list above and the plan's independence section).
- **The Global Constraint "`convert/` must NOT import `ttnn` or `ttml`" had no test for two
  of its four target modules.** `convert.checkpoint_reader`, `convert.tokenizer`, and
  `scripts/backfill_checkpoint_headers.py` all had the subprocess-probe test; `convert.ttml_forward`
  and `convert.to_hf` did not, despite `ttml_forward.py`'s own docstring claiming (in a
  garbled sentence that conceded as much on close reading) that something checked it.
  Added `test_ttml_forward.py::test_ttml_forward_module_imports_no_tenstorrent` and
  `test_to_hf.py::test_convert_to_hf_module_imports_no_tenstorrent`, same pattern, and fixed
  the docstring.
- **Added a fourth not-hollow proof: the epsilon-placement probe.** Task 1 found epsilon
  moved outside the sqrt the *one* perturbation invisible to the CE check (Δ = -0.0002 nats).
  `test_parity_gate_is_not_hollow_it_catches_epsilon_moved_outside_the_sqrt` monkeypatches
  `rms_norm` for one `forward()` call (no reconversion needed) and measures max_abs = 0.0370
  — 37x over the 1e-3 budget. Notably, correlation stays at 0.9999985, *above*
  `MIN_CORRELATION`'s 0.9999 floor: this defect is loud in absolute/relative terms while
  leaving the logit *shape* almost undisturbed, unlike the RoPE bug (corr ~0.93). Documented
  in `convert/ttml_forward.py` that `RMS_NORM_EPS` being a plain hardcoded module constant
  (not threaded from the header) is what makes this probe constructible — a future "read it
  from the header" cleanup would silently remove this coverage.
- **Strengthened the norm-swap blindness test with a byte-identical hash check.** The
  previous version passed identically whether its monkeypatch fired or silently didn't — no
  non-vacuity guard. Now converts both the unpatched and norm-swapped checkpoints into
  separate throwaway directories and asserts their `model.safetensors` files are
  SHA-256-identical (measured: both hash to `3a85bb08e1d2...490462200d`) — proof that *no*
  instrument could see the swap, not just that this one didn't. All three logit metrics
  (`max_abs`, `max_rel`, `corr`) are now asserted, making the "same tolerance" comment
  literally true.
- **Two documentation corrections:** renamed `attention()`'s local `embedding_dim` (q's
  *out-features*, not the model's embedding dim — numerically equal only because
  `head_dim == hidden/num_heads` on this architecture) to `q_out_features`; and
  `docs/ttml-forward-reference.md` §10's summary row and closing line no longer read as if
  the NumPy-vs-HF numerical tolerance were still an open problem — Task 3 resolved it at
  ~5e-6 to ~1.6e-5, two orders of magnitude inside 1e-3. Q1's own body was already correctly
  scoped to the NumPy-vs-*device* comparison and did not need to change.

Test suite: **154 passed** (151 pre-existing + 3 new: the epsilon-probe test and the two
import-purity tests), 0 skipped, 0 failed.

## `feat/packaging` Task 1 — repairing the HF artifact before publication (2026-08-12)

**Why this task exists.** Publication (Task 2+) is gated on the artifact being clean —
three defects, all cheap to fix now and expensive after weights are public, were found
during packaging-plan review of `artifacts/hf/`.

**Fix 1 — `generation_config.json` was missing entirely.** `transformers` logs its absence
and falls back to `config.json`'s token ids. `convert_checkpoint` now builds one via
`transformers.GenerationConfig(bos_token_id=..., eos_token_id=..., pad_token_id=...)` and
calls `.save_pretrained(out_dir)` — using the library's own class rather than a hand-rolled
dict so the on-disk shape matches whatever this environment's `transformers` considers
standard. The three ids are read from `config` (the exact dict `build_config` returned,
already written to `config.json`), not re-derived from `_BOS_TOKEN_ID` et al. directly, so
the two files are structurally incapable of disagreeing.

**Fix 2 — `tokenizer_config.json` declared the wrong class.** `convert/tokenizer.py` exports
via `PreTrainedTokenizerFast.save_pretrained()`, which writes `tokenizer_class:
"PreTrainedTokenizer"` — `transformers` strips the `Fast` suffix on save, an upstream quirk,
not a bug in this project's export code. The tokenizer actually loads as
`PreTrainedTokenizerFast`. Corrected **on the copy in `out_dir` only**, after
`convert_checkpoint`'s existing `shutil.copy2` — `artifacts/tokenizer/` is a separate
artifact on its own publication schedule, and patching it there would invalidate its own
tests. Verified the source is untouched: `artifacts/tokenizer/tokenizer_config.json` still
reads `tokenizer_class: "PreTrainedTokenizer"` after conversion.

**Fix 3 — no guard tied `max_position_embeddings` to the checkpoint's trained sequence
length.** The real trap: `tokenizer_config.json` advertises `model_max_length:
1000000000000000019884624838656` (transformers' "no limit" sentinel), so a serving stack
that derives `max_model_len` from the tokenizer instead of `config.json` would silently
accept ~4k-token contexts from a model trained to a 256-token window — degraded output, no
error (`scripts/chat.py` already carried a comment about exactly this). `build_config`
already derives `max_position_embeddings` from `header["seq_len"]`, so in normal operation
the two can't disagree; the new check in `convert_checkpoint` raises `ValueError` (not a
bare `assert` — see the project's global guard convention) if they ever do, so a future
change to `build_config` that breaks that derivation fails loudly at conversion time rather
than silently at serving time. `test_convert_checkpoint_raises_if_max_position_embeddings_disagrees_with_header`
proves the check is reachable by monkeypatching `build_config` to tamper with its own
output.

**The duplicate-embedding question (plan Task 1 Step 3) — settled empirically, not
re-litigated.** `model.safetensors` was 68.6 MB for a 44 MB model because
`lm_head.weight` duplicated `embed_tokens.weight` under `tie_word_embeddings: true`. Measured
directly before implementing (see `.superpowers/sdd/2026-08-12-packaging/progress.md`):
dropping `lm_head.weight` takes the file from 57 tensors / 68,632,400 B to 56 tensors /
44,056,336 B (36% smaller), `AutoModelForCausalLM.from_pretrained` still loads with **no**
warnings, `torch.equal(embed_tokens.weight, lm_head.weight)` is still `True` after load
(`transformers` reconstructs `lm_head` from the tied embedding), and logits are
bit-identical (max diff 0.0). Implemented as unconditional-on-tying: the tied-embedding
branch in `convert_checkpoint`'s tensor-assembly loop now writes only
`model.embed_tokens.weight`; the untied path (`tok_emb` → `embed_tokens`, `fc` → `lm_head`,
two genuinely distinct tensors) is unchanged. The completeness post-condition's expected-key
count is now conditional on `weight_tying` — `9 × num_hidden_layers + 2` (embed_tokens,
norm) when tied, `+ 3` (adding `lm_head`) when untied — rather than a single hardcoded `+ 3`
that would have made a correct tied conversion fail its own completeness check.

**Verification.** Full suite: **164 passed** (154 pre-existing + 10 new in
`tests/test_to_hf.py`), 0 failed. `tests/test_numpy_parity.py` (the gate that actually proves
numerical correctness, independent of everything else in this task) still passes after
regeneration. Regenerated `artifacts/hf/` via `scripts/convert_checkpoint.py`:
`model.safetensors` 44,056,304 bytes (56 tensors, no `lm_head.weight`), new
`generation_config.json` (`{bos,eos,pad}_token_id` = 1/2/3), `tokenizer_config.json`'s
`tokenizer_class` now `PreTrainedTokenizerFast`, `config.json`'s `max_position_embeddings`
still 256. `scripts/chat.py` smoke-tested against the regenerated artifact: loads with no
warnings, reports `context 256`, generates coherent completions.

## `feat/real-training` — the multi-epoch run: gammas fixed, model measurably better (2026-08-12)

Three tasks. Task 1 fixed the frozen-gamma bug found in `feat/numpy-parity`'s postmortem
(`stochastic_rounding: true`, `train/configs/nanollama3_bpe_v2.yaml`). Task 2 added periodic
real validation (`--val-every`) and an unconditional startup warning when
`stochastic_rounding` is disabled. Task 3 ran the real thing. **Fix round 1 (below) corrects
two overstated interpretations an independent review caught — see
`task-3-review.md` for the full evidence; every underlying measured number was reproduced
exactly and none of them changed.**

**Before the run:** disk was at 98% / 93 GB free (down from the ~140 GB the brief cited at
dispatch). First checkpoint measured (not assumed) at 132,186,302 bytes; `AdamWFullPrecision`
was never needed (Task 1's fix keeps the format unchanged), so all 11 checkpoints from this
run are that same size — 1.454 GB total, in line with the brief's ~1.3 GB estimate.

**The run:** `python train/run.py --config train/configs/nanollama3_bpe_v2.yaml --steps
21034 --save-every 2000 --val-every 1000 --batch-size 64 --checkpoint-dir
artifacts/checkpoints-v2` — 21,034 steps (3.000036 epochs over the 114.9M-token training
split), one p300c, 47m13s wall clock, ~7.42 steps/s. `stochastic_rounding: True` confirmed
at startup before trusting the run; the step-2000 checkpoint's gammas were checked for
degeneracy (sd range 2.02e-2..8.21e-2, all nonzero) before letting the remaining ~43 minutes
proceed. **`artifacts/checkpoints/` was never touched by this task** (still `2026-08-11
17:57`, unchanged). **`artifacts/hf/` was also not written by this task**, but its files do
carry today's date (08:21) — that's Plan 6 Task 1's own deliberate regeneration, an hour
before this run started, not something Task 3 did.

**The curve:** train loss 10.6875 → 1.375. Validation fell steeply for ~8000 steps
(2.1969 → 1.5695) then flattened for the remaining 13,000 steps (62% of the run) into a
1.46–1.59-nat band (one outlier at step 9000, 1.59375; excluding it, the rest sit in
1.45–1.53) while train loss kept falling — a mild overfitting signature, but not a clean
"turn": the best val value (1.4563 @ step 17,000) and the final one (1.4602 @ step 21,034)
differ by only 0.004 nats, well inside noise. Full curve (22 points) in
`artifacts/checkpoints-v2/val_losses.jsonl` and `.superpowers/sdd/2026-08-12-real-training-run/task-3-report.md`.
Read plainly: this corpus/architecture pair has largely exhausted what steps 8000–21,034 had
left to teach it about held-out loss — evidence for Plan 8's dataset-blend rationale, not for
training longer on the same mix.

**Paired comparison (the number that matters):** `convert/ttml_forward.py`'s pure-NumPy
forward pass, 32 seed-0 256-token windows, baseline (`nanollama3_step00003000.pkl`) vs. new
final checkpoint. Baseline mean CE 1.8733 (sd 0.3242, reproducing the brief's cited
1.8781/0.315); new mean CE 1.4228 (sd 0.2908). **Paired diff (baseline − new): +0.4505 nats,
sd of the paired differences 0.0878, SE 0.0155 — every one of 32 windows favors the new
checkpoint.** The two models' per-window losses correlate at r = 0.9651 (hard windows are
hard for both), which is *why* the paired sd (0.0878) is the right yardstick here and not
either model's own unpaired sd (~0.30–0.32) — comparing the paired difference against an
unpaired sd is exactly the mistake this brief warned about, and an earlier draft of this
entry made it (0.27 does not exceed 0.315). Measured correctly: even the **smallest**
per-window improvement (0.2709 nats) is **3.1 paired sds** above zero, and the mean is
**29 SE** from zero (0.4505 / 0.0155). Not noise, by a wide margin.

**Norm-swap ablation, re-measured — two swaps, and the result is real but far below what
this project's tests can detect.** `docs/model-development-troubleshooting.md`'s "+0.0000 ←
blind spot" row and the pinned `test_parity_gate_is_blind_to_a_norm_swap_on_the_real_checkpoint`
both refer to the **canonical** swap — block 0 ↔ block 1 `attention_norm`/`input_layernorm`
gammas, confirmed here still exactly 0.000000 on the baseline. On the new checkpoint that
canonical swap now costs **+0.00652 nats** (sd 0.0066, t = 5.6, 25/32 windows worse). A
second, gentler within-layer swap (block 3's `attention_norm` ↔ `mlp_norm`, the one this
task originally measured) costs **+0.0018 nats** (sd 0.0040, t = 2.5, only 22/32 windows
worse — 10 of 32 actually get *better*, an inconsistent sign that a small sample has a real
chance of reading backwards). **Neither swap is anywhere close to something this project's
loss-based checks would actually catch**: even the larger, canonical effect (0.0065) is ~45×
below either model's per-window sd (~0.29–0.32) and ~31× below the project's own 0.2-nat
detection floor; the smaller swap is ~163× and ~113× below those same floors respectively.
Correctly stated, the Plan-4 blind spot is closed only in the narrow sense that the number is
no longer *identically* zero — **a norm mis-mapping of this kind still slips past every
loss-based gate this project ships**, at the sample sizes those gates actually use (8–32
windows). This ablation alone remains a weak-to-useless instrument for this error class; the
structural/permutation tests (`test_hf_mapping.py`, `test_numpy_parity.py`'s per-destination
gamma checks) stay the actually-reliable defense. (The companion HF-parity-gate figure for
the canonical swap was not re-measured on this checkpoint; the previously-cited `5.86e-6` is
inherited from the plan and does not match the parity test's own docstring number for the
same swap, so it is dropped here rather than repeated unverified.)

**Generated samples, same prompt (`"Once upon a time, there was a little"`), verbatim, not
cherry-picked, unseeded (`do_sample=True, temperature=0.8, top_p=0.95`, no seed — not
reproducible by construction):**

> Baseline (`artifacts/hf`, step 3000): Once upon a time, there was a little girl named Lucy.
> Lucy loved to play with her toys. One day, Lucy saw a big, thick, pretty toy in the box.
> Lucy wanted to play with the toy, so she went to the box and pushed it with her hands. The
> toy made a loud noise and stopped working. Lucy

> New (step 21034), sampled from `artifacts/hf-v2-scratch/` — a scratch conversion made only
> for this comparison, which has **not** been through the parity gate (that gate is pinned to
> the baseline checkpoint): Once upon a time, there was a little boy named Tim. Tim loved to
> play with his toys. One day, Tim saw a big, high chair in the store. He wanted to ride the
> chair, but it was too high for him. Tim saw a tall man named Bob. He asked, "Bob, can you
> help me get

**Visibly better, or only numerically better?** Mostly the latter. Both samples are fluent,
loop-free, single-character TinyStories prose — the new one has a slightly clearer causal
chain on this one draw, but it is a difference of degree, not a qualitative leap. The ~24%
relative reduction in held-out cross-entropy (every window improved) is real, repeatable, and
verified straight from the `.pkl` checkpoints (unaffected by the sample's own unverified
conversion); the prose improvement is real but easy to miss without the paired numbers. Both
are honest findings, not a contradiction.

Test suite: **176 passed, 0 skipped** on this machine (`test_checkpoint_gammas_are_not_degenerate[checkpoints-v2]`
stops skipping once `artifacts/checkpoints-v2/` exists locally — no test was added by this
task, so a fresh clone without `artifacts/` will still show skips). Full detail:
`.superpowers/sdd/2026-08-12-real-training-run/task-3-report.md`.

## `feat/corpus-assembly` — nine sources, a token budget, and a manifest that has to be true (2026-08-13)

The corpus stopped being "TinyStories" and became a blend: nine licence-audited sources
mixed to a 400M-token budget, with a provenance manifest whose entire purpose is to make
*"what was this model trained on"* exactly answerable. Eight tasks: broaden `spine`, strip
residual Gutenberg front matter, measure and settle the shares, blend, retrain the
tokenizer, re-measure and generate the licensing document, freeze an evaluation prompt set,
triage the minors.

**The shares were settled twice, against two different tokenizers.** `scripts/measure_corpus.py`
is a gate, not a report: it counts what each source can actually supply and exits non-zero
when a slice cannot reach its target share within the upsample cap. The first settle moved
`flavour` from 2.00% to its arithmetic ceiling (0.5% — 2.00% needed 12.8x against a 4x cap,
i.e. it was impossible, not merely tight) and gave the freed 1.5 points to `spine`. Then
Task 5 retrained the tokenizer on the blend, which changed what a "token" *is*: measured
availability fell 6–24% for every domain except tinystories (−0.5%, because the old
vocabulary was tinystories-specialised to begin with). That pushed `procedural` over the 4x
working limit, so Task 6 re-settled — 13% → 12% for `procedural`, the point to `tinystories`,
two `upsample` factors raised. **Lesson: a token budget is denominated in a unit the
tokenizer defines. Retraining the tokenizer silently re-denominates every measurement taken
before it.**

**The circularity is real and has to be cut.** tokenizer → availability → shares → blend →
tokenizer. It does not converge on its own; whichever arrow you cut, the tokenizer ends up
one revision behind the corpus it will be used on. We cut it after Task 6 and wrote the
consequence down (`docs/corpus_blend.md`) rather than chasing it. Chasing it is an infinite
loop that produces a new "one revision behind" statement each time round.

### The two content-loss regex bugs — the reusable lesson

Both were the same mistake, found in consecutive review rounds, in the same pattern
(`_FRONT_MATTER` in `scripts/prepare_corpus.py`, which strips Project Gutenberg packaging
from the head of each document).

1. **Blanket `re.IGNORECASE` makes `[A-Z]` match lowercase.** `produced\s+by\s+[A-Z]` under
   `IGNORECASE` matches `produced by nature`, so any word-wrapped prose line beginning
   "produced by …" was classified as a producer credit and deleted. Not hypothetical: 12
   real prose lines were stripped from `poetry.txt`, and in 12 cases that line WAS the whole
   document, so the document vanished. Fixed with a scoped-flag group, `(?-i:...)`.
2. **The scoped fix was scoped too narrowly.** `(?-i:[Pp]roduced\s+by\s+[A-Z])` turned the
   flag off but kept `[Pp]` matching either case, so a lowercase "produced by" still matched
   whenever the NEXT word was capitalised — which 19th-century prose does constantly
   ("produced by Nature herself", "produced by God's providence"). Same failure mode, now
   gated on the next word's case instead of closed. Fixed by requiring the literal capital:
   `(?-i:Produced\s+by\s+[A-Z])`. A genuine PG credit is always line-initial and capitalised.

Carry these forward:

* **`re.IGNORECASE` is not scoped to the literals you were thinking about.** It applies to
  every character class in the pattern, including the `[A-Z]` you wrote precisely *because*
  you wanted case to matter. If one alternative in a case-insensitive pattern needs case
  sensitivity, use `(?-i:...)` — and note that it scopes the **entire group**, not just the
  class next to it. The round-3 comment claimed it covered "only `[A-Z]`", which is wrong
  about the regex engine, and the wrong comment is what let round 4 exist.
* **A deletion rule needs a measured blast radius, not an argument.** Both bugs were found
  by counting documents before and after, not by reading the pattern. `poetry`'s kept-doc
  count going 3,085,102 → 3,085,114 is what proved bug 1 was real, and every source's count
  being unchanged is what proved bug 2 had not yet reached the shipped corpus.
* **A pattern that eats real prose is far worse than one that leaves packaging behind.**
  Front-matter stripping is asymmetric: leftover "Produced by David Price" costs a few
  tokens; a deleted document is gone and nothing downstream can tell.
* **Regenerate the artifact after fixing a content bug.** Both fixes rebuilt
  `artifacts/corpus/*.txt` from the untouched `artifacts/raw/` sources. The raw copies exist
  for exactly this.

### Pre-merge whole-branch review fix wave

Nine findings; the artifact did not match what the manifest claimed.

* **C2 — the blend was not the blend the manifest described.** `_emit` sized its emission
  with a flat `tokens_per_word=1.3` while `plan_blend` gated on tokenizer-MEASURED
  availability. Real tokens/word runs 1.194 (`tinystories`) to 1.559 (`wikipedia_simple`)
  across the nine — a 30% spread — so the emitter over-emitted for eight of them by exactly
  `real_ratio / 1.3`. `wikipedia_simple` declared `upsample=1` and made 1.058 passes,
  duplicating ~5.8% of Simple Wikipedia undeclared; `procedural` made 4.034 passes against
  the 4x limit Task 6 moved a whole share point to stay under; the blend was 425,024,350 real
  tokens against a 400M budget with shares up to ~3 points off — while the manifest reported
  every `achieved_share` as exactly its target to 15 decimal places.

  Fixed by deriving each source's ratio as `available_tokens / file_word_count`. The
  satisfying part: with the ratio right, repetition collapses to `want / available`, which
  the planner's gate ALREADY holds at or below the declared `upsample` — the emitter
  structurally cannot exceed a source's declared repetition any more. **Lesson: when a gate
  and the thing it gates measure in different units, the gate is decorative.** The 1.3 was
  honest where it came from (`measure_corpus.py`'s no-tokenizer fallback, a deliberate
  slight over-estimate so the gate errs toward reporting more supply); it became a bug when
  it was copied into code that had a real measurement available.

  The manifest now records the tokenizer's own count of exactly the text emitted per source,
  chunked the way `measure_corpus.py` chunks it so the two numbers are comparable. Real
  total: **399,594,747 tokens (−0.101% of budget)**, every slice within 0.065 points.
* **C1 — the legacy path wrote TinyStories into `blend.txt`.** `build_tokenizer.py`
  defaulted `--corpus` to the blend, and when that file was absent fetched TinyStories and
  wrote it *into that path*. On a fresh clone the README sequence therefore produced a 512 MB
  TinyStories file named `blend.txt`, and every later run found it, skipped the fetch, and
  trained on TinyStories forever. **A corpus is just a text file; its name is the only claim
  anyone makes about its contents, so the name has to be defended in code.** The legacy path
  may now never create that name. `train/tokenization.py` still defaulted to
  `corpus.txt`, so the documented quickstart crashed at step 2 — both defaults now agree and
  a test holds them equal.
* **I3 — `:.0%` rendered `flavour`'s 0.5% share as `0%`.** In the GENERATED licensing
  document, whose banner promises it cannot go stale. Generation protects against drift, not
  against a format string. `train.corpus.format_share` keeps fractions.
* **I1/I2, I4** — five rationales still quoted the pre-retrain measurement, `spine` claimed
  an upsample computed at a share it no longer holds and a "largest drop of any slice" that
  belonged to `wikipedia_simple`, and `prepare_corpus.py` justified its email rule with
  "every source here is a pre-1929 public-domain text" — false, since `tinystories` is 2023
  GPT-generated and `wikipedia_simple` is a live encyclopedia. The rule is empirically
  harmless and was left alone; only the reason changed, to the measurement it rests on.
  **A false reason is worse than no reason: it tells the next maintainer the wrong thing is
  safe.**
* **I6 — "FROZEN" prompt set that wasn't.** Rewriting every prompt's `text` to garbage left
  the suite green; only ids, probes and count were pinned. Now digested over sorted
  `(id, text)`. **Pinning the labels of a fixture is not pinning the fixture**, and a prompt
  set whose ids are stable while its text drifts is worse than an unfrozen one, because the
  results still look comparable.
* **I7/I8/I9** — tests for `build_tokenizer.py` (it had none and the highest blast radius on
  the branch), the tokenizer-ordering note above, and this section.

Test suite: **419 passed, 1 skipped**.

## The corpus had no document boundaries at all (2026-08-14)

The prompt: `artifacts/corpus/blend.txt` contains **zero** document separators, while the old
TinyStories-only `corpus.txt` — the corpus the *published* model trained on — contains
**662,878**. Find where document identity is lost between the raw jsonl and the prepared
`.txt`, fix it at the right layer, rebuild every artifact, and set up a 2048-context run.

**Where it was lost: `scripts/prepare_corpus.py` wrote each document as `text + "\n\n"`.** A
document boundary was spelled exactly the way a paragraph break *inside* a document is
spelled, so nothing downstream could distinguish them. `train/tokenization.py` then finished
the job — it encodes the corpus one line at a time and drops the newline, so blank lines
contribute no tokens whatsoever. Zero `</s>` in the corpus, zero id 2 in the token arrays
(still verifiable: `artifacts/tokens/` and `artifacts/tokens-stratified/` are kept, and both
have zero in their first 20M tokens).

That is not a tidiness bug. A position-wise loss probe showed per-token loss flat from
position ~64 to 511, on books as much as on short items — with boundaries unmarked, distant
context genuinely *is* unpredictable, so the model was right to ignore it. The mid-generation
topic collapse in the samples is the same fact from the other side. **Lesson: a delimiter that
is indistinguishable from ordinary formatting is not a delimiter.** The old pipeline got this
right by accident, because TinyStories shipped an explicit `<|endoftext|>` line; the
nine-source rewrite dropped the idea along with the format.

* **The separator belongs in `prepare_corpus.py`, and nowhere else.** It is the only stage
  that can see a document at all: `fetch_corpus.py` writes one JSON object per document,
  `prepare_corpus.py` consumes them one at a time, and `blend_corpus.py` sees only
  concatenated text which it repeats and truncates. Putting the boundary anywhere later would
  have meant guessing at it.
* **`</s>` was already the right token, and this was checked rather than assumed.** Id 2, an
  *added* token (so byte-level BPE can neither split it nor absorb a neighbour),
  `special_tokens_map.json`'s `eos_token`, and already written as `eos_token_id` into
  `config.json` *and* `generation_config.json` by `convert/to_hf.py`. The serving path was
  waiting for a token the training data never contained.
* **The fix nearly introduced a worse bug.** `biglam/gutenberg-poetry-corpus` has one row per
  **line** of verse — 3,085,117 rows of ~7 words. A per-row separator would have fired an
  end-of-document token every seven words and, at poetry's 1% share, put roughly a *third* of
  every `</s>` in the blend inside that slice: a seven-word prior for "stop". Caught by asking
  what a "document" is for each source before writing any. `CorpusSource.rows_per_document`
  (64 for poetry, 1 elsewhere) makes it 48,205 documents and 6,002 separators instead.
  **Lesson: "one row = one document" is an assumption about the upstream dataset, not a
  property of jsonl.**
* **The truncated tail is closed deliberately.** `_emit` truncates each source's final pass at
  word level, mid-document. Left open, source A's half-sentence would run into source B's
  first document at each of the nine seams — the same defect, just rarer. The closing
  separator is counted in `emitted_words`, because that number is what the stratified split
  uses to locate each source's boundary in the finished corpus.
* **`train/tokenization.py`'s `add_special_tokens=False` comment was wrong in both halves.**
  It claimed the corpus already carried separators (true of the legacy path, false of this
  one) and that `True` would double them (false outright — this tokenizer's post-processor is
  a plain ByteLevel, so `True` injects nothing; measured). The flag stays `False` for the real
  reason: that function is called once per **line**, so a tokenizer that ever gained a
  template post-processor would wrap every line rather than every document. **A comment that
  survives the code it describes becomes a trap.**

**Verified empirically, not declared.** `blend.txt` holds **798,771** `</s>` lines against
zero before. `artifacts/tokens-v3/` holds 734,978 + 63,793 = **798,771** occurrences of id 2 —
equal to the token, so none were added, lost, or split. Decoded windows put the separator
exactly where documents end: a TinyStories story closing before "Once upon a time"; the last
line of an Oz book before the *next* book's title page; the Vatican City article's category
tail before "Velocity is a measure of how fast something moves".

**Shares did not need re-settling.** Availability rose by exactly **2 tokens per document**
for every source (the separator plus its newline), which can only loosen the scarcity gate.
The blend totals 399,508,203 tokens against the 400M budget (−0.123%), every slice within
0.083 points of target. The expected "+1.4% of tokens" did not happen either: the budget is
fixed, so the separators *displace* text rather than adding to the total. And ~800k, not
~5.5M — most of the document-heavy sources are used fractionally (`tinystories` 0.28x,
`wikipedia_simple` 0.88x), so most of their documents never enter the blend.

**2048 context.** Raised in `train/configs/model/tt-tnt-384.yaml` and `train/sizes.py`
together (the anti-drift test holds them equal). `tt-tnt-1024` deliberately stays at 512 — the
evidence for 2048 is a measurement on the 384 shape, and that size has never been trained.
That divergence is what forced `--seq-len` to default to the selected size's own
`max_sequence_length` instead of a fixed 512: `build_yaml_config` enforces
`seq_len == max_sequence_length`, so a fixed default had become a guaranteed error for the
default size. **Lesson: a constant shared by two things that are allowed to differ is a bug
waiting for the day they do.** No training was started.

Test suite: **511 passed, 1 skipped** for this change (500 before it). A concurrent,
still-untracked `tests/test_probe_context_use.py` — the position-wise loss probe whose
measurement motivated all of the above — adds 36 more, for **547 passed, 1 skipped** when the
whole `tests/` directory is collected.

**A registry rationale is an artifact too.** Re-measuring availability broke
`test_a_rationale_that_cites_availability_cites_the_CURRENT_availability`, exactly as designed:
seven rationales still quoted the pre-separator numbers. That gate has now caught stale prose
on three separate occasions, which is the argument for having it.

## The weight cache that lied (2026-08-14)

**Prompt:** "Design and implement a fix for a stale-cache bug, carried in this model's own
adapter." Hardware paused by the owner — source reading, unit tests, CPU-only work; any
device verification deferred with an exact command.

The bug, from `.superpowers/serve-tt-tnt-v3.md` F7: `tt_transformers` caches converted
weights at `model_cache/<repo_id>/<device>/tensor_cache_bfp8/` and decides to reuse them
with a bare existence check. tt-tnt was retrained, republished to the **same** repo id, and
served — the server came up clean, reported the correct `max_model_len: 2048`, and ran the
**previous** model's weights, logged as an ordinary warm start. Because that model could not
emit EOS by construction, the headline number would have been a confident "0% termination"
and would have read as a real regression in the very fix being tested. Caught only because
someone noticed a directory mtime predated the publish.

**Where it actually lives** — three facts, all now pinned by a canary test:

| what | where |
|---|---|
| cache key computed | `model_config.py:577` — `os.path.join("model_cache", HF_MODEL, self.device_name)` |
| the single funnel every weight path flows through | `model_config.py:3017` — `weight_cache_path(self, dtype)` |
| reuse-vs-reconvert decided | `ttnn/ttnn/operations/core.py:719` — `if not cache_path.exists() or not cache_path.is_file()` |

Only a *deserialisation* failure (`core.py:725`) ever triggers a re-conversion. Nothing
compares the cache against the weights it came from.

**The fix: content-address the cache key**, by appending one component to
`weight_cache_path` —
`…/tensor_cache_bfp8/src-rev-a3c85ec799fe/`. The fingerprint is the HF commit sha
(`hf_config._commit_hash`, which `transformers` stamps on the config at
`configuration_utils.py:812` and `ModelArgs.__init__` has already loaded by
`model_config.py:616`), falling back for local checkpoint directories to a sha256 over
`(name, size, mtime_ns)` of `config.json` and the weight files.

**Why not the alternatives**, all of which were on the table:
* *Validate before use.* To know a cached tensor is wrong you must produce the right one —
  i.e. pay the conversion the cache exists to avoid — unless you keep a side manifest, which
  is this fix with more moving parts. It also overwrites, so flipping back to the previous
  revision pays again. Fingerprinting keeps every revision warm.
* *Refuse a cache older than the weights (mtime).* There is no local file whose mtime means
  "publish time" — the source is a Hub repo id and HF blob mtimes are *download* times, whose
  order against the conversion is arbitrary within one session. Reading a timestamp is what
  the human had to do; it is not a thing to automate.
* *Disable the cache.* Correct and slow, so it gets switched back off the first busy
  afternoon, restoring the bug. **A fix that gets disabled is not a fix.**

**The failure mode was silence, so the fix must not introduce a different silence.** Every
state in which the guarantee does not hold is audible: a new fingerprint beside an existing
one WARNs and names the superseded revision (the log line whose absence made this invisible);
leftover un-fingerprinted `.tensorbin` files WARN that they are now dead (and are *not*
deleted — that is a human's call); an unavailable fingerprint or `TT_TNT_CACHE_FINGERPRINT=0`
WARNs that stale weights are possible again; and if `weight_cache_path` has moved or its
first two parameters are no longer `(self, dtype)`, the patch **declines to install** and
says so rather than crashing the serve or silently no-opping. Each fingerprinted directory
also gets a `tt_tnt_cache_source.json` stamp, so `ls` answers the question that cost an
afternoon.

**Local checkpoints need `mtime_ns`, not just size.** A retrain of the same architecture
writes byte-identical file *sizes* — a size-only digest would reproduce this exact bug. The
cost is a false miss (one redundant conversion, logged) after a re-download. False misses
cost minutes; false hits cost a published measurement. `test_local_checkpoint_retrain_…`
asserts the premise (`len(v1) == len(v2)`) before asserting the behaviour, so the test cannot
pass for the wrong reason.

**Tests run without tt-metal and without hardware** by installing a fake
`models.tt_transformers` into `sys.modules` before loading the adapter — so 18 of the 19 gates
always execute rather than skipping into vacuity. The 19th, the upstream-drift canary, reads
tt-metal's *source text* (never imports it, so no ttnn, no device) and asserts all three
anchors above still exist; it skips with a reason when the tree is absent. **Mutation-checked
three ways**: reverting the patch fails 13, keying on a constant fails 9, dropping `mtime_ns`
fails the retrain test specifically.

**F8 is deliberately not in the adapter.** `~/.cache/tt-kernel/bundles/` is consumed by
`tt-kernel serve` *before* the vLLM process exists: the stale bundle's `vllm_metadata.json`
supplies the launch command (`--max_model_len 512`) that starts the process that would import
this adapter. The adapter cannot reach backwards past its own `argv`. It belongs in tt-kernel
(`cli.py:1276 _serve_vllm` → `_ensure_vllm_pulled`), as a `--refresh` flag plus a revision
comparison against the Hub on serve.

Suite: **600 passed, 1 skipped** (581 + 19 before this change, baseline held).
