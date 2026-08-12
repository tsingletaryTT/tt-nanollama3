# tt-nanollama3 — project notes for Claude

## What this project is

The reference example of a **Tenstorrent-first model**: NanoLlama3, trained from random
init on Blackhole with `ttml` (tt-train), packaged as a **tt-kernel v4 bundle**, and served
through the **Tenstorrent vLLM plugin**. Small model, complete story — the point is to show
end to end what a model built for TT from line one looks like across train → package →
publish → serve.

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
