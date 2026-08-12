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

- First train loss `10.6875` (≈`ln(32000)`, correct init), last train loss `1.9219`, real
  validation loss (our own `evaluate()`, 10 sampled batches, not `train()`'s placeholder)
  `1.8781`. The curve is a clean log shape — steep for the first ~300 steps, then a steadily
  decelerating decline into a noisy 1.9–2.1 band for the back half — not a plateau, not
  divergence, and (as expected for a raw per-batch metric) not step-to-step monotonic: roughly
  half of all individual steps ticked up rather than down, while every windowed average kept
  falling.
- Steady state `~0.134 s/step` (7.44–7.50 it/s), matching Plan 2's 0.12–0.14 s/step. Unlike
  Plan 2, this run's first step showed no visible compiler-warmup stall — most likely because
  Task 2's same-shapes hardware runs earlier today had already warmed the on-disk kernel
  cache; not confirmed by inspecting the cache directly, so treat as a plausible explanation,
  not a verified one.
- Total wall clock `~6 min 47 s` (process start to final printed loss), including all six
  checkpoint writes — the writes did not add a visible cost on top of compute; every 500-step
  chunk took the same ~67.1 s regardless.
- Six checkpoints, `nanollama3_step00000500.pkl` through `nanollama3_step00003000.pkl`
  (`artifacts/checkpoints/`, gitignored), 132,185,963 bytes each, final one at step 3000 as
  requested.

**This is a demonstration, not a capable model.** `3000 × 64 × 256 ≈ 49.2M tokens` is about
**0.43 of one epoch** over the 114.9M-token training split — this run never saw even half the
corpus once. TinyStories is also a synthetic, deliberately simple corpus (short children's
stories, small effective vocabulary, regular grammar, built so small models can fit it), so a
low loss here is expected and does not indicate general language competence — no text was
decoded from this checkpoint to check. Full numbers, the reconstructed per-step curve, and
self-review are in
[`task-3-report.md`](.superpowers/sdd/2026-08-11-checkpointing/task-3-report.md).
