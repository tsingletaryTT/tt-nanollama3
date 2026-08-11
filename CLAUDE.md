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
