<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# tt-nanollama3 — design

**Status:** approved design, not yet implemented
**Date:** 2026-08-11
**Repo:** `~/code/tt-nanollama3` (new; created empty for this spec)

## Summary

Build **NanoLlama3** into the reference example of a Tenstorrent-first model: trained from
random initialization on Blackhole with `ttml` (tt-train), packaged as a **tt-kernel v4
bundle**, and served through the **Tenstorrent vLLM plugin**. The model is small on purpose.
The point is not the model's capability — it is to show, end to end and without gaps, what a
model designed for TT from line one looks like when it is trained, packaged, published, and
served entirely on Tenstorrent tooling.

This replaces an earlier plan to package `tt-animatediff` the same way. That plan was
abandoned for a good reason, recorded in Appendix A: tt-kernel has no forward-looking home
for a non-vLLM model.

## Goals

1. A model we own outright — weights trained from scratch on our own hardware, with no
   third-party checkpoint and therefore no redistribution question.
2. A **v4 tt-kernel bundle** that exercises the current packaging path (`push --manifest`,
   `pull`, `serve`, `publish`, `instances`, `search --target`), not the legacy v3 one.
3. A model package that demonstrates the **"pick your altitude" ladder as a real property**:
   a portable ttnn path and a hand-authored TT-Lang path, bound together by parity tests.
4. Reuse of existing Tenstorrent components wherever they exist. New code only where nothing
   suitable exists.

## Non-goals

- **Competitive model quality.** At ~22M parameters this is a demonstration model. The model
  card will state that plainly rather than implying general capability.
- **Shipping a precompiled kernel cache.** v4 bundles are kernels-less by design; the vLLM
  plugin JITs at first-run warmup. We do not use the v3 kernel-cache path.
- **Serving anything but vLLM.** No custom server, no new tt-kernel backend.
- **Matching Mini-LLM's published run.** See "Training scale" for why, and what we do instead.

## Background — verified inventory

Everything below was confirmed on this machine while writing this spec.

### The existing model

`~/tt-metal/tt-train/checkpoints/` holds `nanollama3_char_3k.pkl_final.pkl` (39 MB) plus step
checkpoints at 500–3000, dated 2026-07-09. Architecture per
`~/tt-metal/tt-train/configs/model_configs/nanollama3_char.yaml`:

```yaml
transformer_config:
  model_type: "llama"
  num_heads: 6
  num_groups: 3          # grouped-query attention
  embedding_dim: 384
  num_blocks: 6
  max_sequence_length: 256
  theta: 500000.0        # RoPE
```

That is **9.81M parameters**, char-tokenized. The recorded run reached loss 3.2344 in 20 steps
at ~65 ms/step, 16.5 TFLOPS, ~11% MFU, on a p300c.

`nanollama3.yaml` is the same architecture with `vocab_size: 32000` — the subword variant we
will use. Switching to it takes the model to roughly **~22M parameters**, almost entirely from
the 32000×384 embedding table.

### Reusable assets

From `~/code/tt-vscode-toolkit/content/templates/llm-from-scratch/` (the lesson arc's source
of truth, `lfs-00`…`lfs-05`):

| Asset | Use |
|---|---|
| `kernels/{rope,rmsnorm,attention,matmul,eltwise_add}.py` | TT-Lang kernels, sim-validated — the hand-authored path |
| `kernels/_ttlang_sim.py` | Simulator harness for kernel tests without hardware |
| `mesh-descriptors/qb2_1x4_ring.textproto` | Maps onto v4 `Mesh(devices, topology, fabric)` if we demo multi-device |
| `reference_gpt.py` | Pure-PyTorch reference with full RoPE — the parity oracle |
| `tokenizer_bpe.py` | Teaching BPE implementation (see "Tokenizer" for why we don't ship it) |
| `train_nano_from_scratch.py` | Thin launcher over tt-train's `train_nanogpt.py` |

From tt-metal: `ttml.ops.rope`, `ttml.ops.rmsnorm`, and ttnn's matmul/softmax. `ttml` is
already built at `~/tt-metal/build_Release/`.

**Reuse policy for this repo: never reimplement what ttnn or ttml already provide.** Where the
lesson has a hand-authored TT-Lang kernel, ship both paths and prove they agree.

### Environment facts that constrain us

- `~/tt-metal` is on `rollback-pre-qwen36-1576-g620793d898`. The lesson arc was verified
  against **v0.73**. v4's `platform.ttnn` range gate resolves a git-sha checkout to
  "assume OK", so this will not block — but we must not declare a range we have not tested.
- Upstream tt-metal runs **no CI for training ops on Blackhole**; `tt-train` `GTEST_SKIP`s
  softmax, cross-entropy, rmsnorm, and SDPA training tests on p100/p150. The lesson arc's
  successful run is our own verification, not an upstream guarantee.
- `tokenizers` 0.21.4 and `sentencepiece` are both installed.
- Training data present is **only** `~/tt-metal/tt-train/data/shakespeare.txt` (1.1 MB).
- The vLLM fork is at `~/dispatch/vllm` on `dev` @ `61bbb1ea` (2026-07-08).
- tt-kernel is at `00dba42` with schema v4; its suite passes (191 tests).

## Architecture

```
tt-nanollama3/
├── nanollama3/
│   ├── model.py            # TT-native forward pass (ttnn path, the portable default)
│   ├── kernels/            # TT-Lang path, adapted from the lesson templates
│   ├── config.py           # architecture constants, single source of truth
│   └── vllm_adapter.py     # TTNanoLlama3 — what Entrypoint.cls points at
├── convert/
│   ├── ckpt_to_hf.py       # ttml .pkl -> config.json + safetensors
│   └── tokenizer.py        # train + export an HF-format 32K BPE tokenizer
├── train/
│   ├── data.py             # corpus fetch + prepare
│   └── run.py              # launcher over tt-train's train_nanogpt.py
├── bundle/
│   └── manifest.json       # the authored v4 manifest
├── tests/
│   ├── test_parity.py      # PyTorch <-> ttnn <-> TT-Lang
│   ├── test_convert.py     # checkpoint conversion round-trip
│   └── test_e2e.py         # generation through the served endpoint
└── scripts/
    └── roundtrip.sh        # push -> pull -> serve -> query
```

Each unit has one job and a testable boundary: `convert/` never imports ttnn; `nanollama3/`
never touches the Hub; `bundle/` is data, not code.

## Stage 1 — Tokenizer and data

**Tokenizer.** Train a 32K-vocab BPE with the `tokenizers` library and export it in HF format
(`tokenizer.json`, `tokenizer_config.json`) so vLLM can load it unmodified. We do **not** ship
`tokenizer_bpe.py` from the template — it is a pure-Python teaching implementation whose
purpose is comprehension, not throughput. It stays referenced in docs as the explanation of
what the real tokenizer does.

**Corpus.** Shakespeare at 1.1 MB is far too small for a 32K vocabulary or ~22M parameters —
it would overfit and prove nothing. Use a modest real corpus (TinyStories is the leading
candidate: small, clean, and known to produce coherent output at this parameter count).
`train/data.py` fetches, deduplicates, and shards it; the exact corpus and its licence are
recorded in the model card.

## Stage 2 — Training scale

**Target: the existing `nanollama3.yaml` unchanged (~22M params, 32K vocab), on a small real
corpus. Hours, not days.**

This was a deliberate calibration. Mini-LLM's published result — ~80M parameters, 361M
tokens, ~5 hours on an A100, final loss ~3.25 — would require authoring a new model config
*and* a 361M-token data pipeline, and would block all packaging work behind a multi-day
training job. The ~22M configuration already exists, is known to run, and at TinyStories scale
produces output that is genuinely coherent rather than gibberish. That is a defensible
exemplar.

Training runs through `train/run.py`, a thin launcher over tt-train's `train_nanogpt.py`,
following the pattern in `train_nano_from_scratch.py`. Two operational details carried over
from the lesson, both learned the hard way:

- **Let `ttml` close the device.** Bypassing the `finally` block triggers a teardown abort in
  `MetalContext::destroy_all_instances`.
- **If the board times out on device open, `tt-smi -r` first.** Common on p300c/QB2; it is
  usually not a real fault.

## Stage 3 — Conversion to HF format

`convert/ckpt_to_hf.py` turns a ttml `.pkl` checkpoint into `config.json` plus safetensors.
The architecture is fully known, so this is mechanical rather than exploratory — but it is the
step most likely to fail silently, so it is guarded by a round-trip test that reloads the
converted weights and compares logits against the PyTorch reference.

`convert/` deliberately does not import ttnn. It must be runnable on any machine, including
one with no Tenstorrent hardware.

## Stage 4 — The model package and the altitude ladder

`nanollama3/model.py` implements the forward pass against `ttnn.Tensor`, using `ttml.ops.rope`
and `ttml.ops.rmsnorm` where they exist. `nanollama3/kernels/` carries the TT-Lang kernels
adapted from the lesson templates.

**Both paths ship.** A runtime switch selects between them; the ttnn path is the default
because it is portable, and the TT-Lang path is the tuned one. `tests/test_parity.py` binds
them: for each component with two implementations (RoPE, RMSNorm, attention, matmul, eltwise
add), assert the PyTorch reference, the ttnn path, and the TT-Lang kernel agree within
tolerance. The simulator harness `_ttlang_sim.py` lets those tests run without hardware.

This is the part that makes the repo an exemplar rather than a packaging demo. A model that
ships its own kernels *and proves they match a reference* is a different artifact from a model
that merely runs.

## Stage 5 — The vLLM adapter

Two steps, deliberately ordered to de-risk.

**Step 1, validation spike (throwaway).** Emit a `config.json` claiming `LlamaForCausalLM` and
see whether the existing TT Llama adapter loads the converted weights and generates. This
proves the conversion is correct end to end without writing any adapter code. It is a test
fixture, not a deliverable.

**Step 2, the real adapter.** `nanollama3/vllm_adapter.py` provides `TTNanoLlama3`, referenced
by `Entrypoint.cls`, registering under `arch_name: NanoLlama3ForCausalLM`. Shipping the model
disguised as Llama would undercut the entire point — the exemplar must show what registering
*your own* architecture with the plugin actually takes.

## Stage 6 — The tt-kernel v4 bundle

`bundle/manifest.json`, authored by hand and passed to `tt-kernel push --manifest`:

```json
{
  "schema_version": "4",
  "name": "tt-nanollama3",
  "arch": "blackhole",
  "device_count": 1,
  "platform":   { "ttnn": ">=0.73" },
  "runtime":    { "kind": "vllm", "version": ">=0.24" },
  "mesh":       { "devices": 1, "topology": "1x1" },
  "entrypoint": { "class": "nanollama3.vllm_adapter:TTNanoLlama3",
                  "arch_name": "NanoLlama3ForCausalLM" },
  "resources":  { "max_model_len": 256, "max_num_seqs": 8 }
}
```

`max_model_len` is 256 because `max_sequence_length` is 256 — the manifest must not promise
more than the model was trained for. `tt_metal_version` is filled by push from the local
environment. `build_key` stays absent: this is a kernels-less bundle and the plugin JITs at
warmup.

Ranges are declared **only as far as we have tested**. `platform.ttnn` gets a floor, not a
speculative ceiling. `runtime.plugin_version` is omitted, which keeps the legacy
presence-only plugin check — correct, since the fork tracks `dev` with no version floor.

Unlike every other bundle on the Hub today, this one **ships its own weights**. `hf_weights`
is not used; there is no upstream repo to point at.

The `qb2_1x4_ring.textproto` descriptor is available if we later want to demonstrate a mesh
topology, but a ~22M model on one device is the honest default.

## Stage 7 — Publish

`tt-kernel push --manifest ... --public --publish`, then verify the catalog listing. Two things
to expect: `filter=tt-kernel-catalog` currently returns zero repos, so this would be the first
listing; and tt-kernel has no `.github` directory, so `web/` is not deployed anywhere. Getting
the catalog hosted is out of scope here but should be raised.

Capability tags are added via `--capability`. Note `web/config.js` only maps `moe` and
`sliding-window-attention` to display labels, so tags like `gqa` or `rope` render no badge
until rows are added there.

## Testing

| Level | What it proves |
|---|---|
| `test_parity.py` | PyTorch ↔ ttnn ↔ TT-Lang agree per component. Runs in the simulator; no hardware needed. |
| `test_convert.py` | Converted weights reload and reproduce reference logits. Catches silent conversion corruption. |
| `test_e2e.py` | A prompt through the served endpoint returns coherent text. |
| `scripts/roundtrip.sh` | `push → pull → serve → query` on a clean bundles dir. The packaging contract. |

The end-to-end success criterion: **on a machine that has never seen this model, `tt-kernel
serve <id>` brings up an endpoint that answers a prompt** — with no manual steps not listed in
the README.

## Risks

**Adapter registration is the critical path.** If registering a new `arch_name` with the TT
plugin turns out to need changes inside the fork rather than in our bundle, Stage 5 Step 2
stalls. Mitigation: the Step 1 spike proves the weights independently, so a stall there is
contained and we can ship the Llama-config route as a documented fallback while we resolve it.

**Training on Blackhole is unwarranted upstream.** No CI covers these ops on this
architecture. Mitigation: the lesson arc's run is prior evidence it works; pin the tt-metal
version we verify against and record it.

**Version drift.** `~/tt-metal` is not on v0.73. Mitigation: verify against whatever we
actually build on, and declare only that.

**Corpus licensing.** Whatever we train on becomes a property of published weights. Mitigation:
record corpus and licence in the model card before the first public push.

## Open questions

1. **HF namespace.** Personal versus `tenstorrent/`. As the first public catalog listing this
   is a governance question, not a technical one.
2. **Corpus choice.** TinyStories is the leading candidate; not yet confirmed.
3. **Catalog hosting.** `web/` is undeployed. Out of scope, but blocks the listing being
   visible to anyone.

## Appendix A — why not tt-animatediff

The original plan was to package `~/code/tt-animatediff` as a tt-kernel bundle. Analysis
killed it, and the findings are worth reporting to the tt-kernel maintainers independently of
this repo:

1. **Both tt-kernel serving backends terminate in `/v1/chat/completions`.** The vLLM backend
   hands to the plugin; the legacy dispatch backend is served by `tt_kernel.legacy_serve`.
   A text-to-video model has no chat-shaped output.
2. **v4 is explicitly "vLLM only, kernels-less"; v3 is "legacy, read-only supported"**
   (`manifest.py:24-27`). A model needing a precompiled kernel cache has only the deprecating
   path, and a non-vLLM model has no path at all. `Runtime.kind` defaults to `"vllm"` and
   nothing consumes another value.
3. **No bundle can ship kernels and arbitrary artifacts together.** `--backend dispatch` gives
   kernels plus wheels; `--backend vllm` gives a folder with no kernels. Converted weight
   tensors have nowhere to live. A `--artifact DIR` option indexed into `Manifest.files` under
   `artifacts/` — mirroring how `python/` already works — would close this.
4. **`cli.py:242` still writes `schema_version="3"`** for dispatch bundles while `manifest.py`
   describes v3 as read-only supported. Those two statements should be reconciled.
5. **Instance selection is wired to the vLLM path only.** `Instance.activation_env()` derives
   `PYTHONPATH` from `tt_metal_home`, which is exactly what makes tt-metal's `models/` tree
   importable — but `legacy_serve` never receives it.

Separately, and unrelated to tt-kernel: `tt-animatediff`'s
`generation_helpers.py:42` calls `preprocess_model_parameters` without `model_name`, which
disables the converted-weights disk cache entirely. The adjacent comment claiming results are
"cached after" the first run is true only within a single process. Worth fixing there on its
own merits.
