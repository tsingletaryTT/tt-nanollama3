<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# tt-nanollama3

A small Llama-3-style language model built **Tenstorrent-first** — trained from random
initialization on Blackhole hardware with `ttml` (tt-train), packaged with
[tt-kernel](https://github.com/tenstorrent/tt-kernel-package-manager), and served through the
Tenstorrent vLLM plugin.

The model is deliberately small. The point is not capability — it is to show, end to end and
without gaps, what a model designed for Tenstorrent from the first line looks like when it is
trained, packaged, published, and served entirely on Tenstorrent tooling.

## Status

**Working today:** corpus preparation, a 32,000-token BPE tokenizer, and a training entrypoint
that runs on hardware with a real validation loop.

**Not done yet:** there is **no trained checkpoint in this repository.** Training has been
proven over short runs (20 steps and 5 steps), not run to completion. Checkpointing lands in
the next plan, followed by a real training run, then Hugging Face conversion and packaging.
See [`docs/superpowers/specs/`](docs/superpowers/specs/) for the full arc.

Do not expect usable weights from this repo yet.

## The model

| Property | Value |
|---|---|
| Architecture | Llama-3 style — RoPE (θ=500000), RMSNorm, SwiGLU, grouped-query attention |
| Embedding dim | 384 |
| Blocks | 6 |
| Heads / KV groups | 6 / 3 |
| Sequence length | 256 |
| Vocabulary | 32,000 (byte-level BPE, trained here) |
| Corpus | TinyStories — 127,635,889 tokens (114.9M train / 12.8M validation) |
| Hardware | Tenstorrent Blackhole (verified on 4× p300c) |

Architecture parameters come from tt-train's `nanollama3.yaml` and are not redefined here.

Measured on a 20-step smoke run: first loss **10.6875** against a theoretical `ln(32000) ≈
10.37` for a freshly initialized model, falling to **7.4688**, with a real validation loss of
**7.0281**. Steady-state throughput was ~0.12–0.14 s/step after an 18.7 s first step that
times the kernel compiler rather than the model.

## History

This model comes out of the **"Build an LLM from Scratch"** lesson arc in
[tt-vscode-toolkit](https://github.com/tenstorrent/tt-vscode-toolkit), which builds a
Llama-3-style model TT-native from the first line of code rather than porting one afterwards.
The lessons are readable without installing anything:

- [Pick Your Altitude](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/lfs-00-intro/) — the arc's introduction
- [Tokenizer & Data](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/lfs-01-tokenizer/)
- [Embeddings](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/lfs-02-embeddings/)
- [Attention](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/lfs-03-attention/)
- [The Transformer Block & the Model](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/lfs-04-block-and-model/)
- [Train It & Run for Real](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/lfs-05-train-and-run/)

Browse the whole microsite at
[docs.tenstorrent.com/tt-vscode-toolkit](https://docs.tenstorrent.com/tt-vscode-toolkit/).

This repository takes that arc past where the lessons stop: it owns its training entrypoint,
its corpus and tokenizer pipeline, and the packaging work that turns a checkpoint into
something servable.

**Why it owns its own training entrypoint.** tt-train's stock Python trainer does not work
against current tt-metal. `examples/python/transformers/training.py` imports a `trainer`
module that is not on its path, calls `train()` with a `val_ids` argument the signature does
not accept, and depends on a `TrainingConfig` that never defines the `seq_len` `train()`
requires. Its data loader also hardcodes `shakespeare.txt`, ignoring any configured path. We
reuse everything ttml genuinely provides — `TransformerModelFactory`, `create_optimizer`,
`train()`, `checkpointing` — and replace only what is broken or hardcoded.

## Provenance and licensing

**This project's source code is Apache-2.0**, matching tt-metal and tt-vscode-toolkit. Every
source file carries an SPDX header. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Apache-2.0 covers *our code*. It does not override the terms of what this project consumes,
and two of those inputs deserve stating plainly rather than being folded into a blanket claim:

**Training corpus — TinyStories, CDLA-Sharing-1.0.** The corpus
([`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories)) is
licensed under the Community Data License Agreement – Sharing, v1.0, which is a share-alike
data license. This repository **does not redistribute the corpus**; `train/data.py` downloads
it from the Hugging Face Hub at a pinned revision. Whether model weights trained on
CDLA-Sharing data constitute a "Data Derivative" under that license is not settled, and we do
not assert that they don't. Anyone publishing weights trained with this code should reach
their own conclusion rather than inheriting ours.

**Architectural inspiration — Mini-LLM.** The lesson arc credits
[Mini-LLM by Ashx098](https://github.com/Ashx098/Mini-LLM) for its component choices — RoPE,
RMSNorm, SwiGLU, GQA, subword BPE. That repository **declares no license**, so no rights are
granted by it. This is a credit, not a license inheritance: the components themselves come
from published papers, and this implementation derives from tt-train's `nanollama3` model
config and the `ttml` library, not from Mini-LLM's source. No code was copied from it.

**Model weights.** None are published yet. When they are, the model card will state the corpus
and its license explicitly, and will describe the model honestly as a demonstration rather
than a capable general model.

**Runtime dependencies** — tt-metal / `ttml` / `ttnn` (Apache-2.0), `transformers` and
`tokenizers` (Apache-2.0), numpy (BSD-3-Clause).

## Getting started

Requires a tt-metal source checkout with `ttml` built, and `TT_METAL_HOME` set. See
[Build TT-Metalium from Source](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/build-tt-metal/).

```bash
pip install -e .

# Fetch the corpus and train the tokenizer (downloads ~2.2 GB, CPU only)
python scripts/build_tokenizer.py --corpus-mb 512

# Tokenize the corpus into training arrays (CPU only)
python train/tokenization.py

# Confirm the training config resolves without touching hardware
python train/run.py --dry-run --steps 20

# Train on Blackhole
python train/run.py --steps 20 --batch-size 64
```

Run the tests with `python -m pytest`. They need no hardware.

## Contributing

This repository follows the plan-then-execute workflow in
[`docs/superpowers/`](docs/superpowers/): a spec, then implementation plans whose steps are
executed and reviewed task by task. The plans record what was verified against source rather
than assumed — including several defects found in the plans themselves.
