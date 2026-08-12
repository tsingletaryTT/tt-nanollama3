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

**Working today:** corpus preparation, a 32,000-token BPE tokenizer, a training entrypoint that
runs on hardware with a real validation loop, checkpointing with resume, and conversion to a
Hugging Face model directory. **The model has been trained** — 3000 steps, in 6 min 47 s on a
single Blackhole p300c — **and the Hugging Face conversion is numerically verified**: HF-side
validation loss on the converted model is **1.9271**, against **1.8781** from the training
run's own held-out evaluation, computed the same way (10 batches of 32 randomly-sampled
256-token windows) so the two numbers are comparable.

**Not done yet:** the trained weights are **not published** — checkpoints are gitignored and
live only on the machine that produced them, so cloning this repo gets you the pipeline, not a
model. tt-kernel packaging is the remaining stage. See
[`docs/superpowers/specs/`](docs/superpowers/specs/) for the full arc.

**Calibrate your expectations.** This is a ~22M-parameter model that has seen 49,152,000
tokens — about **0.43 of one epoch** over its training split — of TinyStories, a synthetic
corpus of simple children's stories with a small vocabulary and deliberately regular grammar.
It demonstrates that the pipeline works end to end. It is not a capable general model. One
generated sample, reported as a single data point rather than proof of general capability:

> Once upon a time, there was a little dog named Max. Max loved to play with his ball. One day,
> Max saw a big ball in the park. Max wanted to play with the ball, but he was very dirty. Max
> had an idea. He would push the ball with his paws to clean it.

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
| Hardware | Tenstorrent Blackhole — trained on **one** p300c (`mesh_shape [1, 1]`, no DDP/TP) |

Architecture parameters come from tt-train's `nanollama3.yaml` and are not redefined here.

Measured on the 3000-step run: first train loss **10.6875** — consistent with a near-uniform
initial distribution, where `ln(32000) = 10.37` — falling to **1.9219**, with a real held-out
validation loss of **1.8781** from this repo's own `evaluate()` over 10 sampled batches.
Steady state ~0.134 s/step (7.44–7.50 it/s); 6 min 47 s wall clock; six checkpoints at steps
500–3000.

**This runs on one chip.** `train/config.py` sets `device_config` to `mesh_shape: [1, 1]` with
`enable_ddp` and `enable_tp` both false, so training opens a single device. The host this was
developed on is a **TT-QuietBox 2** — four Blackhole chips on two dual-chip p300 cards, wired
as a `P300_X2` 2×2 ring mesh, not four independent boards — and three of those four chips sit
idle during a run. During the v2 run the working chip drew **82 W at 73 °C** against
**61–73 W / 63–68 °C** on the idle three; idle Blackhole holds its clock at 1350 MHz, so the
power gap is a clearer signal than temperature.

Multi-chip data parallelism is supported by tt-train (`enable_ddp: true` with a `[1, 4]` mesh)
and is **future work here, not something this repo has demonstrated**. It is not a one-line
config edit — see [`docs/multi-chip-notes.md`](docs/multi-chip-notes.md) for the three known
catches and why the step budget and learning rate move with it.

Validation coming in below training loss is expected here rather than anomalous: at 0.43 of an
epoch there is no overfitting, dropout is 0.0, and the final train figure is a single batch
against a ten-batch validation average. That the two differ at all is what confirms the number
came from our own evaluation rather than tt-train's `val_losses`, which is a documented
placeholder that copies the training loss.

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
