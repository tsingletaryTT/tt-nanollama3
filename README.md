<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# tt-tnt

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

**This repository owns its architectures.** They live in
[`train/configs/model/`](train/configs/model/), one YAML per size, described and validated by
the registry in [`train/sizes.py`](train/sizes.py). `train/run.py --size <name>` selects one.

The 384 config is a verbatim copy of tt-train's own `nanollama3.yaml`, vendored rather than
read out of `$TT_METAL_HOME` so the architecture cannot change under a tt-metal upgrade
without a signal — `tests/test_sizes.py` compares the two whenever tt-metal is present, and
holds the registry and the YAML to describing the same model.

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

## Lineage: from nanollama3 to tt-tnt

This project was originally named **tt-nanollama3**, and this section says plainly what
changed and what didn't, rather than quietly dropping the earlier name.

**What it started as.** The model began as a hand-rolled nanollama3-like model: a Llama-3
architecture, trained from random initialization with tt-train's `ttml` trainer, on a single
downloaded corpus (TinyStories). That is still true today in the parts that matter most for
correctness — the 384 config vendored at
[`train/configs/model/nanollama3-384.yaml`](train/configs/model/nanollama3-384.yaml) is a
verbatim copy of tt-train's own `nanollama3.yaml`, and every checkpoint this project has ever
produced is trained against that same Llama-3 architecture (RoPE, RMSNorm, SwiGLU, grouped-query
attention) through `ttml`. Nothing about the rename touched the model's shape or its trainer.

**What earned the new name.** Two things this project now owns that it didn't at the start:
a **nine-source, licence-audited corpus** blended to a 400M-token budget (see
[`docs/corpus_blend.md`](docs/corpus_blend.md) and [`docs/corpus_licensing.md`](docs/corpus_licensing.md)),
in place of a single downloaded TinyStories dump; and a **32,000-token BPE tokenizer trained
on that blend**, rather than inherited from someone else's vocabulary. Those two changes are
what stopped "nanollama3-like" from being an accurate description of this project's identity,
even though the underlying architecture and trainer it sits on did not change.

**Existing artifacts still carry the old name, on purpose.** Checkpoints from before this
rename are named `nanollama3_step*.pkl` and are left untouched — they are evidence of runs
made under the old name, and renaming them would misrepresent when they were produced. New
checkpoints are written as `tt_tnt_step*.pkl`; `train/checkpoint.latest_checkpoint` reads
both naming schemes so an existing checkpoint directory keeps resolving correctly.

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

# Build the nine-source corpus blend (CPU only; downloads several GB, needs ~45 GB free —
# scripts/check_disk_space.py refuses to start if the volume is too full)
python scripts/fetch_corpus.py       # pinned revisions, one file per source
python scripts/prepare_corpus.py     # normalise, strip Gutenberg packaging
python scripts/measure_corpus.py     # availability gate; writes docs/measurements/
python scripts/blend_corpus.py       # -> artifacts/corpus/blend.txt + its manifest

# Train the tokenizer on the blend (CPU only)
python scripts/build_tokenizer.py

# Tokenize the corpus into training arrays (CPU only)
python train/tokenization.py

# Confirm the training config resolves without touching hardware
python train/run.py --dry-run --steps 20

# Train on Blackhole
python train/run.py --steps 20 --batch-size 64
```

`measure_corpus.py` counts tokens with the trained tokenizer when one exists and falls back
to a word approximation when none does, so the first pass through this sequence on a fresh
clone bootstraps from the approximation. Re-running `measure_corpus.py` and `blend_corpus.py`
once a tokenizer exists settles the numbers against the real vocabulary — see
[`docs/corpus_blend.md`](docs/corpus_blend.md), which records what the shipped blend actually
contains and why that ordering is cut where it is.

**TinyStories only, for a quick smoke test** — a ~512 MB single-source corpus rather than the
blend, downloading ~2.2 GB. It is named `corpus.txt`, never `blend.txt`: `build_tokenizer.py`
refuses to write the blend's name from this path, because a TinyStories file called
`blend.txt` is indistinguishable from the real blend to every later step.

```bash
python scripts/build_tokenizer.py --corpus artifacts/corpus/corpus.txt --corpus-mb 512
python train/tokenization.py --corpus artifacts/corpus/corpus.txt
```

`--corpus-mb` applies only to this path. On a corpus that already exists it is a no-op:
the file is trained on as-is, since a head-truncating byte cap would amputate a blend
written one source at a time rather than sample it.

Run the tests with `python -m pytest`. They need no hardware.

## Contributing

This repository follows the plan-then-execute workflow in
[`docs/superpowers/`](docs/superpowers/): a spec, then implementation plans whose steps are
executed and reviewed task by task. The plans record what was verified against source rather
than assumed — including several defects found in the plans themselves.
