<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!--
  SOURCE OF TRUTH for the Hugging Face model card at episod/tt-nanollama3.

  Kept in this repo because tt-kernel's `tag_repo` (hub.py:56-66) replaces the card's
  front matter wholesale with `ModelCardData(tags=...)` on every `tt-kernel push`,
  destroying `license`, `pipeline_tag`, `library_name`, and `datasets`. The prose body
  survives; the metadata does not. After any tt-kernel operation, re-apply the front
  matter below and verify it stuck.
-->

---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
  - tenstorrent
  - blackhole
  - llama
  - tt-metal
  - ttml
  - trained-from-scratch
datasets:
  - roneneldan/TinyStories
language:
  - en
---

# NanoLlama3 (tt-nanollama3)

A ~22M-parameter Llama-3-style language model **trained from random initialization on
Tenstorrent Blackhole hardware** with `ttml` (tt-train), then converted to Hugging Face
format and numerically verified.

**This is a demonstration of a pipeline, not a capable model.** Please read the limitations
before using it for anything.

## What it demonstrates

That a model can be designed, trained, packaged, and served entirely on Tenstorrent tooling —
TT-native from the first line of code rather than ported afterwards. The full build, including
every dead end, is documented at
[tsingletaryTT/tt-nanollama3](https://github.com/tsingletaryTT/tt-nanollama3).

## Model details

| Property | Value |
|---|---|
| Architecture | Llama-3 style — RoPE (θ=500000), RMSNorm, SwiGLU, grouped-query attention |
| Parameters | 22,025,088 |
| Hidden size | 384 |
| Layers | 6 |
| Attention heads / KV groups | 6 / 3 |
| Context length | **256** |
| Vocabulary | 32,000 (byte-level BPE, trained for this model) |
| Weights dtype | bfloat16 |
| Training hardware | 4× Tenstorrent Blackhole p300c |

Architecture parameters come from tt-train's `nanollama3.yaml` and were not redefined.

## Training

| | |
|---|---|
| Corpus | TinyStories (`roneneldan/TinyStories`), 512 MB subset |
| Tokens seen | **49,152,000** — about **0.43 of one epoch** over the 114.9M-token training split |
| Steps | 3000 at batch 64, sequence length 256 |
| Wall clock | 6 min 47 s (~0.134 s/step steady state) |
| First train loss | 10.6875 — consistent with a near-uniform initial distribution (`ln(32000) = 10.37`) |
| Final train loss | 1.9219 |
| **Held-out validation loss** | **1.8781** |
| Optimizer | AdamW, lr 3e-4, weight decay 0.01 |

## Limitations — please read these

**It has seen less than half of its training data, once.** 0.43 of an epoch is not a trained
model in any conventional sense.

**TinyStories is synthetic and deliberately simple** — children's stories with a small
vocabulary and regular grammar, built so that small models can learn it. A low loss on this
corpus is unsurprising and does **not** transfer to general language ability.

**It is a base completion model.** No instruction tuning, no chat template. Give it the opening
of a simple story; do not ask it questions.

**Its context is 256 tokens.** Note that `tokenizer_config.json` carries the conventional
`model_max_length` sentinel (~1e18). Do not derive a serving length from it — use
`max_position_embeddings`. Serving this model with a 4k context will silently degrade output.

**Its 13 RMSNorm layers never learned.** All gammas are exactly 1.0. The gradients were real
(Adam `exp_avg` ≈ 3.6e-4) but the parameters are bfloat16 at 1.0, where one ulp is 0.0039, and
`stochastic_rounding` was disabled — so every update rounded back to 1.0 and was discarded. The
model trained with its normalization layers effectively frozen. Remedies for anyone reproducing
this: set `stochastic_rounding: true` in the optimizer config, or use
`type: AdamWFullPrecision` (fp32 master weights). Both exist in tt-train today; neither is the
default.

## Verification

The Hugging Face conversion is not merely assumed correct. It is checked against an
**independently derived pure-NumPy reimplementation** of ttml's forward pass — written from
ttml's C++ source rather than from the converter, so the two paths reach logits by different
routes. They agree to a **maximum absolute logit difference of ~6e-6** (correlation ≈ 1 − 1e-13).

This mattered: an earlier conversion loaded cleanly, tied its weights correctly, showed sensible
next-token entropy, and generated fluent prose — while computing the wrong function, because of
a RoPE row-layout mismatch worth 1.3 nats. Only numerical comparison caught it. That story, and
the techniques that catch this class of bug, are written up in
[`docs/model-development-troubleshooting.md`](https://github.com/tsingletaryTT/tt-nanollama3/blob/main/docs/model-development-troubleshooting.md).

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("episod/tt-nanollama3")
model = AutoModelForCausalLM.from_pretrained("episod/tt-nanollama3").eval()

ids = tok("Once upon a time, there was a little", return_tensors="pt").input_ids
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.95)
print(tok.decode(out[0], skip_special_tokens=True))
```

Runs on CPU; no Tenstorrent hardware required for inference.

## Sample output

One sample at temperature 0.8, reported as a single data point rather than as representative:

> Once upon a time, there was a little **girl named Lily. Lily loved to play with her toy car.
> One day, Lily's toy car was broken. Lily was sad. She wanted to fix it. Lily's mom saw her sad
> face and said, "I**

Locally coherent, holds a character, and drifts — which is what 0.43 of an epoch at this scale
looks like.

## Licensing and provenance

**The model weights and this project's code are Apache-2.0.**

**The training corpus is not.** TinyStories (`roneneldan/TinyStories`) is licensed
**CDLA-Sharing-1.0**, a share-alike *data* license. This model was trained on it. Whether model
weights trained on CDLA-Sharing data constitute a "Data Derivative" under that license is **not
settled**, and we do not assert that they fall outside it. Anyone building on these weights
should reach their own conclusion rather than inheriting ours.

**Architectural credit.** The component choices — RoPE, RMSNorm, SwiGLU, GQA, subword BPE —
follow [Mini-LLM by Ashx098](https://github.com/Ashx098/Mini-LLM), which the originating lesson
arc credits. That repository declares no license, so it grants no rights; this is a credit, not
a license inheritance. The components come from published papers, and this implementation
derives from tt-train's `nanollama3` config and the `ttml` library, not from Mini-LLM's source.

## Origin

Built out of the "Build an LLM from Scratch" lesson arc in
[tt-vscode-toolkit](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/lfs-00-intro/), which
builds a Llama-3-style model TT-native from the first line of code. This model takes that arc
past where the lessons stop — real training, checkpointing, conversion, and numerical
verification.
