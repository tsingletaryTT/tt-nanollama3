<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!--
  SOURCE OF TRUTH for the Hugging Face model card at episod/tt-tnt.

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
  - sedthh/gutenberg_english
  - biglam/gutenberg-poetry-corpus
  - wikimedia/wikipedia
language:
  - en
---

# TT-TNT (tt-tnt)

A ~22M-parameter Llama-3-style language model **trained from random initialization on
Tenstorrent Blackhole hardware** with `ttml` (tt-train), then converted to Hugging Face
format and numerically verified.

**This is a demonstration of a pipeline, not a capable model.** Please read the limitations
before using it for anything.

## What it demonstrates

That a model can be designed, trained, packaged, and served entirely on Tenstorrent tooling —
TT-native from the first line of code rather than ported afterwards. The full build, including
every dead end, is documented at
[tsingletaryTT/tt-tnt](https://github.com/tsingletaryTT/tt-tnt).

## Model details

| Property | Value |
|---|---|
| Architecture | Llama-3 style — RoPE (θ=500000), RMSNorm, SwiGLU, grouped-query attention |
| Parameters | 22,025,088 |
| Hidden size | 384 |
| Layers | 6 |
| Attention heads / KV groups | 6 / 3 |
| Context length | **512** |
| Vocabulary | 32,000 (byte-level BPE, trained for this model) |
| Weights dtype | bfloat16 |
| Training hardware | One Tenstorrent Blackhole chip (`mesh_shape [1, 1]`) |

Trained on a **single** Blackhole chip. The host is a TT-QuietBox 2 — four Blackhole chips on
two dual-chip p300 cards — but training used one of them; the other three were idle.

The architecture is vendored in this project as
[`train/configs/model/tt-tnt-384.yaml`](https://github.com/tsingletaryTT/tt-tnt/blob/main/train/configs/model/tt-tnt-384.yaml),
a verbatim copy of tt-train's own `nanollama3.yaml`.

## Training

| | |
|---|---|
| Corpus | Nine-source, licence-audited blend — TinyStories, Simple English Wikipedia, and seven curated Project Gutenberg slices (see [`docs/corpus_blend.md`](https://github.com/tsingletaryTT/tt-tnt/blob/main/docs/corpus_blend.md)). 399,594,747 tokens emitted against a 400,000,000-token budget |
| Tokens seen | **353,495,970** — the full training split, **one epoch** |
| Steps | 10,787 at batch 64, sequence length 512 |
| Wall clock | ~58 minutes on a single Blackhole p300c |
| Final train loss | 3.3125 |
| **Final validation loss** | **4.2203** |
| Optimizer | AdamW, constant lr 3e-4, weight decay 0.01, `stochastic_rounding: true` |

## Limitations — please read these

**It has seen its training corpus once, not memorized it.** At batch 64, sequence length 512,
10,787 steps is one epoch over the blend's 353,495,970-token training split — more data and a
longer context than the original TinyStories-only checkpoint (which saw 0.43 of an epoch), but
still a single pass. The validation curve
(`artifacts/checkpoints-tt-tnt-v1/val_losses.jsonl`) falls from 6.7125 at step 500 to about
4.29 by step 10,000, then **plateaus** for the remainder of the run — oscillating between 4.29
and 4.38 for the last ~2,300 steps, including a rise to 4.378125 at step 9000. Read plainly: it
stopped improving well before training ended; this is not a curve that would keep falling given
more steps at the same settings.

**The corpus is now a nine-source, licence-audited blend, and the model's behavior is a mix to
match.** Read against the frozen evaluation set
([`docs/measurements/samples-tt-tnt-v1.md`](https://github.com/tsingletaryTT/tt-tnt/blob/main/docs/measurements/samples-tt-tnt-v1.md),
greedy decoding, 15 prompts): the model sometimes engages with a prompt's own material — sticks,
roses, a procession, bees — where a TinyStories-only baseline would default to a generic moral,
and one long-form prompt stays coherent for its full 60 tokens. But **TinyStories still
dominates**: four of the fifteen prompts collapse into "a little girl named Lily" regardless of
what the prompt was actually about, and several others degenerate into hard repetition loops
under greedy decoding (e.g. "the procession was ready... the procession was ready..."). The
oblique, observational voice this blend targets — closer to Fabre's insect notebooks than to a
children's story — is **not** present in this checkpoint. Treat the promising examples as
evidence of what the blend can nudge toward, not as evidence the voice has arrived; the
Lily-collapses and the repetition loops are the more representative outcome.

**It is a base completion model.** No instruction tuning, no chat template. Give it the opening
of a simple story; do not ask it questions.

**Its context is 512 tokens** (doubled from the original 256-token checkpoint — see Lineage).
Note that `tokenizer_config.json` carries the conventional `model_max_length` sentinel
(~1e18). Do not derive a serving length from it — use `max_position_embeddings`. Serving this
model with a 4k context will silently degrade output.

**Unlike the original checkpoint, this run's RMSNorm layers did learn.** The very first tt-tnt
checkpoint (see Lineage) trained with `stochastic_rounding` disabled, which silently froze all
13 RMSNorm gammas at bfloat16's rounding fixed-point of 1.0 — the gradients were real, but every
update rounded back to 1.0 and was discarded. This run set `stochastic_rounding: true`, and it
worked: read directly from the final checkpoint (`tt_tnt_step00010787.pkl`), all 13 gammas have
moved off 1.0 and are no longer degenerate (per-tensor means 0.86–1.70, per-tensor standard
deviation 0.036–0.21, values spanning roughly 0.625–2.19 across the set). The model trained with
its normalization layers genuinely live this time.

## Verification

The Hugging Face conversion is not merely assumed correct. It is checked against an
**independently derived pure-NumPy reimplementation** of ttml's forward pass — written from
ttml's C++ source rather than from the converter, so the two paths reach logits by different
routes. They agree to a **maximum absolute logit difference of ~6e-6** (correlation ≈ 1 − 1e-13).

This mattered: an earlier conversion loaded cleanly, tied its weights correctly, showed sensible
next-token entropy, and generated fluent prose — while computing the wrong function, because of
a RoPE row-layout mismatch worth 1.3 nats. Only numerical comparison caught it. That story, and
the techniques that catch this class of bug, are written up in
[`docs/model-development-troubleshooting.md`](https://github.com/tsingletaryTT/tt-tnt/blob/main/docs/model-development-troubleshooting.md).

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("episod/tt-tnt")
model = AutoModelForCausalLM.from_pretrained("episod/tt-tnt").eval()

ids = tok("Once upon a time, there was a little", return_tensors="pt").input_ids
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.95)
print(tok.decode(out[0], skip_special_tokens=True))
```

Runs on CPU; no Tenstorrent hardware required for inference.

## Sample output

Greedy decoding, 60 new tokens, from the frozen evaluation set
([`docs/measurements/samples-tt-tnt-v1.md`](https://github.com/tsingletaryTT/tt-tnt/blob/main/docs/measurements/samples-tt-tnt-v1.md)):

> The old woman kept bees behind the house, and every morning she **would go out and pick up
> the honey and put it in her basket. One day, she was walking in the garden when she saw a
> big, red apple. She was so excited and wanted to pick it. She picked it up and took a big
> bite. It was so sweet and juicy!**

Chosen because it is the most coherent completion in the frozen set of 15, not because it is
typical — see Limitations above and the linked file for the honest range, including four
TinyStories collapses to "a little girl named Lily" and several hard repetition loops.

## Licensing and provenance

**The model weights and this project's code are Apache-2.0.**

**The training corpus is not.** This checkpoint was trained on the nine-source blend described
above. Two of those sources are share-alike: `tinystories`
([`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories), 31% of the
blend, CDLA-Sharing-1.0) and `wikipedia_simple`
([`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia), 15% of the
blend, CC-BY-SA-3.0). Full per-source licence, attribution, and the pinned dataset revisions
are recorded in
[`docs/corpus_licensing.md`](https://github.com/tsingletaryTT/tt-tnt/blob/main/docs/corpus_licensing.md),
which is *generated* from this project's source registry (`train/corpus.py`) rather than
hand-written, specifically so this card cannot drift out of sync with it the way hand-written
licensing prose has before. That document's "unsettled Data Derivative" language for
share-alike sources applies to this checkpoint exactly as written there, for both of the
sources named above; this card does not restate it.

**Architectural credit.** The component choices — RoPE, RMSNorm, SwiGLU, GQA, subword BPE —
follow [Mini-LLM by Ashx098](https://github.com/Ashx098/Mini-LLM), which the originating lesson
arc credits. That repository declares no license, so it grants no rights; this is a credit, not
a license inheritance. The components come from published papers, and this implementation
derives from tt-train's `nanollama3` config and the `ttml` library, not from Mini-LLM's source.

## Lineage

This model was originally published under the name **tt-nanollama3**, and it is worth being
plain about what changed. It started as a hand-rolled nanollama3-like model — a Llama-3
architecture trained from random initialization with tt-train's `ttml` trainer, on TinyStories.
The architecture and trainer have not changed: the config above is a verbatim copy of
tt-train's own `nanollama3.yaml`, and every checkpoint (including the one this card describes)
is trained through `ttml` against it. What changed is the corpus and tokenizer this project
now owns — a nine-source, licence-audited blend and a BPE tokenizer trained on that blend,
rather than a single downloaded corpus and an inherited vocabulary — and that is what earned
the new name, `tt-tnt`. (This is now the first `tt-tnt` checkpoint actually trained on that
corpus: the training table above records a run over the nine-source blend at a 512-token
context, not TinyStories alone at 256. An earlier TinyStories-only, 256-token checkpoint from
before the corpus work existed under this same name for a time; this card now describes the
blend-trained checkpoint that superseded it.)

## Origin

Built out of the "Build an LLM from Scratch" lesson arc in
[tt-vscode-toolkit](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/lfs-00-intro/), which
builds a Llama-3-style model TT-native from the first line of code. This model takes that arc
past where the lessons stop — real training, checkpointing, conversion, and numerical
verification.
