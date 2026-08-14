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
  - episod/tt-tnt-corpus
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
| Context length | **2048** |
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
| Corpus | Nine-source, licence-audited blend — TinyStories, Simple English Wikipedia, and seven curated Project Gutenberg slices (see [`docs/corpus_blend.md`](https://github.com/tsingletaryTT/tt-tnt/blob/main/docs/corpus_blend.md)). **This is the first tt-tnt checkpoint trained on a blend that carries document separators (`</s>`)** — earlier revisions contained none at all, so no previous checkpoint had ever seen an end-of-document token and none could stop generating. Measured on the token arrays directly: the pre-fix array holds **zero** id-2 tokens across the whole training split; this one averages one per ~478 tokens (one per ~210 in the short-document sources, one per ~80,000 in the book-length ones). The blend recipe — source registry, fetch/prepare/measure/blend scripts, and the provenance manifest — is published separately as [`episod/tt-tnt-corpus`](https://huggingface.co/datasets/episod/tt-tnt-corpus) |
| Tokens seen | **352,714,752** — the full training split, **one epoch** |
| Steps | 10,764 at batch 16, sequence length 2048 (32,768 tokens/step, unchanged from the previous run's 64×512) |
| Wall clock | ~91 minutes on a single Blackhole p300c |
| Final train loss | 3.25 |
| **Final validation loss** | **2.9937** — the end-of-run figure from `train/run.py`'s `evaluate()`. The periodic curve's last entry (`artifacts/checkpoints-tt-tnt-v3/val_losses.jsonl`, step 10,764) reads 2.939; the two differ only by which held-out windows each sampled. **Do not read this as an improvement on the previous checkpoint's 4.2203** — see below |
| Optimizer | AdamW, constant lr 3e-4, weight decay 0.01, `stochastic_rounding: true` |

## Limitations — please read these

**The headline validation loss is not comparable to the previous checkpoint's.** 2.9937 against
4.2203 looks like a large gain, and most of it is not one. The previous checkpoint's validation
split was the tail 10% of a token stream whose sources are concatenated in sorted-name order,
so it landed **entirely inside `wikipedia_simple`** — the most out-of-domain source in the
blend, and by per-source measurement the second-hardest (held-out loss 4.28, against
TinyStories' 1.83; see
[`docs/measurements/per-source-loss-tt-tnt-v1.md`](https://github.com/tsingletaryTT/tt-tnt/blob/main/docs/measurements/per-source-loss-tt-tnt-v1.md)).
That number measured domain transfer, not learning. This run's split is **stratified by
source** — a proportional tail from each of the nine — so it is a fair sample of the training
mixture, and a much easier one, because 31% of the mixture is TinyStories. A meaningful share
of the drop from 4.22 to 2.99 is the yardstick changing, not the model improving. The two
numbers should not be subtracted.

**It has seen its training corpus once, not memorized it.** At batch 16, sequence length 2048,
10,764 steps is one epoch over the blend's 352.7M-token training split — the same 32,768
tokens per step as the previous run, at four times the context. The validation curve
(`artifacts/checkpoints-tt-tnt-v3/val_losses.jsonl`) falls from 5.084 at step 500 to ~3.28 by
step 4,000, and then keeps drifting down slowly and noisily — the last ~2,300 steps oscillate
between 2.87 and 3.10 against ~3.12 around step 7,000–8,000. That is a real if modest continued
decline, unlike the previous run, which was flat for its final stretch. Read plainly: this one
had not clearly stopped improving when it ended, but the per-step gain over the last third is
small enough that more steps at these settings would not transform it.

**It can now stop, which no previous checkpoint could.** Every earlier tt-tnt checkpoint was
trained on a corpus containing **zero** `</s>` tokens while its `config.json` nonetheless
declared `eos_token_id: 2` — so generation could never terminate naturally and always ran to
whatever token limit the caller set. This is the first checkpoint trained on a corpus that
marks document boundaries, and it does terminate: on the frozen 15-prompt evaluation set at
128 max new tokens, **5/15 completions end on `</s>` under greedy decoding** (median 48 tokens)
and **11/30 under sampling** at temperature 0.8 / top_p 0.95 (median 91). The comparable
numbers for the previous checkpoint are **0/15 and 0/30** — not "rarely", but never, by
construction. This matters most for chained generation, where you want a passage to end rather
than be cut off mid-sentence at the limit.

It is a partial fix, not a solved problem: two thirds of completions still run to the limit,
and the model is much readier to stop on short-document material (TinyStories-like prompts)
than on the book-length sources, which is what the corpus taught it — separator density in the
blend ranges from one per ~210 tokens in the short sources to one per ~80,000 in the books.

**The corpus is a nine-source, licence-audited blend, and the model's behavior is a mix to
match.** Read against the frozen evaluation set
([`docs/measurements/samples-tt-tnt-v3.md`](https://github.com/tsingletaryTT/tt-tnt/blob/main/docs/measurements/samples-tt-tnt-v3.md),
greedy decoding, 15 prompts): the model sometimes engages with a prompt's own material — sticks,
roses, a procession, bees — where a TinyStories-only baseline would default to a generic moral.
But **TinyStories still dominates**, and under greedy decoding several prompts still degenerate
into hard repetition loops ("the rose is a rose, and the rose is a rose, and..."; "the bees were
busy, and the bees were busy"). The oblique, observational voice this blend targets — closer to
Fabre's insect notebooks than to a children's story — is **not** present in this checkpoint.
Under sampling the output is markedly better behaved
([`docs/measurements/samples-tt-tnt-v3-t0.8.md`](https://github.com/tsingletaryTT/tt-tnt/blob/main/docs/measurements/samples-tt-tnt-v3-t0.8.md)),
which is the honest way to read the greedy loops: greedy decoding on a 22M-parameter model
manufactures repetition attractors that sampling largely avoids. Treat the promising examples
as evidence of what the blend can nudge toward, not as evidence the voice has arrived.

**It is a base completion model.** No instruction tuning, no chat template. Give it the opening
of a simple story; do not ask it questions.

**Its context is 2048 tokens** (256 → 512 → 2048 across the three checkpoints — see Lineage).
Note that `tokenizer_config.json` carries the conventional `model_max_length` sentinel
(~1e18). Do not derive a serving length from it — use `max_position_embeddings`. Serving this
model with a 4k context will silently degrade output; serving it at 512 silently discards three
quarters of the context it was trained to use.

**Whether it actually uses all 2048 tokens is a separate question from whether it was trained
at 2048.** A position-wise loss probe on the *previous* checkpoint found per-token loss flat
from position ~64 onward — that model extracted nothing from distant context, because the
corpus had no document boundaries and so distant context genuinely was unpredictable. Fixing
the separators is what makes a longer window worth having, and is why this run raised it. The
equivalent probe for this checkpoint
([`docs/measurements/context-use-tt-tnt-v3.md`](https://github.com/tsingletaryTT/tt-tnt/blob/main/docs/measurements/context-use-tt-tnt-v3.md))
shows loss still falling well past where the old one went flat — 4.23 at positions [0,32),
3.28 at [32,64), 2.94 at [64,128), 2.85 at [256,512) — but the improvement past ~256 tokens is
about 0.02 nats per bucket against a standard error of ~0.08, i.e. **directionally right and
inside the noise**. The honest claim is that the long window is no longer actively useless, not
that all 2048 tokens are earning their keep.

**Unlike the original checkpoint, this run's RMSNorm layers did learn.** The very first tt-tnt
checkpoint (see Lineage) trained with `stochastic_rounding` disabled, which silently froze all
13 RMSNorm gammas at bfloat16's rounding fixed-point of 1.0 — the gradients were real, but every
update rounded back to 1.0 and was discarded. This run set `stochastic_rounding: true`, and it
worked: read directly from the published weights, all 13 gammas have moved off 1.0 and none is
degenerate (per-tensor means 0.874–1.720, per-tensor standard deviation 0.034–0.207, values
spanning roughly 0.691–2.359 across the set; zero tensors have standard deviation 0). The model
trained with its normalization layers genuinely live.

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
([`docs/measurements/samples-tt-tnt-v3.md`](https://github.com/tsingletaryTT/tt-tnt/blob/main/docs/measurements/samples-tt-tnt-v3.md)):

> Once upon a time, there was a little **girl named Lily. She loved to play outside in the
> park. One day, she saw a big, shiny rock on the ground. She picked it up and showed it to
> her mom. "Look, Mommy! I found a shiny rock!" she said. Her mom smiled and said, "That**

And one that **stops on its own** rather than being cut off at the limit — the new behavior
described in Limitations:

> The ants had learned that being eaten was a way of **helping others. The moral of the story
> is that it's important to be kind to others and to help others.**

Both are near the top of the frozen set of 15, not typical of it — see Limitations above and
the linked file for the honest range, including the TinyStories collapses to "a little girl
named Lily" and several hard repetition loops. Sampled output at temperature 0.8 is in
[`docs/measurements/samples-tt-tnt-v3-t0.8.md`](https://github.com/tsingletaryTT/tt-tnt/blob/main/docs/measurements/samples-tt-tnt-v3-t0.8.md)
and is the more representative read of the model's range.

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

The corpus itself is not redistributed here or anywhere else — this repository only ships the
*recipe* to reconstruct it byte-identically: source registry, pinned revisions, and
fetch/prepare/measure/blend scripts, published as
[`episod/tt-tnt-corpus`](https://huggingface.co/datasets/episod/tt-tnt-corpus) on the Hub.

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
the new name, `tt-tnt`.

Three checkpoints have now been published under this repo id, and it is worth being plain
about which is which, because they differ only in weights and one config field:

| | corpus | context | document separators |
|---|---|---|---|
| the original (TinyStories-only) | TinyStories alone | 256 | none |
| the first blend-trained checkpoint | nine-source blend | 512 | **none** |
| **this one** | nine-source blend, separator-carrying revision | **2048** | **yes** |

The middle checkpoint was the first trained on the blend; this one is the first trained on a
revision of that blend which marks where one document ends and the next begins, and the first
whose declared `eos_token_id: 2` corresponds to a token the training data actually contained.
That, rather than the context length, is the substantive difference.

## Origin

Built out of the "Build an LLM from Scratch" lesson arc in
[tt-vscode-toolkit](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/lfs-00-intro/), which
builds a Llama-3-style model TT-native from the first line of code. This model takes that arc
past where the lessons stop — real training, checkpointing, conversion, and numerical
verification.
