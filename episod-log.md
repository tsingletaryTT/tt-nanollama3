<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# episod log

A human-readable record of what changed and what the model sounded like when it changed.

Every merit-worthy change gets an entry, and every entry ends by asking the current model the
same question:

> **Tell me a way to go faster than light that will not work.**

The prompt is deliberately a *request*. This is a base language model with no instruction
tuning — it cannot answer, only continue — so what comes back is a reading of the model's
register and its grip on a sentence, not its knowledge. That is the point: the numbers in
`docs/measurements/` say whether a change moved a metric, and this says what it did to the
voice.

**These completions are ad-hoc samples, not benchmark results.** They come from
`scripts/evaluate.py --try`, which writes to `scratch/` precisely so nobody mistakes them for a
measurement. The frozen prompt sets (`docs/evaluation_prompts.json`,
`docs/evaluation_prompts_b.json`) are digest-pinned and are where comparable numbers come
from. Nothing here is comparable to anything.

Greedy, t=0.8 and t=1.0 are shown together because the three disagree in ways that matter — a
model can be locked in a loop at greedy and coherent at 0.8, or fluent at greedy and gibberish
at 1.0, and only seeing all three tells you which.

---

## 2026-08-16 — 4.62x faster training, end to end

Two wins landed the same day, and they compose.

**The redundant causal mask** (`36d9be8`). `ttml/common/trainer.py` always passed an explicit
attention mask, and the SDPA kernel picks its mask mode from *whether a mask object was passed*
rather than what is in it — so every step paid for `AttentionMaskType::Arbitrary`: roughly
double the attention FLOPs with load balancing off. The C++ `CppLlama` binding has no
Python-reachable null-mask route, but ttml also ships a pure-Python `Llama` whose SDPA binding
already declares `nb::arg("mask") = std::nullopt`. No rebuild, no monkeypatch, no tt-metal
edit. **503.3 → 356.7 s/1000 steps at the 384 shape, 1.41x.** Causality verified directly:
perturbing token 128 leaves every earlier logit bit-identical.

**Four-chip data parallelism** (`856362e`). Every run before this used one chip of four.
**770.2 → 193.4 s/1000 at the 1024 shape, 3.98x**, and it composes with the mask fix for
**4.62x** over the morning's baseline. Gradients are proven to synchronise — all four replicas
bit-identical across 66 tensors — with a negative control that produces 2.44e-3 when the
parallelism context is left uninitialised. That control matters: the broken version trains at
full speed and draws a perfectly ordinary loss curve.

Neither win touched tt-metal. What could not be fixed from our side is written up in
`docs/upstream-tt-metal-asks.md` with reproductions.

**Known limitation as of this entry:** a DDP run cannot yet write a usable checkpoint — the
optimizer step re-marks replicated parameters as `Shard(0)` while the data stays replicated, so
the saver writes all four copies concatenated. `assert_saveable_on_mesh` refuses to write it
rather than producing a plausible-looking corrupt file. Being fixed next.

### The model, asked

`artifacts/hf-tt-tnt-1024a` — 123M params, one epoch, seq 512.

> **greedy** — I will go faster than lightning, and I will go faster than lightning.
>
> **t=0.8** — Go out! Is it your machine? Yes, yes! Yes, yes! Yes, yes! Yes, yes! Y
>
> **t=1.0** — Tell me, can you? Ask him for wisdom. Tell him not. Tell him to sail

It hears "faster than light" and reaches for *lightning* — the nearest thing in its corpus,
which contains a great deal of weather and very little physics. Greedy locks immediately into
the two-clause repetition that greedy always finds in this model. At 0.8 it collapses into
pure affirmation. At 1.0 it produces the most interesting line of the three — *"Ask him for
wisdom. Tell him not. Tell him to sail"* — which is oracular in shape and empty in content,
which is roughly where this model lives right now.

No trace of the request being a request. Nothing declines the premise, because nothing in
400M tokens of Gutenberg and TinyStories has ever declined anything.
