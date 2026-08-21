---
license: apache-2.0
library_name: transformers
tags:
  - tenstorrent
  - blackhole
  - llama
---

# tt-tnt-1024

A 123M-parameter Llama-3-style model trained from random initialization on
Tenstorrent Blackhole hardware with `ttml` (tt-train), on a nine-source corpus
that includes a small instruction slice.

This card describes what the model does and where it fails. It is not a claim
that the model is good.

## Shape

| | |
|---|---|
| parameters | 122,962,944 |
| hidden size / layers / heads | 1024 / 8 / 16 (4 KV heads) |
| context | 512 |
| vocabulary | 32,000 (BPE, trained on this corpus) |
| training | 10,764 steps, batch 64, seq 512, 4-chip DDP on one p300c |
| final validation loss | 2.8230 |

4 KV heads means it shards across 1, 2 or 4 chips without violating
head-divisibility.

## What it does

It continues text, and — unlike earlier checkpoints in this project — it will
answer a question in the shape of an answer. Under greedy decoding:

> **Q: What is the capital of France?** → Answer: The capital of France is the city of Paris.
>
> **Q: What is the capital of Italy?** → Answer: The capital of Italy is the city of Rome.

Both correct. The corpus carries `wikipedia_simple` at 15%, which is where facts
of that kind live.

## What it gets wrong

> **Q: How many legs does a spider have?** → Answer: The body of a spider is a frog.
>
> **Q: What color is the sky?** → Answer: The sky is a blue color that is blue because it is blue because it is blue…

The second is a measured regression, not an anecdote. Against `tt-tnt-1024a`
(same architecture, same steps, corpus without the dialogue slice):

| signal | delta | vs seed floor | verdict |
|---|---|---|---|
| 4-gram repeat rate | +0.0074 | 3.32× | **worse** |
| termination rate | −0.0076 | 0.52× | not interpretable |
| genre collapse | −0.0035 | 0.06× | not interpretable |
| loss at matched window | +0.0102 | — | no floor for this instrument |

Nine of ten behavioural signals came back NOT INTERPRETABLE against this
project's 0.1944-nat seed-only noise floor. The one finding that cleared both
gates is that repetition got **worse**. A prediction that short question-answer
documents would improve termination was not supported.

Full comparison:
`docs/measurements/evaluation-tt-tnt-1024a-vs-tt-tnt-1024-dialogue.md`.

## Experiments this checkpoint is the base for

Two things were measured on top of this checkpoint on 2026-08-20. Both are recorded with their
limits, because both are easy to overstate.

**Sparse routing (Mixture of Enthusiasts) beats dense from scratch.** Replacing the
feed-forward with `ttml`'s sparse MoE and training both arms one epoch from init, paired on
seed 5489: validation **2.8098 for MoE against 2.8748 for dense** (mean delta +0.0481,
|t| 7.3, 20 of 22 signs), and the gap widens across training. Read it as the ordinary MoE
bargain — the configuration carries **3.62× total parameters at 0.989× active compute**, so
more parameters for the same compute helped. It is *not* evidence about the die-region routing
below. **Replicated at a second seed** (8191: +0.0354, |t| 4.5, 19/22 signs; pooled +0.0417 over
44 points), with the same late-separating trajectory in both runs, so treat ~0.04 as the
estimate.

**Routing by physical die address is nearly free.** Tokens can be routed to experts by where
they live on the harvested Tensix grid rather than by a learned gate. Freezing the gate to that
geography — never letting it learn — costs only **0.0118 nats** against a freely-learned gate
(|t| 5.1, 14/15 signs). Seeding the gate from the die map and then letting it move buys nothing
measurable (+0.0044, signs 8+/7−), even though the seeding demonstrably works as a classifier
(61.2% region recovery against a 10% chance floor). The geometry is real; the loss does not care
where the gate starts, only where it may end up.

**A five-slot think-block can be learned, and does not yet help.** Fine-tuned to emit
`offer / accept / add / stakes / handback` before continuing a story — one slot per improv
failure mode (escalating to the worst place, blocking with the dullest next step, drifting too
far out) — the model produces well-formed blocks in **98%** of generations (784/800; the
no-think control produces them 0% of the time). Substituting another story's block changes
**100%** of continuations, so the block steers rather than decorates. But it moves **none** of
the four failure-mode scores at α = 0.01, and one of those four is saturated on the real
co-occurrence table and cannot discriminate at all. Stage 1 is *partial*.

The generations explain the null better than the scores do. Asked to continue a story, the
model planned `add: dance` / `handback: dance` and then wrote a scary dog; another block set
`stakes: up` and the scene resolved into contentment. The syntax of intention is perfect and the
intention is not honoured. Read alongside the swap test that names it precisely: the block is
*context the model conditions on, not an instruction it obeys* — change it and the output moves,
ask it to mean something and it shrugs. On the same opening the no-think arm writes plainly
better prose. A plainer contributing reason: the slots are telegraphese (*loved play outside
friends*), because derivation lifts content words and drops the rest, so the model was asked to
produce a register nothing in 400M tokens of storybook prose resembles — and then to let that
register steer one it knows fluently.

Next unit is a **skit**: two or more turns with a partner who answers. A single continuation
gives `handback` nothing to hand back to, so the slot that encodes "make your partner look good"
cannot pay off or fail. Close reading in
[`episod-log.md`](https://github.com/tsingletaryTT/tt-tnt/blob/main/episod-log.md), 2026-08-21.

One process note kept deliberately: an earlier pass reported 0% adherence, from a run in which
all 17 RMSNorm gammas were provably frozen because `stochastic_rounding` defaults off on the SFT
path. With the gammas free, 0% became 98%. Both runs are preserved in the repo's measurement
files.

## What it cannot do

No instruction tuning beyond a 2% slice of `databricks-dolly-15k`. No chat
template. No system prompt. It repeats under greedy decoding. It has 512 tokens
of context. It is a small model trained for one epoch on 352.6M tokens, and it
should be treated as an artifact of a hardware-and-tooling project rather than as
a useful assistant.

## Corpus

Nine sources, blended to a 400M-token budget and shipped as a **recipe** rather
than as text, because 46% of it is share-alike under two mutually incompatible
copyleft terms. Reconstruct it from
[`episod/tt-tnt-corpus`](https://huggingface.co/datasets/episod/tt-tnt-corpus).

The dialogue slice is `databricks-dolly-15k` (CC-BY-SA-3.0) at 2%, rendered as
plain `Question: … / Answer: …` prose with no role markers — the tokenizer has no
vocabulary for chat scaffolding.

## Serving

Through the Tenstorrent vLLM plugin. Use a plugin at or after `c127c17`: earlier
builds show a decode defect that degrades free-running generation into repetition
within a few tokens. The plugin reports version `0.1.0` either way, so a version
check cannot detect this; the bundle's adapter warns structurally instead.
