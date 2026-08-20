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
