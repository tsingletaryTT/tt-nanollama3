<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->


> **Conceptual companion:** [Why the numerics are hard, even building Tenstorrent-first](why-the-numerics-are-hard.md) — the retrospective on *why* these defects cluster where they do: continuous gates certifying discrete outputs, block float's correlated error, decode as a feedback loop, and why a small undertrained model is the worst case for all three.

# Model development troubleshooting

Field notes from building, training, and converting tt-tnt on Tenstorrent hardware. Every
entry here is something that actually happened, with the numbers it produced. Nothing is
hypothetical.

Organised by **symptom**, because that is how you will arrive here.

---

## The one idea worth internalising first

"It loads" and "it generates fluent text" are not evidence of correctness.

Our converted model passed four independent checks and was still computing the wrong function:

| Check | Result | Verdict |
|---|---|---|
| Loads as `LlamaForCausalLM` | ✅ 22.025088M params | passed |
| Embedding/lm_head tied | ✅ `torch.equal` true | passed |
| Next-token entropy | 4.75 nats (uniform = 10.37) | passed |
| Generated text | *"Once upon a time, there was a little girl named Lily…"* | passed |
| **Held-out loss vs the training run** | **3.20 vs 1.8781** | ❌ **wrong** |

The cause was a RoPE row-layout mismatch. Every check above is *structurally incapable* of
detecting it — a rotary embedding applied with the wrong pairing still produces a confident,
coherent model. It just computes something else.

The generalisation: for any check you rely on, ask what class of error it *cannot* see.
Then find an instrument that can. Agreement among checks that share a blind spot is not
corroboration.

---

## Symptom: my converted model loads and generates plausible text, but I don't know if it's right

Do not trust fluency. At small scale, a wrong model and a right model produce similarly
plausible prose. Ours did:

- **Broken** — *"She loved to dance. She loved to dance. One day, she danced every day, she found a big, blue flower."*
- **Correct** — *"Max loved to play with his ball. One day, Max saw a big ball in the park."*

The second is better on inspection, but that judgment is one sample against one sample. The
reliable difference was **1.32 nats measured over 320 windows**.

What to do instead — compare against the training run's own loss. This is the single most
valuable check available, and it costs nothing if you recorded the number during training.

```python
# Correct: HF shifts labels internally
out = model(x, labels=x)
```

Two traps here:

1. **Do not pre-shift the labels.** `model(x[:, :-1], labels=x[:, 1:])` double-shifts, because
   `LlamaForCausalLM` already shifts internally. We hit this twice — once in a plan's example
   command, once by accident while verifying the fix. It reports ~9.0 nats on a *correct*
   model, which looks like catastrophic failure rather than a measurement error.
2. **Know your noise floor.** Ours: per-window sd ≈ 0.29 nats; an 8-window mean has SE ≈ 0.11.
   So a 0.05-nat difference is meaningless and a 1.3-nat difference is decisive. Compute the
   standard error before interpreting a gap.

---

## Symptom: I need to know whether a check would actually catch the bug it's meant to catch

Break it on purpose and measure. This is the highest-value technique in this document.

Revert the fix in a scratch copy, re-run, and record the number. If the check doesn't move, it
is not protecting you. Our measurements:

| Deliberate defect | Loss impact |
|---|---|
| RoPE flipped to split-halves | 3.13 (vs 1.84) |
| K/V split order reversed | 7.59 |
| `tile` instead of `repeat_interleave` for GQA | 3.72 |
| Attention layers 0↔1 swapped | +4.70 |
| One layer's `q_proj` left un-permuted | +0.69 |
| gate/up projections swapped | +0.43 |
| **Two RMSNorm layers swapped** | **+0.0000** ← blind spot |
| RMSNorm epsilon moved outside the sqrt | −0.0002 ← invisible, but immaterial |

That last pair is the point. A gate with a 0.2-nat tolerance catches everything above the line
and **nothing** below it. Knowing where your floor sits is the difference between a test and a
comfort blanket.

---

## Symptom: a layer doesn't seem to be learning

Check whether your optimizer updates are smaller than one ulp of your parameter dtype.

All 13 of our RMSNorm gammas were *exactly* 1.0 after 3000 steps. The gradients were real —
`exp_avg` absmax ≈ 3.6e-4 — but the parameters were bf16 at 1.0, where one ulp is 0.0039. With
`stochastic_rounding: False`, every update rounded straight back to 1.0. The optimizer computed
them and threw them away, silently, every step.

How to check: load a checkpoint and look at the parameter statistics, not the loss.

```python
print(gamma.mean(), gamma.std())   # sd of exactly 0.0 is a red flag
```

Fixes: enable `stochastic_rounding`, or use fp32 master weights.

Why this bites hardest at initialisation values: a parameter initialised at 1.0 with small
gradients is the worst case, because the ulp is large relative to the update. Parameters near 0
have much finer resolution.

---

## Symptom: my tokenizer didn't reach the vocabulary size I asked for

`vocab_size` is a ceiling, not a promise. BPE stops when the corpus runs out of pairs worth
merging. Our fixture corpus of five repeated sentences reached **283** against a 400 target;
another reached **378** against 500.

The production corpus reached exactly 32000 — but only because it was large enough.

What to do: after training a tokenizer, reload the export and assert the achieved size,
failing loudly on a mismatch. Otherwise a small `--corpus-mb` silently produces a vocabulary
that disagrees with your model config, and the first symptom is an embedding-shape error much
later, in a completely different component.

---

## Symptom: my model's first loss looks wrong

A freshly initialised model's cross-entropy should be ≈ `ln(vocab_size)`.

For 32000 tokens that is **10.37**. Ours started at **10.6875**.

This is the cheapest possible check that your model, tokenizer, and vocabulary all agree with
each other, and it costs one training step. If the first loss is far from `ln(vocab)`:

- **Much higher** — something is wrong with initialisation, or the vocabulary is larger than
  you think.
- **Much lower** — you may be accidentally resuming from weights, or your loss is not what you
  think it is.

Do the same check at the *other* end: a model that has learned something should show
next-token entropy well below `ln(vocab)`. Ours: 4.75 nats.

---

## Symptom: my resumed run starts at the wrong loss

A resumed run's first loss should be near the previous run's last loss. If it restarts near
`ln(vocab)`, the weights did not load — and the checkpoint file will still exist, still be a
plausible size, and still have a valid header. Nothing but the loss curve catches it.

Ours: run 1 ended at 9.5000, resumed run began at 9.3125. That continuity is the evidence.

Related trap — "latest checkpoint" that isn't. If checkpoint filenames embed an unpadded
step number, sorting them lexicographically gives `step10 < step100 < step9`. Your resume
silently loads step 9 after step 100. Zero-pad the step (`step00003000`), and write a test that
asserts the sorted order differs from the naive one — otherwise the test can quietly become
vacuous.

---

## Symptom: my training-side validation loss looks suspiciously like the training loss

Check whether your framework actually computes validation. ttml's `train()` does this:

```python
# keep existing placeholder behavior for validation loss
val_losses.append(train_losses[-1] if train_losses else 0.0)
```

It copies the training loss. Any "val_loss" from that path is the training loss wearing a
different label — including the one shown live in the progress bar.

How to spot it: if train and val agree to the last decimal, they are the same number. Write
your own evaluation loop over held-out data, and if you display the framework's placeholder
anywhere, label it as such.

---

## Symptom: tensors seem mismatched after loading a checkpoint

Stream order is declaration order, not sorted order. ttml writes tensor records in the
order its module walk emitted them — for us, block 5 came first, then block 4, with block 0 at
record 21. A helper that returns *sorted* names paired against that stream mis-assigns **every
tensor**.

It is shape-silent: `q_linear`, `kv_linear`, and `out_linear` are all `(1,1,384,384)`, so
nothing complains. The model loads and generates nonsense.

What to do: walk the manifest in insertion order, never a sorted list. Pin it with a test
that asserts `declared_order != sorted(declared_order)` — so the test cannot silently become a
no-op if the order ever happens to be alphabetical.

---

## Symptom: `from_pretrained` fails on a shape mismatch in a norm layer

Check your leading-dimension squeeze. ttml stores 2-D weights as `(1, 1, out, in)`, so a
naive `while ndim > 2: squeeze` works for weights — and leaves 4-D norm gammas at `(1, 384)`
instead of `(384,)`.

The condition should be `while ndim > 1 and shape[0] == 1`. Verify it against a tensor of every
rank you actually have, not just the common one. Our tests covered `(1,1,384,384)` and
`(384,384)`, and both the code and the tests were wrong for `(1,1,1,384)` — they agreed with
each other, which is why review didn't catch it.

---

## Symptom: my converted model has no embedding tensor

That is expected under weight tying. With tying enabled, ttml registers only
`llama/fc/weight` — there is no `tok_emb` tensor at all. A converter that expects one produces
a model with a randomly-initialised embedding table **and no error**.

Write `fc/weight` to *both* `model.embed_tokens.weight` and `lm_head.weight`.

And handle the untied case explicitly. When tying is off, ttml registers `tok_emb/weight`
*in addition to* `fc/weight`. If your mapper silently skips unknown names, the real embedding
is discarded while `fc/weight` still lands in both slots — `from_pretrained` reports "All model
checkpoint weights were used" and the model is wrong.

Rule: raise on any unmapped tensor. Never `continue`.

---

## Symptom: attention weights don't split the way I expect

Check for fused projections. ttml packs K and V into a single `kv_linear` weight, with
output dimension `num_groups × head_dim × 2`. It must be split into HF's separate `k_proj` and
`v_proj`, K first.

Reversing that order cost us **7.59 nats** in an ablation — loud, fortunately. But validate the
row count rather than trusting it; a silent mis-split on a differently-shaped model would not
be.

---

## Symptom: RoPE-related, or "everything looks right but the loss is too high"

Rotary embeddings have two incompatible conventions, and they produce identical shapes:

- **Interleaved** — rotate `(x[2i], x[2i+1])`
- **Split-halves** — rotate `(x[i], x[i+d/2])`

ttml uses **interleaved**; Hugging Face Llama uses **split-halves**. Converting between them
requires permuting q and k rows per head — and **only** q and k, since RoPE rotates only those.

Getting it wrong costs ~1.3 nats and is invisible to every structural check.

How we established the convention (four ways, worth copying as a method):

1. The frequency table uses an integer divide `arange(head_dim) / 2`, so adjacent channels
   share a frequency — necessary for interleaved, inconsistent with split-halves.
2. The rotation matrix is 2×2 block-diagonal.
3. The framework's *own* HF importer un-permutes split-halves→interleaved for q and k, and not
   for v — the convention confirmed from the opposite direction.
4. The device kernel applies one 32×32 matrix per tile independently; with `head_dim = 64`,
   split-halves would pair column `i` with `i+32`, crossing a tile boundary — **structurally
   impossible**.

Argument 4 is the strongest because it does not depend on anyone's intent.

---

## Symptom: my parity check fails and I can't tell whose fault it is

Measure what agreement is achievable before asserting what is required.

An unreachable tolerance is the worst property a diagnostic can have: every failure becomes
ambiguous between "the thing under test is wrong" and "the gate was impossible," and you cannot
distinguish them without redoing the analysis that should have come first.

Run a control comparison — same code, same inputs, only the incidental difference you are
trying to tolerate — and set the gate from what you measure.

Be precise about which comparison you are bounding. We nearly asserted that a bf16 rounding
artifact inside the device's RMSNorm made a tight tolerance unachievable — true for a
*host-vs-device* comparison, irrelevant to the *host-vs-host* one we were actually building.
Write down which two things you are comparing before reasoning about what limits them.

---

## Symptom: my first training step is much slower than the rest

You are timing the compiler, not the model. Our first step took 18.7 s; steady state was
0.12–0.14 s/step — a 140× difference. Never quote the first step as throughput.

A later run showed no warm-up stall at all, most likely a warm kernel cache. Treat that as an
observation, not a guarantee.

---

## Symptom: my serving stack accepts contexts far longer than the model was trained on

Check `model_max_length` in `tokenizer_config.json`. Ours reads
`1000000000000000019884624838656` — the "no limit" sentinel. Anyone deriving a serving
`max_model_len` from the tokenizer gets a stack that happily accepts 4k contexts from a model
trained to 256. Degraded output, no error.

Pin the serving length from `max_position_embeddings` in the model config, never the tokenizer.

---

## Symptom: my framework's model cache isn't caching

Some caches are opt-in and silently disabled. ttnn's `preprocess_model_parameters` only
writes to disk when you pass `model_name` — its docstring says "If not provided, the cache will
be disabled." Omit it and every run reconverts from scratch, while any comment nearby claiming
results are "cached after the first run" is true only within a single process.

Generalisation: when a comment claims something is cached, verify a *second process* is
faster. In-process memoisation and on-disk caching are easy to confuse.

---

## Techniques worth having in hand

Ablation. Break something deliberately, measure the impact, restore. This tells you what
your checks can and cannot see, and it converts "the test passes" into "the test would fail if
this were wrong."

Paired comparison. When comparing two implementations, evaluate them on *identical* inputs
rather than independent samples. It turns a noisy two-sample comparison into a paired one and
can improve resolution by an order of magnitude.

Independent reimplementation. The strongest correctness evidence is two implementations
derived by different routes agreeing. But it only works if they *are* independent — if the
second is written by reading the first, it inherits the first's misunderstandings and agrees by
construction, proving nothing.

Write the convention down before implementing it. Our RoPE bug was a derivation error that
survived because nobody wrote the convention anywhere reviewable. A short document with source
citations can be reviewed on its own, before any code depends on it.

Check the arithmetic. `114,872,301 + 12,763,588 = 127,635,889` and `.npy` byte sizes
matching `count × 4 + 128` caught nothing on their own — but they cost seconds and would have
caught a truncated file immediately.

---

## Checks and what they cannot see

| Check | Catches | Blind to |
|---|---|---|
| Model loads | missing/misshaped tensors | any value or layout error |
| Shape assertions | structural mistakes | permutations, swaps within a shape |
| Weight tying assertion | untied when it should be tied | which tensor got tied |
| Entropy vs `ln(vocab)` | catastrophically broken models | layout errors (ours: 4.75, and wrong) |
| Generated text reads well | gross failure | ~1.3-nat errors |
| Held-out loss vs training | most layout errors (>0.2 nats) | anything below the noise floor; **frozen-value tensors entirely** |
| Logit comparison vs an independent implementation | nearly everything | what both implementations get wrong |

Nothing on this list validates a tensor whose values are all identical. Our 13 norm gammas
were all exactly 1.0, so swapping two changed the loss by 0.0000. Validate those with a
synthetic fixture carrying distinct values, not with the real checkpoint.

---

## A closing note on process

Across five plans, nearly every defect originated in a *plan* rather than an implementation —
and specifically in a plan's prose or tables claiming something its own code never delivered.
Two habits caught most of them:

- **Verify assertions against reality before dispatching them.** Checking that a fixture
  tokenizer actually reaches 500 tokens costs one command; not checking costs a full
  implement-review-fix cycle.
- **If a step produces a number that decides pass/fail, make it a test.** We predicted the RoPE
  trap in advance and still shipped the bug, because the check that would have caught it lived
  in a shell command rather than a test.

## The decode gate that cannot fail the way decode breaks

Symptom. A model passes `models/tt_transformers/tests/test_model.py` at PCC 0.994-0.9998
over nine decode steps, then produces repetitive garbage the moment it is served:

```
CPU  : " girl named Lily. She loved to play with her toys and make them look nice..."
TT   : " girl named Lily. Lily. Lily. Lily. She loved to a time, there was a little..."
```

Why the gate passes. The test's decode loop is *teacher-forced*. From the source
(`test_model.py:393-401`), when a reference model is running:

```python
_, pt_out_tok = sample_host(ref_output, temperature=0, top_p=0.8)
pt_decode_input = embd(pt_out_tok)
# Use the same token for TT model (teacher forcing)
tt_decode_input = pt_decode_input
```

The reference model picks each token and **both** models are fed it. The TT model never
consumes its own output, so per-step error cannot accumulate. Nine steps of teacher-forced
decode is nine independent one-step checks, not a nine-step generation.

The structure makes this inescapable rather than incidental: the `else` branch *does* feed
`embd(tt_out_tok)` -- genuine free-running decode -- but only when no reference model is
running, and in that case **no PCC is computed**. The harness offers
compared-but-teacher-forced, or free-running-but-unvalidated. Never both.

What it costs. Serving is free-running by definition. A model can be certified at
PCC 0.999 and diverge from the reference within three tokens of real generation. Measured
on tt-tnt:

| Path | Result |
|---|---|
| Teacher-forced decode, 9 steps (the gate) | PCC 0.9940-0.9998 |
| Teacher-forced prefill, 25 steps | 92% top-1 agreement; every disagreement on a near-tie |
| Free-running decode (serving) | diverges after 2-4 tokens |

Why small models suffer most. Compounding only matters when a step is close enough to
flip. At the divergence point the top two logits were `' She'` 12.375 (p=0.574) and
`' Lily'` 11.938 (p=0.370) -- a 0.44-logit margin, about nine bf16 ulps. A 22M model
trained on 0.43 of an epoch has a flat next-token distribution and therefore near-ties
everywhere. The same absolute error that is invisible on an 8B model flips tokens here.

What to do instead. Do not accept a decode PCC as evidence that generation works. Add a
free-running check: generate N tokens on device, generate N on the CPU reference from the
same prompt, and compare the token sequences directly. It costs one extra run and it is the
only thing that measures the failure mode serving actually has. `$qualitative-check` in
tt-metal's `.agents` skills exists for exactly this, and judges repetition explicitly.

Do not stop at "it is just precision." That may be the answer -- but establish it by
measuring the free-running gap at two precisions, not by assuming. On this model, moving
from the stock `performance` profile (BFLOAT4_B MLP) to `accuracy` (BFLOAT16 attention,
BFLOAT8_B MLP) changed the greedy output not at all, which rules precision out as the
*sole* explanation.
