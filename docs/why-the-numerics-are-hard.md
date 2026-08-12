<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Why the numerics are hard, even building Tenstorrent-first

This project was built for Tenstorrent hardware from the first line — no port, no
translated CUDA kernel, tile-aware dimensions chosen deliberately. It still spent a
substantial fraction of its effort on things that looked like arithmetic problems. This
file is the retrospective on why, written from the defects actually hit rather than from
general principles.

## First: most of it was not math

Running the list honestly, the majority of what went wrong was not arithmetic at all:

| Defect | Actual category |
|---|---|
| RoPE interleaved vs split-halves | **Convention** — two correct implementations, incompatible memory layouts |
| `find_grid` hardcoding `max_cols = 12` | **Resource assumption** — the part is harvested to 11 columns |
| `HF_MODEL` required but never documented | **Contract** — two components disagreeing about who supplies the model id |
| Nine "decode" steps that were teacher-forced | **Harness** — a gate that cannot fail the way the system fails |
| `mesh.devices` vs `env.MESH_DEVICE` unchecked | **Contract** — two independent channels describing one thing |

Building TT-first removes a real class of bugs: there is no foreign kernel to mistranslate.
It does **not** remove seams. This stack crosses ttml (C++, training) → `tt_transformers`
(Python, inference) → the vLLM plugin (serving), and every crossing is a place where two
individually-correct components can disagree about what they mean. Being native to the
hardware does not make you native to every layer above it.

## Where it really is math: continuous measurement, discrete output

This is the one that cost the most and generalises the furthest.

Every gate in the stack reports a **continuous** quantity — PCC, relative error, nats of
loss. The output is `argmax`, a **step function**. At a near-tie, the map from logits to
tokens is a *discontinuity*.

The concrete instance: at the step where this model's served output diverged from its CPU
reference, the top two logits were

```
' She'   12.3750   p = 0.574
' Lily'  11.9375   p = 0.370      margin 0.4375 logits ≈ 9 bf16 ulps
```

Continuous error analysis calls a 9-ulp perturbation a rounding artefact. Token space calls
it a different story. **PCC 0.9940–0.9998 and "the output is wrong" are both true at once**,
and no amount of agreement in the smooth quantity protects you across the discontinuity.

Every instrument we had — the parity gate, the PCC test, held-out loss — measures the
continuous side. That is not an accident of this project; it is what is easy to measure.

## Two properties that make TT arithmetic harder to reason about

**Block float is not iid noise.** `bfp8` and `bfp4` share a single exponent across a tile of
32 values. One large activation degrades the precision of its 31 neighbours. Error is
therefore *data-dependent and correlated between adjacent elements* — precisely the
assumption that standard error-propagation arguments violate. "Raise the precision and the
error shrinks proportionally" is a hypothesis, not a theorem. Tested here: moving attention
to `BFLOAT16` and the MLP off `BFLOAT4_B` bought about **one token** of free-running
agreement (median 3 → 4 of 40).

**Autoregressive decode is a feedback loop.** Prefill is one forward pass and its error is
bounded. Decode feeds output back as input, so the model's own dynamics amplify whatever was
wrong. This is exactly why teacher forcing hides the failure — it breaks the loop — and why
nine teacher-forced decode steps said nothing useful about four real ones.

## And the uncomfortable part: this model is the worst case

Compounding only matters when a step is close enough to flip. Measured on identical text:

| | baseline (0.43 epoch) | v2 (3 epochs) |
|---|---|---|
| median top1−top2 margin | 1.361 | 1.000 |
| positions within 0.5 logits | 21.0% | 32.3% |

A 22M-parameter model trained on a fraction of an epoch has a flat next-token distribution:
roughly a fifth to a third of positions are near-ties. A production 8B model has sharp
distributions where the same absolute numerical error flips nothing.

**The hardware is not less accurate for us than for Llama-70B. Our model is standing on a
knife edge where that accuracy becomes visible.**

Note the direction of the second column: better training made the distribution *flatter*, not
sharper. Lower loss means better *probability assignment*; it does not mean wider top-1/top-2
gaps. A well-calibrated model spreads probability where a continuation is genuinely
ambiguous. The undertrained baseline was blunt and overconfident — which is precisely why it
committed so hard to repetitive continuations.

## What to take from this

1. **A continuous gate cannot certify a discrete output.** If the deliverable is tokens,
   measure tokens. `scripts/free_running_check.py` exists for this reason.
2. **Break the feedback loop only when you mean to.** Teacher forcing is the right tool for
   isolating a single step and the wrong tool for certifying generation.
3. **Treat "more precision" as a hypothesis.** Block float's correlated error means the
   improvement is not proportional and may be small. A/B it.
4. **Know where your model sits on the sharpness distribution.** The same stack that serves
   an 8B model flawlessly can be visibly wrong on a 22M one, with no defect anywhere.
5. **Being native to the hardware is not being native to the stack.** Most defects live at
   seams between correct components, not inside them.
