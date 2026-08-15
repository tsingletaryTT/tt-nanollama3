<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# LR decay tail — a controlled negative result

`tt-tnt-v4` repeats `tt-tnt-v3` exactly — same corpus, tokens, seq_len 2048, batch 16,
10,764 steps — changing one thing: a cosine LR schedule, 3e-4 to 3e-5, **held flat for the
first 50%** and decaying from ~step 5382.

The held phase is the point. It is a *control*: for 5,382 steps v4 and v3 are the same
configuration, so any difference there is run-to-run variance. Every previous comparison in
this project lacked one, and twice that led to a real-looking result that was an artifact.

## Result

| phase | n | mean delta vs v3 | sem | 95% CI |
|---|---:|---:|---:|---|
| held (control) | 10 | -0.0078 | 0.0087 | [-0.0249, +0.0093] |
| decay | 12 | -0.0200 | 0.0037 | [-0.0271, -0.0128] |

**decay minus held = -0.0122 +/- 0.0095, |t| = 1.29 — not distinguishable
from run-to-run noise.**

Read without the control, the decay row looks like a finding: its CI excludes zero. But v4
was already running slightly below v3 *before the decay began*, and the difference between
the phases does not clear the noise. The prior was a 0.1-0.3 nat gain; the observed effect is
-0.0200, an order of magnitude smaller and unattributable.

Final full-pass validation loss, like for like: v2 3.2938, v3 2.9937, **v4 2.9766**.

## The schedule really ran

Traced in `artifacts/checkpoints-tt-tnt-v4/train.log`: `3.000e-04` held, then `2.969e-04`,
`2.274e-04`, `1.130e-04`, `3.603e-05`, `3.040e-05`. The intervention was applied and did
nothing measurable — which is the useful form of a negative result, as distinct from an
intervention that silently failed to apply.

## What this closes

Three fixes were proposed for the model's qualitative failure. All three have now been
tested: sampling (loops were partly a decoding artifact, no voice change), the stratified
split (fixed the measurement and the training mixture; model quality unchanged), and this.
None moved the model. The change that did was a regression none of them anticipated -- a
corpus containing zero document boundaries.
