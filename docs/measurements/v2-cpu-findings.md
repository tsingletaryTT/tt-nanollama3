<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# v2 checkpoint: CPU-side findings

The 21,034-step run (`artifacts/checkpoints-v2/nanollama3_step00021034.pkl`, held-out loss
**1.4602** against the baseline's **1.8781**) converted and verified without hardware.

## Conversion

Converted to `artifacts/384/hf` — the per-size write path, leaving the published baseline at
the legacy flat `artifacts/hf` untouched.

- `generation_config.json` present and `model.safetensors` **44 MB, not 68.6 MB**: the Plan 6
  artifact repairs (drop the duplicate tied `lm_head`, emit a generation config) carried
  through to the new conversion rather than having to be redone.
- Gammas: baseline all exactly `0.0`; v2 spans `4.7e-2 – 6.0e-2`. The retrain did what it
  was for.

## Parity gate

NumPy reimplementation of ttml's forward, run on the raw v2 checkpoint, against the
converted HF model over a 64-token seeded window:

```
max |diff|   8.947e-06
relative     7.332e-07      (baseline was ~1.23e-5)
top-1 agree  100.0% of 64 positions
```

**The v2 conversion is numerically faithful**, and tighter than the baseline's.

## A prediction, and its falsification

The surviving explanation for the serving decode defect is: small per-step numerical error,
compounded by free-running decode, flipping tokens wherever the next-token distribution has
a near-tie. That predicts a better-trained model — sharper distributions, fewer near-ties —
should hold together longer on device.

Measured on identical text, top-1 minus top-2 logit margin per position:

| | baseline | v2 |
|---|---|---|
| median margin | 1.361 | **1.000** |
| mean margin | 1.718 | 1.775 |
| near-ties (< 0.5) | 21.0% | **32.3%** |
| ties (< 1.0) | 37.1% | 50.0% |
| ties (< 2.0) | 71.0% | 66.1% |

**The prediction is wrong.** v2 has *more* very-near ties, and a *lower* median margin
(0.73× the baseline's). Its mean margin is slightly higher and it has fewer moderate ties, so
the distribution changed shape rather than simply flattening — but on the metric that matters
for token flipping, v2 is the more fragile model, not the less.

That is explicable rather than alarming: the undertrained baseline had learned blunt,
overconfident priors — it commits hard to repetitive continuations, which is exactly the
degenerate behaviour observed. v2 is better calibrated, so where a continuation is genuinely
ambiguous it correctly spreads probability. **Lower loss means better probability assignment;
it does not mean wider top-1/top-2 gaps.**

## What this means for the on-device test

The near-tie hypothesis now predicts v2 will show **no improvement, possibly a regression**,
in free-running agreement against the baseline's median of 4/40 tokens
(`scripts/free_running_check.py`). Recording that here *before* the run, so the result is
informative in either direction:

- v2 agreement materially **better** than 4/40 → near-ties are not the mechanism, and
  something else in decode is wrong after all.
- v2 agreement **similar or worse** → consistent with numerical error flipping near-ties,
  and the remedy is precision or a sharper model, not a decode bug hunt.
