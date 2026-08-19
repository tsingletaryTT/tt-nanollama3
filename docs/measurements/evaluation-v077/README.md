<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Does the beta2 0.999 model behave any better? No.

`artifacts/hf-tt-tnt-1024-v077` (the paired control run's final checkpoint, the
one that won the training-curve comparison) against the designated
`tt-tnt-1024-dialogue`, via `scripts/evaluate.py`, matched window, matched token
array, matched prompt set, matched seed.

## Result

| | value |
|---|---|
| mean loss over `[0, 512)` | 2.7726 → **2.7322**, delta **−0.0404** |
| every behavioural signal | **NOT INTERPRETABLE** or **below paired detection** |

The loss delta of −0.040 tracks the training-curve delta of −0.048 closely, which
is a useful consistency check between two independent instruments. It carries no
seed floor of its own and is reported without a ratio.

Everything that *does* have a floor came in at 0.10–0.40× of it: genre collapse,
story-frame collapse, lexical-habit collapse, termination rate, tinystories
margin, nearest-source. Three more — 4-gram repeat, longest repeated span, prompt
engagement — cleared the floor ratio but failed the paired minimum-detectable
difference, which is the mirror-image gate and equally disqualifying.

## What this settles

The prediction made before running it was that 0.048 nats against a 0.1944-nat
seed floor would be invisible downstream. It was. That is now measured rather
than argued.

`beta2 0.999` is a real training-curve improvement and not a better model in any
respect this suite can detect. Both statements are true and neither is the whole
answer; the config change is worth keeping because it is free, not because it
buys capability.

## Designation

Unchanged. `tt-tnt-1024-dialogue` remains current. `docs/current_model.json`
moves on new evidence, and "indistinguishable from the incumbent" is not evidence
for replacing it.

## One gap

The loss-trajectory section did not compute: `evaluate.py` looks for a checkpoint
directory named after the label, and this run's checkpoints live in
`artifacts/checkpoints-v077-beta2-control`. The comparison is unaffected — that
trajectory is the same one already analysed in
`beta2-0999-vs-095-paired.json` — but the naming coupling is a small trap worth
knowing about.
