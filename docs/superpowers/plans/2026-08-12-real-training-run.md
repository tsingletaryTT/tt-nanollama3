<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Real Training Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train NanoLlama3 with all of its layers actually learning, for multiple full epochs, and measure whether the result is meaningfully better than the 0.43-epoch model.

**Architecture:** Three tasks in strict order — unfreeze the normalization layers and *prove* they move, add periodic validation to the chunked training loop, then run multi-epoch training and compare against the existing baseline. Dataset blending is deliberately **not** in this plan.

**Tech Stack:** `ttml` (tt-train), `ttnn`, Tenstorrent Blackhole, numpy, pytest

## Why this order, and why no new data yet

The current model has two separate problems, and mixing their fixes would make both unmeasurable.

**Thirteen of its layers never learned.** Every RMSNorm gamma is exactly 1.0 after 3000 steps. The gradients existed (Adam `exp_avg` ≈ 3.6e-4) but bf16 parameters at 1.0 have a ulp of 0.0039, and `stochastic_rounding` was off — so every update rounded back and was discarded. **We do not know what this architecture can do, because we have never trained all of it.**

**It has seen 43% of its data once.** 49,152,000 tokens against a 114,872,301-token training split.

Both are cheap to fix and neither needs new data. Adding a corpus now would confound three changes at once and leave us unable to say which helped. **Dataset blending is Plan 8**, after this establishes a real baseline.

### The epoch arithmetic

At `batch_size=64`, `seq_len=256`, one step consumes **16,384 tokens**. Confirmed against the existing run: 3000 × 16,384 = 49,152,000, matching the recorded `tokens_seen` exactly.

| Target | Steps | Est. wall clock at 0.134 s/step |
|---|---|---|
| 1 epoch | 7,011 | ~16 min |
| **3 epochs** | **21,033** | **~47 min** |
| 4 epochs | 28,044 | ~63 min |

Under an hour of compute for the whole thing.

## A consequence worth anticipating

Once the gammas are real, **the norm-mapping blind spot closes by itself.** Today, swapping two RMSNorm tensors changes loss by exactly 0.0000 and the parity gate by 5.86e-6 — invisible to both — because all 13 gammas are identical. With distinct learned values, both instruments start seeing that class of error.

`tests/test_to_hf.py`'s synthetic non-unit-gamma test already covers the mapping and will keep working. But after this run, **re-measure the norm-swap ablation** — it should stop being invisible, and that is a real improvement in the verification story worth recording.

## Global Constraints

- SPDX header pair on new files; Python 3.10+.
- **No bare `assert` for guards** in production code.
- `pyproject.toml` must NOT be modified.
- **The existing checkpoints are evidence and must not be destroyed.** `artifacts/checkpoints/` holds the 0.43-epoch run; it is the baseline this plan measures against. Write new checkpoints to `artifacts/checkpoints-v2/`. **Never delete or overwrite the originals.**
- **`artifacts/hf/` must not be regenerated in this plan.** It corresponds to the baseline checkpoint and is currently uploaded to a private Hub repo. Converting the new model is a later decision.
- **If a step produces a number that decides pass/fail, it must be a test**, not a shell command.
- The hardware is shared. Check it is free before long runs, and report if it is not.

---

## Task 1: Unfreeze the normalization layers and prove they move

**Files:**
- Create: `train/configs/nanollama3_bpe_v2.yaml` (or the repo's equivalent location for a training config)
- Test: `tests/test_training_config.py`

**Interfaces:** produces a training config with the fix, plus a check that reads a checkpoint and asserts the gammas are no longer degenerate.

**The fix, in preference order:**

1. **`stochastic_rounding: true`** in the optimizer block. Minimal blast radius — parameters stay bf16, the checkpoint format is unchanged, and nothing downstream (reader, converter, parity gate) is affected. Two shipped tt-train configs already use it: `training_shakespeare_tiny_deepseek_char.yaml`, `training_shakespeare_tinyllama_muon.yaml`.
2. **`type: AdamWFullPrecision`** (registered at `optimizers/optimizer_registry.cpp:66`) keeps fp32 master weights. More principled, but it changes what the optimizer stores — **verify whether master weights land in the checkpoint's optimizer group**, because that would affect `convert/checkpoint_reader.py`'s manifest walk and possibly the converter.

**Start with option 1.** Escalate to option 2 only if gammas still do not move, and if you do, check the checkpoint's optimizer group before assuming the rest of the toolchain is unaffected.

- [ ] **Step 1: Write the test that would have caught the original bug**

```python
def test_checkpoint_gammas_are_not_degenerate(checkpoint_path):
    """13 RMSNorm gammas all exactly 1.0 means the norm layers never learned.

    This is what the original 3000-step run produced: real gradients
    (exp_avg ~3.6e-4) discarded every step because bf16 at 1.0 has ulp 0.0039
    and stochastic_rounding was off. The loss curve looked perfectly healthy
    throughout, so only the parameter statistics reveal it.
    """
    gammas = [t for name, t in read_tensors(checkpoint_path) if name.endswith("/gamma")]
    assert len(gammas) == 13
    for name, g in gammas:
        assert float(g.std()) > 0.0, f"{name} has sd 0.0 — this layer did not learn"
```

Guard it with `skipif` on the checkpoint existing, and make it take a path so it can be pointed at either run.

- [ ] **Step 2: Confirm it fails on the baseline checkpoint** — `artifacts/checkpoints/nanollama3_step00003000.pkl` should fail this test. That is the RED evidence, and it proves the test is not vacuous.

- [ ] **Step 3: Write the v2 config** — a copy of the nanollama3 BPE training config with `stochastic_rounding: true` added, `max_steps` set for the target epoch count, and `model_save_interval` set so checkpoints land at sensible intervals (see Task 3 for storage arithmetic).

- [ ] **Step 4: Prove the fix works with a *short* run before committing an hour**

Run **200 steps** with the new config into a scratch directory, then assert the gammas have moved. Two hundred steps is ~30 seconds and answers the only question that matters before a long run.

If they still do not move, **stop and report** — escalating to `AdamWFullPrecision` is a decision worth surfacing, not making silently.

- [ ] **Step 5: Commit**

---

## Task 2: Periodic validation in the chunked loop

**Files:**
- Modify: `train/run.py`
- Test: `tests/test_run_validation.py`

**Interfaces:** `--val-every` CLI flag; a validation loss recorded at each checkpoint boundary.

A 21,000-step run needs a loss curve, not a single number at the end. We already call `train()` in chunks for checkpointing — evaluating between chunks is nearly free and gives the curve.

ttml's own `val_losses` is a documented placeholder that copies the training loss, so this must use our `evaluate()`.

- [ ] **Step 1: Write the failing test** — that a run with `--val-every` records one validation entry per boundary, and that the recorded values are **not** equal to the corresponding training losses (which would mean ttml's placeholder leaked in).

- [ ] **Step 2: Implement** — call `evaluate()` at each chunk boundary, print `step / train_loss / val_loss`, and append to a JSON-lines file under the checkpoint directory so the curve survives the process.

- [ ] **Step 3: Verify on a short run** (200 steps, `--val-every 100`) and report the recorded curve.

- [ ] **Step 4: Commit**

---

## Task 3: The multi-epoch run

**Files:** modify `CLAUDE.md`

**Interfaces:** produces `artifacts/checkpoints-v2/` and a measured comparison.

- [ ] **Step 1: Check the hardware is free and estimate storage**

Checkpoints are ~132 MB each at the current format. At `--save-every 2000` over 21,033 steps that is 10 checkpoints ≈ 1.3 GB. Confirm free disk before starting (the volume was at 96% with ~140 GB free).

If `AdamWFullPrecision` was used, checkpoints may be larger — measure the first one rather than assuming.

- [ ] **Step 2: Run**

```bash
python train/run.py --steps 21033 --save-every 2000 --val-every 1000 \
  --batch-size 64 --checkpoint-dir artifacts/checkpoints-v2
```

Expect ~47 minutes. Use a long timeout; background and poll rather than assuming failure. **Do not start a second run concurrently.**

- [ ] **Step 3: Report the curve honestly**

First and final train loss, the full validation curve, s/step, wall clock, and **whether validation was still improving at the end or had flattened**. If it plateaued at step 8000, say so — that is a finding about how much data this architecture can use, and it is more useful than a single final number.

If validation loss starts *rising* while training loss falls, that is overfitting and the useful checkpoint is the one before the turn. Report where.

- [ ] **Step 4: Compare against the baseline, fairly**

Evaluate the new final checkpoint and the old `nanollama3_step00003000.pkl` **on the same held-out windows with the same seed** — a paired comparison, not two independent samples. The baseline's held-out loss was **1.8781** measured over 10 sampled batches with sd ≈ 0.315, so an unpaired comparison of two draws could differ by 0.3 nats on noise alone.

Report the paired difference and whether it exceeds the noise floor.

- [ ] **Step 5: Re-measure the norm-swap ablation**

With real gammas, swapping two RMSNorm tensors should no longer be invisible. Measure it and record the number — it was exactly 0.0000 before. This closes the blind spot the verification work has been carrying since Plan 4.

- [ ] **Step 6: Generate samples and read them**

Same prompts as before (`"Once upon a time, there was a little"`), reported verbatim, alongside the baseline's output for the same prompt. Is it visibly better, or only numerically better? **Both answers are informative** — and a 1-nat improvement that produces indistinguishable prose is worth knowing about.

- [ ] **Step 7: Record in CLAUDE.md and commit**

---

## Self-Review

**Every capability named here resolves to a test or a reported measurement.** Task 1's fix → a gamma-degeneracy test with RED evidence on the baseline. Task 2's validation → a test that it is not ttml's placeholder. Task 3's comparison → a paired evaluation with a stated noise floor.

**Does the artifact satisfy the rationale?** The stated goal is a model whose layers all learned, trained on more than half its data, *measurably* better than the baseline. Step 4's paired comparison is what makes "better" a measurement rather than an impression; Step 6 keeps it honest about whether the improvement is visible.

**No load-bearing number lives in a bash block.**

## Known risks

- **The fix may not work.** If gammas still do not move after 200 steps with `stochastic_rounding: true`, that is a finding — escalate to `AdamWFullPrecision`, and check its effect on the checkpoint's optimizer group before assuming the converter is unaffected.
- **More epochs may not help much.** TinyStories is simple and a 22M model may saturate it. If validation flattens at 1.5 epochs, the honest conclusion is that this architecture has learned what this corpus can teach it — which is exactly the finding that would justify Plan 8's dataset blend.
- **Overfitting is plausible** at 3–4 epochs on a small corpus. Periodic validation exists to catch it; the useful checkpoint may not be the last one.
- **The baseline must survive.** `artifacts/checkpoints/` and `artifacts/hf/` correspond to the uploaded private repo. Nothing in this plan may overwrite them.
