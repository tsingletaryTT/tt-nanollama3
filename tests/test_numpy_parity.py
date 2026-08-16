# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The parity gate: compare the NumPy ttml forward pass against the converted HF model.

Two independently-derived paths reach logits from the same checkpoint:

- **NumPy path** (`convert.ttml_forward.forward`) -- raw checkpoint tensors, ttml's own
  conventions (interleaved RoPE, fused KV, ttml's norm order), never touching `convert/`'s
  own interpretation.
- **HF path** (`convert.to_hf.convert_checkpoint` + `transformers`) -- raw checkpoint
  tensors through our converter's split KV, permuted RoPE rows, HF layout.

Agreement between them is real evidence that the converter's interpretation matches ttml's
actual behaviour; it is not circular, because Task 2's `ttml_forward.py` was derived from
`docs/ttml-forward-reference.md` alone (see that module's docstring), never from
`convert/hf_mapping.py` or `convert/to_hf.py`.

**On the tolerance below (see "On the achievable tolerance" in
`docs/superpowers/plans/2026-08-12-numpy-parity.md`).** The plan deliberately did not fix a
number in advance -- an unreachable gate is indistinguishable from a correct one that
happens to fail, which is the worst property a diagnostic can have. Measured directly on
this checkpoint (`artifacts/hf/`, seven 256-token windows -- the model's full
`max_position_embeddings` -- at seeds 0-5 and this file's own committed seed 123; see
`test_numpy_fp32_vs_hf_fp32_agreement_on_fixed_window`'s docstring for the per-seed numbers):

- max absolute logit difference: ~8.3e-6 to ~1.6e-5 (float32 rounding noise, not a real defect)
- max relative difference, restricted to logits with `|value| > 0.01` (unrestricted relative
  error is dominated by meaningless blowups near zero-crossings -- see that test's docstring
  for a worked example): ~3.0e-4 to ~5.6e-4
- correlation: indistinguishable from 1.0 (1 - corr ~= 1.2e-13)

This is dramatically tighter than the loss gate's 0.2-nat tolerance (whose 2-sigma floor is
~0.22 nats) and tighter than the plan's own ballpark suggestion of ~1e-2 relative. The
tolerances below (`MAX_ABS_TOLERANCE = 1e-3`, `MAX_REL_TOLERANCE = 5e-3`,
`MIN_CORRELATION = 0.9999`) sit roughly one to two orders of magnitude above every measured
value -- enough margin that a different fixed window or a harmless numerical-library version
bump will not make this gate flaky, while remaining many orders of magnitude below what an
actual conversion bug produces (`test_parity_gate_is_not_hollow_it_fails_when_rope_permutation_is_disabled`
measures a real bug at max_abs ~8.1, correlation ~0.93).

**What this gate cannot see** (see also CLAUDE.md's "NumPy parity gate" section):

1. **The real checkpoint's RMSNorm gammas are all exactly 1.0** (verified: all 13, min==max==
   1.0 -- see `docs/superpowers/specs/2026-08-11-followups.md`). Swapping two norms'
   destinations in the mapping is therefore invisible to *this* gate on *this* checkpoint,
   exactly as it is to the loss gate -- see
   `test_parity_gate_is_blind_to_a_norm_swap_on_the_real_checkpoint`, which measures this
   directly rather than assuming it. `test_convert_checkpoint_places_each_rmsnorm_gamma_at_its_correct_destination`
   closes this gap with a synthetic checkpoint that has distinct gammas, independent of
   whether the real checkpoint's ever stop being degenerate.
2. **What neither path implements is invisible to both.** If both the NumPy reference and
   the converter drop RoPE scaling (the header records no `scaling_factor`; see plan-5's
   "Known risks"), they agree with each other and are both wrong relative to ttml's actual
   runtime behaviour. This harness validates the *conversion* (does `convert/` correctly
   translate ttml's checkpoint into HF's format), not the *checkpoint's completeness*.
3. **ttnn's own accumulation/output dtype on the actual device is untraced** (plan-5's
   "Untraced" note). Both paths here run entirely on the host in float32/bfloat16-cast-to-
   float32; neither one touches a Tenstorrent device. Agreement here says nothing about
   NumPy-vs-device agreement, which would need a different, hardware-dependent, tolerance
   (Task 1's bf16-RMSNorm-divisor finding bounds that comparison, not this one).
4. **This gate compares two *implementations* of the same architecture, not the checkpoint
   against ground truth.** If ttml's own forward pass has a bug relative to what the
   training run's own loss curve implies, nothing here would catch it -- Task 2 Step 3's
   independent cross-entropy check (`tests/test_ttml_forward.py`) is what anchors the NumPy
   path to an external reference (the training run's own held-out loss), not this file.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
from pathlib import Path
from typing import Dict
from unittest import mock

import ml_dtypes
import numpy as np
import pytest

import convert.hf_mapping as hf_mapping
import convert.to_hf as to_hf
import convert.ttml_forward as ttml_forward
from convert.checkpoint_reader import read_checkpoint_meta, read_tensors
from convert.to_hf import convert_checkpoint
from convert.ttml_forward import attention, forward, rms_norm, squeeze_leading, swiglu

ROOT = Path(__file__).resolve().parent.parent


def _artifact(env_var: str, default: Path) -> Path:
    """``default``, unless ``env_var`` names somewhere else.

    THE DEFAULTS ARE THE COMMITTED GATE and do not change: with none of these variables set,
    this module reads exactly the four artifacts it always read, so the suite's meaning is
    untouched. The overrides exist because this gate is the only instrument that compares
    ``convert/``'s interpretation of a checkpoint against an independently-derived NumPy
    forward pass, and until 2026-08-16 it could only ever be aimed at one checkpoint — the
    protected 384 baseline. ``.superpowers/ddp-bringup.md`` recorded "run the parity gate
    against a DDP-produced checkpoint" as outstanding *specifically* because no valid DDP
    checkpoint existed to aim it at; now one can, and hardcoded paths were the only thing
    still in the way. Point all four at a matched set:

        TT_TNT_PARITY_CHECKPOINT_DIR=/scratch/ckpt \\
        TT_TNT_PARITY_HF_DIR=/scratch/hf \\
        TT_TNT_PARITY_VAL_IDS=artifacts/tokens-v3/val_ids.npy \\
        python -m pytest tests/test_numpy_parity.py

    They must be *matched*: the HF directory has to be the conversion of that same
    checkpoint, and the val-ids file the corpus it was trained on, or the gate measures a
    mismatch rather than the converter.

    **Two tests here are calibrated to the committed baseline and are expected to fail under
    an override.** Both are assertions about *that checkpoint's* properties rather than about
    the converter, so a failure from them is information, not a regression:

    * ``test_parity_gate_is_not_hollow_it_fails_when_rope_permutation_is_disabled`` — its
      three tolerance assertions hold on any checkpoint (measured on a 50-step 1024 DDP
      checkpoint: max_abs 2.93, i.e. ~2900x over budget, max_rel 2.0, corr 0.9988 against a
      0.9999 floor, so the gate is demonstrably not hollow there either). What does not
      transfer is the extra sharpness assertion ``corr < 0.99``, whose 0.934 came from the
      3000-step 384 model: a barely-trained model's logits are less spread out, so the same
      layout bug moves the correlation less while moving the absolute logits just as far.
    * ``test_parity_gate_is_blind_to_a_norm_swap_on_the_real_checkpoint`` — asserts the
      baseline's blind spot, that its RMSNorm gammas are all exactly 1.0 so swapping two
      norms' destinations is a byte-identical no-op. Any checkpoint trained with
      ``stochastic_rounding: true`` has real gammas (measured 0.977-1.031 after 50 steps), the
      swap stops being a no-op, and this test fails **because the blind spot is gone** — which
      is the outcome to want. Its failure message says so directly.
    """
    override = os.environ.get(env_var)
    return Path(override) if override else default


REAL_CHECKPOINT_DIR = _artifact("TT_TNT_PARITY_CHECKPOINT_DIR", ROOT / "artifacts" / "checkpoints")
TOKENIZER_DIR = _artifact("TT_TNT_PARITY_TOKENIZER_DIR", ROOT / "artifacts" / "tokenizer")
HF_DIR = _artifact("TT_TNT_PARITY_HF_DIR", ROOT / "artifacts" / "hf")
VAL_IDS_PATH = _artifact("TT_TNT_PARITY_VAL_IDS", ROOT / "artifacts" / "tokens" / "val_ids.npy")

#: Both naming schemes: ``nanollama3_step*.pkl`` predates the tt-nanollama3 -> tt-tnt rename
#: and is what the committed 384 baseline in ``artifacts/checkpoints/`` uses; ``tt_tnt_step*``
#: is what everything written since uses, and so what an overridden directory will hold.
#: Sorted by the numeric step after "step" rather than by filename, so the two schemes
#: interleave by step instead of one prefix always sorting after the other — same rule as
#: ``train.checkpoint.latest_checkpoint``.
_CHECKPOINTS = (
    sorted(
        list(REAL_CHECKPOINT_DIR.glob("nanollama3_step*.pkl"))
        + list(REAL_CHECKPOINT_DIR.glob("tt_tnt_step*.pkl")),
        key=lambda p: int(p.stem.rsplit("step", 1)[-1]),
    )
    if REAL_CHECKPOINT_DIR.exists()
    else []
)
REAL_CHECKPOINT = _CHECKPOINTS[-1] if _CHECKPOINTS else None

#: Every test that needs the real trained checkpoint, its tokenizer, the converted
#: `artifacts/hf/` model, and/or `val_ids.npy` is individually decorated with this -- NOT a
#: module-wide `pytestmark`, because Step 3's synthetic gamma-mapping test
#: (`test_convert_checkpoint_places_each_rmsnorm_gamma_at_its_correct_destination`) needs
#: none of them and is deliberately left able to run unconditionally. Task 2's review noted
#: that every decisive test being `skipif`-guarded on gitignored `artifacts/` means a CI run
#: can report "all passed" while the load-bearing tests never executed; leaving the one test
#: here that has no such dependency ungated is a direct, if partial, answer to that finding.
_ARTIFACTS_PRESENT = (
    REAL_CHECKPOINT is not None
    and TOKENIZER_DIR.is_dir()
    and (HF_DIR / "config.json").is_file()
    and VAL_IDS_PATH.is_file()
)
_needs_artifacts = pytest.mark.skipif(
    not _ARTIFACTS_PRESENT,
    reason="needs a real checkpoint, tokenizer, converted artifacts/hf/, and val_ids.npy",
)

# ---------------------------------------------------------------------------
# Tolerances -- see this module's docstring for the measurements that justify these numbers.
# ---------------------------------------------------------------------------

#: Absolute margin: measured worst case across seven 256-token windows/seeds was ~1.6e-5;
#: this sits ~60x above that, so ordinary float32 rounding noise (a different BLAS build, a
#: different window) cannot trip it, while a real conversion bug (measured below at ~8.1)
#: trips it by more than five orders of magnitude.
MAX_ABS_TOLERANCE = 1e-3
#: Relative margin, evaluated only on logits with |value| > _REL_FLOOR (see _compare_logits'
#: docstring for why unrestricted relative error is meaningless near zero-crossings).
#: Measured worst case ~5.6e-4; this sits ~9x above that. The plan itself suggested ~1e-2 as
#: a "something is wrong" ceiling; this gate is deliberately tighter than that ceiling by 2x,
#: while still comfortably above measurement noise.
MAX_REL_TOLERANCE = 5e-3
#: Logits below this magnitude are excluded from the relative-difference calculation: two
#: logits of -1.05e-5 and -1.15e-5 differ by "8%" in relative terms while being numerically
#: indistinguishable in every way that matters (softmax over ~13-magnitude logits does not
#: notice a 1e-5 wobble at the bottom of the range).
_REL_FLOOR = 0.01
#: Measured correlation on real data is 1 minus something on the order of 1.2e-13; a real bug
#: (measured below) drops it to ~0.93. 0.9999 sits far below the former and far above the
#: latter.
MIN_CORRELATION = 0.9999

#: Fixed window: a seeded 256-token slice of the validation set -- the model's full
#: `max_position_embeddings` (see `artifacts/hf/config.json`), not a truncated prefix of it.
#: A shorter window only exercises positions 0-63; the remaining 64-255 would then be
#: covered solely by the coarser cross-entropy check (`tests/test_ttml_forward.py`'s
#: `test_numpy_forward_reproduces_the_training_runs_held_out_cross_entropy`), whose detection
#: floor is ~0.22 nats (see docs/ttml-forward-reference.md's Q6) -- far too coarse to catch a
#: defect whose signature grows with position (e.g. a RoPE angle that drifts with sequence
#: length). Distinct from Task 2's own seed-0/256-length CE window by seed alone, so this
#: file's measurements are not just a rerun of that file's under a different name.
_WINDOW_SEED = 123
_WINDOW_LEN = 256


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _fixed_window() -> np.ndarray:
    val_ids = np.load(VAL_IDS_PATH)
    rng = np.random.default_rng(_WINDOW_SEED)
    start = int(rng.integers(0, len(val_ids) - _WINDOW_LEN))
    return val_ids[start : start + _WINDOW_LEN].astype(np.int64)


def _compare_logits(a: np.ndarray, b: np.ndarray, *, rel_floor: float = _REL_FLOOR) -> Dict[str, float]:
    """max abs diff, floored max relative diff, and Pearson correlation between two logit
    arrays of identical shape.

    Relative difference is computed only where `max(|a|, |b|) > rel_floor` -- without that
    floor, a pair like (-1.05e-5, -1.15e-5) reports a "relative difference" of 0.08 (8%)
    purely because both values are near zero, even though the absolute difference (1e-6) is
    negligible and well inside float32 noise. This is not a hypothetical: it is exactly what
    the real checkpoint's actual worst-case relative-difference entry looks like (see this
    module's docstring) -- the floor turns a meaningless outlier into an excluded point
    instead of a false "the gate is loose" reading.
    """
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    diff = np.abs(a64 - b64)
    denom = np.maximum(np.abs(a64), np.abs(b64))
    mask = denom > rel_floor
    max_rel = float((diff[mask] / denom[mask]).max()) if np.any(mask) else 0.0
    corr = float(np.corrcoef(a64.ravel(), b64.ravel())[0, 1])
    return {"max_abs": float(diff.max()), "max_rel": max_rel, "corr": corr}


def _hf_logits_fp32(hf_dir: Path, token_ids: np.ndarray) -> np.ndarray:
    """Logits from an HF model directory, loaded and run in float32.

    float32 on both sides is deliberate (see this module's docstring): it puts the NumPy
    path and the HF path on the same compute dtype from the same bf16-stored weights, so any
    disagreement reflects the *conversion*, not a dtype choice made when loading the model.
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(hf_dir), torch_dtype=torch.float32).eval()
    ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits = model(ids).logits[0]
    return logits.numpy()


def _round_bf16(x: np.ndarray) -> np.ndarray:
    """Round a float32 array down to bfloat16 precision and back up to float32."""
    return np.asarray(x, dtype=ml_dtypes.bfloat16).astype(np.float32)


def _forward_with_bf16_activations(checkpoint_path: Path, token_ids: np.ndarray) -> np.ndarray:
    """A NumPy-only control: the same computation as `convert.ttml_forward.forward`, but
    every activation tensor that crosses a sub-layer boundary (the embedding lookup, each
    norm's output, each sub-layer's output, each residual sum, the final norm) is rounded to
    bfloat16 precision and back before the next op consumes it.

    **What this is for.** Step 1 of the brief asks for a NumPy-vs-NumPy control -- float32
    throughout versus bf16-rounded activations -- to bound how much of any NumPy-vs-HF gap
    could be precision alone, before picking a tolerance for the real (NumPy-vs-HF) gate.

    **What this is not.** This is *not* a claim that it reproduces ttml's actual device
    numerics. `docs/ttml-forward-reference.md`'s Task 1 findings note that `ttnn_fixed::
    matmul`'s accumulation/output dtype is untraced, and that ttml's RMSNorm kernel packs its
    mean divisor as bfloat16 specifically (a narrower, non-obvious rounding point this
    function does not attempt to reproduce). This function rounds coarsely, at sub-layer
    granularity, as a cheap upper-bound-ish probe of "does bf16-level activation noise alone
    explain a gap this size" -- not as a device simulator.

    **Deliberately duplicated from `forward()`'s loop, not refactored into it.** Threading a
    "round activations" flag through the production module for a one-off measurement would
    add a permanent knob to code Task 2 already reviewed as clean, for a control this file
    alone needs. The loop body is ~15 lines; duplicating it here costs little.
    """
    checkpoint_path = Path(checkpoint_path)
    header, _manifest = read_checkpoint_meta(checkpoint_path)
    cfg = header["transformer_config"]
    num_heads = int(cfg["num_heads"])
    num_groups = int(cfg["num_groups"])
    num_blocks = int(cfg["num_blocks"])
    theta = float(cfg["theta"])

    weights: Dict[str, np.ndarray] = {}
    for name, array in read_tensors(checkpoint_path, group="model"):
        weights[name] = squeeze_leading(array).astype(np.float32)

    tokens = np.asarray(token_ids, dtype=np.int64)
    tok_emb = weights["llama/fc/weight"]
    x = _round_bf16(tok_emb[tokens])

    for block in range(num_blocks):
        p = f"llama/llama_block_{block}"

        h = _round_bf16(rms_norm(x, weights[f"{p}/attention_norm/gamma"]))
        a = _round_bf16(
            attention(
                h,
                weights[f"{p}/attention/q_linear/weight"],
                weights[f"{p}/attention/kv_linear/weight"],
                weights[f"{p}/attention/out_linear/weight"],
                num_heads=num_heads,
                num_groups=num_groups,
                theta=theta,
            )
        )
        x = _round_bf16(x + a)

        h = _round_bf16(rms_norm(x, weights[f"{p}/mlp_norm/gamma"]))
        m = _round_bf16(
            swiglu(
                h,
                weights[f"{p}/mlp/w1/weight"],
                weights[f"{p}/mlp/w2/weight"],
                weights[f"{p}/mlp/w3/weight"],
            )
        )
        x = _round_bf16(x + m)

    x = _round_bf16(rms_norm(x, weights["llama/ln_fc/gamma"]))
    logits = x @ tok_emb.T
    return logits


# ---------------------------------------------------------------------------
# Step 1a: NumPy-vs-NumPy control (bf16-rounded activations vs float32 throughout).
# ---------------------------------------------------------------------------


@_needs_artifacts
def test_numpy_fp32_vs_bf16_activation_control_bounds_precision_noise():
    """Step 1's NumPy-vs-NumPy control: how much does activation precision alone move the
    logits, independent of any NumPy-vs-HF question?

    Measured on the fixed window (this test's exact procedure, at the committed
    `_WINDOW_LEN = 256`, run standalone for full precision): max_abs = 0.0403, max_rel
    (floored at 0.01) = 1.814 (a handful of near-zero logits flip sign under bf16 rounding --
    expected and not concerning, which is exactly why the real gate's relative-difference
    check has a floor), correlation = 0.9999969.

    This bounds what bf16-level activation rounding alone contributes: a few hundredths in
    absolute terms, with the overall logit *shape* (correlation) essentially undisturbed.
    Compare against `test_numpy_fp32_vs_hf_fp32_agreement_on_fixed_window`'s measured ~1.2e-5
    absolute agreement (same 256-token window) -- three orders of magnitude tighter -- which
    confirms the fp32-vs-fp32 NumPy-vs-HF comparison is not being helped along by both sides
    sharing some bf16 rounding; if it were, this control's ~0.04 absolute noise floor would
    show up there too, and it does not.
    """
    window = _fixed_window()
    logits_fp32 = forward(REAL_CHECKPOINT, window)
    logits_bf16_act = _forward_with_bf16_activations(REAL_CHECKPOINT, window)
    stats = _compare_logits(logits_fp32, logits_bf16_act)

    assert np.all(np.isfinite(logits_bf16_act))
    # Sanity bounds only -- this is a diagnostic control, not the gate itself. The real
    # figures (~0.04 abs, corr ~0.9999969, measured above) sit comfortably inside these.
    assert stats["max_abs"] < 0.5, stats
    assert stats["corr"] > 0.999, stats


# ---------------------------------------------------------------------------
# Step 1b: the actual gate -- NumPy-vs-HF, both fp32, on the fixed window.
# ---------------------------------------------------------------------------


@_needs_artifacts
def test_numpy_fp32_vs_hf_fp32_agreement_on_fixed_window():
    """**The parity gate.** Compares `convert.ttml_forward.forward`'s logits against
    `artifacts/hf/`'s logits (loaded and run in float32) on a fixed 256-token window of
    `val_ids.npy` -- the model's full `max_position_embeddings`, not a truncated prefix (see
    `_WINDOW_LEN`'s comment for why a shorter window would leave positions 64-255 covered by
    nothing sharper than the 0.22-nat-floor CE check). See this module's docstring for the
    full tolerance derivation; in short:

    Measured directly (this exact procedure, at the committed `_WINDOW_LEN = 256`, seven
    different seeds/windows, run standalone):

    | seed | max_abs   | max_rel (floor 0.01) | correlation        |
    |------|-----------|-----------------------|--------------------|
    | 0    | 1.29e-5   | 4.81e-4               | 1 - 1.19e-13       |
    | 1    | 1.60e-5   | 5.56e-4               | 1 - 1.19e-13       |
    | 2    | 8.85e-6   | 3.54e-4               | 1 - 1.17e-13       |
    | 3    | 1.43e-5   | 3.04e-4               | 1 - 1.21e-13       |
    | 4    | 8.29e-6   | 4.80e-4               | 1 - 1.26e-13       |
    | 5    | 8.91e-6   | 3.73e-4               | 1 - 1.18e-13       |
    | 123 (**this test's actual seed**) | **1.23e-5** | **3.04e-4** | **1 - 1.28e-13** |

    The seed-123 row is the number this committed test actually produces -- the one a future
    debugger should compare against on a failure. The other six rows are the same measurement
    at six other seeds, run standalone with the identical procedure, showing the agreement is
    not an artifact of this particular window.

    This is a NumPy-vs-HF comparison, not NumPy-vs-device: both sides run entirely on the
    host in float32, from identical bfloat16-stored weights (see the plan's corrected
    "achievable tolerance" analysis -- the ~1e-3-unreachable argument from an earlier plan
    amendment bounds a *device* comparison and does not apply here). Agreement this tight is
    exactly what bf16-weights-with-fp32-accumulation predicts, and confirms the two
    independently-written paths are, numerically, computing the same function.
    """
    window = _fixed_window()
    logits_numpy = forward(REAL_CHECKPOINT, window)
    logits_hf = _hf_logits_fp32(HF_DIR, window)

    stats = _compare_logits(logits_numpy, logits_hf)

    assert stats["max_abs"] <= MAX_ABS_TOLERANCE, (
        f"NumPy-vs-HF max abs logit difference {stats['max_abs']:.6g} exceeds "
        f"{MAX_ABS_TOLERANCE} -- measured baseline (this test's exact seed/window) is "
        f"1.23e-5, nearly two orders of magnitude below this tolerance, so exceeding it "
        f"indicates a real divergence between the two paths, not measurement noise."
    )
    assert stats["max_rel"] <= MAX_REL_TOLERANCE, (
        f"NumPy-vs-HF max relative logit difference {stats['max_rel']:.6g} (restricted to "
        f"|logit| > {_REL_FLOOR}) exceeds {MAX_REL_TOLERANCE} -- measured baseline (this "
        f"test's exact seed/window) is 3.04e-4."
    )
    assert stats["corr"] >= MIN_CORRELATION, (
        f"NumPy-vs-HF logit correlation {stats['corr']:.10f} is below {MIN_CORRELATION} -- "
        f"measured baseline (this test's exact seed/window) is indistinguishable from 1.0 "
        f"(1 - corr ~= 1.28e-13)."
    )


# ---------------------------------------------------------------------------
# Step 2: the gate is not hollow.
# ---------------------------------------------------------------------------


def _identity_permute(tensor: np.ndarray, *, num_heads: int, head_dim: int) -> np.ndarray:
    """Stand-in for `convert.hf_mapping.permute_rope_qk` that does nothing -- the exact bug
    Plan 4's Task 3 found and fixed (straight-copied q/k rows, mismatching ttml's interleaved
    RoPE pairing against HF's split-halves pairing).
    """
    return tensor


@_needs_artifacts
def test_parity_gate_is_not_hollow_it_fails_when_rope_permutation_is_disabled():
    """Step 2 of the brief: prove this gate would actually catch a real conversion bug,
    using the same method Plan 4's reviewer used for the loss gate (monkeypatch
    `permute_rope_qk` to identity, reconvert into a throwaway directory, measure).

    Never touches `artifacts/hf/` -- `convert_checkpoint` is called directly against a fresh
    `tempfile.TemporaryDirectory()`, reading the same real checkpoint and tokenizer but
    writing nowhere near the tracked artifacts.

    **Measured divergence** (this exact procedure, at the committed `_WINDOW_LEN = 256`,
    standalone run): max_abs = 8.11, max_rel (floored) = 2.00 (the metric's ceiling --
    opposite-signed logits of similar magnitude), correlation = 0.934. For comparison, Plan
    4's reviewer measured the same bug at the *loss* level as 3.2015 nats against a 1.8781
    target (a 1.32-nat gap) -- this gate catches the identical defect at the logit level, and
    by a much wider margin relative to its own tolerance (8.11 / 1e-3 ~= 8100x over budget,
    versus the loss gate's bug being "only" ~6.6x its own 0.2-nat tolerance).
    """
    window = _fixed_window()
    logits_numpy = forward(REAL_CHECKPOINT, window)

    with tempfile.TemporaryDirectory() as tmp:
        broken_hf_dir = Path(tmp) / "hf_broken"
        with mock.patch.object(to_hf, "permute_rope_qk", _identity_permute):
            to_hf.convert_checkpoint(REAL_CHECKPOINT, TOKENIZER_DIR, broken_hf_dir)
        logits_hf_broken = _hf_logits_fp32(broken_hf_dir, window)

    stats = _compare_logits(logits_numpy, logits_hf_broken)

    # The gate's own tolerances must actually reject this -- if they didn't, the gate above
    # would be measuring nothing.
    assert stats["max_abs"] > MAX_ABS_TOLERANCE, stats
    assert stats["max_rel"] > MAX_REL_TOLERANCE, stats
    assert stats["corr"] < MIN_CORRELATION, stats
    # And the divergence is not marginal -- it is orders of magnitude past the tolerance,
    # matching the "1+ nats means a real layout bug" scale Plan 4 established for the loss
    # gate.
    assert stats["max_abs"] > 1.0, stats
    assert stats["corr"] < 0.99, stats


# ---------------------------------------------------------------------------
# Step 2b: not hollow to an activation-level defect either -- eps moved outside the sqrt in
# the NumPy path, the one perturbation docs/ttml-forward-reference.md's §9 Q6 measured as
# invisible to the held-out cross-entropy check.
# ---------------------------------------------------------------------------


def _rms_norm_eps_outside_sqrt(
    x: np.ndarray, gamma: np.ndarray, eps: float = ttml_forward.RMS_NORM_EPS
) -> np.ndarray:
    """A `rms_norm`-compatible replacement with epsilon moved *outside* the sqrt --
    `sqrt(mean(x^2)) + eps` instead of the correct `sqrt(mean(x^2) + eps)`.

    Not a synthetic strawman: `docs/ttml-forward-reference.md`'s §9 Q6 table identifies
    exactly this perturbation as the *one* thing its end-to-end cross-entropy check cannot
    see (Δ = -0.0002 nats, 0.0 SE, against that check's ≈0.22-nat 2σ detection floor) --
    every other perturbation Task 1 tried (`1+gamma` instead of `gamma`, an embedding scale)
    was loud at the CE level. If a second, sharper instrument is going to earn its keep, this
    is the specific case it has to catch.
    """
    mean_sq = np.mean(np.square(x), axis=-1, keepdims=True)
    rms = np.sqrt(mean_sq) + eps
    return gamma * x / rms


@_needs_artifacts
def test_parity_gate_is_not_hollow_it_catches_epsilon_moved_outside_the_sqrt():
    """A fourth not-hollow proof, and the most persuasive one available: this is the *one*
    perturbation known to be invisible to the CE check (see
    `_rms_norm_eps_outside_sqrt`'s docstring). If the parity gate were blind to it too, this
    gate would add nothing over the loss check for the one defect that matters most.

    Unlike the RoPE not-hollow test above, this needs no reconversion at all -- the defect is
    injected purely in the NumPy path, by monkeypatching `convert.ttml_forward.rms_norm` to
    `_rms_norm_eps_outside_sqrt` for the duration of one `forward()` call (patching the
    module attribute, not the name imported into this file, so every call inside `forward()`
    picks it up -- the same technique `test_parity_gate_is_not_hollow_it_fails_when_rope_permutation_is_disabled`
    uses for `to_hf.permute_rope_qk`). `artifacts/hf/` is read normally, unpatched, as the
    reference; nothing under `artifacts/checkpoints/` or `artifacts/hf/` is written, and no
    conversion runs, which is why this is faster than the other not-hollow tests.

    **Measured** (this exact procedure, at the committed `_WINDOW_LEN = 256`, standalone
    run): max_abs = 0.0370 (37x over `MAX_ABS_TOLERANCE`'s 1e-3 budget), max_rel (floored) =
    1.58, correlation = 0.9999985. That correlation figure is the interesting part: it stays
    *above* `MIN_CORRELATION`'s 0.9999 floor, so correlation alone would call this a pass --
    this perturbation is loud in absolute and relative terms while leaving the overall logit
    *shape* almost undisturbed, unlike the RoPE bug above (which drops correlation to ~0.93).
    That is exactly why the gate checks all three metrics rather than collapsing them into a
    single summary statistic.
    """
    window = _fixed_window()
    logits_hf = _hf_logits_fp32(HF_DIR, window)

    with mock.patch.object(ttml_forward, "rms_norm", _rms_norm_eps_outside_sqrt):
        logits_numpy_bad = ttml_forward.forward(REAL_CHECKPOINT, window)

    stats = _compare_logits(logits_numpy_bad, logits_hf)

    # The gate's own tolerances must actually reject this -- if they didn't, the gate would
    # be adding nothing over the CE check for the one perturbation the CE check cannot see.
    assert stats["max_abs"] > MAX_ABS_TOLERANCE, stats
    assert stats["max_rel"] > MAX_REL_TOLERANCE, stats
    # Correlation is deliberately NOT asserted to drop below MIN_CORRELATION -- measured at
    # 0.9999985, comfortably above it; see the docstring above for why that is the finding,
    # not an oversight.


# ---------------------------------------------------------------------------
# Step 3a: is the parity gate also blind to the norm-mapping bug the loss gate can't see?
# ---------------------------------------------------------------------------


def _swap_two_norm_destinations(ttml_name: str, *, weight_tying: bool = True):
    """A `hf_mapping.map_name`-compatible wrapper that swaps block 0's and block 1's
    `input_layernorm.weight` destinations -- a "swap two norms" mapping bug, applied to the
    real checkpoint's own `map_name` behaviour rather than a synthetic one.
    """
    target = hf_mapping.map_name(ttml_name, weight_tying=weight_tying)
    if target == "model.layers.0.input_layernorm.weight":
        return "model.layers.1.input_layernorm.weight"
    if target == "model.layers.1.input_layernorm.weight":
        return "model.layers.0.input_layernorm.weight"
    return target


@_needs_artifacts
def test_parity_gate_is_blind_to_a_norm_swap_on_the_real_checkpoint():
    """Step 3 of the brief, first half: verify -- don't assume -- that this new, much
    sharper gate is *also* blind to the norm-mapping bug the loss gate cannot see.

    All 13 RMSNorm gammas in the real checkpoint are exactly 1.0 (verified directly:
    `min == max == 1.0` for every one). Swapping which HF slot two of them land in is
    therefore a no-op on *this* checkpoint's actual numbers, regardless of how sharp the
    comparison instrument is -- there is no signal for any instrument to detect, not a gap
    specific to this gate's tolerance.

    **Measured** (this exact procedure, at the committed `_WINDOW_LEN = 256`, standalone
    run): swapping block 0's and block 1's `input_layernorm.weight` destinations and
    reconverting (into a throwaway directory, never touching `artifacts/hf/`) gives a
    NumPy-vs-HF max abs difference of 1.23e-5 -- identical to this test's seed's no-swap
    baseline (per the table in `test_numpy_fp32_vs_hf_fp32_agreement_on_fixed_window`'s
    docstring) to the precision measured, because the swap moves nothing (both gammas are
    exactly 1.0): indistinguishable from ordinary float32 noise, not a detected defect.

    **Honest finding: yes, this gate is blind to it too**, on this checkpoint, for the same
    reason the loss gate is (`docs/superpowers/plans/2026-08-12-numpy-parity.md`'s own
    motivation: "swapping two changes loss by exactly 0.0000"). This is exactly why Step 3's
    second half (`test_convert_checkpoint_places_each_rmsnorm_gamma_at_its_correct_destination`)
    exists as a structural test with synthetic non-unit gammas, independent of both gates.

    **Non-vacuity, the strong way.** A logit-comparison assertion alone cannot distinguish
    "the monkeypatch fired and produced an indistinguishable result" from "the monkeypatch
    silently didn't fire at all" -- both look identical to this test. So this also converts
    the *unpatched* checkpoint into a second throwaway directory and asserts the two
    `model.safetensors` files are **byte-identical by SHA-256**. That is strictly stronger
    than "no instrument here happened to notice": it proves no instrument *could* notice,
    because the two conversions produced the literal same bytes. **Measured** (this exact
    procedure, standalone run): both conversions hash to
    `3a85bb08e1d24f5b6ae499f55be85b594efdfd81503f21e0d35d72490462200d` -- confirming the
    monkeypatch did fire (it changed which HF name each norm's tensor is written under) and
    that the swap is still a genuine no-op on this checkpoint's degenerate gammas, not a
    disguised bug in the test itself.
    """
    window = _fixed_window()
    logits_numpy = forward(REAL_CHECKPOINT, window)

    with tempfile.TemporaryDirectory() as tmp:
        good_hf_dir = Path(tmp) / "hf_good"
        to_hf.convert_checkpoint(REAL_CHECKPOINT, TOKENIZER_DIR, good_hf_dir)

        swapped_hf_dir = Path(tmp) / "hf_normswap"
        with mock.patch.object(to_hf, "map_name", _swap_two_norm_destinations):
            to_hf.convert_checkpoint(REAL_CHECKPOINT, TOKENIZER_DIR, swapped_hf_dir)

        # Byte-identical, not just numerically-close: proves *no* instrument -- this gate,
        # a future sharper one, a bitwise diff -- could ever distinguish the swapped
        # conversion from the correct one on this checkpoint. Strictly stronger than the
        # logit-comparison assertions below, which only show this one instrument didn't.
        good_hash = hashlib.sha256((good_hf_dir / "model.safetensors").read_bytes()).hexdigest()
        swapped_hash = hashlib.sha256((swapped_hf_dir / "model.safetensors").read_bytes()).hexdigest()
        assert good_hash == swapped_hash, (
            f"expected the norm swap to be a byte-identical no-op on this checkpoint's "
            f"degenerate (all-1.0) gammas, but the unpatched conversion hashed to "
            f"{good_hash} and the swapped one to {swapped_hash} -- the monkeypatch changed "
            f"the output, meaning the gammas are no longer degenerate (see this test's "
            f"docstring) or the swap has some other effect this test doesn't expect."
        )

        logits_hf_swapped = _hf_logits_fp32(swapped_hf_dir, window)

    stats = _compare_logits(logits_numpy, logits_hf_swapped)

    # The whole point: this passes the SAME tolerance a correct conversion passes, on a
    # checkpoint where the swap is a real (if currently invisible) mapping bug. All three
    # metrics, not just max_abs, so "same tolerance" in the comment above is actually true.
    assert stats["max_abs"] <= MAX_ABS_TOLERANCE, (
        "expected the norm swap to be invisible on this checkpoint's degenerate (all-1.0) "
        f"gammas, but max_abs={stats['max_abs']:.6g} exceeded the gate's own tolerance -- "
        "if this fails, the checkpoint's gammas are no longer all 1.0 and this test's "
        "premise (and its docstring) needs updating, not the tolerance."
    )
    assert stats["max_rel"] <= MAX_REL_TOLERANCE, (
        f"expected the norm swap to be invisible, but max_rel={stats['max_rel']:.6g} "
        f"exceeded the gate's own tolerance -- see the max_abs assertion above for what a "
        f"failure here would mean."
    )
    assert stats["corr"] >= MIN_CORRELATION, (
        f"expected the norm swap to be invisible, but corr={stats['corr']:.10f} fell below "
        f"the gate's own tolerance -- see the max_abs assertion above for what a failure "
        f"here would mean."
    )


# ---------------------------------------------------------------------------
# Step 3b: close the blind spot directly, with a synthetic checkpoint. No skip guard -- this
# needs no real artifacts, and running unconditionally is the point (see _ARTIFACTS_PRESENT's
# comment above).
# ---------------------------------------------------------------------------


def _write_synthetic_checkpoint_with_distinct_gammas(
    path: Path, *, num_blocks: int = 3
) -> Dict[str, np.ndarray]:
    """A tiny but structurally faithful checkpoint (same tensor names, `[1,1,out,in]` /
    `[1,1,1,C]` leading-dim convention, declaration-order pickle stream as the real ttml
    format -- see `convert.checkpoint_reader`) whose RMSNorm gammas are all **distinct**,
    unlike the real checkpoint's degenerate all-1.0 gammas.

    Every gamma is a small ramp (`base + arange(C) * 0.01`) with a `base` unique to its
    (block, norm-type) pair, so a bug that swaps *which block* a gamma lands in, swaps
    *attention_norm for mlp_norm* (or vice versa) within a block, or transposes/reorders a
    gamma's own elements would all change the array this test compares against -- not just a
    scalar "is it present" check.

    Returns `{hf_destination_name: expected_gamma_array}` for every RMSNorm gamma the
    synthetic checkpoint contains, for the caller to check against the converted
    `model.safetensors`.
    """
    embedding_dim, num_heads, num_groups = 8, 4, 2
    head_dim = embedding_dim // num_heads
    intermediate_dim = 6
    vocab_size = 11
    rng = np.random.default_rng(42)

    def lin(out_f: int, in_f: int) -> np.ndarray:
        return (rng.normal(size=(1, 1, out_f, in_f)) * 0.1).astype(np.float32)

    expected: Dict[str, np.ndarray] = {}
    names_and_arrays: list = []
    for b in range(num_blocks):
        p = f"llama/llama_block_{b}"
        attn_gamma = (100.0 * b + 1.0) + np.arange(embedding_dim, dtype=np.float32) * 0.01
        mlp_gamma = (100.0 * b + 2.0) + np.arange(embedding_dim, dtype=np.float32) * 0.01
        names_and_arrays += [
            (f"{p}/attention_norm/gamma", attn_gamma.reshape(1, 1, 1, embedding_dim)),
            (f"{p}/attention/q_linear/weight", lin(num_heads * head_dim, embedding_dim)),
            (f"{p}/attention/kv_linear/weight", lin(2 * num_groups * head_dim, embedding_dim)),
            (f"{p}/attention/out_linear/weight", lin(embedding_dim, num_heads * head_dim)),
            (f"{p}/mlp_norm/gamma", mlp_gamma.reshape(1, 1, 1, embedding_dim)),
            (f"{p}/mlp/w1/weight", lin(intermediate_dim, embedding_dim)),
            (f"{p}/mlp/w3/weight", lin(intermediate_dim, embedding_dim)),
            (f"{p}/mlp/w2/weight", lin(embedding_dim, intermediate_dim)),
        ]
        expected[f"model.layers.{b}.input_layernorm.weight"] = attn_gamma
        expected[f"model.layers.{b}.post_attention_layernorm.weight"] = mlp_gamma

    ln_fc_gamma = 999.0 + np.arange(embedding_dim, dtype=np.float32) * 0.01
    names_and_arrays += [
        ("llama/ln_fc/gamma", ln_fc_gamma.reshape(1, 1, 1, embedding_dim)),
        ("llama/fc/weight", lin(vocab_size, embedding_dim)),
    ]
    expected["model.norm.weight"] = ln_fc_gamma

    manifest = {"model": {"named_parameters": {name: {} for name, _ in names_and_arrays}}}
    header = {
        "vocab_size": vocab_size,
        "seq_len": 32,
        "intermediate_dim": intermediate_dim,
        "weight_tying": True,
        "rms_norm_eps": 1e-5,
        "weights_dtype": "float32",
        "transformer_config": {
            "embedding_dim": embedding_dim,
            "num_blocks": num_blocks,
            "num_heads": num_heads,
            "num_groups": num_groups,
            "theta": 10000.0,
        },
    }
    record = {"format": 1, "header": header, "manifest": manifest}
    with open(path, "wb") as f:
        pickle.dump(record, f)
        for _name, array in names_and_arrays:
            pickle.dump(array, f)
    return expected


def test_convert_checkpoint_places_each_rmsnorm_gamma_at_its_correct_destination(tmp_path):
    """Step 3 of the brief, second half: close the norm-mapping blind spot directly.

    Runs **unconditionally** -- no `artifacts/` dependency, unlike almost everything else in
    this file (and, per Task 2's review, most of the rest of the suite). Builds a synthetic
    checkpoint whose gammas are distinct non-unit values (see
    `_write_synthetic_checkpoint_with_distinct_gammas`), runs it through the real,
    unmodified `convert.to_hf.convert_checkpoint`, and asserts every gamma landed at its
    correct HF destination -- not merely present, not merely the right shape, but the right
    *values*, which a block-swap or attention/mlp-norm-swap bug would change.

    This does not depend on the real checkpoint's gammas ever stopping being exactly 1.0
    (the current state is attributed to an upstream `stochastic_rounding` issue -- see
    `docs/superpowers/specs/2026-08-11-followups.md`); it keeps working, and keeps meaning
    the same thing, regardless of what the real checkpoint's gammas happen to be.
    """
    ckpt_path = tmp_path / "synthetic_distinct_gammas.pkl"
    expected = _write_synthetic_checkpoint_with_distinct_gammas(ckpt_path, num_blocks=3)
    tokenizer_dir = tmp_path / "empty_tokenizer"
    tokenizer_dir.mkdir()
    out_dir = tmp_path / "hf_out"

    convert_checkpoint(ckpt_path, tokenizer_dir, out_dir)

    from safetensors.numpy import load_file

    tensors = load_file(str(out_dir / "model.safetensors"))

    # Non-vacuity check: every expected gamma must actually differ from every other one, or
    # a swap bug between two of them would produce a false pass below -- exactly the trap
    # that made the real checkpoint's degenerate gammas useless for this purpose in the
    # first place.
    items = list(expected.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            name_i, gamma_i = items[i]
            name_j, gamma_j = items[j]
            assert not np.array_equal(gamma_i, gamma_j), (
                f"synthetic fixture produced identical gammas for {name_i!r} and {name_j!r}; "
                "fix the fixture -- this test would be vacuous for that pair otherwise"
            )

    for hf_name, expected_gamma in expected.items():
        assert hf_name in tensors, (
            f"expected {hf_name!r} among the converted tensors, got {sorted(tensors)}"
        )
        np.testing.assert_array_equal(
            tensors[hf_name],
            expected_gamma,
            err_msg=(
                f"{hf_name}: converted gamma does not match the synthetic checkpoint's "
                "value for this destination -- likely swapped with another norm's slot "
                "(wrong block, or attention_norm/mlp_norm confused)"
            ),
        )
