# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Gamma-degeneracy check: did the RMSNorm layers actually learn?

The original 3000-step run (``artifacts/checkpoints/nanollama3_step00003000.pkl``) produced
13 RMSNorm gammas all exactly 1.0 — every normalization layer in the model was, in effect,
frozen at its initial value for the entire run. This was **not** a missing-gradient bug: the
optimizer's ``exp_avg`` moments for these tensors were real (absmax ~3.6e-4 for block-0
gammas, ~2.5e-3 for ``ln_fc``). The bug is arithmetic. Parameters are stored in bfloat16, and
bfloat16's precision near 1.0 has a step size (ulp) of 0.0039 — roughly an order of magnitude
larger than the ~3e-4 Adam update those gradients produced. With ``stochastic_rounding: False``
(ttml's default, see ``optimizers/optimizer_registry.cpp:37``), every update rounds
*deterministically* back down to the nearest representable bfloat16 value, which is 1.0 again.
The optimizer computed a real step every single time; bfloat16 rounding discarded it every
single time.

**This is invisible in the loss curve.** The original run's training loss fell cleanly from
10.6875 to 1.9219 over 3000 steps — a completely healthy-looking trajectory — because plenty
of *other* parameters (attention/MLP weights) were learning normally. Only inspecting the
gamma tensors' own statistics (here, standard deviation) reveals that an entire class of
layers never moved. That gap between "the loss looks fine" and "13 layers are frozen" is
exactly why this module exists: a test on parameter statistics, not just on the training
metric.

See ``docs/superpowers/specs/2026-08-11-followups.md`` item 1 and
``docs/superpowers/plans/2026-08-12-real-training-run.md`` for the full investigation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from convert.checkpoint_reader import read_tensors

ROOT = Path(__file__).resolve().parent.parent

#: The 0.43-epoch baseline run this whole investigation started from. Evidence, not a
#: scratch artifact — never written to, per this task's global constraints.
BASELINE_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "nanollama3_step00003000.pkl"

#: Directories a *fixed* run's checkpoints might land in: Task 3's real multi-epoch run
#: (``checkpoints-v2``) and this task's own short proof runs (``checkpoints-scratch``). Both
#: are skip-guarded below, so this file stays meaningful (and importable) whether zero, one,
#: or both exist on the machine running the suite.
FIXED_RUN_CHECKPOINT_DIRS = [
    ROOT / "artifacts" / "checkpoints-v2",
    ROOT / "artifacts" / "checkpoints-scratch",
]

#: 6 blocks x (attention_norm + mlp_norm) + the final ln_fc.
EXPECTED_GAMMA_COUNT = 13


def _latest_checkpoint_in(directory: Path) -> Optional[Path]:
    """Highest-step ``.pkl`` in ``directory``, or ``None`` if it has none (or doesn't exist).

    Zero-padded step numbers in the filename (``nanollama3_step00000200.pkl``) sort
    correctly as plain strings, matching ``train/checkpoint.py``'s own ``latest_checkpoint``.
    """
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob("*.pkl"))
    return candidates[-1] if candidates else None


def assert_checkpoint_gammas_are_not_degenerate(checkpoint_path) -> None:
    """Raise ``AssertionError`` if any RMSNorm gamma in ``checkpoint_path`` has sd == 0.0.

    Deliberately **not** itself a pytest test function (pytest has no fixture named
    ``checkpoint_path`` to inject) — this is the reusable assertion body, parameterised by
    path exactly as the task brief asked ("make it take a path so it can be pointed at
    either run"). Both ``test_baseline_checkpoint_is_known_to_have_degenerate_gammas`` below
    and the ad-hoc proof-run check in this task's report call this directly.
    """
    checkpoint_path = Path(checkpoint_path)
    gammas = [
        (name, tensor)
        for name, tensor in read_tensors(checkpoint_path)
        if name.endswith("/gamma")
    ]
    assert len(gammas) == EXPECTED_GAMMA_COUNT, (
        f"expected {EXPECTED_GAMMA_COUNT} RMSNorm gammas (6 attention_norm + 6 mlp_norm + "
        f"1 ln_fc), found {len(gammas)} in {checkpoint_path}"
    )
    degenerate = [name for name, g in gammas if float(g.astype("float32").std()) == 0.0]
    assert not degenerate, (
        f"{len(degenerate)}/{EXPECTED_GAMMA_COUNT} gammas have sd exactly 0.0 (never "
        f"learned) in {checkpoint_path}: {degenerate}"
    )


@pytest.mark.skipif(
    not BASELINE_CHECKPOINT.is_file(),
    reason="baseline checkpoint not present at artifacts/checkpoints/nanollama3_step00003000.pkl",
)
def test_baseline_checkpoint_is_known_to_have_degenerate_gammas():
    """Pins the bug as a fact, not a story — and proves the checker is not vacuous.

    This is a **known-bad historical artifact**: the original 3000-step run, trained before
    ``stochastic_rounding`` was understood to matter here. It is expected, permanently, to
    fail ``assert_checkpoint_gammas_are_not_degenerate`` — if this test ever stops raising,
    either the baseline file changed (it must not: it's the evidence Task 3 measures
    against) or the degeneracy checker itself broke and needs re-verifying, not the training
    run.

    Without this test, ``assert_checkpoint_gammas_are_not_degenerate`` could pass on every
    real checkpoint by pure accident of a broken comparison (e.g. comparing the wrong dtype,
    or never actually iterating the gammas) and nothing in the suite would notice. This test
    is what proves the check can fail, and does, on real data.
    """
    with pytest.raises(AssertionError, match="never learned"):
        assert_checkpoint_gammas_are_not_degenerate(BASELINE_CHECKPOINT)


@pytest.mark.parametrize(
    "checkpoint_dir", FIXED_RUN_CHECKPOINT_DIRS, ids=lambda p: p.name,
)
def test_checkpoint_gammas_are_not_degenerate(checkpoint_dir):
    """The test that would have caught the original bug, pointed at a *fixed* run.

    Skips (rather than fails) when the directory has no checkpoint yet — this test exists
    to hold the line once a ``stochastic_rounding``-fixed run exists (this task's own
    200-step proof run, or Task 3's full multi-epoch run), not to fail permanently against a
    directory nobody has populated on this machine.
    """
    latest = _latest_checkpoint_in(checkpoint_dir)
    if latest is None:
        pytest.skip(f"no checkpoint found under {checkpoint_dir}")
    assert_checkpoint_gammas_are_not_degenerate(latest)
