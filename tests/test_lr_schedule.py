# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The ``--lr-schedule`` learning-rate schedule. Pure Python — no hardware, no ttml import
(``train/run.py`` imports ttml/ttnn lazily inside ``main()``, so importing it at collection
time is safe on a machine with no device).

Two things are pinned here, and they matter for different reasons.

**The shape** (``lr_at_step``): that ``constant`` is inert, that the decays start at the
configured LR and land exactly on ``min_lr`` at the final step, that they are monotonically
non-increasing, and that ``decay_start_frac`` holds the LR genuinely flat before it bites.
That last one is what makes the v3-vs-v4 A/B valid at all — if the "held" portion were not
byte-for-byte the constant-LR behaviour, the comparison would be measuring two changes.

**The plumbing** (``run_training_loop``): that the LR is applied once per chunk via
``optimizer.set_lr()`` at the chunk *midpoint*, and — the regression that would silently
invalidate every previously-recorded run — that with no schedule, ``set_lr`` is **never
called**. The fake optimizer below records every call, so an accidental
``optimizer.set_lr(base_lr)`` in the default path fails the test rather than quietly
re-writing the LR of runs that predate this feature.
"""

from __future__ import annotations

import json

import pytest

from train.run import LR_SCHEDULES, lr_at_step, run_training_loop


class _FakeCfg:
    """Stand-in for train.config.RunConfig: run_training_loop only reads/writes .steps."""

    def __init__(self, steps: int):
        self.steps = steps
        self.seq_len = 8


class _RecordingOptimizer:
    """Records every set_lr() call so the tests can assert on both the values and — for the
    constant default — on the total absence of calls."""

    def __init__(self):
        self.lrs = []

    def set_lr(self, lr):
        self.lrs.append(lr)


def _fake_train_fn(chunks):
    def train_fn(cfg, model, optimizer, train_ids, use_ddp, use_tp):
        chunks.append(cfg.steps)
        return [1.0] * cfg.steps, None

    return train_fn


def _fake_evaluate_fn(model, val_ids, cfg, batches=10, use_ddp=False):
    """``use_ddp`` is accepted but ignored: these tests are about the LR schedule, and the
    real ``evaluate()`` takes it to shard its batch and compose the loss off the mesh."""
    return 0.5


# ---------------------------------------------------------------------------
# lr_at_step — shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("step", [0, 1, 500, 5000, 10764])
def test_constant_is_inert_at_every_step(step):
    """'constant' returns the base LR everywhere and ignores min_lr/decay_start_frac
    entirely — it is the pre-schedule behaviour spelled out, not a decay with flat
    parameters."""
    assert lr_at_step(3e-4, step, 10764, schedule="constant",
                      min_lr=1e-9, decay_start_frac=0.5) == 3e-4


@pytest.mark.parametrize("schedule", ["cosine", "linear"])
def test_decay_starts_at_base_lr_and_ends_at_min_lr(schedule):
    """The two endpoints are the contract: no surprise drop at step 0, and the floor is
    actually reached at the final step rather than approached."""
    total = 10764
    assert lr_at_step(3e-4, 0, total, schedule=schedule, min_lr=3e-5) == pytest.approx(3e-4)
    assert lr_at_step(3e-4, total, total, schedule=schedule,
                      min_lr=3e-5) == pytest.approx(3e-5)


@pytest.mark.parametrize("schedule", ["cosine", "linear"])
def test_decay_is_monotonically_non_increasing(schedule):
    total = 10764
    values = [lr_at_step(3e-4, s, total, schedule=schedule, min_lr=3e-5)
              for s in range(0, total + 1, 100)]
    assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))


@pytest.mark.parametrize("schedule", ["cosine", "linear"])
def test_decay_start_frac_holds_the_lr_genuinely_flat(schedule):
    """Everything at or before the hold fraction is *exactly* base_lr — this is what lets
    the first half of the v4 run reproduce v3 and makes the A/B isolate the decay."""
    total, base = 10764, 3e-4
    for step in [0, 1000, 2500, 5000, int(0.5 * total)]:
        assert lr_at_step(base, step, total, schedule=schedule, min_lr=3e-5,
                          decay_start_frac=0.5) == base
    # ...and it moves immediately after.
    assert lr_at_step(base, int(0.5 * total) + 500, total, schedule=schedule,
                      min_lr=3e-5, decay_start_frac=0.5) < base


@pytest.mark.parametrize("schedule", ["cosine", "linear"])
def test_held_decay_still_lands_exactly_on_min_lr(schedule):
    """The post-hold remainder is re-normalised, so the floor is hit at the final step
    whatever fraction was held — not scaled down by the hold."""
    total = 10764
    assert lr_at_step(3e-4, total, total, schedule=schedule, min_lr=3e-5,
                      decay_start_frac=0.5) == pytest.approx(3e-5)


def test_linear_decay_is_linear_in_the_decay_window():
    """Half way through a full-run linear decay is half way between base and min."""
    assert lr_at_step(3e-4, 5382, 10764, schedule="linear",
                      min_lr=0.0) == pytest.approx(1.5e-4)


def test_cosine_is_above_linear_early_and_below_it_late():
    """Distinguishes the two shapes rather than just asserting 'it decays': cosine holds
    high then falls off, crossing linear at the midpoint."""
    total, kw = 10764, {"min_lr": 0.0}
    quarter = int(0.25 * total)
    assert lr_at_step(3e-4, quarter, total, schedule="cosine", **kw) > \
        lr_at_step(3e-4, quarter, total, schedule="linear", **kw)
    three_q = int(0.75 * total)
    assert lr_at_step(3e-4, three_q, total, schedule="cosine", **kw) < \
        lr_at_step(3e-4, three_q, total, schedule="linear", **kw)


def test_steps_are_clamped_outside_the_run():
    assert lr_at_step(3e-4, -50, 1000, schedule="linear", min_lr=0.0) == pytest.approx(3e-4)
    assert lr_at_step(3e-4, 99999, 1000, schedule="linear", min_lr=0.0) == pytest.approx(0.0)


def test_zero_total_steps_yields_base_lr_rather_than_dividing_by_zero():
    assert lr_at_step(3e-4, 0, 0, schedule="cosine", min_lr=0.0) == 3e-4


def test_decay_start_frac_of_one_never_decays():
    """Guards the 1/(1 - frac) division. The CLI rejects 1.0, but the function is public."""
    assert lr_at_step(3e-4, 1000, 1000, schedule="cosine", min_lr=0.0,
                      decay_start_frac=1.0) == 3e-4


def test_unknown_schedule_raises():
    with pytest.raises(ValueError, match="unknown lr schedule"):
        lr_at_step(3e-4, 0, 100, schedule="exponential")


def test_every_advertised_schedule_is_actually_implemented():
    """LR_SCHEDULES drives argparse's choices=, so a name listed there but unhandled would
    be a CLI option that raises ValueError only once the run is underway."""
    for schedule in LR_SCHEDULES:
        assert isinstance(lr_at_step(3e-4, 10, 100, schedule=schedule, min_lr=1e-5), float)


# ---------------------------------------------------------------------------
# run_training_loop — plumbing
# ---------------------------------------------------------------------------


def test_no_schedule_means_set_lr_is_never_called():
    """THE regression guard. Every measurement recorded before --lr-schedule existed was
    produced by a loop that never touched the optimizer's LR; the default path must keep
    behaving that way, not call set_lr with a value that merely happens to be equal."""
    optimizer = _RecordingOptimizer()
    run_training_loop(
        _FakeCfg(1000), object(), optimizer, None, None,
        save_every=0, val_every=500, start_step=0, val_log_path=None,
        train_fn=_fake_train_fn([]), evaluate_fn=_fake_evaluate_fn,
    )
    assert optimizer.lrs == []


def test_lr_is_applied_once_per_chunk_at_the_chunk_midpoint():
    """1000 steps, --val-every 500 -> two 500-step chunks, midpoints 250 and 750."""
    optimizer = _RecordingOptimizer()
    seen = []

    def lr_fn(position):
        seen.append(position)
        return 0.1 * len(seen)

    chunks = []
    run_training_loop(
        _FakeCfg(1000), object(), optimizer, None, None,
        save_every=0, val_every=500, start_step=0, val_log_path=None,
        train_fn=_fake_train_fn(chunks), evaluate_fn=_fake_evaluate_fn, lr_fn=lr_fn,
    )
    assert chunks == [500, 500]
    assert seen == [250.0, 750.0]
    assert optimizer.lrs == [pytest.approx(0.1), pytest.approx(0.2)]


def test_midpoint_is_used_for_a_short_final_chunk_too():
    """The run's remainder chunk is shorter than the others; its midpoint must reflect its
    real length, which is precisely what a step-counting scheduler object would get wrong."""
    optimizer = _RecordingOptimizer()
    seen = []
    run_training_loop(
        _FakeCfg(764), object(), optimizer, None, None,
        save_every=0, val_every=500, start_step=0, val_log_path=None,
        train_fn=_fake_train_fn([]), evaluate_fn=_fake_evaluate_fn,
        lr_fn=lambda position: seen.append(position) or 1.0,
    )
    # chunks are 500 then 264 -> midpoints 250 and 500 + 132 = 632
    assert seen == [250.0, 632.0]


def test_lr_is_set_before_the_chunk_it_applies_to_runs():
    """Ordering matters: setting the LR after train() would apply each chunk's rate to the
    following chunk, shifting the whole schedule by one step of the staircase."""
    optimizer = _RecordingOptimizer()
    events = []

    def train_fn(cfg, model, opt, train_ids, use_ddp, use_tp):
        events.append(("train", opt.lrs[-1]))
        return [1.0] * cfg.steps, None

    def lr_fn(position):
        events.append(("set_lr", position))
        return position

    run_training_loop(
        _FakeCfg(1000), object(), optimizer, None, None,
        save_every=0, val_every=500, start_step=0, val_log_path=None,
        train_fn=train_fn, evaluate_fn=_fake_evaluate_fn, lr_fn=lr_fn,
    )
    assert [kind for kind, _ in events] == ["set_lr", "train", "set_lr", "train"]
    # each train() saw the LR chosen for its own chunk
    assert events[1] == ("train", 250.0)
    assert events[3] == ("train", 750.0)


def test_val_records_carry_the_lr_when_a_schedule_is_active(tmp_path):
    log = tmp_path / "val_losses.jsonl"
    run_training_loop(
        _FakeCfg(1000), object(), _RecordingOptimizer(), None, None,
        save_every=0, val_every=500, start_step=0, val_log_path=log,
        train_fn=_fake_train_fn([]), evaluate_fn=_fake_evaluate_fn,
        lr_fn=lambda position: 3e-4,
    )
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["lr"] for r in records] == [3e-4, 3e-4]


def test_val_records_omit_lr_under_the_constant_default(tmp_path):
    """Keeps the jsonl shape identical to every curve recorded before this feature, so v3's
    log and a constant v4 log stay directly comparable field-for-field."""
    log = tmp_path / "val_losses.jsonl"
    run_training_loop(
        _FakeCfg(1000), object(), _RecordingOptimizer(), None, None,
        save_every=0, val_every=500, start_step=0, val_log_path=log,
        train_fn=_fake_train_fn([]), evaluate_fn=_fake_evaluate_fn,
    )
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert all("lr" not in r for r in records)
    assert all(set(r) == {"step", "train_loss", "val_loss"} for r in records)


def test_the_planned_v4_schedule_holds_v3s_lr_for_the_first_half():
    """End-to-end sanity on the actual experiment: with the real run's shape, the LR that
    reaches the optimizer is exactly 3e-4 for every chunk in the first half, then strictly
    decreasing, ending near 3e-5. This is the property the v3-vs-v4 comparison rests on."""
    optimizer = _RecordingOptimizer()
    total = 10764
    run_training_loop(
        _FakeCfg(total), object(), optimizer, None, None,
        save_every=1000, val_every=500, start_step=0, val_log_path=None,
        train_fn=_fake_train_fn([]), evaluate_fn=_fake_evaluate_fn,
        save_checkpoint_fn=lambda step: None,
        lr_fn=lambda position: lr_at_step(3e-4, position, total, schedule="cosine",
                                          min_lr=3e-5, decay_start_frac=0.5),
    )
    # Chunk midpoints are 250 + 500k; the halfway mark is step 5382, so midpoints 250
    # through 5250 (eleven chunks, covering steps 0-5500) are held at the full LR.
    held = [lr for lr in optimizer.lrs if lr == 3e-4]
    assert len(held) == 11
    decaying = optimizer.lrs[len(held):]
    assert all(b < a for a, b in zip(decaying, decaying[1:]))
    assert decaying[-1] == pytest.approx(3e-5, rel=0.05)
