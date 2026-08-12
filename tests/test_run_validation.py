# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Periodic validation in the chunked training loop. Pure Python — no hardware, no ttml
import (``train/run.py`` only imports ttml/ttnn lazily inside ``main()``, so importing this
module at test-collection time is safe on a machine with no device).

``run_training_loop`` is the extracted, dependency-injected core of the ``while remaining >
0`` chunk loop that used to live directly in ``main()``. Tests here pass fake ``train_fn``/
``evaluate_fn``/``save_checkpoint_fn`` callables standing in for ttml's ``train()``, this
module's real ``evaluate()``, and the checkpoint-writing closure — so the *cadence* and
*recording* logic (the thing this task adds) is exercised without a board.

The one property these tests must nail down, per the task brief: a run with ``--val-every``
records one validation entry per boundary, and those entries are **not** copies of the
training loss. ttml's own ``val_losses`` is a documented placeholder that does exactly that
(copies ``train_loss``) — if a future edit to ``run_training_loop`` accidentally wired
``losses[-1]`` into the ``val_loss`` field instead of calling ``evaluate_fn``, every test
below that checks ``val_loss != train_loss`` would catch it, because the fake ``evaluate_fn``
returns values from a disjoint numeric range from the fake ``train_fn`` losses.
"""

from __future__ import annotations

import json

import pytest

from train.run import _warn_if_stochastic_rounding_disabled, run_training_loop


class _FakeCfg:
    """Stand-in for train.config.RunConfig: run_training_loop only reads/writes .steps."""

    def __init__(self, steps: int):
        self.steps = steps
        self.seq_len = 8  # unused by the fakes below, but present like the real RunConfig


def _fake_train_fn(losses_log):
    """Returns a train_fn that hands back one descending "loss" per requested step and
    records how many steps it was asked to run, so tests can check chunk sizing."""

    def train_fn(cfg, model, optimizer, train_ids, use_ddp, use_tp):
        n = cfg.steps
        losses_log.append(n)
        # Values deliberately land far below the fake evaluate_fn's range (900s) so an
        # accidental "val_loss = train_loss" bug is unmistakable in any assertion below.
        return [10.0 - 0.001 * i for i in range(n)], None

    return train_fn


def _fake_evaluate_fn(eval_calls):
    """Returns an evaluate_fn that records how many times it fired and returns a value
    from a numeric range disjoint from the fake train losses above."""

    def evaluate_fn(model, val_ids, cfg, batches=10):
        eval_calls.append(cfg.steps)  # cfg.steps at call time, just to prove cfg flows through
        return 900.0 + len(eval_calls)

    return evaluate_fn


# ---------------------------------------------------------------------------
# Recording cadence + non-placeholder values
# ---------------------------------------------------------------------------


def test_val_every_records_one_entry_per_boundary(tmp_path):
    """200 steps, --val-every 100 -> exactly 2 recorded entries, at steps 100 and 200 —
    the same shape as the real 200-step / --val-every 100 verification run."""
    cfg = _FakeCfg(steps=200)
    eval_calls = []
    val_log = tmp_path / "val_losses.jsonl"

    _, val_records = run_training_loop(
        cfg, model=None, optimizer=None, train_ids=None, val_ids=None,
        save_every=0, val_every=100, start_step=0, val_log_path=val_log,
        train_fn=_fake_train_fn([]), evaluate_fn=_fake_evaluate_fn(eval_calls),
        save_checkpoint_fn=None, print_fn=lambda *a, **k: None,
    )

    assert [r["step"] for r in val_records] == [100, 200]
    assert len(eval_calls) == 2


def test_val_loss_is_not_a_copy_of_train_loss(tmp_path):
    """The important assertion: recorded val_loss must differ from the train_loss recorded
    alongside it. Equality would mean ttml's placeholder (a copy of train_loss) leaked in
    instead of the real evaluate() being called."""
    cfg = _FakeCfg(steps=200)
    val_log = tmp_path / "val_losses.jsonl"

    _, val_records = run_training_loop(
        cfg, model=None, optimizer=None, train_ids=None, val_ids=None,
        save_every=0, val_every=100, start_step=0, val_log_path=val_log,
        train_fn=_fake_train_fn([]), evaluate_fn=_fake_evaluate_fn([]),
        save_checkpoint_fn=None, print_fn=lambda *a, **k: None,
    )

    assert len(val_records) == 2
    for record in val_records:
        assert record["val_loss"] != record["train_loss"]
        # Also pin the ranges so this isn't accidentally vacuous (e.g. both being NaN).
        assert record["train_loss"] < 100
        assert record["val_loss"] > 800


def test_val_log_jsonl_file_persists_the_recorded_curve(tmp_path):
    """The curve must survive the process: one JSON object per line, matching what was
    returned in-memory, appended (not overwritten) to <checkpoint-dir>/val_losses.jsonl."""
    cfg = _FakeCfg(steps=200)
    val_log = tmp_path / "val_losses.jsonl"

    _, val_records = run_training_loop(
        cfg, model=None, optimizer=None, train_ids=None, val_ids=None,
        save_every=0, val_every=100, start_step=0, val_log_path=val_log,
        train_fn=_fake_train_fn([]), evaluate_fn=_fake_evaluate_fn([]),
        save_checkpoint_fn=None, print_fn=lambda *a, **k: None,
    )

    assert val_log.is_file()
    lines = val_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed == val_records
    for record in parsed:
        assert set(record) == {"step", "train_loss", "val_loss"}


def test_val_every_zero_disables_periodic_validation(tmp_path):
    """0 (the default) must not create the jsonl file or call evaluate_fn at all — this is
    an opt-in feature, not a behavior change for every existing caller."""
    cfg = _FakeCfg(steps=200)
    val_log = tmp_path / "val_losses.jsonl"
    eval_calls = []

    _, val_records = run_training_loop(
        cfg, model=None, optimizer=None, train_ids=None, val_ids=None,
        save_every=0, val_every=0, start_step=0, val_log_path=val_log,
        train_fn=_fake_train_fn([]), evaluate_fn=_fake_evaluate_fn(eval_calls),
        save_checkpoint_fn=None, print_fn=lambda *a, **k: None,
    )

    assert val_records == []
    assert eval_calls == []
    assert not val_log.exists()


def test_val_boundary_still_fires_once_at_the_end_of_a_run_not_evenly_divisible():
    """steps=90, val_every=40 -> boundaries at 40, 80, and then the final 10-step chunk
    must still trigger one last validation at step 90, matching the pre-existing
    end-of-run checkpoint behaviour (always checkpoint after the final chunk)."""
    cfg = _FakeCfg(steps=90)

    _, val_records = run_training_loop(
        cfg, model=None, optimizer=None, train_ids=None, val_ids=None,
        save_every=0, val_every=40, start_step=0, val_log_path=None,
        train_fn=_fake_train_fn([]), evaluate_fn=_fake_evaluate_fn([]),
        save_checkpoint_fn=None, print_fn=lambda *a, **k: None,
    )

    assert [r["step"] for r in val_records] == [40, 80, 90]


# ---------------------------------------------------------------------------
# Independence from --save-every
# ---------------------------------------------------------------------------


def test_val_every_is_independent_of_save_every_boundaries():
    """save_every=30, val_every=100, steps=200: checkpoint boundaries (every 30, plus the
    final non-aligned chunk at 200) must not be required to coincide with validation
    boundaries (100, 200), and vice versa. This is the interaction called out in the task
    brief -- --val-every must not assume it lines up with --save-every."""
    cfg = _FakeCfg(steps=200)
    checkpoint_calls = []

    all_losses, val_records = run_training_loop(
        cfg, model=None, optimizer=None, train_ids=None, val_ids=None,
        save_every=30, val_every=100, start_step=0, val_log_path=None,
        train_fn=_fake_train_fn([]), evaluate_fn=_fake_evaluate_fn([]),
        save_checkpoint_fn=lambda step: checkpoint_calls.append(step),
        print_fn=lambda *a, **k: None,
    )

    # Checkpoints: every 30 steps, plus the run's final step (200 is not a multiple of 30).
    assert checkpoint_calls == [30, 60, 90, 120, 150, 180, 200]
    # Validations: every 100 steps -- unaffected by the 30-step checkpoint cadence.
    assert [r["step"] for r in val_records] == [100, 200]
    # All 200 steps were actually run, across however many sub-chunks that took.
    assert len(all_losses) == 200


def test_save_checkpoint_fn_not_called_when_save_every_is_zero():
    cfg = _FakeCfg(steps=50)
    checkpoint_calls = []

    run_training_loop(
        cfg, model=None, optimizer=None, train_ids=None, val_ids=None,
        save_every=0, val_every=0, start_step=0, val_log_path=None,
        train_fn=_fake_train_fn([]), evaluate_fn=_fake_evaluate_fn([]),
        save_checkpoint_fn=lambda step: checkpoint_calls.append(step),
        print_fn=lambda *a, **k: None,
    )

    assert checkpoint_calls == []


def test_step_numbers_in_records_are_absolute_offset_by_start_step():
    """--resume runs pass a nonzero start_step; recorded step numbers (and checkpoint
    filenames elsewhere) must be absolute training steps, not steps-within-this-invocation,
    or a curve stitched across resumes would silently overlap."""
    cfg = _FakeCfg(steps=100)

    _, val_records = run_training_loop(
        cfg, model=None, optimizer=None, train_ids=None, val_ids=None,
        save_every=0, val_every=50, start_step=1000, val_log_path=None,
        train_fn=_fake_train_fn([]), evaluate_fn=_fake_evaluate_fn([]),
        save_checkpoint_fn=None, print_fn=lambda *a, **k: None,
    )

    assert [r["step"] for r in val_records] == [1050, 1100]


# ---------------------------------------------------------------------------
# stochastic_rounding runtime warning
# ---------------------------------------------------------------------------


def _yaml_config(stochastic_rounding: bool):
    return {"training_config": {"optimizer": {"stochastic_rounding": stochastic_rounding}}}


def test_warning_fires_when_stochastic_rounding_is_disabled(capsys):
    _warn_if_stochastic_rounding_disabled(_yaml_config(False))
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "stochastic_rounding" in captured.err
    # Names the concrete consequence, not just the flag -- an operator scanning stderr
    # should see *why* this matters without going to read train/config.py.
    assert "0.0039" in captured.err
    assert "3e-4" in captured.err
    # The resolved value is still printed either way (on stdout, alongside normal status).
    assert "False" in captured.out


def test_no_warning_when_stochastic_rounding_is_enabled(capsys):
    _warn_if_stochastic_rounding_disabled(_yaml_config(True))
    captured = capsys.readouterr()
    assert "WARNING" not in captured.err
    assert captured.err == ""
    # Resolved value is still printed on the non-warning path too.
    assert "True" in captured.out


@pytest.mark.parametrize("value", [False, True])
def test_resolved_value_is_always_printed(capsys, value):
    """Regardless of enabled/disabled, the operator must see the resolved value -- this is
    the fix for --config-omitted runs previously printing nothing about the optimizer."""
    _warn_if_stochastic_rounding_disabled(_yaml_config(value))
    captured = capsys.readouterr()
    assert "stochastic_rounding" in captured.out
    assert str(value) in captured.out
