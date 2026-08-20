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
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]

from train.checkpoint import build_header, validate_header
from train.run import (
    _mesh_graph_descriptor_path,
    _warn_if_stochastic_rounding_disabled,
    run_training_loop,
    ttml_cxx_header_fields,
)
from train.sizes import SIZES


class _FakeCfg:
    """Stand-in for train.config.RunConfig: run_training_loop only reads/writes .steps."""

    def __init__(self, steps: int):
        self.steps = steps
        self.seq_len = 8  # unused by the fakes below, but present like the real RunConfig


def _fake_train_fn(losses_log, ddp_log=None):
    """Returns a train_fn that hands back one descending "loss" per requested step and
    records how many steps it was asked to run, so tests can check chunk sizing.

    ``ddp_log``, when given, records the ``use_ddp`` argument of every call — the flag that
    decides whether ttml's real ``train()`` shards the batch and reduces the gradients."""

    def train_fn(cfg, model, optimizer, train_ids, use_ddp, use_tp):
        n = cfg.steps
        losses_log.append(n)
        if ddp_log is not None:
            ddp_log.append((use_ddp, use_tp))
        # Values deliberately land far below the fake evaluate_fn's range (900s) so an
        # accidental "val_loss = train_loss" bug is unmistakable in any assertion below.
        return [10.0 - 0.001 * i for i in range(n)], None

    return train_fn


def _fake_evaluate_fn(eval_calls, ddp_log=None):
    """Returns an evaluate_fn that records how many times it fired and returns a value
    from a numeric range disjoint from the fake train losses above.

    ``use_ddp`` is accepted (and optionally recorded in ``ddp_log``) because the real
    ``evaluate()`` needs it to shard the validation batch and compose the loss back off the
    mesh — see its docstring."""

    def evaluate_fn(model, val_ids, cfg, batches=10, use_ddp=False):
        eval_calls.append(cfg.steps)  # cfg.steps at call time, just to prove cfg flows through
        if ddp_log is not None:
            ddp_log.append(use_ddp)
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


# --- the ttml C++ header fields ------------------------------------------------------
#
# These four values are written into every checkpoint header because they exist nowhere
# else (see ttml_cxx_header_fields). intermediate_dim used to be a hardcoded 1024 at the
# call site, which is the value ttml *derives* for the 384 model -- so it was invisibly
# correct for the only size that had ever been trained, and silently wrong the first time
# --size 1024 ran. These tests pin the derivation to the registry so the constant cannot
# come back.


def test_ttml_cxx_header_fields_derives_intermediate_dim_per_size():
    """The bug this function exists to prevent: one size's derived FFN width written as a
    literal, and therefore wrong for every other size. 384 derives 1024 and 1024 derives
    2816 -- if these ever come out equal, the value is being restated, not derived."""
    f384 = ttml_cxx_header_fields(SIZES["384"])
    f1024 = ttml_cxx_header_fields(SIZES["1024"])
    assert f384["intermediate_dim"] == 1024
    assert f1024["intermediate_dim"] == 2816
    assert f384["intermediate_dim"] != f1024["intermediate_dim"]


@pytest.mark.parametrize("name", sorted(SIZES))
def test_ttml_cxx_header_fields_match_the_registry_for_every_size(name):
    """Every registered size, not just the two we happen to have trained: the header must
    agree with ModelSize.intermediate_dim, which tests/test_sizes.py in turn pins against
    the real converted model's config.json."""
    size = SIZES[name]
    assert ttml_cxx_header_fields(size)["intermediate_dim"] == size.intermediate_dim


@pytest.mark.parametrize("name", sorted(SIZES))
def test_ttml_cxx_header_fields_carries_the_non_derived_cxx_defaults(name):
    """weight_tying is the dangerous one -- a converter that doesn't see it True builds a
    randomly-initialized embedding table and raises nothing. These three are genuine C++
    constants (not size-dependent), so they are asserted the same for every size."""
    fields = ttml_cxx_header_fields(SIZES[name])
    assert fields["weight_tying"] is True
    assert fields["rms_norm_eps"] == 1e-5
    assert fields["weights_dtype"] == "bfloat16"


def test_ttml_cxx_header_fields_passes_validate_header():
    """The fields have to survive build_header's extra= and validate_header, or a
    checkpoint written with them cannot be loaded back."""
    header = build_header(
        step=1000, model_config_path="m.yaml", tokenizer_dir="tok",
        corpus_tokens=1_000, batch_size=64, seq_len=512,
        extra={"transformer_config": {}, **ttml_cxx_header_fields(SIZES["1024"])}, seed=0, tokens_dir="artifacts/tokens-test", optimizer={"type": "AdamW"}, ddp=1)
    validate_header(header)  # must not raise
    assert header["intermediate_dim"] == 2816


# ---------------------------------------------------------------------------
# DDP: use_ddp must reach BOTH train_fn and evaluate_fn, or not reach either
# ---------------------------------------------------------------------------


def test_use_ddp_defaults_off_and_reaches_neither(tmp_path):
    """The pre-2026-08-16 behaviour, pinned: with no ``use_ddp`` argument the loop calls
    ``train_fn(..., False, False)`` and ``evaluate_fn(use_ddp=False)``, so every measurement
    recorded before DDP existed still reproduces exactly."""
    cfg = _FakeCfg(steps=100)
    train_ddp, eval_ddp = [], []

    run_training_loop(
        cfg, model=None, optimizer=None, train_ids=None, val_ids=None,
        save_every=0, val_every=50, start_step=0,
        val_log_path=tmp_path / "val_losses.jsonl",
        train_fn=_fake_train_fn([], train_ddp),
        evaluate_fn=_fake_evaluate_fn([], eval_ddp),
        save_checkpoint_fn=None, print_fn=lambda *a, **k: None,
    )

    assert train_ddp == [(False, False), (False, False)]
    assert eval_ddp == [False, False]


def test_use_ddp_reaches_both_train_and_evaluate(tmp_path):
    """The guard against the silent-failure mode this whole feature is shaped around.

    ``use_ddp`` decides three separate things at once: whether ttml's ``train()`` shards the
    batch across the mesh, whether it calls ``synchronize_gradients`` at all, and whether
    ``evaluate()`` shards its own batch and composes the loss back. If the flag reached the
    trainer but not the evaluator the validation number would be silently computed over a
    quarter of the intended windows; if it reached neither while a ``[1,4]`` mesh was open,
    the run would be four replicas of a single-chip run. So the assertion is not "it was
    passed" but "it was passed to *both*"."""
    cfg = _FakeCfg(steps=100)
    train_ddp, eval_ddp = [], []

    run_training_loop(
        cfg, model=None, optimizer=None, train_ids=None, val_ids=None,
        save_every=0, val_every=50, start_step=0,
        val_log_path=tmp_path / "val_losses.jsonl",
        train_fn=_fake_train_fn([], train_ddp),
        evaluate_fn=_fake_evaluate_fn([], eval_ddp),
        save_checkpoint_fn=None, use_ddp=True, print_fn=lambda *a, **k: None,
    )

    # use_tp stays False in every case: TP shards the weights, which convert/ cannot read.
    assert train_ddp == [(True, False), (True, False)]
    assert eval_ddp == [True, True]


# ---------------------------------------------------------------------------
# Mesh graph descriptor selection
# ---------------------------------------------------------------------------


def test_no_descriptor_for_a_single_chip_run():
    """--ddp 1 arms no fabric, so it must not set TT_MESH_GRAPH_DESC_PATH at all. This is
    what keeps every pre-2026-08-16 measurement reproducible."""
    assert _mesh_graph_descriptor_path(1) is None
    assert _mesh_graph_descriptor_path(0) is None


@pytest.mark.parametrize("devices", [2, 4])
def test_vendored_descriptor_exists_and_declares_the_matching_shape(devices):
    """The descriptor's declared dims must equal the [1, N] mesh --ddp N opens.

    This is the assertion that matters: a descriptor describing the same *chips* in a
    different *shape* (tt-metal's p300_x2 declares [2, 2] for this box) does not fail -- the
    mesh opens, training runs, and the first gradient all-reduce hangs forever. There is no
    error to catch at runtime, so the shape is pinned here instead."""
    path = _mesh_graph_descriptor_path(devices)
    assert path is not None and path.is_file(), f"missing descriptor for {devices} devices"
    text = path.read_text(encoding="utf-8")
    assert f"dims: [ 1, {devices} ]" in text, (
        f"{path} must declare device_topology dims [1, {devices}] to match the mesh "
        f"--ddp {devices} opens"
    )


def test_unsupported_device_count_raises_rather_than_falling_back():
    """3 chips has no descriptor. Falling back to a mismatched one would hang, so the only
    safe behaviour is to refuse."""
    with pytest.raises(ValueError, match="no mesh graph descriptor"):
        _mesh_graph_descriptor_path(3)


def test_operator_supplied_descriptor_is_not_overridden(monkeypatch):
    """An explicitly exported TT_MESH_GRAPH_DESC_PATH wins; returning None tells the caller
    to leave the environment alone."""
    monkeypatch.setenv("TT_MESH_GRAPH_DESC_PATH", "/somewhere/custom.textproto")
    assert _mesh_graph_descriptor_path(4) is None


# ---------------------------------------------------------------------------
# --warm-start and the gate policies (2026-08-20)
# ---------------------------------------------------------------------------
# These are validated BEFORE the device opens, which is the whole point: the first
# version checked them inside the model-construction block, which --dry-run never
# reaches. The code carried a comment promising failure "in a second rather than
# after a mesh has opened" while a mistyped --warm-start path would in fact have
# cost a device open and a full expert build before surfacing.


def _cli(*argv):
    """Run main() with argv under --dry-run and return (exit code, stderr)."""
    import io
    import contextlib
    import train.run as run_mod

    err = io.StringIO()
    with mock.patch.object(sys, "argv", ["run.py", *argv, "--dry-run"]):
        with contextlib.redirect_stderr(err):
            try:
                rc = run_mod.main()
            except SystemExit as e:  # argparse
                rc = e.code
    return rc, err.getvalue()


def test_seeded_gate_requires_a_warm_start():
    """The load-bearing guard.

    A seeded gate scores hidden states against die-region directions. Against the
    random embeddings of a fresh model that encodes NOTHING, and the policy would
    silently degenerate into an unusual initialisation of `learned` -- which would
    look like a result.
    """
    rc, err = _cli("--size", "1024", "--steps", "10", "--moe", "--gate-policy", "seeded")
    assert rc == 2
    assert "requires --warm-start" in err


def test_frozen_gate_requires_a_warm_start():
    rc, err = _cli("--size", "1024", "--steps", "10", "--moe", "--gate-policy", "frozen")
    assert rc == 2
    assert "requires --warm-start" in err


def test_gate_policy_without_moe_is_refused_rather_than_ignored():
    """Silently accepting it would let a run believe it used the die when it did not."""
    rc, err = _cli("--size", "1024", "--steps", "10", "--gate-policy", "frozen",
                   "--warm-start", "latest")
    assert rc == 2
    assert "no effect without --moe" in err


def test_warm_start_and_resume_are_mutually_exclusive():
    """They mean different things and doing both would silently pick one."""
    rc, err = _cli("--size", "1024", "--steps", "10",
                   "--warm-start", "latest", "--resume", "latest")
    assert rc == 2
    assert "mutually exclusive" in err


def test_a_missing_warm_start_checkpoint_fails_before_the_device():
    rc, err = _cli("--size", "1024", "--steps", "10", "--warm-start", "/nope/missing.pkl")
    assert rc == 2
    assert "no checkpoint to warm-start from" in err


def test_a_checkpoint_dir_as_gate_reference_is_refused_with_the_reason():
    """The gate reference must be a CONVERTED artifact, and the error should say so."""
    rc, err = _cli("--size", "1024", "--steps", "10", "--moe",
                   "--reference-hf-dir", "artifacts")
    assert rc == 2
    assert "model.safetensors" in err and "CONVERTED" in err


def test_a_valid_moe_warm_start_combination_is_accepted():
    """The guards must not reject the thing the experiment actually runs."""
    ckpt = ROOT / "artifacts" / "checkpoints-v077-beta2-control" / "tt_tnt_step00010764.pkl"
    if not ckpt.is_file():
        pytest.skip("needs gitignored artifacts: the beta2-control checkpoint")
    rc, err = _cli("--size", "1024", "--steps", "10", "--moe",
                   "--gate-policy", "seeded", "--warm-start", str(ckpt))
    assert rc in (0, None), err
