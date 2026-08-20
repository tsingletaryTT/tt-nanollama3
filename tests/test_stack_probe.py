# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for scripts/stack_probe.py.

Two jobs. First, the probe's SOURCE-LEVEL checks run anywhere, so CI can enforce them
permanently — no device, no instance registry, no installed bundle. Second, guard the
probe itself: an instrument that cannot fail is not measuring anything, and one that
reports PASS when it broke is worse than none.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("stack_probe", ROOT / "scripts" / "stack_probe.py")
stack_probe = importlib.util.module_from_spec(_spec)
sys.modules["stack_probe"] = stack_probe
_spec.loader.exec_module(stack_probe)

PASS, FAIL, SKIP = stack_probe.PASS, stack_probe.FAIL, stack_probe.SKIP


# ------------------------------------------------ the host-independent checks

def test_val_every_flag_stays_wired():
    """CI-enforceable. A whole experiment once ran with validation logging silently off."""
    r = stack_probe.check_val_curve_flag_is_wired()
    assert r.status == PASS, r.detail


def test_checkpoint_format_keeps_recording_provenance():
    """CI-enforceable. Dropping any of these reopens the phantom-regression trap."""
    r = stack_probe.check_checkpoints_record_their_corpus()
    assert r.status == PASS, r.detail


# ------------------------------------------------ guarding the instrument itself

def test_every_check_declares_the_failure_it_exists_for():
    """A check with no observed failure behind it is a claim, not a check.

    This is the file's design rule, so it is asserted rather than trusted: each check
    must name the PR or date of the failure it was written for.
    """
    checks = [
        stack_probe.check_instance_version_matches_its_tree,
        stack_probe.check_toolchain_agrees_with_the_instance,
        stack_probe.check_installed_bundles_point_somewhere_real,
        stack_probe.check_vllm_bundle_pins_an_interpreter_with_vllm,
        stack_probe.check_val_curve_flag_is_wired,
        stack_probe.check_checkpoints_record_their_corpus,
    ]
    for fn in checks:
        r = fn()
        assert r.regression_of, f"{fn.__name__} names no observed failure"
        assert r.status in (PASS, FAIL, SKIP)


def test_a_raising_check_becomes_FAIL_not_a_silent_pass(monkeypatch, capsys):
    """The most important property. A broken instrument must not read as a clean run."""
    def exploding():
        raise RuntimeError("probe itself is broken")

    monkeypatch.setattr(stack_probe, "check_val_curve_flag_is_wired", exploding)
    monkeypatch.setattr(sys, "argv", ["stack_probe"])
    with pytest.raises(SystemExit) as e:
        stack_probe.main()
    out = capsys.readouterr().out
    assert e.value.code == 1, "a raising check must make the run exit non-zero"
    assert "check itself raised" in out


def test_skips_are_enumerated_in_the_summary(monkeypatch, capsys):
    """A summary that counts only what it ran is measuring itself."""
    monkeypatch.setattr(stack_probe, "check_val_curve_flag_is_wired",
                        lambda: stack_probe.Result("faked", SKIP, "deliberately skipped",
                                                   "test"))
    monkeypatch.setattr(sys, "argv", ["stack_probe"])
    with pytest.raises(SystemExit):
        stack_probe.main()
    out = capsys.readouterr().out
    assert "skipped: faked" in out, "a skipped check must be named, not just counted"


def test_unreachable_paths_are_reported_as_such_not_as_absent(tmp_path, monkeypatch):
    """An unprobeable thing is unknown, never a confident 'not installed'.

    Reporting "vllm not found" for an interpreter we merely failed to run would send
    someone to reinstall a stack that is already present.
    """
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    for fn in (stack_probe.check_installed_bundles_point_somewhere_real,
               stack_probe.check_vllm_bundle_pins_an_interpreter_with_vllm,
               stack_probe.check_toolchain_agrees_with_the_instance,
               stack_probe.check_instance_version_matches_its_tree):
        r = fn()
        assert r.status == SKIP, f"{fn.__name__} claimed {r.status} with no data: {r.detail}"


def test_device_check_skips_without_a_lease(monkeypatch):
    """It must never open a device just because someone ran the probe."""
    monkeypatch.delenv("TT_VISIBLE_DEVICES", raising=False)
    r = stack_probe.check_device_training_smoke(steps=1)
    assert r.status == SKIP
    assert "lease" in r.detail
