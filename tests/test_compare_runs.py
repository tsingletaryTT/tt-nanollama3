# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for scripts/compare_runs.py, written from the three failures it exists to stop.

Each test below reproduces something that actually happened on 2026-08-19 and
asserts the tool now refuses it. That is deliberate: a test written from
imagination checks what I thought of, and every one of today's four instrument
failures was something I had not thought of -- including one guarded by a passing
test that sat one layer below the bug.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_runs.py"


def _run_dir(tmp: Path, name: str, curve: dict, manifest: dict | None = None) -> Path:
    d = tmp / name
    d.mkdir(parents=True)
    (d / "val_losses.jsonl").write_text(
        "\n".join(json.dumps({"step": s, "val_loss": v}) for s, v in sorted(curve.items()))
    )
    if manifest is not None:
        (d / "run_manifest.json").write_text(json.dumps(manifest))
    return d


def _manifest(**over) -> dict:
    base = {
        "tokens_dir": "artifacts/tokens-v4", "train_tokens": 352641058,
        "val_tokens": 39182335, "seq_len": 512, "batch_size": 64,
        "gradient_accumulation_steps": 1, "ddp": 4, "size": "1024",
        "optimizer": {"beta2": 0.95}, "tt_metal": "v0.77.0", "seed": 5489,
        "lr_schedule": "constant", "warmup_frac": 0.02,
        "optimizer_override_file": "train/configs/tt-tnt-v077.yaml",
    }
    base.update(over)
    return base


def _invoke(*argv):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, argv)],
                          capture_output=True, text=True)


# --- failure 1: comparing runs trained on different corpora ------------------

def test_refuses_when_tokens_dir_differs(tmp_path):
    """The 1.3-nat false regression. This must be a hard failure, not a warning.

    A warning would not have helped: the original mistake was made while reading
    output attentively. Only a refusal stops it.
    """
    a = _run_dir(tmp_path, "a", {1000: 5.9, 2000: 5.1}, _manifest())
    b = _run_dir(tmp_path, "b", {1000: 4.1, 2000: 3.5},
                 _manifest(tokens_dir="artifacts/tokens", train_tokens=353495970))
    r = _invoke(a, b)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "REFUSING TO COMPARE" in r.stderr
    assert "tokens_dir" in r.stderr


def test_allow_mismatch_is_an_explicit_opt_in(tmp_path):
    a = _run_dir(tmp_path, "a", {1000: 5.9}, _manifest())
    b = _run_dir(tmp_path, "b", {1000: 4.1}, _manifest(tokens_dir="artifacts/tokens"))
    r = _invoke(a, b, "--allow-mismatch")
    assert r.returncode == 0, r.stderr
    assert "COMPARING ANYWAY" in r.stdout


def test_ddp_and_batch_differences_also_block(tmp_path):
    """Effective batch changes the curve as surely as the corpus does."""
    a = _run_dir(tmp_path, "a", {1000: 4.0}, _manifest())
    b = _run_dir(tmp_path, "b", {1000: 4.0}, _manifest(ddp=1, batch_size=16))
    assert _invoke(a, b).returncode == 2


# --- failure 3: a truncated baseline -----------------------------------------

def test_unmatched_steps_are_always_reported(tmp_path):
    """The head -8 failure. Unmatched steps must be visible, never implied.

    The original error was invisible precisely because nothing said "the baseline
    has three more points than you looked at".
    """
    a = _run_dir(tmp_path, "a", {1000: 4.0, 2000: 3.5, 3000: 3.2}, _manifest())
    b = _run_dir(tmp_path, "b", {1000: 4.1, 2000: 3.6}, _manifest())
    r = _invoke(a, b)
    assert r.returncode == 0, r.stderr
    assert "matched       : 2" in r.stdout
    assert "only in A     : [3000]" in r.stdout


def test_reads_every_point_of_a_long_curve(tmp_path):
    """No implicit limit anywhere in the read path."""
    curve_a = {s: 4.0 - s / 100000 for s in range(1000, 12000, 1000)}
    curve_b = {s: 4.0 - s / 100000 for s in range(1000, 12000, 1000)}
    a = _run_dir(tmp_path, "a", curve_a, _manifest())
    b = _run_dir(tmp_path, "b", curve_b, _manifest())
    r = _invoke(a, b)
    assert "matched       : 11" in r.stdout, r.stdout


# --- the floor, and NOT INTERPRETABLE as a real verdict ----------------------

def test_small_difference_is_not_interpretable(tmp_path):
    """A wobbling pair with a tiny mean must NOT be reported as better or worse.

    Today a +0.03 delta was labelled 'worse' by a monitor whose threshold was
    tighter than the runs' own scatter, and that label started accumulating into
    a conclusion.
    """
    a = _run_dir(tmp_path, "a", {1000: 4.0, 2000: 3.5, 3000: 3.6, 4000: 3.1}, _manifest())
    b = _run_dir(tmp_path, "b", {1000: 4.02, 2000: 3.47, 3000: 3.63, 4000: 3.09}, _manifest())
    r = _invoke(a, b)
    assert "NOT INTERPRETABLE" in r.stdout, r.stdout


def test_large_consistent_difference_is_reported(tmp_path):
    """The floor must not swallow a real effect."""
    a = _run_dir(tmp_path, "a", {1000: 3.0, 2000: 2.9, 3000: 2.8, 4000: 2.7}, _manifest())
    b = _run_dir(tmp_path, "b", {1000: 4.0, 2000: 3.9, 3000: 3.8, 4000: 3.7}, _manifest())
    r = _invoke(a, b)
    assert "better (A lower)" in r.stdout, r.stdout


def test_missing_manifest_warns_loudly_and_still_compares(tmp_path):
    """Pre-manifest runs must be usable, but must announce that they cannot prove it."""
    a = _run_dir(tmp_path, "a", {1000: 4.0, 2000: 3.5}, _manifest())
    b = _run_dir(tmp_path, "b", {1000: 4.1, 2000: 3.6})  # no manifest
    r = _invoke(a, b)
    assert r.returncode == 0, r.stderr
    assert "NO run_manifest.json" in r.stdout
    assert "CANNOT be verified" in r.stdout


def test_empty_and_missing_curves_fail_cleanly(tmp_path):
    a = _run_dir(tmp_path, "a", {1000: 4.0}, _manifest())
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "val_losses.jsonl").write_text("")
    assert _invoke(a, empty).returncode != 0
    assert _invoke(a, tmp_path / "nonexistent").returncode != 0


def test_no_overlap_fails_rather_than_reporting_nothing(tmp_path):
    a = _run_dir(tmp_path, "a", {1000: 4.0}, _manifest())
    b = _run_dir(tmp_path, "b", {1500: 4.0}, _manifest())
    r = _invoke(a, b)
    assert r.returncode != 0
    assert "no matched steps" in (r.stdout + r.stderr)
