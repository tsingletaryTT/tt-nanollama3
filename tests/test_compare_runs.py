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


# --- paired vs unpaired ------------------------------------------------------
# Added after the tool's FIRST REAL USE returned a wrong verdict. A paired
# comparison with mean -0.0482, se 0.0081 (six standard errors) and 19 of 22
# signs agreeing was stamped NOT INTERPRETABLE, because |mean| fell under the
# within-run wobble floor. The floor was the wrong instrument for the design:
# the two runs shared a seed, so their oscillation is common to both curves and
# cancels in the delta, which is exactly why se was an order of magnitude below
# the wobble. Judging a paired design by an unpaired floor throws away most of
# its sensitivity.


def _paired_curves(tmp_path, offset: float, seed_a=5489, seed_b=5489, jitter=0.012):
    """Two curves that oscillate TOGETHER, offset by *offset* plus small jitter.

    This is what same-seed runs look like: identical data order, so a large shared
    wobble, with the treatment effect as a small shift underneath it and a little
    independent scatter on top.

    The jitter is not decoration. Without it the deltas are mathematically
    identical, sd is exactly 0, and the test exercises a degenerate branch instead
    of the path real curves take. The first version of these tests made that
    mistake and "failed" against a tool that was behaving correctly.
    """
    wobbly = {s: 3.0 + (0.15 if (s // 500) % 2 else -0.15) for s in range(500, 11000, 500)}
    a = {s: v + offset + (jitter if (s // 500) % 3 == 0 else -jitter / 2)
         for s, v in wobbly.items()}
    return (_run_dir(tmp_path, "a", a, _manifest(seed=seed_a)),
            _run_dir(tmp_path, "b", wobbly, _manifest(seed=seed_b)))


def test_paired_detects_a_small_consistent_offset_the_floor_would_hide(tmp_path):
    """The regression. A steady -0.05 under a +/-0.15 wobble must be found."""
    a, b = _paired_curves(tmp_path, offset=-0.05)
    r = _invoke(a, b)
    assert "design            PAIRED" in r.stdout, r.stdout
    assert "better (A lower)" in r.stdout, r.stdout
    assert "standard errors from zero" in r.stdout


def test_pairing_is_auto_detected_from_matching_seeds(tmp_path):
    a, b = _paired_curves(tmp_path, offset=-0.05)
    r = _invoke(a, b)
    assert "auto: both manifests report seed 5489" in r.stdout, r.stdout


def test_different_seeds_fall_back_to_the_conservative_floor(tmp_path):
    """Different data order means the oscillations do not cancel; stay conservative."""
    a, b = _paired_curves(tmp_path, offset=-0.05, seed_a=1, seed_b=2)
    r = _invoke(a, b)
    assert "design            UNPAIRED" in r.stdout, r.stdout
    assert "NOT INTERPRETABLE" in r.stdout


def test_missing_manifest_falls_back_to_unpaired(tmp_path):
    """Cannot confirm the seeds match, so must not claim the sensitivity."""
    wobbly = {s: 3.0 + (0.15 if (s // 500) % 2 else -0.15) for s in range(500, 11000, 500)}
    a = _run_dir(tmp_path, "a", {s: v - 0.05 for s, v in wobbly.items()}, _manifest())
    b = _run_dir(tmp_path, "b", wobbly)  # no manifest
    r = _invoke(a, b)
    assert "design            UNPAIRED" in r.stdout, r.stdout


def test_paired_still_reports_not_interpretable_when_it_should(tmp_path):
    """Pairing raises sensitivity; it must not manufacture findings from nothing."""
    a, b = _paired_curves(tmp_path, offset=0.0005)
    r = _invoke(a, b)
    assert "NOT INTERPRETABLE" in r.stdout, r.stdout


def test_paired_can_be_forced_and_overrides_autodetect(tmp_path):
    a, b = _paired_curves(tmp_path, offset=-0.05, seed_a=1, seed_b=2)
    r = _invoke(a, b, "--paired")
    assert "forced on the command line" in r.stdout
    assert "better (A lower)" in r.stdout


def test_zero_scatter_is_reported_without_dividing_by_zero(tmp_path):
    """A perfectly constant offset has se == 0 and no t statistic.

    Degenerate, but it must not crash and must not be silently swallowed: the
    answer is obvious and should be stated as what it is.
    """
    a, b = _paired_curves(tmp_path, offset=-0.05, jitter=0.0)
    r = _invoke(a, b)
    assert r.returncode == 0, r.stderr
    assert "constant offset" in r.stdout, r.stdout
    assert "better (A lower)" in r.stdout


def test_identical_curves_are_not_interpretable(tmp_path):
    a, b = _paired_curves(tmp_path, offset=0.0, jitter=0.0)
    r = _invoke(a, b)
    assert "identical" in r.stdout, r.stdout


# ------------------------------------------------------- the MoE variables get NAMED
#
# A dense-vs-MoE comparison used to print no mention of MoE at all: `moe` is (correctly)
# absent from COMPARABILITY_KEYS, so it was neither flagged as a mismatch nor reported as
# the variable under test, and the output read as though the arms were configured alike.

def test_moe_difference_is_reported_as_the_variable_under_test(tmp_path):
    dense = _run_dir(tmp_path, "A-dense", {1000: 3.0, 2000: 2.9},
                     _manifest(moe=None))
    moe = _run_dir(tmp_path, "B-learned", {1000: 3.1, 2000: 2.95},
                   _manifest(moe={"experts": 10, "top_k": 2, "gate_policy": "learned"}))
    out = _invoke(dense, moe, "--paired").stdout
    assert "variable under test -- moe" in out, out
    assert "gate_policy" in out, "the arm's gate policy must appear in the report"


def test_two_moe_arms_differing_only_in_gate_policy_are_still_named(tmp_path):
    """seeded-vs-frozen is the comparison the die-region claim rests on."""
    a = _run_dir(tmp_path, "C-seeded", {1000: 3.0},
                 _manifest(moe={"experts": 10, "gate_policy": "seeded"}))
    b = _run_dir(tmp_path, "D-frozen", {1000: 3.05},
                 _manifest(moe={"experts": 10, "gate_policy": "frozen"}))
    out = _invoke(a, b, "--paired").stdout
    assert "variable under test -- moe" in out
    assert "seeded" in out and "frozen" in out


def test_identical_moe_config_is_not_reported_as_a_variable(tmp_path):
    """Guards the other direction: same config must not be announced as differing."""
    cfg = {"experts": 10, "gate_policy": "seeded"}
    a = _run_dir(tmp_path, "one", {1000: 3.0}, _manifest(moe=cfg))
    b = _run_dir(tmp_path, "two", {1000: 3.0}, _manifest(moe=dict(cfg)))
    assert "variable under test -- moe" not in _invoke(a, b, "--paired").stdout


def test_warm_start_is_reported_not_enforced(tmp_path):
    """Arms warm-started from ONE checkpoint still have differing summaries.

    A dense arm copies every parameter; an MoE arm copies only the shared ones. Putting
    `warm_start` in COMPARABILITY_KEYS would therefore refuse a perfectly valid comparison,
    so it is reported instead — the reader checks the source checkpoint agrees.
    """
    ck = "artifacts/checkpoints-v077-beta2-control/tt_tnt_step00010764.pkl"
    a = _run_dir(tmp_path, "dense", {1000: 3.0},
                 _manifest(warm_start={"source": ck, "copied": 204}))
    b = _run_dir(tmp_path, "moe", {1000: 3.1},
                 _manifest(warm_start={"source": ck, "copied": 48}))
    out = _invoke(a, b, "--paired").stdout
    assert "REFUSING TO COMPARE" not in out, "same-checkpoint arms must remain comparable"
    assert "variable under test -- warm_start" in out
