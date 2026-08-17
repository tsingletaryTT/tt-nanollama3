# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""scripts/probe_core_prng.py: the analysis half, exercised without hardware.

The probe itself needs a device, but the part that decides what we are allowed to
claim -- summarise() and verdict() -- is pure array reduction. That logic is
exactly where a wrong call would be expensive: a verdict that says "intrinsic
core identity" when the cores merely got different seeds would send the whole
sampler design down a false path. So it is tested against synthetic draws whose
answer is known by construction.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location(
        "probe_core_prng", REPO_ROOT / "scripts" / "probe_core_prng.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _draws(num_cores: int, distinct: bool, rng_seed: int = 0) -> np.ndarray:
    """Synthetic (cores, tiles, 32, 32) draws in [0, 1)."""
    rng = np.random.default_rng(rng_seed)
    if distinct:
        return rng.random((num_cores, 1, 32, 32)).astype(np.float32)
    one = rng.random((1, 1, 32, 32)).astype(np.float32)
    return np.repeat(one, num_cores, axis=0)


def test_identical_runs_are_reported_reproducible(probe):
    run = _draws(4, distinct=True)
    summary = probe.summarise("distinct", run, run.copy())
    assert summary["reproducible_across_runs"] is True


def test_differing_runs_are_not_reported_reproducible(probe):
    summary = probe.summarise("distinct", _draws(4, True, 1), _draws(4, True, 2))
    assert summary["reproducible_across_runs"] is False


def test_cores_sharing_one_stream_count_as_identical(probe):
    """The decisive measurement: same stream on every core must not read as distinct."""
    run = _draws(6, distinct=False)
    summary = probe.summarise("identical", run, run.copy())
    assert summary["cores_identical_to_core0"] == 6
    assert summary["cores_distinct_from_core0"] == 0


def test_distinct_cores_are_counted(probe):
    run = _draws(5, distinct=True)
    summary = probe.summarise("distinct", run, run.copy())
    # Core 0 always "matches" itself, so 4 of 5 differ.
    assert summary["cores_distinct_from_core0"] == 4


def test_unit_interval_check_catches_out_of_range_draws(probe):
    run = _draws(3, distinct=True)
    run[0, 0, 0, 0] = 1.5
    summary = probe.summarise("distinct", run, run.copy())
    assert summary["draws_in_unit_interval"] is False


def test_verdict_irreproducible_beats_every_other_finding(probe):
    """If fixed seeds do not replay, nothing else in the run is worth claiming."""
    distinct = {"reproducible_across_runs": False}
    identical = {
        "reproducible_across_runs": False,
        "cores_identical_to_core0": 0,
        "cores_distinct_from_core0": 8,
        "num_cores": 8,
    }
    assert probe.verdict(distinct, identical, {}).startswith("NOT REPRODUCIBLE")


def test_verdict_no_intrinsic_identity_when_same_seed_agrees(probe):
    distinct = {"reproducible_across_runs": True}
    identical = {
        "reproducible_across_runs": True,
        "cores_identical_to_core0": 8,
        "cores_distinct_from_core0": 0,
        "num_cores": 8,
    }
    assert probe.verdict(distinct, identical, {}).startswith("NO INTRINSIC CORE IDENTITY")


def test_verdict_intrinsic_identity_when_same_seed_diverges_stably(probe):
    distinct = {"reproducible_across_runs": True}
    identical = {
        "reproducible_across_runs": True,
        "cores_identical_to_core0": 1,
        "cores_distinct_from_core0": 7,
        "num_cores": 8,
    }
    assert probe.verdict(distinct, identical, {}).startswith("INTRINSIC CORE IDENTITY")


def test_verdict_intrinsic_but_unstable(probe):
    distinct = {"reproducible_across_runs": True}
    identical = {
        "reproducible_across_runs": False,
        "cores_identical_to_core0": 1,
        "cores_distinct_from_core0": 7,
        "num_cores": 8,
    }
    assert probe.verdict(distinct, identical, {}).startswith("INTRINSIC BUT UNSTABLE")
