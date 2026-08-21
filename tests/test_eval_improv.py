# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Statistics tests for the stage-1 evaluation: Bonferroni threshold and the two verdicts.

See scripts/eval_improv.py's module docstring for the CONTROLLER RULING that overrides the
original task-6 brief's `paired_verdict`: a zero-scatter (sd == 0, mean != 0) series is a
PERFECT separation, not an uninterpretable one, and must not collapse t to 0.
"""
from scripts.eval_improv import BONFERRONI_ALPHA, paired_verdict, swap_verdict


def test_bonferroni_threshold_is_stated_for_five_tests():
    """Four scorers plus adherence. Uncorrected 0.05 would read three nulls as real."""
    assert abs(BONFERRONI_ALPHA - 0.01) < 1e-9


def test_identical_curves_are_not_interpretable():
    a = [1.0] * 10
    assert paired_verdict(a, list(a))["verdict"] == "NOT INTERPRETABLE"


def test_a_clear_separation_is_reported():
    a = [1.0] * 10
    b = [2.0] * 10
    assert paired_verdict(a, b)["verdict"] != "NOT INTERPRETABLE"


def test_swap_verdict_fails_when_output_is_invariant():
    """If swapping the think-block changes nothing, the thinking is DECORATIVE."""
    res = swap_verdict(divergence_positions=[None] * 50, n=50)
    assert res["thinking_is_load_bearing"] is False


def test_swap_verdict_passes_when_output_moves():
    res = swap_verdict(divergence_positions=[3] * 50, n=50)
    assert res["thinking_is_load_bearing"] is True
