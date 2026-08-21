# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Statistics tests for the stage-1 evaluation: Bonferroni threshold and the two verdicts.

See scripts/eval_improv.py's module docstring for the CONTROLLER RULING that overrides the
original task-6 brief's `paired_verdict`: a zero-scatter (sd == 0, mean != 0) series is a
PERFECT separation, not an uninterpretable one, and must not collapse t to 0.

FIX 2 (task-6-report.md): `paired_verdict` originally labelled `mean < 0` as "think
better" unconditionally, which is backwards for higher-is-better scorers (`groundedness`,
`affordance`). `direction` is now a required argument and the tests below assert WHICH
arm a significant verdict names for BOTH a lower-is-better and a higher-is-better scorer
-- the previous version of this file only checked that a clear separation was not
reported as "NOT INTERPRETABLE", which is exactly why the inverted-polarity bug passed
28/28 tests before this fix.
"""
from scripts.eval_improv import BONFERRONI_ALPHA, SCORER_DIRECTIONS, paired_verdict, swap_verdict


def test_bonferroni_threshold_is_stated_for_five_tests():
    """Four scorers plus adherence. Uncorrected 0.05 would read three nulls as real."""
    assert abs(BONFERRONI_ALPHA - 0.01) < 1e-9


def test_identical_curves_are_not_interpretable():
    a = [1.0] * 10
    assert paired_verdict(a, list(a), direction="lower")["verdict"] == "NOT INTERPRETABLE"


def test_a_clear_separation_is_reported():
    a = [1.0] * 10
    b = [2.0] * 10
    assert paired_verdict(a, b, direction="lower")["verdict"] != "NOT INTERPRETABLE"


def test_lower_is_better_scorer_names_think_when_think_scores_lower():
    """escalation/new_harm-style scorer: think's scores (a) are lower than no-think's (b)
    -- for a lower-is-better scorer that means THINK should be named as favoured."""
    a = [1.0] * 10
    b = [2.0] * 10
    v = paired_verdict(a, b, direction="lower")
    assert v["verdict"] == "think better"


def test_higher_is_better_scorer_names_think_when_think_scores_higher():
    """groundedness/affordance-style scorer: think's scores (a) are HIGHER than
    no-think's (b) -- for a higher-is-better scorer that means THINK should be named as
    favoured. Under the pre-fix (uniform mean<0-means-think) logic this exact case would
    have been mislabelled "no-think better" (mean_delta is positive here)."""
    a = [2.0] * 10
    b = [1.0] * 10
    v = paired_verdict(a, b, direction="higher")
    assert v["verdict"] == "think better"


def test_direction_flips_the_verdict_for_identical_data():
    """The same delta series must name the OPPOSITE arm depending on direction -- this is
    the exact shape of bug FIX 2 fixes: labelling a higher-is-better scorer as if lower
    were better (or vice versa) silently inverts which arm 'wins'."""
    a = [2.0] * 10
    b = [1.0] * 10
    assert paired_verdict(a, b, direction="higher")["verdict"] == "think better"
    assert paired_verdict(a, b, direction="lower")["verdict"] == "no-think better"


def test_all_four_scorer_directions_are_declared_explicitly():
    assert SCORER_DIRECTIONS == {
        "escalation": "lower",
        "new_harm": "lower",
        "groundedness": "higher",
        "affordance": "higher",
    }


def test_signs_zero_counts_ties_separately_from_signs_neg():
    """FIX 3(a): a tie (delta == 0) must not be folded into signs_neg -- a scorer that is
    saturated (every pair identical) should report signs_zero == n, not signs_neg == n,
    which previously read as 'no-think won every pair' when nothing had differed at all.
    """
    a = [1.0, 1.0, 2.0]
    b = [1.0, 1.0, 1.0]
    v = paired_verdict(a, b, direction="lower")
    assert v["signs_zero"] == 2
    assert v["signs_pos"] == 1
    assert v["signs_neg"] == 0


def test_swap_verdict_fails_when_output_is_invariant():
    """If swapping the think-block changes nothing, the thinking is DECORATIVE."""
    res = swap_verdict(divergence_positions=[None] * 50, n=50)
    assert res["thinking_is_load_bearing"] is False


def test_swap_verdict_passes_when_output_moves():
    res = swap_verdict(divergence_positions=[3] * 50, n=50)
    assert res["thinking_is_load_bearing"] is True
