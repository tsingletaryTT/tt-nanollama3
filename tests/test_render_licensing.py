# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The licensing section is GENERATED so it cannot drift from the registry.

This repo has twice shipped documentation contradicting the facts. A rendered section
cannot go stale; a hand-written one always eventually does.
"""
from train.corpus import SOURCES
from scripts.render_licensing import render_licensing


def test_every_source_appears_with_its_licence():
    out = render_licensing()
    for name, src in SOURCES.items():
        assert name in out, f"{name} missing from the rendered licensing"
        assert src.license_id in out, f"{name}'s licence {src.license_id!r} missing"


def test_share_alike_sources_are_called_out():
    out = render_licensing()
    for src in SOURCES.values():
        if src.share_alike:
            assert src.attribution in out, f"{src.name} needs attribution rendered"
    assert "share-alike" in out.lower()


def test_states_that_the_corpus_is_not_redistributed():
    assert "not redistribute" in render_licensing().lower()


def test_states_the_weights_question_is_unsettled():
    out = render_licensing().lower()
    assert "unsettled" in out or "do not assert" in out


def test_no_source_is_silently_omitted():
    """A source added to the registry without a licence must break this, not slip through."""
    out = render_licensing()
    assert out.count("| ") >= len(SOURCES)


def test_a_fractional_share_renders_with_its_fraction_intact():
    """``:.0%`` rendered flavour's 0.5% as **0%** and spine's 13.5% as 14%. "0%" reads as
    "contributes nothing", in the one document whose banner promises it cannot go stale."""
    out = render_licensing()
    assert "| 0.5% |" in out, "flavour's 0.5% share must not round to 0%"
    assert "| 13.5% |" in out, "spine's 13.5% share must not round to 14%"
    assert "| 0% |" not in out, "no source contributes nothing"


def test_whole_percentages_are_not_padded_with_a_pointless_decimal():
    """Keeping fractions must not make the common case noisier."""
    out = render_licensing()
    assert "| 31% |" in out and "| 31.0% |" not in out
