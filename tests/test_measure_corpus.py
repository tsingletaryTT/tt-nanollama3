# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The scarcity gate.

Pure arithmetic over measured token counts. The point is that a slice which cannot be filled
within its upsample cap is reported BEFORE ratios are committed, rather than discovered when
a training run produces a model dominated by whatever was actually plentiful.
"""
import json

from train.corpus import CorpusSource
from scripts.measure_corpus import (
    achievable_tokens,
    required_tokens,
    shortfall_report,
    Shortfall,
    _json_safe_shortfall,
)


def _src(name, share, upsample=1):
    return CorpusSource(name=name, slice="spine", target_share=share,
                        hf_repo="r", hf_revision="rev", upsample=upsample)


def test_required_tokens_is_share_of_budget():
    assert required_tokens(_src("a", 0.25), 400_000_000) == 100_000_000


def test_achievable_tokens_multiplies_by_upsample():
    assert achievable_tokens(1_000_000, upsample=4) == 4_000_000


def test_no_shortfall_when_supply_meets_demand(monkeypatch):
    sources = {"a": _src("a", 0.5), "b": _src("b", 0.5)}
    monkeypatch.setattr("scripts.measure_corpus.SOURCES", sources)
    available = {"a": 60_000_000, "b": 60_000_000}
    assert shortfall_report(available, total_budget=100_000_000, upsample_cap=8) == []


def test_shortfall_detected_when_supply_is_short(monkeypatch):
    sources = {"a": _src("a", 0.5), "b": _src("b", 0.5)}
    monkeypatch.setattr("scripts.measure_corpus.SOURCES", sources)
    available = {"a": 1_000_000, "b": 60_000_000}
    report = shortfall_report(available, total_budget=100_000_000, upsample_cap=8)
    assert [s.name for s in report] == ["a"]
    assert report[0].required == 50_000_000
    assert report[0].available == 1_000_000
    # 1M * cap 8 = 8M, still short of 50M
    assert report[0].needed_upsample > 8


def test_shortfall_respects_a_sources_own_upsample(monkeypatch):
    """A source already upsampled 4x needs proportionally less raw material."""
    sources = {"a": _src("a", 0.5, upsample=4), "b": _src("b", 0.5)}
    monkeypatch.setattr("scripts.measure_corpus.SOURCES", sources)
    available = {"a": 20_000_000, "b": 60_000_000}
    assert shortfall_report(available, total_budget=100_000_000, upsample_cap=8) == []


def test_missing_source_counts_as_zero_available(monkeypatch):
    sources = {"a": _src("a", 1.0)}
    monkeypatch.setattr("scripts.measure_corpus.SOURCES", sources)
    report = shortfall_report({}, total_budget=1_000_000, upsample_cap=8)
    assert report[0].available == 0


def test_needed_upsample_is_infinite_for_zero_availability(monkeypatch):
    sources = {"a": _src("a", 1.0)}
    monkeypatch.setattr("scripts.measure_corpus.SOURCES", sources)
    report = shortfall_report({}, total_budget=1_000_000, upsample_cap=8)
    assert report[0].needed_upsample == float("inf")


def test_json_safe_shortfall_emits_valid_json_for_infinite_upsample():
    """json.dumps(float('inf')) writes a bare `Infinity` token, invalid per RFC 8259."""
    s = Shortfall(name="a", required=1_000_000, available=0,
                  current_upsample=1, needed_upsample=float("inf"))
    encoded = json.dumps(_json_safe_shortfall(s))
    assert "Infinity" not in encoded
    assert json.loads(encoded)["needed_upsample"] is None


def test_json_safe_shortfall_preserves_finite_values():
    s = Shortfall(name="a", required=1_000_000, available=500_000,
                  current_upsample=1, needed_upsample=2.0)
    assert _json_safe_shortfall(s)["needed_upsample"] == 2.0


def test_gate_row_renders_a_fractional_share_intact():
    """The operator settles the shares against this table. ``:.0%`` printed flavour's
    0.5% as 0%, i.e. as a slice contributing nothing."""
    from scripts.measure_corpus import gate_row
    from train.corpus import SOURCES
    row = gate_row(SOURCES["flavour"], required=2_000_000, available=575_377,
                   method="tokenizer")
    assert "0.5%" in row
    assert " 0% " not in row


def test_gate_row_reports_the_measurement_method():
    """A tokenizer count and a word approximation are not interchangeable evidence."""
    from scripts.measure_corpus import gate_row
    from train.corpus import SOURCES
    assert gate_row(SOURCES["spine"], 54_000_000, 26_200_908, "approx").endswith("approx")
    assert "13.5%" in gate_row(SOURCES["spine"], 54_000_000, 26_200_908, "tokenizer")
