# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""docs/corpus_blend.md is the human-readable half of the blend's provenance record.

The machine-readable half (docs/measurements/blend_manifest.json) is written by the blend
itself and cannot lie. The prose is transcribed, so it can -- and this project has shipped
documentation contradicting reality three times on this branch alone. These tests hold the
transcription to the manifest, which is the only reason the page is allowed to quote
figures at all rather than just linking to the JSON.
"""
import json
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
PAGE = DOCS / "corpus_blend.md"
RECORD = DOCS / "measurements" / "blend_manifest.json"


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _record() -> dict:
    return json.loads(RECORD.read_text())


def test_every_source_is_listed_with_its_real_token_count():
    page, rec = _page(), _record()
    for name, s in rec["sources"].items():
        assert f"`{name}`" in page, f"{name} missing from {PAGE.name}"
        assert f"{s['emitted_tokens']:,}" in page, (
            f"{name}'s emitted token count in {PAGE.name} is not the manifest's "
            f"{s['emitted_tokens']:,}"
        )


def test_the_real_repetition_factors_are_quoted_as_recorded():
    """The declared upsample is a ceiling; the number a reader needs is what actually
    happened, which is fractional and is not in train/corpus.py."""
    page, rec = _page(), _record()
    for name, s in rec["sources"].items():
        assert f"{s['repetition_factor']}x" in page, (
            f"{name}'s real repetition {s['repetition_factor']}x missing from {PAGE.name}"
        )


def test_the_total_against_the_budget_is_stated():
    """The 400M-vs-real figure has to exist in the repository, in prose someone will read
    and not only in a JSON field."""
    page, rec = _page(), _record()
    assert f"{rec['total_emitted_tokens']:,}" in page
    assert f"{rec['budget']:,}" in page
    assert f"{abs(rec['total_vs_budget_tokens']):,}" in page
    assert f"{abs(rec['total_vs_budget_pct'])}" in page


def test_the_recorded_artifact_is_identified_by_digest():
    assert _record()["sha256"] in _page()


def test_the_tokenizer_ordering_caveat_is_recorded():
    """The shipped tokenizer was trained on an earlier revision of this same blend. That
    is inherent -- the dependency is circular and has to be cut somewhere -- but it is
    only acceptable while it is written down."""
    page = _page().lower()
    assert "circular" in page
    assert "tokenizer" in page and "earlier revision" in page


def test_the_page_points_at_the_manifest_as_authoritative():
    assert "measurements/blend_manifest.json" in _page()
