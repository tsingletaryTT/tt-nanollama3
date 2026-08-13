# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Registry anti-drift tests.

The registry is the single place a source's licence, slice and fetch spec live. A source
missing any of them is a source whose provenance cannot be stated, which this project
treats as a defect rather than an omission.
"""
import pytest
from train.corpus import SLICES, SOURCES, CorpusSource, get_source, total_target_share

ALL = sorted(SOURCES)


@pytest.mark.parametrize("name", ALL)
def test_every_source_declares_a_licence(name):
    src = SOURCES[name]
    assert src.license_id, f"{name}: no license_id"
    assert src.license_url, f"{name}: no license_url"


@pytest.mark.parametrize("name", ALL)
def test_every_source_has_a_known_slice(name):
    assert SOURCES[name].slice in SLICES


@pytest.mark.parametrize("name", ALL)
def test_every_source_has_a_resolvable_fetch_spec(name):
    src = SOURCES[name]
    assert src.hf_repo, f"{name}: no hf_repo"
    assert src.hf_revision, f"{name}: no pinned revision — fetches must be reproducible"


@pytest.mark.parametrize("name", ALL)
def test_upsample_is_at_least_one(name):
    assert SOURCES[name].upsample >= 1


def test_target_shares_sum_to_one():
    total = total_target_share()
    assert abs(total - 1.0) < 1e-9, f"target shares sum to {total:.4f}, not 1.0"


def test_get_source_rejects_unknown_with_a_helpful_message():
    with pytest.raises(KeyError) as excinfo:
        get_source("not-a-source")
    assert "registered sources" in str(excinfo.value)


def test_flavour_sources_are_capped():
    """Upsampled flavour sources risk memorisation; the cap is the control."""
    for name, src in SOURCES.items():
        if src.slice == "flavour":
            assert src.upsample <= 8, f"{name}: upsample {src.upsample} exceeds the cap"


def test_corpus_module_does_no_io():
    """The registry is data and arithmetic. I/O belongs in scripts/."""
    import inspect
    import train.corpus as m
    src = inspect.getsource(m)
    for forbidden in ("open(", "requests.", "urllib", "load_dataset", "snapshot_download"):
        assert forbidden not in src, f"train/corpus.py performs I/O: {forbidden}"
