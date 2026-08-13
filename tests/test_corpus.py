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
    # Reject branch refs like "main", "master", "HEAD" — must be a pinned commit sha
    branch_like = {"main", "master", "head", "develop", "development"}
    assert src.hf_revision.lower() not in branch_like, (
        f"{name}: hf_revision '{src.hf_revision}' is a branch ref, not a pinned commit"
    )
    # Assert it looks like a 40-character hex sha
    assert len(src.hf_revision) == 40 and all(
        c in "0123456789abcdef" for c in src.hf_revision.lower()
    ), f"{name}: hf_revision '{src.hf_revision}' is not a 40-char commit sha"


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


def test_spine_is_broad_enough_to_avoid_heavy_repetition():
    """spine had 53 books against a 12% share -- 10x repetition, over the cap of 8.

    Every author here was verified present in the Gutenberg catalogue before being added.
    The count guards against the slice silently narrowing again.
    """
    spine = SOURCES["spine"]
    assert len(spine.authors) >= 17, (
        f"spine has only {len(spine.authors)} author selectors; it was broadened to avoid "
        f"needing 10x upsample"
    )
    for required in ("Fabre, Jean-Henri", "Fort, Charles", "Thoreau, Henry David",
                     "Darwin, Charles", "Jefferies, Richard", "Flammarion, Camille"):
        assert required in spine.authors, f"spine lost its {required!r} selector"


def test_spine_and_folklore_do_not_share_selectors():
    """Andrew Lang belongs to folklore. Listing him in both would double-count him."""
    overlap = set(SOURCES["spine"].authors) & set(SOURCES["folklore"].authors)
    assert not overlap, f"spine and folklore share author selectors: {sorted(overlap)}"


def test_spine_and_weird_do_not_share_selectors():
    """Browne belongs to weird. Listing him in both would double-count him."""
    overlap = set(SOURCES["spine"].authors) & set(SOURCES["weird"].authors)
    assert not overlap, f"spine and weird share author selectors: {sorted(overlap)}"
