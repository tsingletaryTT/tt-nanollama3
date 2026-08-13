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
    """Browne belongs to weird. Listing him in both would double-count him.

    KNOWN, DELIBERATE GAP: this only checks for a shared SELECTOR, not a shared BOOK. Gutenberg
    text_id 30092, "Lords of the Housetops: Thirteen Cat Tales", is a 14-contributor anthology
    matched independently by spine's "Hudson, W. H." and weird's "Blackwood, Algernon" -- no
    single author name is in both lists, so this test correctly cannot see the overlap. Impact
    is one book out of ~296 selected, and the blend built from these selectors is already
    frozen (artifacts/corpus/blend.txt, blend_manifest.json) -- this is a documented, accepted
    duplicate, not an oversight to fix by re-selecting or re-blending.
    """
    overlap = set(SOURCES["spine"].authors) & set(SOURCES["weird"].authors)
    assert not overlap, f"spine and weird share author selectors: {sorted(overlap)}"


# --- share formatting: ":.0%" rendered the smallest slice as "0%".


def test_format_share_keeps_a_fraction_of_a_percent():
    from train.corpus import format_share
    assert format_share(0.005) == "0.5%"
    assert format_share(0.135) == "13.5%"
    assert format_share(0.00575) == "0.575%"   # flavour's arithmetic ceiling at 4x


def test_format_share_leaves_whole_percentages_whole():
    from train.corpus import format_share
    assert format_share(0.31) == "31%"
    assert format_share(0.04) == "4%"
    assert format_share(1.0) == "100%"


def test_no_registered_share_renders_as_zero():
    """A slice that reads as 0% reads as "contributes nothing"."""
    from train.corpus import SOURCES, format_share
    for name, src in SOURCES.items():
        assert format_share(src.target_share) != "0%", name


def test_describe_shows_a_fractional_share():
    from train.corpus import SOURCES
    assert "0.5%" in SOURCES["flavour"].describe()


# --- rationale anti-drift.
#
# Every share in this registry was settled against a measurement, and the rationale is
# where that reasoning is written down. Three separate reviews on this branch found
# rationales still quoting superseded numbers -- availability from a retired tokenizer,
# an upsample computed at a share the source no longer holds. Prose is the only part of
# the registry nothing else checks, so check it here.

def test_a_rationale_that_cites_availability_cites_the_CURRENT_availability():
    """Historical figures are fine and useful -- they show how a share was arrived at --
    but the number in force has to be in there too, or the rationale describes a
    measurement that no longer exists."""
    import json
    from pathlib import Path
    from train.corpus import SOURCES

    report = (Path(__file__).resolve().parents[1]
              / "docs" / "measurements" / "corpus_availability.json")
    available = json.loads(report.read_text())["available"]
    for name, src in SOURCES.items():
        text = src.rationale.lower()
        if "availability" not in text and "measured tokens" not in text:
            continue
        assert f"{available[name]:,}" in src.rationale, (
            f"{name}'s rationale cites availability but not the current "
            f"{available[name]:,} tokens from {report.name}"
        )


def test_no_rationale_claims_an_upsample_the_registry_does_not_declare():
    """`upsample=N` written into prose next to a different declared N is how a reader
    learns to distrust the whole file."""
    import re
    from train.corpus import SOURCES
    for name, src in SOURCES.items():
        for claimed in re.findall(r"upsample\s*=\s*(\d+)", src.rationale):
            # A rationale may recount that a LOWER factor was tried and failed, but must
            # not present one as the factor in force.
            assert int(claimed) <= src.upsample, (
                f"{name}'s rationale claims upsample={claimed}, registry declares "
                f"{src.upsample}"
            )
