# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The corpus source registry — one entry per source, with its provenance.

Mirrors ``train/sizes.py``: a typed description that every other tool reasons about, kept
separate from the I/O that acts on it. Licence lives here as DATA so the model card's
licensing section can be generated rather than written, which is what stops it drifting —
this project has already been bitten twice by prose going stale against reality.

Shares are TARGETS, not measurements. ``scripts/measure_corpus.py`` reports what is actually
available; the spec is explicit that these numbers get revised against it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

#: Functional slices. A source earns its place by what it does to the output.
SLICES = frozenset({
    "backbone",     # well-formed simple prose; makes a small model readable at all
    "grounding",    # facts and real nouns to be strange about
    "spine",        # observational-mystical: Fabre, Fort, Maeterlinck, Hodgson
    "folklore",     # myth and folk narrative
    "weird",        # weird fiction and poetry
    "agentic",      # plan -> act -> observe -> report shapes
    "flavour",      # small, upsampled, capped: Stein, I Ching
})


@dataclass(frozen=True)
class CorpusSource:
    """One corpus source: where it comes from, what it is for, and its licence."""

    name: str
    slice: str
    #: Fraction of the final blend this source targets, in [0, 1].
    target_share: float

    # -- provenance ---------------------------------------------------------------
    hf_repo: str
    #: Pinned revision. Never None: an unpinned fetch is not reproducible, and the whole
    #: point of shipping a recipe rather than the corpus is that the recipe is exact.
    hf_revision: str
    hf_config: Optional[str] = None
    hf_split: str = "train"

    # -- licensing ----------------------------------------------------------------
    #: SPDX identifier where one exists, else a short stable string.
    license_id: str = ""
    license_url: str = ""
    #: Human-readable attribution line, rendered into the generated model-card section.
    attribution: str = ""
    #: True when the licence obliges downstream share-alike. Drives the model card's
    #: "unsettled derivative status" language.
    share_alike: bool = False
    #: Note on the distinction between the packaging licence and the underlying texts.
    license_note: str = ""

    # -- selection and mixing -----------------------------------------------------
    #: Author names to select on, matched case-insensitively as substrings of METADATA
    #: ``authors``. Empty means "no author filter".
    authors: List[str] = field(default_factory=list)
    #: Gutenberg bookshelf names to select on, matched case-insensitively.
    bookshelves: List[str] = field(default_factory=list)
    #: Repetition factor applied when blending. >1 only for deliberately small sources.
    upsample: int = 1
    #: Why this source exists, in one line. Shown by ``describe()``.
    rationale: str = ""

    def describe(self) -> str:
        sel = []
        if self.authors:
            sel.append(f"{len(self.authors)} author(s)")
        if self.bookshelves:
            sel.append(f"{len(self.bookshelves)} bookshelf/-ves")
        selection = ", ".join(sel) if sel else "all rows"
        return (
            f"{self.name}: slice={self.slice} target={self.target_share:.0%} "
            f"upsample={self.upsample}\n"
            f"  from  : {self.hf_repo}@{self.hf_revision[:12]} ({selection})\n"
            f"  licence: {self.license_id}\n"
            f"  {self.rationale}"
        )


#: Every source in the blend.
#:
#: Revisions are pinned to the values current when this plan was written. A revision that
#: no longer resolves is a loud failure at fetch time, which is the intended behaviour:
#: silently training on different data is the thing being prevented.
SOURCES: Dict[str, CorpusSource] = {
    "tinystories": CorpusSource(
        name="tinystories",
        slice="backbone",
        target_share=0.30,
        hf_repo="roneneldan/TinyStories",
        hf_revision="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        license_id="CDLA-Sharing-1.0",
        license_url="https://cdla.dev/sharing-1-0/",
        attribution="TinyStories (Eldan & Li), roneneldan/TinyStories",
        share_alike=True,
        rationale="Simple, regular grammar. The backbone that makes a small model readable.",
    ),
    "gutenberg_children": CorpusSource(
        name="gutenberg_children",
        slice="backbone",
        target_share=0.15,
        hf_repo="sedthh/gutenberg_english",
        hf_revision="28973b04f28fd7be4a6186a042bc26159d4366ca",
        license_id="MIT (packaging); public domain (texts)",
        license_url="https://huggingface.co/datasets/sedthh/gutenberg_english",
        attribution="Project Gutenberg via sedthh/gutenberg_english",
        license_note="MIT covers the aggregation; the underlying pre-1929 texts are public domain.",
        bookshelves=["Children's Literature", "Children's Book Series"],
        rationale="PD children's literature: more narrative backbone in an older register.",
    ),
    "wikipedia_simple": CorpusSource(
        name="wikipedia_simple",
        slice="grounding",
        target_share=0.15,
        hf_repo="wikimedia/wikipedia",
        hf_revision="b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
        hf_config="20231101.simple",
        license_id="CC-BY-SA-3.0",
        license_url="https://creativecommons.org/licenses/by-sa/3.0/",
        attribution="Simple English Wikipedia contributors, via wikimedia/wikipedia",
        share_alike=True,
        rationale="Real nouns and facts to be strange ABOUT. Chimps, ants, sticks, anthills.",
    ),
    "spine": CorpusSource(
        name="spine",
        slice="spine",
        target_share=0.12,
        hf_repo="sedthh/gutenberg_english",
        hf_revision="28973b04f28fd7be4a6186a042bc26159d4366ca",
        license_id="MIT (packaging); public domain (texts)",
        license_url="https://huggingface.co/datasets/sedthh/gutenberg_english",
        attribution="Project Gutenberg via sedthh/gutenberg_english",
        license_note="MIT covers the aggregation; the underlying pre-1929 texts are public domain.",
        authors=[
            # Original four: insect field observation and deadpan anomalism.
            "Fabre, Jean-Henri",              # 10 vols; the spine's spine
            "Maeterlinck, Maurice",           # mystical about insect collectives
            "Fort, Charles",                  # anomalies compiled as data
            "Hodgson, William Hope",          # found-manuscript cosmic dread
            # Naturalists and field observers (verified counts in the catalogue).
            "Darwin, Charles",                # 39
            "Burroughs, John",                # 29
            "Thoreau, Henry David",           # 21
            "Seton, Ernest Thompson",         # 19
            "Jefferies, Richard",             # 18 -- nature writing shading into mysticism
            "Hudson, W. H.",                  # 18
            "Wallace, Alfred Russel",         # 17
            "Muir, John",                     # 12
            "Gosse, Philip Henry",            # 3
            "White, Gilbert",                 # 3  -- Natural History of Selborne
            # Cosmic scale and the possibility of other minds.
            "Flammarion, Camille",            # 7
            "Proctor, Richard A.",            # 7
            "Donnelly, Ignatius",             # 2
        ],
        rationale="Observational-mystical: the model's voice. Fabre is field observation that "
                  "is ALREADY agentic tool-use theatre; Fort applies the same method to things "
                  "that should not happen. Broadened from five authors (53 books, 10x upsample, "
                  "over cap) to 17 (241 unique books, ~2x) with catalogue-verified PD naturalists "
                  "and anomalists in the same register. Browne, Thomas, Sir is deliberately NOT "
                  "here despite being in the pre-task list — he is weird's selector, and listing "
                  "him in both would double-count him. Andrew Lang is excluded for the same reason: "
                  "he is folklore's selector. Blavatsky and Swedenborg are deliberately excluded: "
                  "they assert doctrine where this slice documents the inexplicable.",
    ),
    "folklore": CorpusSource(
        name="folklore",
        slice="folklore",
        target_share=0.08,
        hf_repo="sedthh/gutenberg_english",
        hf_revision="28973b04f28fd7be4a6186a042bc26159d4366ca",
        license_id="MIT (packaging); public domain (texts)",
        license_url="https://huggingface.co/datasets/sedthh/gutenberg_english",
        attribution="Project Gutenberg via sedthh/gutenberg_english",
        license_note="MIT covers the aggregation; the underlying pre-1929 texts are public domain.",
        bookshelves=["Mythology", "Folklore"],
        authors=["Frazer, James George", "Lang, Andrew"],
        rationale="Myth and folk narrative: the dreamlike register with an archaic voice.",
    ),
    "weird": CorpusSource(
        name="weird",
        slice="weird",
        target_share=0.04,
        hf_repo="sedthh/gutenberg_english",
        hf_revision="28973b04f28fd7be4a6186a042bc26159d4366ca",
        license_id="MIT (packaging); public domain (texts)",
        license_url="https://huggingface.co/datasets/sedthh/gutenberg_english",
        attribution="Project Gutenberg via sedthh/gutenberg_english",
        license_note="MIT covers the aggregation; the underlying pre-1929 texts are public domain.",
        authors=["Blackwood, Algernon", "Dunsany", "Machen, Arthur", "Browne, Thomas, Sir"],
        rationale="Weird fiction and baroque prose. Unambiguously PD, unlike Lovecraft.",
    ),
    "poetry": CorpusSource(
        name="poetry",
        slice="weird",
        target_share=0.01,
        hf_repo="biglam/gutenberg-poetry-corpus",
        hf_revision="fcd42e249fed48dbd1d3b9b969528ef9298d3464",
        license_id="CC0-1.0",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        attribution="Gutenberg Poetry Corpus (Allison Parrish), biglam/gutenberg-poetry-corpus",
        rationale="Density and associative leaps, per line rather than per book.",
    ),
    "procedural": CorpusSource(
        name="procedural",
        slice="agentic",
        target_share=0.13,
        hf_repo="sedthh/gutenberg_english",
        hf_revision="28973b04f28fd7be4a6186a042bc26159d4366ca",
        license_id="MIT (packaging); public domain (texts)",
        license_url="https://huggingface.co/datasets/sedthh/gutenberg_english",
        attribution="Project Gutenberg via sedthh/gutenberg_english",
        license_note="MIT covers the aggregation; the underlying pre-1929 texts are public domain.",
        bookshelves=["Cookbooks and Cooking", "Children's Instructional Books"],
        rationale="Recipes and instructional texts: plan -> act -> observe -> report as a SHAPE. "
                  "Models trained on these learn the structure of procedural reasoning.",
    ),
    "flavour": CorpusSource(
        name="flavour",
        slice="flavour",
        target_share=0.02,
        hf_repo="sedthh/gutenberg_english",
        hf_revision="28973b04f28fd7be4a6186a042bc26159d4366ca",
        license_id="MIT (packaging); public domain (texts)",
        license_url="https://huggingface.co/datasets/sedthh/gutenberg_english",
        attribution="Project Gutenberg via sedthh/gutenberg_english",
        license_note="MIT covers the aggregation; the underlying pre-1929 texts are public domain.",
        authors=["Stein, Gertrude", "Legge, James"],
        upsample=4,
        rationale="Stein (grammar intact, semantics dissolved) and the I Ching (Legge 1882: "
                  "terse oracular response). Tiny, so upsampled — capped, because repetition "
                  "at this scale risks memorisation, and Stein IS repetition-as-style.",
    ),
}


def get_source(name: str) -> CorpusSource:
    """Look up a source, raising with the available names rather than a bare miss."""
    try:
        return SOURCES[name]
    except KeyError:
        raise KeyError(
            f"unknown corpus source {name!r}; registered sources: {sorted(SOURCES)}"
        ) from None


def total_target_share() -> float:
    """Sum of every source's target share. Must be 1.0."""
    return sum(s.target_share for s in SOURCES.values())


__all__ = ["SLICES", "SOURCES", "CorpusSource", "get_source", "total_target_share"]
