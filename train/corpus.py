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
        target_share=0.31,
        hf_repo="roneneldan/TinyStories",
        hf_revision="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        license_id="CDLA-Sharing-1.0",
        license_url="https://cdla.dev/sharing-1-0/",
        attribution="TinyStories (Eldan & Li), roneneldan/TinyStories",
        share_alike=True,
        rationale="Simple, regular grammar. The backbone that makes a small model readable. "
                  "Share raised from 30% to 31% in the Task 6 re-settle: retraining the "
                  "tokenizer on the blend compressed every OTHER domain by 6-24% (measured "
                  "against the new 32k vocabulary), which pushed procedural over the 4x "
                  "working limit while barely touching tinystories (-0.5%, since the old "
                  "vocabulary was tinystories-specialised to begin with). Tinystories has "
                  "enormous headroom (443,704,924 measured tokens against a 124,000,000 "
                  "requirement, needing only 0.28x) so it absorbs the point shaved from "
                  "procedural rather than any strange slice giving up share.",
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
        upsample=2,
        rationale="PD children's literature: more narrative backbone in an older register. "
                  "Measured availability (36,437,242 tokens) needs 1.65x upsample at 15% share -- "
                  "upsample=2 covers it with margin, well under the 4x working limit.",
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
        target_share=0.135,
        hf_repo="sedthh/gutenberg_english",
        hf_revision="28973b04f28fd7be4a6186a042bc26159d4366ca",
        license_id="MIT (packaging); public domain (texts)",
        license_url="https://huggingface.co/datasets/sedthh/gutenberg_english",
        attribution="Project Gutenberg via sedthh/gutenberg_english",
        license_note="MIT covers the aggregation; the underlying pre-1929 texts are public domain.",
        upsample=3,
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
                  "over cap) to 17 (241 unique books) with catalogue-verified PD naturalists and "
                  "anomalists in the same register. Measured availability after the broadening: "
                  "29,815,368 tokens (6.2x the old 4,803,988) under the OLD tokenizer -- needs "
                  "1.61x upsample at a 13.5% share, well under the 4x working limit, so "
                  "upsample=2 covered it with margin. Share raised from 12% to 13.5% using the "
                  "1.5 points freed by dropping flavour to its arithmetic ceiling (see flavour's "
                  "rationale) -- spine has the most headroom of any strange slice (29.82% "
                  "ceiling at 4x) and keeping the freed share inside spine+folklore+weird+flavour "
                  "holds their combined share at 26%, unchanged from before the settle. "
                  "Task 6 re-measured against the RETRAINED tokenizer: availability dropped to "
                  "26,200,908 tokens (-12.1%, the largest drop of any slice, consistent with the "
                  "old vocabulary being tinystories-specialised), which needs 2.06x -- upsample=2 "
                  "no longer covers it (54,000,000 required vs 52,401,816 achievable), so "
                  "upsample raised to 3 (achieves 78,602,724, comfortable margin, still well "
                  "under the 4x limit at 2.06x actual need). The 13.5% share and the 26% "
                  "combined strange-slice figure are UNCHANGED by this re-settle; only the "
                  "upsample factor moved. Browne, Thomas, Sir is deliberately NOT here "
                  "despite being in the pre-task list — he is weird's selector, and listing him "
                  "in both would double-count him. Andrew Lang is excluded for the same reason: "
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
        upsample=2,
        rationale="Myth and folk narrative: the dreamlike register with an archaic voice. "
                  "Measured availability (23,540,834 tokens) needs 1.36x upsample at 8% share -- "
                  "upsample=2 covers it with margin, well under the 4x working limit.",
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
        upsample=3,
        rationale="Weird fiction and baroque prose. Unambiguously PD, unlike Lovecraft. Measured "
                  "availability (7,951,195 tokens) needs 2.01x upsample at 4% share -- upsample=3 "
                  "covers it with margin, under the 4x working limit.",
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
        target_share=0.12,
        hf_repo="sedthh/gutenberg_english",
        hf_revision="28973b04f28fd7be4a6186a042bc26159d4366ca",
        license_id="MIT (packaging); public domain (texts)",
        license_url="https://huggingface.co/datasets/sedthh/gutenberg_english",
        attribution="Project Gutenberg via sedthh/gutenberg_english",
        license_note="MIT covers the aggregation; the underlying pre-1929 texts are public domain.",
        bookshelves=["Cookbooks and Cooking", "Children's Instructional Books"],
        upsample=4,
        rationale="Recipes and instructional texts: plan -> act -> observe -> report as a SHAPE. "
                  "Models trained on these learn the structure of procedural reasoning. Measured "
                  "availability under the OLD tokenizer (13,623,510 tokens) needed 3.82x upsample "
                  "at 13% share -- the tightest slice in the registry, right at the 4x working "
                  "limit. Task 6 re-measured against the RETRAINED tokenizer: availability "
                  "dropped to 12,273,087 tokens (-9.9%), which pushed the needed upsample to "
                  "4.24x at the old 13% share -- over the 4x working limit, and 4x is already "
                  "this source's upsample (raising it further would mean more repetition of the "
                  "same ~12.3M raw tokens, which is what the cap exists to prevent, not a share "
                  "problem to solve by repeating harder). Share dropped from 13% to 12% instead: "
                  "the ceiling at upsample=4 is 12,273,087 x 4 / 400,000,000 = 12.27%, so 12% "
                  "needs only 3.91x, with real margin against another small re-measurement "
                  "swing. The freed 1 point moved to tinystories (see its rationale), not to any "
                  "strange slice, so spine+folklore+weird+flavour is untouched by this move. Do "
                  "not raise this share again without re-measuring: it is still the tightest "
                  "slice in the registry.",
    ),
    "flavour": CorpusSource(
        name="flavour",
        slice="flavour",
        target_share=0.005,
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
                  "at this scale risks memorisation, and Stein IS repetition-as-style. Measured "
                  "availability is only 623,814 tokens: at the 4x cap that is a hard ceiling of "
                  "0.62% of the 400M budget, so the previous 2.00% share was arithmetically "
                  "impossible (needed 12.8x). Dropped to 0.5%, comfortably under the ceiling "
                  "(needs 3.21x); the freed 1.5 points moved to spine.",
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
