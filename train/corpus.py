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
    "dialogue",     # question -> answer shapes; the only slice where the corpus is asked
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
    #: How many CONSECUTIVE rows of ``artifacts/raw/<name>/text.jsonl`` make up one
    #: document, i.e. one span that ``scripts/prepare_corpus.py`` terminates with a ``</s>``.
    #:
    #: 1 for every source whose rows really are documents (a story, an article, a book).
    #: It is >1 only where the upstream dataset's row is SMALLER than a document, which is
    #: true of exactly one source here: ``biglam/gutenberg-poetry-corpus`` has one row per
    #: LINE of verse (3,085,117 rows averaging ~7 words), so treating a row as a document
    #: would put an end-of-document token every ~7 words. That is not a boundary the model
    #: should learn: measured against the shipped shares it would have made ``poetry`` --
    #: 1% of the blend -- carry about a third of every ``</s>`` in the corpus, teaching a
    #: ~7-word prior for "stop", which is the opposite of the termination signal document
    #: separators exist to provide.
    #:
    #: Rows arrive in dataset order and that corpus is ordered by Gutenberg id, so N
    #: consecutive rows are N consecutive lines of the same poem (occasionally straddling a
    #: book boundary at the seam). The upstream row's ``gid`` would give exact poem
    #: boundaries, but ``scripts/fetch_corpus.py`` keeps only the text column, so recovering
    #: them means re-fetching. Grouping is the cheaper approximation and is honest about
    #: being one: it restores document-scale spans without claiming to reconstruct poems.
    rows_per_document: int = 1
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
            f"{self.name}: slice={self.slice} target={format_share(self.target_share)} "
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
    "dialogue": CorpusSource(
        name="dialogue",
        slice="dialogue",
        # Settled on measured yield: 15,011 documents, 1,968,868 words, 131 words/doc.
        # At upsample 3 that is ~5.9M words (~7.7M tokens), which supplies ~2% of the
        # blend and no more. 3% would have needed upsample 5.
        target_share=0.020,
        hf_repo="databricks/databricks-dolly-15k",
        hf_revision="bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
        hf_split="train",
        license_id="CC-BY-SA-3.0",
        license_url="https://creativecommons.org/licenses/by-sa/3.0/",
        attribution="Databricks Dolly 15k, CC-BY-SA-3.0, https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        share_alike=True,
        license_note=(
            "Share-alike, and deliberately the SAME licence already carried by "
            "wikipedia_simple. This adds share-alike MASS but no new share-alike KIND: the "
            "blend's two copyleft terms (CDLA-Sharing-1.0, CC-BY-SA-3.0) stay two. "
            "CC-BY-NC alternatives (no_robots, alpaca) were rejected outright -- a "
            "non-commercial term would restrict the whole blend in a way no existing source "
            "does."
        ),
        # 3x. The poetry note in this file warns what upsampling a small source does to the
        # </s> prior, so it was measured here rather than assumed: at upsample 3 this slice
        # is 0.82% of all document separators against 0.92% of all words -- 0.89x, i.e.
        # slightly UNDER-represented in separators, not over. The corpus's actual separator
        # skew is poetry, which supplies 3.08M of 5.49M raw documents at 7 words each.
        upsample=3,
    ),
    "tinystories": CorpusSource(
        # 0.310 -> 0.290 on 2026-08-18 to fund the dialogue slice. TinyStories is the
        # defensible donor: reducing it is the one corpus change this project has measured
        # as a real register gain (1.79x the seed floor, 3/3 seeds).
        name="tinystories",
        slice="backbone",
        target_share=0.290,
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
                  "procedural rather than any strange slice giving up share. Re-measured "
                  "2026-08-14 after document separators were added: 447,943,902 tokens, "
                  "+4,238,978 on the figure above and exactly two tokens per document "
                  "(the </s> and its newline), needing 0.2768x.",
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
                  "Measured availability (34,253,856 tokens against the retrained tokenizer) "
                  "needs 1.75x upsample at a 15% share -- upsample=2 covers it with margin "
                  "(a 17.13% ceiling), well under the 4x working limit. The pre-retrain "
                  "measurement was 36,437,242 tokens needing 1.65x; the -6.0% shift is the "
                  "smallest of any slice. Re-measured 2026-08-14 after document separators "
                  "were added: 34,255,022 tokens (+1,166, two per document), needing "
                  "1.7516x -- upsample=2 still covers it.",
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
                  "29,815,368 tokens (6.2x the old 4,803,988) under the OLD tokenizer -- which "
                  "needed 1.61x at the 12% share it then held, and 1.81x at the 13.5% share it "
                  "moved to, both well under the 4x working limit, so upsample=2 covered it with "
                  "margin. Share raised from 12% to 13.5% using the "
                  "1.5 points freed by dropping flavour to its arithmetic ceiling (see flavour's "
                  "rationale) -- spine has the most headroom of any strange slice (26.20% "
                  "ceiling at 4x against folklore's 10.64%, weird's 5.28% and flavour's 0.575%) "
                  "and keeping the freed share inside spine+folklore+weird+flavour "
                  "holds their combined share at 26%, unchanged from before the settle. "
                  "Task 6 re-measured against the RETRAINED tokenizer: availability dropped to "
                  "26,200,908 tokens (-12.1%, the largest drop of any STRANGE slice and the "
                  "second largest overall -- wikipedia_simple fell -23.8% -- consistent with the "
                  "old vocabulary being tinystories-specialised), which needs 2.06x -- upsample=2 "
                  "no longer covers it (54,000,000 required vs 52,401,816 achievable), so "
                  "upsample raised to 3 (achieves 78,602,724, comfortable margin, still well "
                  "under the 4x limit at 2.06x actual need). The 13.5% share and the 26% "
                  "combined strange-slice figure are UNCHANGED by this re-settle; only the "
                  "upsample factor moved. Re-measured 2026-08-14 after document separators "
                  "were added: 26,201,390 tokens (+482, exactly two per document), needing "
                  "2.0610x -- upsample=3 still covers it."
                  " Browne, Thomas, Sir is deliberately NOT here "
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
                  "Measured availability (21,274,517 tokens against the retrained tokenizer, "
                  "-9.6% on the 23,540,834 measured before it) needs 1.50x upsample at an 8% "
                  "share -- upsample=2 covers it with margin (a 10.64% ceiling), well under "
                  "the 4x working limit. Re-measured 2026-08-14 after document separators "
                  "were added: 21,274,911 tokens (+394, two per document), needing "
                  "1.5041x -- upsample=2 still covers it.",
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
                  "availability (7,040,931 tokens against the retrained tokenizer, -11.5% on the "
                  "7,951,195 measured before it) needs 2.27x upsample at a 4% share -- "
                  "upsample=3 covers it (a 5.28% ceiling), under the 4x working limit. "
                  "Re-measured 2026-08-14 after document separators were added: 7,041,041 "
                  "tokens (+110, two per document), needing 2.2724x -- upsample=3 still "
                  "covers it.",
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
        rows_per_document=64,
        rationale="Density and associative leaps, per line rather than per book. The only "
                  "source whose upstream row is a LINE and not a document, hence "
                  "rows_per_document=64: ~7 words per row means a per-row </s> would fire "
                  "every ~7 words, and this 1% slice would then hold roughly a third of "
                  "every document separator in the blend. 64 lines is ~450 words, the "
                  "scale of a short story, so the separator marks a document-sized span "
                  "the way it does everywhere else.",
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
                  "slice in the registry. Re-measured 2026-08-14 after document separators "
                  "were added: 12,273,447 tokens (+360, two per document), needing 3.9109x "
                  "-- still the tightest slice, and still inside the 4x limit.",
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
                  "availability against the retrained tokenizer is only 575,377 tokens: at the "
                  "4x cap that is a hard ceiling of 0.575% of the 400M budget, so the original "
                  "2.00% share was arithmetically impossible (would need 13.9x). The 0.5% share "
                  "needs 3.48x. It is NOT comfortable: the ceiling sits 0.075 points above the "
                  "share, so a further 13% fall in measured availability would put 0.5% out of "
                  "reach at 4x, and this slice has no headroom to absorb another re-measurement "
                  "swing. It was settled at 0.5% when the pre-retrain measurement (623,814 "
                  "tokens, 0.624% ceiling, 3.21x needed) left 0.12 points of headroom; the "
                  "retrain took -7.8% of it and more than a third of that margin with it. The "
                  "share stayed at 0.5% because it still fits, not because it fits well — do "
                  "not raise it, and re-measure before trusting it after any tokenizer change. "
                  "The 1.5 points freed when it dropped from 2.00% moved to spine. "
                  "Re-measured 2026-08-14 after document separators were added: 575,391 "
                  "tokens (+14 — this slice is seven documents), needing 3.4759x. The "
                  "ceiling moves to 0.5754%, so the headroom is unchanged at 0.075 points.",
    ),
}


def format_share(share: float) -> str:
    """Render a share in [0, 1] as a percentage WITHOUT dropping its fraction.

    ``f"{share:.0%}"`` was used in three places (the generated licensing document, the
    operator gate table, and the blend's shortfall message) and it rounds every fractional
    share away: ``flavour``'s 0.5% rendered as **0%** — "contributes nothing" — and
    ``spine``'s 13.5% as 14%, in a document whose banner promises it cannot go stale.

    Fractions are kept only as far as they exist: 0.31 renders "31%", not "31.0%", so
    whole-number shares stay readable. Three decimal places of a percent is enough for
    every share this registry can hold and for the arithmetic ceilings derived from them
    (``flavour``'s ceiling is 0.575%).
    """
    text = f"{round(share * 100, 3):.3f}".rstrip("0").rstrip(".")
    return f"{text}%"


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


__all__ = ["SLICES", "SOURCES", "CorpusSource", "format_share", "get_source",
           "total_target_share"]
