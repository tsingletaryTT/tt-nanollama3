#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Publish (or verify) the tt-tnt corpus RECIPE to the Hugging Face dataset repo.

``episod/tt-tnt-corpus`` ships a recipe, not the corpus: the source registry
(``train/corpus.py``), the fetch/prepare/measure/blend scripts, the generated licensing
table, and the provenance manifest whose ``sha256`` is the promise that running the recipe
reproduces the blend byte-identically. Those files are a manual copy of what lives in this
git repository, and manual copies drift silently -- which is exactly what happened, twice,
before this script existed:

1. The corpus pipeline added document separators (the blend previously had ZERO), and the
   Hub kept pre-separator scripts sitting next to a manifest ``sha256`` those scripts could
   no longer reproduce. A user following the recipe would get a mismatch with no way to
   tell whether they erred or the recipe was stale. Nothing detected it; a human happened
   to notice.
2. The dataset card carried a hand-transcribed "manifest vs. tokenized" gap figure that was
   simply wrong -- it claimed 0.46% where the real gap was ~1.71%, and this repository's own
   ``docs/corpus_blend.md`` independently claimed 0.42% for what is really 1.90%. A number
   typed by hand into prose that nothing checks goes stale and goes wrong.

This script has two modes:

* ``--publish``  Upload the recipe file set (see ``RECIPE_FILES`` below -- the ONE place
  that set is defined) and a freshly *generated* dataset card to the Hub. Requires ``--yes``.
* ``--verify`` (the default)  Download what is currently on the Hub and diff it, byte for
  byte, against what the working tree would publish right now. Exits non-zero and prints
  exactly which files differ. Read-only. This is what makes drift *detectable* -- run it
  right after any corpus rebuild, not just when someone remembers to ask.

Every number that appears in the generated card -- token totals, the blend's ``sha256``,
per-source target/achieved shares, real repetition factors, and the manifest-vs-tokenized
gap and its percentage -- is COMPUTED at generation time from ``train/corpus.py`` and
``docs/measurements/blend_manifest.json``, plus (for the one figure neither of those files
contains: the actual tokenized-file token count) the real ``artifacts/tokens-v3/*.npy``
arrays that ``train/tokenization.py`` already wrote. Nothing is transcribed by hand. This
follows the precedent of ``scripts/render_licensing.py``, which generates the licensing
table from the registry for exactly this reason.

Safety rules baked into this script, not left to the caller's discipline:

* **Never upload corpus text, tokenizer files, or weights.** ``_assert_recipe_is_redistributable``
  checks the fixed ``RECIPE_FILES`` list against a path/suffix/size blocklist before any
  network call, in both ``--dry-run`` and real publishes. The blend is not redistributable:
  46% of it is share-alike under two mutually incompatible copyleft licences (see
  ``train/corpus.py``'s ``share_alike`` flags), so shipping the recipe rather than the text
  is a licensing necessity, not a style choice.
* **Never change repo visibility.** The repo is PUBLIC (``EXPECTED_PRIVATE = False``) and
  must stay that way. This script never passes a ``private=`` argument to any Hub call, and
  ``_assert_visibility_unchanged`` reads the Hub's own ``private`` flag before and after any
  write to prove nothing moved it. Do NOT use ``tt-model push`` for this repo -- it calls
  ``set_visibility`` unconditionally on every push, which is exactly the failure mode this
  guard exists to catch.

Usage:

    python scripts/publish_corpus_recipe.py                 # --verify (the default)
    python scripts/publish_corpus_recipe.py --verify
    python scripts/publish_corpus_recipe.py --dry-run        # preview a publish, no network writes
    python scripts/publish_corpus_recipe.py --publish --yes  # actually upload
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.corpus import SOURCES, CorpusSource, format_share  # noqa: E402

REPO_ID_DEFAULT = "episod/tt-tnt-corpus"
LICENSE = "apache-2.0"

# PRIVACY. The repo was flipped public on 2026-08-14, out-of-band, with explicit
# authorization (see scripts/publish_to_hub.py's EXPECTED_PRIVATE for the model repo's
# sibling story). This script has no code path that can change visibility in either
# direction -- see test_source_never_sets_private and _assert_visibility_unchanged below --
# so this constant is only ever compared against, never assigned from.
EXPECTED_PRIVATE = False

MANIFEST_PATH = ROOT / "docs" / "measurements" / "blend_manifest.json"

#: Where train/tokenization.py wrote the real token-id arrays for the CURRENT
#: (document-separator-carrying) blend. These are the ground truth for the
#: "manifest vs. tokenized" figure: neither blend_manifest.json nor train/corpus.py
#: contains this number (it only exists as prose, previously copied out by hand -- the
#: exact failure mode this script exists to end). Reading the real arrays instead of
#: re-tokenizing ~400M tokens at every dry-run is far cheaper and cannot drift, because it
#: IS the artifact the number describes, not a description of it.
TOKENS_DIR_DEFAULT = ROOT / "artifacts" / "tokens-v3"
TOKENIZER_DIR_DEFAULT = ROOT / "artifacts" / "tokenizer"


@dataclass(frozen=True)
class RecipeFile:
    """One file in the published recipe.

    The recipe file set is defined ONCE, as the ``RECIPE_FILES`` list below -- adding a
    file to what gets published is a one-line change here, not an edit scattered across an
    upload list, a "what's in this repo" table, and a redistribution guard that all have to
    agree with each other.
    """

    path: str  # relative to ROOT; also the path_in_repo on the Hub (structure preserved)
    description: str  # one line, rendered into the generated card's file table


#: The recipe: source registry, pipeline scripts, and the provenance record. Order here is
#: the order both the upload and the card's file table use.
RECIPE_FILES: List[RecipeFile] = [
    RecipeFile(
        "train/corpus.py",
        "The source registry -- one entry per source: HF repo, pinned revision, licence, "
        "target share, author/bookshelf selectors, repetition factor, and (for `poetry` "
        "only) how many consecutive rows make up one document. This is the payload; "
        "everything else operates on it.",
    ),
    RecipeFile("scripts/fetch_corpus.py", "Downloads each source at its pinned revision."),
    RecipeFile(
        "scripts/prepare_corpus.py",
        "Cleans and normalises fetched text (e.g. strips Project Gutenberg header/footer "
        "boilerplate) and terminates every document with a `</s>` separator.",
    ),
    RecipeFile(
        "scripts/measure_corpus.py",
        "Measures real per-source token availability against the trained tokenizer; "
        "writes the availability report the shares are settled against.",
    ),
    RecipeFile(
        "scripts/blend_corpus.py",
        "Deterministically assembles the final blend from prepared sources and the "
        "availability report, closing each source's truncated tail with a separator; "
        "writes `blend.txt` and both provenance manifests.",
    ),
    RecipeFile(
        "scripts/check_disk_space.py",
        "Preflight check -- refuses to start a fetch if the volume doesn't have room.",
    ),
    RecipeFile(
        "docs/corpus_licensing.md",
        "Per-source licence table, generated from `train/corpus.py` by the code repo's "
        "`scripts/render_licensing.py` (not included here -- this is its output, not the "
        "generator).",
    ),
    RecipeFile(
        "docs/corpus_blend.md",
        "The provenance narrative: what the last real build actually produced, source by "
        "source, the document-boundary accounting, and why real repetition differs from "
        "the declared `upsample` ceiling.",
    ),
    RecipeFile(
        "docs/measurements/blend_manifest.json",
        "The machine-readable provenance record written by `blend_corpus.py` itself: "
        "per-source emitted tokens, achieved share, repetition factor, and the blend's "
        "SHA-256.",
    ),
]

#: Things that must never appear in RECIPE_FILES, checked by path and by suffix. This is
#: belt-and-suspenders around a list that is manually curated above: the list itself is
#: small and reviewed, but a future one-line addition to it is exactly the kind of change
#: that could accidentally add a real artifact instead of a recipe file, and this check
#: fires on that before any network call, not just in code review.
_FORBIDDEN_TOP_LEVEL_DIRS = ("artifacts",)
_FORBIDDEN_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".npy", ".model")
_FORBIDDEN_FILENAMES = (
    "blend.txt", "corpus.txt", "tokenizer.json", "vocab.json", "merges.txt",
    "train_ids.npy", "val_ids.npy", "pytorch_model.bin", "model.safetensors",
)
#: Recipe files are small scripts, docs, and a JSON manifest. Corpus text is gigabytes;
#: tokenizer/weight files are megabytes-to-gigabytes. 1 MB is generous headroom above the
#: largest real recipe file (docs/corpus_blend.md, well under 100 KB) and nowhere near the
#: smallest real artifact this guards against.
_MAX_RECIPE_FILE_BYTES = 1_000_000


def _assert_recipe_is_redistributable(files: Sequence[RecipeFile] = RECIPE_FILES) -> None:
    """Refuse to publish if the recipe file set has come to include anything the blend's
    licences forbid redistributing: corpus text, tokenizer artifacts, or model weights.

    Checked by path prefix, filename, suffix, AND on-disk size -- any one of these firing
    is enough to refuse, because each catches a different mistake (a file moved under
    ``artifacts/``, a familiar dangerous filename appearing at a new path, an unfamiliar
    filename with a giveaway extension, or a recipe-shaped file that is actually huge).
    """
    for rf in files:
        rel = Path(rf.path)
        if rel.parts and rel.parts[0] in _FORBIDDEN_TOP_LEVEL_DIRS:
            raise ValueError(
                f"refusing to publish {rf.path!r}: recipe files must never live under "
                f"{'/'.join(_FORBIDDEN_TOP_LEVEL_DIRS)}/ -- that is exactly the corpus "
                f"text, tokenizer, and weights this repository must not redistribute "
                f"(46% of the blend is share-alike under two mutually incompatible "
                f"copyleft licences; see train/corpus.py)"
            )
        if rel.name.lower() in _FORBIDDEN_FILENAMES:
            raise ValueError(
                f"refusing to publish {rf.path!r}: filename is on the never-redistribute "
                f"blocklist ({_FORBIDDEN_FILENAMES})"
            )
        if rel.suffix.lower() in _FORBIDDEN_SUFFIXES:
            raise ValueError(
                f"refusing to publish {rf.path!r}: suffix {rel.suffix!r} looks like corpus "
                f"text, a tokenizer artifact, or model weights, none of which this "
                f"repository may redistribute"
            )
        abs_path = ROOT / rel
        if abs_path.is_file() and abs_path.stat().st_size > _MAX_RECIPE_FILE_BYTES:
            raise ValueError(
                f"refusing to publish {rf.path!r}: {abs_path.stat().st_size:,} bytes is far "
                f"larger than any real recipe file should be (limit "
                f"{_MAX_RECIPE_FILE_BYTES:,}) -- this looks like it stopped being a recipe "
                f"file and started being an artifact"
            )


def _load_manifest(path: Path = MANIFEST_PATH) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist -- run scripts/blend_corpus.py first to produce it"
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------
# The one figure neither blend_manifest.json nor train/corpus.py contains: the real token
# count of the finished, concatenated blend.txt, as actually tokenized by
# train/tokenization.py. Read from the arrays it already wrote, rather than re-running
# tokenization (expensive, and the arrays already ARE the ground truth being described).
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TokenizedStats:
    """The real, on-disk tokenized totals for the current blend."""

    total_tokens: int
    train_tokens: int
    val_tokens: int
    eos_count: int


def discover_eos_token_id(tokenizer_dir: Path = TOKENIZER_DIR_DEFAULT) -> int:
    """The trained tokenizer's own eos id, not a hardcoded ``2``.

    ``docs/corpus_blend.md`` and the currently-published Hub card both state this is id 2,
    but stating a fact and deriving it are different things, and this repository is
    specifically trying to stop doing the former where it can do the latter.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    if tok.eos_token_id is None:
        raise ValueError(f"{tokenizer_dir}: tokenizer has no eos_token_id")
    return int(tok.eos_token_id)


def compute_tokenized_stats(
    tokens_dir: Path = TOKENS_DIR_DEFAULT, eos_token_id: Optional[int] = None,
) -> TokenizedStats:
    """Read the real ``train_ids.npy`` / ``val_ids.npy`` arrays and summarize them.

    Uses ``mmap_mode='r'`` so this doesn't have to fully materialize ~1.4 GB of int32 in
    memory just to learn its length; the eos count still has to touch every element, but
    that is a single vectorized comparison, not a re-tokenization of the underlying text.
    """
    import numpy as np

    train_path = tokens_dir / "train_ids.npy"
    val_path = tokens_dir / "val_ids.npy"
    for p in (train_path, val_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"{p} does not exist -- run train/tokenization.py against the current "
                f"blend first (see docs/corpus_blend.md's 'Rebuilding it' section), or "
                f"pass a --tokens-dir that already has them"
            )

    if eos_token_id is None:
        eos_token_id = discover_eos_token_id()

    train_ids = np.load(train_path, mmap_mode="r")
    val_ids = np.load(val_path, mmap_mode="r")
    train_eos = int(np.count_nonzero(train_ids == eos_token_id))
    val_eos = int(np.count_nonzero(val_ids == eos_token_id))
    return TokenizedStats(
        total_tokens=int(len(train_ids) + len(val_ids)),
        train_tokens=int(len(train_ids)),
        val_tokens=int(len(val_ids)),
        eos_count=train_eos + val_eos,
    )


@dataclass(frozen=True)
class GapResult:
    """The manifest-vs-tokenized gap: two real counts of two different things, and how far
    apart they are. This is the figure that shipped wrong twice (0.46% and 0.42%, against a
    real gap of ~1.71%/1.90% at the time) -- see the module docstring."""

    manifest_total: int
    tokenized_total: int
    gap_tokens: int
    gap_pct: float


def compute_manifest_vs_tokenized_gap(manifest_total: int, tokenized_total: int) -> GapResult:
    """Pure arithmetic, isolated so it can be pinned by a test with no I/O at all.

    ``gap_pct`` is expressed as a percentage of ``manifest_total``, matching how both the
    Hub card and ``docs/corpus_blend.md`` have always stated it ("the difference is N
    tokens, or X% of <manifest total>") -- getting the denominator right is exactly what
    went wrong before (0.42%/0.46% were each computed, or mis-computed, against the wrong
    baseline / wrong arithmetic).
    """
    if manifest_total <= 0:
        raise ValueError(f"manifest_total must be positive, got {manifest_total}")
    gap_tokens = abs(manifest_total - tokenized_total)
    gap_pct = gap_tokens / manifest_total * 100
    return GapResult(
        manifest_total=manifest_total,
        tokenized_total=tokenized_total,
        gap_tokens=gap_tokens,
        gap_pct=gap_pct,
    )


# ---------------------------------------------------------------------------------------
# Card rendering. Numeric sections are computed from SOURCES / the manifest / TokenizedStats
# at call time. Prose sections that cannot be derived (the recipe-changed note, known
# limitations, cross-links) are preserved verbatim from the card currently on the Hub,
# fetched and read by hand while building this generator, per the plan's instruction not to
# discard content by regenerating from nothing.
# ---------------------------------------------------------------------------------------

FRONT_MATTER = """---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
pretty_name: TT-TNT Corpus Recipe
tags:
  - recipe
  - not-redistributed
  - reproducibility
  - pretraining-corpus
  - gutenberg
  - tinystories
  - wikipedia
  - tenstorrent
---"""

INTRO = """# TT-TNT Corpus Recipe

This is a **recipe**, not a corpus. It contains pinned source revisions and deterministic
scripts that reconstruct a ~400M-token English pretraining blend byte-for-byte, plus the
provenance record from the build that produced it. **No corpus text, tokenizer, or model
weights are stored in this repository.**

The blend is built for [`episod/tt-tnt`](https://huggingface.co/episod/tt-tnt) (public), a
hand-rolled, nanollama3-lineage, Llama-3-architecture language model trained with tt-metal's
`ttml` trainer on Tenstorrent Blackhole hardware. The design goal is a small model with an
oblique, observational voice -- closer to Fabre's insect notebooks or Fort's anomaly-collecting
than to a chat assistant -- built out of nine licence-audited sources rather than one scraped
dump.

Full source, including the training entrypoint, corpus pipeline, and every script named in
this card, lives at
[github.com/tsingletaryTT/tt-tnt](https://github.com/tsingletaryTT/tt-tnt)."""

# Preserved verbatim from the card currently on the Hub -- this is exactly the hand-written
# "recipe changed" note the plan says must survive regeneration.
RECIPE_CHANGED_NOTE = """## The recipe changed on 2026-08-14 -- read this before reproducing

**If you built this corpus before 2026-08-14, your `sha256` will not match the one recorded in
this card, and that is expected -- you built a different (older) recipe, not a wrong one.**

The blend previously carried **zero** document separators: `scripts/prepare_corpus.py` wrote
each document as `text + "\\n\\n"`, which spells a document boundary exactly the way a paragraph
break *inside* a document is spelled, so nothing downstream could tell them apart. That was a
regression against the older, TinyStories-only pipeline the currently-published model's
predecessor used, which carried 662,878 separators. `train/tokenization.py` then encoded the
corpus line-by-line and dropped blank lines, so those unmarked boundaries contributed zero
tokens: the trained model never saw an end-of-document token at all, which is the documented
cause of `tt-tnt-v1`'s mid-generation topic drift and its failure to terminate cleanly.

The fix, landed the same day: `scripts/prepare_corpus.py` now terminates every document with a
line holding exactly `</s>` (the trained tokenizer's eos token, id 2), `scripts/blend_corpus.py`
closes each source's truncated final pass with the same separator so the nine source-to-source
seams aren't unmarked transitions either, and `train/corpus.py` gained
`CorpusSource.rows_per_document` (64 for `poetry` only) so the one source whose upstream row is
a line of verse -- not a document -- doesn't fire an end-of-document token every ~7 words."""

# Preserved verbatim from the card currently on the Hub.
KNOWN_LIMITATIONS = """## Known limitations

These are documented plainly in the source repository, not smoothed over:

- **`flavour` sits close to its arithmetic ceiling.** It draws from only 7 books (Stein and
  Legge's I Ching translation) and its measured availability yields a hard ceiling close to its
  0.5% target share -- about **0.075 percentage points** of headroom. A further ~13% drop in
  measured availability would put even 0.5% out of reach at the current 4x repetition cap.
- **`spine` and `weird` share exactly one book.** Gutenberg text 30092, a 14-contributor
  anthology, is matched independently by a `spine` author (Hudson, W. H.) and a `weird` author
  (Blackwood, Algernon) -- the same underlying text can surface in both slices.
- **Charles Fort has exactly one book** in the source dataset (`sedthh/gutenberg_english`)
  despite being a design anchor for the `spine` slice's observational-mystical voice.
- **The shipped tokenizer was trained on an earlier revision of this blend.** There's a
  circular dependency: `tokenizer -> per-source token availability -> settled shares -> blend
  -> tokenizer`. Availability has to be measured in tokens, which needs a tokenizer; shares are
  settled against that availability; the blend realizes the shares; and retraining the
  tokenizer on the resulting blend would change availability again. The loop doesn't converge
  on its own, so it's deliberately cut: the shipped tokenizer reflects a slightly earlier state
  of the registry than the blend described in this card. This is called out in
  `docs/corpus_blend.md` as known and accepted, with the measured cost bounded. The
  document-separator fix does not deepen this limitation: `</s>` was already in the
  tokenizer's vocabulary as a special token, so the new separators encode to exactly one id
  each with no retrain required to represent them."""

# Preserved verbatim (cross-links).
LINKS = """## Link to the model

[`episod/tt-tnt`](https://huggingface.co/episod/tt-tnt) (public) is the model this corpus
recipe supports: a hand-rolled, nanollama3-lineage, Llama-3-architecture language model --
RoPE, RMSNorm, SwiGLU, grouped-query attention -- trained from random initialization with
tt-metal's `ttml` trainer on Tenstorrent Blackhole hardware. See the model card there for the
full training record and an honest evaluation of the result.

## Link to the code

[github.com/tsingletaryTT/tt-tnt](https://github.com/tsingletaryTT/tt-tnt) is the source
repository this recipe is copied out of -- training entrypoint, corpus pipeline, packaging, and
the full documented build, including the dead ends."""


def render_recipe_file_table(files: Sequence[RecipeFile] = RECIPE_FILES) -> str:
    """The "what's in this repository" table, derived from RECIPE_FILES itself so it can
    never list a file that isn't actually uploaded, or omit one that is."""
    lines = [
        "Copied as-is from a checkout of the `tt-tnt` code repository, directory structure "
        "preserved:",
        "",
        "| Path | What it is |",
        "|---|---|",
    ]
    for rf in files:
        lines.append(f"| `{rf.path}` | {rf.description} |")
    return "\n".join(lines)


def render_why_recipe_not_data(sources: Dict[str, CorpusSource] = SOURCES) -> str:
    """Why the blend can't ship as one file -- with the share-alike names and combined
    percentage derived from the registry rather than restated."""
    share_alike = sorted(
        (s for s in sources.values() if s.share_alike), key=lambda s: -s.target_share
    )
    combined_pct = sum(s.target_share for s in share_alike) * 100
    bullets = "\n".join(
        f"- `{s.name}` -- {format_share(s.target_share)} of the blend -- is "
        f"**{s.license_id}**"
        for s in share_alike
    )
    licence_names = " and ".join(s.license_id for s in share_alike)
    return f"""## Why a recipe and not the data

The blend cannot be redistributed as one file. {combined_pct:.0f}% of it is under
{len(share_alike)} **mutually incompatible** share-alike (copyleft) licences:

{bullets}

{licence_names} both require derivatives to be shared under their own terms; neither is
compatible with being folded into a single file under the other's licence, or under any third
licence. There is no legal way to concatenate them into one redistributable artifact. The
source repository's own generated licensing document says it plainly: **"We do not
redistribute the corpus."**

So this repository ships what actually can be shared: pinned dataset revisions, a
deterministic blending script, and a recorded SHA-256 so that anyone with access to the (still
separately licensed) upstream sources can reconstruct the exact same blend and verify it
byte-for-byte."""


def _format_measured_share(share: float) -> str:
    """Achieved shares are a real measurement, not a target -- rendered to 3 decimal places
    (never rounded to a whole percent, which is exactly the ``:.0%`` bug ``format_share``'s
    own docstring describes for the target-share column)."""
    return f"{share * 100:.3f}%"


def render_share_table(
    sources: Dict[str, CorpusSource] = SOURCES, manifest: Optional[dict] = None,
) -> str:
    """Target shares/licence/revision from ``train/corpus.py``; achieved share and real
    repetition from ``blend_manifest.json`` -- the registry is the target, the manifest is
    the outcome, exactly as ``docs/corpus_licensing.md`` puts it."""
    if manifest is None:
        manifest = _load_manifest()
    manifest_sources = manifest["sources"]

    lines = [
        "| Source | Slice | Target share | Licence | Pinned revision | Declared `upsample` "
        "| Achieved share | Real repetition |",
        "|---|---|---:|---|---|---:|---:|---:|",
    ]
    for s in sorted(sources.values(), key=lambda x: -x.target_share):
        rec = manifest_sources.get(s.name)
        if rec is None:
            raise KeyError(
                f"{s.name!r} is in train/corpus.py's SOURCES but missing from the "
                f"manifest -- re-run scripts/blend_corpus.py"
            )
        share_alike_tag = " (share-alike)" if s.share_alike else ""
        lines.append(
            f"| `{s.name}` | {s.slice} | {format_share(s.target_share)} | "
            f"{s.license_id}{share_alike_tag} | `{s.hf_revision}` | {s.upsample}x | "
            f"{_format_measured_share(rec['achieved_share'])} | "
            f"{rec['repetition_factor']}x |"
        )
    return "\n".join(lines)


def render_headline(manifest: dict) -> str:
    total = manifest["total_emitted_tokens"]
    budget = manifest["budget"]
    diff = manifest["total_vs_budget_tokens"]
    pct = manifest["total_vs_budget_pct"]
    return f"""## The headline number

**{total:,} tokens** against a **{budget:,}** token budget -- **{abs(diff):,} short, {pct}%**.

This is the sum of nine separate tokenizer calls, one per source, each over that source's own
emitted text (see "Honest measurements" below for the second, different count of the same
blend). `blend.txt` SHA-256 `{manifest['sha256']}`."""


def render_honest_measurements(manifest: dict, tokenized: TokenizedStats) -> str:
    """The section that shipped wrong twice. Every number here, including the gap
    percentage, is computed -- not transcribed -- from ``manifest`` and ``tokenized``."""
    gap = compute_manifest_vs_tokenized_gap(manifest["total_emitted_tokens"], tokenized.total_tokens)
    return f"""## Honest measurements

Two token counts exist for this corpus. Both are real and directly verifiable, and neither
corrects the other -- they measure two different things.

**{gap.manifest_total:,} tokens** is in `docs/measurements/blend_manifest.json`, written by the
blend build itself. It is the **sum of nine separate tokenizer calls**, one per source, each
over that source's own emitted text, chunked into paragraphs exactly the way
`scripts/measure_corpus.py` chunks a source file for its availability check.

**{tokenized.total_tokens:,} tokens** ({tokenized.train_tokens:,} train /
{tokenized.val_tokens:,} validation, of which {tokenized.eos_count:,} are the `</s>` document
separator) is what one tokenizer call over the finished, concatenated `blend.txt` actually
produces -- the number `train/tokenization.py` writes out and the number a training run over
this revision would actually read.

Why they differ: BPE merges do not cross an `encode()` call, so tokenizing nine chunks
separately and tokenizing their concatenation as one string can legitimately merge a different
set of byte pairs at every chunk boundary -- millions of them (`measure_corpus.py` splits on
every paragraph break, not just the eight source-to-source seams), each one a place a merge
can differ between the two measurements.

**The gap is {gap.gap_tokens:,} tokens, or {gap.gap_pct:.2f}%** of the manifest total. Use
{gap.manifest_total:,} for per-source blend provenance (how the blend was assembled); use
{tokenized.total_tokens:,} / {tokenized.train_tokens:,} / {tokenized.val_tokens:,} for what a
training run over this revision actually consumes."""


def render_licensing_section(sources: Dict[str, CorpusSource] = SOURCES) -> str:
    """Mirrors ``scripts/render_licensing.py``'s approach: licence facts live as DATA on
    ``CorpusSource`` so this section can't drift from the registry that defines them."""
    lines = ["## Licensing, stated honestly", "",
             "Per-source licences (also in `docs/corpus_licensing.md`, generated from the "
             "registry so it cannot drift from the code that defines it):", ""]
    for s in sorted(sources.values(), key=lambda x: -x.target_share):
        tag = ", share-alike" if s.share_alike else ""
        lines.append(f"- **`{s.name}`** ({format_share(s.target_share)}) -- **{s.license_id}**{tag}.")

    share_alike = [s for s in sources.values() if s.share_alike]
    if share_alike:
        names = ", ".join(f"`{s.name}`" for s in share_alike)
        licences = " or ".join(s.license_id for s in share_alike)
        lines += [
            "",
            f"**The share-alike sources ({names}) are why this repo exists in its current "
            "form.** Whether model weights trained on share-alike data constitute a Data "
            f"Derivative under {licences} is **unsettled law and practice.** This project "
            "does **not** assert that trained weights escape those obligations. Anyone "
            "publishing weights trained with this recipe should reach their own conclusion "
            "rather than inherit this one. This is a stated position, not legal advice.",
        ]
    lines += [
        "",
        "Project Gutenberg material is public domain as text; the aggregation datasets we "
        "fetch it through carry their own separate, permissive terms (recorded per-source "
        "above).",
    ]
    return "\n".join(lines)


def render_reproduction(manifest: dict) -> str:
    return f"""## Reproduction

Requires a checkout of the `tt-tnt` code repository (these files are copies of what lives
there at specific paths) with its Python dependencies installed. From the repo root:

```bash
python scripts/check_disk_space.py     # refuses to start if the volume is too full
python scripts/fetch_corpus.py         # downloads each source at its pinned revision
python scripts/prepare_corpus.py       # cleans/normalises fetched text, adds </s> boundaries
python scripts/measure_corpus.py       # -> docs/measurements/corpus_availability.json
python scripts/blend_corpus.py         # -> artifacts/corpus/blend.txt + both manifests
```

The blend is deterministic: same pinned sources, same availability report, same bytes, same
SHA-256, every time. Verify a reconstruction against the recorded checksum:

```bash
sha256sum artifacts/corpus/blend.txt
# expected: {manifest['sha256']}
```"""


def render_card(
    manifest: Optional[dict] = None,
    sources: Dict[str, CorpusSource] = SOURCES,
    tokenized: Optional[TokenizedStats] = None,
) -> str:
    """Assemble the full dataset card. ``tokenized=None`` renders a clearly-labeled
    placeholder for the "Honest measurements" section instead of a number -- used by
    ``--dry-run`` on a checkout that hasn't run tokenization, never by a real publish."""
    if manifest is None:
        manifest = _load_manifest()

    if tokenized is not None:
        honest = render_honest_measurements(manifest, tokenized)
    else:
        honest = (
            "## Honest measurements\n\n"
            "_(not available -- `artifacts/tokens-v3/{train,val}_ids.npy` were not found on "
            "this machine; run `train/tokenization.py` against the current blend to "
            "populate this section. This placeholder must never be published for real.)_"
        )

    sections = [
        FRONT_MATTER,
        "",
        INTRO,
        "",
        RECIPE_CHANGED_NOTE,
        "",
        render_why_recipe_not_data(sources),
        "",
        "## What's in this repository",
        "",
        render_recipe_file_table(),
        "",
        "**These scripts are not standalone.** `fetch_corpus.py`, `prepare_corpus.py`, "
        "`measure_corpus.py`, and `blend_corpus.py` import `train.paths` and (in "
        "`fetch_corpus.py`'s case) `scripts.build_gutenberg_catalogue` from the code "
        "repository -- neither is duplicated here. To actually run them, clone the "
        "`tt-tnt` code repository and either drop these files into place at the same "
        "relative paths or run them from that checkout directly.",
        "",
        render_headline(manifest),
        "",
        "## The share table",
        "",
        render_share_table(sources, manifest),
        "",
        "\"Real repetition\" is the fraction of a source's available text actually used "
        "(`emitted_tokens / available_tokens`), and it is not the same as `upsample`: "
        "`upsample` in the registry is a *ceiling* enforced by `measure_corpus.py`'s "
        "availability gate, not a claim about how much repetition is applied.",
        "",
        honest,
        "",
        render_licensing_section(sources),
        "",
        render_reproduction(manifest),
        "",
        KNOWN_LIMITATIONS,
        "",
        LINKS,
        "",
    ]
    return "\n".join(sections)


# ---------------------------------------------------------------------------------------
# Hub I/O
# ---------------------------------------------------------------------------------------

def _assert_visibility_unchanged(api, repo_id: str, when: str) -> None:
    """Read-only check that the repo's visibility is exactly ``EXPECTED_PRIVATE``.

    Called before AND after any write, so a write that somehow changed visibility (this
    script has no code path that does, but the check does not trust that claim blindly) is
    caught immediately rather than discovered later.
    """
    info = api.dataset_info(repo_id)
    if info.private is not EXPECTED_PRIVATE:
        raise RuntimeError(
            f"refusing to proceed ({when}): {repo_id} has private={info.private!r}, "
            f"expected {EXPECTED_PRIVATE!r}. This script must never change repo "
            f"visibility, and does not attempt to fix this automatically -- investigate "
            f"before doing anything else."
        )


def _print_upload_plan(repo_id: str, card: str) -> None:
    print(f"repo: {repo_id} (dataset; visibility must remain private={EXPECTED_PRIVATE})")
    print("files:")
    for rf in RECIPE_FILES:
        size = (ROOT / rf.path).stat().st_size
        print(f"  {rf.path:45s} {size:>10,} B")
    print(f"  {'README.md (generated card)':45s} {len(card.encode('utf-8')):>10,} B")


def cmd_dry_run(repo_id: str, tokens_dir: Path) -> int:
    _assert_recipe_is_redistributable()
    try:
        tokenized = compute_tokenized_stats(tokens_dir)
    except FileNotFoundError as e:
        print(f"warning: {e}", file=sys.stderr)
        tokenized = None
    manifest = _load_manifest()
    card = render_card(manifest, SOURCES, tokenized)
    _print_upload_plan(repo_id, card)
    print()
    print("--- generated card ---")
    print(card)
    print("[dry-run] no repo contacted, nothing uploaded.")
    return 0


def cmd_publish(repo_id: str, yes: bool, tokens_dir: Path) -> int:
    _assert_recipe_is_redistributable()
    manifest = _load_manifest()
    # A real publish must never ship the "not available" placeholder -- that would publish
    # a card making a promise (derived figures) it does not keep.
    tokenized = compute_tokenized_stats(tokens_dir)
    card = render_card(manifest, SOURCES, tokenized)
    _print_upload_plan(repo_id, card)

    if not yes:
        print("refusing to publish without --yes (use --dry-run to preview safely)",
              file=sys.stderr)
        return 2

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    _assert_visibility_unchanged(api, repo_id, when="before upload")

    for rf in RECIPE_FILES:
        print(f"uploading {rf.path} ...")
        api.upload_file(
            path_or_fileobj=str(ROOT / rf.path),
            path_in_repo=rf.path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Update recipe: {rf.path}",
        )

    print("uploading README.md (generated card) ...")
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Regenerate dataset card from current recipe",
    )

    _assert_visibility_unchanged(api, repo_id, when="after upload")
    print("done. Visibility is never changed by this script.")
    return 0


def cmd_verify(repo_id: str, tokens_dir: Path) -> int:
    """Read-only: download what's on the Hub, diff it against what the working tree would
    publish right now, and report exactly which files differ."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.dataset_info(repo_id)
    print(f"repo: {repo_id} (private={info.private!r}, expected {EXPECTED_PRIVATE!r})")
    ok = info.private is EXPECTED_PRIVATE
    if not ok:
        print(f"[FAIL] visibility: expected private={EXPECTED_PRIVATE!r}, got {info.private!r}")
    else:
        print(f"[PASS] visibility unchanged (private={EXPECTED_PRIVATE!r})")

    manifest = _load_manifest()
    tokenized = compute_tokenized_stats(tokens_dir)
    expected_card = render_card(manifest, SOURCES, tokenized)

    diffs: List[str] = []

    for rf in RECIPE_FILES:
        local_bytes = (ROOT / rf.path).read_bytes()
        try:
            hub_path = hf_hub_download(repo_id, rf.path, repo_type="dataset")
        except Exception as e:  # noqa: BLE001 -- report as a diff, not a crash
            diffs.append(rf.path)
            print(f"[FAIL] {rf.path}: could not download from Hub ({e})")
            continue
        hub_bytes = Path(hub_path).read_bytes()
        if hub_bytes == local_bytes:
            print(f"[PASS] {rf.path}")
        else:
            diffs.append(rf.path)
            print(f"[FAIL] {rf.path}: Hub copy differs from the working tree "
                  f"({len(hub_bytes):,} B on Hub vs {len(local_bytes):,} B locally)")

    try:
        readme_path = hf_hub_download(repo_id, "README.md", repo_type="dataset")
        hub_readme = Path(readme_path).read_text(encoding="utf-8")
    except Exception as e:  # noqa: BLE001 -- report as a diff, not a crash
        diffs.append("README.md")
        print(f"[FAIL] README.md: could not download from Hub ({e})")
    else:
        if hub_readme == expected_card:
            print("[PASS] README.md (generated card)")
        else:
            diffs.append("README.md")
            print("[FAIL] README.md: Hub card differs from what the working tree would "
                  "generate right now")

    if not ok or diffs:
        print(f"\nDIVERGENCE DETECTED. Differing paths: {diffs if diffs else '(visibility only)'}",
              file=sys.stderr)
        return 1
    print("\nno divergence: the Hub matches the working tree exactly.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=REPO_ID_DEFAULT, help=f"Target dataset repo (default: {REPO_ID_DEFAULT}).")
    p.add_argument("--tokens-dir", default=str(TOKENS_DIR_DEFAULT),
                    help=f"Directory holding train_ids.npy/val_ids.npy for the current "
                         f"blend (default: {TOKENS_DIR_DEFAULT.relative_to(ROOT)}).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--publish", action="store_true", help="Upload the recipe and a freshly generated card.")
    mode.add_argument("--dry-run", action="store_true", help="Preview a publish; never contacts the Hub for writes.")
    mode.add_argument("--verify", action="store_true", help="Read-only: diff the Hub against the working tree (default).")
    p.add_argument("--yes", action="store_true", help="Required for --publish to actually write to the Hub.")
    args = p.parse_args(argv)

    tokens_dir = Path(args.tokens_dir)

    if args.publish:
        return cmd_publish(args.repo_id, yes=args.yes, tokens_dir=tokens_dir)
    if args.dry_run:
        return cmd_dry_run(args.repo_id, tokens_dir=tokens_dir)
    # --verify is the default: runnable with no flags right after a corpus rebuild.
    return cmd_verify(args.repo_id, tokens_dir=tokens_dir)


if __name__ == "__main__":
    raise SystemExit(main())
