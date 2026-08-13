<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Corpus Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the corpus source registry and the fetch/prepare/measure pipeline, ending with a measured answer to "how many tokens does each slice actually contain?" — the gate the design spec requires before any blend ratio is committed.

**Architecture:** A typed registry (`train/corpus.py`) mirroring `train/sizes.py`, holding one `CorpusSource` per source with its licence, functional slice, target share and pinned fetch spec. A metadata-only Gutenberg catalogue is built first — reading just the `METADATA` parquet column rather than 10.75 GB of text — so author presence and per-slice book counts are known cheaply. Only then are texts fetched, normalised, and measured.

**Tech Stack:** Python 3.10+, `huggingface_hub`, `datasets`, `pyarrow`, `pytest`. No hardware.

**Scope note:** This plan deliberately stops at measurement. Blending, the tokenizer retrain, generated licensing and the frozen evaluation prompt set are a **second plan**, because the design spec states the slice shares are "targets to be revised against measured availability" — writing blend ratios before Task 5 has run would be inventing numbers.

## Global Constraints

Copied from `docs/superpowers/specs/2026-08-12-diverse-corpus-design.md`:

- **SPDX header pair on every new file**; Python 3.10+.
- **No bare `assert` for guards** in production code (tests may assert freely).
- **`pyproject.toml` must NOT be modified.**
- **Never write under `artifacts/checkpoints/` or `artifacts/hf/`** — protected baseline evidence. `train/paths.py` already guards this; use `write_dir`/`shared_dir`, never hand-built paths.
- **We do not redistribute the corpus.** Nothing under `artifacts/` is committed. Fetches pin a revision; the recipe is the artifact.
- **Licensing is data, not prose** — every source carries its licence in the registry. No licence text is hand-written into docs by this plan.
- **Gutenberg's texts are public domain; the packaging is separately licensed** (`sedthh/gutenberg_english` MIT, `biglam/gutenberg-poetry-corpus` CC0). Record both.
- **If a step produces a number that decides pass/fail, it must be a test**, not a shell command in prose.
- **DELETE NOTHING.** No task in this plan removes, prunes, or overwrites any pre-existing
  file or cache. That includes `~/.cache/huggingface` (1.2 TB, someone's working set),
  `artifacts/checkpoints*/`, `artifacts/hf*/`, and any dataset cache. If disk pressure
  appears, **stop and report it** — reclaiming space is the human's decision, never the
  implementer's.
- **Disk is tight: check before downloading.** The volume was at **98% (90 GB free)** when
  this plan was written. Task 0 gates on free space and every fetch step re-checks. Streaming
  (`streaming=True`) is used everywhere so the full 10.75 GB Gutenberg dataset is never
  materialised locally; only the filtered subset is written.
- Existing artifact directory names are fixed: `artifacts/raw/`, `artifacts/corpus/`, `artifacts/tokenizer/`, `artifacts/tokens/` — all shared across model sizes (see `train/paths.py::SHARED_KINDS`).

---

## File Structure

| File | Responsibility |
|---|---|
| `train/corpus.py` | The registry. `CorpusSource`, `SOURCES`, `get_source()`, slice constants. Pure data + arithmetic; no I/O. |
| `tests/test_corpus.py` | Registry anti-drift: every source has a licence, slice, fetch spec; shares sum to 100. |
| `scripts/build_gutenberg_catalogue.py` | Metadata-only fetch → `artifacts/raw/gutenberg_catalogue.jsonl`. Verifies named authors exist. |
| `tests/test_gutenberg_catalogue.py` | Selector logic (author/bookshelf matching) tested against fixture rows. |
| `scripts/fetch_corpus.py` | Per-source text download at pinned revision → `artifacts/raw/<source>/`. |
| `scripts/prepare_corpus.py` | Normalise each source → `artifacts/corpus/<source>.txt`. |
| `tests/test_prepare_corpus.py` | Normalisation: boilerplate stripping, whitespace, encoding. |
| `scripts/measure_corpus.py` | Token counts per source vs target share. **The scarcity gate.** |
| `tests/test_measure_corpus.py` | Gate logic: under-supply detection, upsample cap enforcement. |

---

## Task 0: Disk-space preflight

**Files:**
- Create: `scripts/check_disk_space.py`
- Test: `tests/test_check_disk_space.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `free_bytes(path) -> int`; `check_space(path, required_gb) -> tuple[bool, str]`.

**Why this is Task 0.** The volume was at 98% with 90 GB free when this plan was written, and
the HF cache already holds 1.2 TB. This plan downloads several GB. A fetch that fills the
volume damages unrelated work on the machine, so space is checked before anything is written
and re-checked before each fetch. **Nothing is ever deleted to make room** — the script
reports and exits non-zero, and reclaiming space is the human's call.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_check_disk_space.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Preflight tests. The gate must be honest in both directions: it must refuse when space
is short, and it must not block when space is fine."""
from pathlib import Path

import pytest

from scripts.check_disk_space import check_space, free_bytes


def test_free_bytes_is_positive_for_a_real_path(tmp_path: Path):
    assert free_bytes(tmp_path) > 0


def test_check_space_passes_when_requirement_is_tiny(tmp_path: Path):
    ok, msg = check_space(tmp_path, required_gb=0.000001)
    assert ok
    assert "free" in msg


def test_check_space_fails_when_requirement_is_absurd(tmp_path: Path):
    ok, msg = check_space(tmp_path, required_gb=10_000_000)
    assert not ok
    assert "need" in msg.lower()


def test_check_space_never_suggests_deleting_anything(tmp_path: Path):
    """The message must not invite the implementer to reclaim space on its own."""
    _, msg = check_space(tmp_path, required_gb=10_000_000)
    for forbidden in ("rm ", "delete", "prune", "clear the cache"):
        assert forbidden not in msg.lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_check_disk_space.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.check_disk_space'`

- [ ] **Step 3: Implement the preflight**

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Refuse to start a download that could fill the volume.

This machine's root filesystem was at 98% (90 GB free) when this plan was written, with a
1.2 TB Hugging Face cache belonging to other work. Filling it would damage unrelated
projects.

This script REPORTS and EXITS. It never deletes, prunes, or relocates anything: reclaiming
space is a human decision, and an automated tool guessing which gigabytes are expendable is
exactly how someone's dataset disappears.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

#: Rough budget for the full discovery pipeline: filtered Gutenberg subsets, Simple
#: Wikipedia, the normalised text copies, and headroom. Streaming keeps the full 10.75 GB
#: dataset off local disk, so this is far below the raw dataset size.
DEFAULT_REQUIRED_GB = 45.0


def free_bytes(path: Path) -> int:
    """Bytes free on the filesystem holding ``path``."""
    return shutil.disk_usage(Path(path)).free


def check_space(path: Path, required_gb: float) -> tuple:
    """(ok, message). Never proposes reclaiming space."""
    free_gb = free_bytes(path) / 1e9
    total_gb = shutil.disk_usage(Path(path)).total / 1e9
    pct_used = 100.0 * (1.0 - free_gb / total_gb) if total_gb else 0.0
    if free_gb >= required_gb:
        return True, (f"{free_gb:,.1f} GB free of {total_gb:,.1f} GB "
                      f"({pct_used:.0f}% used); need {required_gb:,.1f} GB")
    return False, (f"INSUFFICIENT SPACE: {free_gb:,.1f} GB free of {total_gb:,.1f} GB "
                   f"({pct_used:.0f}% used), need {required_gb:,.1f} GB. "
                   f"Stopping. Report this rather than making room.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", type=Path, default=Path.cwd())
    p.add_argument("--required-gb", type=float, default=DEFAULT_REQUIRED_GB)
    args = p.parse_args()
    ok, msg = check_space(args.path, args.required_gb)
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_check_disk_space.py -q`
Expected: PASS.

- [ ] **Step 5: Run the preflight for real**

Run: `python scripts/check_disk_space.py`
Expected: prints free/total/percentage. **If it exits 1, stop the plan here and report to the
human.** Do not proceed to any fetch task, and do not delete anything to make room.

- [ ] **Step 6: Commit**

```bash
git add scripts/check_disk_space.py tests/test_check_disk_space.py
git commit -m "feat: disk-space preflight; never reclaims, only reports"
```

---

## Task 1: The corpus source registry

**Files:**
- Create: `train/corpus.py`
- Test: `tests/test_corpus.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `CorpusSource` (frozen dataclass), `SOURCES: Dict[str, CorpusSource]`, `get_source(name) -> CorpusSource`, `SLICES: frozenset[str]`, `total_target_share() -> float`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_corpus.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_corpus.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'train.corpus'`

- [ ] **Step 3: Implement the registry**

```python
# train/corpus.py
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
        hf_revision="main",
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
        hf_revision="main",
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
        hf_revision="main",
        license_id="MIT (packaging); public domain (texts)",
        license_url="https://huggingface.co/datasets/sedthh/gutenberg_english",
        attribution="Project Gutenberg via sedthh/gutenberg_english",
        license_note="MIT covers the aggregation; the underlying pre-1929 texts are public domain.",
        authors=["Fabre, Jean-Henri", "Maeterlinck, Maurice", "Fort, Charles",
                 "Hodgson, William Hope"],
        rationale="Observational-mystical. Fabre is field observation that is ALREADY "
                  "agentic tool-use theatre; Fort is the same method applied to the impossible.",
    ),
    "folklore": CorpusSource(
        name="folklore",
        slice="folklore",
        target_share=0.08,
        hf_repo="sedthh/gutenberg_english",
        hf_revision="main",
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
        hf_revision="main",
        license_id="MIT (packaging); public domain (texts)",
        license_url="https://huggingface.co/datasets/sedthh/gutenberg_english",
        attribution="Project Gutenberg via sedthh/gutenberg_english",
        license_note="MIT covers the aggregation; the underlying pre-1929 texts are public domain.",
        authors=["Blackwood, Algernon", "Dunsany", "Machen, Arthur", "Browne, Thomas"],
        rationale="Weird fiction and baroque prose. Unambiguously PD, unlike Lovecraft.",
    ),
    "poetry": CorpusSource(
        name="poetry",
        slice="weird",
        target_share=0.01,
        hf_repo="biglam/gutenberg-poetry-corpus",
        hf_revision="main",
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
        hf_revision="main",
        license_id="MIT (packaging); public domain (texts)",
        license_url="https://huggingface.co/datasets/sedthh/gutenberg_english",
        attribution="Project Gutenberg via sedthh/gutenberg_english",
        license_note="MIT covers the aggregation; the underlying pre-1929 texts are public domain.",
        bookshelves=["How To", "Cookery", "Manuals"],
        rationale="Plan -> act -> observe -> report as a SHAPE. Stands in for IF transcripts, "
                  "which need per-game licence vetting and are deferred to a later release.",
    ),
    "flavour": CorpusSource(
        name="flavour",
        slice="flavour",
        target_share=0.02,
        hf_repo="sedthh/gutenberg_english",
        hf_revision="main",
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_corpus.py -q`
Expected: PASS, all tests.

- [ ] **Step 5: Run the whole suite to check nothing regressed**

Run: `python -m pytest -q`
Expected: all existing tests still pass (257 at the time of writing) plus the new ones.

- [ ] **Step 6: Commit**

```bash
git add train/corpus.py tests/test_corpus.py
git commit -m "feat: corpus source registry with licence as data"
```

---

## Task 2: The Gutenberg catalogue, and verifying the authors exist

**Files:**
- Create: `scripts/build_gutenberg_catalogue.py`
- Test: `tests/test_gutenberg_catalogue.py`

**Interfaces:**
- Consumes: `train.corpus.SOURCES`, `CorpusSource.authors`, `CorpusSource.bookshelves`.
- Produces: `matches_source(metadata: dict, source: CorpusSource) -> bool`; a catalogue file at `artifacts/raw/gutenberg_catalogue.jsonl` with one JSON object per book (`text_id`, `title`, `authors`, `bookshelves`, `subjects`).

**Why this task is first among the I/O tasks.** The design spec records an unverified assumption: that Fabre, Fort, Hodgson, Maeterlinck, Stein, Blackwood, Dunsany, Machen and Browne are actually present in `sedthh/gutenberg_english`. The HF search index returns 500 for that dataset, so it could not be checked while writing the spec. If they are absent the composition changes, so this must be settled **before** any text is downloaded. Reading only the `METADATA` column keeps it cheap — the dataset is 10.75 GB, the metadata is a small fraction of that.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_gutenberg_catalogue.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Selector logic, tested against fixture rows rather than the network."""
import pytest
from train.corpus import CorpusSource
from scripts.build_gutenberg_catalogue import matches_source

FABRE = {"authors": "Fabre, Jean-Henri", "title": "The Life of the Spider",
         "bookshelves": "Science", "subjects": "Spiders"}
STEIN = {"authors": "Stein, Gertrude", "title": "Tender Buttons",
         "bookshelves": "", "subjects": "Prose poetry"}
KIDLIT = {"authors": "Nesbit, E. (Edith)", "title": "Five Children and It",
          "bookshelves": "Children's Literature", "subjects": "Fantasy"}
UNRELATED = {"authors": "Smith, John", "title": "A Treatise on Drainage",
             "bookshelves": "Engineering", "subjects": "Sewerage"}


def _src(**kw):
    base = dict(name="t", slice="spine", target_share=0.1,
                hf_repo="r", hf_revision="rev")
    base.update(kw)
    return CorpusSource(**base)


def test_author_match_is_case_insensitive_substring():
    src = _src(authors=["fabre, jean-henri"])
    assert matches_source(FABRE, src)
    assert not matches_source(UNRELATED, src)


def test_surname_only_author_entry_matches():
    """'Dunsany' must match 'Dunsany, Lord', which is how Gutenberg records him."""
    src = _src(authors=["Dunsany"])
    assert matches_source({"authors": "Dunsany, Lord", "title": "", "bookshelves": "",
                           "subjects": ""}, src)


def test_bookshelf_match_is_case_insensitive_substring():
    src = _src(bookshelves=["children's literature"])
    assert matches_source(KIDLIT, src)
    assert not matches_source(UNRELATED, src)


def test_author_or_bookshelf_matches_when_both_given():
    """A source listing both selects the union, not the intersection."""
    src = _src(authors=["Stein, Gertrude"], bookshelves=["Children's Literature"])
    assert matches_source(STEIN, src)
    assert matches_source(KIDLIT, src)
    assert not matches_source(UNRELATED, src)


def test_no_selectors_matches_nothing_here():
    """A Gutenberg source with no selectors would pull all 48k books by accident."""
    src = _src()
    assert not matches_source(FABRE, src)


def test_missing_metadata_fields_do_not_raise():
    src = _src(authors=["Fabre"])
    assert not matches_source({}, src)
    assert not matches_source({"authors": None}, src)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_gutenberg_catalogue.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.build_gutenberg_catalogue'`

- [ ] **Step 3: Implement the catalogue builder**

Create `scripts/__init__.py` if absent (empty file with the SPDX pair) so `scripts.` is importable by tests.

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Build a metadata-only catalogue of sedthh/gutenberg_english, and verify our authors exist.

WHY METADATA ONLY. The dataset is 48,284 books and 10.75 GB. Everything this plan needs to
decide -- do the named authors exist, how many books does each slice select, is any slice
short -- lives in the METADATA column. Reading one column instead of the whole dataset turns
a multi-gigabyte download into a small one, and it happens BEFORE any text is fetched, so a
composition that cannot be satisfied fails early and cheaply.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.corpus import SOURCES, CorpusSource  # noqa: E402
from train.paths import shared_dir  # noqa: E402

GUTENBERG_REPO = "sedthh/gutenberg_english"
CATALOGUE_NAME = "gutenberg_catalogue.jsonl"


def _norm(value: Optional[str]) -> str:
    return (value or "").casefold()


def matches_source(metadata: Dict[str, object], source: CorpusSource) -> bool:
    """True when a catalogue row belongs to ``source``.

    Author and bookshelf selectors are a UNION: a source listing both takes rows matching
    either. Matching is case-insensitive substring, because Gutenberg records names as
    "Dunsany, Lord" and "Fabre, Jean-Henri" and a registry entry should not have to
    reproduce the punctuation exactly.

    A source with no selectors matches nothing. That is deliberate: an empty selector on a
    48,000-book dataset would silently pull everything.
    """
    if not source.authors and not source.bookshelves:
        return False
    authors = _norm(metadata.get("authors"))
    shelves = _norm(metadata.get("bookshelves"))
    for wanted in source.authors:
        if _norm(wanted) in authors:
            return True
    for wanted in source.bookshelves:
        if _norm(wanted) in shelves:
            return True
    return False


def gutenberg_sources() -> Dict[str, CorpusSource]:
    """Registry sources served by the Gutenberg dataset."""
    return {n: s for n, s in SOURCES.items() if s.hf_repo == GUTENBERG_REPO}


def iter_metadata(revision: str) -> Iterable[Dict[str, object]]:
    """Yield one metadata dict per book, reading only the METADATA column."""
    from datasets import load_dataset

    ds = load_dataset(
        GUTENBERG_REPO, split="train", revision=revision, streaming=True,
        columns=["METADATA"],
    )
    for row in ds:
        raw = row.get("METADATA")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                continue
        if isinstance(raw, dict):
            yield raw


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--revision", default="main")
    p.add_argument("--out", type=Path, default=None,
                   help="Catalogue path (default: artifacts/raw/gutenberg_catalogue.jsonl)")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after N rows (0 = all). For smoke-testing the pipeline.")
    args = p.parse_args()

    out = args.out or (shared_dir("raw") / CATALOGUE_NAME)
    out.parent.mkdir(parents=True, exist_ok=True)

    sources = gutenberg_sources()
    counts = {name: 0 for name in sources}
    total = 0

    with out.open("w", encoding="utf-8") as fh:
        for md in iter_metadata(args.revision):
            total += 1
            record = {
                "text_id": md.get("text_id"),
                "title": md.get("title"),
                "authors": md.get("authors"),
                "bookshelves": md.get("bookshelves"),
                "subjects": md.get("subjects"),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            for name, src in sources.items():
                if matches_source(md, src):
                    counts[name] += 1
            if args.limit and total >= args.limit:
                break

    print(f"catalogued {total:,} books -> {out}")
    print()
    print(f"{'source':22} {'books':>7}  selectors")
    print("-" * 60)
    for name, src in sorted(sources.items()):
        sel = ", ".join(src.authors + src.bookshelves)[:34]
        print(f"{name:22} {counts[name]:>7}  {sel}")

    missing = [n for n, c in counts.items() if c == 0]
    if missing:
        print()
        print(f"WARNING: these sources selected ZERO books: {missing}")
        print("The composition assumes they exist. Investigate before fetching text.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_gutenberg_catalogue.py -q`
Expected: PASS, all tests.

- [ ] **Step 5: Smoke-test the script against a small slice of the real dataset**

Run: `python scripts/build_gutenberg_catalogue.py --limit 2000`
Expected: prints a per-source book count table. With only 2,000 of 48,284 books some sources will legitimately show 0 and the script will exit 1 — that is the warning working, not a failure.

- [ ] **Step 6: Build the full catalogue and record the answer**

Run: `python scripts/build_gutenberg_catalogue.py`
Expected: catalogues ~48,284 books and prints a non-zero count for **every** Gutenberg-backed source.

**This step answers the spec's open question.** Record the printed table verbatim in the commit message. If any source shows zero, do not proceed to Task 3 — report it, because the composition needs revising and that is a decision for the human, not a workaround for the implementer. Specifically: if an author is absent, check whether Gutenberg records them under a different form (initials, alternate spelling) before concluding the work is not there.

- [ ] **Step 7: Commit**

```bash
git add scripts/build_gutenberg_catalogue.py scripts/__init__.py tests/test_gutenberg_catalogue.py
git commit -m "feat: gutenberg metadata catalogue; verify named authors exist

Reads only the METADATA column of a 10.75 GB dataset so author presence and
per-slice book counts are known before any text is downloaded.

Measured counts: <paste the table from Step 6>"
```

---

## Task 3: Fetch source texts

**Files:**
- Create: `scripts/fetch_corpus.py`

**Interfaces:**
- Consumes: `train.corpus.SOURCES`, `scripts.build_gutenberg_catalogue.matches_source`, `train.paths.shared_dir`.
- Produces: `fetch_source(source: CorpusSource, dest: Path, limit_rows: int = 0) -> int` returning the number of documents written; raw text at `artifacts/raw/<source>/text.jsonl` with one `{"text": ...}` per line.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fetch_corpus.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Fetch-layer tests. The network is not touched: row iteration is injected."""
import json
from pathlib import Path

from train.corpus import CorpusSource
from scripts.fetch_corpus import write_documents


def _src(**kw):
    base = dict(name="t", slice="spine", target_share=0.1, hf_repo="r", hf_revision="rev")
    base.update(kw)
    return CorpusSource(**base)


def test_write_documents_writes_one_json_object_per_line(tmp_path: Path):
    rows = [{"text": "alpha"}, {"text": "beta"}]
    n = write_documents(iter(rows), tmp_path / "text.jsonl")
    assert n == 2
    lines = (tmp_path / "text.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["text"] for x in lines] == ["alpha", "beta"]


def test_write_documents_skips_empty_and_whitespace_only(tmp_path: Path):
    rows = [{"text": "alpha"}, {"text": "   "}, {"text": ""}, {"text": None}, {"text": "beta"}]
    n = write_documents(iter(rows), tmp_path / "text.jsonl")
    assert n == 2


def test_write_documents_creates_parent_directories(tmp_path: Path):
    dest = tmp_path / "deep" / "nested" / "text.jsonl"
    assert write_documents(iter([{"text": "x"}]), dest) == 1
    assert dest.is_file()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_fetch_corpus.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.fetch_corpus'`

- [ ] **Step 3: Implement the fetcher**

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Fetch each registered source's text at its pinned revision.

Writes one JSON object per document to artifacts/raw/<source>/text.jsonl. Nothing here is
committed: the project ships a recipe, not a corpus, because CC-BY-SA-3.0 and
CDLA-Sharing-1.0 are not obviously compatible terms on one redistributed work.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_gutenberg_catalogue import GUTENBERG_REPO, matches_source  # noqa: E402
from train.corpus import SOURCES, CorpusSource, get_source  # noqa: E402
from train.paths import shared_dir  # noqa: E402

#: Column holding the document body, per dataset.
TEXT_COLUMN = {
    "sedthh/gutenberg_english": "TEXT",
    "roneneldan/TinyStories": "text",
    "wikimedia/wikipedia": "text",
    "biglam/gutenberg-poetry-corpus": "line",
}


def write_documents(rows: Iterable[Dict[str, object]], dest: Path) -> int:
    """Write ``{"text": ...}`` per line, skipping empties. Returns documents written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with dest.open("w", encoding="utf-8") as fh:
        for row in rows:
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1
    return written


def iter_source_rows(source: CorpusSource, limit_rows: int = 0) -> Iterator[Dict[str, object]]:
    """Stream a source's rows, normalised to ``{"text": str}`` and filtered if Gutenberg."""
    from datasets import load_dataset

    column = TEXT_COLUMN.get(source.hf_repo)
    if column is None:
        raise ValueError(
            f"no text column registered for {source.hf_repo}; add it to TEXT_COLUMN"
        )

    kwargs = {"split": source.hf_split, "revision": source.hf_revision, "streaming": True}
    if source.hf_config:
        kwargs["name"] = source.hf_config
    ds = load_dataset(source.hf_repo, **kwargs)

    seen = 0
    for row in ds:
        if source.hf_repo == GUTENBERG_REPO:
            md = row.get("METADATA")
            if isinstance(md, str):
                try:
                    md = json.loads(md)
                except json.JSONDecodeError:
                    continue
            if not isinstance(md, dict) or not matches_source(md, source):
                continue
        text = row.get(column)
        if not isinstance(text, str):
            continue
        yield {"text": text}
        seen += 1
        if limit_rows and seen >= limit_rows:
            return


def fetch_source(source: CorpusSource, dest: Optional[Path] = None,
                 limit_rows: int = 0) -> int:
    """Fetch one source to ``artifacts/raw/<name>/text.jsonl``. Returns documents written."""
    target = dest or (shared_dir("raw") / source.name / "text.jsonl")
    return write_documents(iter_source_rows(source, limit_rows), target)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", default=None,
                   help="Source name (repeatable). Default: all registered sources.")
    p.add_argument("--limit-rows", type=int, default=0,
                   help="Cap documents per source (0 = all). For smoke tests.")
    args = p.parse_args()

    names = args.source or sorted(SOURCES)
    for name in names:
        try:
            src = get_source(name)
        except KeyError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"fetching {name} from {src.hf_repo}@{src.hf_revision} ...", flush=True)
        n = fetch_source(src, limit_rows=args.limit_rows)
        print(f"  {n:,} documents")
        if n == 0:
            print(f"  WARNING: {name} produced no documents", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_fetch_corpus.py -q`
Expected: PASS.

- [ ] **Step 5: Smoke-test against the network with a small cap**

Run: `python scripts/fetch_corpus.py --source poetry --limit-rows 500`
Expected: prints `500 documents` and creates `artifacts/raw/poetry/text.jsonl`.

- [ ] **Step 6: Commit**

```bash
git add scripts/fetch_corpus.py tests/test_fetch_corpus.py
git commit -m "feat: fetch corpus sources at pinned revisions"
```

---

## Task 4: Normalise sources to plain text

**Files:**
- Create: `scripts/prepare_corpus.py`
- Test: `tests/test_prepare_corpus.py`

**Interfaces:**
- Consumes: raw `text.jsonl` from Task 3.
- Produces: `normalise(text: str) -> str`; `strip_gutenberg_boilerplate(text: str) -> str`; `prepare_source(name: str, src: Path, dest: Path) -> int` returning lines written; normalised text at `artifacts/corpus/<source>.txt`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prepare_corpus.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Normalisation tests.

The Gutenberg boilerplate test matters for licensing, not tidiness: PG applies a trademark
licence to its headers and footers while the underlying pre-1929 text is public domain.
Stripping them is what keeps the "public domain texts" claim accurate.
"""
from scripts.prepare_corpus import normalise, strip_gutenberg_boilerplate

HEADED = """*** START OF THE PROJECT GUTENBERG EBOOK THE LIFE OF THE SPIDER ***

The real text begins here.
And continues.

*** END OF THE PROJECT GUTENBERG EBOOK THE LIFE OF THE SPIDER ***
Produced by Some Volunteer."""


def test_strips_gutenberg_start_and_end_markers():
    out = strip_gutenberg_boilerplate(HEADED)
    assert "The real text begins here." in out
    assert "PROJECT GUTENBERG EBOOK" not in out
    assert "Produced by Some Volunteer." not in out


def test_leaves_text_without_markers_untouched():
    plain = "No markers here.\nJust prose."
    assert strip_gutenberg_boilerplate(plain) == plain


def test_normalise_collapses_carriage_returns():
    assert "\r" not in normalise("line one\r\nline two\r\n")


def test_normalise_collapses_runs_of_blank_lines():
    assert normalise("a\n\n\n\n\nb") == "a\n\nb"


def test_normalise_strips_trailing_whitespace_per_line():
    assert normalise("a   \nb\t\n") == "a\nb"


def test_normalise_is_idempotent():
    once = normalise("a\r\n\n\n\nb   \n")
    assert normalise(once) == once
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_prepare_corpus.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.prepare_corpus'`

- [ ] **Step 3: Implement normalisation**

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Normalise fetched documents into one plain-text file per source.

Boilerplate stripping is a LICENSING step, not a cosmetic one: Project Gutenberg applies a
trademark licence to its headers and footers, while the underlying pre-1929 texts are public
domain. Removing them is what makes "public domain texts" an accurate claim.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.corpus import SOURCES, get_source  # noqa: E402
from train.paths import shared_dir  # noqa: E402

_START = re.compile(r"^\*\*\*\s*START OF TH(?:E|IS) PROJECT GUTENBERG EBOOK.*$",
                    re.IGNORECASE | re.MULTILINE)
_END = re.compile(r"^\*\*\*\s*END OF TH(?:E|IS) PROJECT GUTENBERG EBOOK.*$",
                  re.IGNORECASE | re.MULTILINE)
_BLANKS = re.compile(r"\n{3,}")
_TRAILING = re.compile(r"[ \t]+$", re.MULTILINE)


def strip_gutenberg_boilerplate(text: str) -> str:
    """Keep only what lies between the PG start and end markers, when present."""
    start = _START.search(text)
    if start:
        text = text[start.end():]
    end = _END.search(text)
    if end:
        text = text[: end.start()]
    return text.strip("\n")


def normalise(text: str) -> str:
    """CRLF -> LF, strip trailing whitespace, collapse blank-line runs to one."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING.sub("", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip("\n")


def prepare_source(name: str, src: Path, dest: Path) -> int:
    """Normalise one source's raw jsonl into a plain-text file. Returns documents kept."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with src.open("r", encoding="utf-8") as fin, dest.open("w", encoding="utf-8") as fout:
        for line in fin:
            try:
                text = json.loads(line).get("text", "")
            except json.JSONDecodeError:
                continue
            text = normalise(strip_gutenberg_boilerplate(text))
            if not text:
                continue
            fout.write(text)
            fout.write("\n\n")
            kept += 1
    return kept


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", default=None)
    args = p.parse_args()

    names = args.source or sorted(SOURCES)
    for name in names:
        try:
            get_source(name)
        except KeyError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        src = shared_dir("raw") / name / "text.jsonl"
        if not src.is_file():
            print(f"skipping {name}: {src} not found (run fetch_corpus.py first)")
            continue
        dest = shared_dir("corpus") / f"{name}.txt"
        n = prepare_source(name, src, dest)
        size_mb = dest.stat().st_size / 1e6
        print(f"{name:22} {n:>7,} docs -> {dest.name} ({size_mb:,.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_prepare_corpus.py -q`
Expected: PASS.

- [ ] **Step 5: Run against the smoke-fetched poetry source**

Run: `python scripts/prepare_corpus.py --source poetry`
Expected: prints a document count and creates `artifacts/corpus/poetry.txt`.

- [ ] **Step 6: Commit**

```bash
git add scripts/prepare_corpus.py tests/test_prepare_corpus.py
git commit -m "feat: normalise sources; strip Gutenberg boilerplate"
```

---

## Task 5: Measure availability — the scarcity gate

**Files:**
- Create: `scripts/measure_corpus.py`
- Test: `tests/test_measure_corpus.py`

**Interfaces:**
- Consumes: `artifacts/corpus/<source>.txt`, `train.corpus.SOURCES`.
- Produces: `required_tokens(source, total_budget) -> int`; `achievable_tokens(available, upsample) -> int`; `shortfall_report(available: Dict[str, int], total_budget: int, upsample_cap: int) -> List[Shortfall]`; a report at `docs/measurements/corpus_availability.json`.

**This is the gate the spec requires.** The design states the slice shares are "targets to be revised against measured availability". This task produces that measurement and fails loudly when a slice cannot be filled within its upsample cap, so ratios are revised on evidence rather than discovered to be unsatisfiable during a training run.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_measure_corpus.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The scarcity gate.

Pure arithmetic over measured token counts. The point is that a slice which cannot be filled
within its upsample cap is reported BEFORE ratios are committed, rather than discovered when
a training run produces a model dominated by whatever was actually plentiful.
"""
import pytest
from train.corpus import CorpusSource
from scripts.measure_corpus import achievable_tokens, required_tokens, shortfall_report


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_measure_corpus.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.measure_corpus'`

- [ ] **Step 3: Implement the gate**

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Measure how many tokens each source actually supplies, against its target share.

THE SCARCITY GATE. The design spec records the slice shares as targets "to be revised
against measured availability". This script produces that measurement. A slice that cannot
reach its share within the upsample cap is reported as a shortfall and the script exits
non-zero, so the composition is revised on evidence rather than discovered to be
unsatisfiable after a training run.

Token counts come from the trained tokenizer when one exists, and otherwise from a
whitespace-word approximation scaled by a fixed factor. The approximation is adequate for
the gate's purpose -- deciding whether a slice is short by an order of magnitude -- and the
report records which method was used so the number is never mistaken for exact.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.corpus import SOURCES, CorpusSource  # noqa: E402
from train.paths import shared_dir  # noqa: E402

#: Tokens per whitespace-delimited word, used when no tokenizer is available.
#: Subword tokenizers on English prose run above 1.0; 1.3 is a deliberate slight
#: over-estimate so the gate errs toward reporting MORE supply, never less --
#: a gate that under-reports availability would block on a slice that is actually fine.
TOKENS_PER_WORD = 1.3

DEFAULT_BUDGET = 400_000_000
DEFAULT_UPSAMPLE_CAP = 8


@dataclass(frozen=True)
class Shortfall:
    name: str
    required: int
    available: int
    current_upsample: int
    needed_upsample: float


def required_tokens(source: CorpusSource, total_budget: int) -> int:
    """Tokens this source must supply to hit its target share of the budget."""
    return int(round(source.target_share * total_budget))


def achievable_tokens(available: int, upsample: int) -> int:
    """Tokens obtainable from ``available`` raw tokens at a repetition factor."""
    return available * upsample


def shortfall_report(available: Dict[str, int], total_budget: int,
                     upsample_cap: int) -> List[Shortfall]:
    """Sources that cannot reach their share within the cap, worst first."""
    out: List[Shortfall] = []
    for name, src in SOURCES.items():
        have = available.get(name, 0)
        need = required_tokens(src, total_budget)
        if achievable_tokens(have, src.upsample) >= need:
            continue
        needed = math.inf if have == 0 else need / have
        if needed <= upsample_cap and achievable_tokens(have, int(math.ceil(needed))) >= need:
            # Reachable by raising this source's upsample within the cap: not a shortfall,
            # but the registry's current factor is too low. Reported so it can be raised.
            pass
        out.append(Shortfall(name=name, required=need, available=have,
                             current_upsample=src.upsample, needed_upsample=needed))
    out.sort(key=lambda s: (-s.needed_upsample if s.needed_upsample != math.inf else -1e18))
    return out


def count_tokens(path: Path, tokenizer_dir: Path) -> tuple:
    """(tokens, method) for one prepared source file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    tok_json = tokenizer_dir / "tokenizer.json"
    if tok_json.is_file():
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(str(tok_json))
            total = 0
            for chunk in text.split("\n\n"):
                if chunk.strip():
                    total += len(tok.encode(chunk).ids)
            return total, "tokenizer"
        except Exception:
            pass
    return int(len(text.split()) * TOKENS_PER_WORD), "approx"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                   help=f"Total blend token budget (default: {DEFAULT_BUDGET:,})")
    p.add_argument("--upsample-cap", type=int, default=DEFAULT_UPSAMPLE_CAP,
                   help="Maximum repetition factor considered acceptable.")
    p.add_argument("--report", type=Path,
                   default=ROOT / "docs" / "measurements" / "corpus_availability.json")
    args = p.parse_args()

    corpus_dir = shared_dir("corpus")
    available: Dict[str, int] = {}
    methods: Dict[str, str] = {}
    for name in sorted(SOURCES):
        path = corpus_dir / f"{name}.txt"
        if not path.is_file():
            available[name] = 0
            methods[name] = "missing"
            continue
        available[name], methods[name] = count_tokens(path, shared_dir("tokenizer"))

    print(f"budget {args.budget:,} tokens, upsample cap {args.upsample_cap}")
    print()
    print(f"{'source':22} {'share':>6} {'required':>13} {'available':>13} {'x':>4}  method")
    print("-" * 74)
    for name in sorted(SOURCES):
        src = SOURCES[name]
        need = required_tokens(src, args.budget)
        print(f"{name:22} {src.target_share:>5.0%} {need:>13,} {available[name]:>13,} "
              f"{src.upsample:>4} {methods[name]}")

    short = shortfall_report(available, args.budget, args.upsample_cap)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "budget": args.budget,
        "upsample_cap": args.upsample_cap,
        "available": available,
        "methods": methods,
        "shortfalls": [asdict(s) for s in short],
    }, indent=2, default=str))
    print(f"\nwrote {args.report}")

    if short:
        print("\nSHORTFALL — these slices cannot reach their target share:")
        for s in short:
            need_x = "impossible (no material)" if s.needed_upsample == math.inf \
                else f"{s.needed_upsample:.1f}x"
            print(f"  {s.name:22} needs {need_x}, cap is {args.upsample_cap}x")
        print("\nRevise target shares in train/corpus.py against these numbers before "
              "blending. Do not raise the cap to force a fit: repetition at this scale "
              "risks memorisation rather than style transfer.")
        return 1

    print("\nAll slices can reach their target share within the cap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_measure_corpus.py -q`
Expected: PASS.

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -q`
Expected: all pre-existing tests still pass, plus everything added by this plan.

- [ ] **Step 6: Commit**

```bash
git add scripts/measure_corpus.py tests/test_measure_corpus.py
git commit -m "feat: corpus availability gate

Reports per-source available tokens against target share and exits non-zero
when a slice cannot reach its share within the upsample cap. The design spec
records the shares as targets to be revised against measured availability;
this is that measurement."
```

---

## Task 6: Run the pipeline end to end and record the answer

**Files:**
- Modify: `train/corpus.py` (target shares only, if the measurement requires it)
- Create: `docs/measurements/corpus_availability.json` (produced by the run)

**Interfaces:** none new.

This task runs what the previous five built and turns the result into a decision. It is separate because its deliverable is *evidence*, not code, and because revising the shares is a judgement call that a reviewer should see on its own.

- [ ] **Step 1: Build the full Gutenberg catalogue**

Run: `python scripts/build_gutenberg_catalogue.py`
Expected: ~48,284 books catalogued; a non-zero book count for every Gutenberg-backed source.

If any source is zero, stop and report. Check for alternate name forms first (Gutenberg records authors as `Surname, Forename`, sometimes with dates or initials).

- [ ] **Step 2: Re-check disk space, then fetch every source**

Run: `python scripts/check_disk_space.py`
Expected: exit 0. **If it exits 1, stop and report — do not delete anything to make room.**

Run: `python scripts/fetch_corpus.py`
Expected: a document count per source, all non-zero. This downloads several GB and will take a while.

- [ ] **Step 3: Normalise every source**

Run: `python scripts/prepare_corpus.py`
Expected: `artifacts/corpus/<source>.txt` for every source, with sizes printed.

- [ ] **Step 4: Measure**

Run: `python scripts/measure_corpus.py`
Expected: the availability table, and `docs/measurements/corpus_availability.json`.

Exit code 0 means every slice can hit its target. Exit code 1 means at least one cannot — which is a **successful outcome for this task**, not a failure: it is the gate doing its job.

- [ ] **Step 5: Revise the target shares if the gate reported a shortfall**

Edit `target_share` values in `train/corpus.py` so that they sum to 1.0 and every slice is satisfiable within the cap. Re-run `python scripts/measure_corpus.py` until it exits 0.

Record in the commit message: the original shares, the revised shares, and the measured availability that forced each change. Do not raise `upsample` beyond the cap to avoid revising shares — the cap exists because repetition at this scale risks memorisation rather than style transfer.

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest -q`
Expected: all tests pass, including `test_target_shares_sum_to_one` against any revised numbers.

- [ ] **Step 7: Commit**

```bash
git add train/corpus.py docs/measurements/corpus_availability.json
git commit -m "measure: corpus availability, and revised shares

<paste the availability table>

Shares revised: <before> -> <after>, because <the measured shortfall>."
```

---

## Self-Review

**Spec coverage.**

| Spec requirement | Task |
|---|---|
| Corpus source registry with licence as data | 1 |
| HF-first sourcing, pinned revisions | 1 (registry), 3 (fetch) |
| Verify named authors exist in `sedthh/gutenberg_english` | 2 |
| Fetch pipeline | 3 |
| Prepare/normalise, strip PG boilerplate | 4 |
| Measure before blending — the scarcity gate | 5, 6 |
| Shares revised against measured availability | 6 |
| Disk-space preflight; delete nothing | 0 |
| Blend, tokenizer retrain, generated licensing, evaluation prompt set | **Deliberately deferred to a second plan** — they depend on Task 6's measured ratios |

**Placeholder scan.** No TBD/TODO. Every code step contains complete implementations. Every test step contains runnable test code.

**Type consistency.** `CorpusSource` field names are used identically in Tasks 1–5 (`authors`, `bookshelves`, `upsample`, `target_share`, `hf_repo`, `hf_revision`, `hf_config`, `hf_split`). `matches_source(metadata, source)` has one signature, defined in Task 2 and imported in Task 3. `shared_dir(kind)` is the existing `train/paths.py` function throughout. `Shortfall` fields match between the dataclass, the tests, and the report writer.

**One known imprecision, deliberately kept.** `shortfall_report` contains a branch that computes whether a shortfall is reachable by raising a source's own upsample within the cap, then falls through and reports it anyway. That is intended: raising an upsample factor is a decision for a human reviewing the memorisation risk, not something the gate should quietly absolve. The tests pin the reported values rather than that branch's control flow.
