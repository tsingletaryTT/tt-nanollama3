<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Corpus Assembly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the measured, unsatisfiable corpus composition into a blended, tokenised corpus a model can actually be trained on — with shares settled on evidence, licensing residue removed, and a frozen evaluation set to judge the result.

**Architecture:** Broaden the scarce `spine` slice with catalogue-verified public-domain authors so it needs 2× repetition instead of 10×, strip the Project Gutenberg front matter the marker audit exposed, re-run the gate until it exits 0, then blend deterministically, retrain the tokenizer on that blend, and generate the licensing section from the registry so it cannot drift.

**Tech Stack:** Python 3.10+, `tokenizers`, `pytest`. No hardware, no new dependencies.

## What the previous plan established

`docs/superpowers/plans/2026-08-12-corpus-discovery.md` built the pipeline and ran it. Its results, recorded in `docs/measurements/corpus_availability.json`, are the input to this plan:

```
budget 400,000,000 tokens, cap 8x        (counts via the EXISTING TinyStories tokenizer)

source                share      required      available   verdict
tinystories             30%   120,000,000    445,771,903   ample
wikipedia_simple        15%    60,000,000     89,177,921   ample
poetry                   1%     4,000,000     32,514,616   ample
folklore                 8%    32,000,000     23,541,562   needs 1.4x
gutenberg_children      15%    60,000,000     36,437,848   needs 1.6x
weird                    4%    16,000,000      7,951,195   needs 2.0x
procedural              13%    52,000,000     13,623,510   needs 3.8x
spine                   12%    48,000,000      4,803,988   needs 10.0x   OVER CAP
flavour                  2%     8,000,000        623,814   needs 12.8x   OVER CAP
```

Three facts drive this plan:

1. **Total available is 654M against a 400M budget.** Material is sufficient; distribution is not.
2. **The counts are optimistic.** They were measured with the *existing TinyStories* tokenizer, which over-fragments Fabre, Fort and Machen. Retraining the tokenizer on the blend will yield **fewer** tokens for those sources, so every shortfall above is understated. Task 6 re-measures after retraining for exactly this reason.
3. **A licensing residue exists.** The marker audit found 40/199 `folklore` and 6/583 `gutenberg_children` documents with an end marker but no start marker; inspection confirmed some genuinely retain PG packaging ("There are several editions of this ebook in the Project Gutenberg collection", "This eBook was produced by Les Bowler").

## Global Constraints

- **SPDX header pair on every new file**; Python 3.10+.
- **No bare `assert` for guards** in production code (tests may assert freely).
- **`pyproject.toml` must NOT be modified**; no new dependencies.
- **DELETE NOTHING.** Not `artifacts/`, not `~/.cache/huggingface` (1.2 TB of other projects' data), not any partial output. On disk pressure, **stop and report** — reclaiming space is the human's decision.
- **Disk is at 98%, ~92 GB free.** Run `python scripts/check_disk_space.py` before any step that writes bulk data; stop on non-zero exit.
- **Never write under `artifacts/checkpoints/` or `artifacts/hf/`** — protected baseline evidence. Use `train/paths.py`'s `shared_dir`/`write_dir`.
- **The corpus is never redistributed.** We ship a recipe; `artifacts/` stays gitignored.
- **Licensing is data, not prose** — it lives in `train/corpus.py` and is rendered, never hand-written.
- **`target_share` values must sum to exactly 1.0**; `tests/test_corpus.py::test_target_shares_sum_to_one` enforces it.

---

## File Structure

| File | Responsibility |
|---|---|
| `train/corpus.py` | Registry. This plan edits `spine`'s selectors and (Task 3) the settled shares. |
| `scripts/prepare_corpus.py` | Normalisation. Task 2 adds front-matter stripping. |
| `scripts/blend_corpus.py` | Deterministic mix + `blend_manifest.json`. New in Task 4. |
| `scripts/render_licensing.py` | Generates the licensing section from the registry. New in Task 6. |
| `docs/evaluation_prompts.json` | The frozen prompt set. New in Task 7. |
| `tests/test_blend_corpus.py`, `tests/test_render_licensing.py` | Their gates. |

---

## Task 1: Broaden the `spine` slice

**Files:**
- Modify: `train/corpus.py` (the `spine` entry's `authors` list and `rationale`)
- Test: `tests/test_corpus.py` (add one test)

**Interfaces:**
- Consumes: `CorpusSource.authors` (existing).
- Produces: nothing new; `spine`'s selector list grows.

`spine` is the observational-mystical voice of this model and has 53 books against a 12% share — 10× repetition, over the cap of 8. Every author below was **verified present in the catalogue** (`artifacts/raw/gutenberg_catalogue.jsonl`, 48,284 books) with the book count shown.

Two deliberate exclusions, stated so they are not "fixed" later:

- **Andrew Lang (132 books) is excluded** — he is already `folklore`'s selector. Adding him here would double-count him across slices and quietly turn `spine` into more folklore.
- **Blavatsky (8) and Swedenborg (3) are excluded** — they assert esoteric doctrine, where Fabre and Fort *document* the inexplicable deadpan. That is a different strangeness from the one this corpus is for. Flammarion and Proctor are kept: astronomers writing about cosmic scale and other minds, which is the same register.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_corpus.py

def test_spine_is_broad_enough_to_avoid_heavy_repetition():
    """spine had 53 books against a 12% share -- 10x repetition, over the cap of 8.

    Every author here was verified present in the Gutenberg catalogue before being added.
    The count guards against the slice silently narrowing again.
    """
    spine = SOURCES["spine"]
    assert len(spine.authors) >= 18, (
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_corpus.py -k spine -q`
Expected: FAIL — `spine has only 4 author selectors`

- [ ] **Step 3: Broaden the selectors**

Replace `spine`'s `authors` list and `rationale` in `train/corpus.py` with exactly this:

```python
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
                  "over cap) to 17 (241 unique books, ~2x) with catalogue-verified PD naturalists and "
                  "anomalists in the same register. Browne, Thomas, Sir is deliberately NOT here "
                  "despite being in the pre-task list -- he is weird's selector. Andrew Lang is "
                  "excluded for the same reason: he is folklore's selector. Blavatsky and "
                  "Swedenborg are deliberately excluded: they assert doctrine where this slice "
                  "documents the inexplicable.",
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_corpus.py -q`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add train/corpus.py tests/test_corpus.py
git commit -m "feat: broaden spine from 4 authors to 17, all catalogue-verified"
```

---

## Task 2: Strip the residual Project Gutenberg front matter

**Files:**
- Modify: `scripts/prepare_corpus.py`
- Test: `tests/test_prepare_corpus.py`

**Interfaces:**
- Consumes: `strip_gutenberg_boilerplate(text) -> BoilerplateResult`, `MARKER_*` constants.
- Produces: `strip_front_matter(text: str) -> tuple[str, int]` returning cleaned text and the number of lines removed.

The marker audit found documents with an end marker but no start marker — so nothing bounded their *front*, and PG packaging survived. Confirmed examples: `"There are several editions of this ebook in the Project Gutenberg collection..."` and `"This eBook was produced by Les Bowler."`

**Be conservative.** A pattern that eats real prose is far worse than one that leaves a producer credit. Only strip lines that are unambiguously PG packaging, only from the first 40 lines of a document, and stop at the first line that does not match.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_prepare_corpus.py
from scripts.prepare_corpus import strip_front_matter

REAL_RESIDUE = """This eBook was produced by Les Bowler.

There are several editions of this ebook in the Project Gutenberg collection.
Various characteristics of each ebook are listed to aid in selection.

Even the sandy kitten was neglected, and the story truly begins here."""


def test_strips_producer_credit_and_edition_note():
    out, removed = strip_front_matter(REAL_RESIDUE)
    assert "Les Bowler" not in out
    assert "Project Gutenberg collection" not in out
    assert "Even the sandy kitten was neglected" in out
    assert removed >= 2


def test_leaves_ordinary_prose_completely_alone():
    prose = ("The printer's trademark appeared on the flyleaf.\n"
             "These drawings would have been reproduced by modern processes.\n"
             "The story begins.")
    out, removed = strip_front_matter(prose)
    assert out == prose
    assert removed == 0


def test_stops_at_the_first_non_matching_line():
    """Once real text starts, nothing after it is ever removed."""
    doc = ("Produced by A. Volunteer\n"
           "The real story begins here.\n"
           "This eBook was produced by someone else.\n")
    out, _ = strip_front_matter(doc)
    assert "The real story begins here." in out
    assert "someone else" in out, "stripping must not resume after real text starts"


def test_does_not_scan_beyond_the_document_head():
    body = "\n".join(f"line {i}" for i in range(60))
    doc = body + "\nThis eBook was produced by Someone.\n"
    out, removed = strip_front_matter(doc)
    assert "This eBook was produced by Someone." in out
    assert removed == 0


def test_empty_and_whitespace_documents_do_not_raise():
    assert strip_front_matter("") == ("", 0)
    assert strip_front_matter("   \n\n  ")[1] == 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_prepare_corpus.py -k front_matter -q`
Expected: FAIL — `ImportError: cannot import name 'strip_front_matter'`

- [ ] **Step 3: Implement**

Add to `scripts/prepare_corpus.py`:

```python
#: Lines that are unambiguously Project Gutenberg packaging rather than the work itself.
#:
#: Deliberately narrow. These run only over the head of a document and stop at the first
#: line that does not match, because a pattern that eats real prose is far worse than one
#: that leaves a producer credit behind. "reproduced by" and "the printer's trademark" are
#: real sentences in this corpus and must not match.
_FRONT_MATTER = re.compile(
    r"^\s*(?:"
    r"(?:this\s+)?e-?(?:book|text)\s+was\s+produced\s+by\b"
    r"|produced\s+by\s+[A-Z]"
    r"|there\s+are\s+several\s+editions\s+of\s+this\s+ebook\b"
    r"|various\s+characteristics\s+of\s+each\s+ebook\b"
    r"|transcriber'?s?\s+note\b"
    r"|updated\s+editions\s+will\s+replace\b"
    r"|this\s+file\s+was\s+produced\s+from\b"
    r")",
    re.IGNORECASE,
)

#: How far into a document front matter may appear. Beyond this it is the work, not packaging.
_FRONT_MATTER_WINDOW = 40


def strip_front_matter(text: str) -> tuple:
    """Remove Project Gutenberg packaging lines from a document's head.

    Returns ``(cleaned_text, lines_removed)``. Scans at most the first
    ``_FRONT_MATTER_WINDOW`` lines and stops permanently at the first line that is neither
    blank nor packaging — once the work has started, nothing later is removed even if it
    resembles a credit.
    """
    if not text.strip():
        return text, 0
    lines = text.split("\n")
    keep_from = 0
    removed = 0
    for i, line in enumerate(lines[:_FRONT_MATTER_WINDOW]):
        if not line.strip():
            keep_from = i + 1
            continue
        if _FRONT_MATTER.match(line):
            keep_from = i + 1
            removed += 1
            continue
        break
    if removed == 0:
        return text, 0
    return "\n".join(lines[keep_from:]).lstrip("\n"), removed
```

Then call it from `prepare_source`, immediately after `strip_gutenberg_boilerplate` and before `normalise`, accumulating a `front_matter_lines` count into the returned dict, and print that count in `main()` when it is non-zero.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_prepare_corpus.py -q`
Expected: PASS, all tests.

- [ ] **Step 5: Re-normalise and confirm the residue is gone**

Run: `python scripts/prepare_corpus.py`
Then verify with this exact check, which counts genuine packaging while excluding the two known false positives ("the printer's trademark", "reproduced by"):

```bash
python3 - <<'PY'
import re, pathlib
PACK = re.compile(r"(Project Gutenberg collection|eBook was produced by|"
                  r"^Produced by [A-Z]|Transcriber's Note)", re.I | re.M)
for p in sorted(pathlib.Path("artifacts/corpus").glob("*.txt")):
    hits = len(PACK.findall(p.read_text(encoding="utf-8", errors="replace")))
    print(f"{p.name:28} {hits:>5} packaging hits")
PY
```

Record the before/after counts in your report. `folklore` and `gutenberg_children` should drop substantially. If any source *increases*, stop and report — that would mean the stripping is corrupting text.

- [ ] **Step 6: Commit**

```bash
git add scripts/prepare_corpus.py tests/test_prepare_corpus.py
git commit -m "fix: strip residual Project Gutenberg front matter from document heads"
```

---

## Task 3: Re-measure and settle the shares

**Files:**
- Modify: `train/corpus.py` (`target_share` values only)
- Modify: `docs/measurements/corpus_availability.json` (regenerated)

**Interfaces:** none new.

This task's deliverable is **evidence and a decision**, not code. The gate must exit 0 at the end.

- [ ] **Step 1: Re-fetch the broadened spine**

Run: `python scripts/check_disk_space.py` — confirm exit 0, then:
`python scripts/fetch_corpus.py --source spine`

Task 1 broadened `spine` from five authors to 17, so it must be re-fetched. This streams the Gutenberg dataset once. Expect **241 unique books** (243 summed per-author before deduplication). Re-derived from `artifacts/raw/gutenberg_catalogue.jsonl`; an earlier draft said 259, which was wrong.

- [ ] **Step 2: Re-normalise and re-measure**

```bash
python scripts/prepare_corpus.py --source spine
python scripts/measure_corpus.py
```

Record the full table. `spine` should now need roughly 2× rather than 10×.

- [ ] **Step 3: Settle the shares so the gate exits 0**

Adjust `target_share` values in `train/corpus.py` until `python scripts/measure_corpus.py` exits 0, obeying these rules:

- Shares must still sum to exactly 1.0.
- No source may need more than **4×** upsample. The cap is 8, but 4 is the working limit for this pass: repetition at ~133M parameters risks memorisation, and the measured counts are optimistic (see "What the previous plan established", point 2).
- **`flavour` must drop to at most 0.6%** (use 0.5%). It is 7 books by design and exists to tint, not to bulk. Its 623,814 tokens support a ceiling of 0.62% at 4x upsample, so its current 2.00% is arithmetically impossible under the 4x rule — a forced change, not a judgement call.
- Do not reduce the *combined* strange share (`spine` + `folklore` + `weird` + `flavour`) below **20%**. Below that the model loses the quality it is being built for. If the arithmetic cannot satisfy this, stop and report rather than quietly going under.
- Prefer taking share from `tinystories` (445M available against a 120M requirement) over taking it from a scarce slice.

Record in your report: the shares before, the shares after, and the measured number that forced each change.

- [ ] **Step 4: Confirm the gate passes**

Run: `python scripts/measure_corpus.py`
Expected: exit 0, printing "All slices can reach their target share within the cap."

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: all tests pass, including `test_target_shares_sum_to_one` against the new values.

- [ ] **Step 6: Commit**

```bash
git add train/corpus.py docs/measurements/corpus_availability.json
git commit -m "measure: settle corpus shares on evidence; gate exits 0"
```

---

## Task 4: Deterministic blend

**Files:**
- Create: `scripts/blend_corpus.py`
- Test: `tests/test_blend_corpus.py`

**Interfaces:**
- Consumes: `SOURCES`, `shared_dir`.
- Produces: `plan_blend(available: dict, budget: int) -> dict[str, int]` mapping source name to tokens to emit; `artifacts/corpus/blend.txt`; `artifacts/corpus/blend_manifest.json`.

The manifest is what makes "what was this model trained on" exactly answerable — the question that is usually unanswerable about a model.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_blend_corpus.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Blend planning: deterministic, share-faithful, and honest about repetition."""
import pytest
from train.corpus import CorpusSource
from scripts.blend_corpus import plan_blend


def _src(name, share, upsample=1):
    return CorpusSource(name=name, slice="spine", target_share=share,
                        hf_repo="r", hf_revision="a" * 40, upsample=upsample)


def test_plan_allocates_each_source_its_share_of_the_budget(monkeypatch):
    sources = {"a": _src("a", 0.75), "b": _src("b", 0.25)}
    monkeypatch.setattr("scripts.blend_corpus.SOURCES", sources)
    plan = plan_blend({"a": 10_000_000, "b": 10_000_000}, budget=1_000_000)
    assert plan == {"a": 750_000, "b": 250_000}


def test_plan_is_deterministic(monkeypatch):
    sources = {"a": _src("a", 0.5), "b": _src("b", 0.5)}
    monkeypatch.setattr("scripts.blend_corpus.SOURCES", sources)
    avail = {"a": 9_000_000, "b": 9_000_000}
    assert plan_blend(avail, 1_000_000) == plan_blend(avail, 1_000_000)


def test_plan_refuses_when_a_source_cannot_meet_its_share(monkeypatch):
    """Silently emitting less than the share would produce a corpus nobody ordered."""
    sources = {"a": _src("a", 1.0, upsample=1)}
    monkeypatch.setattr("scripts.blend_corpus.SOURCES", sources)
    with pytest.raises(ValueError, match="cannot supply"):
        plan_blend({"a": 100}, budget=1_000_000)


def test_plan_counts_upsample_toward_supply(monkeypatch):
    sources = {"a": _src("a", 1.0, upsample=4)}
    monkeypatch.setattr("scripts.blend_corpus.SOURCES", sources)
    assert plan_blend({"a": 300_000}, budget=1_000_000) == {"a": 1_000_000}
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_blend_corpus.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.blend_corpus'`

- [ ] **Step 3: Implement**

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Blend the prepared sources into one corpus, and record exactly what went in.

The manifest this writes is the point: it makes "what was this model trained on" an
answerable question, with per-source token counts, repetition factors, achieved shares and
the pinned revision each source came from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.corpus import SOURCES  # noqa: E402
from train.paths import shared_dir  # noqa: E402

DEFAULT_BUDGET = 400_000_000


def plan_blend(available: Dict[str, int], budget: int) -> Dict[str, int]:
    """Tokens to emit per source. Raises ValueError if a share cannot be met."""
    plan: Dict[str, int] = {}
    for name, src in SOURCES.items():
        want = int(round(src.target_share * budget))
        have = available.get(name, 0) * src.upsample
        if have < want:
            raise ValueError(
                f"{name} cannot supply its {src.target_share:.0%} share: needs {want:,} "
                f"tokens, has {available.get(name, 0):,} x{src.upsample} = {have:,}. "
                f"Re-run scripts/measure_corpus.py and settle the shares first."
            )
        plan[name] = want
    return plan


def _emit(src_path: Path, want_tokens: int, out, tokens_per_word: float = 1.3) -> int:
    """Append text from ``src_path`` until ``want_tokens`` is reached, repeating if needed.

    Word-count approximation is used here for the same reason measure_corpus.py uses it:
    the blend only needs to hit its share closely, and an exact tokenizer pass over 400M
    tokens costs far more than the precision is worth. The manifest records the method.
    """
    text = src_path.read_text(encoding="utf-8", errors="replace")
    per_pass = max(1, int(len(text.split()) * tokens_per_word))
    written = 0
    while written < want_tokens:
        out.write(text)
        out.write("\n\n")
        written += per_pass
    return written


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    p.add_argument("--availability", type=Path,
                   default=ROOT / "docs" / "measurements" / "corpus_availability.json")
    args = p.parse_args()

    if not args.availability.is_file():
        print(f"ERROR: {args.availability} not found. Run scripts/measure_corpus.py first.",
              file=sys.stderr)
        return 1
    available = json.loads(args.availability.read_text())["available"]

    try:
        plan = plan_blend(available, args.budget)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    corpus_dir = shared_dir("corpus")
    out_path = corpus_dir / "blend.txt"
    emitted: Dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as out:
        for name in sorted(plan):
            src_path = corpus_dir / f"{name}.txt"
            if not src_path.is_file():
                print(f"ERROR: {src_path} missing; run scripts/prepare_corpus.py",
                      file=sys.stderr)
                return 1
            emitted[name] = _emit(src_path, plan[name], out)
            print(f"  {name:22} {emitted[name]:>13,} tokens "
                  f"({plan[name] / args.budget:.1%})")

    digest = hashlib.sha256()
    with out_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)

    manifest = {
        "budget": args.budget,
        "token_count_method": "approx (words x 1.3)",
        "output": out_path.name,
        "sha256": digest.hexdigest(),
        "sources": {
            name: {
                "planned_tokens": plan[name],
                "emitted_tokens": emitted[name],
                "achieved_share": emitted[name] / sum(emitted.values()),
                "target_share": SOURCES[name].target_share,
                "upsample": SOURCES[name].upsample,
                "available_tokens": available.get(name, 0),
                "hf_repo": SOURCES[name].hf_repo,
                "hf_revision": SOURCES[name].hf_revision,
                "license_id": SOURCES[name].license_id,
            }
            for name in sorted(plan)
        },
    }
    (corpus_dir / "blend_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")
    print(f"wrote {corpus_dir / 'blend_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_blend_corpus.py -q`
Expected: PASS.

- [ ] **Step 5: Build the blend**

Run: `python scripts/check_disk_space.py`, confirm exit 0, then `python scripts/blend_corpus.py`.
Expected: a per-source table, `artifacts/corpus/blend.txt` of roughly 2 GB, and `blend_manifest.json`.

Record the achieved shares. They will not be exactly the targets — the approximation works in whole passes over each source — but each should be within a couple of points. If any is off by more than 5 points, stop and report.

- [ ] **Step 6: Commit**

```bash
git add scripts/blend_corpus.py tests/test_blend_corpus.py
git commit -m "feat: deterministic corpus blend with a provenance manifest"
```

---

## Task 5: Retrain the tokenizer on the blend

**Files:**
- Modify: `scripts/build_tokenizer.py` (default corpus path only)

**Interfaces:**
- Consumes: `artifacts/corpus/blend.txt`.
- Produces: a retrained 32,000-token BPE at `artifacts/tokenizer/`.

The current tokenizer saw only TinyStories and over-fragments every new domain — spending model capacity on spelling rather than meaning, and inflating the token counts Task 3 settled shares against.

- [ ] **Step 1: Point the tokenizer builder at the blend**

In `scripts/build_tokenizer.py`, change the corpus the trainer reads from `artifacts/corpus/corpus.txt` to `artifacts/corpus/blend.txt`, keeping `--corpus` overridable on the command line and leaving `VOCAB_SIZE` at 32,000 so `vocab_size` and every manifest keep their shape.

**Do not delete the existing tokenizer.** Write the new one alongside and let the operator move it, or write to `artifacts/tokenizer/` only after confirming the existing directory has been preserved elsewhere — the baseline model's tokenizer is published-artifact evidence.

- [ ] **Step 2: Train it**

Run: `python scripts/build_tokenizer.py`
Expected: a 32,000-entry vocabulary trained on the blend.

- [ ] **Step 3: Verify it improved fragmentation on the new domains**

```bash
python3 - <<'PY'
from tokenizers import Tokenizer
import pathlib
tok = Tokenizer.from_file("artifacts/tokenizer/tokenizer.json")
for name in ("spine", "folklore", "weird", "tinystories"):
    p = pathlib.Path(f"artifacts/corpus/{name}.txt")
    if not p.is_file():
        continue
    sample = p.read_text(encoding="utf-8", errors="replace")[:200_000]
    words = len(sample.split())
    toks = len(tok.encode(sample).ids)
    print(f"{name:16} {toks/words:.2f} tokens/word")
PY
```

Record the ratios. The new domains should sit close to the backbone's ratio rather than far above it. Report the numbers whatever they are — if `spine` is still much higher than `tinystories`, say so, because it means the vocabulary is still dominated by the backbone.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_tokenizer.py
git commit -m "feat: retrain the tokenizer on the blended corpus"
```

---

## Task 6: Re-measure with the new tokenizer, and generate the licensing section

**Files:**
- Create: `scripts/render_licensing.py`
- Test: `tests/test_render_licensing.py`

**Interfaces:**
- Consumes: `SOURCES`.
- Produces: `render_licensing() -> str` (markdown); written to `docs/corpus_licensing.md`.

Two jobs in one task because both depend on the finished corpus and neither is large.

- [ ] **Step 1: Re-measure against the retrained tokenizer**

Run: `python scripts/measure_corpus.py`

Task 3's shares were settled using the **old** tokenizer's optimistic counts. This re-measure is the honest one. If it now exits 1, settle the shares again under Task 3's rules and re-blend. Record both measurements in your report so the difference the tokenizer made is visible.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_render_licensing.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The licensing section is GENERATED so it cannot drift from the registry.

This repo has twice shipped documentation contradicting the facts. A rendered section
cannot go stale; a hand-written one always eventually does.
"""
from train.corpus import SOURCES
from scripts.render_licensing import render_licensing


def test_every_source_appears_with_its_licence():
    out = render_licensing()
    for name, src in SOURCES.items():
        assert name in out, f"{name} missing from the rendered licensing"
        assert src.license_id in out, f"{name}'s licence {src.license_id!r} missing"


def test_share_alike_sources_are_called_out():
    out = render_licensing()
    for src in SOURCES.values():
        if src.share_alike:
            assert src.attribution in out, f"{src.name} needs attribution rendered"
    assert "share-alike" in out.lower()


def test_states_that_the_corpus_is_not_redistributed():
    assert "not redistribute" in render_licensing().lower()


def test_states_the_weights_question_is_unsettled():
    out = render_licensing().lower()
    assert "unsettled" in out or "do not assert" in out


def test_no_source_is_silently_omitted():
    """A source added to the registry without a licence must break this, not slip through."""
    out = render_licensing()
    assert out.count("| ") >= len(SOURCES)
```

- [ ] **Step 3: Run to verify they fail**

Run: `python -m pytest tests/test_render_licensing.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.render_licensing'`

- [ ] **Step 4: Implement**

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Render the corpus licensing section from the registry.

Generated rather than written, because this project has twice shipped documentation that
contradicted reality and only caught it by grep. A rendered section cannot go stale.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.corpus import SOURCES  # noqa: E402

OUT = ROOT / "docs" / "corpus_licensing.md"


def render_licensing() -> str:
    lines = [
        "<!-- SPDX-License-Identifier: Apache-2.0 -->",
        "<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->",
        "",
        "# Corpus sources and licensing",
        "",
        "**Generated from `train/corpus.py` by `scripts/render_licensing.py`. Do not edit "
        "by hand — regenerate it.**",
        "",
        "| Source | Slice | Share | Licence | Pinned revision |",
        "|---|---|---|---|---|",
    ]
    for name in sorted(SOURCES):
        s = SOURCES[name]
        lines.append(
            f"| `{name}` | {s.slice} | {s.target_share:.0%} | {s.license_id} | "
            f"`{s.hf_revision[:12]}` |"
        )

    share_alike = [s for s in SOURCES.values() if s.share_alike]
    lines += [
        "",
        "## Attribution",
        "",
    ]
    for s in sorted(SOURCES.values(), key=lambda x: x.name):
        if s.attribution:
            lines.append(f"- {s.attribution}")

    lines += [
        "",
        "## What this project does and does not claim",
        "",
        "**We do not redistribute the corpus.** This repository ships a *recipe* — pinned "
        "dataset revisions and a deterministic blend — not the text. Reconstructing it "
        "locally is what `scripts/fetch_corpus.py` and `scripts/blend_corpus.py` are for.",
        "",
    ]
    if share_alike:
        names = ", ".join(f"`{s.name}` ({s.license_id})" for s in share_alike)
        lines += [
            f"**Share-alike sources:** {names}. Whether model weights trained on "
            "share-alike data constitute a Data Derivative is **unsettled**. This project "
            "does not assert that they do not. Anyone publishing weights trained with this "
            "recipe should reach their own conclusion rather than inheriting one.",
            "",
        ]
    lines += [
        "**Project Gutenberg material** is public domain as *text*; the aggregations we "
        "fetch it through carry their own permissive terms, recorded in the table above. "
        "`scripts/prepare_corpus.py` strips Project Gutenberg headers, footers and "
        "front-matter packaging, and this project does not use the Project Gutenberg "
        "trademark.",
        "",
        "This is a stated position, not legal advice.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render_licensing(), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run to verify they pass, and render**

```bash
python -m pytest tests/test_render_licensing.py -q
python scripts/render_licensing.py
```

- [ ] **Step 6: Commit**

```bash
git add scripts/render_licensing.py tests/test_render_licensing.py docs/corpus_licensing.md \
        docs/measurements/corpus_availability.json
git commit -m "feat: generate the corpus licensing section from the registry"
```

---

## Task 7: The frozen evaluation prompt set

**Files:**
- Create: `docs/evaluation_prompts.json`
- Create: `scripts/generate_samples.py`
- Test: `tests/test_evaluation_prompts.py`

**Interfaces:**
- Consumes: a converted HF model directory.
- Produces: `load_prompts() -> list[dict]`; samples written to `docs/measurements/samples-<label>.md`.

Loss cannot see "strangely satisfying", and the v2 result proved lower loss can move *away* from what this project wants. Human judgment on a fixed set is the acceptance gate; this task builds the instrument.

- [ ] **Step 1: Write the prompt set**

Create `docs/evaluation_prompts.json` with exactly these entries:

```json
{
  "note": "Frozen evaluation set. Every checkpoint answers these same prompts with the same seed, and a human reads the results. Do not edit prompts between runs -- that breaks comparability. Add new ones with new ids instead.",
  "prompts": [
    {"id": "voice-01", "probe": "target-voice",
     "text": "The chimp chose the longest stick, then the one that had been"},
    {"id": "voice-02", "probe": "target-voice",
     "text": "The ants had learned that being eaten was a way of"},
    {"id": "voice-03", "probe": "target-voice",
     "text": "Once upon a time, there was a little"},
    {"id": "stutter-01", "probe": "stutter",
     "text": "A rose is a rose is a"},
    {"id": "stutter-02", "probe": "stutter",
     "text": "The little mouse. The little mouse. The"},
    {"id": "oracle-01", "probe": "oracular",
     "text": "The question was whether to go. The answer came back:"},
    {"id": "oracle-02", "probe": "oracular",
     "text": "Above, the mountain. Below, the lake. The image is of"},
    {"id": "agentic-01", "probe": "agentic",
     "text": "I placed a straw across the trench and waited. The procession"},
    {"id": "agentic-02", "probe": "agentic",
     "text": "To find out what the bees would do, I first"},
    {"id": "ground-01", "probe": "grounding",
     "text": "An ant is an insect that lives in"},
    {"id": "ground-02", "probe": "grounding",
     "text": "Chimpanzees use tools such as"},
    {"id": "assoc-01", "probe": "perpendicular",
     "text": "The stick remembered being a"},
    {"id": "assoc-02", "probe": "perpendicular",
     "text": "There is a kind of light that only falls"},
    {"id": "long-01", "probe": "coherence",
     "text": "The old woman kept bees behind the house, and every morning she"},
    {"id": "long-02", "probe": "coherence",
     "text": "In the winter the pond froze, and the children"}
  ]
}
```

Each probe answers a specific question: **target-voice** is the acceptance criterion from the spec; **stutter** distinguishes "learned Stein" from "learned to repeat" given this project's known repetition defect; **oracular** tests whether the I Ching's response shape took; **agentic** tests plan→act→observe→report; **grounding** checks Wikipedia did its job; **perpendicular** tests associative leaps; **coherence** tests whether it holds together over length.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_evaluation_prompts.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The prompt set is FROZEN. These tests protect comparability across checkpoints."""
import json
from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[1] / "docs" / "evaluation_prompts.json"
REQUIRED_PROBES = {"target-voice", "stutter", "oracular", "agentic",
                   "grounding", "perpendicular", "coherence"}


def test_prompt_file_parses():
    data = json.loads(PROMPTS.read_text())
    assert data["prompts"]


def test_ids_are_unique():
    ids = [p["id"] for p in json.loads(PROMPTS.read_text())["prompts"]]
    assert len(ids) == len(set(ids))


def test_every_required_probe_is_present():
    probes = {p["probe"] for p in json.loads(PROMPTS.read_text())["prompts"]}
    missing = REQUIRED_PROBES - probes
    assert not missing, f"prompt set is missing probes: {sorted(missing)}"


def test_stutter_probe_exists_because_this_model_has_a_repetition_defect():
    prompts = json.loads(PROMPTS.read_text())["prompts"]
    assert [p for p in prompts if p["probe"] == "stutter"], (
        "the stutter probe distinguishes 'learned Stein' from 'learned to repeat'"
    )


def test_no_prompt_is_empty_or_whitespace():
    for p in json.loads(PROMPTS.read_text())["prompts"]:
        assert p["text"].strip()
```

- [ ] **Step 3: Run to verify they fail, then pass**

Run: `python -m pytest tests/test_evaluation_prompts.py -q`
Expected: PASS once the JSON from Step 1 exists (the tests validate the data file, so they pass as soon as it is correct — run them first with the file absent to see them fail with `FileNotFoundError`).

- [ ] **Step 4: Implement the sample generator**

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Generate the frozen prompt set's completions from a checkpoint, for human judgment.

Loss cannot see "strangely satisfying", and a better-trained model in this project measured
FLATTER, not sharper. So the acceptance gate is a person reading these samples -- this
script only makes them, identically, every time.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROMPTS = ROOT / "docs" / "evaluation_prompts.json"


def load_prompts() -> list:
    return json.loads(PROMPTS.read_text())["prompts"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model directory")
    p.add_argument("--label", required=True, help="Tag for this run, e.g. blend-v1-step20000")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto").eval()

    out_path = ROOT / "docs" / "measurements" / f"samples-{args.label}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Samples — {args.label}", "",
             f"model: `{args.model}` · greedy · seed {args.seed} · "
             f"{args.max_new_tokens} new tokens", ""]

    for prompt in load_prompts():
        ids = tok(prompt["text"], return_tensors="pt").input_ids
        with torch.no_grad():
            got = model.generate(input_ids=ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=False)
        completion = tok.decode(got[0][ids.shape[1]:], skip_special_tokens=True)
        lines += [f"### {prompt['id']} · _{prompt['probe']}_", "",
                  f"> {prompt['text']}**{completion}**", ""]
        print(f"  {prompt['id']:12} {completion[:60]!r}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Generate a baseline sample set from the existing model**

Run: `python scripts/generate_samples.py --model artifacts/hf --label baseline-384`

This is the *old* TinyStories-only model. Its samples are the "before" picture — they exist so the blended model can be compared against something rather than judged in a vacuum. It will do badly on the oracular and agentic probes; that is the point.

- [ ] **Step 6: Run the full suite and commit**

```bash
python -m pytest -q
git add docs/evaluation_prompts.json scripts/generate_samples.py \
        tests/test_evaluation_prompts.py docs/measurements/samples-baseline-384.md
git commit -m "feat: frozen evaluation prompt set and sample generator"
```

---

## Task 8: Triage the deferred minors

**Files:** various, as triaged.

The previous plan's ledger (`.superpowers/sdd/2026-08-12-corpus-discovery/progress.md`) carries roughly a dozen deferred minor findings, and that plan's final whole-branch review never ran — the branch was merged early at the user's request. This task closes that gap.

- [ ] **Step 1: Read the ledger's deferred list**

Read `.superpowers/sdd/2026-08-12-corpus-discovery/progress.md` and collect every line containing `minor (deferred)`. Expect items including: unused imports in `scripts/check_disk_space.py` and two test files; a dead `dest_files` dict and dead `except KeyError` in `scripts/fetch_corpus.py`; `--revision` defaulting to `"main"` in `scripts/build_gutenberg_catalogue.py`; `json.dumps` emitting bare `Infinity` in `scripts/measure_corpus.py`; a weak carriage-return assertion in `tests/test_prepare_corpus.py`; and Browne appearing in `spine`'s selectors without appearing in its rationale.

- [ ] **Step 2: Fix the ones that are real**

Fix each item that is a genuine defect. Note that Task 1 of this plan rewrites `spine`'s rationale, which resolves the Browne item — confirm it did rather than fixing it twice.

Leave anything that is a matter of taste, and say so in your report with a one-line reason. Do not manufacture work: an unused import in a file transcribed verbatim from a plan is worth removing; a bare `else` that became an `elif` is not worth churn.

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: triage deferred minors from the corpus-discovery plan"
```

---

## Self-Review

**Spec coverage** against `docs/superpowers/specs/2026-08-12-diverse-corpus-design.md`:

| Spec requirement | Task |
|---|---|
| Shares revised against measured availability | 3, and re-settled in 6 |
| Deterministic blend with a recorded manifest | 4 |
| Tokenizer retrained on the blend, 32k vocab | 5 |
| Licensing generated from the registry, not written | 6 |
| Frozen prompt set, human judgment as the gate | 7 |
| Scarcity risk resolved | 1 (broaden) + 3 (settle) |
| Stein/repetition risk has an evaluation probe | 7 (`stutter`) |
| PG boilerplate stripped so the PD claim is accurate | 2 |
| IF transcripts | **Still deferred** — the spec permits shipping without them; `procedural` carries the agentic slice |

**Placeholder scan:** none. Every code step contains a complete implementation; every test step contains runnable tests.

**Type consistency:** `plan_blend(available, budget) -> Dict[str, int]` and `render_licensing() -> str` are each defined once and used consistently. `strip_front_matter(text) -> tuple[str, int]` matches its tests. `shared_dir` and `SOURCES` are the existing symbols throughout.

**One ordering hazard, deliberately kept.** Task 3 settles shares using the *old* tokenizer's optimistic counts, and Task 6 re-measures after retraining, possibly forcing a second settle and re-blend. Doing it in one pass is impossible — the tokenizer must be trained on a blend, and the blend needs settled shares. The plan makes the loop explicit rather than pretending the first measurement is final.
