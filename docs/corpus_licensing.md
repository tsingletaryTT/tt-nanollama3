<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Corpus sources and licensing

**Generated from `train/corpus.py` by `scripts/render_licensing.py`. Do not edit by hand — regenerate it.**

| Source | Slice | Share | Licence | Pinned revision |
|---|---|---|---|---|
| `flavour` | flavour | 0.5% | MIT (packaging); public domain (texts) | `28973b04f28f` |
| `folklore` | folklore | 8% | MIT (packaging); public domain (texts) | `28973b04f28f` |
| `gutenberg_children` | backbone | 15% | MIT (packaging); public domain (texts) | `28973b04f28f` |
| `poetry` | weird | 1% | CC0-1.0 | `fcd42e249fed` |
| `procedural` | agentic | 12% | MIT (packaging); public domain (texts) | `28973b04f28f` |
| `spine` | spine | 13.5% | MIT (packaging); public domain (texts) | `28973b04f28f` |
| `tinystories` | backbone | 31% | CDLA-Sharing-1.0 | `f54c09fd2331` |
| `weird` | weird | 4% | MIT (packaging); public domain (texts) | `28973b04f28f` |
| `wikipedia_simple` | grounding | 15% | CC-BY-SA-3.0 | `b04c8d1ceb2f` |

## Attribution

- Project Gutenberg via sedthh/gutenberg_english
- Project Gutenberg via sedthh/gutenberg_english
- Project Gutenberg via sedthh/gutenberg_english
- Gutenberg Poetry Corpus (Allison Parrish), biglam/gutenberg-poetry-corpus
- Project Gutenberg via sedthh/gutenberg_english
- Project Gutenberg via sedthh/gutenberg_english
- TinyStories (Eldan & Li), roneneldan/TinyStories
- Project Gutenberg via sedthh/gutenberg_english
- Simple English Wikipedia contributors, via wikimedia/wikipedia

## What this project does and does not claim

**We do not redistribute the corpus.** This repository ships a *recipe* — pinned dataset revisions and a deterministic blend — not the text. Reconstructing it locally is what `scripts/fetch_corpus.py` and `scripts/blend_corpus.py` are for.

**What the blend actually contains** — real per-source token counts, achieved shares and repetition factors from the build itself — is recorded in [`corpus_blend.md`](corpus_blend.md) and [`measurements/blend_manifest.json`](measurements/blend_manifest.json). The table above is the *target*; that record is the *outcome*.

**Share-alike sources:** `tinystories` (CDLA-Sharing-1.0), `wikipedia_simple` (CC-BY-SA-3.0). Whether model weights trained on share-alike data constitute a Data Derivative is **unsettled**. This project does not assert that they do not. Anyone publishing weights trained with this recipe should reach their own conclusion rather than inheriting one.

**Project Gutenberg material** is public domain as *text*; the aggregations we fetch it through carry their own permissive terms, recorded in the table above. `scripts/prepare_corpus.py` strips Project Gutenberg headers, footers and front-matter packaging, and this project does not use the Project Gutenberg trademark.

This is a stated position, not legal advice.
