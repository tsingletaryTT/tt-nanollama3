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
