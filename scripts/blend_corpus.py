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

    The final pass is TRUNCATED. An earlier draft of this function wrote
    only whole passes over the source file, which cannot undershoot a large source: with
    tinystories offering 445M tokens against a 120M want, one pass emitted the entire file
    and the slice achieved 53% against a 30% target, with the blend totalling 839M tokens
    against a 400M budget. Truncation is what makes the achieved shares track the targets.

    Streams line by line rather than reading the file into memory: tinystories.txt is ~1.9 GB
    and ``text.split()`` over it would build a list of hundreds of millions of str objects.

    Truncation is at WORD level, not line level: a source whose paragraphs are single long
    lines cannot be trimmed at a line boundary, so overshoot would be bounded by the longest
    line rather than by a couple of percent.

    The truncation, streaming, and word-level-boundary behaviour described above is covered
    by ``tests/test_blend_corpus.py``.
    """
    if src_path.stat().st_size == 0:
        raise ValueError(f"{src_path} is empty; cannot emit {want_tokens:,} tokens from it")
    words = 0
    target_words = want_tokens / tokens_per_word
    while True:
        pass_words = 0
        with src_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.split()
                n = len(parts)
                if words + n >= target_words:
                    need = int(target_words - words)
                    if need > 0:
                        out.write(" ".join(parts[:need]))
                        out.write("\n")
                        words += need
                    return int(words * tokens_per_word)
                out.write(line)
                words += n
                pass_words += n
        if pass_words == 0:
            # size > 0 but nothing but whitespace: the repeat loop would never terminate.
            raise ValueError(f"{src_path} contains no words; cannot emit tokens from it")
        out.write("\n\n")


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
