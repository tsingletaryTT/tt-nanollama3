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
