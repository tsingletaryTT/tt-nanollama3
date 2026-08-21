#!/usr/bin/env python3
# scripts/derive_traces.py
"""Derive extractive think-blocks from real corpus continuations.

The trace is read OFF the continuation that already exists: whatever the next passage
introduces IS the `add`, whatever it intensifies IS the `stakes`, whatever it leaves open
IS the `handback`. No generator is involved, so nothing unvalidated enters the pipeline.

Drop rules make the training set EXEMPLARY rather than representative — deliberate, since
we are teaching good moves, not average ones. Drop rates are reported per rule; above ~50%
the filter rather than the model is choosing the behaviour.

    python3 scripts/derive_traces.py --limit 20000 --out artifacts/improv/traces.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.score_improv import intensity, load_harm_lexicon  # noqa: E402
from train.improv import (content_words, extract_slots, render_think,  # noqa: E402
                          split_sentences)

STORY_SEP = "</s>"
HARM = load_harm_lexicon()
DROPS: Counter = Counter()


def derive_from_story(story: str, *, story_id: int, rng_seed: int,
                      idf: Optional[Dict[str, float]] = None) -> Optional[dict]:
    sents = split_sentences(story)
    if len(sents) < 4:
        DROPS["too_few_sentences"] += 1
        return None
    rng = random.Random(rng_seed + story_id)
    k = rng.randint(2, len(sents) - 2)
    prefix = " ".join(sents[:k])
    continuation = " ".join(sents[k:k + 2])

    slots = extract_slots(prefix, continuation, idf=idf or {},
                          intensity=lambda t: intensity(t, HARM))
    if slots is None:
        DROPS["no_carry_or_no_add"] += 1
        return None
    return {"story_id": story_id, "cut_k": k, "prefix": prefix,
            "think": render_think(slots), "continuation": continuation,
            "slots": slots.as_dict()}


# Tile width ttml's device ops assume. See the Task 2 SFTTrainer smoke finding
# (.superpowers/sdd/2026-08-20-improv-thinking/task-2-report.md): ttml's SDPA backward
# kernel mismatches a collated batch's raw sequence length against a tile-padded buffer
# whenever that length is not a multiple of 32. sft_collate_fn pads each batch to its
# longest example (dynamic, per batch) and does not itself guarantee tile alignment, so
# the guarantee is placed here instead: every example this module hands to sft_collate_fn
# is already a multiple of 32, which makes the batch max necessarily a multiple of 32 too.
# max_seq_len=512 (this project's cap) is itself a multiple of 32, so a truncated example
# stays aligned as well.
TILE = 32


def _sft_example_unaligned(rec: dict, tok, *, with_think: bool) -> dict:
    """Core `{"input_ids", "labels"}` construction, before tile-alignment padding.

    Factored out of `build_sft_examples` so a test can assert the `with_think` token-count
    delta directly against the un-padded lengths: `build_sft_examples`'s tile-alignment
    padding is chosen independently for each arm (it depends on that arm's raw length mod
    ``TILE``), so an aligned-length delta would not reliably equal the think-block's own
    token count and could mask a `with_think` regression on the wrong inputs. This function
    is the one true source of "how are input_ids/labels built from a trace", both for
    `build_sft_examples` and for that test.
    """
    prompt = rec["prefix"]
    completion = (rec["think"] + rec["continuation"]) if with_think else rec["continuation"]
    p_ids = tok.encode(prompt)
    c_ids = tok.encode(completion, add_special_tokens=False)
    return {"input_ids": p_ids + c_ids, "labels": [-100] * len(p_ids) + c_ids}


def build_sft_examples(traces: List[dict], tok, *, with_think: bool,
                       pad_token_id: int) -> List[dict]:
    """`{"input_ids", "labels"}` with -100 on prompt positions, for `sft_collate_fn`.

    Every example is padded up to the next multiple of `TILE` (32) tokens -- `pad_token_id`
    on `input_ids`, `-100` (masked, same as a prompt position) on `labels` -- so the loss is
    unaffected by the padding and the only cost is a little wasted compute. See `TILE`'s
    comment above for why this lives here rather than in a custom collate function.

    Args:
        traces: records with "prefix", "think", "continuation" (see `derive_from_story`).
        tok: a tokenizer exposing `.encode(text, add_special_tokens=...)`.
        with_think: include the think-block on the completion side when True.
        pad_token_id: token id used to pad `input_ids` out to a tile multiple. Passed
            explicitly (not read from a global) so callers own the choice -- this module
            has no tokenizer of its own to default it from.
    """
    out = []
    for rec in traces:
        ex = _sft_example_unaligned(rec, tok, with_think=with_think)
        pad_needed = (-len(ex["input_ids"])) % TILE
        if pad_needed:
            ex["input_ids"] = ex["input_ids"] + [pad_token_id] * pad_needed
            ex["labels"] = ex["labels"] + [-100] * pad_needed
        out.append(ex)
    return out


def build_idf(stories: List[str]) -> Dict[str, float]:
    df: Counter = Counter()
    for s in stories:
        df.update(set(content_words(s)))
    n = max(len(stories), 1)
    return {w: math.log(n / (1 + c)) for w, c in df.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path,
                    default=ROOT / "artifacts" / "corpus" / "tinystories.txt")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=5489)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "improv" / "traces.jsonl")
    args = ap.parse_args()

    stories = [s.strip() for s in args.corpus.read_text(errors="ignore").split(STORY_SEP)
               if s.strip()][:args.limit]
    print(f"stories read: {len(stories):,} (separator {STORY_SEP!r}, NOT a blank line)")
    idf = build_idf(stories)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with args.out.open("w") as fh:
        for i, story in enumerate(stories):
            rec = derive_from_story(story, story_id=i, rng_seed=args.seed, idf=idf)
            if rec is not None:
                fh.write(json.dumps(rec) + "\n")
                kept += 1

    total = len(stories)
    rate = 1 - kept / max(total, 1)
    manifest = {"corpus": str(args.corpus), "separator": STORY_SEP, "seed": args.seed,
                "stories": total, "kept": kept, "drop_rate": round(rate, 4),
                "drops_by_rule": dict(DROPS)}
    (args.out.parent / "derive_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"kept {kept:,}/{total:,}  drop rate {rate:.1%}")
    for rule, n in DROPS.most_common():
        print(f"    {rule:24} {n:,}")
    if rate > 0.5:
        print("WARNING: drop rate above 50% — the FILTER is choosing the behaviour, "
              "not the model. Report this with any result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
