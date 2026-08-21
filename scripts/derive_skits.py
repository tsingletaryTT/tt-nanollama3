#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Corpus -> skits, and skits -> SFT examples.

SEGMENT MASKING, which is the part worth reading twice.
Stage 1 had one prompt and one completion, so its label rule was a single boundary:
`[-100]*(len(p_ids)-1) + c_ids + [-100]`. A skit has NINE alternating segments, so the same
rule is expressed positionally instead: build a per-position `supervised` mask over
`input_ids`, then

    labels[t] = input_ids[t+1] if supervised[t+1] else -100      (and labels[-1] = -100)

That generalises stage 1's boundary correctly and for free: the last position of an
unsupervised segment takes the FIRST token of the following supervised segment as its
target, which is exactly the transition being trained (predicting the first token of a
think-block from the partner's turn). Getting this off by one either drops that transition
or leaks a partner token into the loss.

KNOWN LIMITATION (see train/skit.py:31-38): split_sentences over-splits dialogue with
attribution, so a real fraction of stories drop here on a model turn that follows a
partner turn ending in dialogue. This script does not and must not "fix" the splitter --
that module's output sits behind the published stage-1 measurement. The drop report below
names this explicitly rather than hiding it inside an aggregate rate.

    python3 scripts/derive_skits.py --limit 20000
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.score_improv import intensity, load_harm_lexicon  # noqa: E402
from train.improv import content_words, split_sentences  # noqa: E402
from train.skit import MIN_SENTENCES, Skit, derive_skit, skit_segments  # noqa: E402

STORY_SEP = "</s>"

# Tile width ttml's device ops assume -- see scripts/derive_traces.py's TILE comment
# (task-2 SFTTrainer smoke finding under docs/superpowers/plans) for the underlying
# TT_FATAL. Every example this module hands off is padded to a multiple of this, so a
# batch collated from them is necessarily aligned too.
TILE = 32

HARM = load_harm_lexicon()
DROPS: Counter = Counter()


def build_idf(stories: List[str]) -> Dict[str, float]:
    """Corpus-wide inverse document frequency over `content_words`, one count per story.

    `derive_skit` uses this to rank which fresh word becomes a block's `add` slot --
    rarer (higher-idf) words are preferred, deterministically (the tie-break is the word
    itself, not set() iteration order -- see train/skit.py's _slots_for_turn).
    """
    df: Counter = Counter()
    for s in stories:
        df.update(set(content_words(s)))
    n = max(len(stories), 1)
    return {w: math.log(n / (1 + c)) for w, c in df.items()}


def build_skit_example(skit: Skit, tok, *, with_think: bool, pad_token_id: int) -> dict:
    """`{"input_ids", "labels"}` for one skit, pre-shifted for ttml and tile-aligned.

    Walks `skit_segments` in order, concatenating each segment's token ids and recording
    a per-position `supervised` flag from that segment's `(text, is_supervised)` pair.
    `add_special_tokens` is only True for the very first segment (the prefix) -- every
    later segment is a continuation of the same sequence, not a fresh one, so it must not
    pick up its own BOS.

    When `with_think` is False, think-block segments (identifiable by their rendered
    "<think>" prefix -- see train.improv.render_think) are skipped entirely: they
    contribute no ids and no supervised flags, so the no-think arm never sees the block's
    tokens at all, on either side of the label.

    LABELS: see this module's docstring for the positional generalisation of stage 1's
    boundary rule. `labels[t] = input_ids[t+1] if supervised[t+1] else -100`; the final
    position has no next token, so it is always masked, matching the pad tail.
    """
    ids: List[int] = []
    supervised: List[bool] = []
    first = True
    for text, sup in skit_segments(skit):
        if not with_think and text.lstrip().startswith("<think>"):
            continue
        seg = tok.encode(text, add_special_tokens=first)
        first = False
        ids.extend(seg)
        supervised.extend([sup] * len(seg))

    labels = [(ids[t + 1] if supervised[t + 1] else -100) for t in range(len(ids) - 1)]
    labels.append(-100)                                   # no next token for the last

    pad = (-len(ids)) % TILE
    ids.extend([pad_token_id] * pad)
    labels.extend([-100] * pad)
    return {"input_ids": ids, "labels": labels}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path,
                    default=ROOT / "artifacts" / "corpus" / "tinystories.txt")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "skits" / "skits.jsonl")
    args = ap.parse_args()

    stories = [s.strip() for s in
               args.corpus.read_text(errors="ignore").split(STORY_SEP) if s.strip()][:args.limit]
    print(f"stories read: {len(stories):,} (separator {STORY_SEP!r}, NOT a blank line)")
    idf = build_idf(stories)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with args.out.open("w") as fh:
        for i, story in enumerate(stories):
            # derive_skit itself collapses every drop reason -- too-short story, or any of
            # the three model turns failing accept/add -- into a single None (whole-or-
            # nothing by design, see train/skit.py:119-121). Recomputing split_sentences
            # here is read-only diagnostics on top of that same decision, not a second
            # implementation of it: it lets the report separate "too short to attempt" from
            # "long enough, but a turn failed", and flag the documented dialogue-splitter
            # limitation (train/skit.py:31-38) by name instead of burying it in one number.
            sents = split_sentences(story)
            if len(sents) < MIN_SENTENCES:
                DROPS["too_few_sentences"] += 1
                continue

            skit = derive_skit(story, story_id=i, idf=idf,
                              intensity=lambda t: intensity(t, HARM))
            if skit is None:
                turns = sents[2:7]
                # Signature of the known over-split: a partner turn (index 1 or 3 of the
                # five) that is nothing but a quoted line, split off from its own
                # attribution tail ('"It catches the light!" said her friend.' -> two
                # sentences). That partner turn then shares no content word with the model
                # turn that follows it, and accept/add fails on the boundary.
                dialogue_pattern = any(turns[p].rstrip().endswith('"') for p in (1, 3))
                DROPS["turn_derivation_failed_dialogue_pattern" if dialogue_pattern
                     else "turn_derivation_failed_other"] += 1
                continue
            fh.write(json.dumps(skit.as_dict()) + "\n")
            kept += 1

    total = len(stories)
    rate = 1 - kept / max(total, 1)
    manifest = {"corpus": str(args.corpus), "separator": STORY_SEP, "tile": TILE,
                "stories": total, "kept": kept, "drop_rate": round(rate, 4),
                "drops_by_rule": dict(DROPS)}
    (args.out.parent / "derive_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"kept {kept:,}/{total:,}  drop rate {rate:.1%}")
    for rule, n in DROPS.most_common():
        print(f"    {rule:28} {n:,}")
    if rate > 0.5:
        print("WARNING: drop rate above 50% — the FILTER is choosing the behaviour, not "
              "the model. Report this with any result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
