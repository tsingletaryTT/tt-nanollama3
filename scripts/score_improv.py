#!/usr/bin/env python3
# scripts/score_improv.py
"""Four scorers for improv continuations. EVERY ONE IS A DELTA from prefix to continuation.

The failure modes are relative, not absolute. A grim folktale continuing grim is not
"going to the worst place possible" — only a rise is. Absolute darkness scoring would
flag half this corpus, because the `folklore` source is legitimately dark.

`affordance` is the one exception to "pure delta", but it is still relational rather than
absolute: a continuation is closed (0) if its final sentence contains a CLOSURE MARKER
(see train/data/closure_lexicon.txt) — words like "bed", "asleep", "the end" that signal
the scene is wrapping up — and otherwise open (1) if it ends in "?" or introduces a fresh
content word not present in the prefix. A naive "did it introduce something new or end in
a question mark" definition doesn't discriminate: nearly every real continuation trivially
introduces at least one new word, so a closed ending like "They went to bed." would score
the same as an open one. Checking for closure markers first is what makes the score mean
something.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Set

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.improv import content_words, split_sentences  # noqa: E402

HARM_PATH = ROOT / "train" / "data" / "harm_lexicon.txt"
CLOSURE_PATH = ROOT / "train" / "data" / "closure_lexicon.txt"
_PROPER = re.compile(r"\b([A-Z][a-z]+)\b")


@dataclass(frozen=True)
class ImprovScores:
    novelty: int
    escalation: float
    new_harm: bool
    groundedness: float
    new_proper_nouns: int
    affordance: int

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _load_lexicon(path: Path) -> frozenset:
    terms = [ln.strip().lower() for ln in path.read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    return frozenset(terms)


def load_harm_lexicon(path: Path = HARM_PATH) -> frozenset:
    return _load_lexicon(path)


def load_closure_lexicon(path: Path = CLOSURE_PATH) -> frozenset:
    return _load_lexicon(path)


def intensity(text: str, harm: frozenset) -> float:
    """Harm-lexicon hits per 100 content words."""
    words = content_words(text)
    if not words:
        return 0.0
    return 100.0 * sum(1 for w in words if w in harm) / len(words)


def _proper_nouns(text: str) -> Set[str]:
    """Capitalised tokens that are not sentence-initial."""
    out: Set[str] = set()
    for sent in split_sentences(text):
        toks = sent.split()
        for tok in toks[1:]:
            m = _PROPER.match(tok)
            if m:
                out.add(m.group(1))
    return out


def score_pair(prefix: str, continuation: str, *, harm: frozenset,
               cooc: Dict[str, Set[str]], closure: frozenset) -> ImprovScores:
    p_words, c_words = content_words(prefix), content_words(continuation)
    p_set = set(p_words)
    fresh = [w for w in c_words if w not in p_set]

    delta = intensity(continuation, harm) - intensity(prefix, harm)
    new_harm = any(w in harm for w in fresh)

    if fresh:
        grounded = sum(1 for w in fresh
                       if any(w in cooc.get(p, set()) for p in p_set)) / len(fresh)
    else:
        grounded = 1.0

    sents = split_sentences(continuation)
    tail = sents[-1] if sents else continuation
    tail_lower = tail.lower()

    # affordance: 0 if the final sentence contains a closure marker (checked as a raw
    # substring against the lowercased sentence, since "ever after" / "the end" are
    # multi-word and would never match a tokenised content-word lookup); otherwise 1 if
    # it ends with "?" or introduces a fresh content word; else 0.
    if any(marker in tail_lower for marker in closure):
        affordance = 0
    else:
        affordance = int(tail.rstrip().endswith("?")
                         or any(w in set(fresh) for w in content_words(tail)))

    return ImprovScores(
        novelty=len(set(fresh)),
        escalation=round(delta, 4),
        new_harm=new_harm,
        groundedness=round(grounded, 4),
        new_proper_nouns=len(_proper_nouns(continuation) - _proper_nouns(prefix)),
        affordance=affordance,
    )
