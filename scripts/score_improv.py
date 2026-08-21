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

Closure-marker matching is split by arity, on purpose:
  - Multi-word markers ("ever after", "the end") are matched as a raw lowercased substring
    against the whole final sentence, because they can never surface as a single token.
  - Single-word markers ("bed", "done", ...) are matched against WORD-TOKENISED text
    (word-boundary semantics), never as a raw substring — a raw substring match on a
    single word produces false positives ("done" inside "abandoned"/"condone"/"undone",
    "bed" inside "robbed"/"embedded") that silently mis-score an open ending as closed.
    Tokenisation here uses a plain regex word-split rather than `content_words`, because
    `content_words` drops stopwords and "done" is itself one of `train.improv.STOPWORDS`
    — running it through `content_words` would make "done" impossible to match at all.
"""
from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.improv import content_words, split_sentences  # noqa: E402

HARM_PATH = ROOT / "train" / "data" / "harm_lexicon.txt"
CLOSURE_PATH = ROOT / "train" / "data" / "closure_lexicon.txt"
_PROPER = re.compile(r"\b([A-Z][a-z]+)\b")
_WORD = re.compile(r"[A-Za-z']+")


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


def _is_closed(tail: str, closure: frozenset) -> bool:
    """True if the final sentence contains a closure marker.

    Multi-word markers are matched as a substring of the raw lowercased sentence.
    Single-word markers are matched against word-tokenised text only, so "done"
    cannot match inside "abandoned" and "bed" cannot match inside "robbed".
    """
    tail_lower = tail.lower()
    multi = {m for m in closure if " " in m}
    single = {m for m in closure if " " not in m}
    if any(marker in tail_lower for marker in multi):
        return True
    tail_words = set(_WORD.findall(tail_lower))
    return bool(tail_words & single)


#: Association table for `groundedness`. Counts, not a neighbour SET, because the boolean
#: "did these two words ever co-occur" version SATURATED: measured on the real 18,791-trace
#: table it returned a mean of 0.998 with 99.25% of scores exactly 1.0, so it could not
#: discriminate anything. The cause is structural rather than a bad threshold — the corpus
#: vocabulary is only 9,926 words but averages 548 neighbours each, and 641 hub words
#: (`about`, `after`, `again`, `afraid`, ...) exceed 2,000 neighbours, so 80.1% of prefix
#: words are hubs and an existential "connects to ANY prefix word" is true almost always.
#:
#: Normalised PMI fixes it by weighting association STRENGTH instead of existence, and is
#: bounded to [0, 1] so it still reads as a score. On the same real table: mean 0.259,
#: sd 0.089, range 0.069-0.716, and 0.00% at ceiling.
#:
#: Note the old version DID pass a discrimination test on constructed extremes
#: (grounded 1.000 vs "Gorthax and Vermilion argued about the Treaty of Blunn" 0.333). That
#: test was not wrong, it was UNREPRESENTATIVE — real continuations never look like that.
#: Hence `test_groundedness_is_not_saturated_on_the_real_corpus`, which asserts spread on the
#: actual distribution rather than separation on a hand-built pair.
def build_association(pairs: List[Tuple[str, str]]) -> Dict[str, object]:
    """Build the NPMI counts from (prefix, continuation) pairs. Document = one pair."""
    uni: Dict[str, int] = {}
    co: Dict[str, Dict[str, int]] = {}
    n = 0
    for prefix, continuation in pairs:
        words = set(content_words(prefix)) | set(content_words(continuation))
        n += 1
        for w in words:
            uni[w] = uni.get(w, 0) + 1
        for w in words:
            row = co.setdefault(w, {})
            for v in words:
                if v != w:
                    row[v] = row.get(v, 0) + 1
    return {"uni": uni, "co": co, "n": n}


def npmi(a: str, b: str, assoc: Dict[str, object]) -> float:
    """Normalised PMI in [0, 1]; 0 when either word or the pair is unseen."""
    uni, co, n = assoc["uni"], assoc["co"], assoc["n"]          # type: ignore[index]
    ca, cb = uni.get(a, 0), uni.get(b, 0)
    cab = co.get(a, {}).get(b, 0)
    if not (ca and cb and cab and n):
        return 0.0
    p_ab = cab / n
    denom = -math.log(p_ab)
    if denom <= 0:
        return 0.0
    return max(0.0, math.log(p_ab / ((ca / n) * (cb / n))) / denom)


def score_pair(prefix: str, continuation: str, *, harm: frozenset,
               assoc: Dict[str, object], closure: frozenset) -> ImprovScores:
    p_words, c_words = content_words(prefix), content_words(continuation)
    p_set = set(p_words)
    fresh = [w for w in c_words if w not in p_set]

    p_intensity = intensity(prefix, harm)
    c_intensity = intensity(continuation, harm)
    delta = c_intensity - p_intensity

    # new_harm: True only when the PREFIX carried no harm at all and the continuation
    # introduces some — i.e. harm arriving in a scene that had none, "going to the worst
    # place". A grim scene picking up more grim vocabulary is not that failure mode; that
    # is exactly why `escalation` (the delta) exists to capture rises. A scene that is
    # already grim staying grim (or getting grimmer) must NOT set new_harm — it must stay
    # False in that case, even though fresh words there may themselves be harm-lexicon
    # hits.
    new_harm = (p_intensity == 0.0) and (c_intensity > 0.0)

    if fresh:
        # Mean over fresh words of the STRONGEST association to any prefix word. See the
        # build_association docstring for why this is not a boolean co-occurrence test.
        grounded = sum(max((npmi(w, p, assoc) for p in p_set), default=0.0)
                       for w in fresh) / len(fresh)
    else:
        grounded = 1.0

    sents = split_sentences(continuation)
    tail = sents[-1] if sents else continuation

    if _is_closed(tail, closure):
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
