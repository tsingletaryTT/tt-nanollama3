"""The improv think-block: schema, extraction, rendering, parsing.

Slots hold SPANS LIFTED FROM THE TEXT, never paraphrases. There is no validated
generator here to paraphrase with, and putting an unvalidated model inside the data
pipeline would make every downstream number unattributable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional

SLOT_NAMES = ("offer", "accept", "add", "stakes", "handback")
STAKES_VALUES = ("up", "level", "down")

#: Small closed-class list. Deliberately not a package dependency — the corpus is simple
#: prose and a 40-word list is auditable where an opaque stopword set is not.
STOPWORDS = frozenset("""
a an the and or but if then than so as of to in on at by for with from into onto over
is was were be been being am are it its it's this that these those there here he she
they them his her their him us we you your i me my not no nor do did does done have
has had will would can could should may might must very just
""".split())

_SENT_SPLIT = re.compile(r'(?<=[.!?"])\s+')
_WORD = re.compile(r"[A-Za-z']+")


@dataclass(frozen=True)
class Slots:
    offer: str
    accept: str
    add: str
    stakes: str
    handback: str

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def content_words(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text) if w.lower() not in STOPWORDS]


def render_think(slots: Slots) -> str:
    body = "\n".join(f"{name}: {getattr(slots, name)}" for name in SLOT_NAMES)
    return f"<think>\n{body}\n</think>\n"


def parse_think(text: str) -> Optional[Slots]:
    """Parse a think-block, or None if malformed.

    Returns None rather than a partial object on purpose: schema adherence is reported as
    a rate, and a partial parse would inflate it.
    """
    m = re.search(r"<think>\s*(.*?)\s*</think>", text, re.S)
    if not m:
        return None
    found: Dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if key in SLOT_NAMES and value:
            found[key] = value
    if set(found) != set(SLOT_NAMES):
        return None
    if found["stakes"] not in STAKES_VALUES:
        return None
    return Slots(**found)


#: Below this the delta is noise rather than escalation. Calibrated in Task 3 against the
#: corpus and recorded there; the default is intentionally conservative.
STAKES_EPSILON = 0.5


def extract_slots(prefix: str, continuation: str, *, idf: Dict[str, float],
                  intensity: Callable[[str], float]) -> Optional[Slots]:
    """Derive a think-block from a real continuation, or None to DROP the example."""
    p_sents = split_sentences(prefix)
    if not p_sents or not continuation.strip():
        return None

    last = p_sents[-1]
    p_words = content_words(prefix)
    c_words = content_words(continuation)
    if not c_words:
        return None

    # accept: the longest run of shared content words between the final prefix sentence
    # and the continuation. Falls back to the commonest prefix word that reappears.
    last_words = content_words(last)
    carried = [w for w in last_words if w in set(c_words)]
    if not carried:
        carried = [w for w in sorted(set(p_words), key=p_words.count, reverse=True)
                   if w in set(c_words)][:1]
    if not carried:
        return None                      # nothing carried forward -> a block -> drop

    fresh = [w for w in c_words if w not in set(p_words)]
    if not fresh:
        return None                      # nothing added -> also a block -> drop
    fresh_ranked = sorted(set(fresh), key=lambda w: -idf.get(w, 0.0))
    add = ", ".join(fresh_ranked[:1])

    delta = intensity(continuation) - intensity(prefix)
    stakes = "up" if delta > STAKES_EPSILON else "down" if delta < -STAKES_EPSILON else "level"

    c_sents = split_sentences(continuation)
    tail = content_words(c_sents[-1]) if c_sents else []
    introduced = [w for w in tail if w in set(fresh)]
    handback = introduced[-1] if introduced else "open"

    return Slots(
        offer=" ".join(last_words[:12]) or last[:60],
        accept=" ".join(carried[:6]),
        add=add,
        stakes=stakes,
        handback=handback,
    )
