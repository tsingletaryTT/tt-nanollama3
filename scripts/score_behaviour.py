#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Behavioural quality, as numbers with error bars — genre collapse, repetition, termination,
register, prompt engagement.

THE PROBLEM THIS MEASURES
--------------------------
Loss is honest now (``scripts/eval_per_source.py``), context use is measured
(``scripts/probe_context_use.py``), but the actual goal for this model is qualitative: prose
that is oblique, observational, strange-but-useful. Today that is assessed by a human reading
the 15 greedy completions ``scripts/generate_samples.py`` writes. Fifteen deterministic
completions cannot separate a real improvement from noise and cannot be run in a loop, so the
project's binding constraint is measurement, not training.

This script is the numeric version. It draws MANY sampled completions per frozen prompt and
reports five behaviours with standard errors. Every prompt set it reads is digest-pinned for
cross-checkpoint comparability and none of them is ever written here.

WHICH PROMPT SET, AND WHY THERE ARE TWO
---------------------------------------
``--prompt-set a`` (the default, ``docs/evaluation_prompts.json``, 15 prompts) is the original
frozen set and the one every committed measurement in ``docs/measurements/`` was produced
against. Its output filenames are unchanged, so every existing invocation stays reproducible.

``--prompt-set b`` (``docs/evaluation_prompts_b.json``, 45 prompts) exists because this script's
own comparison report found that power is capped by the PROMPT count, not the sample count: for
story-frame collapse the within-prompt sampling noise is 0.015 of the observed 0.042 paired SEM,
so doubling the completions per prompt buys ~3%. The earlier claim in this docstring -- "power
comes from more samples per prompt, never from more prompts" -- was the design intent and turned
out to be wrong; it is corrected here rather than left standing.

**The two sets are reported separately and are never pooled.** Set B's outputs carry ``-setB`` in
their filenames, its JSON records which set produced it, and ``--compare`` refuses to pair runs
from different sets. Two sets written at different times by different means are not exchangeable;
averaging them would silently redefine the metric rather than sharpen it.

THE FIVE SIGNALS
----------------
1. **Genre collapse** -- the documented failure mode: falling into the TinyStories attractor
   ("Once upon a time", "a little girl named X", "The moral of the story is"). A completion is
   *collapsed* if any marker in :data:`COLLAPSE_MARKERS` fires. The marker set is NOT a guess:
   every marker was selected by measuring its rate per million words in
   ``artifacts/corpus/tinystories.txt`` against the eight other prepared corpora, and the whole
   detector's operating point is re-measured on held-out corpus text every run (the "Detector
   controls" section of the report). At completion length it fires on about a third of genuine
   TinyStories passages and under 2% of passages from every other source -- so the reported
   rate is a **lower bound on collapse, usable as a comparator between models, not as an
   absolute prevalence**. Saying that out loud is the point; a detector whose sensitivity is
   unknown produces a number nobody can act on.

   Reported in three parts, because the union of the markers hides the most important thing
   they say. **Story-frame collapse** ("Once upon a time", "a little girl named X", "The moral
   of the story is") and **lexical-habit collapse** ("One day,", "was so happy", "his mom")
   move independently between checkpoints -- ``tt-tnt-v3`` very nearly eliminated the first and
   barely touched the second -- so each gets its own number alongside the union. See
   :data:`FRAME_MARKERS` for how the split is defined and when it was introduced.

2. **Degenerate repetition** -- word-level 4-gram repeat rate and longest repeated span within
   a completion. Two of the frozen prompts (``stutter-01``, ``stutter-02``) deliberately ASK
   for repetition, so penalising a model for obeying them would be measuring the prompt, not
   the model. Per-prompt numbers always include every prompt; the aggregate is reported twice,
   once over all prompts and once excluding the ``stutter`` probe, and both are labelled.

3. **Termination** -- fraction of completions that emit ``</s>`` (id 2) rather than running
   into the token limit. ``tt-tnt-v1``'s training blend contained no document separators at
   all, so it terminates essentially never *by construction*; this signal exists to show that
   the fix took.

4. **Register** -- "did it write like ``spine`` or like ``tinystories``", answered against the
   prepared per-source corpora rather than by a human squinting. One interpolated
   unigram+bigram language model per source (add-k smoothed, shared vocabulary), and a
   completion is scored under all nine. Reported as (a) which source's vocabulary it most
   resembles and (b) ``tinystories_margin``: log-likelihood per word under ``tinystories``
   minus the best of the other eight, in nats/word -- positive means the completion reads more
   like TinyStories than like anything else in the blend. Deliberately a model simple enough to
   inspect by hand; no classifier that cannot be explained. Its 9-way accuracy on held-out
   corpus text is re-measured every run and printed with the results.

5. **Prompt engagement** -- fraction of the prompt's content words that reappear in the
   completion. The weakest of the five and labelled as such in the report: it cannot tell
   engagement from parroting, so it is only meaningful read alongside the repetition signal.

STATISTICS
----------
Samples drawn from the same prompt are not independent draws of "model behaviour" -- they share
a prompt. So the aggregate is the mean over the 15 **per-prompt** means, with the standard error
taken **over prompts** (n=15), the same "what is the exchangeable sampling unit" convention
``eval_per_source.py`` and ``probe_context_use.py`` apply to windows. Per-prompt rows report the
standard error over that prompt's own samples (n=``--num-samples``). Every row states its own n;
no bare means anywhere. 95% intervals are the normal approximation ``mean ± 1.96 × SEM`` and are
labelled as such.

Comparing two runs (``--compare``) is **paired by prompt**: both runs answered the same frozen
prompts, so the difference is computed per prompt and averaged, which removes between-prompt
variance and is what makes a set this small able to see a real change at all.

CONSTRAINTS THIS SCRIPT RESPECTS
---------------------------------
CPU only. Never imports ttml/ttnn, never opens a Tenstorrent device. Never writes under
``artifacts/``. No dependency beyond numpy/transformers/tokenizers. Both prompt-set files under
``docs/`` are read-only here.

    python scripts/score_behaviour.py --hf-model artifacts/hf-tt-tnt-v3
    python scripts/score_behaviour.py --hf-model artifacts/hf-tt-tnt-v3 --prompt-set b
    python scripts/score_behaviour.py --compare docs/measurements/behaviour-tt-tnt-v1.json \\
        docs/measurements/behaviour-tt-tnt-v3.json --label tt-tnt-v1-vs-v3
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_per_source import mean_and_sem  # noqa: E402
from train.corpus import SOURCES  # noqa: E402

PROMPTS_PATH = ROOT / "docs" / "evaluation_prompts.json"
CORPUS_DIR = ROOT / "artifacts" / "corpus"


@dataclass(frozen=True)
class PromptSet:
    """One frozen prompt set: where it lives and how its outputs are named.

    ``suffix`` is what keeps a reader from ever mistaking one set's numbers for another's.
    Set A's suffix is deliberately EMPTY so that ``behaviour-tt-tnt-v3.md`` still means exactly
    what it meant when it was committed -- renaming set A's outputs would orphan every measurement
    already in ``docs/measurements/`` and every reference to them. Every set added afterwards
    carries its name in the filename.
    """

    key: str
    path: Path
    suffix: str
    description: str


#: The frozen prompt sets, by ``--prompt-set`` key. Adding a third set is a new entry here plus
#: its own digest test; it is never an edit to an existing set's file.
PROMPT_SETS: Dict[str, PromptSet] = {
    "a": PromptSet(
        key="a", path=PROMPTS_PATH, suffix="",
        description="set A (docs/evaluation_prompts.json, 15 prompts) -- the original frozen "
                    "set every committed measurement was produced against"),
    "b": PromptSet(
        key="b", path=ROOT / "docs" / "evaluation_prompts_b.json", suffix="-setB",
        description="set B (docs/evaluation_prompts_b.json, 45 prompts) -- a second frozen set "
                    "added for statistical power, derived from the corpus design; reported "
                    "beside set A and never pooled with it"),
}

#: Which set a run used when its JSON does not say. Every JSON written before ``--prompt-set``
#: existed was produced against set A -- set B did not exist yet -- so this is a fact about the
#: repository's history, not a guess, and it keeps ``--compare`` working on committed files.
LEGACY_PROMPT_SET = "a"

#: Completions drawn per prompt.
#:
#: This was once documented as the ONLY place power comes from. That was wrong, and the
#: correction is worth keeping visible: decomposing the paired SEM of the v1-vs-v3 comparison
#: showed within-prompt sampling noise is 0.015 of the observed 0.042 for story-frame collapse,
#: so doubling to 64 completions buys ~3%. Power is bought with PROMPTS -- which is why there is
#: a set B -- and this number is merely large enough not to be the bottleneck.
#: 32 x 15 prompts x 60 tokens takes well under a minute on CPU for a model this size.
DEFAULT_NUM_SAMPLES = 32

#: Matches ``scripts/generate_samples.py``'s default, so a behaviour run and the human-read
#: samples file describe completions of the same length.
DEFAULT_MAX_NEW_TOKENS = 60

#: Matches the committed ``samples-*-t0.8.md`` runs. Sampling, not greedy: greedy gives one
#: completion per prompt and therefore no variance to measure.
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.95

#: n-gram order for the repetition signal. 4 is long enough that ordinary English does not
#: trip it (real prose sits near 0.02, measured every run in the detector controls) and short
#: enough to catch "The bees were busy, and the bees were busy".
DEFAULT_REPEAT_N = 4

#: Words read per source to fit the register language models, plus a held-out tail used to
#: measure how well those models actually separate the sources.
DEFAULT_REGISTER_WORDS = 1_000_000
DEFAULT_CONTROL_WORDS = 200_000

#: Vocabulary is the union of each source's top-N word types; everything else maps to a single
#: <unk>. Keeps the nine models over one shared event space so their log-likelihoods compare.
DEFAULT_REGISTER_VOCAB = 20_000

#: Add-k smoothing constant and the bigram/unigram interpolation weight. Fixed, not fitted --
#: a fitted knob would be one more thing to explain and this signal's job is to be explainable.
REGISTER_ADD_K = 0.1
REGISTER_BIGRAM_WEIGHT = 0.5

#: Bigrams seen fewer than this many times in a source are dropped and fall back to the
#: smoothing floor. Bounds memory; also stops a single accidental phrase from defining a source.
REGISTER_MIN_BIGRAM_COUNT = 2

#: The probe tag on the frozen prompts that deliberately ask for repetition. Those prompts are
#: excluded from ONE of the two repetition aggregates -- never silently from both.
DELIBERATE_REPETITION_PROBE = "stutter"

#: EOS id. Fixed by the tokenizer this project trained (`artifacts/tokenizer/`), and asserted
#: against the model's own config at run time rather than trusted.
EXPECTED_EOS_ID = 2

_WORD_RE = re.compile(r"[a-z0-9']+")

#: Function words carry no prompt-specific content, so requiring a completion to echo them
#: would score every completion the same. Short, fixed, and inspectable -- no nltk.
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for from by with
without into onto over under again is are was were be been being am do does did doing
have has had having i you he she it we they them him her his hers its their our your my
me us not no so as such very more most other some any each which who whom what when where
why how all both few many much only own same too can will just should now there here
""".split())


# ---------------------------------------------------------------------------------------
# Signal 1: genre collapse
# ---------------------------------------------------------------------------------------

#: The TinyStories attractor, as regexes over the completion text.
#:
#: Every marker here was chosen by measurement, not taste. Rates per million words over the
#: first ~2-3M words of each prepared corpus (``artifacts/corpus/*.txt``), tinystories vs. the
#: mean of the eight other sources:
#:
#:     marker                  tinystories/M   other-sources/M    lift
#:     once_upon_a_time             3457.7               11.9    292x
#:     little_X_named               1542.3                0.3   5602x
#:     moral_of_the_story            163.3                0.1   1623x
#:     happily_ever_after             96.3                1.7     56x
#:     from_that_day_on              599.7                0.1  10808x
#:     learned_a_lesson               75.0                0.3    286x   (merged with the
#:                                                                       "learned that it's
#:                                                                       important" variant)
#:     the_end_terminal              417.3                9.4     45x
#:     one_day_comma                3686.3               21.6    171x
#:     so_very_happy                1701.7                4.0    420x
#:     named_proper                 2428.7               46.5     52x
#:     his_her_mom                  4102.3                0.1  49228x
#:
#: Nothing under ~45x lift is included. The composite detector's real operating point -- what
#: it does to whole passages of completion length rather than to whole corpora -- is measured
#: fresh on held-out corpus text on every run and printed in the report; do not quote the lift
#: table above as if it were the detector's accuracy.
COLLAPSE_MARKERS: Tuple[Tuple[str, str], ...] = (
    ("once_upon_a_time", r"once upon a time"),
    ("little_X_named",
     r"\b(?:little|small|young|tiny)\s+(?:girl|boy|dog|cat|bird|mouse|rabbit|bear|fish|duck"
     r"|frog)\s+named\b"),
    ("moral_of_the_story", r"the moral of the story"),
    ("happily_ever_after", r"happily ever after"),
    ("from_that_day_on", r"from that day on"),
    ("learned_a_lesson",
     r"learned (?:a|an important) lesson|learned that it(?:'s| is) important"),
    ("the_end_terminal", r"the end\."),
    ("one_day_comma", r"\bone day,"),
    ("so_very_happy", r"(?:was|were) (?:so|very) happy"),
    ("named_proper", r"\bnamed\s+[A-Z][a-z]+"),
    ("his_her_mom", r"\b(?:his|her) (?:mom|mommy)\b"),
)

#: Compiled once. ``named_proper`` is the only capitalisation-sensitive marker (it keys on a
#: proper noun following "named"), so it alone is compiled without IGNORECASE.
_COLLAPSE_COMPILED: Tuple[Tuple[str, "re.Pattern[str]"], ...] = tuple(
    (name, re.compile(pattern, 0 if name == "named_proper" else re.IGNORECASE))
    for name, pattern in COLLAPSE_MARKERS
)

#: The markers that are the STORY FRAME -- a formulaic opening, closing, or moral. The rest
#: (``one_day_comma``, ``so_very_happy``, ``named_proper``, ``his_her_mom``) are TinyStories
#: *lexical habits*: vocabulary and syntax the model picked up, carried by prose that need not
#: be a fairy tale at all.
#:
#: WHY THIS SPLIT EXISTS, HONESTLY. It was added AFTER the first v1-vs-v3 comparison, not
#: before, and the reason matters. The composite detector -- markers of both kinds unioned --
#: said v1 and v3 collapse at indistinguishable rates, contradicting a hand count of 9/15 vs
#: 1/15. The per-marker breakdown showed why: ``once_upon_a_time`` went 10.8% -> 0.0% and
#: ``little_X_named`` 8.3% -> 0.4% between the two models, while ``one_day_comma`` and
#: ``so_very_happy`` barely moved. The frame markers are the ones a human reading samples
#: actually counts, and they had improved enormously; the lexical ones had not, and dominated
#: the union. Both facts are true, and reporting only their union hid both.
#:
#: So: the composite ``collapse_rate`` is retained UNCHANGED alongside the two parts, and this
#: split is by what a marker *is* (frame vs. vocabulary), which is inspectable, not by a
#: threshold fitted until the answer came out right. ``the_end_terminal`` is a frame marker and
#: ``one_day_comma`` a lexical one on that definition regardless of which way each makes the
#: numbers move. Judge the split by reading the two lists, not by trusting this note.
FRAME_MARKERS: frozenset = frozenset({
    "once_upon_a_time", "little_X_named", "moral_of_the_story", "happily_ever_after",
    "from_that_day_on", "learned_a_lesson", "the_end_terminal",
})

LEXICAL_MARKERS: frozenset = frozenset(
    name for name, _pattern in COLLAPSE_MARKERS) - FRAME_MARKERS


def collapse_markers_found(text: str) -> List[str]:
    """Names of every TinyStories-attractor marker firing in ``text``, in declaration order.

    Matched against the COMPLETION only, never against prompt+completion: three of the frozen
    prompts hand the model narrative framing outright (``voice-03`` literally opens "Once upon
    a time, there was a little"), and a detector that read the prompt would credit the model
    with the prompt's own genre.
    """
    return [name for name, rx in _COLLAPSE_COMPILED if rx.search(text)]


def is_collapsed(text: str) -> bool:
    """True if ``text`` shows any sign of the TinyStories attractor, of either kind."""
    return bool(collapse_markers_found(text))


def is_frame_collapsed(text: str) -> bool:
    """True if ``text`` uses the TinyStories STORY FRAME -- a formulaic opening/closing/moral.

    This is the sub-signal closest to what a human counts when reading samples, and the one
    that moves most between checkpoints. See :data:`FRAME_MARKERS`.
    """
    return any(name in FRAME_MARKERS for name in collapse_markers_found(text))


def is_lexical_collapsed(text: str) -> bool:
    """True if ``text`` uses TinyStories VOCABULARY HABITS without necessarily the frame.

    A model can stop writing fairy tales and still write in fairy-tale words; that is a
    different (and, on the evidence so far, more stubborn) failure, so it gets its own number.
    """
    return any(name in LEXICAL_MARKERS for name in collapse_markers_found(text))


# ---------------------------------------------------------------------------------------
# Signal 2: degenerate repetition
# ---------------------------------------------------------------------------------------


def words_of(text: str) -> List[str]:
    """Lowercased word tokens. Punctuation and casing dropped -- "The bees. The bees" repeats."""
    return _WORD_RE.findall(text.lower())


def ngram_repeat_rate(words: Sequence[str], n: int = DEFAULT_REPEAT_N) -> Optional[float]:
    """Fraction of ``n``-grams in ``words`` that are not the first occurrence of their type.

    ``0.0`` means every n-gram is distinct; ``1.0`` means a single n-gram repeats forever
    (e.g. "a a a a a" at n=4 has two identical 4-grams, one of which is a repeat: 0.5; at ten
    ``a``s it is 6/7). Returns ``None`` -- not 0.0 -- when the text is shorter than ``n``
    words, so "too short to say" is never averaged in as "perfectly non-repetitive".
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if len(words) < n:
        return None
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def longest_repeated_ngram(words: Sequence[str]) -> int:
    """Length of the longest word n-gram occurring at least twice in ``words``.

    0 when every word is distinct. Complements the rate: a completion can have a low 4-gram
    repeat rate and still contain one enormous verbatim loop, or a high rate made of many
    short ones, and those are different failures.
    """
    best = 0
    # An n-gram of length L repeating implies one of length L-1 repeating, so scanning upward
    # and stopping at the first n with no repeats finds the maximum without checking every n.
    for n in range(1, len(words) + 1):
        grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        if len(set(grams)) == len(grams):
            break
        best = n
    return best


# ---------------------------------------------------------------------------------------
# Signal 3: termination
# ---------------------------------------------------------------------------------------


def is_terminated(generated_ids: Sequence[int], eos_id: int = EXPECTED_EOS_ID) -> bool:
    """True if the model chose to stop -- ``eos_id`` appears among the GENERATED ids.

    ``generated_ids`` must already exclude the prompt. Presence anywhere is the test rather
    than "is the last token", because a batched ``generate`` pads everything after EOS out to
    the longest row in the batch, so the final id of a terminated row is usually the pad token.
    """
    return eos_id in list(generated_ids)


def strip_at_eos(generated_ids: Sequence[int], eos_id: int = EXPECTED_EOS_ID) -> List[int]:
    """``generated_ids`` truncated before the first ``eos_id`` (padding thereby dropped too)."""
    ids = list(generated_ids)
    if eos_id in ids:
        return ids[:ids.index(eos_id)]
    return ids


# ---------------------------------------------------------------------------------------
# Signal 4: register (per-source unigram+bigram language models)
# ---------------------------------------------------------------------------------------


class SourceLM:
    """An interpolated unigram+bigram language model over one corpus source.

    ``P(w | prev) = λ·P_bigram + (1-λ)·P_unigram``, both add-k smoothed over a shared
    vocabulary, with a single ``<unk>`` slot. Chosen because every number it produces can be
    traced back to a word count by hand -- the brief was explicitly a model that can be
    interpreted, not the most accurate discriminator obtainable.
    """

    def __init__(self, ids: Sequence[int], vocab_size: int, *, add_k: float = REGISTER_ADD_K,
                 bigram_weight: float = REGISTER_BIGRAM_WEIGHT,
                 min_bigram_count: int = REGISTER_MIN_BIGRAM_COUNT) -> None:
        if not ids:
            raise ValueError("cannot fit a source language model on zero words")
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
        self.vocab_size = vocab_size
        self.add_k = add_k
        self.bigram_weight = bigram_weight
        self.n_words = len(ids)

        unigram = np.zeros(vocab_size, dtype=np.int64)
        np.add.at(unigram, np.asarray(ids, dtype=np.int64), 1)
        self._unigram = unigram
        self._uni_den = float(self.n_words + add_k * vocab_size)

        counts: Dict[Tuple[int, int], int] = defaultdict(int)
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] += 1
        self._bigram = {k: v for k, v in counts.items() if v >= min_bigram_count}
        context = np.zeros(vocab_size, dtype=np.int64)
        for (a, _b), v in self._bigram.items():
            context[a] += v
        self._bigram_context = context

    def logprob_per_word(self, ids: Sequence[int]) -> float:
        """Mean natural-log probability per word of ``ids`` under this source.

        Per WORD, not summed, so completions of different lengths are comparable -- a long
        completion is not automatically "less like" every source than a short one.
        """
        if not ids:
            raise ValueError("cannot score an empty token sequence")
        total = 0.0
        prev: Optional[int] = None
        k, v_size = self.add_k, self.vocab_size
        for i in ids:
            p_uni = (self._unigram[i] + k) / self._uni_den
            if prev is None:
                p = p_uni
            else:
                c = self._bigram.get((prev, i), 0)
                p_bi = (c + k) / (self._bigram_context[prev] + k * v_size)
                p = self.bigram_weight * p_bi + (1.0 - self.bigram_weight) * p_uni
            total += math.log(p)
            prev = i
        return total / len(ids)


@dataclass
class RegisterProfile:
    """The nine fitted source models plus the shared vocabulary they share an event space on."""

    word_to_id: Dict[str, int]
    unk_id: int
    models: Dict[str, SourceLM]
    train_words: Dict[str, int]
    #: Held-out RAW tokens per source (case and punctuation intact), for the detector controls.
    holdout: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def sources(self) -> List[str]:
        return list(self.models)

    def encode(self, words: Iterable[str]) -> List[int]:
        return [self.word_to_id.get(w, self.unk_id) for w in words]

    def score(self, words: Sequence[str]) -> Dict[str, float]:
        """Log-likelihood per word of ``words`` under each source. Empty input -> ``{}``."""
        if not words:
            return {}
        ids = self.encode(words)
        return {name: lm.logprob_per_word(ids) for name, lm in self.models.items()}

    def nearest_source(self, words: Sequence[str]) -> Optional[str]:
        scores = self.score(words)
        if not scores:
            return None
        return max(scores, key=lambda s: scores[s])

    def tinystories_margin(self, words: Sequence[str],
                           target: str = "tinystories") -> Optional[float]:
        """``logP_target`` minus the best OTHER source, in nats/word. Positive = leans target.

        A margin rather than a raw likelihood because raw likelihood mostly measures how
        ordinary the text is: any fluent English scores well under every source. The margin
        asks the only question that matters here -- of the nine registers the model was trained
        on, is this one the TinyStories one?
        """
        scores = self.score(words)
        if not scores or target not in scores:
            return None
        others = [v for s, v in scores.items() if s != target]
        if not others:
            raise ValueError("register profile has only one source; a margin is meaningless")
        return scores[target] - max(others)


def read_corpus_tokens(path: Path, limit: int) -> List[str]:
    """First ``limit`` whitespace-separated tokens of ``path`` — **verbatim**, case and
    punctuation intact.

    Raw rather than normalised because the held-out tail of this same read is what the detector
    controls run on, and the collapse detector is case- and punctuation-sensitive by design
    (``the end.`` needs its full stop, ``named Lily`` needs its capital). Lowercasing here would
    silently disarm two of the eleven markers and make the measured sensitivity and
    false-positive rate both wrong, in the same direction, without saying so.

    Reads a prefix, not a random sample, and that is a real limitation:
    ``scripts/blend_corpus.py`` builds the training blend from each source's prefix too, so the
    prefix is at least the part of the source the model actually saw -- but Gutenberg-derived
    sources open with title pages and tables of contents, which is register noise the model
    would not associate with the source's body prose.
    """
    tokens: List[str] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            tokens.extend(line.split())
            if len(tokens) >= limit:
                break
    return tokens[:limit]


def build_register_profile(corpus_dir: Path, source_names: Sequence[str], *,
                           train_words: int = DEFAULT_REGISTER_WORDS,
                           control_words: int = DEFAULT_CONTROL_WORDS,
                           vocab_top: int = DEFAULT_REGISTER_VOCAB,
                           log=lambda msg: None) -> RegisterProfile:
    """Fit one :class:`SourceLM` per source, holding out a tail of each for the controls.

    Each source's file is read once, up to ``train_words + control_words`` words. The first
    ``train_words`` fit the model and the remainder is held out; a source with fewer words than
    ``train_words`` in total is split 90/10 instead, so a small source (``flavour``, ~400k
    words) still contributes both a model and a control set rather than being dropped.

    Raises ``FileNotFoundError`` naming the first missing corpus file -- the register signal is
    the one this project most wants and silently degrading it to "8 sources, one of them
    missing" would corrupt every margin computed against it.
    """
    raw: Dict[str, List[str]] = {}
    for name in source_names:
        path = corpus_dir / f"{name}.txt"
        if not path.is_file():
            raise FileNotFoundError(
                f"corpus source {path} not found -- the register signal needs every prepared "
                f"source (run scripts/prepare_corpus.py, or pass --no-register to score the "
                f"other four signals without it)."
            )
        raw[name] = read_corpus_tokens(path, train_words + control_words)
        log(f"  read {len(raw[name]):,} words from {path.name}")

    # Split on RAW tokens, then derive the model's word tokens from the training side only.
    # Splitting raw keeps the held-out side verbatim for the detector controls; deriving the
    # training words with `words_of` means the register models are fit over exactly the same
    # tokenisation that completions are scored with.
    train: Dict[str, List[str]] = {}
    holdout: Dict[str, List[str]] = {}
    for name, tokens in raw.items():
        cut = train_words if len(tokens) > train_words else int(len(tokens) * 0.9)
        train[name] = words_of(" ".join(tokens[:cut]))
        holdout[name] = tokens[cut:]

    vocab: set = set()
    for name, words in train.items():
        vocab.update(w for w, _ in Counter(words).most_common(vocab_top))
    ordered = sorted(vocab)
    word_to_id = {w: i for i, w in enumerate(ordered)}
    unk_id = len(ordered)
    log(f"  shared vocabulary: {unk_id:,} types (+1 <unk>)")

    models: Dict[str, SourceLM] = {}
    for name, words in train.items():
        ids = [word_to_id.get(w, unk_id) for w in words]
        models[name] = SourceLM(ids, unk_id + 1)
        log(f"  fitted {name} on {len(ids):,} words")

    return RegisterProfile(word_to_id=word_to_id, unk_id=unk_id, models=models,
                           train_words={k: len(v) for k, v in train.items()}, holdout=holdout)


# ---------------------------------------------------------------------------------------
# Signal 5: prompt engagement
# ---------------------------------------------------------------------------------------


def _stem(word: str) -> str:
    """Crude singularisation: "bees" -> "bee". Deliberately shallow, and its shallowness is
    reported -- it catches plurals and misses everything else ("children"/"child")."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def content_words(text: str) -> List[str]:
    """Stemmed content words of ``text``: function words and 1-2 character tokens removed."""
    return [_stem(w) for w in words_of(text) if w not in _STOPWORDS and len(w) > 2]


def prompt_engagement(prompt: str, completion: str) -> Optional[float]:
    """Fraction of the PROMPT's content words that reappear in ``completion``.

    ``None`` when the prompt has no content words (never true of the frozen set, but the
    arithmetic must not invent a 0.0). This is the weakest of the five signals and the report
    says so: it cannot distinguish engaging with the prompt from parroting it, which is exactly
    the failure the repetition signal catches, so the two must be read together.
    """
    wanted = set(content_words(prompt))
    if not wanted:
        return None
    got = set(content_words(completion))
    return len(wanted & got) / len(wanted)


# ---------------------------------------------------------------------------------------
# Per-completion scoring and aggregation
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CompletionScore:
    """Every signal for one completion. ``None`` means "not computable", never "zero"."""

    text: str
    n_tokens: int
    terminated: bool
    collapsed: bool
    frame_collapsed: bool
    lexical_collapsed: bool
    collapse_markers: Tuple[str, ...]
    repeat_rate: Optional[float]
    longest_repeat: int
    register_nearest: Optional[str]
    tinystories_margin: Optional[float]
    engagement: Optional[float]


def score_completion(prompt_text: str, text: str, generated_ids: Sequence[int], *,
                     profile: Optional[RegisterProfile] = None,
                     repeat_n: int = DEFAULT_REPEAT_N,
                     eos_id: int = EXPECTED_EOS_ID) -> CompletionScore:
    """All five signals for a single completion.

    ``generated_ids`` are the model's own output ids with the prompt removed; ``text`` is that
    same output decoded with special tokens skipped.
    """
    words = words_of(text)
    markers = collapse_markers_found(text)
    return CompletionScore(
        text=text,
        n_tokens=len(strip_at_eos(generated_ids, eos_id)),
        terminated=is_terminated(generated_ids, eos_id),
        collapsed=bool(markers),
        frame_collapsed=any(m in FRAME_MARKERS for m in markers),
        lexical_collapsed=any(m in LEXICAL_MARKERS for m in markers),
        collapse_markers=tuple(markers),
        repeat_rate=ngram_repeat_rate(words, repeat_n),
        longest_repeat=longest_repeated_ngram(words),
        register_nearest=profile.nearest_source(words) if profile else None,
        tinystories_margin=profile.tinystories_margin(words) if profile else None,
        engagement=prompt_engagement(prompt_text, text),
    )


@dataclass(frozen=True)
class Estimate:
    """A mean with its standard error and the n it rests on. ``n`` is never implied."""

    mean: float
    sem: float
    n: int

    @property
    def ci95_lo(self) -> float:
        return self.mean - 1.96 * self.sem

    @property
    def ci95_hi(self) -> float:
        return self.mean + 1.96 * self.sem

    def as_json(self) -> dict:
        return {"mean": self.mean, "sem": self.sem, "n": self.n,
                "ci95": [self.ci95_lo, self.ci95_hi]}


def estimate(values: Sequence[Optional[float]]) -> Optional[Estimate]:
    """Mean ± SEM over ``values``, skipping ``None``. ``None`` if nothing is computable.

    Uses ``eval_per_source.mean_and_sem`` -- imported rather than reimplemented, so this script
    and the loss scripts can never drift into two subtly different definitions of SEM (ddof=1;
    0.0 rather than NaN at n=1).
    """
    usable = [float(v) for v in values if v is not None]
    if not usable:
        return None
    mean, sem = mean_and_sem(usable)
    return Estimate(mean=mean, sem=sem, n=len(usable))


#: Every reported signal, with the direction that counts as an improvement. Kept as data so the
#: comparison renderer cannot disagree with the measurement renderer about which way is better.
#: ``None`` marks a diagnostic that has no "good" direction.
SIGNALS: Tuple[Tuple[str, str, Optional[str]], ...] = (
    ("collapse_rate", "genre collapse rate (any marker)", "lower"),
    ("frame_collapse_rate", "— story-frame collapse", "lower"),
    ("lexical_collapse_rate", "— lexical-habit collapse", "lower"),
    ("termination_rate", "termination rate (</s>)", "higher"),
    ("repeat_rate", "4-gram repeat rate", "lower"),
    ("longest_repeat", "longest repeated span (words)", "lower"),
    ("tinystories_margin", "tinystories margin (nats/word)", "lower"),
    ("register_tinystories_share", "nearest source == tinystories", "lower"),
    ("engagement", "prompt engagement", "higher"),
    ("n_tokens", "generated tokens before </s>", None),
)


@dataclass
class PromptReport:
    """One frozen prompt's ``--num-samples`` completions, reduced to per-signal estimates."""

    prompt_id: str
    probe: str
    text: str
    n_samples: int
    estimates: Dict[str, Optional[Estimate]]
    #: Marker name -> how many of this prompt's completions it fired on.
    marker_counts: Dict[str, int]
    #: Source name -> how many completions landed nearest it.
    nearest_counts: Dict[str, int]


def summarise_prompt(prompt: dict, scores: Sequence[CompletionScore]) -> PromptReport:
    """Reduce one prompt's completions to a per-signal mean ± SEM over those completions."""
    if not scores:
        raise ValueError(f"prompt {prompt['id']} produced no completions to summarise")
    estimates: Dict[str, Optional[Estimate]] = {
        "collapse_rate": estimate([1.0 if s.collapsed else 0.0 for s in scores]),
        "frame_collapse_rate": estimate([1.0 if s.frame_collapsed else 0.0 for s in scores]),
        "lexical_collapse_rate": estimate(
            [1.0 if s.lexical_collapsed else 0.0 for s in scores]),
        "termination_rate": estimate([1.0 if s.terminated else 0.0 for s in scores]),
        "repeat_rate": estimate([s.repeat_rate for s in scores]),
        "longest_repeat": estimate([float(s.longest_repeat) for s in scores]),
        "tinystories_margin": estimate([s.tinystories_margin for s in scores]),
        "register_tinystories_share": estimate(
            [None if s.register_nearest is None else
             (1.0 if s.register_nearest == "tinystories" else 0.0) for s in scores]),
        "engagement": estimate([s.engagement for s in scores]),
        "n_tokens": estimate([float(s.n_tokens) for s in scores]),
    }
    markers: Counter = Counter()
    for s in scores:
        markers.update(s.collapse_markers)
    nearest: Counter = Counter(
        s.register_nearest for s in scores if s.register_nearest is not None)
    return PromptReport(prompt_id=prompt["id"], probe=prompt["probe"], text=prompt["text"],
                        n_samples=len(scores), estimates=estimates,
                        marker_counts=dict(markers), nearest_counts=dict(nearest))


def aggregate_over_prompts(reports: Sequence[PromptReport], signal: str, *,
                           exclude_probe: Optional[str] = None) -> Optional[Estimate]:
    """Mean ± SEM of ``signal`` **over prompts** -- the honest aggregate.

    The per-prompt mean is one observation; the SEM is over the 15 of them. Pooling all
    15 x --num-samples completions and taking a SEM over that would treat 32 completions of the
    same prompt as 32 independent observations of the model's behaviour, which they are not,
    and would report an interval several times too narrow. Same reasoning as
    ``probe_context_use.py``'s "the window, not the token, is the sampling unit".
    """
    values = [r.estimates[signal].mean for r in reports
              if (exclude_probe is None or r.probe != exclude_probe)
              and r.estimates.get(signal) is not None]
    if not values:
        return None
    mean, sem = mean_and_sem(values)
    return Estimate(mean=mean, sem=sem, n=len(values))


# ---------------------------------------------------------------------------------------
# Detector controls: what the detectors do on text whose answer is already known
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlRow:
    """One source's held-out corpus text run through the detectors, at completion length."""

    source: str
    n_excerpts: int
    collapse_rate: float
    frame_collapse_rate: float
    lexical_collapse_rate: float
    register_accuracy: float
    register_top_confusion: str
    repeat_rate: Optional[Estimate]
    #: marker name -> fraction of this source's excerpts it fired on.
    marker_rates: Dict[str, float] = field(default_factory=dict)


def detector_controls(profile: RegisterProfile, excerpt_words: int,
                      max_excerpts: int = 1000) -> List[ControlRow]:
    """Run collapse/register/repetition over held-out corpus text of completion length.

    This is the part that makes the metric checkable rather than merely computable. The held-out
    tail of ``tinystories.txt`` IS genre collapse, so the collapse detector's rate there is its
    sensitivity; the other eight sources are not, so its rate there is its false-positive rate.
    Likewise the register model's 9-way accuracy here is the accuracy of the register signal at
    the length it is actually used at. Recomputed every run, so the report can never quote a
    calibration that no longer matches the code.
    """
    if excerpt_words < 1:
        raise ValueError(f"excerpt_words must be >= 1, got {excerpt_words}")
    rows: List[ControlRow] = []
    for source, tokens in profile.holdout.items():
        chunks = [tokens[i * excerpt_words:(i + 1) * excerpt_words]
                  for i in range(len(tokens) // excerpt_words)][:max_excerpts]
        if not chunks:
            rows.append(ControlRow(source=source, n_excerpts=0, collapse_rate=float("nan"),
                                   frame_collapse_rate=float("nan"),
                                   lexical_collapse_rate=float("nan"),
                                   register_accuracy=float("nan"), register_top_confusion="n/a",
                                   repeat_rate=None))
            continue
        collapsed = frame = lexical = 0
        markers: Counter = Counter()
        predictions: Counter = Counter()
        repeats: List[Optional[float]] = []
        for chunk in chunks:
            # Verbatim corpus text -- case and punctuation intact -- so every marker, including
            # the two that need them (`the_end_terminal`, `named_proper`), is exercised exactly
            # as it is on a real completion. Each excerpt then goes through the same
            # words_of/nearest_source/ngram_repeat_rate path a completion does.
            text = " ".join(chunk)
            words = words_of(text)
            found = collapse_markers_found(text)
            markers.update(found)
            collapsed += 1 if found else 0
            frame += 1 if any(m in FRAME_MARKERS for m in found) else 0
            lexical += 1 if any(m in LEXICAL_MARKERS for m in found) else 0
            predictions[profile.nearest_source(words) or "n/a"] += 1
            repeats.append(ngram_repeat_rate(words))
        wrong = [(s, c) for s, c in predictions.most_common() if s != source]
        rows.append(ControlRow(
            source=source, n_excerpts=len(chunks),
            collapse_rate=collapsed / len(chunks),
            frame_collapse_rate=frame / len(chunks),
            lexical_collapse_rate=lexical / len(chunks),
            register_accuracy=predictions[source] / len(chunks),
            register_top_confusion=(f"{wrong[0][0]} {wrong[0][1] / len(chunks):.1%}"
                                    if wrong else "none"),
            repeat_rate=estimate(repeats),
            marker_rates={name: markers.get(name, 0) / len(chunks)
                          for name, _pattern in COLLAPSE_MARKERS}))
    return rows


# ---------------------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------------------


def resolve_prompt_set(key: str) -> PromptSet:
    """Look up a prompt set by ``--prompt-set`` key, naming the alternatives on a miss."""
    try:
        return PROMPT_SETS[key]
    except KeyError:
        raise KeyError(f"unknown prompt set {key!r}; known sets: "
                       f"{sorted(PROMPT_SETS)}") from None


def load_prompts(path: Path = PROMPTS_PATH) -> List[dict]:
    """A frozen prompt set's prompts. Read-only: this script never writes one.

    Defaults to set A so that every caller written before ``--prompt-set`` existed keeps reading
    the set it was written against.
    """
    return json.loads(path.read_text())["prompts"]


def resolve_model_dir(model: Path) -> Path:
    """Validate ``model`` looks like a converted HF directory, with no torch import.

    Same fast-fail rule as ``scripts/generate_samples.py::resolve_model_dir``: a bad --hf-model
    is a clear exit-1 before any expensive import and before anything is written under
    ``docs/measurements/``, never a stack trace or a half-written report.
    """
    if not model.is_dir():
        raise FileNotFoundError(
            f"no such directory: {model}. Point --hf-model at a converted HF model directory, "
            f"e.g. artifacts/hf-tt-tnt-v3/.")
    if not (model / "config.json").is_file():
        raise FileNotFoundError(
            f"{model} has no config.json -- it does not look like a converted HF model "
            f"directory. Run scripts/convert_checkpoint.py first.")
    return model


def resolve_eos_id(model_dir: Path) -> int:
    """The model's own ``eos_token_id``, read from config.json rather than assumed.

    The termination signal is a claim about a specific token id; taking it from the config
    means a model converted with a different tokenizer cannot be silently scored against id 2.
    """
    config = json.loads((model_dir / "config.json").read_text())
    eos = config.get("eos_token_id")
    if eos is None:
        raise ValueError(f"{model_dir}/config.json declares no eos_token_id, so the "
                         f"termination signal cannot be computed for it")
    if isinstance(eos, list):
        if len(eos) != 1:
            raise ValueError(f"{model_dir}/config.json declares {len(eos)} eos ids ({eos}); "
                             f"this script scores a single termination token")
        eos = eos[0]
    return int(eos)


def generate_completions(model, tokenizer, prompt_text: str, *, num_samples: int,
                         max_new_tokens: int, temperature: float, top_p: float,
                         pad_token_id: int) -> List[Tuple[str, List[int]]]:
    """``num_samples`` sampled completions of ``prompt_text``: (decoded text, generated ids).

    One batched ``generate`` call per prompt with ``num_return_sequences``; the caller seeds
    torch once before the first prompt, so a whole run is reproducible from ``--seed``.
    """
    import torch

    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens,
                             do_sample=True, temperature=temperature, top_p=top_p,
                             num_return_sequences=num_samples, pad_token_id=pad_token_id)
    results = []
    for row in out:
        generated = [int(t) for t in row[prompt_len:]]
        results.append((tokenizer.decode(generated, skip_special_tokens=True), generated))
    return results


# ---------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------


_SPDX = ("<!-- SPDX-License-Identifier: Apache-2.0 -->",
         "<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->")


def _fmt(est: Optional[Estimate], places: int = 3) -> str:
    if est is None:
        return "n/a"
    return f"{est.mean:.{places}f} ± {est.sem:.{places}f} (n={est.n})"


def _fmt_ci(est: Optional[Estimate], places: int = 3) -> str:
    if est is None:
        return "n/a"
    return f"[{est.ci95_lo:.{places}f}, {est.ci95_hi:.{places}f}]"


def render_markdown(reports: Sequence[PromptReport], *, hf_model: Path, label: str,
                    num_samples: int, max_new_tokens: int, temperature: float, top_p: float,
                    seed: int, eos_id: int, controls: Optional[Sequence[ControlRow]],
                    control_excerpt_words: Optional[int],
                    register_sources: Optional[Sequence[str]],
                    checkpoint_note: str = "",
                    prompt_set: PromptSet = PROMPT_SETS[LEGACY_PROMPT_SET]) -> str:
    lines: List[str] = list(_SPDX)
    lines += ["", f"# Behavioural quality — {label} (prompt set {prompt_set.key.upper()})", ""]
    lines.append(
        f"Model `{hf_model}`, {num_samples} sampled completions per prompt across "
        f"{len(reports)} frozen prompts ({num_samples * len(reports)} completions total), "
        f"{max_new_tokens} new tokens, temperature {temperature}, top_p {top_p}, seed {seed}, "
        f"EOS id {eos_id}. Generated by `scripts/score_behaviour.py`."
    )
    lines += ["", (
        f"**Prompt set {prompt_set.key.upper()}** "
        f"(`{prompt_set.path.name}`, {len(reports)} prompts). Numbers here are comparable ONLY "
        f"with other runs on the same set. The project's frozen sets are reported separately "
        f"and are never pooled into one score: they were written at different times by "
        f"different means and are not exchangeable, so averaging them would redefine the metric "
        f"rather than sharpen it."
    )]
    if checkpoint_note:
        lines += ["", checkpoint_note]
    lines += ["", "## Why this exists", ""]
    lines.append(
        "The goal for this model is qualitative -- prose that is oblique, observational, "
        "strangely useful -- and until now that was assessed by a human reading the 15 greedy "
        "completions `scripts/generate_samples.py` writes. Fifteen deterministic completions "
        "cannot separate a real improvement from noise and cannot be run in a loop. This is "
        "the numeric version of the same judgment: many sampled completions per prompt, five "
        "behaviours, standard errors on all of them."
    )
    lines += ["", "## Method, in brief", ""]
    lines += [
        f"- The frozen prompt set (`docs/{prompt_set.path.name}`) is **unchanged and "
        "unchangeable here**: this script reads it and never writes it, which is what keeps "
        "numbers comparable across checkpoints. Power is bought by adding a NEW frozen set with "
        "new ids and reporting it separately, never by editing an existing one.",
        "- Every signal is computed on the **completion only**, never on prompt+completion: "
        "some of the frozen prompts hand the model a register outright, and a detector that "
        "read the prompt would credit the model with the prompt's own register.",
        "- The aggregate is the mean over the **per-prompt** means, with the standard error "
        "taken **over prompts** (n=" + str(len(reports)) + "). Completions of the same prompt "
        "are not independent observations of model behaviour, so pooling them would report an "
        "interval several times too narrow -- the same \"what is the exchangeable sampling "
        "unit\" convention `probe_context_use.py` applies to windows.",
        "- Per-prompt rows report the standard error over that prompt's own "
        f"{num_samples} completions. 95% intervals are `mean ± 1.96 × SEM`.",
        "- The `" + DELIBERATE_REPETITION_PROBE + "` prompts (" + (", ".join(
            f"`{r.prompt_id}`" for r in reports
            if r.probe == DELIBERATE_REPETITION_PROBE) or "none in this set")
        + ") deliberately ASK for repetition, so the repetition aggregate is reported twice -- "
        "over all prompts and excluding the `" + DELIBERATE_REPETITION_PROBE + "` probe -- and "
        "both are labelled. Per-prompt rows always include every prompt.",
    ]
    lines += ["", "## Headline (aggregate over prompts)", ""]
    lines.append("| signal | better | mean ± SEM over prompts | 95% CI | n prompts |")
    lines.append("|---|---|---|---|---:|")
    for key, title, direction in SIGNALS:
        agg = aggregate_over_prompts(reports, key)
        places = 2 if key in ("longest_repeat", "n_tokens") else 3
        lines.append(f"| {title} | {direction or 'n/a (diagnostic)'} | {_fmt(agg, places)} | "
                     f"{_fmt_ci(agg, places)} | {agg.n if agg else 0} |")
    lines.append("")
    for key, title in (("repeat_rate", "4-gram repeat rate"),
                       ("longest_repeat", "longest repeated span (words)")):
        agg = aggregate_over_prompts(reports, key, exclude_probe=DELIBERATE_REPETITION_PROBE)
        places = 2 if key == "longest_repeat" else 3
        lines.append(f"- **{title}, excluding the `stutter` probe:** {_fmt(agg, places)}, "
                     f"95% CI {_fmt_ci(agg, places)}")
    lines.append("")

    lines += ["## Per prompt", ""]
    lines.append("Each cell is that prompt's own mean ± SEM over its "
                 f"{num_samples} completions.")
    lines.append("")
    lines.append("| prompt | probe | collapse (any) | frame collapse | terminated | "
                 "4-gram repeat | longest repeat | ts margin | nearest=ts | engagement |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in reports:
        e = r.estimates
        lines.append(
            f"| {r.prompt_id} | {r.probe} | {_fmt(e['collapse_rate'], 2)} | "
            f"{_fmt(e['frame_collapse_rate'], 2)} | "
            f"{_fmt(e['termination_rate'], 2)} | {_fmt(e['repeat_rate'], 3)} | "
            f"{_fmt(e['longest_repeat'], 1)} | {_fmt(e['tinystories_margin'], 3)} | "
            f"{_fmt(e['register_tinystories_share'], 2)} | {_fmt(e['engagement'], 2)} |")
    lines.append("")

    lines += ["## Which collapse markers fired", ""]
    total_markers: Counter = Counter()
    for r in reports:
        total_markers.update(r.marker_counts)
    total = sum(r.n_samples for r in reports)
    if total_markers:
        lines.append(f"| marker | completions firing it | of {total} |")
        lines.append("|---|---:|---:|")
        for name, _pattern in COLLAPSE_MARKERS:
            count = total_markers.get(name, 0)
            if count:
                lines.append(f"| `{name}` | {count} | {count / total:.1%} |")
    else:
        lines.append(f"No TinyStories-attractor marker fired on any of the {total} completions.")
    lines.append("")

    if register_sources:
        lines += ["## Register — which source's vocabulary each completion resembles", ""]
        lines.append("Nearest source by interpolated unigram+bigram log-likelihood per word, "
                     "over all completions.")
        lines.append("")
        nearest_total: Counter = Counter()
        for r in reports:
            nearest_total.update(r.nearest_counts)
        scored = sum(nearest_total.values())
        lines.append("| source | slice | completions nearest it | share |")
        lines.append("|---|---|---:|---:|")
        for name in register_sources:
            count = nearest_total.get(name, 0)
            slice_name = SOURCES[name].slice if name in SOURCES else "?"
            lines.append(f"| {name} | {slice_name} | {count} | "
                         f"{(count / scored if scored else 0):.1%} |")
        lines.append("")

    lines += ["## Detector controls — what these detectors do on text whose answer is known", ""]
    if controls is None:
        lines.append("Not run (the register signal was disabled, or "
                     "`artifacts/corpus/` was unavailable). Without this section the numbers "
                     "above have no known sensitivity or false-positive rate, and should be "
                     "treated as uncalibrated.")
    else:
        lines.append(
            f"Held-out tails of `artifacts/corpus/*.txt` -- text never used to fit the register "
            f"models -- cut verbatim (case and punctuation intact) into "
            f"{control_excerpt_words}-word excerpts, the median length of this run's "
            f"completions, and pushed through the same detectors on the same code path. The "
            f"held-out tail of `tinystories` **is** the failure mode, so the collapse "
            f"detector's rate there is its sensitivity; the other eight sources are not, so "
            f"its rate there is its false-positive rate. The register column is the register "
            f"signal's own 9-way accuracy at the length it is actually used at."
        )
        lines.append("")
        lines.append("| source | n excerpts | collapse fires (any) | — frame | — lexical | "
                     "register correct | top confusion | 4-gram repeat rate |")
        lines.append("|---|---:|---:|---:|---:|---:|---|---|")
        for row in controls:
            if row.n_excerpts == 0:
                lines.append(f"| {row.source} | 0 | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue
            lines.append(
                f"| {row.source} | {row.n_excerpts} | {row.collapse_rate:.2%} | "
                f"{row.frame_collapse_rate:.2%} | {row.lexical_collapse_rate:.2%} | "
                f"{row.register_accuracy:.1%} | {row.register_top_confusion} | "
                f"{_fmt(row.repeat_rate, 3)} |")
        lines.append("")
        lines.append("### Per marker, on the same held-out text")
        lines.append("")
        lines.append(
            "Every marker separately, so the frame/lexical split can be audited rather than "
            "taken on trust. `tinystories` is the sensitivity of that one marker; "
            "`worst other` is the highest rate it reaches on any of the eight sources that are "
            "**not** the failure mode -- the number that would make a marker unusable."
        )
        lines.append("")
        lines.append("| marker | kind | tinystories | worst other source | mean of the other 8 |")
        lines.append("|---|---|---:|---|---:|")
        by_source = {c.source: c for c in controls if c.n_excerpts}
        ts_row = by_source.get("tinystories")
        for name, _pattern in COLLAPSE_MARKERS:
            kind = "frame" if name in FRAME_MARKERS else "lexical"
            others = {s: c.marker_rates.get(name, 0.0)
                      for s, c in by_source.items() if s != "tinystories"}
            if ts_row is None or not others:
                lines.append(f"| `{name}` | {kind} | n/a | n/a | n/a |")
                continue
            worst = max(others, key=lambda s: others[s])
            lines.append(
                f"| `{name}` | {kind} | {ts_row.marker_rates.get(name, 0.0):.2%} | "
                f"{worst} {others[worst]:.2%} | "
                f"{sum(others.values()) / len(others):.3%} |")
        lines.append("")
        lines.append(
            "Two things this table is not. It is not a measure of the *model*: it is a measure "
            "of the detectors, on human-written corpus text. And it is not a bound on the "
            "collapse detector's behaviour on model output specifically -- a small model's "
            "TinyStories-flavoured prose is not identical to real TinyStories, so the "
            "sensitivity here is the best available estimate of the detector's recall, not a "
            "guarantee of it."
        )
    lines.append("")

    lines += ["## What to trust, and what not to", ""]
    lines += [
        "- **Termination** is the least ambiguous signal here: it is a token id, either "
        "emitted or not, with no detector in between.",
        "- **Genre collapse** is a lower bound. The controls above give its measured "
        "sensitivity on real TinyStories text; a model's true collapse rate is higher than the "
        "reported one by roughly that factor. Use it to compare models, not to state how often "
        "a model collapses.",
        "- **Read the two collapse sub-signals, not just their union.** Story-frame collapse "
        "(formulaic opening/closing/moral) and lexical-habit collapse (TinyStories vocabulary "
        "in prose that is not a fairy tale) move independently, and a model can fix one while "
        "leaving the other untouched -- which is exactly what happened between `tt-tnt-v1` and "
        "`tt-tnt-v3`. The union is retained because it is the signal that was defined first, "
        "but on its own it hides that split.",
        "- **Register** should be read together with its control column. Sources the register "
        "model cannot separate from each other (typically the narrative trio "
        "`folklore`/`gutenberg_children`/`weird`) cannot support a claim about which of them a "
        "completion resembles; `tinystories` is separated near-perfectly, which is the "
        "comparison this project actually needs.",
        "- **Repetition** is a real measurement but not a quality judgment on its own: this "
        "project's target voice includes deliberate repetition, which is why `stutter-01` and "
        "`stutter-02` exist and why the aggregate is reported both with and without them.",
        "- **Prompt engagement** is the weakest signal. It cannot tell engaging with a prompt "
        "from parroting it, and a completion that echoes the prompt verbatim scores 1.0 -- "
        "always read it next to the repetition row.",
    ]
    lines.append("")
    return "\n".join(lines)


def report_to_json(reports: Sequence[PromptReport], *, hf_model: str, label: str,
                   num_samples: int, max_new_tokens: int, temperature: float, top_p: float,
                   seed: int, eos_id: int, controls: Optional[Sequence[ControlRow]],
                   control_excerpt_words: Optional[int],
                   register_sources: Optional[Sequence[str]],
                   prompt_set: str = LEGACY_PROMPT_SET) -> dict:
    """The same numbers as the markdown, keyed for `--compare` to pair by prompt id.

    ``prompt_set`` is recorded so a comparison can REFUSE to pair runs from different sets
    rather than discovering the mismatch as an empty intersection of prompt ids -- or worse,
    silently pairing two sets that happened to share an id.
    """
    payload: dict = {
        "hf_model": hf_model,
        "label": label,
        "prompt_set": prompt_set,
        "num_samples": num_samples,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "eos_id": eos_id,
        "n_prompts": len(reports),
        "signals": [{"key": k, "title": t, "better": d} for k, t, d in SIGNALS],
        "aggregate": {
            key: (aggregate_over_prompts(reports, key).as_json()
                  if aggregate_over_prompts(reports, key) else None)
            for key, _t, _d in SIGNALS
        },
        "aggregate_excluding_stutter": {
            key: (aggregate_over_prompts(reports, key,
                                         exclude_probe=DELIBERATE_REPETITION_PROBE).as_json()
                  if aggregate_over_prompts(
                      reports, key, exclude_probe=DELIBERATE_REPETITION_PROBE) else None)
            for key in ("repeat_rate", "longest_repeat")
        },
        "per_prompt": {
            r.prompt_id: {
                "probe": r.probe,
                "n_samples": r.n_samples,
                "estimates": {k: (v.as_json() if v else None) for k, v in r.estimates.items()},
                "collapse_markers": r.marker_counts,
                "register_nearest": r.nearest_counts,
            }
            for r in reports
        },
        "register_sources": list(register_sources) if register_sources else None,
        "control_excerpt_words": control_excerpt_words,
        "detector_controls": ([
            {"source": c.source, "n_excerpts": c.n_excerpts,
             "collapse_rate": None if c.n_excerpts == 0 else c.collapse_rate,
             "frame_collapse_rate": None if c.n_excerpts == 0 else c.frame_collapse_rate,
             "lexical_collapse_rate": None if c.n_excerpts == 0 else c.lexical_collapse_rate,
             "register_accuracy": None if c.n_excerpts == 0 else c.register_accuracy,
             "register_top_confusion": c.register_top_confusion,
             "marker_rates": c.marker_rates,
             "repeat_rate": c.repeat_rate.as_json() if c.repeat_rate else None}
            for c in controls] if controls is not None else None),
    }
    return payload


# ---------------------------------------------------------------------------------------
# Comparison of two runs (paired by prompt)
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedDifference:
    """``candidate - baseline`` for one signal, averaged over the prompts both runs answered."""

    signal: str
    title: str
    better: Optional[str]
    baseline: Optional[Estimate]
    candidate: Optional[Estimate]
    difference: Optional[Estimate]
    n_prompts: int

    @property
    def min_detectable(self) -> Optional[float]:
        """Smallest difference this comparison could have called, i.e. ``1.96 × SEM``.

        The number to quote when a signal says "no change": it distinguishes "we looked and
        there is nothing there" from "we could not have seen it either way". Without it a table
        of null results is unreadable.
        """
        return None if self.difference is None else 1.96 * self.difference.sem

    @property
    def verdict(self) -> str:
        """"better"/"worse"/"no change"/"n/a" -- 'no change' when the 95% CI spans zero.

        Deliberately conservative: a difference whose interval includes zero is reported as no
        change, so a run of noise cannot be written up as an improvement. A metric that can
        only move one way is a vanity metric.
        """
        if self.difference is None or self.better is None:
            return "n/a"
        lo, hi = self.difference.ci95_lo, self.difference.ci95_hi
        if lo <= 0.0 <= hi:
            return "no change"
        improved = (self.difference.mean < 0) if self.better == "lower" else (
            self.difference.mean > 0)
        return "better" if improved else "worse"


def paired_differences(baseline: dict, candidate: dict) -> List[PairedDifference]:
    """Per-signal paired comparison over the prompts both runs answered.

    Paired, not two-sample: both runs answered the same frozen prompts, so differencing within
    a prompt removes between-prompt variance -- which is most of the variance, and is what makes
    15 prompts enough to see a real change. Raises if the two runs share no prompts, rather than
    reporting a comparison of nothing.
    """
    base_set = baseline.get("prompt_set", LEGACY_PROMPT_SET)
    cand_set = candidate.get("prompt_set", LEGACY_PROMPT_SET)
    if base_set != cand_set:
        raise ValueError(
            f"the two runs used different prompt sets ({base_set!r} vs {cand_set!r}). They are "
            f"reported separately by design: two sets written at different times by different "
            f"means are not exchangeable, and a comparison across them would be a comparison of "
            f"the sets as much as of the models. Score both models on the SAME set and compare "
            f"those.")
    shared = [p for p in baseline["per_prompt"] if p in candidate["per_prompt"]]
    if not shared:
        raise ValueError(
            f"the two runs share no prompt ids ({sorted(baseline['per_prompt'])[:3]}... vs "
            f"{sorted(candidate['per_prompt'])[:3]}...); they cannot be paired")
    out: List[PairedDifference] = []
    for key, title, direction in SIGNALS:
        base_vals, cand_vals, deltas = [], [], []
        for pid in shared:
            b = baseline["per_prompt"][pid]["estimates"].get(key)
            c = candidate["per_prompt"][pid]["estimates"].get(key)
            if b is None or c is None:
                continue
            base_vals.append(b["mean"])
            cand_vals.append(c["mean"])
            deltas.append(c["mean"] - b["mean"])
        out.append(PairedDifference(
            signal=key, title=title, better=direction,
            baseline=estimate(base_vals), candidate=estimate(cand_vals),
            difference=estimate(deltas), n_prompts=len(deltas)))
    return out


def render_comparison(baseline: dict, candidate: dict, diffs: Sequence[PairedDifference],
                      *, label: str) -> str:
    n_shared = max((d.n_prompts for d in diffs), default=0)
    set_key = baseline.get("prompt_set", LEGACY_PROMPT_SET)
    set_name = PROMPT_SETS[set_key].path.name if set_key in PROMPT_SETS else f"set {set_key}"
    lines: List[str] = list(_SPDX)
    lines += ["", f"# Behavioural quality — {label} (prompt set {set_key.upper()})", ""]
    lines.append(
        f"Baseline `{baseline['hf_model']}` ({baseline['label']}) vs candidate "
        f"`{candidate['hf_model']}` ({candidate['label']}), both scored on **prompt set "
        f"{set_key.upper()}** (`{set_name}`). "
        f"{baseline['num_samples']} and {candidate['num_samples']} completions per prompt "
        f"respectively, over {n_shared} shared prompts. "
        f"Generated by `scripts/score_behaviour.py --compare`."
    )
    lines += ["", "## Method, in brief", ""]
    lines += [
        "- **Paired by prompt.** Both runs answered the same frozen prompts, so the difference "
        "is computed within each prompt and averaged over prompts. That removes between-prompt "
        "variance, which is most of the variance here.",
        "- The `difference` column is `candidate - baseline`, mean ± SEM **over prompts**, with "
        "its 95% interval. A difference whose interval spans zero is reported as **no change** "
        "-- not as a small improvement.",
        "- `better` states which direction counts as an improvement for that signal, so a "
        "regression is reported as a regression.",
        "- `min. detectable` is `1.96 × SEM` of the difference: the smallest change this "
        "comparison had the power to call. Read every **no change** verdict against it -- a "
        "null result only means \"nothing there\" for effects larger than that number.",
        f"- **Both runs used prompt set {set_key.upper()}**, and a comparison across sets is "
        "refused rather than computed. The project's frozen sets are reported side by side, "
        "never averaged together.",
    ]
    lines += ["", "## Signal by signal", ""]
    lines.append("| signal | better | baseline | candidate | difference (cand − base) | "
                 "95% CI of difference | min. detectable | verdict |")
    lines.append("|---|---|---|---|---|---|---:|---|")
    for d in diffs:
        places = 2 if d.signal in ("longest_repeat", "n_tokens") else 3
        mdd = "n/a" if d.min_detectable is None else f"{d.min_detectable:.{places}f}"
        lines.append(
            f"| {d.title} | {d.better or 'n/a'} | {_fmt(d.baseline, places)} | "
            f"{_fmt(d.candidate, places)} | {_fmt(d.difference, places)} | "
            f"{_fmt_ci(d.difference, places)} | {mdd} | **{d.verdict}** |")
    lines.append("")
    improved = [d.title for d in diffs if d.verdict == "better"]
    regressed = [d.title for d in diffs if d.verdict == "worse"]
    flat = [d.title for d in diffs if d.verdict == "no change"]
    lines += ["## Verdict", ""]
    lines.append(f"- **better on {len(improved)}**: " + (", ".join(improved) or "none"))
    lines.append(f"- **worse on {len(regressed)}**: " + (", ".join(regressed) or "none"))
    lines.append(f"- **no measurable change on {len(flat)}**: " + (", ".join(flat) or "none"))
    lines.append("")
    wrong_way = [d.title for d in diffs
                 if d.better is not None and d.difference is not None
                 and ((d.difference.mean > 0) if d.better == "lower"
                      else (d.difference.mean < 0))]
    lines.append(
        "Signals whose point estimate moved in the **wrong** direction (significant or not): "
        + (", ".join(wrong_way) or "none")
        + ". This line exists so a run cannot be read as uniformly good when it was not; a "
        "score that only goes up is a vanity metric."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-model", type=Path, default=ROOT / "artifacts" / "hf-tt-tnt-v3",
                   help="Converted HF model directory to score (CPU only, default: "
                        "%(default)s).")
    p.add_argument("--label", type=str, default=None,
                   help="Tag for this run; output goes to docs/measurements/behaviour-LABEL"
                        "[-setX].{md,json}. Defaults to --hf-model's directory name minus 'hf-'.")
    p.add_argument("--prompt-set", type=str, default=LEGACY_PROMPT_SET,
                   choices=sorted(PROMPT_SETS),
                   help="Which frozen prompt set to score against (default: %(default)s). "
                        + "  ".join(f"{k}: {s.description}."
                                    for k, s in sorted(PROMPT_SETS.items()))
                        + "  The sets are reported SEPARATELY and never pooled: output "
                          "filenames carry the set, and --compare refuses to pair runs from "
                          "different sets.")
    p.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                   help="Completions drawn per frozen prompt (default: %(default)s). Large "
                        "enough not to be the bottleneck; power is bought with PROMPTS, which "
                        "is what --prompt-set b is for.")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeat-n", type=int, default=DEFAULT_REPEAT_N,
                   help="n-gram order for the repetition signal (default: %(default)s).")
    p.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR,
                   help="Directory of prepared per-source corpora the register models are fit "
                        "on (default: %(default)s).")
    p.add_argument("--register-words", type=int, default=DEFAULT_REGISTER_WORDS,
                   help="Words per source used to fit its register model (default: "
                        "%(default)s).")
    p.add_argument("--control-words", type=int, default=DEFAULT_CONTROL_WORDS,
                   help="Additional words per source held out for the detector controls "
                        "(default: %(default)s).")
    p.add_argument("--register-vocab", type=int, default=DEFAULT_REGISTER_VOCAB,
                   help="Per-source vocabulary cap; the shared vocabulary is the union "
                        "(default: %(default)s).")
    p.add_argument("--no-register", action="store_true",
                   help="Skip the register signal and the detector controls entirely. The "
                        "remaining signals are still reported, and the report says the "
                        "detectors are uncalibrated.")
    p.add_argument("--checkpoint-note", type=str, default="",
                   help="Freeform provenance line recorded verbatim in the markdown report.")
    p.add_argument("--compare", type=Path, nargs=2, default=None,
                   metavar=("BASELINE_JSON", "CANDIDATE_JSON"),
                   help="Compare two previous runs' JSON outputs, paired by prompt, and write "
                        "a comparison report. Loads no model.")
    p.add_argument("--out", type=Path, default=None, help="Markdown output path.")
    p.add_argument("--json-out", type=Path, default=None, help="JSON output path.")
    return p.parse_args(argv)


def default_label(hf_model: Path) -> str:
    """``artifacts/hf-tt-tnt-v3`` -> ``tt-tnt-v3``, matching probe_context_use.py's convention."""
    tag = hf_model.name
    return tag[len("hf-"):] if tag.startswith("hf-") else tag


def output_paths(label: str, prompt_set: str = LEGACY_PROMPT_SET) -> Tuple[Path, Path]:
    """Markdown and JSON paths for a run, with the prompt set baked into the filename.

    Set A keeps its historical names (``behaviour-tt-tnt-v3.md``); every other set appends its
    suffix (``behaviour-tt-tnt-v3-setB.md``). A reader who sees only a filename can therefore
    never mistake one set's numbers for the other's, and the two can never overwrite each other.
    """
    out_dir = ROOT / "docs" / "measurements"
    suffix = resolve_prompt_set(prompt_set).suffix
    return (out_dir / f"behaviour-{label}{suffix}.md",
            out_dir / f"behaviour-{label}{suffix}.json")


def _run_compare(args: argparse.Namespace) -> int:
    baseline_path, candidate_path = args.compare
    for path in (baseline_path, candidate_path):
        if not path.is_file():
            print(f"ERROR: --compare input {path} not found.", file=sys.stderr)
            return 1
    baseline = json.loads(baseline_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    try:
        diffs = paired_differences(baseline, candidate)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    label = args.label or f"{baseline['label']}-vs-{candidate['label']}"
    # The set comes from the JSONs being compared, not from --prompt-set: a comparison names the
    # set its INPUTS were produced against, so a stray flag cannot mislabel the output.
    prompt_set = baseline.get("prompt_set", LEGACY_PROMPT_SET)
    out, json_out = output_paths(label, prompt_set)
    out = args.out or out
    json_out = args.json_out or json_out

    md = render_comparison(baseline, candidate, diffs, label=label)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"wrote {out}")
    json_out.write_text(json.dumps({
        "prompt_set": prompt_set,
        "baseline": {"label": baseline["label"], "hf_model": baseline["hf_model"],
                     "json": str(baseline_path)},
        "candidate": {"label": candidate["label"], "hf_model": candidate["hf_model"],
                      "json": str(candidate_path)},
        "paired": [{"signal": d.signal, "better": d.better, "n_prompts": d.n_prompts,
                    "baseline": d.baseline.as_json() if d.baseline else None,
                    "candidate": d.candidate.as_json() if d.candidate else None,
                    "difference": d.difference.as_json() if d.difference else None,
                    "min_detectable": d.min_detectable,
                    "verdict": d.verdict} for d in diffs],
    }, indent=2), encoding="utf-8")
    print(f"wrote {json_out}")
    for d in diffs:
        print(f"  {d.title:34} {d.verdict}")
    return 0


def main() -> int:
    args = _parse_args()
    if args.compare is not None:
        return _run_compare(args)

    if args.num_samples < 2:
        print(f"ERROR: --num-samples must be >= 2 (got {args.num_samples}); a single "
              f"completion per prompt has no variance to report a standard error over.",
              file=sys.stderr)
        return 1
    try:
        model_dir = resolve_model_dir(args.hf_model)
        eos_id = resolve_eos_id(model_dir)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    label = args.label or default_label(model_dir)
    try:
        prompt_set = resolve_prompt_set(args.prompt_set)
        prompts = load_prompts(prompt_set.path)
    except (KeyError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"prompt set {prompt_set.key}: {prompt_set.path.relative_to(ROOT)}")
    print(f"{len(prompts)} frozen prompts x {args.num_samples} samples = "
          f"{len(prompts) * args.num_samples} completions")

    profile: Optional[RegisterProfile] = None
    if not args.no_register:
        print(f"fitting register models from {args.corpus_dir} ...")
        try:
            profile = build_register_profile(
                args.corpus_dir, sorted(SOURCES), train_words=args.register_words,
                control_words=args.control_words, vocab_top=args.register_vocab,
                log=print)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {model_dir} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForCausalLM.from_pretrained(str(model_dir),
                                                     torch_dtype="auto").eval()
    except (OSError, ValueError) as exc:
        print(f"ERROR: could not load model from {model_dir}: {exc}", file=sys.stderr)
        return 1

    # Seeded once, before any generation, exactly as generate_samples.py does: prompts are
    # visited in a fixed order, so one seed makes the whole run reproducible.
    torch.manual_seed(args.seed)
    pad_id = json.loads((model_dir / "config.json").read_text()).get("pad_token_id", eos_id)

    reports: List[PromptReport] = []
    all_word_counts: List[int] = []
    for prompt in prompts:
        completions = generate_completions(
            model, tokenizer, prompt["text"], num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p, pad_token_id=pad_id)
        scores = [score_completion(prompt["text"], text, ids, profile=profile,
                                   repeat_n=args.repeat_n, eos_id=eos_id)
                  for text, ids in completions]
        all_word_counts.extend(len(words_of(s.text)) for s in scores)
        report = summarise_prompt(prompt, scores)
        reports.append(report)
        e = report.estimates
        print(f"  {report.prompt_id:12} collapse={e['collapse_rate'].mean:.2f} "
              f"frame={e['frame_collapse_rate'].mean:.2f} "
              f"term={e['termination_rate'].mean:.2f} "
              f"repeat={e['repeat_rate'].mean if e['repeat_rate'] else float('nan'):.3f} "
              f"ts_margin="
              f"{e['tinystories_margin'].mean if e['tinystories_margin'] else float('nan'):+.3f}",
              flush=True)

    controls: Optional[List[ControlRow]] = None
    control_excerpt_words: Optional[int] = None
    if profile is not None:
        # Controls are run at the median completion length, so the sensitivity and
        # false-positive rates they report describe the detectors as actually used here.
        control_excerpt_words = max(1, int(np.median(all_word_counts)))
        print(f"running detector controls on held-out corpus text "
              f"({control_excerpt_words}-word excerpts) ...")
        controls = detector_controls(profile, control_excerpt_words)
        for row in controls:
            print(f"  {row.source:22} collapse-fires={row.collapse_rate:.2%} "
                  f"register-correct={row.register_accuracy:.1%} (n={row.n_excerpts})")

    md_out, json_out = output_paths(label, prompt_set.key)
    md_out = args.out or md_out
    json_out = args.json_out or json_out
    register_sources = profile.sources if profile else None

    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(render_markdown(
        reports, hf_model=args.hf_model, label=label, num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
        seed=args.seed, eos_id=eos_id, controls=controls,
        control_excerpt_words=control_excerpt_words, register_sources=register_sources,
        checkpoint_note=args.checkpoint_note, prompt_set=prompt_set), encoding="utf-8")
    print(f"wrote {md_out}")

    json_out.write_text(json.dumps(report_to_json(
        reports, hf_model=str(args.hf_model), label=label, num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
        seed=args.seed, eos_id=eos_id, controls=controls,
        control_excerpt_words=control_excerpt_words,
        register_sources=register_sources, prompt_set=prompt_set.key), indent=2),
        encoding="utf-8")
    print(f"wrote {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
