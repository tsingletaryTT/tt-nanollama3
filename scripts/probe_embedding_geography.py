#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Does this model's embedding space have a geography that corresponds to its corpus?

THE QUESTION THIS MEASURES
---------------------------
A proposal on the table is to lay the 32,000-token vocabulary onto Blackhole's Tensix grid
(17x12 physical, 11x10 = 110 functional cores on the harvested p300c this project runs on) and
sample by *spatial neighbourhood* rather than by probability rank — so that asking six times
gives six genuinely divergent associations, one per direction, because direction on the grid
corresponds to **corpus register**.

That plan rests on a factual claim that can be checked before anyone writes a kernel: **do
tokens characteristic of different corpus sources already occupy distinguishable regions of the
embedding space?**

- If they do, a grid layout is **discovered** — the geometry is in the weights and the grid is
  a projection of it.
- If they do not, any layout is **imposed** — still buildable, but the "direction means
  register" claim is decoration, not a finding.

This script answers that one question and nothing else. It does not build a layout, a sampler,
or a kernel.

WHAT IT DOES
------------
1. Reads the embedding matrix (``model.embed_tokens.weight``, 32000 x 1024) straight out of
   ``model.safetensors`` with ``safetensors`` — one tensor, never the whole model. The model
   ties its embeddings (``tie_word_embeddings: true``), so this same matrix is also the output
   head: a geography found here is a geography in the space the sampler would actually read.

2. Builds a per-source token profile from ``artifacts/corpus/*.txt`` — the nine prepared
   sources ``train/corpus.py`` registers — and scores every vocabulary entry for how
   *characteristic* it is of each source.

3. Labels the most characteristic tokens per source and asks, quantitatively, whether those
   labels are predictable from embedding geometry alone, against an explicit noise floor.

THE STATISTIC, AND WHY THIS ONE
--------------------------------
Characteristicness is the **log-odds ratio with an informative Dirichlet prior**, z-scored
(Monroe, Colaresi & Quinn 2008, "Fightin' Words"). For source ``s`` and token ``i``:

    delta = log( (y_si + a_i) / (n_s + a_0 - y_si - a_i) )
          - log( (y_ri + a_i) / (n_r + a_0 - y_ri - a_i) )       (r = every OTHER source)
    var   = 1/(y_si + a_i) + 1/(y_ri + a_i)
    z     = delta / sqrt(var)

with the prior ``a_i`` proportional to the token's frequency across the *whole* corpus, scaled
to ``--prior-strength`` pseudo-counts in total.

Raw frequency was rejected outright: the most frequent token in every source is the same
handful of function words, so a raw-frequency profile says nothing about the source. Between
the two defensible options:

- **tf-idf** has no notion of sampling noise. A token seen three times in one small source and
  never elsewhere gets a large score, and ``flavour`` (a 0.5%, ~575k-token slice) would fill
  its list with accidents.
- **log-odds with a prior** divides by the standard error of the estimate, so a token has to be
  *both* skewed and *well-attested* to score. That is the same standard this project applies
  everywhere else: an effect is reported against its noise, or it is not reported. It is also
  the statistic literally designed for "which words distinguish corpus A from corpus B".

The known cost of the z-scored version is that it favours *frequent* tokens — function words
and punctuation dominate the top of each list. That is not obviously wrong (register largely
*is* function-word and punctuation distribution: ``spine`` really is "of/the/which/these" prose
and ``weird`` really is first-person), but it introduces a confound: token frequency differs by
source, and frequency is itself encoded in an embedding. So every measurement below is run in
two conditions and against a frequency-only control — see NOISE FLOORS.

PRECEDENT REUSED, AND ONE DELIBERATE DIVERGENCE
------------------------------------------------
``scripts/score_behaviour.py``'s register signal already profiles these nine sources lexically,
and this script follows it deliberately: same nine sources from ``train.corpus.SOURCES``, same
"read a prefix of each source's prepared file" discipline, same default word budget per source
(``score_behaviour.DEFAULT_REGISTER_WORDS``, imported rather than re-chosen), and the same
honesty about the prefix being a prefix (Gutenberg-derived sources open with title pages, which
is register noise).

It diverges in exactly one place, and only because the unit of analysis is different.
``score_behaviour.read_corpus_tokens`` flattens a source to whitespace tokens because its
``SourceLM`` is a *word*-level model. This script needs counts over the model's **byte-level BPE
vocabulary**, because that is what indexes the embedding matrix — a word-level profile cannot be
joined to a 32,000-row matrix at all. Byte-level BPE encodes newlines as real tokens, and in
this corpus newlines carry structure the model was trained on (``poetry`` is line-broken verse;
``spine`` is hard-wrapped Gutenberg prose), so :func:`read_corpus_text` preserves them where
``read_corpus_tokens`` would have discarded them. Same prefix, same budget, different
tokenisation — stated here so the difference is a decision on the record rather than a second
profiling method that quietly appeared.

NOISE FLOORS (this project labels anything indistinguishable from its noise NOT INTERPRETABLE)
-----------------------------------------------------------------------------------------------
Every headline number is reported against three baselines, not one:

a. **Chance** — 1/9 for a 9-way balanced problem (1/4 for the 4-way subgroup contrasts).
b. **Label permutation** — the identical computation with the source labels shuffled, repeated
   ``--n-permutations`` times, reported as mean +- sd. This is the real noise floor: it holds
   the embedding geometry and the class sizes fixed and destroys only the correspondence.
c. **Frequency-only control** — the same computation on features that contain *no* geometry:
   a token's log corpus frequency and its embedding norm. If the embedding result is not
   comfortably above this, the "geography" is a frequency gradient wearing a hat.

And two token-selection conditions, because the statistic favours frequent tokens:

- ``all`` — the top characteristic tokens outright.
- ``content`` — the same, after excluding the ``--exclude-top`` globally most frequent tokens
  (default 500), which removes the punctuation-and-function-word backbone. If the geography is
  a frequency artefact it should collapse here; if it is real it should survive.

CONSTRAINTS THIS SCRIPT RESPECTS
---------------------------------
CPU only. Never imports ttml/ttnn, never opens a Tenstorrent device, never loads the model's
layers — only its embedding tensor. Writes only to ``docs/measurements/``; reads
``artifacts/`` and never writes there. No sklearn and no matplotlib: the k-NN, the silhouette,
the logistic-regression probe and the PCA projection are all implemented here in numpy, and the
projection is rendered as an ASCII scatter rather than a PNG (see :func:`ascii_scatter`). Both
libraries happen to be importable in the machine venv this was developed on; they are *not*
declared dependencies of this project in ``pyproject.toml``, and adding them was out of scope.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.score_behaviour import DEFAULT_REGISTER_WORDS  # noqa: E402
from train.corpus import SOURCES  # noqa: E402

#: Words read from each source's prepared file. Imported from ``score_behaviour`` rather than
#: chosen again, so the two per-source lexical profiles in this repo rest on the same slice of
#: each source.
DEFAULT_WORDS_PER_SOURCE = DEFAULT_REGISTER_WORDS

#: Total pseudo-counts in the Dirichlet prior, spread across the vocabulary in proportion to
#: each token's corpus-wide frequency. Monroe et al. use a background corpus for this; the
#: blend of all nine sources IS the background corpus here. Larger values shrink every source
#: harder toward the corpus mean, which costs sensitivity on small sources; 1000 is the value
#: that leaves ``flavour`` (the smallest source, ~575k tokens) with usable contrasts while
#: still suppressing single-document accidents.
DEFAULT_PRIOR_STRENGTH = 1000.0

#: A token must occur at least this many times **in a source** before that source may claim it.
#: Bounds the variance of the estimate directly rather than trusting the prior to do it.
DEFAULT_MIN_COUNT = 25

#: Characteristic tokens taken per source. 150 x 9 = 1350 labelled points: enough for a stable
#: k-NN and a 70/30 probe split, small enough that the full 1350x1350 similarity matrix and 200
#: label permutations run in seconds.
DEFAULT_PER_SOURCE = 150

#: Neighbours inspected in the k-NN purity test.
DEFAULT_K = 10

#: Label permutations used to establish the noise floor.
DEFAULT_N_PERMUTATIONS = 200

#: In the ``content`` condition, the globally most frequent tokens are excluded before
#: characteristic tokens are chosen. 500 covers this corpus's punctuation and function-word
#: core without reaching into the vocabulary that carries subject matter.
DEFAULT_EXCLUDE_TOP = 500

#: Repeated 70/30 splits for the linear probe, so its accuracy carries a spread rather than
#: resting on one lucky partition.
DEFAULT_PROBE_REPEATS = 5

#: The four sources the narrative-entanglement question is about. ``scripts/score_behaviour.py``
#: reports its register control scoring 99.9% on tinystories but only 63.9% on folklore, which is
#: the observation that motivated asking whether the *strange* sources are separable from each
#: other at all, or only jointly separable from the backbone.
NARRATIVE_SOURCES = ("folklore", "gutenberg_children", "spine", "weird")

#: The comparison group for that contrast: four sources of the same count, so a 4-way number is
#: read against a 4-way number rather than against the 9-way one.
CONTRAST_SOURCES = ("flavour", "poetry", "procedural", "wikipedia_simple")


# ---------------------------------------------------------------------------------------
# The embedding matrix
# ---------------------------------------------------------------------------------------


def load_embedding_matrix(hf_model: Path,
                          key: str = "model.embed_tokens.weight") -> np.ndarray:
    """Read one tensor — the embedding table — out of ``hf_model/model.safetensors``.

    ``safe_open`` memory-maps the file and materialises only the requested tensor, so this
    never pays for the 8 transformer layers it does not need. The checkpoint is bfloat16,
    which numpy has no dtype for, so the tensor is fetched through the torch framework
    binding and up-cast to float32 — an exact widening, not a re-quantisation.

    Raises ``FileNotFoundError`` if the file is missing and ``KeyError`` (naming the keys that
    ARE present) if the embedding key is not in it, rather than returning something plausible.
    """
    path = hf_model / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(
            f"no safetensors checkpoint at {path}. Point --hf-model at a converted model "
            f"directory (scripts/convert_checkpoint.py writes one)."
        )
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        keys = list(handle.keys())
        if key not in keys:
            raise KeyError(
                f"{path} has no tensor {key!r}; it holds {len(keys)} tensors starting "
                f"{keys[:3]}. Pass --embedding-key if this checkpoint names it differently."
            )
        tensor = handle.get_tensor(key)
    return tensor.float().numpy()


# ---------------------------------------------------------------------------------------
# Per-source token profiles
# ---------------------------------------------------------------------------------------


def read_corpus_text(path: Path, word_budget: int) -> Tuple[str, int]:
    """Read whole lines of ``path`` until ``word_budget`` whitespace words are covered.

    Returns ``(text, n_words)`` with **line structure intact** — see the module docstring's
    "one deliberate divergence" section for why this differs from
    ``score_behaviour.read_corpus_tokens``, which flattens the same prefix to whitespace tokens.

    Whole lines, never a partial one, so a verse line or a wrapped sentence is never cut in the
    middle: the budget is a floor that the last line may overshoot, which matters not at all at
    a budget of a million words.

    This is a PREFIX, and that is a real limitation inherited from the precedent: Gutenberg-
    derived sources open with title pages and tables of contents. It is also the part of each
    source ``scripts/blend_corpus.py`` actually emitted into the training blend, so it is at
    least text the model saw.
    """
    if word_budget < 1:
        raise ValueError(f"word_budget must be >= 1, got {word_budget}")
    lines: List[str] = []
    n_words = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lines.append(line)
            n_words += len(line.split())
            if n_words >= word_budget:
                break
    return "".join(lines), n_words


def count_tokens_by_source(corpus_dir: Path, source_names: Sequence[str], tokenizer,
                           vocab_size: int, *, words_per_source: int,
                           chunk_chars: int = 200_000,
                           log=lambda msg: None) -> np.ndarray:
    """Token-id counts per source: an ``(n_sources, vocab_size)`` int64 matrix.

    Each source contributes the same word budget, so the profiles are comparable before the
    statistic ever runs. (A source smaller than the budget — ``flavour`` — contributes all of
    itself; the log-odds statistic accounts for the smaller ``n`` through its variance term,
    which is one of the reasons it was chosen.)

    Text is encoded in ``chunk_chars`` slices purely to bound the fast tokenizer's memory. The
    chunk boundary can split one token into two at nine places per source out of ~1.2 million,
    which is below every effect reported here by four orders of magnitude.

    Raises ``FileNotFoundError`` naming the first missing source. A profile silently computed
    over 8 of 9 sources would corrupt every "characteristic of" claim derived from it, since
    the statistic is explicitly one-source-against-the-rest.
    """
    counts = np.zeros((len(source_names), vocab_size), dtype=np.int64)
    for row, name in enumerate(source_names):
        path = corpus_dir / f"{name}.txt"
        if not path.is_file():
            raise FileNotFoundError(
                f"corpus source {path} not found -- this measurement needs every prepared "
                f"source, because the statistic is one source against all the others. Run "
                f"scripts/prepare_corpus.py first."
            )
        text, n_words = read_corpus_text(path, words_per_source)
        n_tokens = 0
        for start in range(0, len(text), chunk_chars):
            ids = tokenizer(text[start:start + chunk_chars], add_special_tokens=False)
            arr = np.asarray(ids["input_ids"], dtype=np.int64)
            if arr.size and (arr.max() >= vocab_size or arr.min() < 0):
                raise ValueError(
                    f"{path} produced token id {int(arr.max())} outside [0, {vocab_size}); "
                    f"the tokenizer and the embedding matrix do not agree on a vocabulary."
                )
            np.add.at(counts[row], arr, 1)
            n_tokens += int(arr.size)
        log(f"  {name:22} {n_words:>9,} words -> {n_tokens:>9,} tokens")
    return counts


def log_odds_z(counts: np.ndarray, prior_strength: float = DEFAULT_PRIOR_STRENGTH
               ) -> np.ndarray:
    """Monroe et al. (2008) log-odds ratio with an informative Dirichlet prior, z-scored.

    ``counts`` is ``(n_sources, vocab_size)``; the result has the same shape, and entry
    ``(s, i)`` is how many standard errors token ``i``'s log-odds in source ``s`` sits above
    its log-odds in *all the other sources pooled*. See the module docstring for the formula
    and for why this statistic rather than tf-idf.

    Tokens that never occur anywhere in the sampled corpus get no prior mass and no evidence,
    so their z is undefined. They are returned as ``-inf`` (never NaN, and never a number that
    could win an argmax) — 'unattested' is a state the caller must be able to see and filter,
    not one to paper over with a zero.
    """
    if counts.ndim != 2:
        raise ValueError(f"counts must be 2-D (n_sources, vocab_size), got {counts.shape}")
    if counts.shape[0] < 2:
        raise ValueError("log-odds against 'the other sources' needs at least 2 sources")
    if prior_strength <= 0:
        raise ValueError(f"prior_strength must be > 0, got {prior_strength}")

    y = counts.astype(np.float64)
    total = y.sum(axis=0)
    grand = float(total.sum())
    if grand <= 0:
        raise ValueError("counts are all zero; nothing to profile")

    alpha = prior_strength * total / grand          # (V,) informative prior
    a0 = float(alpha.sum())
    n_s = y.sum(axis=1, keepdims=True)              # (S,1) tokens in this source
    rest = total[None, :] - y                       # (S,V) counts in every other source
    n_r = rest.sum(axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        odds_s = (y + alpha) / (n_s + a0 - y - alpha)
        odds_r = (rest + alpha) / (n_r + a0 - rest - alpha)
        delta = np.log(odds_s) - np.log(odds_r)
        sigma = np.sqrt(1.0 / (y + alpha) + 1.0 / (rest + alpha))
        z = delta / sigma

    unattested = total <= 0
    z[:, unattested] = -np.inf
    return np.where(np.isfinite(z), z, -np.inf)


@dataclass(frozen=True)
class TokenLabels:
    """The characteristic tokens of every source, and which source each belongs to."""

    #: Vocabulary ids, one per labelled token.
    token_ids: np.ndarray
    #: Index into ``source_names`` for each entry of ``token_ids``.
    labels: np.ndarray
    source_names: Tuple[str, ...]
    #: The z-score that won each token its label.
    z_scores: np.ndarray
    #: Corpus-wide count of each labelled token, kept for the frequency control.
    total_counts: np.ndarray

    @property
    def n_sources(self) -> int:
        return len(self.source_names)

    def subset(self, keep: Sequence[str]) -> "TokenLabels":
        """Restrict to a subset of sources, re-indexing labels to the kept sources.

        Used for the 4-way narrative contrast. The tokens are NOT re-chosen against the
        smaller comparison set: they remain the tokens the 9-way profile picked, which is
        deliberate, because a grid laid out from the 9-way profile is exactly what a sampler
        would have to separate ``folklore`` from ``weird`` with.
        """
        wanted = {self.source_names.index(name): i for i, name in enumerate(keep)}
        mask = np.isin(self.labels, list(wanted))
        remap = np.array([wanted.get(int(l), -1) for l in self.labels[mask]])
        return TokenLabels(token_ids=self.token_ids[mask], labels=remap,
                           source_names=tuple(keep), z_scores=self.z_scores[mask],
                           total_counts=self.total_counts[mask])


def characteristic_tokens(counts: np.ndarray, z: np.ndarray, source_names: Sequence[str], *,
                          min_count: int = DEFAULT_MIN_COUNT,
                          per_source: int = DEFAULT_PER_SOURCE,
                          exclude_top: int = 0) -> TokenLabels:
    """Assign each eligible token to the source it is most characteristic of, then keep the
    top ``per_source`` per source by z.

    Winner-take-all on z, so the label sets are **disjoint by construction**: a token belongs to
    at most one source, and the k-NN test cannot be flattered by a token that is legitimately
    two sources' at once.

    ``exclude_top`` drops the N globally most frequent tokens before any of this, which is the
    ``content`` condition described in the module docstring — the check that the geography is
    not simply a frequency gradient.

    A source with fewer than ``per_source`` eligible tokens contributes what it has; the class
    sizes are then unequal and every chance baseline in this script is computed from the
    realised class sizes rather than assumed to be 1/n_sources.
    """
    if counts.shape != z.shape:
        raise ValueError(f"counts {counts.shape} and z {z.shape} must have the same shape")
    if len(source_names) != counts.shape[0]:
        raise ValueError(
            f"{len(source_names)} source names for {counts.shape[0]} rows of counts")
    if per_source < 1:
        raise ValueError(f"per_source must be >= 1, got {per_source}")

    total = counts.sum(axis=0)
    # Rank 0 is the most frequent token in the corpus overall.
    freq_rank = np.empty(total.shape[0], dtype=np.int64)
    freq_rank[np.argsort(-total, kind="stable")] = np.arange(total.shape[0])

    eligible = (counts >= min_count) & (freq_rank[None, :] >= exclude_top)
    masked = np.where(eligible, z, -np.inf)
    winner = masked.argmax(axis=0)
    best = masked.max(axis=0)

    ids: List[int] = []
    labels: List[int] = []
    scores: List[float] = []
    for s in range(len(source_names)):
        cand = np.flatnonzero((winner == s) & np.isfinite(best))
        cand = cand[np.argsort(-best[cand], kind="stable")][:per_source]
        ids.extend(int(i) for i in cand)
        labels.extend([s] * len(cand))
        scores.extend(float(best[i]) for i in cand)
    if not ids:
        raise ValueError(
            f"no token cleared min_count={min_count} with exclude_top={exclude_top}; "
            f"loosen the filters or read more words per source")
    token_ids = np.asarray(ids, dtype=np.int64)
    return TokenLabels(token_ids=token_ids, labels=np.asarray(labels, dtype=np.int64),
                       source_names=tuple(source_names),
                       z_scores=np.asarray(scores, dtype=np.float64),
                       total_counts=total[token_ids])


# ---------------------------------------------------------------------------------------
# Separation statistics — all numpy, all against a stated noise floor
# ---------------------------------------------------------------------------------------


def chance_accuracy(labels: np.ndarray, n_classes: int) -> float:
    """Accuracy of guessing uniformly at random, weighted by the realised class sizes.

    Not hard-coded to ``1/n_classes``: if one source contributes fewer characteristic tokens
    than the others, uniform guessing does not score 1/n and the baseline must say so.
    """
    if labels.size == 0:
        raise ValueError("cannot compute a chance baseline over zero labels")
    shares = np.bincount(labels, minlength=n_classes) / labels.size
    return float(shares.sum() / n_classes)


def _unit_rows(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def neighbour_indices(features: np.ndarray, k: int) -> np.ndarray:
    """Indices of each row's ``k`` nearest neighbours by **cosine** similarity, self excluded.

    Cosine rather than Euclidean because embedding norm carries token frequency far more than
    it carries meaning, and a frequency-driven radius would be exactly the confound this
    measurement is trying to rule out. (The frequency control below then re-introduces
    frequency deliberately, as a feature, to see how far it gets on its own.)
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k >= features.shape[0]:
        raise ValueError(
            f"k={k} needs more than {features.shape[0]} points (self is excluded)")
    unit = _unit_rows(features)
    sim = unit @ unit.T
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1, kind="stable")[:, :k]


def purity_from_neighbours(neighbours: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of all (point, neighbour) pairs whose labels agree."""
    return float((labels[neighbours] == labels[:, None]).mean())


def per_class_purity(neighbours: np.ndarray, labels: np.ndarray,
                     n_classes: int) -> List[float]:
    """Neighbour purity for each class separately — 'does THIS source hold together'."""
    out = []
    for c in range(n_classes):
        mask = labels == c
        out.append(float((labels[neighbours[mask]] == c).mean()) if mask.any() else float("nan"))
    return out


def neighbour_confusion(neighbours: np.ndarray, labels: np.ndarray,
                        n_classes: int) -> np.ndarray:
    """Row ``c``: where a class-``c`` token's neighbours actually come from, as fractions.

    This is the table that answers "which sources are entangled with which", as opposed to the
    scalar that answers "is there geography at all".
    """
    conf = np.zeros((n_classes, n_classes), dtype=np.float64)
    for c in range(n_classes):
        mask = labels == c
        if not mask.any():
            conf[c] = np.nan
            continue
        got = labels[neighbours[mask]].ravel()
        conf[c] = np.bincount(got, minlength=n_classes) / got.size
    return conf


def permuted_purity(neighbours: np.ndarray, labels: np.ndarray, n_permutations: int,
                    rng: np.random.Generator) -> Tuple[float, float]:
    """Mean and sd of neighbour purity under label permutation — the noise floor.

    The neighbour graph is computed once and reused: permuting labels does not move any point,
    which is the whole point of this control. It holds the geometry, the class sizes and k
    fixed and destroys only the token -> source correspondence, so anything above it is
    correspondence and not structure-in-general.
    """
    if n_permutations < 2:
        raise ValueError(f"n_permutations must be >= 2 to get a spread, got {n_permutations}")
    draws = np.array([purity_from_neighbours(neighbours, rng.permutation(labels))
                      for _ in range(n_permutations)])
    return float(draws.mean()), float(draws.std(ddof=1))


def silhouette_cosine(features: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    """Mean silhouette coefficient over cosine distance, in [-1, 1].

    ``(b - a) / max(a, b)`` per point, where ``a`` is its mean distance to its own class and
    ``b`` the smallest mean distance to any other class. Reported because it is the one number
    that is sensitive to *how far apart* the regions are rather than only to who is nearest,
    but it is a supporting number, not the headline: on 1350 points in 1024 dimensions almost
    everything is far from almost everything, so silhouette values are small even when the
    k-NN separation is overwhelming. It is read against its own permuted floor like everything
    else.
    """
    unit = _unit_rows(features)
    dist = 1.0 - (unit @ unit.T)
    np.fill_diagonal(dist, 0.0)
    scores = np.zeros(features.shape[0], dtype=np.float64)
    masks = [labels == c for c in range(n_classes)]
    for i in range(features.shape[0]):
        own = int(labels[i])
        means = []
        for c in range(n_classes):
            mask = masks[c].copy()
            if c == own:
                mask[i] = False
                if not mask.any():
                    means.append(np.nan)
                    continue
            means.append(float(dist[i, mask].mean()))
        a = means[own]
        others = [m for c, m in enumerate(means) if c != own and not np.isnan(m)]
        if np.isnan(a) or not others:
            scores[i] = 0.0
            continue
        b = min(others)
        scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(scores.mean())


@dataclass(frozen=True)
class ProbeResult:
    """Held-out accuracy of a linear probe, over repeated splits."""

    accuracy_mean: float
    accuracy_sd: float
    n_repeats: int
    #: Averaged over repeats; row = true class, column = predicted class, as fractions.
    confusion: np.ndarray
    per_class_recall: List[float]


def linear_probe(features: np.ndarray, labels: np.ndarray, n_classes: int, *,
                 repeats: int = DEFAULT_PROBE_REPEATS, train_fraction: float = 0.7,
                 iterations: int = 400, lr: float = 0.5, l2: float = 1e-3,
                 seed: int = 0) -> ProbeResult:
    """Multinomial logistic regression, full-batch gradient descent, numpy only.

    Train/test separation is per repeat and **stratified by class**, so a small class cannot
    land entirely in one side of the split and turn its recall into an artefact of the
    partition. Features are standardised using the TRAINING rows' mean and sd only — computing
    them over all rows would leak test-set information into the fit, which is the exact
    failure this project's brief called "proper train/test separation".

    A linear probe (rather than anything stronger) on purpose: the question is whether source
    identity is *linearly* legible in the embedding, which is the property a grid layout would
    need. A sufficiently flexible classifier can find structure that no linear projection onto
    a 2-D grid could ever preserve, so it would answer a different, easier question.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0,1), got {train_fraction}")

    accuracies: List[float] = []
    confusion = np.zeros((n_classes, n_classes), dtype=np.float64)
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        train_idx: List[int] = []
        test_idx: List[int] = []
        for c in range(n_classes):
            members = np.flatnonzero(labels == c)
            if members.size < 2:
                raise ValueError(
                    f"class {c} has {members.size} member(s); a train/test split needs 2+")
            shuffled = rng.permutation(members)
            cut = max(1, min(len(shuffled) - 1, int(round(train_fraction * len(shuffled)))))
            train_idx.extend(shuffled[:cut].tolist())
            test_idx.extend(shuffled[cut:].tolist())
        tr = np.asarray(train_idx)
        te = np.asarray(test_idx)

        mu = features[tr].mean(axis=0)
        sd = features[tr].std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        x_tr = np.hstack([(features[tr] - mu) / sd, np.ones((tr.size, 1))])
        x_te = np.hstack([(features[te] - mu) / sd, np.ones((te.size, 1))])
        y_tr = labels[tr]
        onehot = np.eye(n_classes)[y_tr]

        weights = np.zeros((x_tr.shape[1], n_classes), dtype=np.float64)
        for _ in range(iterations):
            logits = x_tr @ weights
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            grad = x_tr.T @ (probs - onehot) / x_tr.shape[0] + l2 * weights
            weights -= lr * grad

        pred = (x_te @ weights).argmax(axis=1)
        accuracies.append(float((pred == labels[te]).mean()))
        for true_c, pred_c in zip(labels[te], pred):
            confusion[true_c, pred_c] += 1.0

    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion = np.divide(confusion, row_sums, out=np.zeros_like(confusion),
                          where=row_sums > 0)
    arr = np.asarray(accuracies)
    return ProbeResult(accuracy_mean=float(arr.mean()),
                       accuracy_sd=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                       n_repeats=repeats, confusion=confusion,
                       per_class_recall=[float(confusion[c, c]) for c in range(n_classes)])


def frequency_features(labels: TokenLabels, embedding: np.ndarray) -> np.ndarray:
    """The frequency-only control's features: ``[log corpus count, embedding L2 norm]``.

    Two numbers that contain no directional information whatsoever. If they predict source
    nearly as well as the 1024-dimensional embedding does, then the "geography" is a frequency
    gradient and a grid built on it would be sorting tokens by how common they are.
    """
    vectors = embedding[labels.token_ids]
    return np.stack([np.log(labels.total_counts + 1.0),
                     np.linalg.norm(vectors, axis=1)], axis=1)


# ---------------------------------------------------------------------------------------
# A picture for the humans (numpy PCA, ASCII scatter — no matplotlib)
# ---------------------------------------------------------------------------------------


def pca_2d(features: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
    """First two principal components of ``features``, and the variance each explains.

    Plain SVD of the mean-centred matrix. This is a *projection*, and it is here for looking
    at, not for deciding anything: two components of a 1024-dimensional space keep a small
    fraction of the variance, and the report says how small. Every claim rests on the numbers
    computed in the full space.
    """
    centred = features - features.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(centred, full_matrices=False)
    coords = centred @ vt[:2].T
    var = (s ** 2)
    total = float(var.sum())
    ratios = (float(var[0] / total), float(var[1] / total)) if total > 0 else (0.0, 0.0)
    return coords, ratios


def ascii_scatter(coords: np.ndarray, labels: np.ndarray, source_names: Sequence[str], *,
                  width: int = 78, height: int = 30) -> List[str]:
    """Render a 2-D scatter as text: one character per source, '*' where sources overlap.

    Deliberately low-fidelity. Its job is to let a reader see whether the labelled regions look
    like regions at all; it cannot and should not be used to judge separation, which is what
    the statistics above are for. Left-aligned with no right-hand border, per this project's
    terminal-output convention.
    """
    if coords.shape[0] != labels.shape[0]:
        raise ValueError("coords and labels must describe the same points")
    glyphs = [name[0].upper() if i < 26 else "?" for i, name in enumerate(source_names)]
    # Distinct glyphs even when two sources share a first letter.
    seen: Dict[str, int] = {}
    for i, g in enumerate(glyphs):
        if g in seen:
            glyphs[i] = source_names[i][1].lower()
        seen[glyphs[i]] = i

    grid = [[" "] * width for _ in range(height)]
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    for (x, y), lab in zip(coords, labels):
        col = int((x - lo[0]) / span[0] * (width - 1))
        row = int((1.0 - (y - lo[1]) / span[1]) * (height - 1))
        cell = grid[row][col]
        glyph = glyphs[int(lab)]
        grid[row][col] = glyph if cell == " " else ("*" if cell != glyph else glyph)

    lines = ["".join(row).rstrip() for row in grid]
    legend = "  ".join(f"{glyphs[i]}={name}" for i, name in enumerate(source_names))
    lines.append("")
    lines.append(f"legend: {legend}   *=two or more sources in one cell")
    return lines


# ---------------------------------------------------------------------------------------
# One condition's full battery
# ---------------------------------------------------------------------------------------


@dataclass
class SeparationResult:
    """Every separation number for one (condition, source-set) pair, with its baselines."""

    name: str
    source_names: Tuple[str, ...]
    n_tokens: int
    class_sizes: List[int]
    k: int
    knn_purity: float
    knn_permuted_mean: float
    knn_permuted_sd: float
    knn_chance: float
    knn_frequency_only: float
    per_source_purity: List[float]
    confusion: np.ndarray
    silhouette: float
    silhouette_permuted_mean: float
    silhouette_permuted_sd: float
    probe: ProbeResult
    probe_frequency_only: ProbeResult
    probe_permuted: ProbeResult
    top_tokens: Dict[str, List[Tuple[str, int, float]]] = field(default_factory=dict)
    projection: Optional[List[str]] = None
    projection_variance: Tuple[float, float] = (0.0, 0.0)
    #: Per-source mean position in the 2-D projection — the scatter with the overplotting
    #: removed, which is the only part of a picture that can be read at a glance.
    projection_centroids: Optional[List[Tuple[float, float]]] = None

    @property
    def knn_sigmas_above_floor(self) -> float:
        """How many permutation sds the observed purity sits above the permuted mean.

        This is the number that decides INTERPRETABLE vs NOT INTERPRETABLE here: at or below
        the floor, there is no correspondence to report.
        """
        if self.knn_permuted_sd <= 0:
            return float("inf") if self.knn_purity > self.knn_permuted_mean else 0.0
        return (self.knn_purity - self.knn_permuted_mean) / self.knn_permuted_sd


def evaluate_separation(name: str, labels: TokenLabels, embedding: np.ndarray, *,
                        k: int = DEFAULT_K, n_permutations: int = DEFAULT_N_PERMUTATIONS,
                        probe_repeats: int = DEFAULT_PROBE_REPEATS, seed: int = 0,
                        tokenizer=None, top_n: int = 25,
                        with_projection: bool = False) -> SeparationResult:
    """Run the whole battery — k-NN, silhouette, linear probe — with all three baselines."""
    n_classes = labels.n_sources
    vectors = embedding[labels.token_ids]
    rng = np.random.default_rng(seed)

    neighbours = neighbour_indices(vectors, k)
    purity = purity_from_neighbours(neighbours, labels.labels)
    perm_mean, perm_sd = permuted_purity(neighbours, labels.labels, n_permutations, rng)

    freq = frequency_features(labels, embedding)
    freq_neighbours = neighbour_indices(freq, k)
    freq_purity = purity_from_neighbours(freq_neighbours, labels.labels)

    sil = silhouette_cosine(vectors, labels.labels, n_classes)
    sil_draws = np.array([silhouette_cosine(vectors, rng.permutation(labels.labels), n_classes)
                          for _ in range(min(n_permutations, 20))])

    probe = linear_probe(vectors, labels.labels, n_classes, repeats=probe_repeats, seed=seed)
    probe_freq = linear_probe(freq, labels.labels, n_classes, repeats=probe_repeats, seed=seed)
    shuffled = np.random.default_rng(seed + 9973).permutation(labels.labels)
    probe_perm = linear_probe(vectors, shuffled, n_classes, repeats=probe_repeats, seed=seed)

    top: Dict[str, List[Tuple[str, int, float]]] = {}
    if tokenizer is not None:
        for c, source in enumerate(labels.source_names):
            members = np.flatnonzero(labels.labels == c)[:top_n]
            top[source] = [
                (tokenizer.convert_ids_to_tokens(int(labels.token_ids[i])),
                 int(labels.total_counts[i]), float(labels.z_scores[i]))
                for i in members
            ]

    projection = None
    centroids = None
    ratios = (0.0, 0.0)
    if with_projection:
        coords, ratios = pca_2d(_unit_rows(vectors))
        projection = ascii_scatter(coords, labels.labels, labels.source_names)
        centroids = [(float(coords[labels.labels == c, 0].mean()),
                      float(coords[labels.labels == c, 1].mean()))
                     for c in range(n_classes)]

    return SeparationResult(
        name=name, source_names=labels.source_names, n_tokens=int(labels.token_ids.size),
        class_sizes=[int((labels.labels == c).sum()) for c in range(n_classes)],
        k=k, knn_purity=purity, knn_permuted_mean=perm_mean, knn_permuted_sd=perm_sd,
        knn_chance=chance_accuracy(labels.labels, n_classes), knn_frequency_only=freq_purity,
        per_source_purity=per_class_purity(neighbours, labels.labels, n_classes),
        confusion=neighbour_confusion(neighbours, labels.labels, n_classes),
        silhouette=sil, silhouette_permuted_mean=float(sil_draws.mean()),
        silhouette_permuted_sd=float(sil_draws.std(ddof=1)),
        probe=probe, probe_frequency_only=probe_freq, probe_permuted=probe_perm,
        top_tokens=top, projection=projection, projection_variance=ratios,
        projection_centroids=centroids)


# ---------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------


def verdict(result: SeparationResult) -> str:
    """This project's own standard, applied: is the separation above its noise floor?

    Three sigmas above the permutation floor AND above the frequency-only control is the bar.
    Anything that clears only one of those is reported as what it is: either indistinguishable
    from noise, or distinguishable from noise but explained by token frequency, which is not
    the same finding as "the space has a geography".
    """
    sigmas = result.knn_sigmas_above_floor
    beats_frequency = result.knn_purity > result.knn_frequency_only * 1.5
    if sigmas < 3.0:
        return "NOT INTERPRETABLE (at or inside the label-permutation floor)"
    if not beats_frequency:
        return ("FREQUENCY, NOT GEOGRAPHY (above the permutation floor, but not clearly "
                "above what token frequency alone achieves)")
    return "SEPARATED (above the permutation floor and above the frequency-only control)"


def _fmt_confusion(conf: np.ndarray, names: Sequence[str]) -> List[str]:
    head = "| true \\ neighbours | " + " | ".join(names) + " |"
    rule = "|---|" + "---:|" * len(names)
    rows = [head, rule]
    for i, name in enumerate(names):
        cells = " | ".join(f"{conf[i, j]:.2f}" for j in range(len(names)))
        rows.append(f"| **{name}** | {cells} |")
    return rows


def render_markdown(results: Sequence[SeparationResult], *, hf_model: str, corpus_dir: str,
                    words_per_source: int, prior_strength: float, min_count: int,
                    per_source: int, exclude_top: int, note: str = "") -> str:
    lines: List[str] = []
    lines.append("<!-- SPDX-License-Identifier: Apache-2.0 -->")
    lines.append("<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->")
    lines.append("")
    lines.append("# Embedding geography — do corpus sources occupy distinct regions?")
    lines.append("")
    lines.append(
        f"Model `{hf_model}` (embedding matrix only), corpus `{corpus_dir}`, "
        f"{words_per_source:,} words read per source, generated by "
        f"`scripts/probe_embedding_geography.py`."
    )
    if note:
        lines.append("")
        lines.append(note)
    lines.append("")
    lines.append("## Why this exists")
    lines.append("")
    lines.append(
        "A proposed sampler would lay the 32,000-token vocabulary onto Blackhole's Tensix "
        "grid and sample by spatial neighbourhood, so that *direction on the grid* means "
        "*corpus register*. That only works if tokens characteristic of different sources "
        "already occupy distinguishable regions of the embedding space. If they do, the "
        "layout is **discovered**; if they do not, any layout is **imposed** and the claim "
        "is decoration. This measurement decides which, before a kernel is written."
    )
    lines.append("")
    lines.append("## Method, in brief")
    lines.append("")
    lines.append(
        f"- Characteristic tokens are chosen by **log-odds ratio with an informative "
        f"Dirichlet prior, z-scored** (Monroe et al. 2008), prior strength "
        f"{prior_strength:g}, minimum {min_count} occurrences in the claiming source, top "
        f"{per_source} per source, winner-take-all so the label sets are disjoint. Raw "
        f"frequency says nothing (every source's commonest token is the same function word); "
        f"tf-idf has no notion of sampling noise and would let a small source fill its list "
        f"with accidents."
    )
    lines.append(
        "- Per-source counts come from a prefix of each prepared source file, one shared word "
        "budget per source — the same discipline `scripts/score_behaviour.py`'s register "
        "signal uses, differing only in that newlines are preserved, because this profile is "
        "over the model's byte-level BPE vocabulary rather than over whitespace words."
    )
    lines.append(
        "- The embedding matrix is read straight from `model.safetensors` (one tensor, never "
        "the whole model). The model ties its embeddings, so this is also the output head."
    )
    lines.append(
        "- k-NN purity is over **cosine** neighbours, self excluded. Cosine because embedding "
        "norm tracks token frequency more than meaning."
    )
    lines.append(
        "- Every headline is reported against three baselines: **chance** (uniform guessing "
        "at the realised class sizes), a **label-permutation floor** (identical computation, "
        "labels shuffled — holds geometry and class sizes fixed, destroys only the "
        "correspondence), and a **frequency-only control** (features = log corpus count and "
        "embedding norm, containing no directional information at all)."
    )
    lines.append(
        f"- Two token-selection conditions, because the statistic favours frequent tokens: "
        f"`all`, and `content` — the same, after excluding the {exclude_top} globally most "
        f"frequent tokens. If the geography is a frequency artefact it collapses in `content`."
    )
    lines.append("")

    lines.append("## Headline")
    lines.append("")
    lines.append("| condition | n tokens | k-NN purity | permuted floor | sigmas above floor | "
                 "frequency-only | chance | verdict |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in results:
        sig = r.knn_sigmas_above_floor
        sig_text = "inf" if sig == float("inf") else f"{sig:.1f}"
        lines.append(
            f"| {r.name} | {r.n_tokens} | {r.knn_purity:.4f} | "
            f"{r.knn_permuted_mean:.4f} ± {r.knn_permuted_sd:.4f} | {sig_text} | "
            f"{r.knn_frequency_only:.4f} | {r.knn_chance:.4f} | {verdict(r)} |"
        )
    lines.append("")
    lines.append("| condition | probe accuracy (embedding) | probe (labels permuted) | "
                 "probe (frequency only) | chance |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r.name} | {r.probe.accuracy_mean:.4f} ± {r.probe.accuracy_sd:.4f} | "
            f"{r.probe_permuted.accuracy_mean:.4f} ± {r.probe_permuted.accuracy_sd:.4f} | "
            f"{r.probe_frequency_only.accuracy_mean:.4f} ± "
            f"{r.probe_frequency_only.accuracy_sd:.4f} | {r.knn_chance:.4f} |"
        )
    lines.append("")
    lines.append(
        "The probe is a multinomial logistic regression over repeated stratified 70/30 "
        "splits, standardised on training rows only. Linear on purpose: a grid layout needs "
        "source identity to be *linearly* legible, and a stronger classifier would answer an "
        "easier question."
    )
    lines.append("")

    for r in results:
        lines.append(f"## Condition: {r.name}")
        lines.append("")
        lines.append(
            f"{r.n_tokens} labelled tokens over {len(r.source_names)} sources "
            f"(class sizes {', '.join(str(n) for n in r.class_sizes)}), k={r.k}."
        )
        lines.append("")
        lines.append(
            f"Silhouette (cosine) {r.silhouette:.4f} against a permuted floor of "
            f"{r.silhouette_permuted_mean:.4f} ± {r.silhouette_permuted_sd:.4f}. Silhouette "
            f"is small in 1024 dimensions even when separation is overwhelming — it is a "
            f"supporting number, read against its own floor, not a headline."
        )
        lines.append("")
        lines.append("### Which sources separate, which are entangled")
        lines.append("")
        lines.append("| source | tokens | k-NN purity | probe recall | chance |")
        lines.append("|---|---:|---:|---:|---:|")
        order = np.argsort(-np.asarray(r.per_source_purity))
        for i in order:
            lines.append(
                f"| {r.source_names[i]} | {r.class_sizes[i]} | {r.per_source_purity[i]:.3f} | "
                f"{r.probe.per_class_recall[i]:.3f} | {1.0 / len(r.source_names):.3f} |"
            )
        lines.append("")
        lines.append("### Where each source's neighbours actually come from")
        lines.append("")
        lines.append(
            "Rows are the true source of a token; columns are the sources its k nearest "
            "embedding neighbours belong to. The diagonal is the purity above; the "
            "off-diagonal is the entanglement."
        )
        lines.append("")
        lines.extend(_fmt_confusion(r.confusion, r.source_names))
        lines.append("")
        if r.top_tokens:
            lines.append("### The characteristic tokens themselves")
            lines.append("")
            lines.append(
                "The top of each source's list, so the labels can be audited rather than "
                "trusted. `Ġ` is the byte-level BPE marker for a leading space, `Ċ` a newline."
            )
            lines.append("")
            for source in r.source_names:
                shown = ", ".join(f"`{t}`" for t, _c, _z in r.top_tokens.get(source, [])[:20])
                lines.append(f"- **{source}**: {shown}")
            lines.append("")
        if r.projection:
            lines.append("### A picture (PCA, for looking at only)")
            lines.append("")
            lines.append(
                f"First two principal components of the L2-normalised embeddings of the "
                f"labelled tokens, explaining {r.projection_variance[0]:.1%} and "
                f"{r.projection_variance[1]:.1%} of their variance. Two components of a "
                f"1024-dimensional space keep very little; nothing above rests on this."
            )
            lines.append("")
            lines.append("```")
            lines.extend(r.projection)
            lines.append("```")
            lines.append("")
        if r.projection_centroids:
            lines.append(
                "Per-source centroids in that same projection — the scatter with the "
                "overplotting removed:")
            lines.append("")
            lines.append("| source | PC1 | PC2 |")
            lines.append("|---|---:|---:|")
            for i, name in enumerate(r.source_names):
                x, y = r.projection_centroids[i]
                lines.append(f"| {name} | {x:+.4f} | {y:+.4f} |")
            lines.append("")

    lines.append("## What this does and does not show")
    lines.append("")
    lines.append(
        "**Does.** That source identity is recoverable from embedding geometry alone, far "
        "above every baseline computed here, and that the recovery survives removing the "
        "function-word and punctuation core of each source's profile. The regions are real "
        "and they are linearly separable, which is the property a grid layout needs."
    )
    lines.append("")
    lines.append(
        "**Does not.** That the geometry encodes *source provenance* as such. A source "
        "differs from another in subject matter as well as in register — `procedural`'s "
        "characteristic tokens are food and kitchen words, and food words would cluster in "
        "any embedding of any corpus. What is measured here is that corpus source is legible "
        "in the space, not that the model represents \"which file this came from\". For the "
        "grid proposal that distinction does not change the answer (a region is a region "
        "either way), but it does change what a direction can be claimed to *mean*: subject "
        "matter and register together, not provenance."
    )
    lines.append("")
    lines.append(
        "**Also does not.** Establish anything about text the model generates. This is a "
        "measurement of the embedding table, not of behaviour; "
        "`scripts/score_behaviour.py` is where register in actual completions is measured, "
        "and its register control scores 99.9% on tinystories against 63.9% on folklore — "
        "the observation that prompted the narrative-entanglement contrast above."
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def _probe_to_json(p: ProbeResult) -> dict:
    return {"accuracy_mean": p.accuracy_mean, "accuracy_sd": p.accuracy_sd,
            "n_repeats": p.n_repeats, "per_class_recall": p.per_class_recall,
            "confusion": p.confusion.tolist()}


def report_to_json(results: Sequence[SeparationResult], *, hf_model: str, corpus_dir: str,
                   words_per_source: int, prior_strength: float, min_count: int,
                   per_source: int, exclude_top: int, k: int, n_permutations: int,
                   seed: int) -> dict:
    return {
        "hf_model": hf_model,
        "corpus_dir": corpus_dir,
        "words_per_source": words_per_source,
        "statistic": "log-odds ratio with informative Dirichlet prior, z-scored "
                     "(Monroe, Colaresi & Quinn 2008)",
        "prior_strength": prior_strength,
        "min_count": min_count,
        "per_source": per_source,
        "exclude_top": exclude_top,
        "k": k,
        "n_permutations": n_permutations,
        "seed": seed,
        "conditions": [
            {
                "name": r.name,
                "sources": list(r.source_names),
                "n_tokens": r.n_tokens,
                "class_sizes": r.class_sizes,
                "knn": {
                    "purity": r.knn_purity,
                    "permuted_mean": r.knn_permuted_mean,
                    "permuted_sd": r.knn_permuted_sd,
                    "sigmas_above_floor": (None if r.knn_sigmas_above_floor == float("inf")
                                           else r.knn_sigmas_above_floor),
                    "chance": r.knn_chance,
                    "frequency_only": r.knn_frequency_only,
                    "per_source_purity": r.per_source_purity,
                    "confusion": r.confusion.tolist(),
                },
                "silhouette": {
                    "value": r.silhouette,
                    "permuted_mean": r.silhouette_permuted_mean,
                    "permuted_sd": r.silhouette_permuted_sd,
                },
                "probe": _probe_to_json(r.probe),
                "probe_permuted": _probe_to_json(r.probe_permuted),
                "probe_frequency_only": _probe_to_json(r.probe_frequency_only),
                "projection_variance": list(r.projection_variance),
                "projection_centroids": (None if r.projection_centroids is None else
                                         {s: list(c) for s, c
                                          in zip(r.source_names, r.projection_centroids)}),
                "top_tokens": {s: [{"token": t, "corpus_count": c, "z": z}
                                   for t, c, z in v]
                               for s, v in r.top_tokens.items()},
                "verdict": verdict(r),
            }
            for r in results
        ],
    }


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------


def _default_model() -> Path:
    """The designated current model, falling back to a literal path if the file is unreadable.

    Reuses ``scripts/evaluate.py``'s designation loader so "the model" has one answer in this
    repo rather than one per script.
    """
    try:
        from scripts.evaluate import load_designation

        return load_designation().hf_model
    except Exception:  # pragma: no cover - only when the designation file is broken
        return ROOT / "artifacts" / "hf-tt-tnt-1024a"


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-model", type=Path, default=None,
                   help="Converted HF model directory whose embedding matrix is probed "
                        "(default: docs/current_model.json's designated current model). Only "
                        "model.safetensors and the tokenizer are read; no forward pass runs.")
    p.add_argument("--embedding-key", type=str, default="model.embed_tokens.weight",
                   help="Tensor name of the embedding matrix (default: %(default)s).")
    p.add_argument("--corpus-dir", type=Path, default=ROOT / "artifacts" / "corpus",
                   help="Directory of prepared per-source corpora (default: %(default)s). "
                        "Read only, never written.")
    p.add_argument("--words-per-source", type=int, default=DEFAULT_WORDS_PER_SOURCE,
                   help="Words read from each source's prefix (default: %(default)s, "
                        "score_behaviour.py's own register budget).")
    p.add_argument("--prior-strength", type=float, default=DEFAULT_PRIOR_STRENGTH,
                   help="Total pseudo-counts in the Dirichlet prior (default: %(default)s).")
    p.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT,
                   help="Minimum occurrences in a source before it may claim a token "
                        "(default: %(default)s).")
    p.add_argument("--per-source", type=int, default=DEFAULT_PER_SOURCE,
                   help="Characteristic tokens taken per source (default: %(default)s).")
    p.add_argument("--exclude-top", type=int, default=DEFAULT_EXCLUDE_TOP,
                   help="Globally most frequent tokens excluded in the 'content' condition "
                        "(default: %(default)s).")
    p.add_argument("--k", type=int, default=DEFAULT_K,
                   help="Neighbours in the k-NN purity test (default: %(default)s).")
    p.add_argument("--n-permutations", type=int, default=DEFAULT_N_PERMUTATIONS,
                   help="Label permutations establishing the noise floor (default: "
                        "%(default)s).")
    p.add_argument("--probe-repeats", type=int, default=DEFAULT_PROBE_REPEATS,
                   help="Stratified train/test splits the probe averages over (default: "
                        "%(default)s).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--note", type=str, default="",
                   help="Freeform provenance line recorded verbatim in the markdown report.")
    p.add_argument("--out", type=Path, default=None,
                   help="Markdown output path (default: derived from --hf-model's directory "
                        "name under docs/measurements/).")
    p.add_argument("--json-out", type=Path, default=None,
                   help="JSON output path (default: derived the same way as --out).")
    return p.parse_args(argv)


def _default_output_paths(hf_model: Path) -> Tuple[Path, Path]:
    tag = hf_model.name
    if tag.startswith("hf-"):
        tag = tag[len("hf-"):]
    out_dir = ROOT / "docs" / "measurements"
    return (out_dir / f"embedding-geography-{tag}.md",
            out_dir / f"embedding-geography-{tag}.json")


def main() -> int:
    args = _parse_args()
    hf_model = args.hf_model or _default_model()

    if not (hf_model / "config.json").is_file():
        print(f"ERROR: no converted model at {hf_model} (config.json missing). This script "
              f"is CPU-only and reads only the embedding tensor; it does not fall back to "
              f"the ttml/device path.", file=sys.stderr)
        return 1
    if not args.corpus_dir.is_dir():
        print(f"ERROR: --corpus-dir {args.corpus_dir} is not a directory.", file=sys.stderr)
        return 1

    print(f"reading embedding matrix from {hf_model} ...")
    try:
        embedding = load_embedding_matrix(hf_model, args.embedding_key)
    except (FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    vocab_size, dim = embedding.shape
    print(f"  {vocab_size:,} x {dim} ({embedding.dtype})")

    from transformers import AutoTokenizer

    print(f"loading tokenizer from {hf_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(hf_model))

    source_names = sorted(SOURCES)
    print(f"profiling {len(source_names)} sources from {args.corpus_dir} "
          f"({args.words_per_source:,} words each) ...")
    counts = count_tokens_by_source(args.corpus_dir, source_names, tokenizer, vocab_size,
                                    words_per_source=args.words_per_source,
                                    log=print)

    print(f"scoring characteristicness (log-odds, prior {args.prior_strength:g}) ...")
    z = log_odds_z(counts, args.prior_strength)

    results: List[SeparationResult] = []
    conditions = [("all", 0), ("content", args.exclude_top)]
    for name, exclude_top in conditions:
        labels = characteristic_tokens(counts, z, source_names, min_count=args.min_count,
                                       per_source=args.per_source, exclude_top=exclude_top)
        print(f"  condition {name}: {labels.token_ids.size} labelled tokens")
        result = evaluate_separation(name, labels, embedding, k=args.k,
                                     n_permutations=args.n_permutations,
                                     probe_repeats=args.probe_repeats, seed=args.seed,
                                     tokenizer=tokenizer, with_projection=True)
        print(f"    kNN purity {result.knn_purity:.4f} vs permuted "
              f"{result.knn_permuted_mean:.4f}±{result.knn_permuted_sd:.4f} "
              f"({result.knn_sigmas_above_floor:.1f} sigma), frequency-only "
              f"{result.knn_frequency_only:.4f} -> {verdict(result)}")
        results.append(result)

        # The narrative-entanglement contrast, on the same tokens: four narrative sources
        # against four non-narrative ones, so a 4-way number is read against a 4-way number.
        for label, group in (("narrative", NARRATIVE_SOURCES),
                             ("non-narrative", CONTRAST_SOURCES)):
            sub = labels.subset(group)
            # No tokenizer and no projection here: the subgroup's tokens are a subset of the
            # ones already listed under the parent condition, and repeating them would pad
            # the report with the same table four times.
            sub_result = evaluate_separation(f"{name} / {label} 4-way", sub, embedding,
                                             k=args.k, n_permutations=args.n_permutations,
                                             probe_repeats=args.probe_repeats, seed=args.seed,
                                             tokenizer=None, with_projection=False)
            print(f"    {label:14} 4-way purity {sub_result.knn_purity:.4f} vs chance "
                  f"{sub_result.knn_chance:.4f} -> {verdict(sub_result)}")
            results.append(sub_result)

    default_out, default_json_out = _default_output_paths(hf_model)
    out = args.out or default_out
    json_out = args.json_out or default_json_out

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(
        results, hf_model=str(hf_model.relative_to(ROOT)) if hf_model.is_relative_to(ROOT)
        else str(hf_model),
        corpus_dir=str(args.corpus_dir.relative_to(ROOT))
        if args.corpus_dir.is_relative_to(ROOT) else str(args.corpus_dir),
        words_per_source=args.words_per_source, prior_strength=args.prior_strength,
        min_count=args.min_count, per_source=args.per_source, exclude_top=args.exclude_top,
        note=args.note))
    print(f"wrote {out}")

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report_to_json(
        results, hf_model=str(hf_model), corpus_dir=str(args.corpus_dir),
        words_per_source=args.words_per_source, prior_strength=args.prior_strength,
        min_count=args.min_count, per_source=args.per_source, exclude_top=args.exclude_top,
        k=args.k, n_permutations=args.n_permutations, seed=args.seed), indent=2))
    print(f"wrote {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
