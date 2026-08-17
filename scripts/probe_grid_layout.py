#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Does the corpus geography survive being flattened onto a Tensix grid?

THE GATE THIS DECIDES
---------------------
``scripts/probe_embedding_geography.py`` established that this model's embedding space has a
corpus geography: k-NN purity over **cosine** neighbours of 1,350 source-labelled tokens is
0.5458 (``content`` condition) against a label-permutation floor of 0.1103 ± 0.0031 — 139
sigma. That measurement also named the one risk it could not retire:

    the geography is emphatically **not 2-D**. PC1 and PC2 of the labelled tokens explain
    2.7% and 1.8% of their variance. A Tensix grid is two-dimensional.

So the proposal — lay the 32,000-token vocabulary onto Blackhole's grid of Tensix cores and
sample by *spatial neighbourhood*, so that six neighbours give six corpus registers rather
than six synonyms — rests on a squash from 1024 dimensions to 2, and nobody knows what
survives it. This script answers exactly that, and nothing else. It builds no sampler and no
kernel.

The gate, as the earlier report itself proposed it: **assign every token to a grid cell, then
re-run the same purity statistic with grid distance substituted for cosine distance.** Purity
at or near its permutation floor means the squash destroyed the geography and the idea needs a
different substrate. Purity at a useful fraction of 0.5458 means the layout is worth a kernel.

WHY AGGREGATE PURITY IS NECESSARY BUT NOT SUFFICIENT
----------------------------------------------------
A layout can pass the aggregate gate while failing the actual proposal, in two distinct ways,
so this script separates them by construction:

1. **Purity that is really clustering, not placement.** Tokens sharing a cell are at grid
   distance 0 from each other. If a token's ten nearest grid neighbours are mostly its own
   cell-mates, the purity statistic is measuring the *clustering* and would score the same
   with the cells scattered over the die at random. Every number below is therefore also
   computed for a **shuffled-placement** control: identical cells, positions permuted. Where
   the annealed layout and the shuffled control agree, the two-dimensional arrangement is
   doing nothing, whatever the headline says.

2. **Purity by blob.** Nine sources laid down as nine contiguous continents maximise purity
   and destroy the proposal: every interior cell's neighbours are the same register, so asking
   six times gives six synonyms — the exact failure the idea was invented to fix. So the
   aggregate is reported next to a **source-diversity** statistic: how many *distinct* sources
   a token's spatial neighbours represent, against the chance expectation from the source
   proportions and against the shuffled-placement control.

Those two pull in opposite directions, and that tension — not either number alone — is the
finding. A layout is viable only if it holds purity **and** diversity at once.

HOW THE LAYOUT IS BUILT
-----------------------
Three stages, all numpy, no sklearn (see CONSTRAINTS):

1. **Balanced spherical k-means** over the L2-normalised embeddings of the *whole* 32,000-token
   vocabulary into one cluster per grid cell. Balanced because each Tensix core has its own
   fixed L1 SRAM: a layout that puts 4,000 tokens on one core and 3 on another is not a layout
   this machine can run. Balance is enforced by a capacity cap of ``ceil(V / cells)`` and a
   regret-ordered greedy assignment (points with the most to lose choose first) — the standard
   capacity-constrained Lloyd variant. Spherical (cosine) because that is the metric the
   geography was measured in, embedding norm tracking frequency more than meaning.

   The clustering sees the **entire vocabulary**, not just the 1,350 labelled tokens, and never
   sees a label. A real layout has to place every token; and an unsupervised layout cannot
   flatter itself by arranging the very points it is scored on.

2. **Spectral initialisation.** Cluster centroids are projected onto their first two principal
   components and dealt onto the grid column-by-column (sort by PC1, cut into ``width`` groups
   of ``height``, sort each group by PC2). This is the 2-D squash done naively, and it is a
   deliberate starting point rather than an answer: the annealer's job is to improve on it, and
   how much it improves is itself informative.

3. **Simulated-annealing QAP.** The placement of clusters onto cells is a quadratic assignment
   problem with the objective

       cost(p) = sum_{i != j} similarity(i, j) * grid_distance(p(i), p(j))

   — the classic VLSI-placement form, minimised when similar clusters sit few hops apart. It is
   also literally the NoC cost model: ``grid_distance`` is **torus** Manhattan distance, because
   Blackhole's NoC wraps. Sum over all ordered pairs of distinct cells of ``grid_distance`` is
   invariant under permutation, so no centring of ``similarity`` is needed; pair swaps are
   evaluated by an O(cells) delta rather than a full recompute.

   Simulated annealing rather than anything smarter because QAP is NP-hard, the objective is
   cheap, and — the part that matters for honesty — **restarts give a distribution**. This
   script runs several clusterings x several annealing seeds per grid and reports the spread,
   not the best one. Tuning a layout until it passes would be the easiest way to fake this
   result, so the tool is built not to offer that option.

WHAT IS COMPARED, AND AGAINST WHAT
----------------------------------
Three layout variants, all with identical class sizes and identical k:

- ``annealed`` — the layout above.
- ``shuffled`` — the same clusters, their positions on the grid randomly permuted. Isolates
  what the *placement* contributes from what the *clustering* contributes.
- ``random-cells`` — tokens dealt into equal-sized cells at random, placement identity.
  Destroys both. A floor for the whole construction.

And for each variant, three readings of the same k-NN purity:

- ``all neighbours`` — the gate as stated: k nearest by grid distance, ties broken at random.
- ``off-cell`` — the same with cell-mates excluded outright, so only *adjacency between cells*
  can contribute. This is the number that speaks to "direction on the grid means register".
- ``within-cell`` — the fraction of same-cell labelled pairs that share a source: "does one
  core hold one register".

Every one of them is reported against the **label-permutation floor** computed exactly as
``probe_embedding_geography.permuted_purity`` computes it (that function is imported, not
reimplemented, so the floors are the same statistic), and the headline against the 0.5458
cosine baseline it has to be a useful fraction of. This project labels anything indistinguishable
from its floor **NOT INTERPRETABLE**, regardless of how good a picture it makes.

CONSTRAINTS THIS SCRIPT RESPECTS
--------------------------------
CPU only. Never imports ttml/ttnn, never opens a Tenstorrent device, never runs a forward pass;
it reads one tensor — the embedding matrix — out of ``model.safetensors``. Writes only to
``docs/measurements/``; reads ``artifacts/`` and never writes there. numpy only: the k-means,
the k-means++ seeding, the PCA, the annealer and every statistic are implemented here. sklearn
and matplotlib are importable in the venv this was developed on and are *not* declared
dependencies of this project, so neither is used — matching
``scripts/probe_embedding_geography.py``, whose statistics this reuses by import.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.probe_embedding_geography import (  # noqa: E402
    DEFAULT_EXCLUDE_TOP,
    DEFAULT_K,
    DEFAULT_MIN_COUNT,
    DEFAULT_PER_SOURCE,
    DEFAULT_PRIOR_STRENGTH,
    DEFAULT_WORDS_PER_SOURCE,
    TokenLabels,
    characteristic_tokens,
    count_tokens_by_source,
    load_embedding_matrix,
    log_odds_z,
    neighbour_indices,
    permuted_purity,
    purity_from_neighbours,
)
from train.corpus import SOURCES  # noqa: E402

#: The cosine k-NN purity this gate is measured against — the ``content`` condition headline of
#: ``docs/measurements/embedding-geography-tt-tnt-1024a.md``. Recomputed at run time from the
#: same tokens (see :func:`cosine_baseline`) rather than trusted; this constant only names the
#: published value so a drift between the two is visible in the report.
PUBLISHED_COSINE_PURITY = 0.5458

#: Label permutations behind every floor. Fewer than the geography probe's 200 because each
#: floor here is computed once per tie-break draw and there are several draws per layout.
DEFAULT_N_PERMUTATIONS = 100

#: Independent random tie-break draws per layout. Grid distance is an integer number of hops,
#: so a token's k nearest neighbours are massively tied and the ordering within a tie is
#: arbitrary. Rather than fix an arbitrary order, every purity is averaged over this many
#: independent draws and reported with the spread they produce.
DEFAULT_TIE_BREAKS = 8

#: Distinct clusterings (k-means seeds) per grid.
DEFAULT_N_CLUSTERINGS = 3

#: Annealing restarts per clustering. Total layouts per grid is the product, and the report
#: gives the distribution over all of them.
DEFAULT_N_ANNEALS = 3

#: Lloyd iterations in the balanced k-means. The balanced assignment is a greedy approximation,
#: so the objective is not guaranteed monotone and more iterations are not automatically better;
#: this is enough for the assignment to stop moving appreciably on this vocabulary.
DEFAULT_KMEANS_ITERS = 12

#: "Asking six times" from the proposal, made literal: six neighbouring cells, one token drawn
#: from each, how many distinct sources come back.
DEFAULT_N_DIRECTIONS = 6


# ---------------------------------------------------------------------------------------
# Grid geometry — the actual silicon
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GridSpec:
    """One candidate substrate: a rectangle of Tensix cores, wrapped or not."""

    name: str
    width: int
    height: int
    #: Torus (Blackhole's NoC wraps) or open mesh. Torus is the default and the reported case;
    #: the open mesh exists so the wrap can be shown to matter or not.
    torus: bool = True
    #: How many physical dies this rectangle stands for. > 1 is an idealisation — see ``note``.
    n_chips: int = 1
    note: str = ""

    @property
    def n_cells(self) -> int:
        return self.width * self.height

    @property
    def label(self) -> str:
        return f"{self.width}x{self.height}"


#: The substrates tried, smallest first. 11x10 is what the harvested p300c in this box actually
#: exposes; 17x12 is the full Blackhole die; the multi-chip rows tile dies 2x2.
#:
#: The multi-chip rows are deliberately OPTIMISTIC and are labelled as such in the report: a
#: real 4-chip mesh is a torus per die joined by Ethernet, and an inter-die hop is nothing like
#: the cost of a NoC hop. Treating 2x2 dies as one flat torus gives the layout more room than
#: the hardware would, which is the right direction for a gate — if the idea fails even on the
#: generous version of a bigger substrate, it fails on the real one too.
DEFAULT_GRIDS: Tuple[GridSpec, ...] = (
    GridSpec("harvested-p300c", 11, 10, note="110 functional Tensix on this box's harvested die"),
    GridSpec("full-die", 17, 12, note="140 functional Tensix on an unharvested die; 204 sites"),
    GridSpec("4x harvested", 22, 20, n_chips=4,
             note="four harvested dies tiled 2x2, treated as one flat torus (optimistic)"),
    GridSpec("4x full-die", 34, 24, n_chips=4,
             note="four full dies tiled 2x2, treated as one flat torus (optimistic)"),
)


def grid_distance_matrix(spec: GridSpec) -> np.ndarray:
    """``(cells, cells)`` matrix of hop counts between grid positions.

    Manhattan, because a NoC routes in x then y and every hop costs the same; **torus** when
    ``spec.torus``, because Blackhole's NoC wraps at the die edge, so the two ends of a row are
    neighbours rather than the furthest points on it. The wrap is not cosmetic: on an 11x10 open
    mesh the maximum distance is 19 hops, on the torus it is 10, which changes what "far" means
    by a factor of two.
    """
    if spec.width < 1 or spec.height < 1:
        raise ValueError(f"grid must be at least 1x1, got {spec.width}x{spec.height}")
    idx = np.arange(spec.n_cells)
    rows = idx // spec.width
    cols = idx % spec.width
    dcol = np.abs(cols[:, None] - cols[None, :])
    drow = np.abs(rows[:, None] - rows[None, :])
    if spec.torus:
        dcol = np.minimum(dcol, spec.width - dcol)
        drow = np.minimum(drow, spec.height - drow)
    return (dcol + drow).astype(np.float64)


# ---------------------------------------------------------------------------------------
# Stage 1 — balanced spherical k-means over the whole vocabulary
# ---------------------------------------------------------------------------------------


def unit_rows(features: np.ndarray) -> np.ndarray:
    """L2-normalise rows, guarding the zero vector (unused rows of a vocabulary happen)."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def kmeans_plusplus_cosine(unit: np.ndarray, n_clusters: int,
                           rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding under cosine distance: spread the initial centroids out.

    Random initial centroids on a 32,000-point vocabulary reliably leave several cells empty
    and several doing all the work, which a capacity cap then papers over by force. Seeding
    proportional to squared distance-from-the-nearest-chosen-centre costs one pass per cluster
    and removes that failure mode.
    """
    if not 1 <= n_clusters <= unit.shape[0]:
        raise ValueError(f"n_clusters must be in [1, {unit.shape[0]}], got {n_clusters}")
    chosen = [int(rng.integers(unit.shape[0]))]
    dist = 1.0 - unit @ unit[chosen[0]]
    for _ in range(1, n_clusters):
        weights = np.maximum(dist, 0.0) ** 2
        total = float(weights.sum())
        pick = (int(rng.integers(unit.shape[0])) if total <= 0
                else int(rng.choice(unit.shape[0], p=weights / total)))
        chosen.append(pick)
        dist = np.minimum(dist, 1.0 - unit @ unit[pick])
    return unit[np.asarray(chosen)].copy()


def balanced_assign(similarity: np.ndarray, capacity: int) -> np.ndarray:
    """Assign every row to a cluster, no cluster over ``capacity``, by regret-ordered greed.

    ``similarity`` is ``(n_points, n_clusters)`` cosine similarity to each centroid. Points are
    served in order of **regret** — how much worse their second choice is than their first — so
    a point with a strong preference gets it and a point that is nearly indifferent absorbs the
    compromise. That ordering is what makes a hard capacity cap cheap: an optimal balanced
    assignment is a transportation problem, and this greedy approximation costs one argsort.

    The cap is what makes the layout physical. Each Tensix core has its own L1 SRAM, so cells
    have to hold comparable numbers of tokens; an unbalanced clustering is not a layout this
    machine could run, however good its objective.
    """
    n_points, n_clusters = similarity.shape
    if capacity * n_clusters < n_points:
        raise ValueError(
            f"capacity {capacity} x {n_clusters} clusters cannot hold {n_points} points")
    preference = np.argsort(-similarity, axis=1)
    best = np.take_along_axis(similarity, preference[:, :1], axis=1)[:, 0]
    second = (np.take_along_axis(similarity, preference[:, 1:2], axis=1)[:, 0]
              if n_clusters > 1 else best)
    order = np.argsort(-(best - second), kind="stable")

    counts = np.zeros(n_clusters, dtype=np.int64)
    assignment = np.full(n_points, -1, dtype=np.int64)
    for point in order:
        for cluster in preference[point]:
            if counts[cluster] < capacity:
                assignment[point] = cluster
                counts[cluster] += 1
                break
    if int((assignment < 0).sum()):
        raise RuntimeError("balanced assignment left points unplaced; capacity arithmetic is wrong")
    return assignment


def balanced_spherical_kmeans(unit: np.ndarray, n_clusters: int, *, seed: int = 0,
                              iterations: int = DEFAULT_KMEANS_ITERS,
                              log=lambda msg: None) -> Tuple[np.ndarray, np.ndarray]:
    """Capacity-capped spherical k-means. Returns ``(assignment, centroids)``.

    Centroids are re-normalised to the sphere after each mean, which is what makes this
    *spherical* k-means rather than Euclidean k-means on normalised inputs — the update that
    matches the cosine objective the geography was measured in.

    A cluster that loses every member keeps its previous centroid rather than becoming NaN; the
    capacity cap makes that vanishingly unlikely, but a silent NaN would poison the placement
    stage without ever raising.
    """
    capacity = math.ceil(unit.shape[0] / n_clusters)
    rng = np.random.default_rng(seed)
    centroids = kmeans_plusplus_cosine(unit, n_clusters, rng)
    assignment = np.zeros(unit.shape[0], dtype=np.int64)
    for iteration in range(iterations):
        similarity = unit @ centroids.T
        new_assignment = balanced_assign(similarity, capacity)
        moved = int((new_assignment != assignment).sum()) if iteration else unit.shape[0]
        assignment = new_assignment
        for cluster in range(n_clusters):
            members = unit[assignment == cluster]
            if members.shape[0]:
                centroids[cluster] = members.mean(axis=0)
        centroids = unit_rows(centroids)
        log(f"    kmeans iter {iteration + 1:>2}: {moved:>6,} points moved")
        if iteration and moved == 0:
            break
    return assignment, centroids


# ---------------------------------------------------------------------------------------
# Stage 2/3 — placing clusters on the grid
# ---------------------------------------------------------------------------------------


def pca_2d(features: np.ndarray) -> np.ndarray:
    """First two principal components of ``features`` — the 2-D squash itself, done plainly."""
    centred = features - features.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ vt[:2].T


def spectral_grid_init(centroids: np.ndarray, spec: GridSpec) -> np.ndarray:
    """Deal clusters onto the grid by their 2-D principal coordinates.

    Returns ``placement``: ``placement[c]`` is the grid position holding cluster ``c``.

    Sort by PC1, cut into ``width`` columns of ``height``, sort each column by PC2. That is the
    naive squash — the thing the whole gate is suspicious of — and it is used only as a starting
    point for the annealer, so that "how much did annealing help" is a number the report can
    show rather than a claim.
    """
    if centroids.shape[0] != spec.n_cells:
        raise ValueError(f"{centroids.shape[0]} clusters for {spec.n_cells} cells")
    coords = pca_2d(centroids)
    placement = np.empty(spec.n_cells, dtype=np.int64)
    by_x = np.argsort(coords[:, 0], kind="stable")
    for col in range(spec.width):
        column = by_x[col * spec.height:(col + 1) * spec.height]
        column = column[np.argsort(coords[column, 1], kind="stable")]
        for row, cluster in enumerate(column):
            placement[cluster] = row * spec.width + col
    return placement


def qap_cost(similarity: np.ndarray, distance: np.ndarray, placement: np.ndarray) -> float:
    """``sum_{i != j} similarity(i,j) * distance(placement(i), placement(j))``.

    Lower is better: similar clusters end up few hops apart. The diagonal is excluded because a
    cluster's distance to itself is zero anyway and its self-similarity is a constant.
    """
    placed = distance[np.ix_(placement, placement)]
    return float((similarity * placed).sum() - np.trace(similarity * placed))


def _swap_delta(similarity: np.ndarray, distance: np.ndarray, placement: np.ndarray,
                a: int, b: int) -> float:
    """Change in :func:`qap_cost` from exchanging the positions of clusters ``a`` and ``b``.

    O(cells) instead of O(cells^2): only terms touching ``a`` or ``b`` move, the ``(a, b)`` term
    itself is unchanged because both matrices are symmetric, and the factor 2 covers ``(i, j)``
    and ``(j, i)``.
    """
    dsim = similarity[a] - similarity[b]
    ddist = distance[placement[b], placement] - distance[placement[a], placement]
    terms = dsim * ddist
    return float(2.0 * (terms.sum() - terms[a] - terms[b]))


def anneal_placement(similarity: np.ndarray, distance: np.ndarray, placement: np.ndarray, *,
                     n_steps: int, seed: int = 0,
                     log=lambda msg: None) -> Tuple[np.ndarray, float, float]:
    """Simulated annealing over pair swaps. Returns ``(placement, start_cost, end_cost)``.

    The temperature schedule is geometric from a start scaled to the observed spread of swap
    deltas (so it does not depend on the arbitrary units of the similarity matrix) down to a
    thousandth of it. The best placement ever seen is kept, not the last one.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    n = placement.shape[0]
    if n < 2:
        return placement.copy(), 0.0, 0.0
    rng = np.random.default_rng(seed)

    probe = np.array([abs(_swap_delta(similarity, distance, placement,
                                      *rng.choice(n, size=2, replace=False)))
                      for _ in range(min(200, n * 4))])
    t0 = float(probe.mean()) if probe.mean() > 0 else 1.0
    t1 = t0 / 1000.0
    decay = (t1 / t0) ** (1.0 / max(n_steps - 1, 1))

    current = placement.copy()
    cost = qap_cost(similarity, distance, current)
    start_cost = cost
    best = current.copy()
    best_cost = cost
    temperature = t0
    for step in range(n_steps):
        a, b = rng.choice(n, size=2, replace=False)
        delta = _swap_delta(similarity, distance, current, int(a), int(b))
        if delta <= 0.0 or rng.random() < math.exp(-delta / max(temperature, 1e-12)):
            current[a], current[b] = current[b], current[a]
            cost += delta
            if cost < best_cost:
                best_cost = cost
                best = current.copy()
        temperature *= decay
        if log is not None and n_steps >= 10 and (step + 1) % (n_steps // 5) == 0:
            log(f"    anneal {step + 1:>7,}/{n_steps:,}  cost {cost:,.0f} "
                f"(best {best_cost:,.0f}, from {start_cost:,.0f})")
    return best, start_cost, best_cost


# ---------------------------------------------------------------------------------------
# The statistics — grid distance substituted for cosine distance
# ---------------------------------------------------------------------------------------


def token_positions(assignment: np.ndarray, placement: np.ndarray,
                    token_ids: np.ndarray) -> np.ndarray:
    """Grid position of each labelled token: cluster of the token, position of the cluster."""
    return placement[assignment[token_ids]]


def pair_distances(positions: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """``(n_tokens, n_tokens)`` grid distance between labelled tokens. Zero iff same cell."""
    return distance[np.ix_(positions, positions)]


def tie_broken_neighbours(pair_dist: np.ndarray, k: int, rng: np.random.Generator, *,
                          exclude_same_cell: bool = False) -> np.ndarray:
    """``k`` nearest tokens by grid distance, self excluded, **ties broken at random**.

    Grid distance is an integer hop count, so a token typically has far more than ``k``
    equidistant neighbours — everything in its own cell is at distance 0. Fixing an arbitrary
    order (say, token id) would let vocabulary ordering leak into the statistic. Instead each
    distance gets independent uniform jitter in ``[0, 0.5)``, which can never reorder two
    genuinely different hop counts (they differ by at least 1) and randomises every tie. The
    caller repeats this and reports the spread.

    ``exclude_same_cell`` removes cell-mates from consideration entirely, which is the reading
    that isolates *adjacency between cells* — the only part of the statistic that a
    two-dimensional arrangement can be responsible for.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    scored = pair_dist + rng.random(pair_dist.shape) * 0.5
    if exclude_same_cell:
        scored = np.where(pair_dist <= 0.0, np.inf, scored)
    np.fill_diagonal(scored, np.inf)
    usable = int((np.isfinite(scored)).sum(axis=1).min())
    if usable < k:
        raise ValueError(
            f"k={k} but some token has only {usable} usable neighbours "
            f"(exclude_same_cell={exclude_same_cell})")
    return np.argsort(scored, axis=1, kind="stable")[:, :k]


def within_cell_purity(positions: np.ndarray, labels: np.ndarray) -> Tuple[float, float, int]:
    """Fraction of same-cell labelled pairs sharing a source, its chance level, and the pair count.

    "Does one Tensix core hold one register." Chance here is the probability two tokens drawn
    independently from the labelled pool share a source, ``sum_s p_s^2`` — not 1/n_sources,
    which would be wrong the moment the class sizes are unequal.
    """
    same_cell = positions[:, None] == positions[None, :]
    np.fill_diagonal(same_cell, False)
    agree = labels[:, None] == labels[None, :]
    n_pairs = int(same_cell.sum())
    shares = np.bincount(labels) / labels.size
    chance = float((shares ** 2).sum())
    if n_pairs == 0:
        return float("nan"), chance, 0
    return float(agree[same_cell].mean()), chance, n_pairs // 2


def purity_by_hop(pair_dist: np.ndarray, labels: np.ndarray,
                  max_hops: int = 5) -> List[Tuple[int, float, int]]:
    """Fraction of token pairs at *exactly* d hops that share a source, for d = 0, 1, 2, ...

    Returns ``[(hops, purity, n_pairs), ...]``. This is the k-NN statistic with the ranking
    taken out of it: no ties to break, no k to choose, no cell-mates to argue about — just how
    far the register correlation reaches across the die. Hop 0 is the within-cell number; the
    rate at which the rest decays toward the pair-share baseline (``sum_s p_s^2``) is the
    clearest single view of what "direction on the grid means register" is worth.
    """
    agree = labels[:, None] == labels[None, :]
    out: List[Tuple[int, float, int]] = []
    for hops in range(max_hops + 1):
        mask = pair_dist == float(hops)
        np.fill_diagonal(mask, False)
        n_pairs = int(mask.sum())
        out.append((hops, float(agree[mask].mean()) if n_pairs else float("nan"), n_pairs // 2))
    return out


def expected_distinct_sources(labels: np.ndarray, n_draws: int) -> float:
    """Distinct sources expected among ``n_draws`` tokens drawn i.i.d. from the labelled pool.

    ``sum_s (1 - (1 - p_s)^n)``. This is the brief's "chance expectation from source
    proportions": what a neighbourhood would show if position carried no information at all
    and every neighbour were a fresh draw from the corpus mixture.
    """
    if n_draws < 1:
        raise ValueError(f"n_draws must be >= 1, got {n_draws}")
    shares = np.bincount(labels) / labels.size
    return float((1.0 - (1.0 - shares) ** n_draws).sum())


def distinct_sources_per_row(neighbours: np.ndarray, labels: np.ndarray) -> float:
    """Mean number of distinct sources among each row's neighbours."""
    return float(np.mean([np.unique(labels[row]).size for row in neighbours]))


def six_direction_diversity(positions: np.ndarray, labels: np.ndarray, distance: np.ndarray,
                            n_directions: int, rng: np.random.Generator) -> float:
    """"Ask six times, in six directions": distinct sources among one token per nearby cell.

    For each labelled token, take the ``n_directions`` nearest *other occupied cells* (ties at
    random), draw one labelled token uniformly from each, and count how many distinct sources
    come back. This is the proposal's own claim made literal — it is not the aggregate purity
    statistic under another name, because it deliberately takes **one** token per cell, so a
    single crowded cell cannot supply the whole answer.

    The ceiling is ``min(n_directions, n_sources)``; the floor is 1 (every direction the same
    register), which is the "one big blob" failure.
    """
    occupied = np.unique(positions)
    if occupied.size <= n_directions:
        raise ValueError(
            f"{occupied.size} occupied cells cannot supply {n_directions} distinct directions")
    members = {int(pos): np.flatnonzero(positions == pos) for pos in occupied}
    cell_dist = distance[np.ix_(occupied, occupied)]
    order = np.argsort(cell_dist + rng.random(cell_dist.shape) * 0.5, axis=1, kind="stable")
    home = {int(pos): i for i, pos in enumerate(occupied)}

    counts = []
    for token, pos in enumerate(positions):
        row = order[home[int(pos)]]
        picked = []
        for cell_index in row:
            cell = int(occupied[cell_index])
            if cell == int(pos):
                continue
            pool = members[cell]
            picked.append(labels[pool[rng.integers(pool.size)]])
            if len(picked) == n_directions:
                break
        counts.append(np.unique(picked).size)
    return float(np.mean(counts))


def oracle_continent_placement(assignment: np.ndarray, labels: TokenLabels,
                               spec: GridSpec) -> np.ndarray:
    """A **cheating** placement that reads the labels: same-source cells laid down contiguously.

    Each cluster is tagged with the source that dominates its labelled tokens (clusters holding
    none are tagged last), and the tagged clusters are dealt onto the grid in boustrophedon
    order, so every source's cells form one contiguous run.

    This is **not a candidate layout** — it uses the very labels the statistic scores, so it
    could never be built without already knowing the answer. It is here as a *ceiling*: it is
    approximately the best any placement of this clustering could do at adjacency-purity, so it
    says whether a weak annealed result is the annealer's fault or the squash's. It is also the
    "nine continents" failure mode made concrete, so the purity-versus-diversity trade-off can
    be read off a real layout rather than argued about.
    """
    dominant = np.full(spec.n_cells, -1, dtype=np.int64)
    token_cluster = assignment[labels.token_ids]
    for cluster in range(spec.n_cells):
        members = labels.labels[token_cluster == cluster]
        if members.size:
            dominant[cluster] = int(np.bincount(members).argmax())
    # Boustrophedon: consecutive clusters stay adjacent when a row ends.
    snake: List[int] = []
    for row in range(spec.height):
        cols = range(spec.width) if row % 2 == 0 else range(spec.width - 1, -1, -1)
        snake.extend(row * spec.width + col for col in cols)
    key = np.where(dominant < 0, labels.n_sources, dominant)
    placement = np.empty(spec.n_cells, dtype=np.int64)
    placement[np.argsort(key, kind="stable")] = np.asarray(snake, dtype=np.int64)
    return placement


@dataclass
class PurityReading:
    """One purity statistic with the floor it is read against, over tie-break draws."""

    name: str
    mean: float
    sd: float
    permuted_mean: float
    permuted_sd: float
    chance: float

    @property
    def sigmas(self) -> float:
        """Distance above the label-permutation floor, in floor sds. Below 3 is not a finding."""
        if self.permuted_sd <= 0:
            return float("inf") if self.mean > self.permuted_mean else 0.0
        return (self.mean - self.permuted_mean) / self.permuted_sd


def purity_reading(name: str, pair_dist: np.ndarray, labels: np.ndarray, k: int, *,
                   tie_breaks: int, n_permutations: int, rng: np.random.Generator,
                   exclude_same_cell: bool = False) -> PurityReading:
    """k-NN purity over grid distance, averaged over tie-break draws, with its own floor.

    The floor is ``probe_embedding_geography.permuted_purity`` — imported, not reimplemented,
    so this gate's floor and the cosine measurement's floor are the same computation on the
    same class sizes, and the ratio between the two headlines is meaningful.
    """
    values: List[float] = []
    floors: List[float] = []
    spreads: List[float] = []
    for _ in range(tie_breaks):
        nb = tie_broken_neighbours(pair_dist, k, rng, exclude_same_cell=exclude_same_cell)
        values.append(purity_from_neighbours(nb, labels))
        floor_mean, floor_sd = permuted_purity(nb, labels, n_permutations, rng)
        floors.append(floor_mean)
        spreads.append(floor_sd)
    shares = np.bincount(labels) / labels.size
    return PurityReading(
        name=name,
        mean=float(np.mean(values)),
        sd=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        permuted_mean=float(np.mean(floors)),
        permuted_sd=float(np.mean(spreads)),
        chance=float((shares ** 2).sum()),
    )


@dataclass
class LayoutEvaluation:
    """Everything measured about one (layout variant, token condition) pair."""

    variant: str
    condition: str
    grid: str
    n_cells: int
    purity_all: PurityReading
    purity_offcell: PurityReading
    within_cell: float
    within_cell_chance: float
    diversity_knn: float
    diversity_knn_chance: float
    diversity_directions: float
    diversity_directions_chance: float
    n_directions: int
    occupied_cells: int
    tokens_per_cell_max: int
    #: ``[(hops, purity, n_pairs), ...]`` — the correlation's reach across the die.
    hop_profile: List[Tuple[int, float, int]] = field(default_factory=list)


def evaluate_layout(variant: str, condition: str, spec: GridSpec, distance: np.ndarray,
                    assignment: np.ndarray, placement: np.ndarray, labels: TokenLabels, *,
                    k: int, tie_breaks: int, n_permutations: int, n_directions: int,
                    seed: int) -> LayoutEvaluation:
    """Run the whole battery for one layout on one labelled token set."""
    rng = np.random.default_rng(seed)
    positions = token_positions(assignment, placement, labels.token_ids)
    pair_dist = pair_distances(positions, distance)
    y = labels.labels

    all_read = purity_reading("all neighbours", pair_dist, y, k, tie_breaks=tie_breaks,
                              n_permutations=n_permutations, rng=rng)
    off_read = purity_reading("off-cell", pair_dist, y, k, tie_breaks=tie_breaks,
                              n_permutations=n_permutations, rng=rng, exclude_same_cell=True)
    within, within_chance, _pairs = within_cell_purity(positions, y)

    nb = tie_broken_neighbours(pair_dist, k, rng)
    diversity = distinct_sources_per_row(nb, y)
    directions = six_direction_diversity(positions, y, distance, n_directions, rng)

    counts = np.bincount(positions, minlength=spec.n_cells)
    return LayoutEvaluation(
        variant=variant, condition=condition, grid=spec.label, n_cells=spec.n_cells,
        purity_all=all_read, purity_offcell=off_read,
        within_cell=within, within_cell_chance=within_chance,
        diversity_knn=diversity, diversity_knn_chance=expected_distinct_sources(y, k),
        diversity_directions=directions,
        diversity_directions_chance=expected_distinct_sources(y, n_directions),
        n_directions=n_directions,
        occupied_cells=int((counts > 0).sum()),
        tokens_per_cell_max=int(counts.max()),
        hop_profile=purity_by_hop(pair_dist, y))


def _mean_reading(readings: Sequence[PurityReading]) -> PurityReading:
    """Average a set of readings, with ``sd`` becoming the spread ACROSS them."""
    values = np.array([r.mean for r in readings])
    return PurityReading(
        name=readings[0].name,
        mean=float(values.mean()),
        sd=float(values.std(ddof=1)) if values.size > 1 else 0.0,
        permuted_mean=float(np.mean([r.permuted_mean for r in readings])),
        permuted_sd=float(np.mean([r.permuted_sd for r in readings])),
        chance=readings[0].chance)


def mean_evaluation(evaluations: Sequence[LayoutEvaluation]) -> LayoutEvaluation:
    """The element-wise mean of several layouts' statistics.

    Verdicts are computed from this rather than from one arbitrarily chosen layout, so that a
    lucky restart cannot decide whether a kernel gets written. ``sd`` on the averaged purity
    readings becomes the spread *across layouts* rather than across tie-break draws, which is
    the spread a reader of a verdict wants.
    """
    if not evaluations:
        raise ValueError("cannot average zero layouts")
    first = evaluations[0]
    hops = range(len(first.hop_profile))
    return LayoutEvaluation(
        variant=first.variant, condition=first.condition, grid=first.grid,
        n_cells=first.n_cells,
        purity_all=_mean_reading([e.purity_all for e in evaluations]),
        purity_offcell=_mean_reading([e.purity_offcell for e in evaluations]),
        within_cell=float(np.mean([e.within_cell for e in evaluations])),
        within_cell_chance=first.within_cell_chance,
        diversity_knn=float(np.mean([e.diversity_knn for e in evaluations])),
        diversity_knn_chance=first.diversity_knn_chance,
        diversity_directions=float(np.mean([e.diversity_directions for e in evaluations])),
        diversity_directions_chance=first.diversity_directions_chance,
        n_directions=first.n_directions,
        occupied_cells=int(np.mean([e.occupied_cells for e in evaluations])),
        tokens_per_cell_max=int(np.mean([e.tokens_per_cell_max for e in evaluations])),
        hop_profile=[(hop, _hop_mean(evaluations, hop),
                      int(np.mean([e.hop_profile[hop][2] for e in evaluations])))
                     for hop in hops])


def cosine_baseline(embedding: np.ndarray, labels: TokenLabels, k: int, *,
                    n_permutations: int, seed: int) -> PurityReading:
    """The cosine k-NN purity this gate is a fraction of, recomputed on the same tokens.

    Recomputed rather than quoted, so the report can show that this script's token selection
    reproduces ``probe_embedding_geography.py``'s published headline before asking anyone to
    believe its ratio to it.
    """
    rng = np.random.default_rng(seed)
    nb = neighbour_indices(embedding[labels.token_ids], k)
    value = purity_from_neighbours(nb, labels.labels)
    floor_mean, floor_sd = permuted_purity(nb, labels.labels, n_permutations, rng)
    shares = np.bincount(labels.labels) / labels.labels.size
    return PurityReading(name="cosine (1024-d)", mean=value, sd=0.0, permuted_mean=floor_mean,
                         permuted_sd=floor_sd, chance=float((shares ** 2).sum()))


# ---------------------------------------------------------------------------------------
# Verdicts — this project's standard, applied
# ---------------------------------------------------------------------------------------


def purity_verdict(reading: PurityReading) -> str:
    """Is this purity distinguishable from its own label-permutation floor?"""
    return ("NOT INTERPRETABLE (at or inside the label-permutation floor)"
            if reading.sigmas < 3.0 else "above floor")


def headroom_retention(value: float, floor: float, cosine: PurityReading) -> float:
    """What fraction of the cosine measurement's headroom above its floor this value keeps.

    A ratio of raw purities flatters every grid result, because a third of the cosine purity is
    the floor itself and the floor is free — a layout that learned nothing would still score
    0.11/0.55 = 20%. Measuring ``(value - floor) / (cosine - cosine_floor)`` prices the floor
    out of both sides, so 0 means "learned nothing" and 1 means "the squash cost nothing".
    Both ratios appear in the report; this one is the one the verdict uses.
    """
    span = cosine.mean - cosine.permuted_mean
    return (value - floor) / span if span > 0 else 0.0


def diversity_headroom(observed: float, chance: float) -> float:
    """Where a neighbourhood's distinct-source count sits between "one register" and chance.

    ``(observed - 1) / (chance - 1)``. A plain ratio to chance is a bad scale here because its
    floor is not zero: every neighbourhood returns at least one source, so the worst possible
    layout — the "six synonyms" failure the proposal exists to fix — still scores 1/4.56 = 22%
    of chance on nine sources. Anchoring at 1 makes 0 mean exactly that failure and 1 mean "as
    diverse as drawing from the corpus at random".
    """
    return (observed - 1.0) / (chance - 1.0) if chance > 1.0 else 0.0


def grid_verdicts(annealed: LayoutEvaluation, shuffled: LayoutEvaluation,
                  oracle: LayoutEvaluation, cosine: PurityReading, *,
                  retention_bar: float = 0.5, direction_bar: float = 0.5,
                  diversity_bar: float = 0.5) -> Dict[str, str]:
    """Verdicts on the two *separate* claims the proposal makes, plus the joint one.

    The proposal bundles two claims that this measurement can pull apart, and they do not have
    the same answer, so reporting one blended verdict would hide the finding:

    - **"neighbourhood sampling returns divergent, register-coherent draws"** — a claim about
      cells. It needs cells that are individually register-coherent (``within-cell purity``
      above the chance that two tokens share a source) and neighbourhoods that still span
      several sources (``N-direction diversity`` at ``diversity_bar`` of chance or better).
      Nothing about it requires the *arrangement* of cells to mean anything.

    - **"spatial direction corresponds to corpus register"** — a claim about placement, and the
      one the 2-D squash threatens. It needs adjacency between cells to carry register on its
      own: ``off-cell`` purity above its floor, and the annealed placement beating a random
      placement of the *same cells* by ``direction_bar`` of the cosine headroom. The
      ``oracle-continents`` ceiling says whether that bar was even reachable with this
      clustering, so a failure can be attributed to the squash rather than to the annealer.

    The joint verdict is deliberately the conjunction: the proposal as pitched needs both.
    """
    verdicts: Dict[str, str] = {}

    aggregate = headroom_retention(annealed.purity_all.mean,
                                   annealed.purity_all.permuted_mean, cosine)
    if annealed.purity_all.sigmas < 3.0:
        verdicts["aggregate_purity"] = ("NOT INTERPRETABLE (grid purity is inside its "
                                        "label-permutation floor)")
    elif aggregate < retention_bar:
        verdicts["aggregate_purity"] = (f"COLLAPSED ({aggregate:.0%} of the cosine headroom "
                                        f"survives the squash)")
    else:
        verdicts["aggregate_purity"] = (f"SURVIVES ({aggregate:.0%} of the cosine headroom "
                                        f"survives the squash)")

    coherence = headroom_retention(annealed.within_cell, annealed.within_cell_chance, cosine)
    coherent = coherence >= retention_bar
    diversity = diversity_headroom(annealed.diversity_directions,
                                   annealed.diversity_directions_chance)
    if coherent and diversity >= diversity_bar:
        verdicts["neighbourhood_sampling"] = (
            f"SUPPORTED (a cell is {coherence:.0%} as register-coherent as a cosine "
            f"neighbourhood, and {annealed.n_directions} neighbouring cells still return "
            f"{annealed.diversity_directions:.2f} distinct sources — {diversity:.0%} of the way "
            f"from one register to chance)")
    elif not coherent:
        verdicts["neighbourhood_sampling"] = (
            f"REFUTED (cells are not register-coherent: {annealed.within_cell:.3f} against "
            f"{annealed.within_cell_chance:.3f} chance keeps only {coherence:.0%} of the "
            f"cosine headroom)")
    else:
        verdicts["neighbourhood_sampling"] = (
            f"REFUTED ({annealed.n_directions} neighbouring cells return "
            f"{annealed.diversity_directions:.2f} distinct sources, only {diversity:.0%} of the "
            f"way from one register to chance — six synonyms, not six registers)")

    placement = headroom_retention(annealed.purity_offcell.mean,
                                   shuffled.purity_offcell.mean, cosine)
    ceiling = headroom_retention(oracle.purity_offcell.mean,
                                 shuffled.purity_offcell.mean, cosine)
    if annealed.purity_offcell.sigmas < 3.0:
        verdicts["direction_means_register"] = (
            "REFUTED (adjacency-only purity is inside its label-permutation floor)")
    elif placement < direction_bar:
        verdicts["direction_means_register"] = (
            f"REFUTED (adjacency carries {placement:.0%} of the cosine headroom; even the "
            f"label-cheating continent layout reaches only {ceiling:.0%}, so the limit is the "
            f"2-D squash, not the annealer)")
    else:
        verdicts["direction_means_register"] = (
            f"SUPPORTED (adjacency carries {placement:.0%} of the cosine headroom, against a "
            f"label-cheating ceiling of {ceiling:.0%})")

    failed = [name for name, text in verdicts.items()
              if text.startswith(("REFUTED", "COLLAPSED", "NOT INTERPRETABLE"))]
    verdicts["overall"] = ("VIABLE AS PITCHED" if not failed
                           else "NOT VIABLE AS PITCHED (" + ", ".join(failed) + ")")
    return verdicts


# ---------------------------------------------------------------------------------------
# A picture — which source dominates each cell
# ---------------------------------------------------------------------------------------


def ascii_grid_map(positions: np.ndarray, labels: np.ndarray, source_names: Sequence[str],
                   spec: GridSpec) -> List[str]:
    """One character per grid cell: the source that dominates its labelled tokens, '.' if none.

    For looking at, not for deciding: it shows *labelled* tokens only, which are 1,350 of 32,000.
    Its job is to let a reader see at a glance whether the layout is continents or confetti,
    since that distinction is the whole difference between "six registers" and "six synonyms".
    Left-aligned with no right-hand border, per this project's terminal-output convention.
    """
    glyphs = [name[0].upper() for name in source_names]
    seen: Dict[str, int] = {}
    for i, glyph in enumerate(glyphs):
        if glyph in seen:
            glyphs[i] = source_names[i][1].lower()
        seen[glyphs[i]] = i

    lines = []
    for row in range(spec.height):
        chars = []
        for col in range(spec.width):
            cell = row * spec.width + col
            members = labels[positions == cell]
            chars.append("." if members.size == 0
                         else glyphs[int(np.bincount(members).argmax())])
            chars.append(" ")
        lines.append("".join(chars).rstrip())
    lines.append("")
    lines.append("legend: " + "  ".join(f"{glyphs[i]}={n}" for i, n in enumerate(source_names))
                 + "   .=no labelled token in that cell")
    return lines


# ---------------------------------------------------------------------------------------
# Driving one grid
# ---------------------------------------------------------------------------------------


@dataclass
class GridResult:
    """Every layout tried on one grid, and the map of the best-costing annealed one."""

    spec: GridSpec
    n_layouts: int
    evaluations: List[LayoutEvaluation] = field(default_factory=list)
    anneal_cost_start: List[float] = field(default_factory=list)
    anneal_cost_end: List[float] = field(default_factory=list)
    grid_map: Optional[List[str]] = None
    seconds: float = 0.0

    def select(self, variant: str, condition: str) -> List[LayoutEvaluation]:
        return [e for e in self.evaluations
                if e.variant == variant and e.condition == condition]


def _hop_mean(evaluations: Sequence["LayoutEvaluation"], hop: int) -> float:
    """Mean purity at exactly ``hop`` hops across layouts, ignoring layouts with no such pair."""
    values = [e.hop_profile[hop][1] for e in evaluations
              if hop < len(e.hop_profile) and not math.isnan(e.hop_profile[hop][1])]
    return float(np.mean(values)) if values else float("nan")


def _spread(values: Sequence[float]) -> Tuple[float, float, float, float]:
    """``(mean, sd, min, max)`` — the distribution, never just the best."""
    arr = np.asarray(values, dtype=np.float64)
    return (float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            float(arr.min()), float(arr.max()))


def run_grid(spec: GridSpec, unit: np.ndarray, conditions: Dict[str, TokenLabels], *,
             n_clusterings: int, n_anneals: int, kmeans_iters: int, k: int,
             tie_breaks: int, n_permutations: int, n_directions: int, anneal_steps: int,
             seed: int, log=print) -> GridResult:
    """Build every layout for one grid and evaluate all three variants on it."""
    started = time.time()
    distance = grid_distance_matrix(spec)
    result = GridResult(spec=spec, n_layouts=n_clusterings * n_anneals)
    steps = max(anneal_steps, 400 * spec.n_cells)
    primary_condition = next(iter(conditions))
    best_cost = float("inf")

    for c_index in range(n_clusterings):
        log(f"  clustering {c_index + 1}/{n_clusterings} "
            f"({spec.n_cells} balanced cells over {unit.shape[0]:,} tokens)")
        assignment, centroids = balanced_spherical_kmeans(
            unit, spec.n_cells, seed=seed + 101 * c_index, iterations=kmeans_iters, log=log)
        similarity = centroids @ centroids.T
        init = spectral_grid_init(centroids, spec)

        for a_index in range(n_anneals):
            layout_seed = seed + 1009 * c_index + 31 * a_index
            placement, cost0, cost1 = anneal_placement(
                similarity, distance, init, n_steps=steps, seed=layout_seed, log=log)
            result.anneal_cost_start.append(cost0)
            result.anneal_cost_end.append(cost1)
            rng = np.random.default_rng(layout_seed + 7)
            # Same cells, positions permuted: what the CLUSTERING alone is worth.
            shuffled = rng.permutation(spec.n_cells)
            # Tokens dealt into equal-sized cells at random: what NEITHER is worth.
            random_assignment = np.empty(unit.shape[0], dtype=np.int64)
            random_assignment[rng.permutation(unit.shape[0])] = (
                np.arange(unit.shape[0]) % spec.n_cells)

            for condition, labels in conditions.items():
                for variant, (assign_v, place_v) in {
                    "annealed": (assignment, placement),
                    "oracle-continents": (assignment,
                                          oracle_continent_placement(assignment, labels, spec)),
                    "shuffled": (assignment, shuffled),
                    "random-cells": (random_assignment, np.arange(spec.n_cells)),
                }.items():
                    result.evaluations.append(evaluate_layout(
                        variant, condition, spec, distance, assign_v, place_v, labels,
                        k=k, tie_breaks=tie_breaks, n_permutations=n_permutations,
                        n_directions=n_directions, seed=layout_seed + 13))
            if cost1 < best_cost:
                best_cost = cost1
                positions = token_positions(assignment, placement,
                                            conditions[primary_condition].token_ids)
                result.grid_map = ascii_grid_map(
                    positions, conditions[primary_condition].labels,
                    conditions[primary_condition].source_names, spec)
            hit = [e for e in result.evaluations
                   if e.variant == "annealed" and e.condition == primary_condition][-1]
            log(f"    layout {c_index * n_anneals + a_index + 1}/{result.n_layouts}: "
                f"cost {cost0:,.0f} -> {cost1:,.0f}; purity {hit.purity_all.mean:.4f} "
                f"(floor {hit.purity_all.permuted_mean:.4f}), off-cell "
                f"{hit.purity_offcell.mean:.4f}, {hit.diversity_directions:.2f} distinct "
                f"sources in {n_directions} directions")
    result.seconds = time.time() - started
    return result


# ---------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------


def render_markdown(results: Sequence[GridResult], cosine: Dict[str, PurityReading], *,
                    hf_model: str, condition: str, k: int, n_directions: int,
                    tie_breaks: int, n_permutations: int, note: str = "") -> str:
    lines: List[str] = []
    add = lines.append
    add("<!-- SPDX-License-Identifier: Apache-2.0 -->")
    add("<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->")
    add("")
    add("# Grid-distance gate — does the corpus geography survive the 2-D squash?")
    add("")
    add(f"Model `{hf_model}` (embedding matrix only), generated by "
        f"`scripts/probe_grid_layout.py`. CPU only; no device opened.")
    if note:
        add("")
        add(note)
    add("")
    add("## What is being decided")
    add("")
    add("`docs/measurements/embedding-geography-tt-tnt-1024a.md` established that corpus "
        "sources occupy distinguishable regions of this model's embedding space — k-NN purity "
        f"**{PUBLISHED_COSINE_PURITY:.4f}** over cosine neighbours against a permutation floor "
        "of 0.1103, 139 sigma. It also named the risk it could not retire: that geography is "
        "**not 2-D** (PC1 and PC2 explain 2.7% and 1.8% of variance), and a Tensix grid is. "
        "This measurement substitutes **grid distance for cosine distance** in the identical "
        "purity statistic and asks what is left.")
    add("")
    add("Three things are measured, because the aggregate alone can pass while the proposal "
        "fails:")
    add("")
    add("1. **Grid-distance k-NN purity** — the gate as stated.")
    add("2. **Off-cell purity** — the same with cell-mates excluded, so only adjacency between "
        "cells can contribute. This is the only part a two-dimensional *arrangement* can be "
        "responsible for; the rest is the clustering, which would score the same with the cells "
        "thrown onto the die at random.")
    add(f"3. **Source diversity** — how many distinct sources a neighbourhood returns, against "
        f"the chance expectation from the source proportions. A layout that paints nine "
        f"continents maximises purity and returns {n_directions} synonyms, which is the failure "
        f"the proposal exists to fix.")
    add("")
    add("Controls, all with identical class sizes and identical k: **shuffled** (same cells, "
        "positions permuted — isolates placement from clustering) and **random-cells** (tokens "
        "dealt into equal cells at random — destroys both). Every purity carries its own "
        "label-permutation floor, computed by the same function the cosine measurement used.")
    add("")

    add("## The baseline this is a fraction of")
    add("")
    add("| statistic | value | permuted floor | sigmas | chance |")
    add("|---|---:|---:|---:|---:|")
    for name, reading in cosine.items():
        add(f"| cosine k-NN purity, condition `{name}` | {reading.mean:.4f} | "
            f"{reading.permuted_mean:.4f} ± {reading.permuted_sd:.4f} | "
            f"{reading.sigmas:.1f} | {reading.chance:.4f} |")
    add("")
    add(f"Recomputed here on the same tokens rather than quoted; the published `content` "
        f"headline is {PUBLISHED_COSINE_PURITY:.4f}.")
    add("")

    base = cosine[condition].mean
    add(f"## Headline — condition `{condition}`, k={k}")
    add("")
    add(f"Every cell is mean ± sd over the layouts tried for that grid ({tie_breaks} random "
        f"tie-break draws each, since grid distance is an integer hop count and neighbours are "
        f"massively tied). **Never the best layout — the distribution.**")
    add("")
    add("| grid | cells | layouts | grid purity | permuted floor | sigmas | ratio to cosine | "
        "headroom kept | within-cell | off-cell | shuffled off-cell | oracle off-cell | "
        "adjacency headroom |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        ann = r.select("annealed", condition)
        shuf = r.select("shuffled", condition)
        orc = r.select("oracle-continents", condition)
        pm, ps, _lo, _hi = _spread([e.purity_all.mean for e in ann])
        floor = float(np.mean([e.purity_all.permuted_mean for e in ann]))
        floor_sd = float(np.mean([e.purity_all.permuted_sd for e in ann]))
        sig = (pm - floor) / floor_sd if floor_sd > 0 else float("inf")
        om, os_, _a, _b = _spread([e.purity_offcell.mean for e in ann])
        sm, ss, _c, _d = _spread([e.purity_offcell.mean for e in shuf])
        cm, _cs, _g, _h = _spread([e.purity_offcell.mean for e in orc])
        wm, ws, _e, _f = _spread([e.within_cell for e in ann])
        add(f"| {r.spec.name} {r.spec.label} | {r.spec.n_cells} | {len(ann)} | "
            f"{pm:.4f} ± {ps:.4f} | {floor:.4f} ± {floor_sd:.4f} | {sig:.1f} | "
            f"{pm / base:.2f} | {headroom_retention(pm, floor, cosine[condition]):.0%} | "
            f"{wm:.4f} ± {ws:.4f} | {om:.4f} ± {os_:.4f} | {sm:.4f} ± {ss:.4f} | "
            f"{cm:.4f} | {headroom_retention(om, sm, cosine[condition]):.0%} |")
    add("")
    add("**The columns after `headroom kept` are where the answer is.** `headroom kept` prices "
        "the floor out of both sides — `(grid purity - floor) / (cosine purity - cosine "
        "floor)` — because a third of the raw cosine number is floor, and a layout that learned "
        "nothing would still score a fifth of it on the naive ratio.")
    add("")
    add("- `within-cell` is the fraction of same-cell labelled pairs that share a source: **does "
        "one Tensix core hold one register**. It is a property of the clustering; the placement "
        "cannot change it.")
    add("- `off-cell` excludes cell-mates outright, so only **adjacency between cells** can "
        "contribute — the only part a two-dimensional *arrangement* is responsible for.")
    add("- `shuffled off-cell` is the same number with the cells scattered over the die at "
        "random, and it is the honest reference for `off-cell`, not the permutation floor. "
        "(It sits slightly *below* the permutation floor by construction: a token's own source "
        "is concentrated in its own cell, so excluding that cell leaves a pool depleted of its "
        "source. Permuting labels destroys that concentration and so misses the depletion. Both "
        "annealed and shuffled carry the bias equally, which is why they are compared to each "
        "other.)")
    add("- `oracle off-cell` is a **label-cheating** layout that groups same-source cells into "
        "contiguous continents. It cannot be built without already knowing the answer; it is "
        "here as a ceiling, so a weak annealed result can be attributed to the squash rather "
        "than to the annealer.")
    add("- `adjacency headroom` is `(off-cell - shuffled off-cell) / cosine headroom`: **the "
        "fraction of the register signal that survives as direction on the grid.** This is the "
        "number the pitch actually rests on.")
    add("")

    add("## Source diversity — does a neighbourhood give divergent registers?")
    add("")
    add(f"`{n_directions} directions` takes the {n_directions} nearest *other occupied cells* "
        f"and draws one token from each: the proposal's \"ask six times\" made literal. Chance "
        f"is the distinct-source count expected from {n_directions} independent draws at the "
        f"realised source proportions; the ceiling is {n_directions} on nine sources.")
    add("")
    add("| grid | annealed, k nearest | chance | annealed, N directions | oracle continents | "
        "shuffled | chance | ratio to chance | diversity headroom |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        ann = r.select("annealed", condition)
        shuf = r.select("shuffled", condition)
        orc = r.select("oracle-continents", condition)
        km, ks, _a, _b = _spread([e.diversity_knn for e in ann])
        dm, ds, _c, _d = _spread([e.diversity_directions for e in ann])
        om, _os, _g, _h = _spread([e.diversity_directions for e in orc])
        sm, ss, _e, _f = _spread([e.diversity_directions for e in shuf])
        chance_k = ann[0].diversity_knn_chance
        chance_d = ann[0].diversity_directions_chance
        add(f"| {r.spec.name} {r.spec.label} | {km:.2f} ± {ks:.2f} | {chance_k:.2f} | "
            f"{dm:.2f} ± {ds:.2f} | {om:.2f} | {sm:.2f} ± {ss:.2f} | {chance_d:.2f} | "
            f"{dm / chance_d:.2f} | {diversity_headroom(dm, chance_d):.0%} |")
    add("")
    add("`diversity headroom` is `(observed - 1) / (chance - 1)`: 0 is the **six synonyms** "
        "failure — every direction the same register — and 1 is as diverse as drawing from the "
        "corpus at random. A plain ratio to chance has its floor at 22%, not 0, so it flatters "
        "every layout including the worst one.")
    add("")
    add("The `k nearest` column falls well below chance while the `N directions` column sits "
        "near it, and that difference is not a contradiction — it is the mechanism. A token's "
        "ten nearest grid neighbours are mostly its own cell-mates, so they concentrate on one "
        "register; one token drawn from each of several *different* cells spreads back out.")
    add("")

    add("## Verdicts")
    add("")
    add("The proposal bundles two claims, and they do not have the same answer, so they are "
        "scored separately. `neighbourhood sampling` is a claim about **cells** — coherent "
        "registers per core, several registers within reach. `direction means register` is a "
        "claim about **placement** — the one the 2-D squash threatens, and the only one the "
        "arrangement of cells can be credited or blamed for.")
    add("")
    add("| grid | aggregate purity | neighbourhood sampling | direction means register | overall |")
    add("|---|---|---|---|---|")
    for r in results:
        v = grid_verdicts(mean_evaluation(r.select("annealed", condition)),
                          mean_evaluation(r.select("shuffled", condition)),
                          mean_evaluation(r.select("oracle-continents", condition)),
                          cosine[condition])
        add(f"| {r.spec.name} {r.spec.label} | {v['aggregate_purity']} | "
            f"{v['neighbourhood_sampling']} | {v['direction_means_register']} | "
            f"{v['overall']} |")
    add("")
    add("Bars, stated so they can be disagreed with: aggregate purity must keep **half** the "
        "cosine headroom; adjacency must carry **half** of it for direction to be said to mean "
        "register; neighbourhoods must return **half** the distinct sources chance would give. "
        "Every raw number is above, so a reader who prefers different bars can apply them.")
    add("")

    for r in results:
        add(f"## Grid {r.spec.name} ({r.spec.label}, {r.spec.n_cells} cells)")
        add("")
        add(f"{r.spec.note}. {r.n_layouts} layouts "
            f"({len(r.anneal_cost_start)} annealing runs), {r.seconds:.0f}s.")
        add("")
        if r.anneal_cost_start:
            s_mean, _s_sd, _s_lo, _s_hi = _spread(r.anneal_cost_start)
            e_mean, e_sd, e_lo, e_hi = _spread(r.anneal_cost_end)
            add(f"QAP cost from the spectral initialisation {s_mean:,.0f}, after annealing "
                f"{e_mean:,.0f} ± {e_sd:,.0f} (range {e_lo:,.0f} to {e_hi:,.0f}) — "
                f"a {100 * (1 - e_mean / s_mean):.1f}% improvement on the naive 2-D squash.")
            add("")
        add("| variant | condition | grid purity | floor | sigmas | off-cell | floor | sigmas | "
            "within-cell | N-direction diversity |")
        add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for cond in dict.fromkeys(e.condition for e in r.evaluations):
            for variant in ("annealed", "oracle-continents", "shuffled", "random-cells"):
                sel = r.select(variant, cond)
                if not sel:
                    continue
                pm, ps, _a, _b = _spread([e.purity_all.mean for e in sel])
                pf = float(np.mean([e.purity_all.permuted_mean for e in sel]))
                pfs = float(np.mean([e.purity_all.permuted_sd for e in sel]))
                om, os_, _c, _d = _spread([e.purity_offcell.mean for e in sel])
                of = float(np.mean([e.purity_offcell.permuted_mean for e in sel]))
                ofs = float(np.mean([e.purity_offcell.permuted_sd for e in sel]))
                wm, _ws, _e, _f = _spread([e.within_cell for e in sel])
                dm, _ds, _g, _h = _spread([e.diversity_directions for e in sel])
                add(f"| {variant} | {cond} | {pm:.4f} ± {ps:.4f} | {pf:.4f} | "
                    f"{(pm - pf) / pfs if pfs > 0 else float('inf'):.1f} | "
                    f"{om:.4f} ± {os_:.4f} | {of:.4f} | "
                    f"{(om - of) / ofs if ofs > 0 else float('inf'):.1f} | {wm:.4f} | "
                    f"{dm:.2f} |")
        add("")
        ann = r.select("annealed", condition)
        shuf = r.select("shuffled", condition)
        orc = r.select("oracle-continents", condition)
        if ann:
            add("### How far the register correlation reaches")
            add("")
            add("Fraction of labelled token pairs at *exactly* d hops that share a source — the "
                "same statistic with the ranking taken out of it: no ties, no k, no argument "
                "about cell-mates. Hop 0 is the within-cell number. The baseline is the chance "
                f"that two tokens drawn at random share a source, {ann[0].within_cell_chance:.4f}.")
            add("")
            add("| hops | annealed | oracle continents | shuffled | pairs (annealed) |")
            add("|---:|---:|---:|---:|---:|")
            for hop in range(len(ann[0].hop_profile)):
                pairs = int(np.mean([e.hop_profile[hop][2] for e in ann]))
                add(f"| {hop} | {_hop_mean(ann, hop):.4f} | {_hop_mean(orc, hop):.4f} | "
                    f"{_hop_mean(shuf, hop):.4f} | {pairs:,} |")
            add("")
        if r.grid_map:
            add("### The layout, by dominant source per cell")
            add("")
            add("Labelled tokens only (1,350 of 32,000), lowest-cost annealed layout on this "
                "grid. Continents mean high purity and synonymous neighbours; confetti means "
                "diverse neighbours and no usable direction. For looking at, not for deciding.")
            add("")
            add("```")
            lines.extend(r.grid_map)
            add("```")
            add("")

    return "\n".join(lines) + "\n"


def _reading_json(r: PurityReading) -> dict:
    return {"mean": r.mean, "sd": r.sd, "permuted_mean": r.permuted_mean,
            "permuted_sd": r.permuted_sd, "chance": r.chance,
            "sigmas_above_floor": None if r.sigmas == float("inf") else r.sigmas,
            "verdict": purity_verdict(r)}


def report_to_json(results: Sequence[GridResult], cosine: Dict[str, PurityReading], *,
                   hf_model: str, condition: str, k: int, n_directions: int, tie_breaks: int,
                   n_permutations: int, seed: int) -> dict:
    payload = {
        "hf_model": hf_model,
        "gate": "k-NN purity with grid distance substituted for cosine distance",
        "published_cosine_purity": PUBLISHED_COSINE_PURITY,
        "headline_condition": condition,
        "k": k,
        "n_directions": n_directions,
        "tie_breaks": tie_breaks,
        "n_permutations": n_permutations,
        "seed": seed,
        "cosine_baseline": {name: _reading_json(r) for name, r in cosine.items()},
        "grids": [],
    }
    for r in results:
        ann = r.select("annealed", condition)
        shuf = r.select("shuffled", condition)
        grid = {
            "name": r.spec.name,
            "width": r.spec.width,
            "height": r.spec.height,
            "cells": r.spec.n_cells,
            "n_chips": r.spec.n_chips,
            "torus": r.spec.torus,
            "note": r.spec.note,
            "n_layouts": r.n_layouts,
            "seconds": r.seconds,
            "qap_cost_start_mean": float(np.mean(r.anneal_cost_start)),
            "qap_cost_end_mean": float(np.mean(r.anneal_cost_end)),
            "verdicts": grid_verdicts(mean_evaluation(ann), mean_evaluation(shuf),
                                      mean_evaluation(r.select("oracle-continents", condition)),
                                      cosine[condition]),
            # The two headline retentions, precomputed so a reader does not have to reassemble
            # them from the variant table: how much of the cosine measurement's headroom above
            # its floor the grid keeps in aggregate, and how much of it survives as ADJACENCY
            # (annealed off-cell against the same cells randomly placed) — the pitch's claim.
            "aggregate_headroom_kept": headroom_retention(
                float(np.mean([e.purity_all.mean for e in ann])),
                float(np.mean([e.purity_all.permuted_mean for e in ann])), cosine[condition]),
            "adjacency_headroom_kept": headroom_retention(
                float(np.mean([e.purity_offcell.mean for e in ann])),
                float(np.mean([e.purity_offcell.mean for e in shuf])), cosine[condition]),
            "adjacency_headroom_oracle_ceiling": headroom_retention(
                float(np.mean([e.purity_offcell.mean
                               for e in r.select("oracle-continents", condition)])),
                float(np.mean([e.purity_offcell.mean for e in shuf])), cosine[condition]),
            "variants": {},
        }
        for cond in dict.fromkeys(e.condition for e in r.evaluations):
            for variant in ("annealed", "oracle-continents", "shuffled", "random-cells"):
                sel = r.select(variant, cond)
                if not sel:
                    continue
                pm, ps, plo, phi = _spread([e.purity_all.mean for e in sel])
                om, os_, olo, ohi = _spread([e.purity_offcell.mean for e in sel])
                dm, ds, dlo, dhi = _spread([e.diversity_directions for e in sel])
                km, ks, _klo, _khi = _spread([e.diversity_knn for e in sel])
                grid["variants"][f"{cond}/{variant}"] = {
                    "n_layouts": len(sel),
                    "purity_all": {"mean": pm, "sd": ps, "min": plo, "max": phi,
                                   "permuted_mean": float(np.mean(
                                       [e.purity_all.permuted_mean for e in sel])),
                                   "permuted_sd": float(np.mean(
                                       [e.purity_all.permuted_sd for e in sel])),
                                   "ratio_to_cosine": pm / cosine[cond].mean},
                    "purity_offcell": {"mean": om, "sd": os_, "min": olo, "max": ohi,
                                       "permuted_mean": float(np.mean(
                                           [e.purity_offcell.permuted_mean for e in sel])),
                                       "permuted_sd": float(np.mean(
                                           [e.purity_offcell.permuted_sd for e in sel]))},
                    "within_cell_purity": float(np.mean([e.within_cell for e in sel])),
                    "within_cell_chance": sel[0].within_cell_chance,
                    "diversity_knn": {"mean": km, "sd": ks,
                                      "chance": sel[0].diversity_knn_chance},
                    "diversity_directions": {"mean": dm, "sd": ds, "min": dlo, "max": dhi,
                                             "chance": sel[0].diversity_directions_chance},
                    "occupied_cells": sel[0].occupied_cells,
                    "tokens_per_cell_max": sel[0].tokens_per_cell_max,
                    "purity_by_hop": [
                        {"hops": hop,
                         "purity": float(np.nanmean([e.hop_profile[hop][1] for e in sel])),
                         "pairs": int(np.mean([e.hop_profile[hop][2] for e in sel]))}
                        for hop in range(len(sel[0].hop_profile))],
                }
        payload["grids"].append(grid)
    return payload


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------


def _default_model() -> Path:
    """The designated current model, matching ``probe_embedding_geography.py``."""
    try:
        from scripts.evaluate import load_designation

        return load_designation().hf_model
    except Exception:  # pragma: no cover - only when the designation file is broken
        return ROOT / "artifacts" / "hf-tt-tnt-1024a"


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-model", type=Path, default=None)
    p.add_argument("--embedding-key", type=str, default="model.embed_tokens.weight")
    p.add_argument("--corpus-dir", type=Path, default=ROOT / "artifacts" / "corpus")
    p.add_argument("--counts-cache", type=Path, default=None,
                   help="Optional .npy of the (n_sources, vocab) count matrix. Loaded if it "
                        "exists, written if it does not — the tokenisation is the only slow "
                        "part of this script and it is identical between runs.")
    p.add_argument("--words-per-source", type=int, default=DEFAULT_WORDS_PER_SOURCE)
    p.add_argument("--prior-strength", type=float, default=DEFAULT_PRIOR_STRENGTH)
    p.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT)
    p.add_argument("--per-source", type=int, default=DEFAULT_PER_SOURCE)
    p.add_argument("--exclude-top", type=int, default=DEFAULT_EXCLUDE_TOP)
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--n-directions", type=int, default=DEFAULT_N_DIRECTIONS)
    p.add_argument("--tie-breaks", type=int, default=DEFAULT_TIE_BREAKS)
    p.add_argument("--n-permutations", type=int, default=DEFAULT_N_PERMUTATIONS)
    p.add_argument("--n-clusterings", type=int, default=DEFAULT_N_CLUSTERINGS)
    p.add_argument("--n-anneals", type=int, default=DEFAULT_N_ANNEALS)
    p.add_argument("--kmeans-iters", type=int, default=DEFAULT_KMEANS_ITERS)
    p.add_argument("--anneal-steps", type=int, default=50_000,
                   help="Floor on annealing proposals per layout; the effective count is "
                        "max(this, 400 x cells).")
    p.add_argument("--grids", type=str, default="",
                   help="Comma-separated WxH list overriding the default substrates "
                        "(11x10,17x12,22x20,34x24).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--note", type=str, default="")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args(argv)


def _grids_from_arg(text: str) -> Tuple[GridSpec, ...]:
    if not text:
        return DEFAULT_GRIDS
    specs = []
    for chunk in text.split(","):
        w, _, h = chunk.strip().partition("x")
        specs.append(GridSpec(f"{w}x{h}", int(w), int(h)))
    return tuple(specs)


def main() -> int:
    args = _parse_args()
    hf_model = args.hf_model or _default_model()
    if not (hf_model / "config.json").is_file():
        print(f"ERROR: no converted model at {hf_model} (config.json missing).", file=sys.stderr)
        return 1

    print(f"reading embedding matrix from {hf_model} ...")
    embedding = load_embedding_matrix(hf_model, args.embedding_key)
    vocab_size, dim = embedding.shape
    print(f"  {vocab_size:,} x {dim}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(hf_model))
    source_names = sorted(SOURCES)

    if args.counts_cache is not None and args.counts_cache.is_file():
        counts = np.load(args.counts_cache)
        print(f"loaded cached counts {counts.shape} from {args.counts_cache}")
        if counts.shape != (len(source_names), vocab_size):
            print(f"ERROR: cached counts are {counts.shape}, expected "
                  f"{(len(source_names), vocab_size)}", file=sys.stderr)
            return 1
    else:
        if not args.corpus_dir.is_dir():
            print(f"ERROR: --corpus-dir {args.corpus_dir} is not a directory.", file=sys.stderr)
            return 1
        print(f"profiling {len(source_names)} sources ({args.words_per_source:,} words each) ...")
        counts = count_tokens_by_source(args.corpus_dir, source_names, tokenizer, vocab_size,
                                        words_per_source=args.words_per_source, log=print)
        if args.counts_cache is not None:
            args.counts_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.counts_cache, counts)

    z = log_odds_z(counts, args.prior_strength)
    conditions: Dict[str, TokenLabels] = {}
    for name, exclude_top in (("content", args.exclude_top), ("all", 0)):
        conditions[name] = characteristic_tokens(
            counts, z, source_names, min_count=args.min_count, per_source=args.per_source,
            exclude_top=exclude_top)
        print(f"  condition {name}: {conditions[name].token_ids.size} labelled tokens")

    cosine = {name: cosine_baseline(embedding, labels, args.k,
                                    n_permutations=args.n_permutations, seed=args.seed)
              for name, labels in conditions.items()}
    for name, reading in cosine.items():
        print(f"  cosine baseline {name}: {reading.mean:.4f} (floor {reading.permuted_mean:.4f}"
              f" ± {reading.permuted_sd:.4f}, {reading.sigmas:.1f} sigma)")

    unit = unit_rows(embedding.astype(np.float32))
    results: List[GridResult] = []
    for spec in _grids_from_arg(args.grids):
        print(f"grid {spec.name} {spec.label} ({spec.n_cells} cells, "
              f"torus={spec.torus}) ...")
        results.append(run_grid(
            spec, unit, conditions, n_clusterings=args.n_clusterings,
            n_anneals=args.n_anneals, kmeans_iters=args.kmeans_iters, k=args.k,
            tie_breaks=args.tie_breaks, n_permutations=args.n_permutations,
            n_directions=args.n_directions, anneal_steps=args.anneal_steps, seed=args.seed))

    out = args.out or (ROOT / "docs" / "measurements" / "grid-layout-gate-tt-tnt-1024a.md")
    json_out = args.json_out or (ROOT / "docs" / "measurements"
                                 / "grid-layout-gate-tt-tnt-1024a.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(
        results, cosine,
        hf_model=str(hf_model.relative_to(ROOT)) if hf_model.is_relative_to(ROOT)
        else str(hf_model),
        condition="content", k=args.k, n_directions=args.n_directions,
        tie_breaks=args.tie_breaks, n_permutations=args.n_permutations, note=args.note))
    print(f"wrote {out}")
    json_out.write_text(json.dumps(report_to_json(
        results, cosine, hf_model=str(hf_model), condition="content", k=args.k,
        n_directions=args.n_directions, tie_breaks=args.tie_breaks,
        n_permutations=args.n_permutations, seed=args.seed), indent=2))
    print(f"wrote {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
