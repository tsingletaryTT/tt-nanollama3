# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for scripts/probe_grid_layout.py.

Everything here runs without the model and without the corpus, on synthetic data with a KNOWN
answer. The bar is the one tests/test_probe_embedding_geography.py set: a gate that cannot
report failure is worthless, so the planted-null tests matter more than the planted-signal
ones. A layout tool that says "viable" whatever it is handed would have quietly approved a
kernel.

Three properties get the most attention, because they are the ones the report's conclusions
actually rest on:

- ``_swap_delta`` must equal a full recompute of :func:`qap_cost`. The annealer accepts or
  rejects millions of swaps on that delta alone; if it drifts, every cost figure is fiction.
- ``tie_broken_neighbours`` must randomise ties and never reorder genuinely different hop
  counts. Grid distance is an integer, so almost every neighbour is tied, and a fixed
  tie-break would let token id leak into the statistic.
- ``grid_verdicts`` must return REFUTED on a layout with no spatial structure, and must
  separate the two claims: cells can be register-coherent while placement means nothing.

The two tests that would need ``artifacts/hf-tt-tnt-1024a`` skip explicitly with a reason.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.probe_embedding_geography import TokenLabels  # noqa: E402
from scripts.probe_grid_layout import (  # noqa: E402
    DEFAULT_GRIDS,
    GridSpec,
    PurityReading,
    _spread,
    _swap_delta,
    anneal_placement,
    ascii_grid_map,
    balanced_assign,
    balanced_spherical_kmeans,
    cosine_baseline,
    distinct_sources_per_row,
    diversity_headroom,
    evaluate_layout,
    expected_distinct_sources,
    grid_distance_matrix,
    grid_verdicts,
    headroom_retention,
    kmeans_plusplus_cosine,
    load_embedding_matrix,
    mean_evaluation,
    oracle_continent_placement,
    pair_distances,
    purity_by_hop,
    purity_reading,
    purity_verdict,
    qap_cost,
    render_markdown,
    report_to_json,
    six_direction_diversity,
    spectral_grid_init,
    tie_broken_neighbours,
    token_positions,
    unit_rows,
    within_cell_purity,
)
from scripts.probe_grid_layout import GridResult  # noqa: E402

MODEL_DIR = ROOT / "artifacts" / "hf-tt-tnt-1024a"


# ---------------------------------------------------------------------------------------
# Synthetic fixtures with a known answer
# ---------------------------------------------------------------------------------------


def planted_layout(spec, n_sources=4, per_cell=6, blocked=True, seed=0):
    """A vocabulary laid out on ``spec`` with a known source structure.

    ``blocked`` puts each source in one contiguous band of grid positions -- perfect cells AND
    meaningful adjacency. ``blocked=False`` keeps the cells perfectly pure but scatters them
    over the grid at random, which is the case that must score high on within-cell purity and
    at the floor on adjacency.
    """
    rng = np.random.default_rng(seed)
    n_cells = spec.n_cells
    vocab = n_cells * per_cell
    assignment = np.repeat(np.arange(n_cells), per_cell)
    placement = (np.arange(n_cells) if blocked else rng.permutation(n_cells))
    cell_source = (np.arange(n_cells) * n_sources) // n_cells
    labels = TokenLabels(
        token_ids=np.arange(vocab),
        labels=cell_source[assignment],
        source_names=tuple(f"s{i}" for i in range(n_sources)),
        z_scores=np.zeros(vocab),
        total_counts=np.full(vocab, 100),
    )
    return assignment, placement, labels


def structureless_layout(spec, n_sources=4, per_cell=6, seed=1):
    """Same cells and same class sizes, but labels unrelated to position: the null."""
    assignment, placement, labels = planted_layout(spec, n_sources, per_cell, seed=seed)
    shuffled = np.random.default_rng(seed).permutation(labels.labels)
    return assignment, placement, TokenLabels(
        token_ids=labels.token_ids, labels=shuffled, source_names=labels.source_names,
        z_scores=labels.z_scores, total_counts=labels.total_counts)


# ---------------------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------------------


def test_torus_distance_wraps_and_open_mesh_does_not():
    torus = grid_distance_matrix(GridSpec("t", 11, 10, torus=True))
    mesh = grid_distance_matrix(GridSpec("m", 11, 10, torus=False))
    # Columns 0 and 10 of row 0 are 10 apart on a mesh and 1 apart on a torus.
    assert mesh[0, 10] == pytest.approx(10.0)
    assert torus[0, 10] == pytest.approx(1.0)
    assert torus.max() == pytest.approx(5.0 + 5.0), "11x10 torus diameter is floor(11/2)+5"
    assert mesh.max() == pytest.approx(10.0 + 9.0)


def test_grid_distance_is_a_metric_shaped_matrix():
    dist = grid_distance_matrix(GridSpec("t", 5, 4))
    assert dist.shape == (20, 20)
    assert np.allclose(dist, dist.T)
    assert np.all(np.diag(dist) == 0.0)
    assert np.all(dist[np.triu_indices(20, 1)] > 0), "distinct cells are never zero apart"


def test_grid_distance_rejects_a_degenerate_grid():
    with pytest.raises(ValueError, match="at least 1x1"):
        grid_distance_matrix(GridSpec("bad", 0, 4))


def test_default_grids_describe_the_real_substrates():
    cells = {spec.n_cells for spec in DEFAULT_GRIDS}
    assert 110 in cells, "the harvested p300c exposes 11x10 = 110 functional Tensix"
    assert 204 in cells, "the full Blackhole die is 17x12"
    assert any(spec.n_chips > 1 for spec in DEFAULT_GRIDS), "a multi-chip option must be tried"


# ---------------------------------------------------------------------------------------
# Balanced spherical k-means
# ---------------------------------------------------------------------------------------


def test_balanced_assign_never_exceeds_capacity_and_places_everyone():
    rng = np.random.default_rng(0)
    similarity = rng.normal(size=(50, 5))
    assignment = balanced_assign(similarity, capacity=10)
    assert assignment.shape == (50,)
    assert np.all(assignment >= 0)
    assert np.all(np.bincount(assignment, minlength=5) == 10)


def test_balanced_assign_gives_the_strongest_preference_its_first_choice():
    # Point 0 overwhelmingly prefers cluster 0; points 1 and 2 are nearly indifferent.
    similarity = np.array([[9.0, 0.0], [0.51, 0.5], [0.5, 0.51]])
    assignment = balanced_assign(similarity, capacity=2)
    assert assignment[0] == 0


def test_balanced_assign_refuses_an_impossible_capacity():
    with pytest.raises(ValueError, match="cannot hold"):
        balanced_assign(np.zeros((10, 2)), capacity=4)


def test_kmeans_plusplus_picks_distinct_spread_out_seeds():
    rng = np.random.default_rng(0)
    points = unit_rows(np.concatenate([np.eye(4), np.eye(4) + 0.01 * rng.normal(size=(4, 4))]))
    centroids = kmeans_plusplus_cosine(points, 4, np.random.default_rng(3))
    assert centroids.shape == (4, 4)
    similarity = centroids @ centroids.T
    np.fill_diagonal(similarity, -np.inf)
    assert similarity.max() < 0.9, "seeds must not land on top of each other"


def test_kmeans_plusplus_validates_the_cluster_count():
    with pytest.raises(ValueError, match="n_clusters"):
        kmeans_plusplus_cosine(unit_rows(np.eye(3)), 4, np.random.default_rng(0))


def test_balanced_kmeans_recovers_planted_clusters_and_stays_balanced():
    rng = np.random.default_rng(0)
    centres = np.eye(4, 12)
    points = unit_rows(np.repeat(centres, 15, axis=0) + rng.normal(0, 0.02, (60, 12)))
    assignment, centroids = balanced_spherical_kmeans(points, 4, seed=0, iterations=8)
    assert centroids.shape == (4, 12)
    assert np.all(np.bincount(assignment, minlength=4) == 15), "cells must hold equal load"
    truth = np.repeat(np.arange(4), 15)
    # Recovery up to a relabelling: every planted cluster lands in exactly one cell.
    for c in range(4):
        assert np.unique(assignment[truth == c]).size == 1


def test_balanced_kmeans_centroids_are_on_the_unit_sphere():
    rng = np.random.default_rng(1)
    points = unit_rows(rng.normal(size=(40, 6)))
    _assignment, centroids = balanced_spherical_kmeans(points, 4, seed=0, iterations=3)
    assert np.allclose(np.linalg.norm(centroids, axis=1), 1.0)


# ---------------------------------------------------------------------------------------
# Placement: initialisation, cost, and the delta the annealer trusts
# ---------------------------------------------------------------------------------------


def test_spectral_init_is_a_permutation_of_the_grid_positions():
    rng = np.random.default_rng(0)
    spec = GridSpec("g", 5, 4)
    centroids = unit_rows(rng.normal(size=(20, 8)))
    placement = spectral_grid_init(centroids, spec)
    assert sorted(placement.tolist()) == list(range(20))


def test_spectral_init_requires_one_cluster_per_cell():
    with pytest.raises(ValueError, match="clusters for"):
        spectral_grid_init(unit_rows(np.random.default_rng(0).normal(size=(7, 3))),
                           GridSpec("g", 5, 4))


def test_swap_delta_equals_a_full_recompute():
    """The annealer's whole correctness rests on this identity."""
    rng = np.random.default_rng(0)
    spec = GridSpec("g", 4, 3)
    distance = grid_distance_matrix(spec)
    similarity = rng.normal(size=(12, 12))
    similarity = similarity + similarity.T
    placement = rng.permutation(12)
    for a, b in [(0, 1), (3, 7), (2, 11), (5, 5)]:
        before = qap_cost(similarity, distance, placement)
        delta = _swap_delta(similarity, distance, placement, a, b)
        swapped = placement.copy()
        swapped[a], swapped[b] = swapped[b], swapped[a]
        after = qap_cost(similarity, distance, swapped)
        assert delta == pytest.approx(after - before, abs=1e-8)


def test_qap_cost_prefers_similar_clusters_to_be_close():
    spec = GridSpec("row", 4, 1)
    distance = grid_distance_matrix(spec)
    # Clusters 0,1 are similar and 2,3 are similar; nothing else is.
    similarity = np.array([[1.0, 1.0, 0.0, 0.0],
                           [1.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 1.0],
                           [0.0, 0.0, 1.0, 1.0]])
    good = qap_cost(similarity, distance, np.array([0, 1, 2, 3]))
    bad = qap_cost(similarity, distance, np.array([0, 2, 1, 3]))
    assert good < bad


def test_annealing_returns_a_permutation_and_never_a_worse_cost():
    rng = np.random.default_rng(0)
    spec = GridSpec("g", 5, 4)
    distance = grid_distance_matrix(spec)
    centroids = unit_rows(rng.normal(size=(20, 6)))
    similarity = centroids @ centroids.T
    init = spectral_grid_init(centroids, spec)
    placement, start, end = anneal_placement(similarity, distance, init, n_steps=4000, seed=0,
                                             log=None)
    assert sorted(placement.tolist()) == list(range(20))
    assert end <= start, "the best-seen placement can never be worse than the start"
    assert qap_cost(similarity, distance, placement) == pytest.approx(end, rel=1e-6)


def test_annealing_solves_a_layout_whose_answer_is_known():
    """Four clusters on a 1-D row, similar in a chain: annealing must recover the chain order."""
    spec = GridSpec("row", 6, 1)
    distance = grid_distance_matrix(spec)
    order = np.arange(6)
    similarity = 1.0 / (1.0 + np.abs(order[:, None] - order[None, :]))
    rng = np.random.default_rng(0)
    placement, _start, _end = anneal_placement(similarity, distance, rng.permutation(6),
                                               n_steps=20_000, seed=1, log=None)
    # On a 6-cell torus the chain can be recovered in either direction and any rotation; what
    # must hold is that consecutive clusters end up adjacent.
    hops = [distance[placement[i], placement[i + 1]] for i in range(5)]
    assert max(hops) <= 1.0, f"chain neighbours ended up {hops} hops apart"


def test_annealing_validates_its_budget():
    with pytest.raises(ValueError, match="n_steps"):
        anneal_placement(np.eye(4), grid_distance_matrix(GridSpec("g", 2, 2)),
                         np.arange(4), n_steps=0, log=None)


def test_oracle_placement_makes_same_source_cells_contiguous():
    spec = GridSpec("g", 4, 3)
    assignment, _placement, labels = planted_layout(spec, n_sources=3, per_cell=4)
    oracle = oracle_continent_placement(assignment, labels, spec)
    assert sorted(oracle.tolist()) == list(range(12))
    distance = grid_distance_matrix(spec)
    positions = token_positions(assignment, oracle, labels.token_ids)
    # Every source's cells form one run in boustrophedon order, so its span is tight.
    for source in range(3):
        cells = np.unique(positions[labels.labels == source])
        assert distance[np.ix_(cells, cells)].max() <= 3.0


# ---------------------------------------------------------------------------------------
# The statistics
# ---------------------------------------------------------------------------------------


def test_tie_broken_neighbours_excludes_self_and_randomises_ties():
    spec = GridSpec("g", 4, 4)
    distance = grid_distance_matrix(spec)
    positions = np.repeat(np.arange(16), 4)          # 4 tokens per cell: massive ties
    pair = pair_distances(positions, distance)
    first = tie_broken_neighbours(pair, 3, np.random.default_rng(0))
    second = tie_broken_neighbours(pair, 3, np.random.default_rng(1))
    for i, row in enumerate(first):
        assert i not in row.tolist()
    assert not np.array_equal(first, second), "ties must not resolve the same way every draw"


def test_tie_break_jitter_never_reorders_different_hop_counts():
    spec = GridSpec("g", 6, 6)
    distance = grid_distance_matrix(spec)
    positions = np.arange(36)
    pair = pair_distances(positions, distance)
    for seed in range(5):
        nb = tie_broken_neighbours(pair, 4, np.random.default_rng(seed))
        for i, row in enumerate(nb):
            chosen = pair[i, row]
            # Everything chosen must be no further than the closest thing NOT chosen.
            rest = np.delete(pair[i], np.append(row, i))
            assert chosen.max() <= rest.min()


def test_off_cell_neighbours_exclude_cell_mates():
    spec = GridSpec("g", 4, 4)
    distance = grid_distance_matrix(spec)
    positions = np.repeat(np.arange(16), 3)
    pair = pair_distances(positions, distance)
    nb = tie_broken_neighbours(pair, 3, np.random.default_rng(0), exclude_same_cell=True)
    assert np.all(pair[np.arange(48)[:, None], nb] > 0), "a cell-mate slipped through"


def test_tie_broken_neighbours_rejects_an_unsatisfiable_k():
    pair = pair_distances(np.arange(4), grid_distance_matrix(GridSpec("g", 2, 2)))
    with pytest.raises(ValueError, match="k must be"):
        tie_broken_neighbours(pair, 0, np.random.default_rng(0))
    with pytest.raises(ValueError, match="usable neighbours"):
        tie_broken_neighbours(pair, 9, np.random.default_rng(0))


def test_within_cell_purity_is_one_for_pure_cells_and_chance_for_scrambled_ones():
    positions = np.repeat(np.arange(4), 5)
    pure = np.repeat(np.arange(4), 5)
    value, chance, pairs = within_cell_purity(positions, pure)
    assert value == pytest.approx(1.0)
    assert chance == pytest.approx(0.25)
    assert pairs == 4 * (5 * 4 // 2)

    scrambled = np.tile(np.arange(4), 5)             # every cell holds one of each source
    value, _chance, _pairs = within_cell_purity(positions, scrambled)
    assert value < 0.3


def test_purity_by_hop_decays_from_one_on_a_blocked_layout_and_is_flat_on_the_null():
    spec = GridSpec("g", 8, 8)
    distance = grid_distance_matrix(spec)

    assignment, placement, labels = planted_layout(spec, n_sources=4, per_cell=4)
    pair = pair_distances(token_positions(assignment, placement, labels.token_ids), distance)
    profile = purity_by_hop(pair, labels.labels, max_hops=4)
    assert [hops for hops, _p, _n in profile] == [0, 1, 2, 3, 4]
    assert profile[0][1] == pytest.approx(1.0), "cell-mates share a source by construction"
    assert profile[1][1] > profile[4][1], "correlation must fall off with distance"
    assert all(pairs > 0 for _h, _p, pairs in profile)

    assignment, placement, labels = structureless_layout(spec, n_sources=4, per_cell=4)
    pair = pair_distances(token_positions(assignment, placement, labels.token_ids), distance)
    flat = purity_by_hop(pair, labels.labels, max_hops=4)
    values = np.array([p for _h, p, _n in flat])
    assert np.all(np.abs(values - 0.25) < 0.06), "no structure means no decay to find"


def test_purity_by_hop_reports_nan_rather_than_inventing_a_number():
    spec = GridSpec("g", 3, 3)
    distance = grid_distance_matrix(spec)
    pair = pair_distances(np.array([0, 0, 1]), distance)   # nothing is 4 hops away
    profile = purity_by_hop(pair, np.array([0, 0, 1]), max_hops=4)
    assert np.isnan(profile[4][1])
    assert profile[4][2] == 0


def test_expected_distinct_sources_matches_the_closed_form():
    labels = np.repeat(np.arange(4), 10)             # balanced, p = 1/4
    assert expected_distinct_sources(labels, 1) == pytest.approx(1.0)
    assert expected_distinct_sources(labels, 2) == pytest.approx(4 * (1 - 0.75 ** 2))
    assert expected_distinct_sources(labels, 500) == pytest.approx(4.0, abs=1e-6)
    with pytest.raises(ValueError, match="n_draws"):
        expected_distinct_sources(labels, 0)


def test_distinct_sources_per_row_counts_uniques():
    neighbours = np.array([[0, 1, 2], [0, 0, 0]])
    labels = np.array([0, 1, 1])
    assert distinct_sources_per_row(neighbours, labels) == pytest.approx((2 + 1) / 2)


def test_six_direction_diversity_is_high_when_neighbours_differ_and_one_for_a_blob():
    spec = GridSpec("g", 6, 6)
    distance = grid_distance_matrix(spec)
    positions = np.arange(36)                     # one token per cell
    rng = np.random.default_rng(0)

    interleaved = np.arange(36) % 4               # neighbouring cells are different sources
    assert six_direction_diversity(positions, interleaved, distance, 4, rng) > 2.5

    one_source = np.zeros(36, dtype=np.int64)     # every cell the same source
    assert six_direction_diversity(positions, one_source, distance, 4, rng) == pytest.approx(1.0)


def test_diversity_headroom_puts_six_synonyms_at_zero_and_chance_at_one():
    assert diversity_headroom(1.0, 4.56) == pytest.approx(0.0)
    assert diversity_headroom(4.56, 4.56) == pytest.approx(1.0)
    assert diversity_headroom(2.78, 4.56) == pytest.approx(0.5)
    assert diversity_headroom(1.0, 1.0) == 0.0, "a one-source corpus has no headroom to keep"


def test_six_direction_diversity_refuses_more_directions_than_cells():
    distance = grid_distance_matrix(GridSpec("g", 2, 2))
    with pytest.raises(ValueError, match="distinct directions"):
        six_direction_diversity(np.arange(4), np.arange(4), distance, 6,
                                np.random.default_rng(0))


def test_purity_reading_is_high_on_a_blocked_layout_and_at_the_floor_on_the_null():
    spec = GridSpec("g", 6, 6)
    distance = grid_distance_matrix(spec)

    assignment, placement, labels = planted_layout(spec, n_sources=4, per_cell=5)
    pair = pair_distances(token_positions(assignment, placement, labels.token_ids), distance)
    good = purity_reading("planted", pair, labels.labels, 5, tie_breaks=2, n_permutations=30,
                          rng=np.random.default_rng(0))
    assert good.mean > 0.9
    assert good.sigmas > 10.0
    assert purity_verdict(good) == "above floor"

    assignment, placement, labels = structureless_layout(spec, n_sources=4, per_cell=5)
    pair = pair_distances(token_positions(assignment, placement, labels.token_ids), distance)
    null = purity_reading("null", pair, labels.labels, 5, tie_breaks=2, n_permutations=30,
                          rng=np.random.default_rng(0))
    assert null.sigmas < 3.0, "labels unrelated to position must not look like a geography"
    assert purity_verdict(null).startswith("NOT INTERPRETABLE")


def test_headroom_retention_prices_the_floor_out_of_both_sides():
    cosine = PurityReading("c", mean=0.55, sd=0.0, permuted_mean=0.11, permuted_sd=0.003,
                           chance=0.11)
    assert headroom_retention(0.55, 0.11, cosine) == pytest.approx(1.0)
    assert headroom_retention(0.11, 0.11, cosine) == pytest.approx(0.0)
    assert headroom_retention(0.33, 0.11, cosine) == pytest.approx(0.5)


# ---------------------------------------------------------------------------------------
# The verdicts -- the part that must be able to say no
# ---------------------------------------------------------------------------------------


def _evaluate(spec, assignment, placement, labels, variant):
    return evaluate_layout(variant, "content", spec, grid_distance_matrix(spec), assignment,
                           placement, labels, k=5, tie_breaks=2, n_permutations=30,
                           n_directions=4, seed=0)


def test_verdicts_support_a_layout_that_really_has_directional_structure():
    spec = GridSpec("g", 6, 6)
    assignment, blocked, labels = planted_layout(spec, n_sources=4, per_cell=5, blocked=True)
    _a, scattered, _l = planted_layout(spec, n_sources=4, per_cell=5, blocked=False, seed=4)
    cosine = PurityReading("c", mean=1.0, sd=0.0, permuted_mean=0.25, permuted_sd=0.01,
                           chance=0.25)
    verdicts = grid_verdicts(_evaluate(spec, assignment, blocked, labels, "annealed"),
                             _evaluate(spec, assignment, scattered, labels, "shuffled"),
                             _evaluate(spec, assignment,
                                       oracle_continent_placement(assignment, labels, spec),
                                       labels, "oracle-continents"),
                             cosine)
    assert verdicts["aggregate_purity"].startswith("SURVIVES")
    assert verdicts["direction_means_register"].startswith("SUPPORTED")


def test_verdicts_refute_direction_when_the_cells_are_scattered():
    """The case this whole script exists to detect: good cells, meaningless placement."""
    spec = GridSpec("g", 6, 6)
    assignment, _blocked, labels = planted_layout(spec, n_sources=4, per_cell=5)
    scattered_a = planted_layout(spec, 4, 5, blocked=False, seed=2)[1]
    scattered_b = planted_layout(spec, 4, 5, blocked=False, seed=3)[1]
    cosine = PurityReading("c", mean=1.0, sd=0.0, permuted_mean=0.25, permuted_sd=0.01,
                           chance=0.25)
    annealed = _evaluate(spec, assignment, scattered_a, labels, "annealed")
    shuffled = _evaluate(spec, assignment, scattered_b, labels, "shuffled")
    oracle = _evaluate(spec, assignment, oracle_continent_placement(assignment, labels, spec),
                       labels, "oracle-continents")
    verdicts = grid_verdicts(annealed, shuffled, oracle, cosine)
    assert verdicts["direction_means_register"].startswith("REFUTED")
    assert verdicts["overall"].startswith("NOT VIABLE")
    # ... while the cells themselves are still perfectly register-coherent.
    assert annealed.within_cell == pytest.approx(1.0)
    assert verdicts["neighbourhood_sampling"].startswith("SUPPORTED")


def test_verdicts_refute_neighbourhood_sampling_when_the_layout_is_two_continents():
    """The 'one big blob' failure: perfect purity by construction, and six synonyms.

    Two sources laid down as two contiguous halves of a 12x12 torus. Every cell is perfectly
    register-coherent and adjacency is maximally informative, so the purity clauses pass — and
    the proposal still fails, because only the two boundary rows have a neighbour of the other
    register and every other neighbourhood is one register repeated.

    The grid has to be this large for the failure to be visible, which is itself the point: on
    a small grid the continents are only a couple of cells across and even a blob layout keeps
    its neighbourhood diversity. That is not a quirk of the test — it is the reason the real
    110-cell result passes this clause.
    """
    spec = GridSpec("g", 12, 12)
    assignment, placement, labels = planted_layout(spec, n_sources=2, per_cell=3)
    cosine = PurityReading("c", mean=1.0, sd=0.0, permuted_mean=0.5, permuted_sd=0.01,
                           chance=0.5)
    annealed = _evaluate(spec, assignment, placement, labels, "annealed")
    scattered = planted_layout(spec, 2, 3, blocked=False, seed=6)[1]
    shuffled = _evaluate(spec, assignment, scattered, labels, "shuffled")
    verdicts = grid_verdicts(annealed, shuffled, annealed, cosine)
    assert annealed.within_cell == pytest.approx(1.0), "cells really are perfectly coherent"
    assert verdicts["aggregate_purity"].startswith("SURVIVES")
    assert verdicts["neighbourhood_sampling"].startswith("REFUTED")
    assert "synonyms" in verdicts["neighbourhood_sampling"]
    assert verdicts["overall"].startswith("NOT VIABLE")


def test_verdicts_report_not_interpretable_when_there_is_nothing_there():
    spec = GridSpec("g", 6, 6)
    assignment, placement, labels = structureless_layout(spec, n_sources=4, per_cell=5)
    cosine = PurityReading("c", mean=1.0, sd=0.0, permuted_mean=0.25, permuted_sd=0.01,
                           chance=0.25)
    evaluation = _evaluate(spec, assignment, placement, labels, "annealed")
    verdicts = grid_verdicts(evaluation, evaluation, evaluation, cosine)
    assert verdicts["aggregate_purity"].startswith("NOT INTERPRETABLE")
    assert verdicts["overall"].startswith("NOT VIABLE")


# ---------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------


def _one_grid_result(spec):
    assignment, placement, labels = planted_layout(spec, n_sources=4, per_cell=5)
    scattered = planted_layout(spec, 4, 5, blocked=False, seed=9)[1]
    result = GridResult(spec=spec, n_layouts=1)
    result.evaluations = [
        _evaluate(spec, assignment, placement, labels, "annealed"),
        _evaluate(spec, assignment, oracle_continent_placement(assignment, labels, spec),
                  labels, "oracle-continents"),
        _evaluate(spec, assignment, scattered, labels, "shuffled"),
        _evaluate(spec, assignment, scattered, labels, "random-cells"),
    ]
    result.anneal_cost_start = [100.0]
    result.anneal_cost_end = [90.0]
    result.grid_map = ascii_grid_map(token_positions(assignment, placement, labels.token_ids),
                                     labels.labels, labels.source_names, spec)
    return result


def test_ascii_grid_map_has_one_glyph_per_cell_and_no_right_hand_border():
    spec = GridSpec("g", 6, 5)
    assignment, placement, labels = planted_layout(spec, n_sources=4, per_cell=5)
    lines = ascii_grid_map(token_positions(assignment, placement, labels.token_ids),
                           labels.labels, labels.source_names, spec)
    rows = lines[:spec.height]
    assert len(rows) == spec.height
    assert not any(line.endswith(" ") for line in lines), "no trailing right-hand border"
    assert "legend:" in lines[-1]


def test_ascii_grid_map_marks_empty_cells_rather_than_inventing_a_source():
    spec = GridSpec("g", 3, 2)
    labels = TokenLabels(token_ids=np.array([0]), labels=np.array([0]),
                         source_names=("a", "b"), z_scores=np.zeros(1),
                         total_counts=np.array([1]))
    lines = ascii_grid_map(np.array([0]), labels.labels, labels.source_names, spec)
    assert "." in "".join(lines[:2]), "cells with no labelled token must show as empty"


def test_markdown_report_names_every_grid_and_both_verdict_claims():
    spec = GridSpec("g", 6, 6)
    results = [_one_grid_result(spec)]
    cosine = {"content": PurityReading("c", 1.0, 0.0, 0.25, 0.01, 0.25)}
    text = render_markdown(results, cosine, hf_model="artifacts/fake", condition="content",
                           k=5, n_directions=4, tie_breaks=2, n_permutations=30)
    assert "Grid-distance gate" in text
    assert "adjacency headroom" in text
    assert "neighbourhood sampling" in text
    assert "oracle" in text.lower()
    assert spec.label in text


def test_json_report_round_trips_and_carries_the_two_retentions():
    spec = GridSpec("g", 6, 6)
    results = [_one_grid_result(spec)]
    cosine = {"content": PurityReading("c", 1.0, 0.0, 0.25, 0.01, 0.25)}
    payload = report_to_json(results, cosine, hf_model="fake", condition="content", k=5,
                             n_directions=4, tie_breaks=2, n_permutations=30, seed=0)
    text = json.dumps(payload)
    assert json.loads(text) == payload
    grid = payload["grids"][0]
    assert "aggregate_headroom_kept" in grid
    assert "adjacency_headroom_kept" in grid
    assert "content/oracle-continents" in grid["variants"]
    assert set(grid["verdicts"]) >= {"aggregate_purity", "neighbourhood_sampling",
                                     "direction_means_register", "overall"}


def test_mean_evaluation_averages_layouts_rather_than_picking_one():
    """A verdict must read the distribution; one lucky restart must not decide a kernel."""
    spec = GridSpec("g", 6, 6)
    assignment, blocked, labels = planted_layout(spec, n_sources=4, per_cell=5)
    scattered = planted_layout(spec, 4, 5, blocked=False, seed=8)[1]
    good = _evaluate(spec, assignment, blocked, labels, "annealed")
    bad = _evaluate(spec, assignment, scattered, labels, "annealed")
    averaged = mean_evaluation([good, bad])
    assert averaged.purity_offcell.mean == pytest.approx(
        (good.purity_offcell.mean + bad.purity_offcell.mean) / 2)
    assert averaged.purity_offcell.sd > 0.0, "sd must become the spread across layouts"
    assert averaged.hop_profile[1][1] == pytest.approx(
        (good.hop_profile[1][1] + bad.hop_profile[1][1]) / 2)
    with pytest.raises(ValueError, match="zero layouts"):
        mean_evaluation([])


def test_spread_reports_the_distribution_not_the_best():
    mean, sd, low, high = _spread([1.0, 2.0, 3.0])
    assert (mean, low, high) == (2.0, 1.0, 3.0)
    assert sd > 0.0
    assert _spread([5.0]) == (5.0, 0.0, 5.0, 5.0)


# ---------------------------------------------------------------------------------------
# The two tests that need the model, skipped explicitly when it is absent
# ---------------------------------------------------------------------------------------


@pytest.mark.skipif(not (MODEL_DIR / "model.safetensors").is_file(),
                    reason="artifacts/hf-tt-tnt-1024a is an optional local artifact, not "
                           "repository content; the rest of this suite covers the layout "
                           "and the statistics on synthetic data")
def test_embedding_matrix_is_the_shape_the_layout_assumes():
    embedding = load_embedding_matrix(MODEL_DIR)
    assert embedding.shape[0] == 32000
    assert embedding.ndim == 2


@pytest.mark.skipif(not (MODEL_DIR / "model.safetensors").is_file(),
                    reason="artifacts/hf-tt-tnt-1024a is an optional local artifact, not "
                           "repository content")
def test_cosine_baseline_reproduces_the_published_headline():
    """This script's baseline must match probe_embedding_geography.py's, or its ratio is void."""
    from scripts.probe_embedding_geography import characteristic_tokens, log_odds_z

    counts_path = ROOT / "scratch" / "grid-gate" / "counts.npy"
    if not counts_path.is_file():
        pytest.skip("no cached corpus profile; the corpus is an optional local artifact")
    counts = np.load(counts_path)
    from train.corpus import SOURCES

    z = log_odds_z(counts, 1000.0)
    labels = characteristic_tokens(counts, z, sorted(SOURCES), min_count=25, per_source=150,
                                   exclude_top=500)
    reading = cosine_baseline(load_embedding_matrix(MODEL_DIR), labels, 10,
                              n_permutations=50, seed=0)
    assert reading.mean == pytest.approx(0.5458, abs=0.002)
