# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for scripts/probe_embedding_geography.py.

Everything here runs without the model and without the corpus. The statistic, the token
selection, the k-NN/silhouette/probe machinery, the PCA projection and the rendering are all
exercised on synthetic data with a KNOWN answer -- separable clusters must score high, and
random labels over the same points must score at chance, because a separation statistic that
cannot report "no separation" is worthless for the question this script exists to answer.

The two tests that would need `artifacts/hf-tt-tnt-1024a` or `artifacts/corpus` skip
explicitly with a reason (they are optional local artifacts, not repository contents), matching
tests/test_probe_context_use.py's convention that this suite must pass non-vacuously on a
machine that has neither.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.probe_embedding_geography import (  # noqa: E402
    ProbeResult,
    SeparationResult,
    TokenLabels,
    _default_output_paths,
    ascii_scatter,
    chance_accuracy,
    characteristic_tokens,
    count_tokens_by_source,
    evaluate_separation,
    frequency_features,
    linear_probe,
    load_embedding_matrix,
    log_odds_z,
    neighbour_confusion,
    neighbour_indices,
    pca_2d,
    per_class_purity,
    permuted_purity,
    purity_from_neighbours,
    read_corpus_text,
    render_markdown,
    report_to_json,
    silhouette_cosine,
    verdict,
)

MODEL_DIR = ROOT / "artifacts" / "hf-tt-tnt-1024a"
CORPUS_DIR = ROOT / "artifacts" / "corpus"


# ---------------------------------------------------------------------------------------
# Synthetic fixtures with a known answer
# ---------------------------------------------------------------------------------------


def separable_clusters(n_classes=3, per_class=40, dim=8, spread=0.05, seed=0):
    """Tight clusters around orthogonal directions: the "obvious geography" case."""
    rng = np.random.default_rng(seed)
    centres = np.eye(n_classes, dim)
    features = np.repeat(centres, per_class, axis=0) + rng.normal(0, spread,
                                                                  (n_classes * per_class, dim))
    labels = np.repeat(np.arange(n_classes), per_class)
    return features, labels


def structureless(n_classes=3, per_class=40, dim=8, seed=1):
    """Points with real structure but labels unrelated to it: the "no geography" case."""
    features, _ = separable_clusters(n_classes, per_class, dim, seed=seed)
    labels = np.random.default_rng(seed).integers(0, n_classes, features.shape[0])
    return features, labels


# ---------------------------------------------------------------------------------------
# log_odds_z -- the characteristicness statistic
# ---------------------------------------------------------------------------------------


def test_log_odds_z_ranks_an_exclusive_token_above_a_shared_one():
    # token 0 only ever appears in source 0; token 1 is split evenly; token 2 is source 1's.
    counts = np.array([[500, 500, 0],
                       [0, 500, 500]], dtype=np.int64)
    z = log_odds_z(counts, prior_strength=10.0)
    assert z[0, 0] > z[0, 1] > z[0, 2]
    assert z[1, 2] > z[1, 1] > z[1, 0]


def test_log_odds_z_is_near_zero_for_a_token_the_sources_share_equally():
    counts = np.array([[400, 600], [400, 600]], dtype=np.int64)
    z = log_odds_z(counts, prior_strength=10.0)
    assert np.all(np.abs(z) < 1e-6)


def test_log_odds_z_prefers_the_better_attested_of_two_equally_skewed_tokens():
    """The whole reason this statistic was chosen over tf-idf: evidence counts, not just ratio."""
    counts = np.array([[1000, 30, 500],
                       [0, 0, 500]], dtype=np.int64)
    z = log_odds_z(counts, prior_strength=10.0)
    assert z[0, 0] > z[0, 1], "a 1000-count exclusive token must outrank a 30-count one"


def test_log_odds_z_marks_unattested_tokens_minus_inf_not_nan():
    counts = np.array([[10, 0], [20, 0]], dtype=np.int64)
    z = log_odds_z(counts, prior_strength=10.0)
    assert np.all(np.isneginf(z[:, 1]))
    assert not np.any(np.isnan(z))


def test_log_odds_z_rejects_degenerate_input():
    with pytest.raises(ValueError, match="at least 2 sources"):
        log_odds_z(np.array([[1, 2, 3]]))
    with pytest.raises(ValueError, match="prior_strength"):
        log_odds_z(np.array([[1, 2], [3, 4]]), prior_strength=0.0)
    with pytest.raises(ValueError, match="all zero"):
        log_odds_z(np.zeros((2, 3), dtype=np.int64))


# ---------------------------------------------------------------------------------------
# characteristic_tokens -- selection, disjointness, filters
# ---------------------------------------------------------------------------------------


def test_characteristic_tokens_are_disjoint_and_capped_per_source():
    rng = np.random.default_rng(0)
    counts = rng.integers(30, 400, size=(3, 60))
    z = log_odds_z(counts)
    labels = characteristic_tokens(counts, z, ["a", "b", "c"], min_count=1, per_source=5)
    assert len(labels.token_ids) == 15
    assert len(set(labels.token_ids.tolist())) == 15, "a token must belong to at most one source"
    assert sorted(np.bincount(labels.labels).tolist()) == [5, 5, 5]


def test_characteristic_tokens_respects_min_count():
    counts = np.array([[100, 5, 100], [1, 100, 100]], dtype=np.int64)
    z = log_odds_z(counts)
    labels = characteristic_tokens(counts, z, ["a", "b"], min_count=50, per_source=10)
    # token 1 has only 5 occurrences in source a, so a may not claim it.
    claimed_by_a = labels.token_ids[labels.labels == 0]
    assert 1 not in claimed_by_a.tolist()


def test_characteristic_tokens_excludes_the_globally_most_frequent():
    counts = np.array([[10_000, 100, 60], [9_000, 60, 100]], dtype=np.int64)
    z = log_odds_z(counts)
    kept = characteristic_tokens(counts, z, ["a", "b"], min_count=1, per_source=10,
                                 exclude_top=1)
    assert 0 not in kept.token_ids.tolist(), "the most frequent token must be excluded"
    unfiltered = characteristic_tokens(counts, z, ["a", "b"], min_count=1, per_source=10)
    assert 0 in unfiltered.token_ids.tolist()


def test_characteristic_tokens_raises_rather_than_returning_an_empty_set():
    counts = np.array([[5, 5], [5, 5]], dtype=np.int64)
    z = log_odds_z(counts)
    with pytest.raises(ValueError, match="no token cleared"):
        characteristic_tokens(counts, z, ["a", "b"], min_count=1000, per_source=3)


def test_token_labels_subset_reindexes_to_the_kept_sources():
    counts = np.array([[100, 10, 10], [10, 100, 10], [10, 10, 100]], dtype=np.int64)
    z = log_odds_z(counts)
    labels = characteristic_tokens(counts, z, ["a", "b", "c"], min_count=1, per_source=1)
    sub = labels.subset(["b", "c"])
    assert sub.source_names == ("b", "c")
    assert sorted(sub.labels.tolist()) == [0, 1]
    assert sub.token_ids.size == 2


# ---------------------------------------------------------------------------------------
# Separation statistics: they must find structure AND report its absence
# ---------------------------------------------------------------------------------------


def test_knn_purity_is_perfect_on_separable_clusters_and_at_chance_on_random_labels():
    features, labels = separable_clusters()
    nb = neighbour_indices(features, k=5)
    assert purity_from_neighbours(nb, labels) == pytest.approx(1.0)

    features, labels = structureless()
    nb = neighbour_indices(features, k=5)
    assert purity_from_neighbours(nb, labels) < 0.55, "random labels must not look separated"


def test_neighbour_indices_excludes_self_and_validates_k():
    features, _ = separable_clusters(per_class=5)
    nb = neighbour_indices(features, k=3)
    for i, row in enumerate(nb):
        assert i not in row.tolist()
    with pytest.raises(ValueError, match="k must be"):
        neighbour_indices(features, k=0)
    with pytest.raises(ValueError, match="needs more than"):
        neighbour_indices(features, k=features.shape[0])


def test_permutation_floor_sits_at_chance_for_balanced_classes():
    features, labels = separable_clusters(n_classes=4, per_class=25)
    nb = neighbour_indices(features, k=5)
    mean, sd = permuted_purity(nb, labels, 100, np.random.default_rng(0))
    assert mean == pytest.approx(0.25, abs=0.03), "the floor for 4 balanced classes is 1/4"
    assert sd > 0.0


def test_permutation_floor_needs_a_spread_to_be_meaningful():
    features, labels = separable_clusters()
    nb = neighbour_indices(features, k=3)
    with pytest.raises(ValueError, match="n_permutations"):
        permuted_purity(nb, labels, 1, np.random.default_rng(0))


def test_chance_accuracy_accounts_for_unequal_class_sizes():
    balanced = np.array([0, 0, 1, 1, 2, 2])
    assert chance_accuracy(balanced, 3) == pytest.approx(1 / 3)
    skewed = np.array([0, 0, 0, 0, 1, 2])
    # Uniform guessing over 3 classes on a set that is 2/3 class 0 still scores 1/3 overall,
    # but the helper must derive that from the realised sizes rather than assume it.
    assert chance_accuracy(skewed, 3) == pytest.approx(1 / 3)


def test_silhouette_is_high_when_separated_and_near_zero_when_labels_are_random():
    features, labels = separable_clusters()
    assert silhouette_cosine(features, labels, 3) > 0.7

    features, labels = structureless()
    assert abs(silhouette_cosine(features, labels, 3)) < 0.15


def test_per_class_purity_and_confusion_agree_on_the_diagonal():
    features, labels = separable_clusters(n_classes=3, per_class=20, spread=0.35, seed=3)
    nb = neighbour_indices(features, k=5)
    purity = per_class_purity(nb, labels, 3)
    conf = neighbour_confusion(nb, labels, 3)
    for c in range(3):
        assert conf[c, c] == pytest.approx(purity[c])
    assert np.allclose(conf.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------------------
# The linear probe
# ---------------------------------------------------------------------------------------


def test_linear_probe_separates_clusters_and_falls_to_chance_on_shuffled_labels():
    features, labels = separable_clusters(n_classes=3, per_class=40)
    good = linear_probe(features, labels, 3, repeats=3)
    assert good.accuracy_mean > 0.95

    shuffled = np.random.default_rng(7).permutation(labels)
    bad = linear_probe(features, shuffled, 3, repeats=3)
    assert bad.accuracy_mean < 0.55, "a probe on shuffled labels must not generalise"


def test_linear_probe_confusion_rows_are_normalised_and_recall_is_its_diagonal():
    features, labels = separable_clusters(n_classes=3, per_class=30, spread=0.4, seed=5)
    result = linear_probe(features, labels, 3, repeats=2)
    assert np.allclose(result.confusion.sum(axis=1), 1.0)
    for c in range(3):
        assert result.per_class_recall[c] == pytest.approx(result.confusion[c, c])


def test_linear_probe_test_rows_are_never_in_training():
    """A probe that scores its own training rows is not a measurement of anything.

    Verified behaviourally: a class whose points are pure noise cannot be learned, so if the
    reported accuracy were in-sample it would rise far above chance as capacity is added.
    """
    rng = np.random.default_rng(11)
    features = rng.normal(size=(120, 60))          # 60 dims, 84 training rows: over-capacity
    labels = np.repeat(np.arange(3), 40)
    result = linear_probe(features, labels, 3, repeats=3, iterations=800)
    assert result.accuracy_mean < 0.55, "noise features must not yield an above-chance score"


def test_linear_probe_rejects_impossible_splits():
    features, labels = separable_clusters(n_classes=2, per_class=5)
    with pytest.raises(ValueError, match="train_fraction"):
        linear_probe(features, labels, 2, train_fraction=1.0)
    with pytest.raises(ValueError, match="repeats"):
        linear_probe(features, labels, 2, repeats=0)
    lonely = np.array([0, 1, 1, 1])
    with pytest.raises(ValueError, match="train/test split needs"):
        linear_probe(features[:4], lonely, 2)


def test_frequency_features_contain_only_count_and_norm():
    embedding = np.array([[3.0, 4.0], [0.0, 1.0], [1.0, 0.0]])
    labels = TokenLabels(token_ids=np.array([0, 1]), labels=np.array([0, 1]),
                         source_names=("a", "b"), z_scores=np.array([1.0, 2.0]),
                         total_counts=np.array([99, 0]))
    feats = frequency_features(labels, embedding)
    assert feats.shape == (2, 2)
    assert feats[0, 0] == pytest.approx(np.log(100.0))
    assert feats[0, 1] == pytest.approx(5.0)
    assert feats[1, 0] == pytest.approx(0.0), "log(0 + 1) must be 0, never -inf"


# ---------------------------------------------------------------------------------------
# Projection and rendering
# ---------------------------------------------------------------------------------------


def test_pca_2d_recovers_a_plane_embedded_in_a_higher_dimensional_space():
    rng = np.random.default_rng(0)
    plane = rng.normal(size=(200, 2)) * np.array([10.0, 5.0])
    high = np.zeros((200, 7))
    high[:, :2] = plane
    high += rng.normal(0, 0.01, high.shape)
    coords, ratios = pca_2d(high)
    assert coords.shape == (200, 2)
    assert ratios[0] > ratios[1] > 0.0
    assert ratios[0] + ratios[1] > 0.99


def test_ascii_scatter_gives_distinct_glyphs_to_sources_sharing_a_first_letter():
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    labels = np.array([0, 1, 2])
    lines = ascii_scatter(coords, labels, ["poetry", "procedural", "spine"], width=20,
                          height=10)
    legend = lines[-1]
    glyphs = {chunk.split("=")[0] for chunk in legend.split()
              if "=" in chunk and chunk.split("=", 1)[1] in
              ("poetry", "procedural", "spine")}
    assert len(glyphs) == 3, f"glyphs collided: {legend}"
    assert not any(line.endswith(" ") for line in lines), "no trailing right-hand border"


def test_ascii_scatter_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="same points"):
        ascii_scatter(np.zeros((3, 2)), np.array([0, 1]), ["a", "b"])


def _fake_result(name="all", purity=0.5, floor=0.11, floor_sd=0.003, freq=0.13):
    probe = ProbeResult(accuracy_mean=0.7, accuracy_sd=0.02, n_repeats=3,
                        confusion=np.eye(2), per_class_recall=[0.7, 0.7])
    return SeparationResult(
        name=name, source_names=("spine", "weird"), n_tokens=4, class_sizes=[2, 2], k=3,
        knn_purity=purity, knn_permuted_mean=floor, knn_permuted_sd=floor_sd,
        knn_chance=0.5, knn_frequency_only=freq, per_source_purity=[0.6, 0.4],
        confusion=np.array([[0.6, 0.4], [0.4, 0.6]]), silhouette=0.02,
        silhouette_permuted_mean=0.0, silhouette_permuted_sd=0.001, probe=probe,
        probe_frequency_only=probe, probe_permuted=probe,
        top_tokens={"spine": [("Ġof", 10, 5.0)], "weird": [("ĠI", 12, 6.0)]},
        projection=["S  W", "", "legend: S=spine  W=weird"],
        projection_variance=(0.03, 0.02),
        projection_centroids=[(-0.1, 0.2), (0.3, -0.4)])


def test_verdict_applies_the_projects_noise_floor_standard():
    assert verdict(_fake_result(purity=0.5)).startswith("SEPARATED")
    # Inside the permutation floor: no correspondence to report.
    assert verdict(_fake_result(purity=0.113)).startswith("NOT INTERPRETABLE")
    # Above the floor but no better than what token frequency alone achieves.
    assert verdict(_fake_result(purity=0.30, freq=0.29)).startswith("FREQUENCY, NOT GEOGRAPHY")


def test_render_markdown_carries_the_baselines_next_to_the_headline():
    md = render_markdown([_fake_result()], hf_model="artifacts/hf-x", corpus_dir="artifacts/c",
                         words_per_source=1000, prior_strength=1000.0, min_count=25,
                         per_source=150, exclude_top=500, note="a note")
    assert md.startswith("<!-- SPDX-License-Identifier: Apache-2.0 -->")
    assert "a note" in md
    for required in ("permuted floor", "frequency-only", "chance", "Monroe",
                     "What this does and does not show", "sigmas above floor"):
        assert required in md, f"the report must state {required!r}"


def test_render_markdown_renders_the_centroid_table_and_the_projection():
    md = render_markdown([_fake_result()], hf_model="m", corpus_dir="c", words_per_source=1,
                         prior_strength=1.0, min_count=1, per_source=1, exclude_top=1)
    assert "| spine | -0.1000 | +0.2000 |" in md
    assert "legend: S=spine" in md


def test_report_to_json_round_trips_and_states_its_statistic():
    payload = report_to_json([_fake_result()], hf_model="m", corpus_dir="c",
                             words_per_source=10, prior_strength=1000.0, min_count=25,
                             per_source=150, exclude_top=500, k=10, n_permutations=200,
                             seed=0)
    text = json.dumps(payload)          # must contain nothing json cannot serialise
    reloaded = json.loads(text)
    assert "Monroe" in reloaded["statistic"]
    condition = reloaded["conditions"][0]
    assert condition["knn"]["chance"] == 0.5
    assert condition["knn"]["frequency_only"] == 0.13
    assert condition["verdict"].startswith("SEPARATED")
    assert condition["projection_centroids"]["weird"] == [0.3, -0.4]


def test_report_to_json_writes_null_not_inf_for_an_infinite_sigma():
    """`inf` is not valid JSON; a zero-variance floor must not produce an unparseable file."""
    payload = report_to_json([_fake_result(purity=0.5, floor_sd=0.0)], hf_model="m",
                             corpus_dir="c", words_per_source=1, prior_strength=1.0,
                             min_count=1, per_source=1, exclude_top=0, k=1,
                             n_permutations=2, seed=0)
    text = json.dumps(payload, allow_nan=False)
    assert json.loads(text)["conditions"][0]["knn"]["sigmas_above_floor"] is None


def test_default_output_paths_strip_the_hf_prefix():
    md, js = _default_output_paths(Path("/x/artifacts/hf-tt-tnt-1024a"))
    assert md.name == "embedding-geography-tt-tnt-1024a.md"
    assert js.name == "embedding-geography-tt-tnt-1024a.json"
    assert md.parent.name == "measurements"


# ---------------------------------------------------------------------------------------
# Corpus reading and counting, against a fake corpus and a fake tokenizer
# ---------------------------------------------------------------------------------------


def test_read_corpus_text_keeps_newlines_and_stops_at_whole_lines(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("one two three\nfour five six\nseven eight nine\n")
    text, n_words = read_corpus_text(path, 4)
    assert text == "one two three\nfour five six\n", "newlines are structure here, not padding"
    assert n_words == 6, "the budget is a floor; the last line is never cut in half"


def test_read_corpus_text_rejects_a_meaningless_budget(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("a b\n")
    with pytest.raises(ValueError, match="word_budget"):
        read_corpus_text(path, 0)


class _FakeTokenizer:
    """Maps each whitespace word to a stable id by hashing -- enough to count with."""

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [hash(w) % self.vocab_size for w in text.split()]}


def test_count_tokens_by_source_returns_one_row_per_source(tmp_path):
    (tmp_path / "a.txt").write_text("alpha beta alpha\n")
    (tmp_path / "b.txt").write_text("gamma gamma gamma\n")
    counts = count_tokens_by_source(tmp_path, ["a", "b"], _FakeTokenizer(64), 64,
                                    words_per_source=10)
    assert counts.shape == (2, 64)
    assert counts[0].sum() == 3 and counts[1].sum() == 3
    assert counts[1].max() == 3, "b's three identical words must land on one id"


def test_count_tokens_by_source_names_the_missing_source(tmp_path):
    (tmp_path / "a.txt").write_text("x\n")
    with pytest.raises(FileNotFoundError, match="missing.txt"):
        count_tokens_by_source(tmp_path, ["a", "missing"], _FakeTokenizer(8), 8,
                               words_per_source=5)


def test_count_tokens_by_source_rejects_a_tokenizer_that_overflows_the_vocabulary(tmp_path):
    (tmp_path / "a.txt").write_text("x\n")

    class Overflowing:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [99]}

    with pytest.raises(ValueError, match="do not agree on a vocabulary"):
        count_tokens_by_source(tmp_path, ["a"], Overflowing(), 8, words_per_source=5)


# ---------------------------------------------------------------------------------------
# End-to-end on synthetic data: the battery must find planted geography, and only that
# ---------------------------------------------------------------------------------------


def test_evaluate_separation_finds_planted_geography_and_calls_it_separated():
    rng = np.random.default_rng(0)
    embedding = rng.normal(0, 0.01, size=(300, 6))
    labels_arr = np.repeat(np.arange(3), 30)
    token_ids = np.arange(90)
    embedding[token_ids] = np.eye(3, 6)[labels_arr] + rng.normal(0, 0.05, (90, 6))
    labels = TokenLabels(token_ids=token_ids, labels=labels_arr,
                         source_names=("a", "b", "c"),
                         z_scores=np.ones(90), total_counts=np.full(90, 100))
    result = evaluate_separation("planted", labels, embedding, k=5, n_permutations=20,
                                 probe_repeats=2, with_projection=True)
    assert result.knn_purity > 0.9
    assert result.knn_permuted_mean == pytest.approx(1 / 3, abs=0.06)
    assert result.knn_sigmas_above_floor > 3.0
    assert result.probe.accuracy_mean > 0.9
    assert verdict(result).startswith("SEPARATED")
    assert result.projection is not None and result.projection_centroids is not None


def test_evaluate_separation_reports_no_geography_when_there_is_none():
    """The measurement must be capable of returning the negative answer, or it proves nothing."""
    rng = np.random.default_rng(3)
    embedding = rng.normal(size=(200, 6))
    token_ids = np.arange(90)
    labels = TokenLabels(token_ids=token_ids, labels=np.repeat(np.arange(3), 30),
                         source_names=("a", "b", "c"), z_scores=np.ones(90),
                         total_counts=np.full(90, 100))
    result = evaluate_separation("unplanted", labels, embedding, k=5, n_permutations=50,
                                 probe_repeats=2)
    assert result.knn_sigmas_above_floor < 3.0
    assert verdict(result).startswith("NOT INTERPRETABLE")


# ---------------------------------------------------------------------------------------
# The real artifacts, when (and only when) they are on this machine
# ---------------------------------------------------------------------------------------


@pytest.mark.skipif(not (MODEL_DIR / "model.safetensors").is_file(),
                    reason=f"{MODEL_DIR}/model.safetensors is a local training artifact, not "
                           f"repository content; this suite must pass without it")
def test_load_embedding_matrix_reads_the_real_table_without_loading_the_model():
    embedding = load_embedding_matrix(MODEL_DIR)
    config = json.loads((MODEL_DIR / "config.json").read_text())
    assert embedding.shape == (config["vocab_size"], config["hidden_size"])
    assert embedding.dtype == np.float32, "bfloat16 must be widened, not reinterpreted"


@pytest.mark.skipif(not (MODEL_DIR / "model.safetensors").is_file(),
                    reason=f"{MODEL_DIR} is a local training artifact, not repository content")
def test_load_embedding_matrix_names_the_keys_it_does_have():
    with pytest.raises(KeyError, match="no tensor"):
        load_embedding_matrix(MODEL_DIR, key="model.not_a_tensor")


def test_load_embedding_matrix_reports_a_missing_checkpoint_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="no safetensors checkpoint"):
        load_embedding_matrix(tmp_path)
