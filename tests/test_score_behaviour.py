# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for scripts/score_behaviour.py.

Every signal's computation is pinned against synthetic inputs whose right answer is obvious by
inspection -- "a a a a a" must be maximally repetitive, a completion ending in `</s>` must count
as terminated, a completion containing "Once upon a time" must count as collapsed.

MUTATION CHECKS. A test that still passes when the thing it tests is reverted is worth nothing,
and this repo has shipped two of those. So the collapse and repetition detectors are additionally
tested against a DEFEATED version of themselves: `test_*_mutation_*` re-runs the same assertion
through a deliberately broken implementation and requires it to fail. If the detector is ever
weakened back to that broken form, those tests go red rather than staying green.

Nothing here needs a trained model or a corpus on disk. The two tests that exercise real
generation build a tiny random-initialised LlamaForCausalLM and skip explicitly (with a reason
in the pytest report) if torch/transformers are absent -- matching tests/test_probe_context_use.py.
No test touches Tenstorrent hardware; this script is CPU-only by construction.
"""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.score_behaviour import (  # noqa: E402
    COLLAPSE_MARKERS,
    FRAME_MARKERS,
    LEXICAL_MARKERS,
    SIGNALS,
    ControlRow,
    Estimate,
    PromptReport,
    RegisterProfile,
    SourceLM,
    aggregate_over_prompts,
    build_register_profile,
    collapse_markers_found,
    content_words,
    default_label,
    detector_controls,
    estimate,
    is_collapsed,
    is_frame_collapsed,
    is_lexical_collapsed,
    is_terminated,
    load_prompts,
    longest_repeated_ngram,
    ngram_repeat_rate,
    paired_differences,
    prompt_engagement,
    read_corpus_tokens,
    render_comparison,
    render_markdown,
    report_to_json,
    resolve_eos_id,
    resolve_model_dir,
    score_completion,
    strip_at_eos,
    summarise_prompt,
    words_of,
)


# ---------------------------------------------------------------------------------------
# Signal 2: repetition -- the obvious cases
# ---------------------------------------------------------------------------------------


def test_a_completion_that_is_literally_a_a_a_a_a_is_maximally_repetitive():
    # Five identical words at n=4 give two 4-grams, one distinct: exactly half are repeats,
    # and that is the ceiling for a five-word text.
    assert ngram_repeat_rate("a a a a a".split(), n=4) == pytest.approx(0.5)
    # Longer, and the rate marches toward 1.0 -- the definition of degenerate.
    assert ngram_repeat_rate(["a"] * 40, n=4) > 0.9
    assert ngram_repeat_rate(["a"] * 400, n=4) > 0.99


def test_non_repeating_prose_scores_near_zero():
    words = words_of("The chimp chose the longest stick, then the one that had been bitten "
                     "through by something with a mouth much smaller than its own.")
    assert ngram_repeat_rate(words, n=4) == 0.0


def test_repeat_rate_is_none_not_zero_when_the_text_is_shorter_than_n():
    # The distinction matters: averaging "too short to say" in as 0.0 would make a model that
    # emits three-word completions look perfectly non-repetitive.
    assert ngram_repeat_rate("one two three".split(), n=4) is None
    assert ngram_repeat_rate([], n=4) is None
    assert ngram_repeat_rate("one two three four".split(), n=4) == 0.0


def test_repeat_rate_rejects_a_nonsense_order():
    with pytest.raises(ValueError, match="n must be >= 1"):
        ngram_repeat_rate("a b c".split(), n=0)


def test_longest_repeated_span_finds_the_whole_loop():
    # "the bees were busy" twice: a four-word span repeats, a five-word one does not.
    words = words_of("the bees were busy and the bees were busy again")
    assert longest_repeated_ngram(words) == 4


def test_longest_repeated_span_is_zero_when_every_word_is_distinct():
    assert longest_repeated_ngram("alpha beta gamma delta".split()) == 0


def test_longest_repeated_span_counts_a_single_repeated_word():
    assert longest_repeated_ngram("the cat sat on the mat".split()) == 1


def test_deliberate_repetition_is_measured_not_excused():
    # A stutter-prompt completion IS repetitive, and the detector must say so. Fairness to the
    # stutter probes is handled at aggregation time (a separate aggregate), never by teaching
    # the detector to look the other way -- which would make it blind to real degeneration too.
    words = words_of("rose is a rose is a rose is a rose")
    assert ngram_repeat_rate(words, n=4) > 0.5


# --- mutation check: a defeated repetition detector must FAIL these same assertions -------


def _mutant_repeat_rate_counts_types_not_repeats(words, n=4):
    """A plausible-looking but WRONG repetition rate: distinct-n ratio, not its complement.

    This is the mutation. If the real implementation were ever "simplified" into this, the
    assertions below would break -- which is what the paired test proves.
    """
    if len(words) < n:
        return None
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def test_repetition_mutation_the_inverted_detector_fails_the_obvious_cases():
    obvious = ["a"] * 40
    clean = words_of("The chimp chose the longest stick, then the one that had been bitten "
                     "through by something with a mouth much smaller than its own.")
    # The real detector gets both right...
    assert ngram_repeat_rate(obvious, n=4) > 0.9
    assert ngram_repeat_rate(clean, n=4) == 0.0
    # ...and the mutant gets both exactly backwards, so this test cannot pass on the mutant.
    assert _mutant_repeat_rate_counts_types_not_repeats(obvious, 4) < 0.1
    assert _mutant_repeat_rate_counts_types_not_repeats(clean, 4) == 1.0


def _mutant_longest_repeat_returns_text_length(words):
    """A detector that reports length instead of repetition -- passes a lazy "it's a number"
    test, fails any test that actually pins the value."""
    return len(words)


def test_repetition_mutation_a_length_returning_detector_fails():
    words = words_of("the bees were busy and the bees were busy again")
    assert longest_repeated_ngram(words) == 4
    assert _mutant_longest_repeat_returns_text_length(words) != 4


# ---------------------------------------------------------------------------------------
# Signal 1: genre collapse
# ---------------------------------------------------------------------------------------


def test_the_documented_attractor_phrases_are_all_detected():
    for text in ("Once upon a time, there was a duck.",
                 "There was a little girl named Lily who lived by the sea.",
                 "The moral of the story is that it is good to share.",
                 "and they lived happily ever after.",
                 "From that day on, the mouse was careful.",
                 "One day, the bees decided to leave.",
                 "She was so happy she could not speak.",
                 "a boy named Tom",
                 "his mom said no",
                 "The end."):
        assert is_collapsed(text), text


def test_target_voice_prose_is_not_flagged_as_collapsed():
    # Lines in the register the project is actually aiming for. None may trip the detector, or
    # the metric would punish the model for succeeding.
    for text in ("The ants had learned that being eaten was a way of travelling further than "
                 "any ant walks in a season.",
                 "I placed a straw across the trench and waited; the procession divided, "
                 "considered, and re-formed on the far side.",
                 "Above, the mountain. Below, the lake. The image is of a thing about to be "
                 "understood.",
                 "An ant is an insect that lives in colonies of up to several million "
                 "individuals."):
        assert not is_collapsed(text), text


def test_collapse_is_judged_on_the_completion_alone_not_the_prompt():
    # voice-03's frozen prompt IS the attractor's opening line. Scoring prompt+completion would
    # mark every model's answer collapsed regardless of what the model actually wrote.
    prompt = "Once upon a time, there was a little"
    completion = " ridge of pumice the sea had been working on for a hundred years."
    assert is_collapsed(prompt + completion)
    assert not is_collapsed(completion)


def test_frame_and_lexical_sub_signals_split_the_marker_set_exactly():
    names = {name for name, _pattern in COLLAPSE_MARKERS}
    assert FRAME_MARKERS | LEXICAL_MARKERS == names
    assert not (FRAME_MARKERS & LEXICAL_MARKERS)


def test_frame_collapse_and_lexical_collapse_fire_independently():
    frame_only = "Once upon a time there lived a stone."
    lexical_only = "One day, the tide came in over the flats."
    assert is_frame_collapsed(frame_only) and not is_lexical_collapsed(frame_only)
    assert is_lexical_collapsed(lexical_only) and not is_frame_collapsed(lexical_only)
    # And the union is exactly "either of them".
    for text in (frame_only, lexical_only, "One day, they lived happily ever after."):
        assert is_collapsed(text) == (is_frame_collapsed(text) or is_lexical_collapsed(text))


def test_named_proper_is_the_only_case_sensitive_marker():
    # It keys on a capitalised proper noun; lowercasing it would make it fire on "named after".
    assert "named_proper" in collapse_markers_found("a bird named Aster")
    assert "named_proper" not in collapse_markers_found("a bird named after the river")
    # Every other marker must survive arbitrary casing.
    assert "once_upon_a_time" in collapse_markers_found("ONCE UPON A TIME")
    assert "moral_of_the_story" in collapse_markers_found("THE MORAL OF THE STORY IS")


def test_collapse_markers_are_reported_by_name_so_a_result_can_be_audited():
    found = collapse_markers_found("The moral of the story is that his mom was very happy.")
    assert found == ["moral_of_the_story", "so_very_happy", "his_her_mom"]


# --- mutation check: a defeated collapse detector must FAIL these same assertions ----------


_MUTANT_MARKERS = (("once_upon_a_time", r"once upon a time"),)


def _mutant_is_collapsed(text):
    """The detector reduced to its single most obvious marker -- the shape a "simplification"
    would most plausibly take, and the shape that would silently stop detecting most collapse."""
    return any(re.search(p, text, re.IGNORECASE) for _n, p in _MUTANT_MARKERS)


def test_collapse_mutation_a_single_marker_detector_misses_what_the_real_one_catches():
    missed = ("There was a little girl named Lily who lived by the sea.",
              "The moral of the story is that it is good to share.",
              "From that day on, the mouse was careful.",
              "One day, the bees decided to leave.",
              "his mom said no")
    for text in missed:
        assert is_collapsed(text), text
        assert not _mutant_is_collapsed(text), text


def _mutant_is_collapsed_always_true(text):
    """A detector with no specificity at all. Catches everything, means nothing."""
    return True


def test_collapse_mutation_an_always_true_detector_fails_the_target_voice_cases():
    clean = ("The ants had learned that being eaten was a way of travelling further.",
             "Above, the mountain. Below, the lake.")
    for text in clean:
        assert not is_collapsed(text), text
        assert _mutant_is_collapsed_always_true(text)


# ---------------------------------------------------------------------------------------
# Signal 3: termination
# ---------------------------------------------------------------------------------------


def test_a_completion_ending_in_eos_counts_as_terminated():
    assert is_terminated([10, 11, 12, 2])


def test_a_completion_that_ran_out_of_budget_does_not_count_as_terminated():
    assert not is_terminated([10, 11, 12, 13])
    assert not is_terminated([])


def test_eos_followed_by_padding_still_counts_as_terminated():
    # Batched generate() pads every row out to the longest in the batch, so a terminated row's
    # LAST id is normally the pad token, not </s>. Testing "ends with eos" would report 0%.
    assert is_terminated([10, 2, 3, 3, 3])
    assert strip_at_eos([10, 2, 3, 3, 3]) == [10]


def test_termination_respects_a_models_own_eos_id():
    assert is_terminated([10, 7], eos_id=7)
    assert not is_terminated([10, 7], eos_id=2)


def test_generated_length_stops_at_eos_not_at_the_padding():
    score = score_completion("a prompt", "text", [10, 11, 2, 3, 3])
    assert score.n_tokens == 2
    assert score.terminated


# ---------------------------------------------------------------------------------------
# Signal 5: prompt engagement
# ---------------------------------------------------------------------------------------


def test_engagement_is_one_when_every_content_word_comes_back():
    assert prompt_engagement("The bees were busy", "busy bees everywhere") == 1.0


def test_engagement_is_zero_when_the_completion_ignores_the_prompt():
    assert prompt_engagement("Chimpanzees use tools such as", "the lake is very deep") == 0.0


def test_engagement_ignores_function_words_and_matches_plurals_crudely():
    # "The"/"was" are stopwords; "bees" must match "bee".
    assert content_words("The bees was") == ["bee"]
    assert prompt_engagement("The bees", "a single bee") == 1.0


def test_engagement_is_none_when_a_prompt_has_no_content_words():
    assert prompt_engagement("the and of it", "anything at all") is None


# ---------------------------------------------------------------------------------------
# Signal 4: register
# ---------------------------------------------------------------------------------------


def _toy_profile():
    """Two deliberately unmistakable 'sources' plus tinystories, built from tiny word lists."""
    vocab = ["once", "upon", "time", "little", "girl", "ant", "colony", "mandible", "trench",
             "the", "a"]
    word_to_id = {w: i for i, w in enumerate(vocab)}
    unk = len(vocab)
    corpora = {
        "tinystories": "once upon a time the little girl the little girl once upon a time".split(),
        "spine": "the ant colony the mandible the trench the ant colony the mandible".split(),
    }
    models = {name: SourceLM([word_to_id.get(w, unk) for w in words], unk + 1)
              for name, words in corpora.items()}
    return RegisterProfile(word_to_id=word_to_id, unk_id=unk, models=models,
                           train_words={k: len(v) for k, v in corpora.items()})


def test_register_puts_a_tinystories_sentence_nearest_tinystories():
    profile = _toy_profile()
    assert profile.nearest_source(words_of("once upon a time the little girl")) == "tinystories"


def test_register_puts_an_observational_sentence_nearest_spine():
    profile = _toy_profile()
    assert profile.nearest_source(words_of("the ant colony the mandible")) == "spine"


def test_tinystories_margin_has_the_sign_its_name_promises():
    profile = _toy_profile()
    assert profile.tinystories_margin(words_of("once upon a time the little girl")) > 0
    assert profile.tinystories_margin(words_of("the ant colony the mandible")) < 0


def test_register_scores_are_per_word_so_length_does_not_decide_the_answer():
    profile = _toy_profile()
    short = profile.score(words_of("the ant colony"))
    long = profile.score(words_of("the ant colony the ant colony the ant colony"))
    # Repeating the same text must not move the per-word likelihood much, and must not flip
    # which source wins -- a summed score would fall off a cliff with length instead.
    assert max(short, key=short.get) == max(long, key=long.get) == "spine"
    assert abs(short["spine"] - long["spine"]) < 0.5


def test_register_declines_to_guess_on_empty_input():
    profile = _toy_profile()
    assert profile.score([]) == {}
    assert profile.nearest_source([]) is None
    assert profile.tinystories_margin([]) is None


def test_source_lm_rejects_degenerate_construction():
    with pytest.raises(ValueError, match="zero words"):
        SourceLM([], 5)
    with pytest.raises(ValueError, match="vocab_size"):
        SourceLM([0], 0)
    with pytest.raises(ValueError, match="empty token sequence"):
        SourceLM([0, 1], 3).logprob_per_word([])


def test_corpus_reader_keeps_case_and_punctuation(tmp_path):
    # The detector controls run on this text, and two markers need case/punctuation to fire.
    # Normalising here would silently disarm them and misreport the detector's own accuracy.
    path = tmp_path / "src.txt"
    path.write_text("The end. A bird named Aster.\nOne day, it left.\n")
    tokens = read_corpus_tokens(path, 100)
    assert "The" in tokens and "end." in tokens and "Aster." in tokens
    assert is_collapsed(" ".join(tokens))


def test_building_a_register_profile_names_the_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.txt"):
        build_register_profile(tmp_path, ["nope"])


def test_a_small_source_is_split_ninety_ten_rather_than_dropped(tmp_path):
    (tmp_path / "small.txt").write_text(" ".join(["alpha beta"] * 50))
    profile = build_register_profile(tmp_path, ["small"], train_words=1_000_000,
                                     control_words=10)
    assert profile.train_words["small"] == 90
    assert len(profile.holdout["small"]) == 10


# ---------------------------------------------------------------------------------------
# Aggregation -- prompts are the sampling unit
# ---------------------------------------------------------------------------------------


def test_estimate_reports_mean_sem_and_n_and_skips_missing_values():
    est = estimate([1.0, 1.0, 1.0, None])
    assert (est.mean, est.sem, est.n) == (1.0, 0.0, 3)
    assert estimate([None, None]) is None


def test_estimate_ci95_is_the_stated_normal_approximation():
    est = Estimate(mean=0.5, sem=0.1, n=10)
    assert est.ci95_lo == pytest.approx(0.5 - 1.96 * 0.1)
    assert est.ci95_hi == pytest.approx(0.5 + 1.96 * 0.1)


def _report(prompt_id, probe, collapse, samples=10):
    """A PromptReport carrying one signal, for aggregation tests."""
    return PromptReport(prompt_id=prompt_id, probe=probe, text="t", n_samples=samples,
                        estimates={"collapse_rate": Estimate(mean=collapse, sem=0.0,
                                                             n=samples)},
                        marker_counts={}, nearest_counts={})


def test_the_aggregate_is_over_prompts_not_over_pooled_completions():
    # Three prompts of 10 completions each. The aggregate's n must be 3 (prompts), not 30
    # (completions) -- completions of one prompt are not independent observations of the model.
    reports = [_report("a", "x", 0.0), _report("b", "x", 0.5), _report("c", "x", 1.0)]
    agg = aggregate_over_prompts(reports, "collapse_rate")
    assert agg.n == 3
    assert agg.mean == pytest.approx(0.5)
    assert agg.sem == pytest.approx(0.2886751, rel=1e-4)


def test_the_aggregate_can_exclude_the_deliberate_repetition_probe():
    reports = [_report("a", "target-voice", 0.0), _report("s", "stutter", 1.0)]
    assert aggregate_over_prompts(reports, "collapse_rate").mean == pytest.approx(0.5)
    excluded = aggregate_over_prompts(reports, "collapse_rate", exclude_probe="stutter")
    assert excluded.n == 1 and excluded.mean == 0.0


def test_summarising_a_prompt_with_no_completions_raises_rather_than_inventing_a_zero():
    with pytest.raises(ValueError, match="no completions"):
        summarise_prompt({"id": "x", "probe": "p", "text": "t"}, [])


def test_summarise_prompt_computes_every_declared_signal():
    scores = [score_completion("The bees were busy", "One day, the bees were busy. The end.",
                               [5, 6, 2]),
              score_completion("The bees were busy", "a quiet unrepeating observation",
                               [5, 6, 7])]
    report = summarise_prompt({"id": "x", "probe": "p", "text": "The bees were busy"}, scores)
    for key, _title, _direction in SIGNALS:
        assert key in report.estimates, key
    assert report.estimates["collapse_rate"].mean == pytest.approx(0.5)
    assert report.estimates["termination_rate"].mean == pytest.approx(0.5)
    assert report.marker_counts["one_day_comma"] == 1


# ---------------------------------------------------------------------------------------
# Comparison -- paired, and able to say "worse"
# ---------------------------------------------------------------------------------------


def _payload(label, per_prompt):
    return {"label": label, "hf_model": f"artifacts/hf-{label}", "num_samples": 8,
            "per_prompt": {pid: {"probe": "p", "n_samples": 8,
                                 "estimates": {k: {"mean": v, "sem": 0.0, "n": 8}
                                               for k, v in signals.items()}}
                           for pid, signals in per_prompt.items()}}


def test_a_clear_improvement_is_reported_as_better():
    base = _payload("v1", {f"p{i}": {"collapse_rate": 0.9} for i in range(10)})
    cand = _payload("v2", {f"p{i}": {"collapse_rate": 0.1} for i in range(10)})
    diff = next(d for d in paired_differences(base, cand) if d.signal == "collapse_rate")
    assert diff.verdict == "better"
    assert diff.difference.mean == pytest.approx(-0.8)


def test_a_clear_regression_is_reported_as_worse():
    # THE point of this test: the metric must be able to say a model got worse. A score that
    # only ever goes up is a vanity metric.
    base = _payload("v1", {f"p{i}": {"collapse_rate": 0.1} for i in range(10)})
    cand = _payload("v2", {f"p{i}": {"collapse_rate": 0.9} for i in range(10)})
    diff = next(d for d in paired_differences(base, cand) if d.signal == "collapse_rate")
    assert diff.verdict == "worse"


def test_a_regression_on_a_higher_is_better_signal_is_also_worse():
    # Direction is per-signal: termination FALLING is the regression.
    base = _payload("v1", {f"p{i}": {"termination_rate": 0.9} for i in range(10)})
    cand = _payload("v2", {f"p{i}": {"termination_rate": 0.1} for i in range(10)})
    diff = next(d for d in paired_differences(base, cand) if d.signal == "termination_rate")
    assert diff.verdict == "worse"


def test_noise_is_reported_as_no_change_not_as_a_small_improvement():
    base = _payload("v1", {f"p{i}": {"collapse_rate": v}
                           for i, v in enumerate([0.1, 0.9, 0.2, 0.8, 0.5])})
    cand = _payload("v2", {f"p{i}": {"collapse_rate": v}
                           for i, v in enumerate([0.9, 0.1, 0.8, 0.2, 0.5])})
    diff = next(d for d in paired_differences(base, cand) if d.signal == "collapse_rate")
    assert diff.verdict == "no change"
    assert diff.min_detectable > abs(diff.difference.mean)


def test_comparison_is_paired_which_is_what_makes_fifteen_prompts_enough():
    # A consistent +0.1 on every prompt, on top of huge between-prompt spread. Paired, the
    # difference is unmistakable (SEM 0); unpaired, the prompt spread would swamp it.
    spread = [0.0, 0.2, 0.4, 0.6, 0.8]
    base = _payload("v1", {f"p{i}": {"collapse_rate": v} for i, v in enumerate(spread)})
    cand = _payload("v2", {f"p{i}": {"collapse_rate": v + 0.1} for i, v in enumerate(spread)})
    diff = next(d for d in paired_differences(base, cand) if d.signal == "collapse_rate")
    assert diff.difference.mean == pytest.approx(0.1)
    assert diff.difference.sem == pytest.approx(0.0)
    assert diff.verdict == "worse"          # collapse going UP is a regression
    assert diff.baseline.sem > 0.1          # ...while the unpaired spread is far larger


def test_comparing_runs_with_no_shared_prompts_raises():
    base = _payload("v1", {"a": {"collapse_rate": 0.1}})
    cand = _payload("v2", {"z": {"collapse_rate": 0.1}})
    with pytest.raises(ValueError, match="share no prompt ids"):
        paired_differences(base, cand)


def test_the_comparison_report_names_signals_that_moved_the_wrong_way():
    base = _payload("v1", {f"p{i}": {"collapse_rate": 0.1, "termination_rate": 0.9}
                           for i in range(6)})
    cand = _payload("v2", {f"p{i}": {"collapse_rate": 0.9, "termination_rate": 0.1}
                           for i in range(6)})
    diffs = paired_differences(base, cand)
    md = render_comparison(base, cand, diffs, label="x")
    assert "**worse on 2**" in md
    assert "moved in the **wrong** direction" in md
    assert "none. This line exists" not in md


# ---------------------------------------------------------------------------------------
# Rendering and provenance
# ---------------------------------------------------------------------------------------


def _one_real_report():
    prompt = {"id": "voice-01", "probe": "target-voice", "text": "The bees were busy"}
    scores = [score_completion(prompt["text"], "One day, the bees were busy.", [5, 6, 2]),
              score_completion(prompt["text"], "the ant colony went on regardless", [5, 6, 7])]
    return summarise_prompt(prompt, scores)


def test_markdown_states_the_decoding_settings_and_the_n_it_rests_on():
    md = render_markdown([_one_real_report()], hf_model=Path("artifacts/hf-x"), label="x",
                         num_samples=2, max_new_tokens=60, temperature=0.8, top_p=0.95,
                         seed=0, eos_id=2, controls=None, control_excerpt_words=None,
                         register_sources=None)
    assert "temperature 0.8" in md and "top_p 0.95" in md and "seed 0" in md
    assert "EOS id 2" in md
    assert "2 sampled completions per prompt" in md
    assert "(n=2)" in md          # per-prompt n
    assert "(n=1)" in md          # aggregate n = number of prompts


def test_markdown_says_the_detectors_are_uncalibrated_when_controls_were_skipped():
    md = render_markdown([_one_real_report()], hf_model=Path("artifacts/hf-x"), label="x",
                         num_samples=2, max_new_tokens=60, temperature=0.8, top_p=0.95,
                         seed=0, eos_id=2, controls=None, control_excerpt_words=None,
                         register_sources=None)
    assert "uncalibrated" in md


def test_markdown_renders_the_controls_including_the_per_marker_audit():
    controls = [ControlRow(source="tinystories", n_excerpts=100, collapse_rate=0.48,
                           frame_collapse_rate=0.25, lexical_collapse_rate=0.40,
                           register_accuracy=0.99, register_top_confusion="folklore 0.5%",
                           repeat_rate=Estimate(0.005, 0.001, 100),
                           marker_rates={"once_upon_a_time": 0.16}),
                ControlRow(source="spine", n_excerpts=100, collapse_rate=0.003,
                           frame_collapse_rate=0.0, lexical_collapse_rate=0.003,
                           register_accuracy=0.90, register_top_confusion="folklore 1.6%",
                           repeat_rate=Estimate(0.0005, 0.0001, 100),
                           marker_rates={"once_upon_a_time": 0.0})]
    md = render_markdown([_one_real_report()], hf_model=Path("artifacts/hf-x"), label="x",
                         num_samples=2, max_new_tokens=60, temperature=0.8, top_p=0.95,
                         seed=0, eos_id=2, controls=controls, control_excerpt_words=52,
                         register_sources=["tinystories", "spine"])
    assert "Detector controls" in md
    assert "Per marker" in md
    assert "48.00%" in md          # sensitivity
    assert "0.30%" in md           # false-positive rate on a non-collapse source
    assert "`once_upon_a_time` | frame" in md


def test_json_carries_per_prompt_values_so_a_later_run_can_pair_against_it():
    payload = report_to_json([_one_real_report()], hf_model="artifacts/hf-x", label="x",
                             num_samples=2, max_new_tokens=60, temperature=0.8, top_p=0.95,
                             seed=0, eos_id=2, controls=None, control_excerpt_words=None,
                             register_sources=None)
    assert payload["per_prompt"]["voice-01"]["estimates"]["collapse_rate"]["n"] == 2
    assert payload["aggregate"]["termination_rate"]["mean"] == pytest.approx(0.5)
    # Round-trips through JSON, and pairs with itself to a difference of exactly zero.
    reloaded = json.loads(json.dumps(payload))
    for diff in paired_differences(reloaded, reloaded):
        if diff.difference is not None:
            assert diff.difference.mean == 0.0
            assert diff.verdict in ("no change", "n/a")


def test_default_label_strips_the_hf_prefix_like_the_other_measurement_scripts():
    assert default_label(Path("artifacts/hf-tt-tnt-v3")) == "tt-tnt-v3"
    assert default_label(Path("artifacts/some-model")) == "some-model"


def test_a_missing_model_directory_fails_before_any_expensive_import(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such directory"):
        resolve_model_dir(tmp_path / "absent")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no config.json"):
        resolve_model_dir(tmp_path / "empty")


def test_the_eos_id_comes_from_the_models_own_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": 7}))
    assert resolve_eos_id(tmp_path) == 7
    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": [7]}))
    assert resolve_eos_id(tmp_path) == 7
    (tmp_path / "config.json").write_text(json.dumps({}))
    with pytest.raises(ValueError, match="no eos_token_id"):
        resolve_eos_id(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": [2, 7]}))
    with pytest.raises(ValueError, match="single termination token"):
        resolve_eos_id(tmp_path)


# ---------------------------------------------------------------------------------------
# The frozen prompt set is read, never written
# ---------------------------------------------------------------------------------------


def test_the_frozen_prompt_set_is_loaded_intact():
    prompts = load_prompts()
    assert len(prompts) == 15
    ids = [p["id"] for p in prompts]
    assert ids[0] == "voice-01" and "stutter-01" in ids and "stutter-02" in ids
    for prompt in prompts:
        assert {"id", "probe", "text"} <= set(prompt)


def test_the_stutter_probe_tag_this_script_excludes_actually_exists_in_the_prompt_set():
    # If the prompt set were ever re-tagged, the "excluding deliberate repetition" aggregate
    # would silently become a duplicate of the plain one. This is the tripwire for that.
    probes = {p["probe"] for p in load_prompts()}
    assert "stutter" in probes


def test_score_behaviour_does_not_write_the_frozen_prompt_set():
    source = (ROOT / "scripts" / "score_behaviour.py").read_text()
    assert "evaluation_prompts.json" in source
    for forbidden in ("PROMPTS_PATH.write_text", "PROMPTS_PATH.open(\"w\"", "json.dump("):
        assert forbidden not in source, forbidden


def test_score_behaviour_never_writes_under_artifacts():
    # artifacts/ is read-only to this tool: corpora and weights go in, nothing comes back out.
    source = (ROOT / "scripts" / "score_behaviour.py").read_text()
    for line in source.splitlines():
        if "write_text" in line or "mkdir(" in line:
            assert "artifacts" not in line, line


def test_score_behaviour_never_touches_tenstorrent_hardware():
    source = (ROOT / "scripts" / "score_behaviour.py").read_text()
    for forbidden in ("import ttnn", "import ttml", "tt_smi", "tt-smi"):
        assert forbidden not in source, forbidden


def test_production_code_uses_no_bare_assert():
    source = (ROOT / "scripts" / "score_behaviour.py").read_text()
    for number, line in enumerate(source.splitlines(), start=1):
        assert not re.match(r"\s*assert\b", line), f"bare assert at line {number}: {line}"


# ---------------------------------------------------------------------------------------
# End-to-end against a tiny real model -- skipped explicitly if torch is absent
# ---------------------------------------------------------------------------------------


@pytest.fixture
def tiny_model(tmp_path):
    """A random-initialised 2-layer Llama, so the generation path is exercised for real.

    Skipped -- with a reason visible in the pytest report -- when torch/transformers are not
    installed, rather than passing vacuously. Deliberately does NOT depend on
    artifacts/hf-tt-tnt-* existing: the suite must be meaningful on a machine with no trained
    model at all.
    """
    torch = pytest.importorskip(
        "torch", reason="torch is not installed in this environment; skipping the "
                        "generation-path tests explicitly rather than passing vacuously")
    transformers = pytest.importorskip(
        "transformers", reason="transformers is not installed in this environment")
    config = transformers.LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        bos_token_id=1, eos_token_id=2, pad_token_id=3)
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).eval()
    model.save_pretrained(tmp_path)
    return tmp_path


def test_a_saved_tiny_model_passes_the_pre_import_validation(tiny_model):
    assert resolve_model_dir(tiny_model) == tiny_model
    assert resolve_eos_id(tiny_model) == 2


def test_generation_produces_the_requested_number_of_scoreable_completions(tiny_model):
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM

    from scripts.score_behaviour import generate_completions

    class _IdTokenizer:
        """Minimal stand-in: the tiny model has no tokenizer, and this test is about the
        generate/slice/decode plumbing, not about tokenisation."""

        def __call__(self, text, return_tensors=None):
            ids = torch.tensor([[4, 5, 6]])
            return type("Enc", (), {"input_ids": ids})()

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(f"w{int(i)}" for i in ids if int(i) not in (1, 2, 3))

    model = AutoModelForCausalLM.from_pretrained(str(tiny_model)).eval()
    torch.manual_seed(0)
    out = generate_completions(model, _IdTokenizer(), "anything", num_samples=6,
                               max_new_tokens=8, temperature=0.8, top_p=0.95, pad_token_id=3)
    assert len(out) == 6
    for text, ids in out:
        assert len(ids) == 8            # prompt sliced off, budget respected
        score = score_completion("anything", text, ids)
        assert score.n_tokens <= 8
        assert isinstance(score.collapsed, bool)


def test_detector_controls_report_sensitivity_on_tinystories_and_specificity_elsewhere(
        tmp_path):
    """The metric's own validation, in miniature: known-collapsed text vs known-clean text."""
    (tmp_path / "tinystories.txt").write_text(
        " ".join(["Once upon a time there was a little girl named Lily and she was so happy."]
                 * 40))
    (tmp_path / "spine.txt").write_text(
        " ".join(["The ant colony repaired the trench before the light had wholly gone."] * 40))
    profile = build_register_profile(tmp_path, ["spine", "tinystories"], train_words=200,
                                     control_words=400)
    rows = {row.source: row for row in detector_controls(profile, excerpt_words=14)}
    assert rows["tinystories"].collapse_rate > 0.5     # sensitivity
    assert rows["spine"].collapse_rate == 0.0          # specificity
    assert rows["tinystories"].register_accuracy > 0.9
    assert rows["spine"].register_accuracy > 0.9


def test_detector_controls_reject_a_nonsense_excerpt_length():
    with pytest.raises(ValueError, match="excerpt_words"):
        detector_controls(_toy_profile(), excerpt_words=0)
