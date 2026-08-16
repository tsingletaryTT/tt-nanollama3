# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for scripts/benchmark_external.py.

The thing worth testing hardest here is the **chance-baseline labelling rule**, because it is
what stops this project from quoting a coin flip as a capability. Everything about it is pure
arithmetic over numbers lm-eval already computed, so all of it is exercised on hand-built
inputs with no venv, no model, no torch and no network: :func:`chance_verdict`,
:func:`reportable_score`, :func:`headline`, and the fact that an ``AT CHANCE`` row's
``reportable_score`` is ``null`` in the JSON and that the markdown never states its number as
a finding.

Report generation is likewise tested against synthetic lm-eval results dicts, and the
truncation analysis against a synthetic ``--log_samples`` JSONL with a stub tokenizer -- so
"did the 512-token window cut this prompt off" is verified without loading a 123M-parameter
model.

The handful of tests that genuinely need the external venv (its package versions, and the
WikiText-2/WikiText-103 test-split identity claim the report makes) skip with an explicit
reason naming the missing venv, in the convention of tests/test_probe_context_use.py: this
suite must still pass, non-vacuously, on a machine that has never installed lm-eval.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_external import (  # noqa: E402
    ABOVE_CHANCE,
    AT_CHANCE,
    BELOW_CHANCE,
    CHANCE_SE_MULTIPLE,
    DEFAULT_VENV,
    GPT2_PUBLISHED,
    NO_CHANCE_BASELINE,
    NO_STDERR,
    TASKS,
    TASKS_BY_NAME,
    WIKITEXT_TEST_CHARS,
    Metric,
    ModelFacts,
    ReportInputs,
    TaskSpec,
    TruncationReport,
    VenvProvenance,
    analyse_truncation,
    as_float,
    build_metric_results,
    chance_verdict,
    context_window,
    default_label,
    find_results_json,
    find_sample_paths,
    headline,
    iter_request_texts,
    lm_eval_command,
    missing_tasks,
    parse_training_tokens,
    read_model_facts,
    reference_lookup,
    render_markdown,
    report_to_json,
    reportable_score,
    require_cpu_device,
    venv_provenance,
    venv_python,
    verify_wikitext_test_identity,
    z_from_chance,
)


# ---------------------------------------------------------------------------------------
# chance_verdict -- the rule this whole script exists to enforce
# ---------------------------------------------------------------------------------------


def test_a_score_sitting_on_chance_is_at_chance():
    # 4-way multiple choice, dead on 0.25.
    assert chance_verdict(0.25, 0.25, 0.004) == AT_CHANCE


def test_a_score_just_inside_the_gate_is_at_chance():
    # 1.9 standard errors above chance -- significant-looking, and still not reportable.
    assert chance_verdict(0.25 + 1.9 * 0.004, 0.25, 0.004) == AT_CHANCE


def test_a_score_just_outside_the_gate_clears_it():
    assert chance_verdict(0.25 + 2.1 * 0.004, 0.25, 0.004) == ABOVE_CHANCE


def test_the_gate_is_exactly_the_documented_multiple():
    # Exactly CHANCE_SE_MULTIPLE standard errors clears (the comparison is `<`), so the
    # constant in the module is the real boundary and not an approximate one.
    stderr = 0.01
    at = 0.5 + (CHANCE_SE_MULTIPLE - 1e-9) * stderr
    beyond = 0.5 + (CHANCE_SE_MULTIPLE + 1e-9) * stderr
    assert chance_verdict(at, 0.5, stderr) == AT_CHANCE
    assert chance_verdict(beyond, 0.5, stderr) == ABOVE_CHANCE


def test_a_score_far_below_chance_is_reported_as_below_not_hidden():
    # Anti-correlated length-normalised scoring is a real outcome, not a bug to suppress.
    assert chance_verdict(0.20, 0.25, 0.005) == BELOW_CHANCE


def test_a_continuous_metric_has_no_chance_baseline():
    assert chance_verdict(199.5, None, None) == NO_CHANCE_BASELINE


def test_a_missing_standard_error_refuses_to_call_it():
    assert chance_verdict(0.31, 0.25, None) == NO_STDERR


def test_a_nan_standard_error_refuses_to_call_it():
    assert chance_verdict(0.31, 0.25, float("nan")) == NO_STDERR


def test_zero_stderr_exactly_on_chance_is_at_chance_not_uninterpretable():
    # 0/N correct on LAMBADA, whose chance baseline is ~0: unambiguously at chance.
    assert chance_verdict(0.0, 0.0, 0.0) == AT_CHANCE


def test_zero_stderr_off_chance_is_still_uninterpretable():
    assert chance_verdict(0.4, 0.25, 0.0) == NO_STDERR


def test_z_from_chance_matches_hand_arithmetic():
    assert z_from_chance(0.30, 0.25, 0.01) == pytest.approx(5.0)


def test_z_from_chance_is_none_without_a_usable_spread():
    assert z_from_chance(0.30, 0.25, 0.0) is None
    assert z_from_chance(0.30, None, 0.01) is None
    assert z_from_chance(None, 0.25, 0.01) is None


# ---------------------------------------------------------------------------------------
# The suppression itself: an at-chance score is not available as a quantity
# ---------------------------------------------------------------------------------------


def test_reportable_score_is_none_at_chance():
    assert reportable_score(0.2504, AT_CHANCE) is None


def test_reportable_score_is_none_without_a_standard_error():
    assert reportable_score(0.31, NO_STDERR) is None


def test_reportable_score_survives_a_real_result():
    assert reportable_score(0.3312, ABOVE_CHANCE) == 0.3312
    assert reportable_score(199.52, NO_CHANCE_BASELINE) == 199.52


def test_headline_says_the_words_not_the_number():
    assert headline(0.2504, AT_CHANCE) == "at chance"
    assert "0.25" not in headline(0.2504, AT_CHANCE)


def test_headline_prints_a_real_result():
    assert headline(0.3312, ABOVE_CHANCE) == "0.3312"


def test_headline_handles_a_missing_score():
    assert headline(None, NO_CHANCE_BASELINE) == "n/a"


# ---------------------------------------------------------------------------------------
# The fixed task list is internally coherent
# ---------------------------------------------------------------------------------------


def test_every_task_declares_a_chance_baseline_or_is_continuous():
    for spec in TASKS:
        for metric in spec.metrics:
            if metric.chance is not None:
                assert 0.0 <= metric.chance < 1.0, f"{spec.task}/{metric.key}"


def test_multiple_choice_chance_baselines_are_the_expected_ones():
    expected = {
        "hellaswag": 0.25, "arc_easy": 0.25, "arc_challenge": 0.25, "mmlu": 0.25,
        "piqa": 0.5, "winogrande": 0.5,
    }
    for task, chance in expected.items():
        accuracies = [m for m in TASKS_BY_NAME[task].metrics if m.chance is not None]
        assert accuracies, task
        for metric in accuracies:
            assert metric.chance == chance, f"{task}/{metric.key}"


def test_perplexity_metrics_have_no_chance_baseline():
    for task in ("wikitext", "lambada_openai"):
        for metric in TASKS_BY_NAME[task].metrics:
            if "perplexity" in metric.key or metric.key == "bits_per_byte":
                assert metric.chance is None, f"{task}/{metric.key}"


def test_wikitext_is_the_rolling_task_and_the_others_are_not():
    assert TASKS_BY_NAME["wikitext"].rolling is True
    assert all(not s.rolling for s in TASKS if s.task != "wikitext")


def test_the_gpt2_reference_table_only_names_tasks_we_actually_run():
    for (task, metric) in GPT2_PUBLISHED:
        assert task in TASKS_BY_NAME, task
        assert metric in {m.key for m in TASKS_BY_NAME[task].metrics}, (task, metric)


# ---------------------------------------------------------------------------------------
# Parsing lm-eval's results dict
# ---------------------------------------------------------------------------------------


def _fake_results() -> dict:
    """A results dict shaped exactly like lm-eval 0.4.9's, with hand-chosen scores.

    piqa is 1.0 s.e. above its 0.5 chance (at chance); hellaswag is 10 s.e. above its 0.25
    chance (clears); wikitext is continuous.
    """
    return {
        "piqa": {"alias": "piqa", "acc,none": 0.505, "acc_stderr,none": 0.005,
                 "acc_norm,none": 0.51, "acc_norm_stderr,none": 0.005},
        "hellaswag": {"acc,none": 0.30, "acc_stderr,none": 0.005},
        "wikitext": {"word_perplexity,none": 199.5222, "byte_perplexity,none": 2.8133,
                     "bits_per_byte,none": 1.4923},
    }


def _specs(*names: str):
    return [TASKS_BY_NAME[n] for n in names]


def test_build_metric_results_reads_lm_evals_comma_suffixed_keys():
    rows = build_metric_results(_fake_results(), _specs("piqa"))
    by_metric = {r.metric: r for r in rows}
    assert by_metric["acc"].score == 0.505
    assert by_metric["acc"].stderr == 0.005
    assert by_metric["acc"].chance == 0.5


def test_build_metric_results_applies_the_chance_rule_per_row():
    rows = build_metric_results(_fake_results(), _specs("piqa", "hellaswag", "wikitext"))
    verdicts = {(r.task, r.metric): r.verdict for r in rows}
    assert verdicts[("piqa", "acc")] == AT_CHANCE          # 1.0 s.e. above chance
    assert verdicts[("piqa", "acc_norm")] == ABOVE_CHANCE  # 2.0 s.e. -> clears
    assert verdicts[("hellaswag", "acc")] == ABOVE_CHANCE  # 10 s.e.
    assert verdicts[("wikitext", "word_perplexity")] == NO_CHANCE_BASELINE


def test_a_task_absent_from_the_results_produces_no_row_and_is_named_as_missing():
    results = _fake_results()
    specs = _specs("piqa", "mmlu")
    rows = build_metric_results(results, specs)
    assert all(r.task != "mmlu" for r in rows)
    assert missing_tasks(results, specs) == ["mmlu"]


def test_missing_tasks_is_empty_when_everything_returned():
    assert missing_tasks(_fake_results(), _specs("piqa", "wikitext")) == []


def test_a_metric_the_harness_did_not_emit_is_skipped_rather_than_invented():
    # hellaswag's spec asks for acc AND acc_norm; the fake results only have acc.
    rows = build_metric_results(_fake_results(), _specs("hellaswag"))
    assert [r.metric for r in rows] == ["acc"]


# ---------------------------------------------------------------------------------------
# The GPT-2 column: measured beats published, and neither is invented
# ---------------------------------------------------------------------------------------


def test_reference_lookup_falls_back_to_the_published_figure():
    value, kind, source, caveat = reference_lookup("wikitext", "word_perplexity", None)
    assert value == pytest.approx(37.50)
    assert kind == "published"
    assert "Radford" in source
    assert caveat


def test_reference_lookup_prefers_a_measured_run():
    reference = {
        "model": {"label": "gpt2-small"},
        "harness": {"lm_eval_version": "0.4.9", "max_length": 1024},
        "results": [{"task": "wikitext", "metric": "word_perplexity", "score": 33.1}],
    }
    value, kind, source, _ = reference_lookup("wikitext", "word_perplexity", reference)
    assert value == 33.1
    assert kind == "measured"
    assert "gpt2-small" in source
    assert "0.4.9" in source
    # The reference's own context window must travel with the number.
    assert "1024-token context" in source


def test_reference_lookup_returns_nothing_when_nothing_is_known():
    assert reference_lookup("piqa", "acc", None) == (None, "", "", "")


def test_reference_lookup_does_not_invent_a_measured_row_for_another_task():
    reference = {"model": {"label": "gpt2-small"}, "harness": {},
                 "results": [{"task": "piqa", "metric": "acc", "score": 0.62}]}
    value, kind, _, _ = reference_lookup("hellaswag", "acc", reference)
    assert (value, kind) == (None, "")


# ---------------------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------------------


def _inputs(**overrides) -> ReportInputs:
    facts = ModelFacts(
        path=Path("artifacts/hf-tt-tnt-1024a"), label="tt-tnt-1024a",
        max_position_embeddings=512, hidden_size=1024, num_hidden_layers=8,
        training_tokens=352_714_752, training_tokens_note="10,764 steps x batch 64 x seq_len 512",
    )
    rows = build_metric_results(_fake_results(), _specs("wikitext", "piqa", "hellaswag"))
    defaults = dict(
        model=facts, rows=rows,
        truncation=[
            TruncationReport("wikitext", 62, 0, 8228, 512, True, note="rolling note"),
            TruncationReport("piqa", 3676, 0, 61, 512, False),
            TruncationReport("hellaswag", 40168, 17, 640, 512, False),
        ],
        venv=VenvProvenance(python=Path("/tmp/venv/bin/python"),
                            versions={"lm_eval": "0.4.9", "torch": "2.7.0+cpu"}),
        command=["/tmp/venv/bin/python", "-m", "lm_eval", "--tasks", "piqa"],
        lm_eval_version="0.4.9", n_params=122_962_944, max_length=512, batch_size=16,
        dtype="float32", limit=None, device="cpu",
        results_json=Path("/tmp/run/results_x.json"),
        wikitext_identity="Verified at run time: ... byte-identical ...",
    )
    defaults.update(overrides)
    return ReportInputs(**defaults)


def test_markdown_marks_an_at_chance_row_in_bold_in_the_table():
    md = render_markdown(_inputs())
    piqa_rows = [line for line in md.splitlines()
                 if line.startswith("| piqa | accuracy |")]
    assert len(piqa_rows) == 1
    assert f"**{AT_CHANCE}**" in piqa_rows[0]


def test_markdown_names_at_chance_rows_in_words_without_quoting_their_score():
    md = render_markdown(_inputs())
    section = md.split("## What this says, in words")[1].split("## Did the context")[0]
    at_chance_line = [l for l in section.splitlines() if l.startswith("**At chance**")][0]
    assert "`piqa` accuracy" in at_chance_line
    # The suppressed number itself must not appear as a claim in the prose.
    assert "0.505" not in at_chance_line


def test_markdown_states_the_chance_baseline_for_every_multiple_choice_row():
    md = render_markdown(_inputs())
    row = [l for l in md.splitlines() if l.startswith("| piqa | accuracy |")][0]
    assert "| 0.50 |" in row


def test_markdown_flags_a_truncated_task_loudly():
    md = render_markdown(_inputs())
    hellaswag_row = [l for l in md.splitlines()
                     if l.startswith("| hellaswag |") and "| 17 |" in l][0]
    assert "NO — truncated" in hellaswag_row


def test_markdown_does_not_call_an_untruncated_task_truncated():
    md = render_markdown(_inputs())
    piqa_row = [l for l in md.splitlines()
                if l.startswith("| piqa |") and "3,676" in l][0]
    assert piqa_row.rstrip().endswith("| yes |")


def test_markdown_describes_a_rolling_task_as_windowed_not_truncated():
    md = render_markdown(_inputs())
    wikitext_row = [l for l in md.splitlines()
                    if l.startswith("| wikitext |") and "8,228" in l][0]
    assert "windowed, not truncated" in wikitext_row


def test_markdown_records_the_data_gap_and_the_venv():
    md = render_markdown(_inputs())
    assert "122,962,944" in md
    assert "352,714,752" in md
    assert "40,000,000,000" in md
    assert "/tmp/venv/bin/python" in md
    assert "0.4.9" in md


def test_markdown_says_when_training_tokens_could_not_be_established():
    facts = ModelFacts(path=Path("x"), label="x", max_position_embeddings=512,
                       hidden_size=1, num_hidden_layers=1, training_tokens=None,
                       training_tokens_note="no train.log at x/train.log")
    md = render_markdown(_inputs(model=facts))
    assert "Training tokens could not be established: no train.log" in md
    assert "| training tokens | unknown |" in md


def test_markdown_names_a_missing_task_rather_than_omitting_it():
    md = render_markdown(_inputs(missing=["mmlu"]))
    assert "**Requested but absent from the harness output:** mmlu" in md
    assert "A missing task is a failed task" in md


def test_markdown_accounts_for_every_row_in_the_prose():
    # Nothing may fall out of the "in words" section: a reader who skipped the table must
    # still see every task named under one of the four headings.
    results = dict(_fake_results())
    results["winogrande"] = {"acc,none": 0.51, "acc_stderr,none": 0.0}   # -> NO STANDARD ERROR
    rows = build_metric_results(results, _specs("wikitext", "piqa", "hellaswag", "winogrande"))
    md = render_markdown(_inputs(rows=rows))
    section = md.split("## What this says, in words")[1].split("## Did the context")[0]
    for row in rows:
        assert f"`{row.task}`" in section, row.task
    assert "Not interpretable" in section


def test_markdown_shouts_about_limit_when_one_was_used():
    md = render_markdown(_inputs(limit=100))
    assert "**--limit 100**" in md


def test_a_reference_run_does_not_compare_gpt2_to_itself():
    md = render_markdown(_inputs(is_reference_run=True))
    assert "This is a reference run" in md
    assert "the gap to its reference class" not in md
    assert "first this project has that our corpus did not produce" not in md


def test_json_drops_the_score_of_an_at_chance_row():
    payload = report_to_json(_inputs())
    rows = {(r["task"], r["metric"]): r for r in payload["results"]}
    piqa = rows[("piqa", "acc")]
    assert piqa["verdict"] == AT_CHANCE
    assert piqa["reportable_score"] is None
    # ...but the raw value is still auditable.
    assert piqa["score"] == 0.505
    assert piqa["headline"] == "at chance"


def test_json_keeps_the_score_of_a_row_that_cleared_chance():
    payload = report_to_json(_inputs())
    rows = {(r["task"], r["metric"]): r for r in payload["results"]}
    assert rows[("hellaswag", "acc")]["reportable_score"] == 0.30
    assert rows[("wikitext", "word_perplexity")]["reportable_score"] == pytest.approx(199.5222)


def test_json_records_the_rule_it_applied():
    payload = report_to_json(_inputs())
    assert payload["chance_rule"]["se_multiple"] == CHANCE_SE_MULTIPLE
    assert payload["schema"] == "tt-tnt/external-benchmark/1"


def test_json_records_truncation_per_task():
    payload = report_to_json(_inputs())
    trunc = {t["task"]: t for t in payload["truncation"]}
    assert trunc["hellaswag"]["truncated"] is True
    assert trunc["piqa"]["truncated"] is False
    assert trunc["wikitext"]["rolling"] is True


def test_json_round_trips():
    # The report is consumed by --reference-json, so it has to actually be JSON.
    json.loads(json.dumps(report_to_json(_inputs())))


# ---------------------------------------------------------------------------------------
# Truncation analysis -- no model, no torch, a stub tokenizer
# ---------------------------------------------------------------------------------------


class _WhitespaceTokenizer:
    """One token per whitespace-separated word. Enough to test the arithmetic exactly."""

    def __call__(self, text):
        return {"input_ids": text.split()}


def _write_samples(path: Path, pairs):
    """Write a --log_samples-shaped JSONL: one line per document, N requests per line."""
    with path.open("w", encoding="utf-8") as handle:
        for options in pairs:
            arguments = {f"gen_args_{i}": {"arg_0": ctx, "arg_1": cont}
                         for i, (ctx, cont) in enumerate(options)}
            handle.write(json.dumps({"doc_id": 0, "arguments": arguments}) + "\n")


def test_analyse_truncation_counts_requests_over_the_window(tmp_path):
    samples = tmp_path / "samples_piqa_x.jsonl"
    # max_length 5 -> a request is truncated when context+continuation > 6 tokens.
    _write_samples(samples, [
        [("a b c", " d"), ("a b c", " d e")],       # 4 and 5 tokens: fine
        [("a b c d e f", " g"), ("a b", " c")],     # 7 tokens: truncated; 3: fine
    ])
    report = analyse_truncation([samples], _WhitespaceTokenizer(), task="piqa", max_length=5,
                                rolling=False)
    assert report.n_requests == 4
    assert report.n_truncated == 1
    assert report.max_tokens == 7
    assert report.truncated is True


def test_analyse_truncation_boundary_is_max_length_plus_one(tmp_path):
    # lm-eval scores (context + continuation)[-(max_length + 1):-1], so exactly
    # max_length + 1 tokens still fits and one more does not.
    samples = tmp_path / "samples_piqa_x.jsonl"
    _write_samples(samples, [[(" ".join("abcde"), " f")]])   # 5 + 1 = 6 = max_length + 1
    fits = analyse_truncation([samples], _WhitespaceTokenizer(), task="piqa", max_length=5,
                              rolling=False)
    assert fits.n_truncated == 0
    _write_samples(samples, [[(" ".join("abcdef"), " g")]])  # 7 tokens
    over = analyse_truncation([samples], _WhitespaceTokenizer(), task="piqa", max_length=5,
                              rolling=False)
    assert over.n_truncated == 1


def test_analyse_truncation_reports_zero_truncation_for_a_rolling_task(tmp_path):
    samples = tmp_path / "samples_wikitext_x.jsonl"
    _write_samples(samples, [[(" ".join(str(i) for i in range(50)), "")]])
    report = analyse_truncation([samples], _WhitespaceTokenizer(), task="wikitext", max_length=5,
                               rolling=True)
    assert report.n_truncated == 0
    assert report.max_tokens == 50          # the long document is still recorded honestly
    assert "windows" in report.note


def test_analyse_truncation_says_so_when_there_is_no_sample_log(tmp_path):
    report = analyse_truncation([tmp_path / "absent.jsonl"], _WhitespaceTokenizer(), task="piqa",
                                max_length=512, rolling=False)
    assert report.n_requests == 0
    assert "truncation could not be checked" in report.note
    # Crucially it does NOT claim the task was fine.
    assert report.truncated is False and report.note != ""


def test_iter_request_texts_reads_the_dict_shape():
    sample = {"arguments": {"gen_args_0": {"arg_0": "ctx", "arg_1": " cont"}}}
    assert list(iter_request_texts(sample)) == [("ctx", " cont")]


def test_iter_request_texts_reads_the_older_list_shape():
    sample = {"arguments": [["ctx", " cont"], ["ctx2", " cont2"]]}
    assert list(iter_request_texts(sample)) == [("ctx", " cont"), ("ctx2", " cont2")]


def test_iter_request_texts_handles_a_rolling_request_with_no_continuation():
    sample = {"arguments": {"gen_args_0": {"arg_0": "the whole document"}}}
    assert list(iter_request_texts(sample)) == [("the whole document", "")]


def test_iter_request_texts_refuses_an_unrecognised_shape():
    with pytest.raises(ValueError, match="no usable 'arguments'"):
        list(iter_request_texts({"arguments": "not a container"}))


# ---------------------------------------------------------------------------------------
# Model facts, read from disk only
# ---------------------------------------------------------------------------------------


def test_default_label_strips_the_hf_prefix():
    assert default_label(Path("artifacts/hf-tt-tnt-1024a")) == "tt-tnt-1024a"
    assert default_label(Path("scratch/hf-gpt2")) == "gpt2"
    assert default_label(Path("somewhere/my-model")) == "my-model"


def test_context_window_prefers_max_position_embeddings():
    assert context_window({"max_position_embeddings": 512, "n_positions": 1024}) == 512


def test_context_window_falls_back_to_gpt2s_spelling():
    assert context_window({"n_positions": 1024}) == 1024
    assert context_window({"n_ctx": 2048}) == 2048


def test_context_window_is_none_when_the_config_states_nothing():
    assert context_window({"hidden_size": 768}) is None


def test_read_model_facts_refuses_a_config_with_no_context_window(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 768}))
    with pytest.raises(ValueError, match="states no context window"):
        read_model_facts(tmp_path)


def test_read_model_facts_refuses_a_missing_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="no config.json"):
        read_model_facts(tmp_path / "nope")


def test_read_model_facts_reads_a_llama_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(
        {"max_position_embeddings": 512, "hidden_size": 1024, "num_hidden_layers": 8}))
    facts = read_model_facts(tmp_path)
    assert facts.max_position_embeddings == 512
    assert facts.hidden_size == 1024
    assert facts.num_hidden_layers == 8
    # No sibling checkpoints-* directory -> tokens unknown, and said so.
    assert facts.training_tokens is None
    assert "checkpoint" in facts.training_tokens_note


# ---------------------------------------------------------------------------------------
# Training-token provenance
# ---------------------------------------------------------------------------------------


def test_parse_training_tokens_multiplies_the_summary_line(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("noise\ntt-tnt training — steps=10764 batch=64 seq_len=512 arch=blackhole\n")
    tokens, note = parse_training_tokens(log)
    assert tokens == 10764 * 64 * 512
    assert "10,764 steps" in note


def test_parse_training_tokens_reports_a_missing_log_rather_than_guessing(tmp_path):
    tokens, note = parse_training_tokens(tmp_path / "absent.log")
    assert tokens is None
    assert "no train.log" in note


def test_parse_training_tokens_reports_an_incomplete_summary_line(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("tt-tnt training — steps=100 arch=blackhole\n")
    tokens, note = parse_training_tokens(log)
    assert tokens is None
    assert "batch" in note and "seq_len" in note


def test_parse_training_tokens_reports_an_unparseable_field(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("tt-tnt training — steps=lots batch=64 seq_len=512\n")
    tokens, note = parse_training_tokens(log)
    assert tokens is None
    assert "unparseable" in note


# ---------------------------------------------------------------------------------------
# The command, and the CPU-only refusal
# ---------------------------------------------------------------------------------------


def test_lm_eval_command_passes_the_window_explicitly():
    argv = lm_eval_command(Path("/v/bin/python"), Path("/m"), ["piqa", "wikitext"],
                           max_length=512, batch_size=16, dtype="float32",
                           output_path=Path("/out"))
    joined = " ".join(argv)
    assert "pretrained=/m,dtype=float32,max_length=512" in joined
    assert "--tasks piqa,wikitext" in joined
    assert "--device cpu" in joined
    assert "--log_samples" in argv


def test_lm_eval_command_omits_limit_unless_asked():
    argv = lm_eval_command(Path("/v/bin/python"), Path("/m"), ["piqa"], max_length=512,
                           batch_size=16, dtype="float32", output_path=Path("/out"))
    assert "--limit" not in argv
    argv = lm_eval_command(Path("/v/bin/python"), Path("/m"), ["piqa"], max_length=512,
                           batch_size=16, dtype="float32", output_path=Path("/out"), limit=5)
    assert argv[argv.index("--limit") + 1] == "5"


def test_require_cpu_device_refuses_anything_else():
    assert require_cpu_device("cpu") == "cpu"
    for device in ("cuda", "tt", "cuda:0"):
        with pytest.raises(ValueError, match="CPU-only by design"):
            require_cpu_device(device)


def test_lm_eval_command_cannot_be_pointed_at_a_device():
    with pytest.raises(ValueError, match="CPU-only by design"):
        lm_eval_command(Path("/v/bin/python"), Path("/m"), ["piqa"], max_length=512,
                        batch_size=16, dtype="float32", output_path=Path("/out"),
                        device="cuda")


def test_venv_python_error_message_says_how_to_build_the_venv(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        venv_python(tmp_path / "no-such-venv")
    message = str(exc.value)
    assert "lm-eval" in message
    assert "Do NOT install lm-eval into this project's environment." in message


# ---------------------------------------------------------------------------------------
# Locating lm-eval's own output
# ---------------------------------------------------------------------------------------


def test_find_results_json_picks_the_newest(tmp_path):
    nested = tmp_path / "model"
    nested.mkdir()
    old = nested / "results_2020.json"
    new = nested / "results_2026.json"
    old.write_text("{}")
    new.write_text("{}")
    import os
    os.utime(old, (1, 1))
    assert find_results_json(tmp_path) == new


def test_find_results_json_raises_on_an_empty_run_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="did not complete a run"):
        find_results_json(tmp_path)


def test_find_sample_paths_matches_the_task(tmp_path):
    results = tmp_path / "results_x.json"
    results.write_text("{}")
    (tmp_path / "samples_piqa_x.jsonl").write_text("")
    assert [p.name for p in find_sample_paths(results, "piqa")] == ["samples_piqa_x.jsonl"]
    assert find_sample_paths(results, "mmlu") == []


def test_find_sample_paths_collects_every_subtask_of_a_group(tmp_path):
    # The MMLU bug: a group task writes one log per subtask, and checking only one of them
    # would report the other 56 as fair without having looked at them.
    results = tmp_path / "results_x.json"
    results.write_text("{}")
    for subject in ("anatomy", "astronomy", "professional_law"):
        (tmp_path / f"samples_mmlu_{subject}_x.jsonl").write_text("")
    found = [p.name for p in find_sample_paths(results, "mmlu")]
    assert len(found) == 3
    assert "samples_mmlu_professional_law_x.jsonl" in found


def test_find_sample_paths_does_not_mix_two_runs_in_the_same_directory(tmp_path):
    (tmp_path / "results_run1.json").write_text("{}")
    (tmp_path / "results_run2.json").write_text("{}")
    (tmp_path / "samples_mmlu_anatomy_run1.jsonl").write_text("")
    (tmp_path / "samples_mmlu_anatomy_run2.jsonl").write_text("")
    found = find_sample_paths(tmp_path / "results_run2.json", "mmlu")
    assert [p.name for p in found] == ["samples_mmlu_anatomy_run2.jsonl"]


def test_analyse_truncation_aggregates_across_a_groups_subtask_logs(tmp_path):
    a = tmp_path / "samples_mmlu_anatomy_x.jsonl"
    b = tmp_path / "samples_mmlu_professional_law_x.jsonl"
    _write_samples(a, [[("a b", " c")]])                       # 3 tokens: fine
    _write_samples(b, [[("a b c d e f g h", " i")]])           # 9 tokens: truncated
    report = analyse_truncation([a, b], _WhitespaceTokenizer(), task="mmlu", max_length=5,
                                rolling=False)
    assert report.n_requests == 2
    assert report.n_truncated == 1
    assert report.max_tokens == 9


# ---------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------


def test_as_float_coerces_lm_evals_stringified_numbers():
    assert as_float("16.4") == pytest.approx(16.4)
    assert as_float(3) == 3.0
    assert as_float(None) is None
    assert as_float("not a number") is None


def test_venv_provenance_reads_versions_through_the_injected_runner():
    class _Result:
        returncode = 0
        stdout = json.dumps({"lm_eval": "0.4.9", "torch": "2.7.0+cpu"})
        stderr = ""

    def _runner(argv, **kwargs):
        assert argv[0] == "/v/bin/python"
        return _Result()

    provenance = venv_provenance(Path("/v/bin/python"), ("lm_eval", "torch"), runner=_runner)
    assert provenance.versions["lm_eval"] == "0.4.9"


def test_venv_provenance_raises_when_the_interpreter_fails():
    class _Result:
        returncode = 1
        stdout = ""
        stderr = "boom"

    with pytest.raises(RuntimeError, match="could not read package versions"):
        venv_provenance(Path("/v/bin/python"), ("lm_eval",), runner=lambda *a, **k: _Result())


def test_wikitext_identity_check_never_returns_a_false_confirmation():
    class _Failed:
        returncode = 1
        stdout = ""
        stderr = "ImportError: no datasets"

    sentence = verify_wikitext_test_identity(Path("/v/bin/python"),
                                             runner=lambda *a, **k: _Failed())
    assert "UNVERIFIED" in sentence


def test_wikitext_identity_check_reports_a_negative_result_as_a_warning():
    class _Different:
        returncode = 0
        stdout = json.dumps({"identical": False, "chars": 1})
        stderr = ""

    sentence = verify_wikitext_test_identity(Path("/v/bin/python"),
                                             runner=lambda *a, **k: _Different())
    assert "NOT identical" in sentence
    assert "do not read the wikitext row as WikiText-103" in sentence


def test_wikitext_identity_check_confirms_a_positive_result():
    class _Same:
        returncode = 0
        stdout = json.dumps({"identical": True, "chars": WIKITEXT_TEST_CHARS})
        stderr = ""

    sentence = verify_wikitext_test_identity(Path("/v/bin/python"),
                                             runner=lambda *a, **k: _Same())
    assert "byte-identical" in sentence
    assert f"{WIKITEXT_TEST_CHARS:,}" in sentence


# ---------------------------------------------------------------------------------------
# Tests that genuinely need the external venv -- skipped with a reason, never vacuous
# ---------------------------------------------------------------------------------------

_venv_missing = not (DEFAULT_VENV / "bin" / "python").is_file()
_needs_venv = pytest.mark.skipif(
    _venv_missing,
    reason=f"no lm-eval venv at {DEFAULT_VENV} (it is deliberately not a dependency of this "
           f"repo; see scripts/benchmark_external.py's docstring for how to build one)")


@_needs_venv
def test_the_real_venv_has_lm_eval_installed():
    provenance = venv_provenance(venv_python(DEFAULT_VENV))
    assert not provenance.versions["lm_eval"].startswith("not installed"), provenance.versions


@_needs_venv
def test_wikitext2_and_wikitext103_test_splits_really_are_identical():
    """The report's load-bearing claim, checked against the real datasets.

    If this ever fails, the `wikitext` row in every external-*.md stops being WikiText-103
    perplexity and the comparison to the published literature has to be withdrawn.
    """
    sentence = verify_wikitext_test_identity(venv_python(DEFAULT_VENV))
    if "UNVERIFIED" in sentence:
        pytest.skip(f"datasets unavailable in the venv: {sentence}")
    assert "byte-identical" in sentence, sentence
    assert f"{WIKITEXT_TEST_CHARS:,}" in sentence, sentence
