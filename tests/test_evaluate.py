# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for scripts/evaluate.py -- the joined benchmark entry point.

WHAT THESE TESTS ARE FOR
------------------------
``evaluate.py`` exists because every significant error in this project came from JOINING
numbers, not from measuring them. So the tests that matter here are the ones that pin the
two guards which would have caught those errors, and both are **mutation-checked** (see
CLAUDE.md for the exact mutations applied and the failures they produced):

1. **The different-window refusal.** A 512-token-window loss and a 2048-token-window loss
   are not commensurable, and comparing them is what turned a -0.2994-nat capacity effect
   into "5.6x the parameters bought nothing". The refusal must fire, must name BOTH windows,
   and the default window must NOT be a function of the model.
2. **The seed-floor-ratio labelling.** A delta at or below ~1.2x the run-to-run floor is
   NOT INTERPRETABLE whatever its confidence interval says. The precedents are pinned by
   name and by number: 1.03x (an LR-decay register finding that was published and later
   refuted), 1.01x and 1.05x (two collapse signals correctly excluded), and the mirror-image
   2.99x that failed its own paired minimum-detectable difference.

NO MODEL IS REQUIRED. Everything here is filesystem/JSON/arithmetic, exercised against
synthetic fixtures built in ``tmp_path``: a "converted HF model" for this module's purposes
is a directory containing ``config.json``, which is exactly what ``read_model_facts`` reads.
The two tests that genuinely need this machine's trained checkpoints skip with an explicit
reason rather than passing vacuously.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import needs_artifacts

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate import (  # noqa: E402
    ADHOC_BANNER,
    assert_writable_out_dir,
    DEFAULT_WINDOW,
    FLOOR_BEHAVIOUR_JSON,
    FLOOR_RATIO_MIN,
    NO_FLOOR,
    NOT_INTERPRETABLE,
    Comparison,
    DesignationError,
    ModelFacts,
    ScratchPathViolation,
    SeedFloor,
    SingleEvaluation,
    TokenArray,
    WindowMismatch,
    adhoc_output_path,
    assert_scratch_path,
    checkpoint_dir_for,
    common_window,
    compare,
    compare_trajectories,
    derive_behaviour_floor,
    derive_seed_floor,
    floor_label,
    floor_ratio,
    label_differences,
    load_designation,
    pair_trajectories,
    pooled_window_loss,
    read_model_facts,
    read_prompt_file,
    read_val_losses,
    render_adhoc,
    render_comparison,
    render_single,
    require_matched_window,
    require_window_fits,
    sign_test,
)
from scripts.score_behaviour import Estimate, PairedDifference  # noqa: E402

CURRENT_MODEL_JSON = ROOT / "docs" / "current_model.json"


# ---------------------------------------------------------------------------------------
# Fixtures -- a "model" here is a directory with a config.json, which is all this reads
# ---------------------------------------------------------------------------------------


def make_model(tmp_path: Path, name: str, max_pos: int, *,
               losses=None, hidden=384) -> ModelFacts:
    """A synthetic converted-model directory, optionally with a checkpoint trajectory."""
    model_dir = tmp_path / f"hf-{name}"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps({
        "max_position_embeddings": max_pos, "hidden_size": hidden,
        "num_hidden_layers": 6, "vocab_size": 32000,
    }))
    if losses is not None:
        ckpt = tmp_path / f"checkpoints-{name}"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "val_losses.jsonl").write_text("\n".join(
            json.dumps({"step": step, "train_loss": loss, "val_loss": loss})
            for step, loss in losses) + "\n")
    return read_model_facts(model_dir)


def make_paired(signal: str, title: str, better: str, delta: float, sem: float,
                n: int = 45) -> PairedDifference:
    return PairedDifference(
        signal=signal, title=title, better=better,
        baseline=Estimate(mean=0.5, sem=0.01, n=n),
        candidate=Estimate(mean=0.5 + delta, sem=0.01, n=n),
        difference=Estimate(mean=delta, sem=sem, n=n), n_prompts=n)


def make_evaluation(facts: ModelFacts, *, window: int, behaviour=None,
                    buckets=None) -> SingleEvaluation:
    from scripts.evaluate import InstrumentRun

    runs = []
    if behaviour is not None:
        run = InstrumentRun(name="behaviour", argv=["python", "score_behaviour.py"],
                            json_path=Path("behaviour.json"))
        run.payload = behaviour
        run.status = "ok"
        runs.append(run)
    if buckets is not None:
        run = InstrumentRun(name="context", argv=["python", "probe_context_use.py"],
                            json_path=Path("context.json"))
        run.payload = {"overall": buckets, "seq_len": window}
        run.status = "ok"
        runs.append(run)
    return SingleEvaluation(
        facts=facts, window=window,
        tokens=TokenArray(path=Path("artifacts/tokens-v3/val_ids.npy"),
                          size_bytes=1, n_tokens=1),
        prompt_set="b", num_samples=32, n_windows=256, seed=0, runs=runs,
        designation=None, started_utc="2026-08-15T00:00:00+00:00")


def behaviour_payload(prompt_set: str, values: dict, prompt_ids=("b-null-01", "b-null-02")):
    """A minimal score_behaviour JSON: enough for paired_differences to pair on."""
    from scripts.score_behaviour import SIGNALS

    return {
        "hf_model": "artifacts/hf-x", "label": "x", "prompt_set": prompt_set,
        "num_samples": 32, "n_prompts": len(prompt_ids),
        "signals": [{"key": k, "title": t, "better": d} for k, t, d in SIGNALS],
        "aggregate": {},
        "per_prompt": {
            pid: {"probe": "default-register", "n_samples": 32,
                  "estimates": {k: {"mean": values.get(k, 0.0) + i * 1e-9,
                                    "sem": 0.001, "n": 32,
                                    "ci95": [0.0, 0.0]}
                                for k, _t, _d in SIGNALS},
                  "collapse_markers": {}, "register_nearest": {}}
            for i, pid in enumerate(prompt_ids)
        },
    }


# =======================================================================================
# GUARD 1 -- the different-window refusal
# =======================================================================================


def test_matched_windows_are_allowed_and_return_the_common_window():
    assert require_matched_window("a", 512, "b", 512, what="losses") == 512


def test_different_windows_are_refused():
    with pytest.raises(WindowMismatch):
        require_matched_window("tt-tnt-v3", 2048, "tt-tnt-1024a", 512, what="losses")


def test_the_refusal_names_BOTH_windows_and_BOTH_models():
    """A refusal that does not say which number is which sends the reader back to guess.

    Deliberately uses windows that appear nowhere else in the message (777/333, not the
    real 2048/512, which the message also quotes as the historical precedent), and requires
    each window to sit NEAR its own model's name -- otherwise "the two were measured at
    different windows" would pass while telling the reader nothing.
    """
    with pytest.raises(WindowMismatch) as exc:
        require_matched_window("model-alpha", 777, "model-beta", 333,
                               what="training validation losses")
    message = str(exc.value)
    assert re.search(r"model-alpha.{0,60}777", message), \
        "the refusal must attach 777 to model-alpha, not merely mention both"
    assert re.search(r"model-beta.{0,60}333", message), \
        "the refusal must attach 333 to model-beta, not merely mention both"
    assert "training validation losses" in message, "it must say WHAT it refused to compare"


def test_the_refusal_is_symmetric():
    """512-vs-2048 must fail exactly as 2048-vs-512 does; order is not a loophole."""
    for a, b in ((2048, 512), (512, 2048)):
        with pytest.raises(WindowMismatch):
            require_matched_window("a", a, "b", b, what="losses")


def test_a_window_longer_than_the_model_is_refused_naming_both_numbers():
    facts = ModelFacts(path=Path("x"), label="tt-tnt-1024a", max_position_embeddings=512,
                       hidden_size=1024, num_hidden_layers=8, checkpoint_dir=None)
    with pytest.raises(WindowMismatch) as exc:
        require_window_fits(facts, 2048)
    assert "2048" in str(exc.value) and "512" in str(exc.value)


def test_the_default_window_is_a_constant_not_the_models_own_context(tmp_path):
    """The mutation this guards against: ``window = capacity``.

    Defaulting the window to the model's ``max_position_embeddings`` is *exactly* the
    mechanism that produced this project's wrong headline -- the window rides along with the
    model, so two 'validation losses' silently answer different questions. A 2048-context
    model must still be evaluated at 512 by default.
    """
    long_context = make_model(tmp_path, "v3", 2048)
    assert long_context.max_position_embeddings == 2048
    assert common_window([long_context], None) == DEFAULT_WINDOW == 512


def test_the_default_window_narrows_to_the_smallest_model_but_never_widens(tmp_path):
    short = make_model(tmp_path, "1024a", 256)
    long = make_model(tmp_path, "v3", 2048)
    assert common_window([short, long], None) == 256, "narrow to what both can reach"
    assert common_window([long], None) == 512, "never widen past the constant"


def test_an_explicit_window_past_a_models_context_is_still_refused(tmp_path):
    short = make_model(tmp_path, "1024a", 512)
    long = make_model(tmp_path, "v3", 2048)
    with pytest.raises(WindowMismatch):
        common_window([short, long], 2048)


def test_training_window_is_read_from_max_position_embeddings(tmp_path):
    """``convert/to_hf.py`` sets max_position_embeddings from the header's seq_len, and
    ``evaluate()`` windows validation at that same seq_len -- so this field IS the units of
    val_losses.jsonl, not an approximation of them."""
    facts = make_model(tmp_path, "v3", 2048)
    assert facts.training_window == 2048


def test_trajectories_at_different_windows_are_refused_before_any_arithmetic(tmp_path):
    """The real shape of the mistake: v3 (2048) against 1024a (512)."""
    v3 = make_model(tmp_path, "v3", 2048, losses=[(500, 5.0), (1000, 4.0), (1500, 3.0)])
    a1024 = make_model(tmp_path, "1024a", 512,
                       losses=[(500, 4.5), (1000, 3.5), (1500, 2.9)])
    with pytest.raises(WindowMismatch) as exc:
        compare_trajectories(v3, a1024, None)
    assert "2048" in str(exc.value) and "512" in str(exc.value)


def test_trajectories_at_a_matched_window_are_compared(tmp_path):
    base = make_model(tmp_path, "384s512", 512, losses=[(500, 5.0), (1000, 4.0), (1500, 3.2)])
    cand = make_model(tmp_path, "1024a", 512, losses=[(500, 4.5), (1000, 3.5), (1500, 2.9)])
    result = compare_trajectories(base, cand, None)
    assert result.window == 512
    assert result.n_steps == 3
    assert result.final_delta == pytest.approx(-0.3)
    assert result.sign.n_negative == 3


def test_a_comparison_records_the_window_refusal_instead_of_silently_dropping_it(tmp_path):
    """A refused trajectory must leave a stated reason in the report, not a blank."""
    v3 = make_model(tmp_path, "v3", 2048, losses=[(500, 5.0), (1000, 4.0), (1500, 3.0)])
    a1024 = make_model(tmp_path, "1024a", 512,
                       losses=[(500, 4.5), (1000, 3.5), (1500, 2.9)])
    cmp = compare(make_evaluation(v3, window=512), make_evaluation(a1024, window=512),
                  floor=None, floor_problems=[], skip_trajectory=False)
    assert cmp.trajectory is None
    assert cmp.trajectory_refusal is not None
    assert "2048" in cmp.trajectory_refusal and "512" in cmp.trajectory_refusal
    rendered = render_comparison(cmp)
    assert "Not computed" in rendered
    assert "2048" in rendered


@needs_artifacts("artifacts/hf-tt-tnt-v3")
def test_the_cli_refuses_a_cross_window_comparison_with_a_nonzero_exit(tmp_path):
    """End to end, through argv: the guard must fire before any model is loaded."""
    make_model(tmp_path, "v3", 2048, losses=[(500, 5.0), (1000, 4.0), (1500, 3.0)])
    make_model(tmp_path, "1024a", 512, losses=[(500, 4.5), (1000, 3.5), (1500, 2.9)])
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate.py"),
         "--model", str(tmp_path / "hf-1024a"), "--against", str(tmp_path / "hf-v3"),
         "--out-dir", str(tmp_path / "out")],
        capture_output=True, text=True, cwd=str(ROOT))
    assert completed.returncode != 0, "a cross-window comparison must not exit 0"
    assert "2048" in completed.stderr and "512" in completed.stderr
    assert not (tmp_path / "out").exists(), "nothing may be written on a refusal"


def test_pair_trajectories_pairs_by_step_not_by_position():
    """Two runs of different lengths must not be zipped into step 500 vs step 1000."""
    a = [(500, 5.0), (1000, 4.0), (1500, 3.0)]
    b = [(1000, 4.4), (1500, 3.3)]
    assert pair_trajectories(a, b) == [(1000, 4.0, 4.4), (1500, 3.0, 3.3)]


def test_val_losses_are_read_with_their_steps(tmp_path):
    path = tmp_path / "val_losses.jsonl"
    path.write_text('{"step": 500, "val_loss": 5.0}\n\n{"step": 1000, "val_loss": 4.0}\n')
    assert read_val_losses(path) == [(500, 5.0), (1000, 4.0)]


def test_a_malformed_trajectory_line_raises_rather_than_being_skipped(tmp_path):
    path = tmp_path / "val_losses.jsonl"
    path.write_text('{"step": 500, "val_loss": 5.0}\n{"step": 1000}\n')
    with pytest.raises(ValueError):
        read_val_losses(path)


# =======================================================================================
# GUARD 2 -- the seed-floor ratio and its labelling
# =======================================================================================


def test_the_lr_decay_precedent_is_not_interpretable():
    """-0.041 at 1.03x the floor: it cleared its CI, was published, and was refuted.

    The CI verdict is deliberately passed as "better" here -- the whole point of the rule is
    that the floor OVERRIDES a significant interval.
    """
    assert floor_label(1.03, "better") == NOT_INTERPRETABLE


def test_the_two_collapse_signals_from_the_1024_run_are_not_interpretable():
    """1.01x and 1.05x came back "better" from the paired test and were correctly excluded."""
    assert floor_label(1.01, "better") == NOT_INTERPRETABLE
    assert floor_label(1.05, "better") == NOT_INTERPRETABLE


def test_the_threshold_is_inclusive_at_exactly_the_floor_ratio():
    """"within ~1.2x" includes 1.2 itself. A boundary that lets 1.20 through is a boundary
    at 1.2000001, and this project has already been burned at 1.03."""
    assert floor_label(FLOOR_RATIO_MIN, "better") == NOT_INTERPRETABLE
    assert floor_label(FLOOR_RATIO_MIN - 1e-9, "worse") == NOT_INTERPRETABLE
    assert floor_label(FLOOR_RATIO_MIN + 0.01, "better") == "better"


def test_a_real_register_effect_clears_the_floor():
    """3.51x / 3.72x -- the 1024 run's register move, the first real one in this project."""
    assert floor_label(3.51, "better") == "better"
    assert floor_label(3.72, "worse") == "worse"


def test_the_mirror_image_error_is_caught_by_the_second_gate():
    """engagement: 2.99x the floor over a tiny denominator, but +0.0198 against a 0.0275 MDE.

    A finding has to clear BOTH gates. Above the floor and inside its own paired interval is
    not a finding.
    """
    assert floor_label(2.99, "no change") == "below paired detection"


def test_no_floor_means_no_interpretation_not_a_default_one():
    assert floor_label(None, "better") == NO_FLOOR
    assert floor_label(None, "no change") == NO_FLOOR


def test_a_zero_floor_produces_no_ratio_rather_than_infinity():
    """A seed control that moved a signal by exactly nothing is an empty denominator.
    Dividing by it manufactures an unbounded ratio -- the same error, upside down."""
    assert floor_ratio(0.05, 0.0) is None
    assert floor_ratio(0.05, None) is None
    assert floor_ratio(None, 0.05) is None


def test_the_ratio_is_a_magnitude_so_direction_does_not_change_it():
    assert floor_ratio(-0.06, 0.05) == pytest.approx(1.2)
    assert floor_ratio(0.06, 0.05) == pytest.approx(1.2)


def test_labelling_joins_the_paired_verdict_with_the_floor():
    floor = SeedFloor(behaviour={"collapse_rate": 0.0542, "tinystories_margin": 0.0745},
                      behaviour_source="x", behaviour_prompt_set="b", loss_sd=None,
                      loss_mean=None, loss_sign=None, loss_window=None, loss_sources=(),
                      provenance="test")
    diffs = [
        # The 1024 run's real numbers: -0.0549 against a -0.0542 floor is 1.01x.
        make_paired("collapse_rate", "genre collapse rate", "lower", -0.0549, 0.01),
        # And its real register move: -0.2613 against -0.0745 is 3.51x.
        make_paired("tinystories_margin", "tinystories margin", "lower", -0.2613, 0.02),
    ]
    labelled = {d.signal: d for d in label_differences(diffs, floor)}
    assert labelled["collapse_rate"].ci_verdict == "better", "the CI does call it better"
    assert labelled["collapse_rate"].ratio == pytest.approx(1.0129, abs=1e-3)
    assert labelled["collapse_rate"].label == NOT_INTERPRETABLE, \
        "and the floor must override that"
    assert labelled["tinystories_margin"].ratio == pytest.approx(3.507, abs=1e-2)
    assert labelled["tinystories_margin"].label == "better"


def test_the_rendered_report_prints_the_ratio_beside_every_delta(tmp_path):
    """Mutation check at the render layer: dropping the ratio column must break this."""
    floor = SeedFloor(behaviour={"collapse_rate": 0.0542, "tinystories_margin": 0.0745},
                      behaviour_source="docs/measurements/seed.json",
                      behaviour_prompt_set="b", loss_sd=0.1944, loss_mean=0.041,
                      loss_sign=sign_test([-1.0] * 8 + [1.0] * 14), loss_window=2048,
                      loss_sources=("a", "b"), provenance="derived-live")
    base = make_model(tmp_path, "384s512", 512, losses=[(500, 5.0), (1000, 4.0), (1500, 3.2)])
    cand = make_model(tmp_path, "1024a", 512, losses=[(500, 4.5), (1000, 3.5), (1500, 2.9)])
    cmp = Comparison(
        baseline=make_evaluation(base, window=512), candidate=make_evaluation(cand, window=512),
        window=512, floor=floor, floor_problems=[],
        behaviour=label_differences([
            make_paired("collapse_rate", "genre collapse rate", "lower", -0.0549, 0.01),
            make_paired("tinystories_margin", "tinystories margin", "lower", -0.2613, 0.02),
        ], floor),
        trajectory=compare_trajectories(base, cand, floor), trajectory_refusal=None,
        window_loss_delta=-0.29, started_utc="2026-08-15T00:00:00+00:00")
    out = render_comparison(cmp)
    # "Beside" means in the signal's own row, not merely somewhere in the document -- a
    # summary line further down is not what a reader scanning the table sees.
    rows = {line.split("|")[1].strip(): line
            for line in out.splitlines() if line.startswith("| ")}
    assert "1.01x" in rows["genre collapse rate"], \
        "the floor ratio must sit in the signal's own row"
    assert NOT_INTERPRETABLE in rows["genre collapse rate"]
    assert "3.51x" in rows["tinystories margin"]
    assert "0.0542" in rows["genre collapse rate"], "the floor itself must be shown too"
    assert "0.1944" in out, "the derived loss floor must be stated, not implied"
    assert "8/22 negative" in out, "the floor's own sign split makes the candidate readable"


def test_a_report_with_no_floor_prints_no_ratios_and_says_why(tmp_path):
    """"Say so and refuse to print ratios rather than inventing them." """
    base = make_model(tmp_path, "384s512", 512)
    cand = make_model(tmp_path, "1024a", 512)
    cmp = Comparison(
        baseline=make_evaluation(base, window=512), candidate=make_evaluation(cand, window=512),
        window=512, floor=None,
        floor_problems=["the seed-only loss trajectories are missing"],
        behaviour=label_differences(
            [make_paired("collapse_rate", "genre collapse rate", "lower", -0.9, 0.01)], None),
        trajectory=None, trajectory_refusal="no trajectory", window_loss_delta=None,
        started_utc="2026-08-15T00:00:00+00:00")
    out = render_comparison(cmp)
    assert "No seed-only floor is available" in out
    assert "the seed-only loss trajectories are missing" in out
    assert NO_FLOOR in out
    assert "No verdict" in out, "no floor means no verdict, not a quiet one"
    # Not a substring check: ANY "N.NNx" in the report is a ratio, and there is nothing to
    # compute one from -- including the threshold itself, which is a number nothing here was
    # measured against.
    leaked = re.findall(r"\d+\.\d+x", out)
    assert leaked == [], f"ratios printed with no floor to compute them from: {leaked}"


# ---------------------------------------------------------------------------------------
# The floor is DERIVED, not hardcoded
# ---------------------------------------------------------------------------------------


def test_the_behavioural_floor_comes_out_of_the_committed_seed_control():
    """Against the real committed file, so a change to it moves the floor."""
    if not FLOOR_BEHAVIOUR_JSON.is_file():
        pytest.skip(f"seed-only control {FLOOR_BEHAVIOUR_JSON} is not present in this "
                    f"checkout; the behavioural floor cannot be derived")
    floor, prompt_set = derive_behaviour_floor(FLOOR_BEHAVIOUR_JSON)
    assert prompt_set == "b"
    # The two numbers this project's own reports quote for the seed-only control.
    assert floor["tinystories_margin"] == pytest.approx(0.0745, abs=1e-3)
    assert floor["register_tinystories_share"] == pytest.approx(0.0368, abs=1e-3)
    assert all(v >= 0 for v in floor.values()), "a floor is a magnitude"


def test_the_floor_follows_its_input_rather_than_a_constant(tmp_path):
    """Mutation check on "derive, do not hardcode": a synthetic control must be believed.

    If anyone replaces the derivation with the numbers that happen to be true today, this
    fails -- the fixture's floor is deliberately nothing like 0.1944 or 0.0745.
    """
    path = tmp_path / "seed-control.json"
    path.write_text(json.dumps({
        "prompt_set": "b",
        "paired": [
            {"signal": "collapse_rate", "difference": {"mean": -0.5, "sem": 0.1, "n": 45}},
            {"signal": "tinystories_margin", "difference": {"mean": 0.25, "sem": 0.1, "n": 45}},
        ]}))
    floor, prompt_set = derive_behaviour_floor(path)
    assert floor == {"collapse_rate": 0.5, "tinystories_margin": 0.25}
    assert prompt_set == "b"


def test_the_loss_floor_is_the_spread_of_the_seed_controls_paired_deltas(tmp_path):
    """sd, not mean: the seed control's mean delta is near zero because its sign wanders."""
    behaviour = tmp_path / "seed-control.json"
    behaviour.write_text(json.dumps({
        "prompt_set": "b",
        "paired": [{"signal": "collapse_rate",
                    "difference": {"mean": -0.5, "sem": 0.1, "n": 45}}]}))
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text("\n".join(json.dumps({"step": s, "val_loss": 4.0})
                           for s in (500, 1000, 1500, 2000)))
    b.write_text("\n".join(json.dumps({"step": s, "val_loss": v})
                           for s, v in ((500, 5.0), (1000, 3.0), (1500, 5.0), (2000, 3.0))))
    floor = derive_seed_floor(behaviour_json=behaviour, trajectory_a=a, trajectory_b=b)
    # deltas are +1, -1, +1, -1: mean 0, sd 1.1547 (ddof=1). The mean is useless; the
    # spread is the floor.
    assert floor.loss_mean == pytest.approx(0.0)
    assert floor.loss_sd == pytest.approx(1.1547, abs=1e-3)
    assert floor.loss_sign.n_negative == 2 and floor.loss_sign.n == 4


def test_a_missing_loss_floor_is_reported_not_invented(tmp_path):
    behaviour = tmp_path / "seed-control.json"
    behaviour.write_text(json.dumps({
        "prompt_set": "b",
        "paired": [{"signal": "collapse_rate",
                    "difference": {"mean": -0.5, "sem": 0.1, "n": 45}}]}))
    floor = derive_seed_floor(behaviour_json=behaviour,
                              trajectory_a=tmp_path / "nope-a.jsonl",
                              trajectory_b=tmp_path / "nope-b.jsonl")
    assert floor.loss_sd is None, "a missing floor must stay missing"
    assert floor.notes, "and must say so"
    assert "missing" in " ".join(floor.notes)


def test_the_floor_derivation_checks_its_own_inputs_share_a_window(tmp_path):
    """The floor is the one number nobody may fudge, including this module."""
    behaviour = tmp_path / "seed-control.json"
    behaviour.write_text(json.dumps({
        "prompt_set": "b",
        "paired": [{"signal": "collapse_rate",
                    "difference": {"mean": -0.5, "sem": 0.1, "n": 45}}]}))
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    for path in (a, b):
        path.write_text("\n".join(json.dumps({"step": s, "val_loss": 4.0})
                                  for s in (500, 1000, 1500)))
    with pytest.raises(WindowMismatch):
        derive_seed_floor(behaviour_json=behaviour, trajectory_a=a, trajectory_b=b,
                          window_a=2048, window_b=512)


# ---------------------------------------------------------------------------------------
# The sign test -- preferred for trajectories
# ---------------------------------------------------------------------------------------


def test_a_unanimous_trajectory_matches_the_committed_capacity_result():
    """22/22 negative, p ~ 5e-7 -- the number the 384s512 control commit reports."""
    result = sign_test([-0.3] * 22)
    assert result.n == 22 and result.n_negative == 22
    assert result.p_two_sided == pytest.approx(4.77e-7, rel=1e-2)


def test_the_seed_floors_own_split_matches_the_committed_control():
    """8/22 negative -- a floor that changes sign, which is what makes 22/22 convincing."""
    result = sign_test([-1.0] * 8 + [1.0] * 14)
    assert result.n_negative == 8 and result.n_positive == 14
    assert result.p_two_sided == pytest.approx(0.2863, abs=1e-3)


def test_the_sign_test_is_symmetric_and_drops_ties():
    assert sign_test([-1.0] * 22).p_two_sided == sign_test([1.0] * 22).p_two_sided
    result = sign_test([-1.0, 1.0, 0.0, 0.0])
    assert result.n == 2 and result.n_zero == 2


def test_the_sign_test_reports_nothing_rather_than_a_p_of_one_when_all_ties():
    result = sign_test([0.0, 0.0, 0.0])
    assert result.n == 0 and result.p_two_sided is None


def test_the_report_gives_the_endpoint_AND_the_average_and_says_which_is_the_headline(tmp_path):
    """Mistake (2): a trajectory AVERAGE reported where the ENDPOINT was the number."""
    base = make_model(tmp_path, "384s512", 512,
                      losses=[(500, 9.0), (1000, 4.0), (1500, 3.2)])
    cand = make_model(tmp_path, "1024a", 512,
                      losses=[(500, 4.5), (1000, 3.5), (1500, 3.1)])
    result = compare_trajectories(base, cand, None)
    assert result.final_delta == pytest.approx(-0.1), "endpoint"
    assert result.mean_delta == pytest.approx(-1.7), "average, dominated by the early steps"
    assert result.final_delta != result.mean_delta
    cmp = Comparison(baseline=make_evaluation(base, window=512),
                     candidate=make_evaluation(cand, window=512), window=512, floor=None,
                     floor_problems=[], behaviour=[], trajectory=result,
                     trajectory_refusal=None, window_loss_delta=None,
                     started_utc="2026-08-15T00:00:00+00:00")
    out = render_comparison(cmp)
    assert "endpoint delta" in out and "the headline" in out
    assert "average over checkpoints" in out and "*not* the headline" in out


# ---------------------------------------------------------------------------------------
# Prompt sets are never pooled
# ---------------------------------------------------------------------------------------


def test_a_cross_prompt_set_comparison_is_refused_even_with_identical_prompt_ids(tmp_path):
    """The existing refusal in score_behaviour must be preserved, not routed around."""
    base = make_model(tmp_path, "v3", 512)
    cand = make_model(tmp_path, "v5", 512)
    with pytest.raises(ValueError) as exc:
        compare(make_evaluation(base, window=512,
                                behaviour=behaviour_payload("a", {"collapse_rate": 0.1})),
                make_evaluation(cand, window=512,
                                behaviour=behaviour_payload("b", {"collapse_rate": 0.2})),
                floor=None, floor_problems=[], skip_trajectory=True)
    assert "different prompt sets" in str(exc.value)


def test_one_prompt_set_pairs_normally(tmp_path):
    base = make_model(tmp_path, "v3", 512)
    cand = make_model(tmp_path, "v5", 512)
    cmp = compare(make_evaluation(base, window=512,
                                  behaviour=behaviour_payload("b", {"collapse_rate": 0.10})),
                  make_evaluation(cand, window=512,
                                  behaviour=behaviour_payload("b", {"collapse_rate": 0.04})),
                  floor=None, floor_problems=[], skip_trajectory=True)
    collapse = next(d for d in cmp.behaviour if d.signal == "collapse_rate")
    assert collapse.delta == pytest.approx(-0.06, abs=1e-6)


# ---------------------------------------------------------------------------------------
# Mode 3 -- the ad-hoc escape valve stays out of the measurement namespace
# ---------------------------------------------------------------------------------------


def test_adhoc_output_may_not_land_in_the_measurements_directory():
    with pytest.raises(ScratchPathViolation):
        assert_scratch_path(ROOT / "docs" / "measurements" / "samples.md")


def test_adhoc_output_may_not_borrow_the_behaviour_namespace():
    from scripts.evaluate import SCRATCH

    with pytest.raises(ScratchPathViolation):
        assert_scratch_path(SCRATCH / "behaviour-tt-tnt-1024a-setB.md")


def test_adhoc_output_may_not_escape_via_a_sibling_with_the_same_prefix():
    """``…/adhoc-promptsEVIL`` is not inside ``…/adhoc-prompts``; a string prefix test says
    it is."""
    from scripts.evaluate import SCRATCH

    with pytest.raises(ScratchPathViolation):
        assert_scratch_path(Path(str(SCRATCH) + "EVIL") / "x.md")


def test_adhoc_output_may_not_escape_via_dot_dot():
    from scripts.evaluate import SCRATCH

    with pytest.raises(ScratchPathViolation):
        assert_scratch_path(SCRATCH / ".." / ".." / "docs" / "measurements" / "x.md")


def test_a_scratch_path_is_accepted():
    from scripts.evaluate import SCRATCH

    accepted = assert_scratch_path(SCRATCH / "ADHOC-20260815T000000Z-x.md")
    assert accepted.name.startswith("ADHOC-")


def test_the_default_adhoc_path_is_outside_docs_and_marked_as_adhoc():
    path = adhoc_output_path("The lighthouse keeper wrote in the log:")
    assert "ADHOC-" in path.name
    assert "docs" not in path.relative_to(ROOT).parts
    assert path.parent.name == "adhoc-prompts"


def test_two_adhoc_runs_do_not_overwrite_each_other():
    from datetime import datetime, timezone

    first = adhoc_output_path("x", now=datetime(2026, 8, 15, 1, tzinfo=timezone.utc))
    second = adhoc_output_path("x", now=datetime(2026, 8, 15, 2, tzinfo=timezone.utc))
    assert first != second


def test_the_adhoc_markdown_shouts_that_it_is_not_a_measurement():
    out = render_adhoc(["a prompt"], {(0, 0.0): " completion"}, model=Path("artifacts/hf-x"),
                       temperatures=[0.0], max_new_tokens=24, seed=0,
                       out_path=Path("scratch/adhoc-prompts/ADHOC-x.md"))
    assert out.startswith(f"# {ADHOC_BANNER}")
    assert "NOT a measurement" in out
    assert out.rstrip().endswith(f"_{ADHOC_BANNER}_")


def test_the_adhoc_markdown_says_how_to_promote_a_good_prompt():
    """Never by editing an existing set -- that silently invalidates every measurement."""
    out = render_adhoc(["a prompt"], {(0, 0.0): " c"}, model=Path("artifacts/hf-x"),
                       temperatures=[0.0], max_new_tokens=24, seed=0,
                       out_path=Path("scratch/adhoc-prompts/ADHOC-x.md"))
    assert "new" in out and "new ids" in out
    assert "digest-pinned" in out
    assert "evaluation_prompts_b.json" in out


def test_help_explains_how_to_promote_a_prompt_into_a_new_set():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--help"],
        capture_output=True, text=True, cwd=str(ROOT))
    assert completed.returncode == 0
    assert "NEW set with new" in completed.stdout
    assert "never by editing an existing set" in completed.stdout


def test_prompts_can_be_read_from_a_plain_text_file(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text("# a comment\nfirst prompt\n\nsecond prompt\n")
    assert read_prompt_file(path) == ["first prompt", "second prompt"]


def test_prompts_can_be_read_from_json(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({"prompts": [{"text": "a"}, {"text": "b"}]}))
    assert read_prompt_file(path) == ["a", "b"]


def test_an_empty_prompt_file_raises(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text("# nothing but comments\n")
    with pytest.raises(ValueError):
        read_prompt_file(path)


def test_the_frozen_prompt_sets_are_refused_as_an_adhoc_write_target():
    """The mechanism, not a grep: every --try write goes through assert_scratch_path, and
    neither frozen set is reachable through it."""
    for frozen in ("evaluation_prompts.json", "evaluation_prompts_b.json"):
        with pytest.raises(ScratchPathViolation):
            assert_scratch_path(ROOT / "docs" / frozen)


def test_the_frozen_prompt_sets_digests_are_untouched_by_this_module():
    """Belt and braces against the one edit that would silently invalidate every committed
    measurement: the digest tests own this, and this asserts the files still parse as the
    sets they are so a failure here points at the right place."""
    for name, expected in (("evaluation_prompts.json", 15), ("evaluation_prompts_b.json", 45)):
        payload = json.loads((ROOT / "docs" / name).read_text())
        assert len(payload["prompts"]) == expected


# ---------------------------------------------------------------------------------------
# The current-model designation
# ---------------------------------------------------------------------------------------


def test_the_designation_exists_and_carries_its_own_justification():
    designation = load_designation()
    assert designation.label
    assert designation.reason and designation.qualification
    assert designation.evidence


def test_the_designation_points_at_a_converted_model_directory():
    designation = load_designation()
    relative = designation.hf_model.relative_to(ROOT)
    assert relative.parts[0] == "artifacts"
    assert relative.name.startswith("hf-")


def test_the_designation_states_the_context_caveat_honestly():
    """1024a is trained at 512 while v3 is at 2048, so "best" is not unqualified. A
    designation that omits that is the wrong headline waiting to happen again."""
    qualification = load_designation().qualification
    assert "512" in qualification and "2048" in qualification
    assert "not unqualified" in qualification.lower(), \
        "the designation must say in so many words that 'best' is qualified"


def test_the_designations_evidence_paths_are_named_not_gestured_at():
    evidence = load_designation().evidence
    assert any(e.endswith("val_losses.jsonl") for e in evidence), \
        "the loss claim needs its trajectories named"
    assert any("behaviour-tt-tnt-v3-vs-tt-tnt-v5" in e for e in evidence), \
        "the seed-only control is what makes any of it interpretable"


def test_a_designation_without_a_reason_is_refused(tmp_path):
    path = tmp_path / "current_model.json"
    path.write_text(json.dumps({"current": {"label": "x", "hf_model": "artifacts/hf-x"}}))
    with pytest.raises(DesignationError) as exc:
        load_designation(path)
    assert "reason" in str(exc.value)


def test_a_missing_designation_is_refused_rather_than_guessed(tmp_path):
    with pytest.raises(DesignationError):
        load_designation(tmp_path / "nothing.json")


def test_the_shipped_designation_file_is_valid_json_with_every_candidate_listed():
    payload = json.loads(CURRENT_MODEL_JSON.read_text())
    labels = {c["label"] for c in payload["candidates"]}
    assert payload["current"]["label"] in labels
    # All five trained checkpoints, so a reader can see what was NOT chosen.
    assert labels >= {"tt-tnt-v3", "tt-tnt-v4", "tt-tnt-v5", "tt-tnt-384s512",
                      "tt-tnt-1024a"}


# ---------------------------------------------------------------------------------------
# Comparability facts recorded in the single-model report
# ---------------------------------------------------------------------------------------


def test_the_single_report_records_every_comparability_fact(tmp_path):
    facts = make_model(tmp_path, "1024a", 512)
    ev = make_evaluation(facts, window=512,
                         behaviour=behaviour_payload("b", {"collapse_rate": 0.04}),
                         buckets=[{"lo": 0, "hi": 256, "mean": 3.0, "sem": 0.1,
                                   "n_windows": 256},
                                  {"lo": 256, "hi": 512, "mean": 2.0, "sem": 0.1,
                                   "n_windows": 256}])
    payload = ev.as_json()["comparability"]
    for key in ("eval_window", "model_max_position_embeddings", "tokens", "prompt_set",
                "num_samples", "seed"):
        assert key in payload, f"{key} is a comparability fact and must be recorded"
    out = render_single(ev)
    assert "eval window" in out and "512" in out
    assert "max_position_embeddings" in out
    assert "tokens-v3/val_ids.npy" in out
    assert "prompt set" in out


def test_a_shorter_eval_window_than_the_model_is_flagged_in_the_report(tmp_path):
    facts = make_model(tmp_path, "v3", 2048)
    ev = make_evaluation(facts, window=512)
    out = render_single(ev)
    assert "shorter" in out
    assert "2048" in out and "512" in out


def test_the_pooled_window_loss_is_the_position_mean_over_the_window():
    buckets = [{"lo": 0, "hi": 256, "mean": 3.0}, {"lo": 256, "hi": 512, "mean": 2.0}]
    assert pooled_window_loss(buckets, 512) == pytest.approx(2.5)


def test_the_pooled_loss_refuses_buckets_that_do_not_tile_the_window():
    """Half a window's buckets averaged into "the loss" is the same class of error."""
    assert pooled_window_loss([{"lo": 0, "hi": 256, "mean": 3.0}], 512) is None


def test_checkpoint_dir_is_derived_from_the_hf_dir_name(tmp_path):
    (tmp_path / "checkpoints-tt-tnt-1024a").mkdir()
    (tmp_path / "hf-tt-tnt-1024a").mkdir()
    assert checkpoint_dir_for(tmp_path / "hf-tt-tnt-1024a") == \
        tmp_path / "checkpoints-tt-tnt-1024a"


def test_a_missing_checkpoint_dir_is_None_not_an_invented_path(tmp_path):
    (tmp_path / "hf-tt-tnt-v9").mkdir()
    assert checkpoint_dir_for(tmp_path / "hf-tt-tnt-v9") is None


def test_no_evaluation_run_may_write_under_artifacts():
    """The evidence tree is read-only to this tool: it produces reports and nothing else."""
    for target in ("artifacts", "artifacts/hf-tt-tnt-1024a",
                   "artifacts/checkpoints-tt-tnt-v3", "artifacts/tokens-v3",
                   "artifacts/corpus"):
        with pytest.raises(ValueError) as exc:
            assert_writable_out_dir(ROOT / target)
        assert "artifacts" in str(exc.value)


def test_the_default_and_scratch_out_dirs_are_accepted():
    from scripts.evaluate import MEASUREMENTS, SCRATCH

    assert assert_writable_out_dir(MEASUREMENTS) == MEASUREMENTS
    assert assert_writable_out_dir(SCRATCH) == SCRATCH


def test_a_config_without_max_position_embeddings_is_refused(tmp_path):
    model_dir = tmp_path / "hf-broken"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"hidden_size": 384}))
    with pytest.raises(ValueError) as exc:
        read_model_facts(model_dir)
    assert "max_position_embeddings" in str(exc.value)


# ---------------------------------------------------------------------------------------
# The two tests that genuinely need this machine's checkpoints -- skipped, never vacuous
# ---------------------------------------------------------------------------------------


def test_the_designated_model_is_present_and_its_window_matches_the_designation():
    designation = load_designation()
    if not (designation.hf_model / "config.json").is_file():
        pytest.skip(f"the designated model {designation.hf_model} is not converted in this "
                    f"checkout (artifacts/ is not committed); its config.json cannot be "
                    f"read, so its training window cannot be checked against the "
                    f"designation")
    facts = read_model_facts(designation.hf_model)
    declared = json.loads(CURRENT_MODEL_JSON.read_text())["current"]["training_window"]
    assert facts.training_window == declared, \
        "docs/current_model.json's training_window must match the converted config"


def test_the_real_seed_floor_reproduces_the_committed_numbers():
    from scripts.evaluate import FLOOR_TRAJECTORY_A, FLOOR_TRAJECTORY_B

    missing = [p for p in (FLOOR_TRAJECTORY_A, FLOOR_TRAJECTORY_B) if not p.is_file()]
    if missing:
        pytest.skip(f"the seed-only control's loss trajectories are not in this checkout "
                    f"({', '.join(str(p) for p in missing)}); artifacts/ is gitignored, so "
                    f"the loss floor cannot be derived live here")
    floor = derive_seed_floor()
    # The numbers the 384s512 control commit reports, reproduced by derivation.
    assert floor.loss_sd == pytest.approx(0.1944, abs=1e-4)
    assert floor.loss_sign.n_negative == 8 and floor.loss_sign.n == 22
