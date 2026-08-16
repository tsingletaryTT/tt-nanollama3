#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""One entry point for evaluating a tt-tnt checkpoint -- and for comparing two of them.

WHY THIS EXISTS
---------------
This project already has good instruments: ``scripts/eval_per_source.py``,
``scripts/probe_context_use.py``, ``scripts/score_behaviour.py``. It has never had a way to
run them as **one benchmark**, so every comparison so far has been hand-assembled -- and
**every significant error this project has made came from the joining, not the measuring**:

1. **A 512-token-window loss was compared against a 2048-token-window loss as if the two
   were commensurable.** ``evaluate()`` windows at ``cfg.seq_len``, so the window silently
   rides along with the model: the 1024 run's "5.6x the parameters bought nothing" was a
   harder evaluation condition being read as a null result. At a matched window the same
   two runs differ by -0.2994 nats.
2. **A trajectory *average* was reported where the *endpoint* was the meaningful number**,
   making an effect look 9.7x its true size.
3. **Deltas were quoted against SAMPLING error when RUN-TO-RUN error was the relevant
   floor**, producing a confident "LR decay improves register" finding that a seed-only
   control later refuted.

A single command that always joins the numbers correctly would have caught all three, so
that is what this is: not a convenience wrapper, an error-prevention tool. Every guard
below exists because the corresponding mistake has actually been made here.

THE THREE MODES
---------------
``--model DIR``
    Evaluate one model, emit one report (markdown + JSON) under ``docs/measurements/``.
    Runs the existing instruments as subprocesses -- nothing is reimplemented -- and records
    the facts that make a result comparable or not: the eval window, the token array, the
    prompt set, the sample count, and the model's ``max_position_embeddings``. Defaults to
    the model designated in ``docs/current_model.json``.

``--model A --against B``
    The mode that matters. Refuses to compare losses measured at different windows, prints
    the seed-floor ratio beside every delta, and labels anything within
    :data:`FLOOR_RATIO_MIN` of the floor **NOT INTERPRETABLE** regardless of its confidence
    interval. Prefers the sign test over the loss trajectory, and reports both.

``--try "some prompt text"``
    The escape valve. The frozen prompt sets are digest-pinned so checkpoints stay
    comparable, which by design forbids trying a new prompt on impulse. This mode generates
    from an arbitrary prompt at several temperatures and writes to a clearly-marked
    **scratch** location -- never to the frozen JSONs, never into the ``behaviour-*``
    measurement namespace, and (see :func:`assert_scratch_path`) it refuses rather than
    letting a stray ``--out`` put an ad-hoc sample where a benchmark result lives.

WHAT "THE FLOOR" IS, AND WHERE IT COMES FROM
--------------------------------------------
The floor is the **seed-only control**: two runs identical in everything but the RNG seed
(``tt-tnt-v3`` vs ``tt-tnt-v5``). Whatever that pair moves, a seed moves, so a candidate
that moves a signal by the same amount has shown nothing. It is **derived at runtime** --
the way ``scripts/render_licensing.py`` derives its table from the registry -- never
hardcoded, from:

* ``docs/measurements/behaviour-tt-tnt-v3-vs-tt-tnt-v5-setB.json`` (committed) for the
  behavioural signals, and
* ``artifacts/checkpoints-tt-tnt-{v3,v5}/val_losses.jsonl`` for the loss trajectory.

``artifacts/`` is not committed (see ``artifacts/.gitignore``), so on a fresh clone the loss
half of the floor is absent. ``--refresh-floor`` therefore renders a committable snapshot,
``docs/measurements/seed-noise-floor.json``, from those raw sources; runs prefer the live
derivation and fall back to the snapshot, recording in every report which one they used. If
neither is available the ratio columns are **not printed** and the report says so -- an
invented floor is worse than no floor.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.generate_samples import (  # noqa: E402
    resolve_model_dir,
    validate_sampling_args,
)
from scripts.score_behaviour import (  # noqa: E402
    LEGACY_PROMPT_SET,
    PROMPT_SETS,
    PairedDifference,
    default_label,
    paired_differences,
)

# ---------------------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------------------

MEASUREMENTS = ROOT / "docs" / "measurements"

#: Ad-hoc output lives OUTSIDE docs/ entirely, and outside git (see .gitignore). A benchmark
#: result and a thing someone typed once must not be neighbours in the filesystem.
SCRATCH = ROOT / "scratch" / "adhoc-prompts"

#: The designated subject of the benchmark. Data, hand-maintained; nothing here writes it.
CURRENT_MODEL_PATH = ROOT / "docs" / "current_model.json"

#: Where ``--refresh-floor`` renders the committable seed-floor snapshot.
FLOOR_SNAPSHOT_PATH = MEASUREMENTS / "seed-noise-floor.json"

#: The seed-only control's behavioural comparison -- committed, so it is always available.
FLOOR_BEHAVIOUR_JSON = MEASUREMENTS / "behaviour-tt-tnt-v3-vs-tt-tnt-v5-setB.json"

#: The seed-only control's two loss trajectories. NOT committed (artifacts/ is gitignored).
FLOOR_TRAJECTORY_A = ROOT / "artifacts" / "checkpoints-tt-tnt-v3" / "val_losses.jsonl"
FLOOR_TRAJECTORY_B = ROOT / "artifacts" / "checkpoints-tt-tnt-v5" / "val_losses.jsonl"

#: Default evaluation window.
#:
#: **Deliberately a fixed constant and NOT derived from the model.** Defaulting the window to
#: the model's own ``max_position_embeddings`` is precisely the mechanism that produced this
#: project's wrong headline: the window rides along with the model, and two "validation
#: losses" turn out to be answers to different questions. 512 is the largest window every
#: trained checkpoint in this repo can be evaluated at (1024a and 384s512 are 512-context
#: models), so it is the one window at which all five are commensurable.
DEFAULT_WINDOW = 512

#: The token array evaluated against, by default.
#:
#: Recorded in every report, because this repo has three token generations on disk
#: (``tokens``, ``tokens-stratified``, ``tokens-v3``) with different byte counts. Two models
#: scored on different arrays are not comparable no matter how matched everything else is.
DEFAULT_TOKENS = ROOT / "artifacts" / "tokens-v3" / "val_ids.npy"

#: Prompt set for behavioural scoring. ``b`` (45 prompts) rather than ``a`` (15) because
#: power here is bought with prompts, not samples. The sets are NEVER pooled -- one set is
#: chosen and both sides of a comparison are scored on it.
DEFAULT_PROMPT_SET = "b"

#: Completions drawn per frozen prompt, passed straight to ``score_behaviour.py``.
DEFAULT_NUM_SAMPLES = 32

#: Windows sampled by the context probe.
DEFAULT_N_WINDOWS = 256

#: A delta at or below this multiple of the seed floor is NOT INTERPRETABLE, whatever its
#: confidence interval says.
#:
#: Not a round number picked for comfort -- it is where this project's own history puts it.
#: An LR-decay register delta of -0.041 sat at **1.03x** the floor, cleared its paired CI,
#: was written up as a finding, and was refuted by a seed-only control. Two collapse signals
#: in the 1024 run came back "better" from the paired test at **1.01x** and **1.05x** and
#: were correctly excluded. 1.2 leaves a little air above the largest observed false
#: positive without pretending to a precision the evidence does not support -- which is why
#: the reports say "~1.2x" and print the raw ratio next to every label, so a reader can see
#: how close a call was rather than trusting the threshold.
FLOOR_RATIO_MIN = 1.2

#: Temperatures ``--try`` samples at. 0.0 means greedy/deterministic.
DEFAULT_TRY_TEMPERATURES = (0.0, 0.8, 1.0)

DEFAULT_TRY_MAX_NEW_TOKENS = 80

_SPDX = ("<!-- SPDX-License-Identifier: Apache-2.0 -->",
         "<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->")


class WindowMismatch(RuntimeError):
    """Raised when two losses measured at different windows are about to be compared."""


class ScratchPathViolation(RuntimeError):
    """Raised when an ad-hoc sample is about to be written where measurements live."""


class DesignationError(RuntimeError):
    """Raised when ``docs/current_model.json`` is missing or does not say what it must."""


# ---------------------------------------------------------------------------------------
# The designated current model
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Designation:
    """What ``docs/current_model.json`` says the benchmark's subject is."""

    label: str
    hf_model: Path
    reason: str
    qualification: str
    evidence: Tuple[str, ...]
    designated: str
    source: Path

    def as_json(self) -> dict:
        return {"label": self.label, "hf_model": str(self.hf_model),
                "reason": self.reason, "qualification": self.qualification,
                "evidence": list(self.evidence), "designated": self.designated,
                "source": str(self.source.relative_to(ROOT))}


def load_designation(path: Path = CURRENT_MODEL_PATH) -> Designation:
    """Read the current-model designation, insisting it carries its own justification.

    A bare "the current model is X" is how a designation goes stale without anyone noticing.
    ``reason``, ``qualification`` and ``evidence`` are therefore required, not optional: the
    file has to say what the claim rests on and what it does **not** mean, or it is not a
    designation this tool will act on.
    """
    if not path.is_file():
        raise DesignationError(
            f"no current-model designation at {path}. Either create it (see the schema in "
            f"the file this repo ships) or pass --model explicitly; this tool will not guess "
            f"which of five trained checkpoints a benchmark is about.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DesignationError(f"{path} is not valid JSON: {exc}") from exc
    current = payload.get("current")
    if not isinstance(current, dict):
        raise DesignationError(f"{path} has no 'current' object")
    missing = [k for k in ("label", "hf_model", "reason", "qualification", "evidence")
               if not current.get(k)]
    if missing:
        raise DesignationError(
            f"{path}'s 'current' is missing {missing}. A designation must carry its reason, "
            f"its evidence, and the qualification that says what it does not mean -- "
            f"otherwise the next reader inherits a claim with no way to check it.")
    evidence = current["evidence"]
    if not isinstance(evidence, list) or not all(isinstance(e, str) for e in evidence):
        raise DesignationError(f"{path}'s 'current.evidence' must be a list of paths")
    return Designation(
        label=str(current["label"]),
        hf_model=ROOT / str(current["hf_model"]),
        reason=str(current["reason"]),
        qualification=str(current["qualification"]),
        evidence=tuple(evidence),
        designated=str(current.get("designated", "unknown")),
        source=path,
    )


# ---------------------------------------------------------------------------------------
# Model facts -- everything that decides whether two numbers may be compared
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelFacts:
    """The comparability facts about one converted HF model directory."""

    path: Path
    label: str
    max_position_embeddings: int
    hidden_size: Optional[int]
    num_hidden_layers: Optional[int]
    checkpoint_dir: Optional[Path]

    @property
    def training_window(self) -> int:
        """The window this model's ``val_losses.jsonl`` was measured at.

        Not a guess. ``train/run.py``'s ``evaluate()`` windows at ``cfg.seq_len``; that same
        ``seq_len`` is written into the checkpoint header; and ``convert/to_hf.py`` sets
        ``max_position_embeddings = int(header["seq_len"])`` and *verifies* it. So the
        converted config's ``max_position_embeddings`` IS the training-time evaluation
        window, and reading it here is reading the trajectory's units.
        """
        return self.max_position_embeddings

    def as_json(self) -> dict:
        return {
            "path": _rel(self.path),
            "label": self.label,
            "max_position_embeddings": self.max_position_embeddings,
            "training_window": self.training_window,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "checkpoint_dir": _rel(self.checkpoint_dir) if self.checkpoint_dir else None,
        }


def _rel(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise -- reports should be portable."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def checkpoint_dir_for(model_dir: Path) -> Optional[Path]:
    """``artifacts/hf-tt-tnt-1024a`` -> ``artifacts/checkpoints-tt-tnt-1024a``, if it exists.

    Returns ``None`` rather than a non-existent path, so a caller can say "no trajectory
    available" instead of failing to open a file it invented.
    """
    name = Path(model_dir).name
    if name == "hf":
        candidate = Path(model_dir).parent / "checkpoints"
    elif name.startswith("hf-"):
        candidate = Path(model_dir).parent / ("checkpoints-" + name[len("hf-"):])
    else:
        return None
    return candidate if candidate.is_dir() else None


def read_model_facts(model_dir: Path) -> ModelFacts:
    """Read a converted model's comparability facts. Filesystem + JSON only, no torch."""
    model_dir = resolve_model_dir(str(model_dir))
    config_path = model_dir / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{config_path} is not valid JSON: {exc}") from exc
    max_pos = config.get("max_position_embeddings")
    if not isinstance(max_pos, int) or max_pos <= 0:
        raise ValueError(
            f"{config_path} has no usable max_position_embeddings ({max_pos!r}). That field "
            f"is this tool's only record of the window the model was trained and evaluated "
            f"at, and without it no loss from this model can be safely compared to any "
            f"other.")
    return ModelFacts(
        path=model_dir,
        label=default_label(model_dir),
        max_position_embeddings=max_pos,
        hidden_size=config.get("hidden_size"),
        num_hidden_layers=config.get("num_hidden_layers"),
        checkpoint_dir=checkpoint_dir_for(model_dir),
    )


# ---------------------------------------------------------------------------------------
# The window guard -- the trap that produced a wrong headline
# ---------------------------------------------------------------------------------------


def require_matched_window(a_label: str, a_window: int, b_label: str, b_window: int,
                           *, what: str) -> int:
    """Return the common window, or raise :class:`WindowMismatch` naming BOTH windows.

    This is the guard that mistake (1) in the module docstring needed and did not have. It
    is deliberately impossible to pass accidentally: there is no tolerance, no "close
    enough", and the message names each side's window next to each side's label so the
    reader can see immediately which number is which.
    """
    if a_window != b_window:
        raise WindowMismatch(
            f"refusing to compare {what}: {a_label} was measured at a {a_window}-token "
            f"window, and {b_label} was measured at a {b_window}-token window. These are "
            f"answers to different questions -- a longer window is an easier prediction "
            f"problem at every position, so the difference between them is part window and "
            f"part model and cannot be split after the fact. This project has already "
            f"published one headline built on exactly this comparison: a 2048-window loss "
            f"against a 512-window loss, which turned a -0.2994-nat capacity effect into "
            f"'5.6x the parameters bought nothing'. Evaluate both models at one window "
            f"(--window {min(a_window, b_window)}) and compare those numbers instead.")
    return a_window


def require_window_fits(facts: ModelFacts, window: int) -> None:
    """Refuse a window longer than the model's trained context, naming both numbers."""
    if window > facts.max_position_embeddings:
        raise WindowMismatch(
            f"refusing to evaluate {facts.label} at a {window}-token window: its "
            f"max_position_embeddings is {facts.max_position_embeddings}. Positions past "
            f"the trained context have no learned RoPE support, and the loss there measures "
            f"extrapolation rather than the model. Use --window "
            f"{facts.max_position_embeddings} or smaller.")


def common_window(models: Sequence[ModelFacts], requested: Optional[int]) -> int:
    """The window to evaluate every model at, checked against each model's own capacity.

    When ``requested`` is ``None`` the default is :data:`DEFAULT_WINDOW` -- a constant, not
    a function of the models -- narrowed only if some model cannot reach it. Narrowing is
    reported by the caller; silently *widening* to a model's own context is exactly the
    behaviour this module exists to prevent.
    """
    capacity = min(m.max_position_embeddings for m in models)
    if requested is None:
        window = min(DEFAULT_WINDOW, capacity)
    else:
        window = requested
    for m in models:
        require_window_fits(m, window)
    return window


# ---------------------------------------------------------------------------------------
# Token-array fingerprint -- the other thing that must match for a comparison to mean anything
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenArray:
    path: Path
    size_bytes: int
    n_tokens: Optional[int]

    def as_json(self) -> dict:
        return {"path": _rel(self.path), "size_bytes": self.size_bytes,
                "n_tokens": self.n_tokens}


def fingerprint_tokens(path: Path) -> TokenArray:
    """Identify a token array well enough to tell this repo's three generations apart.

    Size in bytes and element count, not a content hash: the arrays are 150 MB+ and the
    generations differ in length, so this separates them at negligible cost. It is an
    identity record for the report, not a cryptographic guarantee.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"token array {path} not found. Every loss in this report is a loss ON a "
            f"specific token array -- this repo has three generations on disk with "
            f"different contents -- so the array is required rather than defaulted around.")
    size = path.stat().st_size
    n_tokens: Optional[int] = None
    try:
        import numpy as np

        n_tokens = int(np.load(path, mmap_mode="r").shape[0])
    except Exception:  # noqa: BLE001 - the fingerprint is provenance, never a gate
        n_tokens = None
    return TokenArray(path=path, size_bytes=size, n_tokens=n_tokens)


# ---------------------------------------------------------------------------------------
# The seed-only noise floor, derived rather than hardcoded
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SignTest:
    """A sign test over paired differences: how often the delta pointed the same way."""

    n: int
    n_negative: int
    n_positive: int
    n_zero: int
    p_two_sided: Optional[float]

    def as_json(self) -> dict:
        return {"n": self.n, "n_negative": self.n_negative, "n_positive": self.n_positive,
                "n_zero": self.n_zero, "p_two_sided": self.p_two_sided}


def sign_test(deltas: Sequence[float]) -> SignTest:
    """Exact two-sided sign test.

    Preferred over mean-vs-spread for loss trajectories because it is **independent of the
    spread**: capacity was negative at 22/22 checkpoints (p ~ 5e-7) while the seed-only
    floor changed sign at 8/22, and no summary that divides a mean by a standard deviation
    states that as clearly. Zero differences are dropped, which is the standard convention
    and the conservative one (a tie is evidence for neither side).
    """
    values = [float(d) for d in deltas]
    n_zero = sum(1 for d in values if d == 0.0)
    n_neg = sum(1 for d in values if d < 0.0)
    n_pos = sum(1 for d in values if d > 0.0)
    n = n_neg + n_pos
    if n == 0:
        return SignTest(n=0, n_negative=0, n_positive=0, n_zero=n_zero, p_two_sided=None)
    m = min(n_neg, n_pos)
    tail = sum(math.comb(n, k) for k in range(m + 1))
    p = min(1.0, 2.0 * tail / float(2 ** n))
    return SignTest(n=n, n_negative=n_neg, n_positive=n_pos, n_zero=n_zero, p_two_sided=p)


def read_val_losses(path: Path) -> List[Tuple[int, float]]:
    """Read a ``val_losses.jsonl`` trajectory as ``(step, val_loss)`` pairs."""
    rows: List[Tuple[int, float]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
        if "step" not in record or "val_loss" not in record:
            raise ValueError(f"{path}:{lineno} has no 'step'/'val_loss': {record!r}")
        rows.append((int(record["step"]), float(record["val_loss"])))
    if not rows:
        raise ValueError(f"{path} contains no validation losses")
    return rows


def pair_trajectories(a: Sequence[Tuple[int, float]],
                      b: Sequence[Tuple[int, float]]) -> List[Tuple[int, float, float]]:
    """Pair two trajectories by STEP, keeping only steps both ran.

    Pairing by step rather than by position: two runs of different length would otherwise be
    zipped into a comparison of step 500 against step 1000.
    """
    b_by_step = dict(b)
    return [(step, loss, b_by_step[step]) for step, loss in a if step in b_by_step]


@dataclass(frozen=True)
class SeedFloor:
    """The run-to-run floor: what a change of SEED ALONE moves each signal by.

    ``behaviour`` maps a ``score_behaviour`` signal key to the absolute paired difference
    the seed-only control produced for it. ``loss_sd`` is the standard deviation of the
    seed-only control's per-checkpoint loss differences -- the *spread*, not the mean,
    because that control's mean delta is near zero while its sign wanders, so the spread is
    what a candidate has to beat.
    """

    behaviour: Dict[str, float]
    behaviour_source: str
    behaviour_prompt_set: str
    loss_sd: Optional[float]
    loss_mean: Optional[float]
    loss_sign: Optional[SignTest]
    loss_window: Optional[int]
    loss_sources: Tuple[str, ...]
    provenance: str
    notes: Tuple[str, ...] = ()

    def as_json(self) -> dict:
        return {
            "provenance": self.provenance,
            "behaviour": self.behaviour,
            "behaviour_source": self.behaviour_source,
            "behaviour_prompt_set": self.behaviour_prompt_set,
            "loss_sd": self.loss_sd,
            "loss_mean": self.loss_mean,
            "loss_sign": self.loss_sign.as_json() if self.loss_sign else None,
            "loss_window": self.loss_window,
            "loss_sources": list(self.loss_sources),
            "notes": list(self.notes),
        }


def derive_behaviour_floor(path: Path) -> Tuple[Dict[str, float], str]:
    """Per-signal seed floor from a committed ``--compare`` JSON, as |mean difference|.

    The seed-only control is a comparison of two models that differ only in RNG seed, so its
    per-signal paired difference IS the amount a seed moves that signal. Absolute value
    because the floor is a magnitude: a candidate moving a signal the other way by the same
    amount has shown just as little.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    paired = payload.get("paired")
    if not isinstance(paired, list) or not paired:
        raise ValueError(f"{path} has no 'paired' array; it is not a --compare output")
    floor: Dict[str, float] = {}
    for row in paired:
        difference = row.get("difference")
        if not isinstance(difference, dict) or difference.get("mean") is None:
            continue
        floor[str(row["signal"])] = abs(float(difference["mean"]))
    if not floor:
        raise ValueError(f"{path} has no usable per-signal differences")
    return floor, str(payload.get("prompt_set", LEGACY_PROMPT_SET))


def derive_seed_floor(
    *,
    behaviour_json: Path = FLOOR_BEHAVIOUR_JSON,
    trajectory_a: Path = FLOOR_TRAJECTORY_A,
    trajectory_b: Path = FLOOR_TRAJECTORY_B,
    window_a: Optional[int] = None,
    window_b: Optional[int] = None,
) -> SeedFloor:
    """Derive the floor from the raw seed-only control. Raises if the inputs are absent.

    The loss half needs the two trajectories to have been measured at the same window --
    they are two runs of the same configuration, so they are, but this asserts it through
    :func:`require_matched_window` rather than assuming it, because the whole point of the
    floor is that it is a number nobody may fudge.
    """
    behaviour, prompt_set = derive_behaviour_floor(behaviour_json)
    notes: List[str] = []

    loss_sd: Optional[float] = None
    loss_mean: Optional[float] = None
    loss_sign: Optional[SignTest] = None
    loss_window: Optional[int] = None
    sources: Tuple[str, ...] = ()

    if trajectory_a.is_file() and trajectory_b.is_file():
        if window_a is not None and window_b is not None:
            loss_window = require_matched_window(
                trajectory_a.parent.name, window_a, trajectory_b.parent.name, window_b,
                what="the seed-only control's own loss trajectories")
        paired = pair_trajectories(read_val_losses(trajectory_a),
                                   read_val_losses(trajectory_b))
        if len(paired) < 3:
            notes.append(
                f"the seed-only trajectories share only {len(paired)} checkpoint steps; "
                f"too few for a loss floor, so loss ratios are not reported")
        else:
            deltas = [b - a for _step, a, b in paired]
            loss_sd = statistics.stdev(deltas)
            loss_mean = statistics.mean(deltas)
            loss_sign = sign_test(deltas)
            sources = (_rel(trajectory_a), _rel(trajectory_b))
    else:
        missing = [str(p) for p in (trajectory_a, trajectory_b) if not p.is_file()]
        notes.append(
            f"the seed-only loss trajectories are missing ({', '.join(missing)}); "
            f"artifacts/ is not committed, so a fresh clone has no loss floor until a "
            f"snapshot is rendered with --refresh-floor")

    return SeedFloor(
        behaviour=behaviour,
        behaviour_source=_rel(behaviour_json),
        behaviour_prompt_set=prompt_set,
        loss_sd=loss_sd,
        loss_mean=loss_mean,
        loss_sign=loss_sign,
        loss_window=loss_window,
        loss_sources=sources,
        provenance="derived-live",
        notes=tuple(notes),
    )


def load_floor_snapshot(path: Path = FLOOR_SNAPSHOT_PATH) -> SeedFloor:
    """Read a previously rendered floor snapshot."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    floor = payload.get("floor")
    if not isinstance(floor, dict):
        raise ValueError(f"{path} has no 'floor' object; re-render it with --refresh-floor")
    sign = floor.get("loss_sign")
    return SeedFloor(
        behaviour={str(k): float(v) for k, v in (floor.get("behaviour") or {}).items()},
        behaviour_source=str(floor.get("behaviour_source", "?")),
        behaviour_prompt_set=str(floor.get("behaviour_prompt_set", LEGACY_PROMPT_SET)),
        loss_sd=floor.get("loss_sd"),
        loss_mean=floor.get("loss_mean"),
        loss_sign=(SignTest(n=sign["n"], n_negative=sign["n_negative"],
                            n_positive=sign["n_positive"], n_zero=sign["n_zero"],
                            p_two_sided=sign["p_two_sided"]) if sign else None),
        loss_window=floor.get("loss_window"),
        loss_sources=tuple(floor.get("loss_sources") or ()),
        provenance=f"snapshot ({_rel(path)})",
        notes=tuple(floor.get("notes") or ()),
    )


def resolve_seed_floor() -> Tuple[Optional[SeedFloor], List[str]]:
    """Live derivation first, committed snapshot second, ``None`` third.

    Returning ``None`` rather than a default is the point: with no floor the reports print
    no ratios and say why. "Derive the floor, do not hardcode it" has to include the case
    where derivation is impossible, or the rule quietly becomes "hardcode it in a fallback".
    """
    problems: List[str] = []
    try:
        floor = derive_seed_floor()
        if floor.loss_sd is not None:
            return floor, list(floor.notes)
        problems.extend(floor.notes)
        live_behaviour: Optional[SeedFloor] = floor
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        problems.append(f"live derivation failed: {exc}")
        live_behaviour = None

    if FLOOR_SNAPSHOT_PATH.is_file():
        try:
            snapshot = load_floor_snapshot()
            problems.append(
                f"using the committed floor snapshot {_rel(FLOOR_SNAPSHOT_PATH)} because "
                f"the raw seed-only sources were not fully available")
            return snapshot, problems
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            problems.append(f"floor snapshot unusable: {exc}")

    if live_behaviour is not None:
        problems.append(
            "behavioural floor available, loss floor NOT -- loss ratios will not be printed")
        return live_behaviour, problems
    return None, problems


def write_floor_snapshot(path: Path = FLOOR_SNAPSHOT_PATH) -> SeedFloor:
    """Render the committable snapshot from the raw seed-only control.

    Same shape as ``scripts/render_licensing.py``: a generated artifact with a banner saying
    it is generated, so it can be committed (and therefore survive a fresh clone, where
    ``artifacts/`` does not exist) without becoming a hardcoded constant that nobody can
    trace back to a measurement.
    """
    facts_a = _facts_for_checkpoint_dir(FLOOR_TRAJECTORY_A.parent)
    facts_b = _facts_for_checkpoint_dir(FLOOR_TRAJECTORY_B.parent)
    floor = derive_seed_floor(window_a=facts_a, window_b=facts_b)
    payload = {
        "_generated_by": "scripts/evaluate.py --refresh-floor",
        "_do_not_edit": (
            "Generated from the seed-only control (tt-tnt-v3 vs tt-tnt-v5). Re-render it "
            "rather than editing it; a hand-edited noise floor is a hardcoded constant "
            "wearing a measurement's clothes."),
        "rendered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "floor": floor.as_json(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return floor


def _facts_for_checkpoint_dir(checkpoint_dir: Path) -> Optional[int]:
    """The training window for a checkpoint directory, via its converted HF sibling."""
    name = checkpoint_dir.name
    if not name.startswith("checkpoints-"):
        return None
    hf_dir = checkpoint_dir.parent / ("hf-" + name[len("checkpoints-"):])
    if not (hf_dir / "config.json").is_file():
        return None
    try:
        return read_model_facts(hf_dir).training_window
    except (FileNotFoundError, ValueError):
        return None


# ---------------------------------------------------------------------------------------
# Labelling a delta against the floor
# ---------------------------------------------------------------------------------------

#: Verdict strings. NOT_INTERPRETABLE is the one that exists because of this project's
#: history: it OVERRIDES a significant confidence interval rather than being combined with it.
NOT_INTERPRETABLE = "NOT INTERPRETABLE"
NO_FLOOR = "no floor"


def floor_ratio(delta: Optional[float], floor: Optional[float]) -> Optional[float]:
    """``|delta| / floor``, or ``None`` when the ratio would be meaningless.

    A zero floor returns ``None``, not infinity: a seed control that moved a signal by
    exactly nothing usually means the signal is pinned at its own floor in both runs, and
    dividing by it manufactures an unbounded ratio out of a tiny (here, empty) denominator
    -- the mirror-image error to the one this function is mainly here to prevent.
    """
    if delta is None or floor is None or floor <= 0.0:
        return None
    return abs(float(delta)) / float(floor)


def floor_label(ratio: Optional[float], verdict: str) -> str:
    """Join a paired-CI verdict with the seed-floor ratio. Both gates must pass.

    - No ratio -> ``no floor``: the delta is reported, the interpretation is not.
    - ratio <= :data:`FLOOR_RATIO_MIN` -> ``NOT INTERPRETABLE``, **whatever the CI says**.
      This is the rule that would have caught the LR-decay finding (1.03x, CI cleared) and
      that correctly excluded the 1024 run's two collapse signals (1.01x, 1.05x).
    - ratio above the floor but the paired CI spans zero -> ``below paired detection``: the
      mirror-image error, a large ratio over a tiny denominator that cannot clear its own
      minimum detectable difference (the 1024 run's engagement, 2.99x but +0.0198 against a
      0.0275 MDE).
    - otherwise the CI's own verdict.
    """
    if ratio is None:
        return NO_FLOOR
    if ratio <= FLOOR_RATIO_MIN:
        return NOT_INTERPRETABLE
    if verdict == "no change":
        return "below paired detection"
    return verdict


@dataclass(frozen=True)
class LabelledDifference:
    """One behavioural signal's delta, its floor ratio, and the joined label."""

    signal: str
    title: str
    better: Optional[str]
    baseline_mean: Optional[float]
    candidate_mean: Optional[float]
    delta: Optional[float]
    ci95: Optional[Tuple[float, float]]
    min_detectable: Optional[float]
    ci_verdict: str
    floor: Optional[float]
    ratio: Optional[float]
    label: str

    def as_json(self) -> dict:
        return {
            "signal": self.signal, "title": self.title, "better": self.better,
            "baseline": self.baseline_mean, "candidate": self.candidate_mean,
            "delta": self.delta,
            "ci95": list(self.ci95) if self.ci95 else None,
            "min_detectable": self.min_detectable,
            "ci_verdict": self.ci_verdict,
            "seed_floor": self.floor, "floor_ratio": self.ratio,
            "label": self.label,
        }


def label_differences(diffs: Sequence[PairedDifference],
                      floor: Optional[SeedFloor]) -> List[LabelledDifference]:
    """Attach the seed-floor ratio and the joined label to every paired difference."""
    out: List[LabelledDifference] = []
    for d in diffs:
        signal_floor = floor.behaviour.get(d.signal) if floor else None
        delta = d.difference.mean if d.difference else None
        ratio = floor_ratio(delta, signal_floor)
        out.append(LabelledDifference(
            signal=d.signal, title=d.title, better=d.better,
            baseline_mean=d.baseline.mean if d.baseline else None,
            candidate_mean=d.candidate.mean if d.candidate else None,
            delta=delta,
            ci95=((d.difference.ci95_lo, d.difference.ci95_hi) if d.difference else None),
            min_detectable=d.min_detectable,
            ci_verdict=d.verdict,
            floor=signal_floor,
            ratio=ratio,
            label=floor_label(ratio, d.verdict),
        ))
    return out


# ---------------------------------------------------------------------------------------
# Loss trajectory comparison -- window-guarded, sign-tested
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryComparison:
    """``candidate - baseline`` per checkpoint step, with endpoint AND average reported."""

    window: int
    n_steps: int
    final_step: int
    final_delta: float
    mean_delta: float
    sd_delta: float
    baseline_final: float
    candidate_final: float
    sign: SignTest
    floor_sd: Optional[float]
    floor_sign: Optional[SignTest]
    ratio: Optional[float]
    label: str
    floor_window: Optional[int]

    def as_json(self) -> dict:
        return {
            "window": self.window, "n_steps": self.n_steps,
            "final_step": self.final_step,
            "final_delta": self.final_delta,
            "mean_delta": self.mean_delta, "sd_delta": self.sd_delta,
            "baseline_final": self.baseline_final, "candidate_final": self.candidate_final,
            "sign_test": self.sign.as_json(),
            "seed_floor_sd": self.floor_sd,
            "seed_floor_sign_test": self.floor_sign.as_json() if self.floor_sign else None,
            "seed_floor_window": self.floor_window,
            "floor_ratio": self.ratio, "label": self.label,
        }


def compare_trajectories(baseline: ModelFacts, candidate: ModelFacts,
                         floor: Optional[SeedFloor]) -> TrajectoryComparison:
    """Compare two training-time loss trajectories, refusing a window mismatch.

    Reports the **endpoint** delta and the average separately and says which is which. That
    distinction is mistake (2) in the module docstring: an average over a trajectory where
    the two runs start far apart and converge is dominated by the early steps, and reporting
    it as "the effect" once made an effect look 9.7x its true size.
    """
    window = require_matched_window(
        baseline.label, baseline.training_window,
        candidate.label, candidate.training_window,
        what="training validation losses")
    if baseline.checkpoint_dir is None or candidate.checkpoint_dir is None:
        missing = [m.label for m in (baseline, candidate) if m.checkpoint_dir is None]
        raise FileNotFoundError(
            f"no checkpoint directory found for {', '.join(missing)}; the loss trajectory "
            f"comparison needs val_losses.jsonl from both runs. Pass --skip-trajectory to "
            f"compare behaviour only.")
    a_path = baseline.checkpoint_dir / "val_losses.jsonl"
    b_path = candidate.checkpoint_dir / "val_losses.jsonl"
    for path in (a_path, b_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found; the loss trajectory comparison needs both runs' "
                f"val_losses.jsonl. Pass --skip-trajectory to compare behaviour only.")
    paired = pair_trajectories(read_val_losses(a_path), read_val_losses(b_path))
    if len(paired) < 3:
        raise ValueError(
            f"{baseline.label} and {candidate.label} share only {len(paired)} checkpoint "
            f"steps; a sign test over that is not worth reporting")
    deltas = [b - a for _step, a, b in paired]
    final_step, base_final, cand_final = paired[-1]
    sd = statistics.stdev(deltas)
    mean = statistics.mean(deltas)
    ratio = floor_ratio(deltas[-1], floor.loss_sd if floor else None)
    # The sign test is the headline for a trajectory; the ratio is the second gate. A
    # trajectory whose endpoint sits at the floor is labelled NOT INTERPRETABLE even if the
    # sign test is unanimous, because 22 checkpoints of one run are not 22 independent runs.
    label = floor_label(ratio, "better" if deltas[-1] < 0 else "worse")
    return TrajectoryComparison(
        window=window, n_steps=len(paired), final_step=final_step,
        final_delta=deltas[-1], mean_delta=mean, sd_delta=sd,
        baseline_final=base_final, candidate_final=cand_final,
        sign=sign_test(deltas),
        floor_sd=floor.loss_sd if floor else None,
        floor_sign=floor.loss_sign if floor else None,
        ratio=ratio, label=label,
        floor_window=floor.loss_window if floor else None,
    )


# ---------------------------------------------------------------------------------------
# Running the existing instruments (as subprocesses -- nothing is reimplemented)
# ---------------------------------------------------------------------------------------


@dataclass
class InstrumentRun:
    """One instrument invocation: the exact argv, and where its JSON landed."""

    name: str
    argv: List[str]
    json_path: Path
    md_path: Optional[Path] = None
    status: str = "pending"
    payload: Optional[dict] = field(default=None, repr=False)

    def as_json(self) -> dict:
        return {"name": self.name, "argv": self.argv, "json": _rel(self.json_path),
                "markdown": _rel(self.md_path) if self.md_path else None,
                "status": self.status}


def run_dir_for(label: str, out_dir: Path = MEASUREMENTS) -> Path:
    """Per-run output directory, kept OUT of the committed ``behaviour-*`` namespace.

    A run of this tool must never overwrite a committed measurement: re-running with a
    different ``--num-samples`` would otherwise silently replace the numbers a published
    finding was built on. ``--out-dir`` moves the whole run elsewhere, which is how a smoke
    test proves the plumbing without leaving something that looks like a finding in
    ``docs/measurements/`` -- the convention this project already follows by hand.
    """
    return Path(out_dir) / f"evaluation-{label}"


def run_instrument(run: InstrumentRun, *, log=print) -> InstrumentRun:
    """Invoke an instrument as a subprocess and read back its JSON.

    Subprocess rather than in-process import on purpose: the argv IS the provenance record,
    so the report can state the exact command that produced each number and a reader can
    paste it and get the same file back.
    """
    log(f"\n$ {' '.join(run.argv)}")
    completed = subprocess.run(run.argv, cwd=str(ROOT))
    if completed.returncode != 0:
        run.status = f"FAILED (exit {completed.returncode})"
        return run
    if not run.json_path.is_file():
        run.status = f"FAILED (no JSON at {run.json_path})"
        return run
    run.payload = json.loads(run.json_path.read_text(encoding="utf-8"))
    run.status = "ok"
    return run


def behaviour_invocation(facts: ModelFacts, out_dir: Path, *, prompt_set: str,
                         num_samples: int, max_new_tokens: int, seed: int,
                         no_register: bool) -> InstrumentRun:
    md = out_dir / f"behaviour-set{prompt_set.upper()}.md"
    js = out_dir / f"behaviour-set{prompt_set.upper()}.json"
    argv = [sys.executable, str(ROOT / "scripts" / "score_behaviour.py"),
            "--hf-model", str(facts.path),
            "--label", facts.label,
            "--prompt-set", prompt_set,
            "--num-samples", str(num_samples),
            "--max-new-tokens", str(max_new_tokens),
            "--seed", str(seed),
            "--out", str(md), "--json-out", str(js)]
    if no_register:
        argv.append("--no-register")
    return InstrumentRun(name="behaviour", argv=argv, json_path=js, md_path=md)


def context_invocation(facts: ModelFacts, out_dir: Path, *, window: int, tokens: Path,
                       n_windows: int, seed: int) -> InstrumentRun:
    md = out_dir / "context-use.md"
    js = out_dir / "context-use.json"
    argv = [sys.executable, str(ROOT / "scripts" / "probe_context_use.py"),
            "--hf-model", str(facts.path),
            "--tokens", str(tokens),
            "--seq-len", str(window),
            "--n-windows", str(n_windows),
            "--seed", str(seed),
            "--out", str(md), "--json-out", str(js)]
    return InstrumentRun(name="context", argv=argv, json_path=js, md_path=md)


def per_source_invocation(facts: ModelFacts, out_dir: Path, *, window: int,
                          n_windows: int, seed: int) -> InstrumentRun:
    md = out_dir / "per-source-loss.md"
    js = out_dir / "per-source-loss.json"
    argv = [sys.executable, str(ROOT / "scripts" / "eval_per_source.py"),
            "--hf-model", str(facts.path),
            "--seq-len", str(window),
            "--n-windows", str(n_windows),
            "--seed", str(seed),
            "--out", str(md), "--json-out", str(js)]
    return InstrumentRun(name="per-source", argv=argv, json_path=js, md_path=md)


def pooled_window_loss(buckets: Sequence[dict], window: int) -> Optional[float]:
    """Mean loss over the whole window, pooled from ``probe_context_use``'s buckets.

    Each bucket's ``mean`` is an unweighted mean over the positions it covers, and the
    buckets tile ``[0, window)`` exactly, so weighting by bucket width recovers the
    position-mean over the window. This is the number that makes two models comparable on
    loss **by construction**: same instrument, same window, same token array, one run.
    """
    total = 0.0
    span = 0
    for bucket in buckets:
        width = int(bucket["hi"]) - int(bucket["lo"])
        if bucket.get("mean") is None:
            return None
        total += float(bucket["mean"]) * width
        span += width
    if span != window or span == 0:
        return None
    return total / span


# ---------------------------------------------------------------------------------------
# Mode 1: evaluate one model
# ---------------------------------------------------------------------------------------


@dataclass
class SingleEvaluation:
    facts: ModelFacts
    window: int
    tokens: TokenArray
    prompt_set: str
    num_samples: int
    n_windows: int
    seed: int
    runs: List[InstrumentRun]
    designation: Optional[Designation]
    started_utc: str

    @property
    def behaviour(self) -> Optional[dict]:
        return next((r.payload for r in self.runs if r.name == "behaviour" and r.payload),
                    None)

    @property
    def context(self) -> Optional[dict]:
        return next((r.payload for r in self.runs if r.name == "context" and r.payload), None)

    @property
    def window_loss(self) -> Optional[float]:
        payload = self.context
        if not payload:
            return None
        return pooled_window_loss(payload.get("overall") or [], self.window)

    def comparability(self) -> dict:
        """The facts that decide whether this report may be compared to another one."""
        return {
            "eval_window": self.window,
            "model_max_position_embeddings": self.facts.max_position_embeddings,
            "training_window": self.facts.training_window,
            "tokens": self.tokens.as_json(),
            "prompt_set": self.prompt_set,
            "num_samples": self.num_samples,
            "n_windows": self.n_windows,
            "seed": self.seed,
        }

    def as_json(self) -> dict:
        return {
            "kind": "tt-tnt/evaluation/1",
            "mode": "single",
            "started_utc": self.started_utc,
            "model": self.facts.as_json(),
            "comparability": self.comparability(),
            "window_loss": self.window_loss,
            "instruments": [r.as_json() for r in self.runs],
            "designation": self.designation.as_json() if self.designation else None,
        }


def render_single(ev: SingleEvaluation) -> str:
    """The one-model report. Leads with comparability, because that is what gets lost."""
    f = ev.facts
    lines: List[str] = list(_SPDX)
    lines += ["", f"# Evaluation — {f.label}", ""]
    lines.append(
        f"Generated by `scripts/evaluate.py --model {_rel(f.path)}` on {ev.started_utc}. "
        f"This is a **joined** report: the instruments below were run as one benchmark so "
        f"their numbers share a window, a token array and a prompt set. Every number in it "
        f"is CPU-only.")
    lines += ["", "## Comparability — read this before quoting any number", ""]
    lines.append(
        "These are the facts that decide whether a number here may be placed beside a number "
        "from another report. This project has published a headline that compared a "
        "512-token-window loss against a 2048-token-window loss; the table exists so that "
        "cannot happen quietly again.")
    lines.append("")
    lines.append("| fact | value |")
    lines.append("|---|---|")
    lines.append(f"| eval window (`seq_len`) | **{ev.window}** |")
    lines.append(f"| model `max_position_embeddings` | {f.max_position_embeddings} |")
    lines.append(
        f"| training-time eval window (what `val_losses.jsonl` is in) | {f.training_window} |")
    lines.append(f"| token array | `{_rel(ev.tokens.path)}` |")
    lines.append(
        f"| token array size | {ev.tokens.size_bytes:,} bytes"
        + (f", {ev.tokens.n_tokens:,} tokens |" if ev.tokens.n_tokens else " |"))
    lines.append(f"| prompt set | **{ev.prompt_set.upper()}** "
                 f"(`{PROMPT_SETS[ev.prompt_set].path.name}`) |")
    lines.append(f"| completions per prompt | {ev.num_samples} |")
    lines.append(f"| windows sampled per probe | {ev.n_windows} |")
    lines.append(f"| seed | {ev.seed} |")
    lines.append("")
    if ev.window < f.max_position_embeddings:
        lines.append(
            f"⚠️ The eval window ({ev.window}) is **shorter** than this model's context "
            f"({f.max_position_embeddings}). That is deliberate — the window is a fixed "
            f"constant, not a function of the model, so that models with different contexts "
            f"stay commensurable — but it means this report does not measure what the model "
            f"does with its full context. `probe_context_use.py --seq-len "
            f"{f.max_position_embeddings}` does, and its numbers may not be compared with "
            f"these.")
        lines.append("")

    if ev.designation is not None and ev.designation.hf_model.resolve() == f.path.resolve():
        lines += ["## This model is the designated current model", ""]
        lines.append(f"- **reason:** {ev.designation.reason}")
        lines.append(f"- **qualification:** {ev.designation.qualification}")
        lines.append(f"- recorded in `{_rel(ev.designation.source)}` "
                     f"({ev.designation.designated})")
        lines.append("")

    lines += ["## Instruments run", ""]
    lines.append("| instrument | status | output | invocation |")
    lines.append("|---|---|---|---|")
    for run in ev.runs:
        lines.append(f"| {run.name} | {run.status} | `{_rel(run.json_path)}` | "
                     f"`{' '.join(_short_argv(run.argv))}` |")
    lines.append("")

    loss = ev.window_loss
    lines += ["## Loss", ""]
    if loss is None:
        lines.append("The context probe did not produce a usable pooled loss, so no "
                     "window-mean loss is reported. A blank is the honest output here; an "
                     "estimate assembled from whatever partial numbers exist is not.")
    else:
        lines.append(
            f"**Mean loss over a {ev.window}-token window: {loss:.4f} nats.** Pooled from "
            f"`probe_context_use.py`'s position buckets, which tile `[0, {ev.window})` "
            f"exactly, so this is the position-mean over the window and not a re-derivation "
            f"of anything. Comparable **only** to another model's number at the same window "
            f"on the same token array.")
        lines.append("")
        lines.append("| position bucket | mean | SEM | n windows |")
        lines.append("|---|---:|---:|---:|")
        for bucket in (ev.context or {}).get("overall", []):
            lines.append(f"| [{bucket['lo']}, {bucket['hi']}) | {bucket['mean']:.4f} | "
                         f"{bucket['sem']:.4f} | {bucket['n_windows']} |")
    lines.append("")

    behaviour = ev.behaviour
    lines += ["## Behaviour", ""]
    if behaviour is None:
        lines.append("The behavioural instrument did not complete; no behavioural numbers "
                     "are reported.")
    else:
        behaviour_run = next((r for r in ev.runs if r.name == "behaviour"), None)
        full = (f" Full report: `{_rel(behaviour_run.md_path)}`."
                if behaviour_run and behaviour_run.md_path else "")
        lines.append(
            f"`score_behaviour.py` on prompt set "
            f"**{str(behaviour.get('prompt_set', '?')).upper()}**, "
            f"{behaviour.get('n_prompts')} prompts x {behaviour.get('num_samples')} "
            f"completions.{full}")
        lines.append("")
        lines.append("| signal | mean | SEM | 95% CI |")
        lines.append("|---|---:|---:|---|")
        for signal in behaviour.get("signals", []):
            est = (behaviour.get("aggregate") or {}).get(signal["key"])
            if not est:
                continue
            lo, hi = est["ci95"]
            lines.append(f"| {signal['title']} | {est['mean']:.4f} | {est['sem']:.4f} | "
                         f"[{lo:.4f}, {hi:.4f}] |")
    lines.append("")

    lines += ["## What this report does NOT say", ""]
    lines += [
        "- **Nothing about another model.** A single-model report has no noise floor in it: "
        "with one run there is nothing to compare a number against, so no number here is "
        "'good' or 'bad'. Use `--model A --against B` for that, which applies the seed-only "
        "floor automatically.",
        "- **Nothing at a window other than the one in the table above.** That is the whole "
        "point of the table.",
        f"- **Nothing about prompt set "
        f"{'A' if ev.prompt_set != 'a' else 'B'}.** The frozen sets are reported separately "
        "and never pooled.",
        "",
    ]
    return "\n".join(lines)


def _short_argv(argv: Sequence[str]) -> List[str]:
    """Shorten an argv for a markdown table: interpreter and repo prefix elided."""
    out: List[str] = []
    for i, item in enumerate(argv):
        if i == 0:
            out.append("python")
        else:
            out.append(item.replace(str(ROOT) + "/", ""))
    return out


def evaluate_single(facts: ModelFacts, *, window: int, tokens: TokenArray, prompt_set: str,
                    num_samples: int, max_new_tokens: int, n_windows: int, seed: int,
                    instruments: Sequence[str], no_register: bool,
                    designation: Optional[Designation], reuse: bool,
                    root_out_dir: Path = MEASUREMENTS,
                    log=print) -> SingleEvaluation:
    """Run the requested instruments for one model into its own run directory."""
    out_dir = run_dir_for(facts.label, root_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: List[InstrumentRun] = []
    if "behaviour" in instruments:
        runs.append(behaviour_invocation(facts, out_dir, prompt_set=prompt_set,
                                         num_samples=num_samples,
                                         max_new_tokens=max_new_tokens, seed=seed,
                                         no_register=no_register))
    if "context" in instruments:
        runs.append(context_invocation(facts, out_dir, window=window, tokens=tokens.path,
                                       n_windows=n_windows, seed=seed))
    if "per-source" in instruments:
        runs.append(per_source_invocation(facts, out_dir, window=window,
                                          n_windows=n_windows, seed=seed))
    for run in runs:
        if reuse and run.json_path.is_file():
            run.payload = json.loads(run.json_path.read_text(encoding="utf-8"))
            run.status = "reused"
            log(f"reusing {_rel(run.json_path)} (--reuse)")
            continue
        run_instrument(run, log=log)
    return SingleEvaluation(
        facts=facts, window=window, tokens=tokens, prompt_set=prompt_set,
        num_samples=num_samples, n_windows=n_windows, seed=seed, runs=runs,
        designation=designation,
        started_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


# ---------------------------------------------------------------------------------------
# Mode 2: compare two models
# ---------------------------------------------------------------------------------------


@dataclass
class Comparison:
    baseline: SingleEvaluation
    candidate: SingleEvaluation
    window: int
    floor: Optional[SeedFloor]
    floor_problems: List[str]
    behaviour: List[LabelledDifference]
    trajectory: Optional[TrajectoryComparison]
    trajectory_refusal: Optional[str]
    window_loss_delta: Optional[float]
    started_utc: str

    def as_json(self) -> dict:
        return {
            "kind": "tt-tnt/evaluation/1",
            "mode": "comparison",
            "started_utc": self.started_utc,
            "baseline": self.baseline.as_json(),
            "candidate": self.candidate.as_json(),
            "eval_window": self.window,
            "seed_floor": self.floor.as_json() if self.floor else None,
            "seed_floor_problems": self.floor_problems,
            "floor_ratio_threshold": FLOOR_RATIO_MIN,
            "matched_window_loss_delta": self.window_loss_delta,
            "behaviour": [d.as_json() for d in self.behaviour],
            "trajectory": self.trajectory.as_json() if self.trajectory else None,
            "trajectory_refusal": self.trajectory_refusal,
        }


def compare(baseline: SingleEvaluation, candidate: SingleEvaluation, *,
            floor: Optional[SeedFloor], floor_problems: List[str],
            skip_trajectory: bool) -> Comparison:
    """Join two evaluations: behaviour paired by prompt, loss guarded by window."""
    window = require_matched_window(
        baseline.facts.label, baseline.window, candidate.facts.label, candidate.window,
        what="window-mean losses from this run")

    behaviour: List[LabelledDifference] = []
    base_payload, cand_payload = baseline.behaviour, candidate.behaviour
    if base_payload and cand_payload:
        # paired_differences() itself refuses to pair runs from different prompt sets; that
        # refusal is preserved here rather than re-implemented or worked around.
        behaviour = label_differences(paired_differences(base_payload, cand_payload), floor)

    trajectory: Optional[TrajectoryComparison] = None
    refusal: Optional[str] = None
    if skip_trajectory:
        refusal = ("skipped by --skip-trajectory; no training-time loss comparison is "
                   "reported")
    else:
        try:
            trajectory = compare_trajectories(baseline.facts, candidate.facts, floor)
        except (WindowMismatch, FileNotFoundError, ValueError) as exc:
            refusal = str(exc)

    base_loss, cand_loss = baseline.window_loss, candidate.window_loss
    delta = None if (base_loss is None or cand_loss is None) else cand_loss - base_loss

    return Comparison(
        baseline=baseline, candidate=candidate, window=window, floor=floor,
        floor_problems=floor_problems, behaviour=behaviour, trajectory=trajectory,
        trajectory_refusal=refusal, window_loss_delta=delta,
        started_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _fmt(value: Optional[float], places: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def render_comparison(cmp: Comparison) -> str:
    """The comparison report: floor ratio beside every delta, no exceptions."""
    base, cand = cmp.baseline, cmp.candidate
    lines: List[str] = list(_SPDX)
    lines += ["", f"# Comparison — {base.facts.label} → {cand.facts.label}", ""]
    lines.append(
        f"Baseline `{_rel(base.facts.path)}` vs candidate `{_rel(cand.facts.path)}`, "
        f"generated by `scripts/evaluate.py` on {cmp.started_utc}. Every delta is "
        f"`candidate − baseline`.")

    lines += ["", "## Matched conditions", ""]
    lines.append("| fact | baseline | candidate |")
    lines.append("|---|---|---|")
    lines.append(f"| eval window | **{base.window}** | **{cand.window}** |")
    lines.append(f"| `max_position_embeddings` | {base.facts.max_position_embeddings} | "
                 f"{cand.facts.max_position_embeddings} |")
    lines.append(f"| training-time eval window | {base.facts.training_window} | "
                 f"{cand.facts.training_window} |")
    lines.append(f"| token array | `{_rel(base.tokens.path)}` | "
                 f"`{_rel(cand.tokens.path)}` |")
    lines.append(f"| prompt set | {base.prompt_set.upper()} | {cand.prompt_set.upper()} |")
    lines.append(f"| completions/prompt | {base.num_samples} | {cand.num_samples} |")
    lines.append(f"| seed | {base.seed} | {cand.seed} |")
    lines.append("")
    if base.facts.training_window != cand.facts.training_window:
        lines.append(
            f"⚠️ **The two models were TRAINED at different windows** "
            f"({base.facts.training_window} vs {cand.facts.training_window}). This run "
            f"evaluates both at {cmp.window}, so the numbers below are matched — but their "
            f"training-time `val_losses.jsonl` trajectories are **not** commensurable, and "
            f"the trajectory section says so rather than differencing them.")
        lines.append("")

    lines += ["## The noise floor", ""]
    if cmp.floor is None:
        lines.append(
            "**No seed-only floor is available, so no ratio is printed anywhere in this "
            "report.** Every delta below should be read as uninterpretable until a floor "
            "exists. Reasons:")
        for problem in cmp.floor_problems:
            lines.append(f"- {problem}")
    else:
        floor = cmp.floor
        lines.append(
            f"The floor is the **seed-only control**: two runs identical in everything but "
            f"the RNG seed. Whatever it moves, a seed moves. Derived at runtime "
            f"(`{floor.provenance}`) from `{floor.behaviour_source}`"
            + (f" and {', '.join(f'`{s}`' for s in floor.loss_sources)}"
               if floor.loss_sources else "")
            + ", never hardcoded.")
        lines.append("")
        if floor.loss_sd is not None:
            lines.append(
                f"- **loss floor:** sd of the seed control's per-checkpoint differences = "
                f"**{floor.loss_sd:.4f} nats** (mean {_fmt(floor.loss_mean)}, sign "
                f"{floor.loss_sign.n_negative}/{floor.loss_sign.n} negative). The *spread*, "
                f"not the mean: that control's mean delta is near zero precisely because "
                f"its sign wanders, so the spread is what a candidate has to beat.")
        for problem in cmp.floor_problems:
            lines.append(f"- ⚠️ {problem}")
        lines.append(
            f"- **the rule:** a delta at or below **{FLOOR_RATIO_MIN}x** the floor is "
            f"`{NOT_INTERPRETABLE}`, *whatever its confidence interval says*. A −0.041 "
            f"register delta at 1.03x the floor cleared its CI, was written up, and was "
            f"refuted by exactly this control; two collapse signals at 1.01x and 1.05x were "
            f"correctly excluded on the same rule.")
        lines.append(
            "- **and the mirror-image rule:** a large ratio over a tiny denominator that "
            "cannot clear its own paired minimum-detectable difference is "
            "`below paired detection`, not a finding. Both gates have to pass.")
    lines.append("")

    lines += ["## Loss at a matched window", ""]
    base_loss, cand_loss = base.window_loss, cand.window_loss
    if cmp.window_loss_delta is None:
        lines.append(
            "Not available: at least one side has no pooled window loss (the context probe "
            "did not run or did not complete).")
    else:
        lines.append(
            f"Both models measured by the **same instrument, at the same "
            f"{cmp.window}-token window, on the same token array, in this run** — which is "
            f"the one construction under which two losses in this project are known to be "
            f"commensurable.")
        lines.append("")
        lines.append("| | baseline | candidate | delta |")
        lines.append("|---|---:|---:|---:|")
        lines.append(f"| mean loss over `[0, {cmp.window})` | {_fmt(base_loss)} | "
                     f"{_fmt(cand_loss)} | {cmp.window_loss_delta:+.4f} |")
        lines.append("")
        lines.append(
            "This number has **no seed floor of its own** — the seed control's floor is over "
            "training-time trajectories, not over this instrument — so it is reported "
            "without a ratio and should be read alongside the trajectory section below "
            "rather than on its own.")
    lines.append("")

    lines += ["## Loss trajectory (training-time validation)", ""]
    if cmp.trajectory is None:
        lines.append(f"**Not computed.** {cmp.trajectory_refusal}")
    else:
        t = cmp.trajectory
        lines.append(
            f"Both trajectories are at a **{t.window}-token window** (checked, not assumed: "
            f"a mismatch here is refused outright). {t.n_steps} shared checkpoint steps.")
        lines.append("")
        lines.append("| statistic | value | note |")
        lines.append("|---|---:|---|")
        lines.append(
            f"| **endpoint delta** (step {t.final_step}) | **{t.final_delta:+.4f}** | "
            f"the headline: {_fmt(t.candidate_final)} vs {_fmt(t.baseline_final)} |")
        lines.append(
            f"| average over checkpoints | {t.mean_delta:+.4f} | *not* the headline — an "
            f"average over a trajectory where two runs start apart and converge is dominated "
            f"by the early steps, which once made an effect look 9.7x its true size |")
        lines.append(f"| sd of per-checkpoint deltas | {t.sd_delta:.4f} | |")
        lines.append(
            f"| **sign test** | {t.sign.n_negative}/{t.sign.n} negative, p = "
            + (f"{t.sign.p_two_sided:.2g}" if t.sign.p_two_sided is not None else "n/a")
            + " | preferred for trajectories: independent of spread |")
        if t.floor_sign is not None:
            lines.append(
                f"| seed floor's own sign split | {t.floor_sign.n_negative}/"
                f"{t.floor_sign.n} negative | the floor *changes sign*, which is what makes "
                f"a unanimous candidate convincing |")
        lines.append(
            "| endpoint / floor | " + ("n/a" if t.ratio is None else f"**{t.ratio:.2f}x**")
            + f" | floor sd {_fmt(t.floor_sd)} |")
        lines.append(f"| **label** | **{t.label}** | both gates applied |")
        lines.append("")
        lines.append(
            "Both the sign test and the mean-vs-sd summary are given because they answer "
            "different questions: the sign test says *how consistently* the difference "
            "pointed one way, mean-vs-sd says *how large* it was relative to run-to-run "
            "variation. A finding wants both.")
        if (t.floor_window is not None and t.floor_window != t.window):
            lines.append("")
            lines.append(
                f"⚠️ The seed floor was measured at a **{t.floor_window}-token** window and "
                f"these trajectories are at **{t.window}**. The ratio is still printed "
                f"because a run-to-run spread is the closest thing to a floor this project "
                f"has, and it is the same floor the committed 384s512 analysis used — but it "
                f"is a floor borrowed across windows, and that is a caveat on the ratio, not "
                f"a property of it.")
    lines.append("")

    lines += ["## Behaviour, signal by signal", ""]
    if not cmp.behaviour:
        lines.append("No behavioural comparison: the behavioural instrument did not produce "
                     "output for both models.")
    else:
        lines.append(
            f"Paired by prompt over frozen set **{base.prompt_set.upper()}**. The sets are "
            f"never pooled — `score_behaviour.paired_differences` refuses a cross-set pair "
            f"outright, and this tool scores both models on one set by construction.")
        lines.append("")
        lines.append("| signal | better | baseline | candidate | delta | 95% CI | "
                     "min. detectable | seed floor | ratio | **label** |")
        lines.append("|---|---|---:|---:|---:|---|---:|---:|---:|---|")
        for d in cmp.behaviour:
            ci = ("n/a" if d.ci95 is None
                  else f"[{d.ci95[0]:+.4f}, {d.ci95[1]:+.4f}]")
            ratio = "n/a" if d.ratio is None else f"{d.ratio:.2f}x"
            lines.append(
                f"| {d.title} | {d.better or 'n/a'} | {_fmt(d.baseline_mean)} | "
                f"{_fmt(d.candidate_mean)} | {_fmt(d.delta)} | {ci} | "
                f"{_fmt(d.min_detectable)} | {_fmt(d.floor)} | {ratio} | "
                f"**{d.label}** |")
        lines.append("")
        lines += ["### Verdict", ""]
        if cmp.floor is None:
            # With no floor there are no findings to summarise, only deltas. Quoting the
            # threshold here would put a number in the report that nothing in it was
            # measured against, which is the habit this whole section exists to break.
            lines.append(
                f"**No verdict.** Without a seed-only floor none of the {len(cmp.behaviour)} "
                f"signals above can be told from run-to-run variation, so every one of them "
                f"is `{NO_FLOOR}` and none is a finding.")
        else:
            interpretable = [d for d in cmp.behaviour if d.label in ("better", "worse")]
            excluded = [d for d in cmp.behaviour if d.label == NOT_INTERPRETABLE]
            lines.append(
                "- **findings (clear the floor AND the paired interval):** "
                + (", ".join(f"{d.title} ({d.label}, {d.ratio:.2f}x)" for d in interpretable)
                   or "none"))
            lines.append(
                f"- **at or below {FLOOR_RATIO_MIN}x the seed floor — {NOT_INTERPRETABLE} "
                f"whatever their intervals say:** "
                + (", ".join(d.title if d.ratio is None else f"{d.title} ({d.ratio:.2f}x)"
                             for d in excluded) or "none"))
            lines.append(
                "- **above the floor but inside their own paired interval:** "
                + (", ".join(d.title for d in cmp.behaviour
                             if d.label == "below paired detection") or "none"))
            lines.append(
                "- **no floor available for:** "
                + (", ".join(d.title for d in cmp.behaviour if d.label == NO_FLOOR)
                   or "none"))
    lines.append("")

    lines += ["## What would refute this", ""]
    lines += [
        "- A **seed replicate** of either arm. Every claim above rests on one run per arm; "
        "the floor is one seed pair, and a second pair could widen it.",
        "- A **matched-window** re-measurement of anything quoted here from a different "
        "window. Where the two models' training windows differ, the trajectory comparison is "
        "refused rather than estimated, and that refusal is not a formality.",
        "- The other frozen prompt set. A behavioural effect that appears on set B and not "
        "on set A is a fact about the sets as much as about the models.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# Mode 3: the ad-hoc escape valve
# ---------------------------------------------------------------------------------------


def assert_scratch_path(path: Path) -> Path:
    """Refuse to write an ad-hoc sample anywhere a benchmark result could live.

    The frozen prompt sets are digest-pinned so checkpoints stay comparable. This mode is the
    deliberate way around that, which means its output is the single most mistakable thing
    this tool produces: it is one person's impulse, generated once, at temperatures chosen on
    the spot. So it is kept physically apart — outside ``docs/``, outside git — and a stray
    ``--out`` is refused rather than obeyed.
    """
    resolved = Path(path).expanduser().resolve()
    try:
        # relative_to(), not startswith(): "…/adhoc-promptsEVIL" is not inside
        # "…/adhoc-prompts", and a string prefix test cannot tell the two apart.
        resolved.relative_to(SCRATCH.resolve())
    except ValueError:
        raise ScratchPathViolation(
            f"refusing to write an ad-hoc sample to {resolved}: --try output belongs under "
            f"{_rel(SCRATCH)}/ and nowhere else. An ad-hoc completion is not a measurement — "
            f"it comes from a prompt nobody froze, at temperatures chosen on the spot, with "
            f"n=1 per temperature — and it must not be able to sit in the same directory as "
            f"a result a finding was built on.")
    if "behaviour" in resolved.name:
        raise ScratchPathViolation(
            f"refusing to write {resolved}: 'behaviour-*' is the measurement namespace "
            f"produced by scripts/score_behaviour.py. Ad-hoc output must not borrow it.")
    return resolved


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (cleaned[:40] or "prompt")


def adhoc_output_path(first_prompt: str, *, now: Optional[datetime] = None) -> Path:
    """Timestamped scratch path. Timestamped so two ad-hoc runs never overwrite each other."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return SCRATCH / f"ADHOC-{stamp}-{_slug(first_prompt)}.md"


def read_prompt_file(path: Path) -> List[str]:
    """One prompt per line, or a JSON list/``{"prompts": [...]}`` object.

    Blank lines and ``#`` comments are dropped, so a scratch file can be annotated.
    """
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        payload = json.loads(text)
        items = payload if isinstance(payload, list) else payload.get("prompts", [])
        prompts = [(p["text"] if isinstance(p, dict) else str(p)) for p in items]
    else:
        prompts = [line for line in (ln.strip() for ln in text.splitlines())
                   if line and not line.startswith("#")]
    if not prompts:
        raise ValueError(f"{path} contains no prompts")
    return prompts


ADHOC_BANNER = "SCRATCH — AD-HOC SAMPLES, NOT A BENCHMARK RESULT"


def render_adhoc(prompts: Sequence[str], completions: Dict[Tuple[int, float], str], *,
                 model: Path, temperatures: Sequence[float], max_new_tokens: int,
                 seed: int, out_path: Path) -> str:
    """Render the ad-hoc markdown, banner first, so the file cannot be mistaken."""
    lines = [
        f"# {ADHOC_BANNER}",
        "",
        f"**These completions are NOT a measurement.** They come from "
        f"{len(prompts)} prompt(s) that nobody froze, at temperatures chosen on the spot, "
        f"with one sample per (prompt, temperature). Nothing here is comparable to any "
        f"other checkpoint, and nothing here belongs in `docs/measurements/`.",
        "",
        f"Written to `{_rel(out_path)}` — outside `docs/`, outside git.",
        "",
        f"model: `{_rel(model)}` · seed {seed} · {max_new_tokens} new tokens · "
        f"temperatures {', '.join('greedy' if t == 0 else str(t) for t in temperatures)}",
        "",
        "## If one of these turns out to be diagnostic",
        "",
        "Promote it into a **new** frozen set with **new ids**, in a deliberate commit — "
        "`docs/evaluation_prompts.json` (set A) and `docs/evaluation_prompts_b.json` (set B) "
        "are digest-pinned so every checkpoint scored against them stays comparable, and "
        "editing either one silently invalidates every measurement already taken with it. "
        "A third set is cheap; a broken digest is not.",
        "",
    ]
    for i, prompt in enumerate(prompts):
        lines += [f"## prompt {i + 1}", "", f"> {prompt}", ""]
        for temp in temperatures:
            name = "greedy (deterministic)" if temp == 0 else f"temperature {temp}"
            text = completions.get((i, temp), "")
            lines += [f"**{name}:**", "", f"> {prompt}**{text}**", ""]
    lines += [f"_{ADHOC_BANNER}_", ""]
    return "\n".join(lines)


def run_adhoc(model_dir: Path, prompts: Sequence[str], *, temperatures: Sequence[float],
              max_new_tokens: int, seed: int, out_path: Path, log=print) -> Path:
    """Generate at each temperature and write the scratch file. Torch is imported here only."""
    out_path = assert_scratch_path(out_path)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"loading {model_dir} ...")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype="auto").eval()
    torch.manual_seed(seed)

    completions: Dict[Tuple[int, float], str] = {}
    for i, prompt in enumerate(prompts):
        ids = tok(prompt, return_tensors="pt").input_ids
        for temp in temperatures:
            kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temp > 0}
            if temp > 0:
                kwargs.update(temperature=float(temp), top_p=0.95, top_k=0)
            with torch.no_grad():
                got = model.generate(input_ids=ids, **kwargs)
            text = tok.decode(got[0][ids.shape[1]:], skip_special_tokens=True)
            completions[(i, temp)] = text
            name = "greedy" if temp == 0 else f"t={temp}"
            log(f"  prompt {i + 1} {name:>8}: {text[:70]!r}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_adhoc(prompts, completions, model=model_dir,
                                     temperatures=temperatures,
                                     max_new_tokens=max_new_tokens, seed=seed,
                                     out_path=out_path), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------

_EPILOG = """
examples
--------
  # one model (defaults to the designated current model in docs/current_model.json)
  python scripts/evaluate.py --model artifacts/hf-tt-tnt-1024a

  # the comparison, with the seed-only noise floor applied automatically
  python scripts/evaluate.py --model artifacts/hf-tt-tnt-1024a \\
      --against artifacts/hf-tt-tnt-384s512

  # the ad-hoc escape valve -- scratch output, never a measurement
  python scripts/evaluate.py --try "The lighthouse keeper wrote in the log:"

  # re-render the committable seed-floor snapshot from the raw seed-only control
  python scripts/evaluate.py --refresh-floor

on --try, and on prompts that turn out to be good
-------------------------------------------------
The frozen sets (docs/evaluation_prompts.json, docs/evaluation_prompts_b.json) are
digest-pinned so every checkpoint scored against them stays comparable, which by design
forbids trying a new prompt on impulse. --try is the escape valve, and its output goes to
scratch/adhoc-prompts/ -- outside docs/, outside git, banner-marked, and refused if pointed
anywhere else.

A prompt that proves genuinely DIAGNOSTIC should be promoted into a **NEW set with new
ids**, in a deliberate commit -- never by editing an existing set. Editing set A or set B
silently invalidates every measurement already taken against it: the digest tests
(tests/test_evaluation_prompts.py, tests/test_evaluation_prompts_b.py) will fail, and that
failure is the feature. A third set costs one file and one test module; a broken digest
costs every comparison in docs/measurements/.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scripts/evaluate.py",
        description=__doc__,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--model", type=Path, default=None,
                   help="Converted HF model directory to evaluate (CPU only). Defaults to "
                        "the model designated in docs/current_model.json.")
    p.add_argument("--against", type=Path, default=None, metavar="BASELINE",
                   help="Compare --model (candidate) against this BASELINE model. Turns on "
                        "comparison mode: the seed-only noise floor is applied to every "
                        "delta, and losses measured at different windows are REFUSED.")
    p.add_argument("--try", dest="try_prompt", type=str, default=None, metavar="TEXT",
                   help="Ad-hoc mode: generate from this prompt at several temperatures and "
                        "write to scratch/adhoc-prompts/. Never touches the frozen prompt "
                        "sets or the behaviour-* measurement namespace.")
    p.add_argument("--try-file", type=Path, default=None, metavar="PATH",
                   help="Ad-hoc mode over several prompts: one per line (# comments and "
                        "blank lines ignored), or a JSON list / {\"prompts\": [...]} object.")
    p.add_argument("--refresh-floor", action="store_true",
                   help="Re-render docs/measurements/seed-noise-floor.json from the raw "
                        "seed-only control (tt-tnt-v3 vs tt-tnt-v5) and exit. The snapshot "
                        "is committable, so a fresh clone still has a floor; it is GENERATED "
                        "and must never be hand-edited.")

    g = p.add_argument_group("evaluation conditions (recorded in every report)")
    g.add_argument("--window", type=int, default=None,
                   help=f"Evaluation window / seq_len (default: {DEFAULT_WINDOW}, narrowed "
                        f"if a model cannot reach it). Deliberately a CONSTANT and not the "
                        f"model's own max_position_embeddings -- a window that rides along "
                        f"with the model is how this project compared a 512-window loss to a "
                        f"2048-window loss.")
    g.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS,
                   help="Token array losses are measured on (default: %(default)s). Recorded "
                        "in the report: this repo has three token generations on disk.")
    g.add_argument("--prompt-set", default=DEFAULT_PROMPT_SET, choices=sorted(PROMPT_SETS),
                   help="Frozen prompt set for the behavioural instrument (default: "
                        "%(default)s). Both sides of a comparison use ONE set; the sets are "
                        "never pooled.")
    g.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                   help="Completions per frozen prompt (default: %(default)s).")
    g.add_argument("--max-new-tokens", type=int, default=60)
    g.add_argument("--n-windows", type=int, default=DEFAULT_N_WINDOWS,
                   help="Windows sampled by the context probe (default: %(default)s).")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--instruments", default="behaviour,context",
                   help="Comma-separated subset of behaviour,context,per-source (default: "
                        "%(default)s). per-source is off by default because it replays the "
                        "entire ~1.7 GB blend through the tokenizer, which takes minutes and "
                        "needs artifacts/corpus/blend.txt.")
    g.add_argument("--no-register", action="store_true",
                   help="Pass --no-register to score_behaviour.py (skips fitting the "
                        "per-source register models and their detector controls).")
    g.add_argument("--out-dir", type=Path, default=MEASUREMENTS,
                   help="Where reports land (default: %(default)s). Point a SMOKE run "
                        "somewhere else -- scratch/, say -- so a 2-sample plumbing check can "
                        "never be mistaken for a measurement, which is the convention this "
                        "project already follows by hand.")
    g.add_argument("--reuse", action="store_true",
                   help="Reuse an instrument's JSON if it already exists under this run's "
                        "output directory instead of re-running it. Convenient, and a way to "
                        "quote stale numbers -- the report records which outputs were reused.")
    g.add_argument("--skip-trajectory", action="store_true",
                   help="Comparison mode only: drop the training-time loss trajectory "
                        "comparison. The escape hatch for comparing two models trained at "
                        "different windows -- the refusal is not bypassed, the comparison is "
                        "simply not made, and the report says so.")

    t = p.add_argument_group("ad-hoc mode (--try / --try-file)")
    t.add_argument("--temperatures", default=None,
                   help="Comma-separated temperatures for --try (default: "
                        + ",".join(str(x) for x in DEFAULT_TRY_TEMPERATURES)
                        + "; 0 means greedy).")
    t.add_argument("--try-max-new-tokens", type=int, default=DEFAULT_TRY_MAX_NEW_TOKENS)
    t.add_argument("--try-out", type=Path, default=None,
                   help=f"Override the ad-hoc output path. Must still be under "
                        f"{_rel(SCRATCH)}/ -- anything else is refused.")
    return p


def assert_writable_out_dir(out_dir: Path) -> Path:
    """Refuse an ``--out-dir`` anywhere under ``artifacts/``.

    ``artifacts/`` holds the irreplaceable evidence -- checkpoints, converted weights, the
    tokenizer, the token arrays, the corpus -- none of which can be regenerated without
    retraining or a multi-hour rebuild. This tool only ever produces reports, so it has no
    business writing there at all, and saying so in one place is cheaper than trusting every
    future invocation. Same rule and same spirit as ``train/paths.py``'s protected paths.
    """
    resolved = Path(out_dir).expanduser().resolve()
    artifacts = (ROOT / "artifacts").resolve()
    if resolved == artifacts or artifacts in resolved.parents:
        raise ValueError(
            f"refusing --out-dir {resolved}: nothing under artifacts/ may be written by an "
            f"evaluation run. That tree holds checkpoints, converted weights, the tokenizer "
            f"and the token arrays -- evidence that cannot be regenerated without "
            f"retraining. Reports belong under docs/measurements/ (the default) or "
            f"scratch/.")
    return Path(out_dir)


def _resolve_subject(args) -> Tuple[ModelFacts, Optional[Designation]]:
    """The model under test: ``--model``, else the designated current model.

    The designation is returned either way, so a report can say "this IS the current model"
    or "this is not" rather than leaving the reader to work it out. A missing designation is
    fatal only when it was the thing being relied on.
    """
    try:
        designation: Optional[Designation] = load_designation()
    except DesignationError:
        if args.model is None:
            raise
        designation = None
    if args.model is not None:
        return read_model_facts(args.model), designation
    print(f"no --model given; using the designated current model "
          f"'{designation.label}' from {_rel(designation.source)}")
    print(f"  reason: {designation.reason}")
    print(f"  qualification: {designation.qualification}")
    return read_model_facts(designation.hf_model), designation


def _mode_try(args) -> int:
    prompts: List[str] = []
    if args.try_prompt:
        prompts.append(args.try_prompt)
    if args.try_file:
        prompts.extend(read_prompt_file(args.try_file))
    temperatures = ([float(x) for x in args.temperatures.split(",")]
                    if args.temperatures else list(DEFAULT_TRY_TEMPERATURES))
    for temp in temperatures:
        if temp < 0:
            raise ValueError(f"temperature must be >= 0 (0 means greedy), got {temp}")
        if temp > 0:
            validate_sampling_args(temp, 0.95, 0, 1)

    designation = None
    if args.model is None:
        designation = load_designation()
        model_dir = resolve_model_dir(str(designation.hf_model))
    else:
        model_dir = resolve_model_dir(str(args.model))

    # Validate the destination BEFORE the banner and before loading a model: a refused
    # --try-out must fail as a one-line error, not after minutes of generation.
    out_path = assert_scratch_path(args.try_out or adhoc_output_path(prompts[0]))
    print("=" * 78)
    print(f"  {ADHOC_BANNER}")
    print("=" * 78)
    print(f"  model     : {_rel(model_dir)}"
          + (f"  (designated current: {designation.label})" if designation else ""))
    print(f"  prompts   : {len(prompts)} (ad-hoc, NOT from a frozen set)")
    print(f"  writing to: {_rel(out_path)}   <- scratch, outside docs/, outside git")
    print("  These completions are not comparable to any checkpoint and must not be "
          "quoted as one.")
    print("=" * 78)

    written = run_adhoc(model_dir, prompts, temperatures=temperatures,
                        max_new_tokens=args.try_max_new_tokens, seed=args.seed,
                        out_path=out_path)
    print(f"\nwrote {written}")
    print(f"({ADHOC_BANNER})")
    print("A prompt that proves genuinely diagnostic belongs in a NEW frozen set with NEW "
          "ids, in a deliberate commit -- never by editing an existing set.")
    return 0


def _mode_single(args) -> int:
    assert_writable_out_dir(args.out_dir)
    facts, designation = _resolve_subject(args)
    window = common_window([facts], args.window)
    tokens = fingerprint_tokens(args.tokens)
    instruments = [s.strip() for s in args.instruments.split(",") if s.strip()]

    ev = evaluate_single(facts, window=window, tokens=tokens, prompt_set=args.prompt_set,
                         num_samples=args.num_samples, max_new_tokens=args.max_new_tokens,
                         n_windows=args.n_windows, seed=args.seed, instruments=instruments,
                         no_register=args.no_register, designation=designation,
                         reuse=args.reuse, root_out_dir=args.out_dir)

    md_path = args.out_dir / f"evaluation-{facts.label}.md"
    json_path = args.out_dir / f"evaluation-{facts.label}.json"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_single(ev), encoding="utf-8")
    json_path.write_text(json.dumps(ev.as_json(), indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {md_path}")
    print(f"wrote {json_path}")
    failed = [r.name for r in ev.runs if r.status.startswith("FAILED")]
    if failed:
        print(f"WARNING: instruments that failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def _mode_compare(args) -> int:
    assert_writable_out_dir(args.out_dir)
    candidate_facts, designation = _resolve_subject(args)
    baseline_facts = read_model_facts(args.against)

    window = common_window([baseline_facts, candidate_facts], args.window)
    tokens = fingerprint_tokens(args.tokens)
    instruments = [s.strip() for s in args.instruments.split(",") if s.strip()]

    floor, problems = resolve_seed_floor()
    if floor is None:
        print("WARNING: no seed-only noise floor could be derived; NO ratios will be "
              "printed. Reasons:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
    else:
        print(f"seed floor: {floor.provenance}; "
              f"{len(floor.behaviour)} behavioural signals"
              + (f", loss sd {floor.loss_sd:.4f}" if floor.loss_sd is not None
                 else ", NO loss floor"))
        for problem in problems:
            print(f"  ! {problem}")

    # Fail on a window mismatch BEFORE spending an hour of CPU on two model evaluations
    # whose losses would then be refused. The refusal must be cheap to hit and impossible
    # to miss.
    if not args.skip_trajectory:
        try:
            require_matched_window(
                baseline_facts.label, baseline_facts.training_window,
                candidate_facts.label, candidate_facts.training_window,
                what="training validation losses")
        except WindowMismatch as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            print("\nPass --skip-trajectory to compare BEHAVIOUR only (this run's own "
                  "matched-window losses are still reported); the trajectory comparison "
                  "will be recorded as refused, with the reason.", file=sys.stderr)
            return 2

    evaluations = []
    for facts in (baseline_facts, candidate_facts):
        print(f"\n{'=' * 78}\nevaluating {facts.label}\n{'=' * 78}")
        evaluations.append(evaluate_single(
            facts, window=window, tokens=tokens, prompt_set=args.prompt_set,
            num_samples=args.num_samples, max_new_tokens=args.max_new_tokens,
            n_windows=args.n_windows, seed=args.seed, instruments=instruments,
            no_register=args.no_register, designation=designation, reuse=args.reuse,
            root_out_dir=args.out_dir))
    baseline, candidate = evaluations

    try:
        cmp = compare(baseline, candidate, floor=floor, floor_problems=problems,
                      skip_trajectory=args.skip_trajectory)
    except (WindowMismatch, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    label = f"{baseline.facts.label}-vs-{candidate.facts.label}"
    md_path = args.out_dir / f"evaluation-{label}.md"
    json_path = args.out_dir / f"evaluation-{label}.json"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_comparison(cmp), encoding="utf-8")
    json_path.write_text(json.dumps(cmp.as_json(), indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {md_path}")
    print(f"wrote {json_path}")
    for d in cmp.behaviour:
        ratio = "  n/a" if d.ratio is None else f"{d.ratio:5.2f}x"
        print(f"  {d.title:34} {_fmt(d.delta):>10}  {ratio}  {d.label}")
    if cmp.trajectory:
        t = cmp.trajectory
        print(f"  {'loss endpoint (matched window)':34} {t.final_delta:>+10.4f}  "
              + ("  n/a" if t.ratio is None else f"{t.ratio:5.2f}x")
              + f"  {t.label}  [sign {t.sign.n_negative}/{t.sign.n}]")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    modes = [bool(args.against), bool(args.try_prompt or args.try_file),
             bool(args.refresh_floor)]
    if sum(modes) > 1:
        parser.error("--against, --try/--try-file and --refresh-floor are different modes; "
                     "pick one")

    try:
        if args.refresh_floor:
            floor = write_floor_snapshot()
            print(f"wrote {FLOOR_SNAPSHOT_PATH}")
            print(f"  behavioural signals: {len(floor.behaviour)}")
            print("  loss floor sd: "
                  + (f"{floor.loss_sd:.4f}" if floor.loss_sd is not None else "unavailable"))
            for note in floor.notes:
                print(f"  ! {note}")
            return 0
        if args.try_prompt or args.try_file:
            return _mode_try(args)
        if args.against:
            return _mode_compare(args)
        return _mode_single(args)
    except (DesignationError, WindowMismatch, ScratchPathViolation, FileNotFoundError,
            ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
