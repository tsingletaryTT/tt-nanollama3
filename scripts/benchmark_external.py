#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""External benchmarks: the first numbers in this project that our own corpus did not produce.

THE PROBLEM THIS MEASURES
--------------------------
Every quantity this project reports is **self-referential**. Validation loss is computed on a
tail of our own blend. ``scripts/eval_per_source.py`` slices that same blend by source.
``scripts/score_behaviour.py`` scores two prompt sets we wrote ourselves, against a register
metric whose whole definition is "similarity to text drawn from our own training backbone".
``scripts/evaluate.py``'s noise floor is derived from our own seed-only control run. All of it
is internally consistent and none of it is anchored to anything outside the repo: a model that
had learned to imitate our corpus and nothing else would score exactly as well on every one of
those instruments as a model that had learned English.

This script runs the model against benchmarks **someone else built, on data we did not
choose**, using EleutherAI's ``lm-evaluation-harness`` -- the same tool and the same task
definitions the published literature reports against. That makes it a categorically different
kind of evidence from everything else in ``docs/measurements/``, and it is the only kind that
can contradict us.

WHAT TO EXPECT (READ THIS BEFORE READING A SCORE)
--------------------------------------------------
The model is ~123M parameters trained on ~0.35B tokens. Its reference class, GPT-2 small
(124M), saw roughly 25x more data. **Chance-level scores on the multiple-choice tasks are the
predicted outcome, not a malfunction**, and this script is built to say so out loud rather than
to print a number that a reader could mistake for a capability.

That is why every task here carries an explicit ``chance`` baseline and why
:func:`chance_verdict` refuses to let a score within :data:`CHANCE_SE_MULTIPLE` standard errors
of chance be reported as a quantity: it is rendered as ``AT CHANCE`` and its
``reportable_score`` in the JSON is ``null``. This is the same move ``scripts/evaluate.py``
makes when it labels a delta inside ~1.2x of the seed-noise floor ``NOT INTERPRETABLE``
regardless of what its confidence interval says -- a measurement that a null model would have
produced just as easily is not a measurement of this model.

The two tasks that can say something *graded* about a model this small:

* **WikiText perplexity** -- a pure language-modelling metric with no chance floor to sit at,
  reported by essentially every LM paper since 2016. lm-eval's ``wikitext`` task uses the
  ``wikitext-2-raw-v1`` test split, which is **byte-identical** to the ``wikitext-103-raw-v1``
  test split (both are the same 62 documents / 1,288,493 characters; verified by
  :func:`verify_wikitext_test_identity`, and by ``tests/test_benchmark_external.py``). So the
  number this task produces IS WikiText-103 test perplexity, and is comparable to the figure
  every paper quotes -- with the protocol caveats in :data:`PROTOCOL_CAVEATS`.
* **LAMBADA** -- predict the final word of a passage that is only predictable from long-range
  context. This is the external analogue of ``scripts/probe_context_use.py``, which found on
  our own data that per-token loss stops improving past position ~64. Agreement between the
  two instruments is itself a finding; disagreement is a bigger one.

HOW IT RUNS (AND THE DEPENDENCY RULE IT RESPECTS)
--------------------------------------------------
``pyproject.toml`` lists three runtime dependencies and this project has repeatedly declined to
add a fourth. ``lm-eval`` pulls in dozens (torch, datasets, sqlitedict, ...) and would dwarf
the repo's own dependency set, so it is **not** installed into this project's environment.
Instead it lives in a throwaway virtualenv (default ``scratch/lm-eval-venv``, which is
gitignored) and this script **shells out** to it: see :func:`run_lm_eval`. Nothing in this file
imports ``lm_eval``, ``torch``, or ``datasets``; it uses only ``transformers`` (already a
dependency) and the standard library.

The venv's absolute path, the exact ``lm_eval`` version, and the versions of ``torch``,
``transformers`` and ``datasets`` inside it are recorded in both reports (see
:func:`venv_provenance`), because a benchmark number without the harness version attached is
not reproducible -- lm-eval has changed task definitions between minor releases.

CPU ONLY. This script passes ``--device cpu`` unconditionally and refuses any other device
(:func:`require_cpu_device`). It never imports ttml/ttnn and never opens a Tenstorrent device.

THE CONTEXT-WINDOW CAVEAT
--------------------------
``artifacts/hf-tt-tnt-1024a`` has ``max_position_embeddings=512``, half of GPT-2's 1024. That
matters in two different ways, which this script keeps separate because conflating them would
be dishonest in opposite directions:

1. **Truncation** (multiple-choice / LAMBADA tasks). If a prompt plus its continuation exceeds
   the window, lm-eval drops tokens off the FRONT of the context. The model then answers a
   question it was shown only part of, and the resulting score is not a fair score for that
   task. :func:`analyse_truncation` re-tokenizes every request lm-eval actually issued (from
   its ``--log_samples`` output) and counts exactly how many were over the limit, so a
   truncated task is flagged in the report instead of being quietly averaged in.
2. **Windowing** (WikiText, a rolling-loglikelihood task). Nothing is dropped: the document is
   scored in consecutive ``max_length`` chunks. But every token still sees at most
   ``max_length - 1`` tokens of context, so a 512-window model is genuinely handicapped against
   a 1024-window one on the same text. That is a property of our model, not of the benchmark,
   and it is reported as a caveat rather than as truncation.

THE GPT-2 COMPARISON
---------------------
:data:`GPT2_PUBLISHED` carries the figures from the GPT-2 paper for the 117M/124M model, with
their source and an explicit note that the paper's protocol is NOT lm-eval's. Published numbers
under a different protocol are weak evidence, so the preferred path is ``--reference-json``:
run this same script against ``gpt2`` first, then pass the resulting JSON as the reference for
the real model. Then the comparison is *measured* on identical task definitions, identical
harness version and identical machine, and the report says which kind of comparison each row is
(see :func:`build_report`).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: Where the throwaway lm-eval virtualenv lives by default. Under ``scratch/``, which
#: ``.gitignore`` already excludes ("an ad-hoc completion is not a measurement" -- and a
#: 1.7 GB torch install is certainly not one either).
DEFAULT_VENV = ROOT / "scratch" / "lm-eval-venv"

#: Where lm-eval's own raw output (results JSON + per-sample JSONL) is written. Also under
#: ``scratch/``: it is an intermediate, not the report. The report is the two files this
#: script writes into ``docs/measurements/``.
DEFAULT_RUN_ROOT = ROOT / "scratch" / "lm-eval-runs"

#: Forward-pass batch size. Throughput knob only -- lm-eval sorts requests by length and the
#: loglikelihoods it computes do not depend on how they were batched.
DEFAULT_BATCH_SIZE = 16

#: Weights are stored bfloat16; they are up-cast to float32 for CPU evaluation. CPU bf16
#: matmul is both slower and lower-precision than fp32 here, so fp32 is the honest choice --
#: it removes numerical precision as an explanation for a low score.
DEFAULT_DTYPE = "float32"

#: How many standard errors a score must sit away from its chance baseline before this script
#: will report it as a quantity at all.
#:
#: 2.0 is ~95% two-sided for the (approximately normal) binomial proportion lm-eval's
#: ``*_stderr`` estimates. It is deliberately a *gate*, in the same spirit as
#: ``scripts/evaluate.py::FLOOR_RATIO_MIN``: below it, the score is not reported as a number,
#: it is reported as ``AT CHANCE``. The raw score is always still printed next to the verdict
#: so a reader can see exactly what was suppressed and why -- this is a labelling rule, not a
#: concealment rule.
CHANCE_SE_MULTIPLE = 2.0

#: Verdict strings. ``AT_CHANCE`` is the one this whole script exists to be able to say.
AT_CHANCE = "AT CHANCE"
ABOVE_CHANCE = "ABOVE CHANCE"
BELOW_CHANCE = "BELOW CHANCE"
NO_CHANCE_BASELINE = "NO CHANCE BASELINE"
NO_STDERR = "NO STANDARD ERROR"


# ---------------------------------------------------------------------------------------
# The task list, with what each one's chance baseline actually is
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Metric:
    """One reported number from one task, and the null it has to beat to mean anything.

    ``chance`` is ``None`` only for continuous metrics (perplexity, bits-per-byte) where there
    is no "guess at random" strategy to compare against. Those are never labelled ``AT
    CHANCE``; they are reported directly, because a perplexity is informative at any value.
    """

    key: str
    """lm-eval's own metric name, e.g. ``acc`` or ``word_perplexity``."""

    title: str
    """How the metric is named in the report."""

    chance: Optional[float]
    """Score a model that has learned nothing would get. ``None`` for continuous metrics."""

    higher_is_better: bool

    chance_note: str = ""
    """Why ``chance`` is the number it is, when that is not self-evident."""


@dataclass(frozen=True)
class TaskSpec:
    """One lm-eval task, its metrics, and what it is doing in this report."""

    task: str
    """The ``--tasks`` name passed to lm-eval. Also the key in its results JSON."""

    title: str
    metrics: Tuple[Metric, ...]
    why: str
    """One sentence: what this task tells us that the others do not."""

    rolling: bool = False
    """True for ``loglikelihood_rolling`` tasks (WikiText). Their documents are *windowed*,
    not truncated -- see the module docstring's CONTEXT-WINDOW CAVEAT."""


#: The fixed task list. Ordered by how much a model this small can be expected to say
#: through it: the two graded language-modelling tasks first, then the multiple-choice
#: benchmarks where chance is the predicted outcome.
TASKS: Tuple[TaskSpec, ...] = (
    TaskSpec(
        task="wikitext",
        title="WikiText-103 (test) perplexity",
        metrics=(
            Metric("word_perplexity", "word perplexity", None, False),
            Metric("byte_perplexity", "byte perplexity", None, False),
            Metric("bits_per_byte", "bits per byte", None, False),
        ),
        why="A pure language-modelling metric with no chance floor -- the one number here on "
            "which a 123M-parameter model can be scored meaningfully rather than at chance. "
            "lm-eval's `wikitext` task is the wikitext-2-raw-v1 test split, which is "
            "byte-identical to wikitext-103-raw-v1's test split, so this IS the "
            "WikiText-103 test perplexity the literature quotes.",
        rolling=True,
    ),
    TaskSpec(
        task="lambada_openai",
        title="LAMBADA (OpenAI variant)",
        metrics=(
            Metric("acc", "last-word accuracy", 0.0, True,
                   chance_note="Open-vocabulary: the model must produce the exact final word, "
                               "so chance is ~1/|vocab| (~3e-5 for our 32,000-token "
                               "vocabulary), i.e. indistinguishable from zero. A nonzero "
                               "score here is therefore always above chance -- the "
                               "interesting question for this task is the magnitude, and the "
                               "comparison to GPT-2."),
            Metric("perplexity", "perplexity of the final word", None, False),
        ),
        why="Predict the last word of a passage that is only predictable from long-range "
            "context. This is the external analogue of `scripts/probe_context_use.py`, which "
            "found on our own data that per-token loss stops improving past position ~64.",
    ),
    TaskSpec(
        task="hellaswag",
        title="HellaSwag",
        metrics=(
            Metric("acc", "accuracy", 0.25, True,
                   chance_note="4-way multiple choice."),
            Metric("acc_norm", "length-normalised accuracy", 0.25, True,
                   chance_note="4-way multiple choice."),
        ),
        why="Commonsense sentence completion. Notoriously hard below ~1B parameters; GPT-2 "
            "small itself barely clears chance.",
    ),
    TaskSpec(
        task="piqa",
        title="PIQA",
        metrics=(
            Metric("acc", "accuracy", 0.5, True, chance_note="2-way multiple choice."),
            Metric("acc_norm", "length-normalised accuracy", 0.5, True,
                   chance_note="2-way multiple choice."),
        ),
        why="Physical commonsense, 2-way. The most forgiving of the multiple-choice tasks and "
            "so the likeliest place for a small model to show any signal at all.",
    ),
    TaskSpec(
        task="winogrande",
        title="WinoGrande",
        metrics=(
            Metric("acc", "accuracy", 0.5, True, chance_note="2-way multiple choice."),
        ),
        why="Pronoun resolution requiring world knowledge, 2-way. Adversarially filtered "
            "against exactly the surface statistics a small LM has.",
    ),
    TaskSpec(
        task="arc_easy",
        title="ARC-Easy",
        metrics=(
            Metric("acc", "accuracy", 0.25, True,
                   chance_note="Mostly 4-way multiple choice (a handful of items have 3 or 5 "
                               "options, so 0.25 is approximate)."),
            Metric("acc_norm", "length-normalised accuracy", 0.25, True),
        ),
        why="Grade-school science questions, the easy split. Requires factual knowledge our "
            "corpus (public-domain literature and Simple Wikipedia) only partly contains.",
    ),
    TaskSpec(
        task="arc_challenge",
        title="ARC-Challenge",
        metrics=(
            Metric("acc", "accuracy", 0.25, True,
                   chance_note="Mostly 4-way multiple choice."),
            Metric("acc_norm", "length-normalised accuracy", 0.25, True),
        ),
        why="The split ARC built specifically to defeat retrieval and co-occurrence "
            "heuristics.",
    ),
    TaskSpec(
        task="mmlu",
        title="MMLU (0-shot)",
        metrics=(
            Metric("acc", "accuracy", 0.25, True, chance_note="4-way multiple choice."),
        ),
        why="57 subjects of exam knowledge. Included as a floor check: nothing at this scale "
            "moves on MMLU, so a result here that is NOT at chance would be the surprising "
            "outcome and worth investigating as a bug before believing it.",
    ),
)

TASKS_BY_NAME: Dict[str, TaskSpec] = {t.task: t for t in TASKS}


# ---------------------------------------------------------------------------------------
# The GPT-2 reference class
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PublishedFigure:
    """A number someone else published, and enough context to distrust it correctly."""

    value: float
    source: str
    caveat: str = ""


#: GPT-2 small (117M in the paper, 124M as shipped) zero-shot figures from Radford et al.
#: 2019, "Language Models are Unsupervised Multitask Learners", Table 3.
#:
#: These are here so a reader can see the DATA GAP -- GPT-2 saw ~40B tokens of WebText against
#: our ~0.35B -- rather than mistaking a low score for a broken model. They are NOT a like-for-
#: like comparison: the GPT-2 paper used its own invertible detokenizers and its own scoring
#: code, and lm-eval's task definitions differ in detail. Prefer ``--reference-json`` (an
#: actual gpt2 run through THIS harness) wherever it is available; the report marks which kind
#: of comparison each row is.
GPT2_PUBLISHED: Dict[Tuple[str, str], PublishedFigure] = {
    ("wikitext", "word_perplexity"): PublishedFigure(
        37.50, "GPT-2 paper (Radford et al. 2019) Table 3, WikiText-103, 117M, zero-shot",
        "The paper applies invertible detokenizers before scoring and normalises by its own "
        "word count; lm-eval's word_perplexity uses a different detokenizer and word count, "
        "so the two are close in spirit and not identical in protocol."),
    ("lambada_openai", "acc"): PublishedFigure(
        0.4599, "GPT-2 paper Table 3, LAMBADA accuracy, 117M, zero-shot",
        "The paper's LAMBADA accuracy uses a stopword filter on generation; lm-eval's "
        "lambada_openai scores the exact continuation by loglikelihood, which typically "
        "reads lower (~0.33 for gpt2 in lm-eval)."),
    ("lambada_openai", "perplexity"): PublishedFigure(
        35.13, "GPT-2 paper Table 3, LAMBADA perplexity, 117M, zero-shot"),
}

#: GPT-2 small as shipped (`gpt2` on the Hub), counted by lm-eval's own
#: ``model_num_parameters`` on the real checkpoint rather than quoted as the paper's "117M".
GPT2_SMALL_PARAMS = 124_439_808

#: WebText is ~40 GB of text and was never released with an official token count; ~40 billion
#: is the figure usually quoted for it and is right to within a factor that does not matter at
#: the ~100x scale this comparison is making. Stated as approximate everywhere it is used.
GPT2_SMALL_TRAINING_TOKENS = 40_000_000_000

#: Caveats that apply to EVERY published comparison, printed once in the report rather than
#: repeated per row.
PROTOCOL_CAVEATS: Tuple[str, ...] = (
    "Published figures were produced by their authors' own code, not by lm-eval. Task "
    "definitions, detokenizers and normalisation all differ in detail, so treat a published "
    "row as an order-of-magnitude reference and not as a head-to-head result.",
    "A row marked `(measured)` is a real head-to-head: GPT-2 small was run through this same "
    "harness — same lm-eval version, same task definitions, same machine, same day — rather "
    "than quoted. The one thing it does not match is the context window, which is each "
    "model's own; GPT-2 small's is 1024.",
)


# ---------------------------------------------------------------------------------------
# Reading the subject model's facts (filesystem + JSON only -- no torch)
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelFacts:
    """What the report needs to say about the subject before any score is shown."""

    path: Path
    label: str
    max_position_embeddings: int
    hidden_size: Optional[int]
    num_hidden_layers: Optional[int]
    n_params: Optional[int] = None
    training_tokens: Optional[int] = None
    training_tokens_note: str = ""


def default_label(model_dir: Path) -> str:
    """``artifacts/hf-tt-tnt-1024a`` -> ``tt-tnt-1024a``; anything else -> its own basename.

    Matches ``scripts/probe_context_use.py::_default_output_paths``'s convention so the
    ``docs/measurements/`` filenames from the two tools line up.
    """
    name = model_dir.name
    return name[len("hf-"):] if name.startswith("hf-") else name


def read_model_config(model_dir: Path) -> dict:
    """Load ``config.json``, raising a message that says what was missing and why it matters."""
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"no config.json at {model_dir}; this script cannot record the model's context "
            f"window, and a benchmark score without the window it was produced at is not "
            f"interpretable (see this module's CONTEXT-WINDOW CAVEAT)")
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{config_path} is not valid JSON: {exc}") from exc


def checkpoint_dir_for(model_dir: Path) -> Optional[Path]:
    """``artifacts/hf-tt-tnt-1024a`` -> ``artifacts/checkpoints-tt-tnt-1024a``, if it exists.

    Mirrors ``scripts/evaluate.py::checkpoint_dir_for``. Used only to find ``train.log``, which
    is where the training-token count can be derived from; a missing directory is not an error.
    """
    name = model_dir.name
    if not name.startswith("hf-"):
        return None
    candidate = model_dir.parent / f"checkpoints-{name[len('hf-'):]}"
    return candidate if candidate.is_dir() else None


def parse_training_tokens(train_log: Path) -> Tuple[Optional[int], str]:
    """Tokens actually seen in training, read off ``train.log``'s own summary line.

    ``train/`` logs one line of the form::

        tt-tnt training — steps=10764 batch=64 seq_len=512 arch=blackhole

    and ``steps * batch * seq_len`` is the number of tokens the optimiser actually consumed.
    That is the honest denominator for "how much data did this model see", and it is NOT the
    same as the corpus size: a run can stop short of one epoch or go round more than once.

    Returns ``(tokens, note)``. ``tokens`` is ``None`` when the line is absent or malformed --
    a missing provenance line is reported as missing, never guessed at.
    """
    if not train_log.is_file():
        return None, f"no train.log at {train_log}"
    fields: Dict[str, int] = {}
    for line in train_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "tt-tnt training" not in line:
            continue
        for token in line.split():
            for key in ("steps", "batch", "seq_len"):
                prefix = f"{key}="
                if token.startswith(prefix):
                    try:
                        fields[key] = int(token[len(prefix):])
                    except ValueError:
                        return None, f"unparseable {key!r} in {train_log}: {token!r}"
        break
    missing = [k for k in ("steps", "batch", "seq_len") if k not in fields]
    if missing:
        return None, (f"{train_log} has no usable 'tt-tnt training' summary line "
                      f"(missing {', '.join(missing)})")
    tokens = fields["steps"] * fields["batch"] * fields["seq_len"]
    note = (f"{fields['steps']:,} steps x batch {fields['batch']} x seq_len "
            f"{fields['seq_len']} (from {train_log.relative_to(ROOT) if train_log.is_relative_to(ROOT) else train_log})")
    return tokens, note


#: Config keys that hold the trained context window, in preference order. Llama-family configs
#: (ours) use ``max_position_embeddings``; GPT-2's config -- which matters because the GPT-2
#: reference run goes through this same code path -- writes ``n_positions`` instead. Guessing
#: between them is not the same as guessing a value: if none of these keys is present the
#: model is refused, never given a default window.
CONTEXT_WINDOW_KEYS: Tuple[str, ...] = ("max_position_embeddings", "n_positions", "n_ctx")


def context_window(config: dict) -> Optional[int]:
    """The trained context window from a config dict, or ``None`` if it does not state one."""
    for key in CONTEXT_WINDOW_KEYS:
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def read_model_facts(model_dir: Path) -> ModelFacts:
    """Everything the report states about the subject model, from disk only."""
    config = read_model_config(model_dir)
    max_pos = context_window(config)
    if max_pos is None:
        raise ValueError(
            f"{model_dir / 'config.json'} states no context window (looked for "
            f"{', '.join(CONTEXT_WINDOW_KEYS)}). Without it this script cannot tell whether a "
            f"benchmark prompt was truncated, and an unflagged truncated score is worse than "
            f"no score.")
    ckpt = checkpoint_dir_for(model_dir)
    tokens, note = (None, "no checkpoint directory found for this model")
    if ckpt is not None:
        tokens, note = parse_training_tokens(ckpt / "train.log")
    return ModelFacts(
        path=model_dir,
        label=default_label(model_dir),
        max_position_embeddings=max_pos,
        hidden_size=config.get("hidden_size"),
        num_hidden_layers=config.get("num_hidden_layers"),
        training_tokens=tokens,
        training_tokens_note=note,
    )


def current_model_path() -> Optional[Path]:
    """``docs/current_model.json``'s designated subject, or ``None`` if that file is absent.

    Same source of truth ``scripts/evaluate.py --model`` defaults to, so "benchmark the model"
    has one unambiguous answer here too.
    """
    path = ROOT / "docs" / "current_model.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    hf_model = payload.get("current", {}).get("hf_model")
    return (ROOT / hf_model) if isinstance(hf_model, str) else None


# ---------------------------------------------------------------------------------------
# The venv: provenance, and the refusal to proceed without one
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VenvProvenance:
    """Which interpreter produced these numbers, and what was installed in it."""

    python: Path
    versions: Dict[str, str]

    def as_json(self) -> dict:
        return {"python": str(self.python), "versions": dict(self.versions)}


def venv_python(venv: Path) -> Path:
    """The interpreter inside ``venv``, checked to exist with an actionable message."""
    python = venv / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(
            f"no interpreter at {python}. lm-eval is deliberately NOT a dependency of this "
            f"repo (see this module's docstring); it lives in a separate throwaway venv. "
            f"Create one with:\n"
            f"    python3 -m venv {venv}\n"
            f"    {venv}/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            f"    {venv}/bin/pip install lm-eval==0.4.9 'transformers<5'\n"
            f"then re-run. Do NOT install lm-eval into this project's environment.")
    return python


#: Packages whose versions are recorded with every run. lm-eval has changed task definitions
#: between minor releases, and transformers/torch changes can move a loglikelihood in the last
#: decimal place -- a score without these attached cannot be reproduced.
PROVENANCE_PACKAGES: Tuple[str, ...] = ("lm_eval", "torch", "transformers", "datasets")


def venv_provenance(python: Path, packages: Sequence[str] = PROVENANCE_PACKAGES,
                    runner=subprocess.run) -> VenvProvenance:
    """Import each package in the venv and record its ``__version__``.

    A package that fails to import is recorded as ``"not installed"`` rather than raising:
    the run itself will fail loudly enough if something essential is missing, and a partial
    provenance record is more useful than none when diagnosing that failure.
    """
    program = (
        "import json\n"
        f"names = {list(packages)!r}\n"
        "out = {}\n"
        "for n in names:\n"
        "    try:\n"
        "        out[n] = __import__(n).__version__\n"
        "    except Exception as exc:\n"
        "        out[n] = 'not installed (%s)' % type(exc).__name__\n"
        "print(json.dumps(out))\n"
    )
    result = runner([str(python), "-c", program], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"could not read package versions from {python}: {result.stderr.strip()}")
    return VenvProvenance(python=python, versions=json.loads(result.stdout.strip()))


def require_cpu_device(device: str) -> str:
    """Refuse anything but CPU. This script must never touch a Tenstorrent device.

    The lm-eval HF backend would happily accept ``tt`` or ``cuda`` here; this project's
    measurement scripts are CPU-only by design (see ``scripts/probe_context_use.py``'s
    CONSTRAINTS section), and a benchmark harness silently grabbing an accelerator would
    also silently take a device lease out from under whatever else is running.
    """
    if device != "cpu":
        raise ValueError(
            f"--device {device!r} refused: scripts/benchmark_external.py is CPU-only by "
            f"design. It never opens a Tenstorrent device and never imports ttml/ttnn.")
    return device


# ---------------------------------------------------------------------------------------
# Running lm-eval
# ---------------------------------------------------------------------------------------


def lm_eval_command(python: Path, model_dir: Path, tasks: Sequence[str], *, max_length: int,
                    batch_size: int, dtype: str, output_path: Path, device: str = "cpu",
                    limit: Optional[int] = None, num_fewshot: Optional[int] = None,
                    log_samples: bool = True) -> List[str]:
    """The exact argv used, built in one place so the report can quote it verbatim.

    ``max_length`` is passed explicitly rather than left to lm-eval's autodetection: lm-eval
    would read it from ``config.json`` anyway, but stating it makes the truncation analysis
    downstream check the same number the harness actually used, instead of a number it
    assumed.
    """
    require_cpu_device(device)
    model_args = (f"pretrained={model_dir},dtype={dtype},max_length={max_length}")
    argv = [
        str(python), "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", ",".join(tasks),
        "--device", device,
        "--batch_size", str(batch_size),
        "--output_path", str(output_path),
    ]
    if log_samples:
        argv.append("--log_samples")
    if limit is not None:
        argv += ["--limit", str(limit)]
    if num_fewshot is not None:
        argv += ["--num_fewshot", str(num_fewshot)]
    return argv


def find_results_json(output_path: Path) -> Path:
    """lm-eval writes ``<output_path>/<model_sanitized>/results_<timestamp>.json``.

    Returns the most recent one. Raises if there is none -- which is what a crashed or
    silently-empty run looks like, and it must not be mistaken for "no results yet".
    """
    candidates = sorted(output_path.glob("**/results_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"no results_*.json under {output_path}; lm-eval did not complete a run there")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_lm_eval(argv: Sequence[str], output_path: Path, *, runner=subprocess.run) -> Path:
    """Run the harness, streaming its output, and return the results JSON it wrote."""
    output_path.mkdir(parents=True, exist_ok=True)
    result = runner(list(argv), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"lm-eval exited {result.returncode}. Command:\n    {' '.join(argv)}\n"
            f"Nothing has been written to docs/measurements/ -- a partial benchmark run is "
            f"not a measurement.")
    return find_results_json(output_path)


# ---------------------------------------------------------------------------------------
# The chance-baseline rule -- this script's whole character
# ---------------------------------------------------------------------------------------


def chance_verdict(score: Optional[float], chance: Optional[float], stderr: Optional[float],
                   *, se_multiple: float = CHANCE_SE_MULTIPLE) -> str:
    """Is this score distinguishable from what a model that learned nothing would get?

    - No chance baseline (a perplexity) -> :data:`NO_CHANCE_BASELINE`. Continuous metrics are
      informative at any value and are reported directly.
    - No usable standard error -> :data:`NO_STDERR`, **except** when the score is exactly the
      chance baseline. lm-eval reports ``"N/A"`` for some aggregated metrics, and a zero
      standard error is what a task returns when every item came out the same way; without a
      spread there is generally no way to tell a real gap from sampling noise, so this script
      declines to call it either way rather than eyeballing the margin. The one case it will
      call is a score sitting exactly on chance with no spread at all -- e.g. 0/N correct on
      LAMBADA, whose chance is zero -- which is unambiguously :data:`AT_CHANCE` and would be
      misleading to report as merely uninterpretable.
    - Within ``se_multiple`` standard errors of chance -> :data:`AT_CHANCE`. **This is the
      predicted outcome for a 123M-parameter model on every multiple-choice task here**, and
      it is a verdict, not a failure to measure.
    - Otherwise :data:`ABOVE_CHANCE` / :data:`BELOW_CHANCE`.

    Note ``BELOW_CHANCE`` is a real and reportable outcome, not a bug: length-normalised
    multiple-choice scoring can be systematically anti-correlated with the answer for a model
    whose likelihoods are dominated by surface frequency.
    """
    if chance is None:
        return NO_CHANCE_BASELINE
    if score is None:
        return NO_STDERR
    if stderr is None or not math.isfinite(stderr) or stderr <= 0.0:
        return AT_CHANCE if score == chance else NO_STDERR
    z = (score - chance) / stderr
    if abs(z) < se_multiple:
        return AT_CHANCE
    return ABOVE_CHANCE if z > 0 else BELOW_CHANCE


def z_from_chance(score: Optional[float], chance: Optional[float],
                  stderr: Optional[float]) -> Optional[float]:
    """``(score - chance) / stderr``, or ``None`` when that would be meaningless."""
    if score is None or chance is None or stderr is None:
        return None
    if not math.isfinite(stderr) or stderr <= 0.0:
        return None
    return (score - chance) / stderr


def reportable_score(score: Optional[float], verdict: str) -> Optional[float]:
    """The score, or ``None`` when the verdict says it must not be read as a quantity.

    This is the mechanical form of the rule: an ``AT CHANCE`` result has no
    ``reportable_score`` in the JSON at all, so a downstream consumer cannot accidentally
    plot it, average it, or quote it as a capability. The raw value is still present as
    ``score`` for anyone who needs to see what was suppressed.
    """
    return None if verdict in (AT_CHANCE, NO_STDERR) else score


def headline(score: Optional[float], verdict: str, *, digits: int = 4) -> str:
    """How a score is rendered in prose: the number, or the words ``at chance``."""
    if verdict == AT_CHANCE:
        return "at chance"
    if verdict == NO_STDERR:
        return "not interpretable (no standard error)"
    if score is None:
        return "n/a"
    return f"{score:.{digits}f}"


# ---------------------------------------------------------------------------------------
# Truncation: did the 512-token window cut prompts off?
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TruncationReport:
    """How many of a task's requests did not fit inside the model's context window."""

    task: str
    n_requests: int
    n_truncated: int
    max_tokens: int
    max_length: int
    rolling: bool
    note: str = ""

    @property
    def truncated(self) -> bool:
        return self.n_truncated > 0

    def as_json(self) -> dict:
        return {
            "task": self.task, "n_requests": self.n_requests,
            "n_truncated": self.n_truncated, "max_request_tokens": self.max_tokens,
            "max_length": self.max_length, "rolling": self.rolling,
            "truncated": self.truncated, "note": self.note,
        }


def iter_request_texts(sample: dict) -> Iterable[Tuple[str, str]]:
    """Every ``(context, continuation)`` pair lm-eval actually scored for one document.

    lm-eval's ``--log_samples`` writes ``arguments`` as ``{"gen_args_0": {"arg_0": context,
    "arg_1": continuation}, ...}`` -- one entry per multiple-choice option. Rolling tasks have
    only ``arg_0`` (the whole document). Older harness versions wrote a list of pairs instead,
    so both shapes are accepted; anything else raises rather than being silently skipped,
    because a truncation count computed over a subset of the requests is a false all-clear.
    """
    arguments = sample.get("arguments")
    if isinstance(arguments, dict):
        for value in arguments.values():
            if isinstance(value, dict):
                yield str(value.get("arg_0", "")), str(value.get("arg_1", "") or "")
            elif isinstance(value, (list, tuple)):
                yield str(value[0]), str(value[1]) if len(value) > 1 else ""
            else:
                raise ValueError(f"unrecognised lm-eval argument entry: {type(value).__name__}")
        return
    if isinstance(arguments, (list, tuple)):
        for value in arguments:
            if isinstance(value, (list, tuple)):
                yield str(value[0]), str(value[1]) if len(value) > 1 else ""
            else:
                raise ValueError(f"unrecognised lm-eval argument entry: {type(value).__name__}")
        return
    raise ValueError(
        f"lm-eval sample has no usable 'arguments' field (got {type(arguments).__name__}); "
        f"cannot verify whether this task's prompts fit the model's context window")


def analyse_truncation(samples_paths: Sequence[Path], tokenizer, *, task: str, max_length: int,
                       rolling: bool) -> TruncationReport:
    """Count how many of a task's requests exceeded ``max_length`` tokens.

    lm-eval's HF backend scores ``(context + continuation)[-(max_length + 1):-1]`` -- so a
    request is truncated exactly when ``len(context) + len(continuation) > max_length + 1``,
    and what is dropped is the FRONT of the context. A task with any truncated request did
    not get a fair score and is flagged as such in the report.

    Takes **every** per-sample log belonging to the task, not one: a group task like ``mmlu``
    writes 57 of them, and checking a single subtask's prompts would report the other 56 as
    fair without having looked at them. See :func:`find_sample_paths`.

    Rolling tasks (WikiText) are exempt from the count and get a note instead: their documents
    are split into consecutive ``max_length`` windows with nothing dropped, so "truncation" is
    the wrong word -- the real caveat is that every token sees less context, which is stated
    separately.
    """
    present = [p for p in samples_paths if p.is_file()]
    if not present:
        return TruncationReport(task=task, n_requests=0, n_truncated=0, max_tokens=0,
                                max_length=max_length, rolling=rolling,
                                note=f"no per-sample log found for {task}; truncation could "
                                     f"not be checked (re-run with --log_samples)")
    n_requests = 0
    n_truncated = 0
    max_tokens = 0
    for samples_path in present:
        with samples_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                for context, continuation in iter_request_texts(json.loads(line)):
                    total = len(tokenizer(context)["input_ids"])
                    if continuation:
                        total += len(tokenizer(continuation)["input_ids"])
                    n_requests += 1
                    max_tokens = max(max_tokens, total)
                    if total > max_length + 1:
                        n_truncated += 1
    note = ""
    if rolling:
        note = (f"rolling-loglikelihood task: documents are scored in consecutive "
                f"{max_length}-token windows, so nothing is dropped, but every token sees at "
                f"most {max_length - 1} tokens of context")
        n_truncated = 0
    return TruncationReport(task=task, n_requests=n_requests, n_truncated=n_truncated,
                            max_tokens=max_tokens, max_length=max_length, rolling=rolling,
                            note=note)


def find_sample_paths(results_json: Path, task: str) -> List[Path]:
    """Every per-sample log belonging to ``task`` in the run that produced ``results_json``.

    lm-eval names these ``samples_<task>_<timestamp>.jsonl`` -- but a **group** task writes one
    file per member, not one file: ``mmlu`` is 57 subtasks and produces
    ``samples_mmlu_anatomy_<timestamp>.jsonl``, ``samples_mmlu_astronomy_<timestamp>.jsonl``
    and so on. Returning a single file would check 1/57th of MMLU's prompts against the
    context window and then report the whole task as fair -- precisely the false all-clear
    this analysis exists to prevent, and precisely the task where a 512-token model is most
    likely to be truncated.

    The run's timestamp comes from ``results_json``'s own filename, so a run directory holding
    several runs cannot mix one run's sample logs into another's count.
    """
    stem = results_json.stem
    prefix = "results_"
    stamp = stem[len(prefix):] if stem.startswith(prefix) else ""
    pattern = f"samples_{task}*_{stamp}.jsonl" if stamp else f"samples_{task}*.jsonl"
    return sorted(results_json.parent.glob(pattern))


def load_tokenizer(model_dir: Path):
    """The subject model's own tokenizer, for the truncation count.

    ``transformers`` is already a dependency of this repo (``pyproject.toml``), so this needs
    nothing new installed -- unlike the harness itself, which is why the harness is shelled
    out to instead of imported.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir))


# ---------------------------------------------------------------------------------------
# Turning lm-eval's results JSON into this project's report
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricResult:
    """One task/metric row: the score, the null, the spread, and the verdict joining them."""

    task: str
    task_title: str
    metric: str
    metric_title: str
    score: Optional[float]
    stderr: Optional[float]
    chance: Optional[float]
    higher_is_better: bool
    verdict: str
    z: Optional[float]
    chance_note: str = ""
    reference: Optional[float] = None
    reference_kind: str = ""
    reference_source: str = ""
    reference_caveat: str = ""

    def as_json(self) -> dict:
        return {
            "task": self.task, "task_title": self.task_title,
            "metric": self.metric, "metric_title": self.metric_title,
            "score": self.score,
            "stderr": self.stderr,
            "chance": self.chance,
            "higher_is_better": self.higher_is_better,
            "verdict": self.verdict,
            "standard_errors_from_chance": self.z,
            # `None` whenever the verdict says this number must not be read as a quantity --
            # see reportable_score's docstring.
            "reportable_score": reportable_score(self.score, self.verdict),
            "headline": headline(self.score, self.verdict),
            "chance_note": self.chance_note,
            "gpt2_reference": self.reference,
            "gpt2_reference_kind": self.reference_kind,
            "gpt2_reference_source": self.reference_source,
            "gpt2_reference_caveat": self.reference_caveat,
        }


def _get(results: dict, task: str, key: str) -> Optional[float]:
    """lm-eval stores metrics as ``"<key>,<filter>"``; find the first matching key."""
    task_results = results.get(task)
    if not isinstance(task_results, dict):
        return None
    for name, value in task_results.items():
        if name.split(",")[0] == key and isinstance(value, (int, float)):
            return float(value)
    return None


def _stderr(results: dict, task: str, key: str) -> Optional[float]:
    """The standard error lm-eval reports for ``key``; ``None`` when it wrote ``"N/A"``."""
    task_results = results.get(task)
    if not isinstance(task_results, dict):
        return None
    for name, value in task_results.items():
        if name.split(",")[0] == f"{key}_stderr":
            return float(value) if isinstance(value, (int, float)) else None
    return None


def reference_lookup(task: str, metric: str, reference: Optional[dict]
                     ) -> Tuple[Optional[float], str, str, str]:
    """The GPT-2 figure for one row: measured if we have one, published if we do not.

    Returns ``(value, kind, source, caveat)`` where ``kind`` is ``"measured"`` (a real gpt2 run
    through this same harness, via ``--reference-json``), ``"published"`` (GPT-2 paper, a
    different protocol), or ``""`` (no figure known -- reported as blank rather than as a
    guess).
    """
    if reference is not None:
        for row in reference.get("results", []):
            if row.get("task") == task and row.get("metric") == metric:
                value = row.get("score")
                if isinstance(value, (int, float)):
                    label = reference.get("model", {}).get("label", "reference model")
                    harness = reference.get("harness", {})
                    version = harness.get("lm_eval_version", "?")
                    window = harness.get("max_length")
                    # The reference model's own context window belongs in the source string:
                    # GPT-2 small runs at 1024 and our current model at 512, and a reader
                    # comparing a perplexity across the two has to know that.
                    at = f" at a {window}-token context" if window else ""
                    return (float(value), "measured",
                            f"{label} run through this same harness (lm_eval {version}){at}",
                            "")
    published = GPT2_PUBLISHED.get((task, metric))
    if published is not None:
        return published.value, "published", published.source, published.caveat
    return None, "", "", ""


def build_metric_results(results: dict, tasks: Sequence[TaskSpec],
                         reference: Optional[dict] = None) -> List[MetricResult]:
    """One :class:`MetricResult` per (task, metric) the task list asks for.

    A task lm-eval did not return is skipped silently only in the sense that it produces no
    row; :func:`missing_tasks` is what turns that into a visible statement in the report, so
    "the task crashed" can never look like "the task was not requested".
    """
    rows: List[MetricResult] = []
    for spec in tasks:
        for metric in spec.metrics:
            score = _get(results, spec.task, metric.key)
            if score is None:
                continue
            stderr = _stderr(results, spec.task, metric.key)
            verdict = chance_verdict(score, metric.chance, stderr)
            ref, kind, source, caveat = reference_lookup(spec.task, metric.key, reference)
            rows.append(MetricResult(
                task=spec.task, task_title=spec.title,
                metric=metric.key, metric_title=metric.title,
                score=score, stderr=stderr, chance=metric.chance,
                higher_is_better=metric.higher_is_better,
                verdict=verdict,
                z=z_from_chance(score, metric.chance, stderr),
                chance_note=metric.chance_note,
                reference=ref, reference_kind=kind, reference_source=source,
                reference_caveat=caveat))
    return rows


def missing_tasks(results: dict, tasks: Sequence[TaskSpec]) -> List[str]:
    """Requested tasks with no row in lm-eval's results -- reported, never swallowed."""
    return [spec.task for spec in tasks if not isinstance(results.get(spec.task), dict)]


# ---------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if not math.isfinite(value):
        return "inf"
    return f"{value:.{digits}f}"


def _fmt_int(value: Optional[int]) -> str:
    return "unknown" if value is None else f"{value:,}"


def as_float(value) -> Optional[float]:
    """Coerce a JSON field to ``float``, or ``None``. lm-eval writes some numbers as strings.

    ``total_evaluation_time_seconds`` in particular arrives as a string in lm-eval 0.4.9;
    formatting it as a number without this raises, and coercing it blindly with ``float()``
    would raise on the ``None`` of a reused run. Neither is worth crashing a finished
    benchmark over.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class ReportInputs:
    """Everything :func:`render_markdown` / :func:`report_to_json` need, gathered in one place."""

    model: ModelFacts
    rows: List[MetricResult]
    truncation: List[TruncationReport]
    venv: VenvProvenance
    command: List[str]
    lm_eval_version: str
    n_params: Optional[int]
    max_length: int
    batch_size: int
    dtype: str
    limit: Optional[int]
    device: str
    results_json: Path
    missing: List[str] = field(default_factory=list)
    reference_label: str = ""
    total_seconds: Optional[float] = None
    wikitext_identity: str = ""
    is_reference_run: bool = False
    """True when the subject IS an external reference model (e.g. gpt2) rather than one of
    ours. Only changes wording: a report that compared GPT-2 to GPT-2 in a column headed
    'the gap to its reference class' would be nonsense."""


def _verdict_cell(row: MetricResult) -> str:
    if row.verdict == AT_CHANCE:
        return f"**{AT_CHANCE}**"
    return row.verdict


def render_markdown(inputs: ReportInputs) -> str:
    """The report. Structured so a reader meets the caveats before meeting a number."""
    m = inputs.model
    lines: List[str] = []
    lines.append("<!-- SPDX-License-Identifier: Apache-2.0 -->")
    lines.append("<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->")
    lines.append("")
    lines.append(f"# External benchmarks — `{m.label}`")
    lines.append("")
    lines.append(
        f"Model `{m.path}`, evaluated on CPU by EleutherAI's `lm-evaluation-harness` "
        f"{inputs.lm_eval_version} at a {inputs.max_length}-token context, "
        f"dtype {inputs.dtype}, batch size {inputs.batch_size}"
        + (f", **--limit {inputs.limit}**" if inputs.limit else "")
        + ". Generated by `scripts/benchmark_external.py`."
    )
    lines.append("")

    # -- Why this file is different from everything else in docs/measurements/ ------------
    lines.append("## Why this exists")
    lines.append("")
    if inputs.is_reference_run:
        lines.append(
            "**This is a reference run, not a measurement of one of our models.** It exists so "
            "that the GPT-2-small column in this project's own external-benchmark reports is a "
            "*measured* head-to-head — same harness version, same task definitions, same "
            "machine, same day — instead of a figure quoted from a paper that used different "
            "detokenizers and different scoring code. Feed this file to "
            "`scripts/benchmark_external.py --reference-json`."
        )
        lines.append("")
        lines.append(
            "The chance-baseline labelling rule is applied here unchanged. GPT-2 small is also "
            "at chance on several of these tasks, and that is exactly the context a reader "
            "needs: it is not only *our* model that cannot do them."
        )
    else:
        lines.append(
            "Every other measurement in this repository is self-referential: validation loss "
            "on a tail of our own blend, per-source loss on that same blend, behaviour scores "
            "on two prompt sets we wrote ourselves, and a noise floor derived from our own "
            "seed-only control. A model that had learned to imitate our corpus and nothing "
            "else would score exactly as well on all of it. **These numbers are the first this "
            "project has that our corpus did not produce** — someone else's benchmarks, "
            "someone else's data, someone else's scoring code."
        )
        lines.append("")
        lines.append(
            "**Chance-level scores were the prediction, not a disappointment.** Read the next "
            "section before reading any number below."
        )
        lines.append("")

    # -- The data gap ---------------------------------------------------------------------
    if inputs.is_reference_run:
        lines.append("## The model")
        lines.append("")
        lines.append("| parameters | context window | layers x hidden |")
        lines.append("|---:|---:|---:|")
        lines.append(f"| {_fmt_int(inputs.n_params)} | {m.max_position_embeddings} | "
                     f"{m.num_hidden_layers} x {m.hidden_size} |")
        lines.append("")
        lines.append(
            "GPT-2 small was trained on WebText, ~40 GB of text — on the order of 40 billion "
            "tokens, roughly 100x what this project's own models have seen. That ratio is the "
            "single most important number for reading the comparison reports this file feeds."
        )
        lines.append("")
    else:
        lines.append("## The model, and the gap to its reference class")
        lines.append("")
        lines.append("| | this model | GPT-2 small |")
        lines.append("|---|---:|---:|")
        lines.append(f"| parameters | {_fmt_int(inputs.n_params)} | "
                     f"{GPT2_SMALL_PARAMS:,} |")
        lines.append(f"| training tokens | {_fmt_int(m.training_tokens)} | "
                     f"~{GPT2_SMALL_TRAINING_TOKENS:,} (WebText, ~40 GB) |")
        lines.append(f"| context window | {m.max_position_embeddings} | 1,024 |")
        lines.append(f"| layers x hidden | {m.num_hidden_layers} x {m.hidden_size} | 12 x 768 |")
        lines.append("")
        if m.training_tokens is not None:
            lines.append(f"Training tokens counted as {m.training_tokens_note}.")
        else:
            lines.append(f"Training tokens could not be established: {m.training_tokens_note}.")
        lines.append("")
        # Both sentences below are derived from the numbers in the table rather than asserted,
        # because this same renderer produces reports for models of very different sizes: the
        # 123M current model IS GPT-2 small's parameter twin, and the 22M v3 is emphatically
        # not, and a fixed "within 1.2% of each other" would be false in the second report.
        gap_sentences: List[str] = []
        if inputs.n_params:
            ratio = GPT2_SMALL_PARAMS / inputs.n_params
            if 0.95 <= ratio <= 1.05:
                gap_sentences.append(
                    f"The parameter counts are within {abs(1 - ratio) * 100:.1f}% of each "
                    f"other, so this is a like-for-like comparison on capacity.")
            else:
                gap_sentences.append(
                    f"GPT-2 small has **{ratio:.1f}x** this model's parameters, so capacity is "
                    f"part of the gap here as well as data.")
        if m.training_tokens:
            token_ratio = GPT2_SMALL_TRAINING_TOKENS / m.training_tokens
            gap_sentences.append(
                f"It saw roughly **{token_ratio:.0f}x** as many training tokens.")
        else:
            gap_sentences.append(
                "This model's training-token count could not be established from disk (see "
                "above), so the data gap can only be stated for GPT-2's side of it.")
        gap_sentences.append(
            "Every score below should be read against those ratios. A low score here is the "
            "expected consequence of the gap and is not evidence that anything is broken.")
        lines.append(" ".join(gap_sentences))
        lines.append("")

    # -- How to read the table ------------------------------------------------------------
    lines.append("## How to read this table")
    lines.append("")
    lines.append(
        f"- **chance** is what a model that had learned nothing would score. For 4-way "
        f"multiple choice that is 0.25; for 2-way, 0.50; for open-vocabulary last-word "
        f"prediction it is ~1/32000, i.e. zero. Perplexities have no chance baseline — there "
        f"is no \"guess at random\" strategy for them — and are reported directly."
    )
    lines.append(
        f"- **verdict** applies one rule: a score within **{CHANCE_SE_MULTIPLE:g} standard "
        f"errors** of chance is `{AT_CHANCE}`, and `{AT_CHANCE}` means the number in the "
        f"score column **is not a measurement of this model**. It is printed so you can see "
        f"what was suppressed and check the arithmetic, not so it can be quoted. In the "
        f"companion JSON such a row has `reportable_score: null`."
    )
    lines.append(
        "- This mirrors `scripts/evaluate.py`, which labels any delta inside ~1.2x of the "
        "seed-noise floor `NOT INTERPRETABLE` no matter what its confidence interval says. "
        "Same principle, different null: there, another run of the same recipe; here, a model "
        "that learned nothing."
    )
    lines.append(
        "- **stderr** is lm-eval's own standard error over benchmark items. It covers "
        "sampling of the benchmark, and nothing else — not training-seed variance, not "
        "prompt-format sensitivity. It is a lower bound on the real uncertainty."
    )
    # The gate is applied once per row, so the table as a whole has more chances to produce a
    # spurious "clears chance" than any single row does. Saying so is the same instinct as
    # evaluate.py's floor: a threshold quoted without its false-positive rate invites exactly
    # the over-reading this project has already published once.
    tested = [r for r in inputs.rows if r.chance is not None and r.z is not None]
    if tested:
        expected_false = len(tested) * (1.0 - 0.9545)
        lines.append(
            f"- **The gate is applied per row, and this table has {len(tested)} of them.** "
            f"If every one of those scores were truly at chance, roughly "
            f"**{expected_false:.1f}** would still clear a {CHANCE_SE_MULTIPLE:g}-standard-"
            f"error gate by luck alone. A row sitting just past the threshold is therefore "
            f"weaker evidence than its verdict makes it look; a row at 5+ standard errors, or "
            f"a pattern of several rows moving the same way, is not."
        )
    lines.append("")

    # -- The results ----------------------------------------------------------------------
    lines.append("## Results")
    lines.append("")
    lines.append("| task | metric | score | stderr | chance | s.e. from chance | verdict | "
                 "GPT-2 small |")
    lines.append("|---|---|---:|---:|---:|---:|---|---:|")
    for row in inputs.rows:
        ref = "—"
        if row.reference is not None:
            marker = "measured" if row.reference_kind == "measured" else "published"
            ref = f"{_fmt(row.reference)} ({marker})"
        lines.append(
            f"| {row.task} | {row.metric_title} | {_fmt(row.score)} | {_fmt(row.stderr)} | "
            f"{'—' if row.chance is None else _fmt(row.chance, 2)} | "
            f"{'—' if row.z is None else f'{row.z:+.1f}'} | {_verdict_cell(row)} | {ref} |"
        )
    lines.append("")
    if inputs.missing:
        lines.append(
            f"**Requested but absent from the harness output:** {', '.join(inputs.missing)}. "
            f"A missing task is a failed task — do not read its absence as a null result."
        )
        lines.append("")

    # -- Headline -------------------------------------------------------------------------
    lines.append("## What this says, in words")
    lines.append("")
    at_chance = [r for r in inputs.rows if r.verdict == AT_CHANCE]
    moved = [r for r in inputs.rows if r.verdict in (ABOVE_CHANCE, BELOW_CHANCE)]
    continuous = [r for r in inputs.rows if r.verdict == NO_CHANCE_BASELINE]
    if at_chance:
        lines.append(
            "**At chance** (score not reportable as a quantity): "
            + ", ".join(f"`{r.task}` {r.metric_title}" for r in at_chance) + "."
        )
        lines.append("")
    if moved:
        lines.append("**Distinguishable from chance:**")
        lines.append("")
        for r in moved:
            direction = "above" if r.verdict == ABOVE_CHANCE else "below"
            se = "n/a" if r.z is None else f"{abs(r.z):.1f}"
            lines.append(
                f"- `{r.task}` {r.metric_title}: {_fmt(r.score)} vs chance "
                f"{_fmt(r.chance, 2)} — {se} standard errors {direction}."
            )
            # Clearing chance is not the same as being good. Where chance is a degenerate
            # baseline (LAMBADA's is ~zero, so ANY correct answer clears it by many standard
            # errors), the note says so right next to the claim rather than in a footnote.
            if r.chance_note:
                lines.append(f"  - *{r.chance_note}*")
            if r.reference is not None:
                lines.append(f"  - *GPT-2 small on the same metric: {_fmt(r.reference)} "
                             f"({r.reference_kind}).*")
        lines.append("")
    if continuous:
        lines.append("**Graded (no chance floor to sit at):**")
        lines.append("")
        for r in continuous:
            lines.append(f"- `{r.task}` {r.metric_title}: {_fmt(r.score)}"
                         + (f" (GPT-2 small: {_fmt(r.reference)}, {r.reference_kind})"
                            if r.reference is not None else ""))
        lines.append("")
    # A row the harness gave no spread for belongs here too. Leaving it out of the prose
    # would let a reader who skipped the table assume every task was accounted for.
    no_spread = [r for r in inputs.rows if r.verdict == NO_STDERR]
    if no_spread:
        lines.append(
            "**Not interpretable** (the harness reported no standard error, so there is no "
            "way to tell these apart from chance either way): "
            + ", ".join(f"`{r.task}` {r.metric_title}" for r in no_spread) + "."
        )
        lines.append("")

    # -- Context window -------------------------------------------------------------------
    lines.append("## Did the context window truncate anything?")
    lines.append("")
    lines.append(
        f"This model's window is **{m.max_position_embeddings} tokens**. When a prompt plus "
        f"its continuation exceeds it, lm-eval drops tokens off the **front of the context** "
        f"and the model answers a question it was only shown part of. Every request the "
        f"harness actually issued was re-tokenized with this model's own tokenizer and "
        f"checked against the limit."
    )
    lines.append("")
    lines.append("| task | requests | over the window | longest request (tokens) | fair score? |")
    lines.append("|---|---:|---:|---:|---|")
    for t in inputs.truncation:
        if t.rolling:
            fair = "windowed, not truncated — see note"
        elif t.n_requests == 0:
            fair = "unchecked"
        elif t.truncated:
            fair = "**NO — truncated, do not read as a fair score**"
        else:
            fair = "yes"
        lines.append(f"| {t.task} | {t.n_requests:,} | {t.n_truncated:,} | {t.max_tokens:,} | "
                     f"{fair} |")
    lines.append("")
    for t in inputs.truncation:
        if t.note:
            lines.append(f"- *{t.task}: {t.note}*")
    lines.append("")

    # -- Caveats --------------------------------------------------------------------------
    lines.append("## Caveats on the GPT-2 column")
    lines.append("")
    # Only warn about the kinds of row this report actually contains. A caveat about
    # published-under-another-protocol figures, printed over a table where every cell is a
    # measured head-to-head, trains a reader to skip caveats.
    kinds = {row.reference_kind for row in inputs.rows if row.reference is not None}
    if "published" in kinds:
        lines.append(f"- {PROTOCOL_CAVEATS[0]}")
    if "measured" in kinds:
        lines.append(f"- {PROTOCOL_CAVEATS[1]}")
    if not kinds:
        lines.append("- No GPT-2 figure is available for any row in this table.")
    for row in inputs.rows:
        if row.reference is not None and row.reference_caveat:
            lines.append(f"- `{row.task}` {row.metric_title}: {row.reference_caveat}")
    # Where a cell has BOTH a measured value and a published one, the two together say how far
    # apart the protocols really are -- which is the only honest way to calibrate how much to
    # trust a published figure elsewhere. Computed, not asserted.
    for row in inputs.rows:
        if row.reference is None or row.reference_kind != "measured":
            continue
        published = GPT2_PUBLISHED.get((row.task, row.metric))
        if published is None or published.value == 0:
            continue
        drift = abs(row.reference - published.value) / abs(published.value)
        verdict = ("the two protocols agree closely here"
                   if drift < 0.05 else
                   "the two protocols do NOT agree here, so published figures for other rows "
                   "should be treated with corresponding suspicion")
        lines.append(
            f"- **Protocol cross-check, `{row.task}` {row.metric_title}:** measured "
            f"{_fmt(row.reference)} against the published {_fmt(published.value)} "
            f"({drift * 100:.1f}% apart) — {verdict}."
        )
    lines.append("")
    if inputs.wikitext_identity:
        lines.append(f"- {inputs.wikitext_identity}")
        lines.append("")

    # -- Reproducibility ------------------------------------------------------------------
    lines.append("## Reproducing this")
    lines.append("")
    lines.append(
        "`lm-eval` is **not** a dependency of this repository and must not become one — "
        "`pyproject.toml` lists three runtime dependencies and this project has repeatedly "
        "declined to add a fourth. The harness lives in a throwaway virtualenv under "
        "`scratch/` (gitignored) and `scripts/benchmark_external.py` shells out to it."
    )
    lines.append("")
    lines.append(f"- venv: `{inputs.venv.python}`")
    for name, version in inputs.venv.versions.items():
        lines.append(f"- `{name}`: {version}")
    lines.append(f"- device: `{inputs.device}` (CPU-only by design; this script refuses any "
                 f"other device and never opens a Tenstorrent device)")
    if inputs.total_seconds is not None:
        lines.append(f"- harness wall time: {inputs.total_seconds:,.0f} s")
    lines.append(f"- raw harness output: `{inputs.results_json}`")
    lines.append("")
    lines.append("```")
    lines.append(" ".join(inputs.command))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def report_to_json(inputs: ReportInputs) -> dict:
    """The machine-readable twin of the markdown, with the same suppression rule applied."""
    m = inputs.model
    return {
        "schema": "tt-tnt/external-benchmark/1",
        "model": {
            "label": m.label,
            "path": str(m.path),
            "n_params": inputs.n_params,
            "max_position_embeddings": m.max_position_embeddings,
            "hidden_size": m.hidden_size,
            "num_hidden_layers": m.num_hidden_layers,
            "training_tokens": m.training_tokens,
            "training_tokens_note": m.training_tokens_note,
        },
        "harness": {
            "name": "EleutherAI/lm-evaluation-harness",
            "lm_eval_version": inputs.lm_eval_version,
            "venv": inputs.venv.as_json(),
            "command": list(inputs.command),
            "device": inputs.device,
            "dtype": inputs.dtype,
            "batch_size": inputs.batch_size,
            "max_length": inputs.max_length,
            "limit": inputs.limit,
            "results_json": str(inputs.results_json),
            "total_evaluation_time_seconds": inputs.total_seconds,
        },
        "chance_rule": {
            "se_multiple": CHANCE_SE_MULTIPLE,
            "explanation":
                f"A score within {CHANCE_SE_MULTIPLE:g} standard errors of its chance "
                f"baseline is labelled {AT_CHANCE} and has reportable_score: null. The raw "
                f"score is kept in `score` so the suppression is auditable.",
        },
        "results": [row.as_json() for row in inputs.rows],
        "missing_tasks": list(inputs.missing),
        "truncation": [t.as_json() for t in inputs.truncation],
        "reference_model": inputs.reference_label,
        "is_reference_run": inputs.is_reference_run,
        "protocol_caveats": list(PROTOCOL_CAVEATS),
        "wikitext_identity": inputs.wikitext_identity,
    }


# ---------------------------------------------------------------------------------------
# The WikiText-2 / WikiText-103 test-split identity claim
# ---------------------------------------------------------------------------------------

#: What :func:`verify_wikitext_test_identity` checks, stated so the claim in the report is
#: falsifiable: both configs' test splits are these 62 documents and this many characters.
WIKITEXT_TEST_DOCS = 62
WIKITEXT_TEST_CHARS = 1_288_493


def verify_wikitext_test_identity(python: Path, *, runner=subprocess.run) -> str:
    """Check inside the venv that wikitext-2's and wikitext-103's test splits are identical.

    The report claims lm-eval's ``wikitext`` task (which uses ``wikitext-2-raw-v1``) yields
    *WikiText-103* test perplexity. That claim is load-bearing -- it is what makes the number
    comparable to the published literature -- so it is checked at run time rather than
    asserted from memory. Returns a human-readable sentence for the report; on any failure it
    returns a sentence saying the check could not be made, and never a false confirmation.
    """
    program = (
        "import json\n"
        "from datasets import load_dataset\n"
        "a = ''.join(load_dataset('EleutherAI/wikitext_document_level',"
        " 'wikitext-2-raw-v1', split='test')['page'])\n"
        "b = ''.join(load_dataset('EleutherAI/wikitext_document_level',"
        " 'wikitext-103-raw-v1', split='test')['page'])\n"
        "print(json.dumps({'identical': a == b, 'chars': len(a)}))\n"
    )
    result = runner([str(python), "-c", program], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return ("The wikitext-2 / wikitext-103 test-split identity check could not be run "
                f"({result.stderr.strip().splitlines()[-1] if result.stderr.strip() else 'no output'}), "
                "so the claim that this is WikiText-103 perplexity is UNVERIFIED here.")
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return ("The wikitext-2 / wikitext-103 test-split identity check produced no parseable "
                "answer, so the claim that this is WikiText-103 perplexity is UNVERIFIED here.")
    if payload.get("identical"):
        return (f"Verified at run time: the `wikitext-2-raw-v1` and `wikitext-103-raw-v1` test "
                f"splits are byte-identical ({payload.get('chars', 0):,} characters), so "
                f"lm-eval's `wikitext` task reports WikiText-103 test perplexity.")
    return ("Run-time check says the `wikitext-2-raw-v1` and `wikitext-103-raw-v1` test splits "
            "are NOT identical — do not read the wikitext row as WikiText-103 perplexity.")


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=None,
                   help="Converted HF model directory to benchmark. Defaults to "
                        "docs/current_model.json's designated current model.")
    p.add_argument("--venv", type=Path, default=DEFAULT_VENV,
                   help="Virtualenv with lm-eval installed (default: %(default)s). This is "
                        "deliberately NOT this project's environment -- see the module "
                        "docstring.")
    p.add_argument("--tasks", type=str, default=",".join(t.task for t in TASKS),
                   help="Comma-separated lm-eval task names (default: the fixed list).")
    p.add_argument("--max-length", type=int, default=None,
                   help="Context window to evaluate at. Defaults to the model's own "
                        "max_position_embeddings; passing more would let RoPE silently "
                        "zero-fill past its cache.")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--dtype", type=str, default=DEFAULT_DTYPE)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap documents per task. TESTING ONLY -- it inflates every standard "
                        "error and is recorded prominently in the report when used.")
    p.add_argument("--num-fewshot", type=int, default=None,
                   help="Passed through to lm-eval. Left unset (each task's own default, "
                        "0-shot for everything in the fixed list) so prompts stay inside a "
                        "512-token window.")
    p.add_argument("--device", type=str, default="cpu",
                   help="CPU only. Any other value is refused (default: %(default)s).")
    p.add_argument("--label", type=str, default=None,
                   help="Report label (default: derived from --model's directory name).")
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Where lm-eval writes its raw output (default: "
                        f"{DEFAULT_RUN_ROOT}/<label>).")
    p.add_argument("--reference-json", type=Path, default=None,
                   help="A previous external-*.json from this script (e.g. a real gpt2 run) "
                        "to use as the GPT-2 column. Turns the comparison from 'published "
                        "under another protocol' into a measured head-to-head.")
    p.add_argument("--reference-run", action="store_true",
                   help="Mark this as a run of an EXTERNAL reference model (e.g. gpt2) rather "
                        "than one of ours. Wording only -- the chance-baseline rule and every "
                        "number are computed identically.")
    p.add_argument("--reuse-run", action="store_true",
                   help="Re-render the report from an existing --run-dir without re-running "
                        "the harness.")
    p.add_argument("--skip-truncation-check", action="store_true",
                   help="Skip re-tokenizing every request to count truncations. Only useful "
                        "when the per-sample logs are unavailable; the report then says the "
                        "check was not made rather than implying it passed.")
    p.add_argument("--out", type=Path, default=None,
                   help="Markdown output path (default: docs/measurements/external-<label>.md)")
    p.add_argument("--json-out", type=Path, default=None,
                   help="JSON output path (default: docs/measurements/external-<label>.json)")
    return p.parse_args(argv)


def _default_output_paths(label: str) -> Tuple[Path, Path]:
    out_dir = ROOT / "docs" / "measurements"
    return out_dir / f"external-{label}.md", out_dir / f"external-{label}.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    try:
        device = require_cpu_device(args.device)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    model_dir = args.model or current_model_path()
    if model_dir is None:
        print("ERROR: no --model given and docs/current_model.json does not name one.",
              file=sys.stderr)
        return 1
    model_dir = Path(model_dir)
    if not model_dir.is_absolute():
        model_dir = (ROOT / model_dir).resolve()

    try:
        facts = read_model_facts(model_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    label = args.label or facts.label
    max_length = args.max_length or facts.max_position_embeddings
    if max_length > facts.max_position_embeddings:
        print(f"ERROR: --max-length {max_length} exceeds {model_dir}'s "
              f"max_position_embeddings ({facts.max_position_embeddings}). RoPE does not "
              f"bounds-check the input length against anything but its own cos/sin cache, so "
              f"this would not raise -- it would silently zero-fill and produce a confidently "
              f"wrong score. Refusing.", file=sys.stderr)
        return 1

    try:
        python = venv_python(args.venv)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in task_names if t not in TASKS_BY_NAME]
    if unknown:
        print(f"ERROR: no chance baseline is defined for {', '.join(unknown)}. Add a TaskSpec "
              f"to scripts/benchmark_external.py rather than reporting a score whose null is "
              f"unknown.", file=sys.stderr)
        return 1
    specs = [TASKS_BY_NAME[t] for t in task_names]

    run_dir = args.run_dir or (DEFAULT_RUN_ROOT / label)
    command = lm_eval_command(python, model_dir, task_names, max_length=max_length,
                              batch_size=args.batch_size, dtype=args.dtype,
                              output_path=run_dir, device=device, limit=args.limit,
                              num_fewshot=args.num_fewshot)

    print(f"model      {model_dir}")
    print(f"label      {label}")
    print(f"venv       {python}")
    print(f"max_length {max_length} (model max_position_embeddings "
          f"{facts.max_position_embeddings})")
    print(f"tasks      {', '.join(task_names)}")

    try:
        provenance = venv_provenance(python)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"versions   {provenance.versions}")

    if args.reuse_run:
        try:
            results_json = find_results_json(run_dir)
        except FileNotFoundError as exc:
            print(f"ERROR: --reuse-run given but {exc}", file=sys.stderr)
            return 1
        print(f"reusing    {results_json}")
    else:
        print(f"running    {' '.join(command)}")
        try:
            results_json = run_lm_eval(command, run_dir)
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    raw = json.loads(results_json.read_text(encoding="utf-8"))
    results = raw.get("results", {})
    n_params = raw.get("config", {}).get("model_num_parameters")
    lm_eval_version = raw.get("lm_eval_version", provenance.versions.get("lm_eval", "unknown"))

    reference = None
    reference_label = ""
    if args.reference_json is not None:
        if not args.reference_json.is_file():
            print(f"ERROR: --reference-json {args.reference_json} not found.", file=sys.stderr)
            return 1
        reference = json.loads(args.reference_json.read_text(encoding="utf-8"))
        reference_label = reference.get("model", {}).get("label", str(args.reference_json))
        print(f"reference  {reference_label} ({args.reference_json})")

    rows = build_metric_results(results, specs, reference)
    missing = missing_tasks(results, specs)
    if missing:
        print(f"WARNING: no results for {', '.join(missing)} -- reported as missing, not null.")

    truncation: List[TruncationReport] = []
    if args.skip_truncation_check:
        truncation = [TruncationReport(task=s.task, n_requests=0, n_truncated=0, max_tokens=0,
                                       max_length=max_length, rolling=s.rolling,
                                       note="truncation check skipped (--skip-truncation-check)")
                      for s in specs]
    else:
        print("checking every issued request against the context window ...")
        tokenizer = load_tokenizer(model_dir)
        for spec in specs:
            samples = find_sample_paths(results_json, spec.task)
            report = analyse_truncation(samples, tokenizer, task=spec.task,
                                        max_length=max_length, rolling=spec.rolling)
            truncation.append(report)
            flag = "TRUNCATED" if report.truncated else "ok"
            print(f"    {spec.task:16} {len(samples):>3} log(s), "
                  f"{report.n_requests:>7,} requests, max {report.max_tokens:>6,} tokens  "
                  f"{flag}")

    wikitext_identity = ""
    if any(s.task == "wikitext" for s in specs):
        wikitext_identity = verify_wikitext_test_identity(python)
        print(f"wikitext   {wikitext_identity}")

    inputs = ReportInputs(
        model=facts, rows=rows, truncation=truncation, venv=provenance, command=command,
        lm_eval_version=lm_eval_version, n_params=n_params, max_length=max_length,
        batch_size=args.batch_size, dtype=args.dtype, limit=args.limit, device=device,
        results_json=results_json, missing=missing, reference_label=reference_label,
        total_seconds=as_float(raw.get("total_evaluation_time_seconds")),
        wikitext_identity=wikitext_identity,
        is_reference_run=args.reference_run,
    )

    default_md, default_json = _default_output_paths(label)
    out = args.out or default_md
    json_out = args.json_out or default_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(inputs), encoding="utf-8")
    print(f"wrote {out}")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report_to_json(inputs), indent=2), encoding="utf-8")
    print(f"wrote {json_out}")

    for row in rows:
        print(f"    {row.task:16} {row.metric:18} {headline(row.score, row.verdict):>12}  "
              f"[{row.verdict}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
