#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Train tt-tnt on Tenstorrent hardware.

We own this entrypoint because tt-train's own Python trainer does not work against the
current tree: ``examples/python/transformers/training.py`` imports a ``trainer`` module
that is not on its path, calls ``train()`` with an extra ``val_ids`` argument the signature
does not accept, and relies on a ``TrainingConfig`` that lacks the ``seq_len`` ``train()``
requires. Its data loader also hardcodes ``$TT_METAL_HOME/tt-train/data/shakespeare.txt``.

What we reuse from ttml (never reimplemented): both of its Llama implementations,
``create_optimizer``, ``initialize_device``, ``set_seed``, and the ``train()`` loop itself.
What we supply: our corpus, our tokenizer, ``seq_len``, and a **real** validation loss —
ttml's ``train()`` fills ``val_losses`` with a copy of the training loss under a comment
calling it placeholder behavior, so a val number from it means nothing.

*Which* Llama is chosen by ``--model-impl``. The default (``python``) is ttml's Python
``Llama`` wrapped by ``train.model``, which drops the redundant causal mask and so runs SDPA
on its causal path — 1.41x faster per step at ``--size 384``, 1.15x at ``--size 1024``.
``cpp`` is ttml's ``TransformerModelFactory``/``CppLlama``, what this entrypoint used before
2026-08-16, kept as an A/B lever. Checkpoints from the two are interchangeable; see
``train/model.py`` for why the two exist and what the wrapper has to reconcile.

ttml's ``train()`` has no checkpoint hook of its own, so periodic checkpointing is done by
calling it repeatedly in chunks of ``--save-every`` steps and saving between chunks (see
``train.checkpoint``); the optimizer object persists across those calls, so AdamW's moments
carry over rather than resetting each chunk. ``--resume latest`` (or a specific checkpoint
path) restores model and optimizer state before training resumes — note that ``--steps`` in
a ``--resume`` run counts steps to run *from* the checkpoint, not an absolute target step;
see ``--resume``'s help below. ``--checkpoint-dir`` selects where checkpoints are read from
and written to.

The same chunk loop also drives periodic validation (``--val-every``): since we already stop
between chunks to checkpoint, evaluating there too is nearly free and produces a loss curve
instead of one number at the end of a long run — see ``run_training_loop`` and ``evaluate()``.
It is independent of ``--save-every``: the two boundaries need not coincide.

    python train/run.py --steps 20
    python train/run.py --steps 100 --save-every 25
    python train/run.py --steps 200 --val-every 100
    python train/run.py --steps 50 --resume latest
    python train/run.py --steps 20                # seq_len defaults to --size's max_sequence_length
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train import checkpoint  # noqa: E402
from train import model as tt_tnt_model  # noqa: E402
from train.config import (  # noqa: E402
    DEFAULT_SEED,
    SEQ_LEN,
    VOCAB_SIZE,
    apply_optimizer_override,
    build_yaml_config,
    run_config_from_yaml,
)
from train.paths import ProtectedPathError, assert_not_protected, write_dir  # noqa: E402
from train.sizes import DEFAULT_SIZE, SIZES, get_size  # noqa: E402


def _default_tt_metal_home() -> str:
    return os.environ.get("TT_METAL_HOME", os.path.expanduser("~/tt-metal"))


def _prepare_env(tt_metal_home: str, arch: str) -> None:
    """ttml needs all three of these before import; it aborts without RUNTIME_ROOT."""
    os.environ.setdefault("TT_METAL_HOME", tt_metal_home)
    os.environ.setdefault("TT_METAL_RUNTIME_ROOT", tt_metal_home)
    os.environ.setdefault("TT_METAL_ARCH_NAME", arch)
    os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
    sys.path.append(f"{tt_metal_home}/tt-train/sources/ttml")


#: Mesh graph descriptors this repo ships, keyed by the device count they describe. See
#: :func:`_mesh_graph_descriptor_path` for why they are vendored rather than referenced.
MESH_DESCRIPTORS = {2: "mesh-1x2.textproto", 4: "mesh-1x4.textproto"}


def _mesh_graph_descriptor_path(devices: int) -> Optional[Path]:
    """The mesh-graph descriptor a ``devices``-chip DDP run needs, or ``None`` if unneeded.

    TWO SEPARATE THINGS GO WRONG WITHOUT THIS, and only the first was anticipated.

    **1. There is no descriptor at all for 4 devices.**
    ``ttml.core.distributed.enable_fabric`` ships defaults for **8 and 32 devices only**
    (``ttnn_fixed/distributed/tt_metal.cpp:80-88``); any other count returns ``std::nullopt``
    and falls back to a bare ``FABRIC_2D``. Measured on this box, that hangs before the mesh
    even opens — killed at 600s with no output at all.

    **2. The descriptor's declared shape must MATCH THE LOGICAL MESH, not the cabling.**
    This is the part that cost the most to find, because it fails silently. tt-metal's
    ``p300_x2_mesh_graph_descriptor.textproto`` declares ``device_topology { dims: [2, 2] }``,
    which correctly describes a TT-QuietBox 2 (two dual-chip p300 cards, four Blackhole chips
    in a 2x2 ring). Pointing at it makes the ``[1, 4]`` mesh **open successfully** and the
    model train at full speed — and then the run stops dead the first time a gradient
    all-reduce crosses the mesh axis. It is not an error, a warning, or a diagnosable
    mismatch; it is a hang. Supplying a descriptor that declares ``[1, 4]`` instead — the
    logical shape ``--ddp 4`` opens — makes the identical run work, with every replica's
    parameters bit-identical afterwards.

    So the descriptors are vendored under ``train/configs/mesh/`` rather than selected from
    tt-metal's directory: the file this project needs is chosen by the mesh *it* opens, and
    no shipped descriptor names that shape. See ``.superpowers/ddp-bringup.md`` for the full
    experiment, including the no-sync control that proves the difference is real.

    An operator who has already exported ``TT_MESH_GRAPH_DESC_PATH`` keeps their value: this
    returns ``None`` in that case so the caller leaves the environment alone. ``devices <= 1``
    also returns ``None`` — a single-chip run arms no fabric and needs no descriptor, which
    is why every measurement recorded before 2026-08-16 never met this.

    Raises:
        ValueError: if ``devices`` has no vendored descriptor. Better than falling back to a
            shape that hangs.
    """
    if devices <= 1 or os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        return None
    if devices not in MESH_DESCRIPTORS:
        raise ValueError(
            f"no mesh graph descriptor for {devices} devices; this repo ships "
            f"{sorted(MESH_DESCRIPTORS)} (train/configs/mesh/). A descriptor whose declared "
            f"dims do not match the [1, {devices}] mesh opened will HANG in the first "
            f"gradient all-reduce rather than fail, so falling back to one is not safe."
        )
    return ROOT / "train" / "configs" / "mesh" / MESH_DESCRIPTORS[devices]


def _init_parallelism_context(ddp: int, print_fn: Callable[..., None] = print) -> None:
    """Initialise ttml's parallelism context for a ``[1, ddp]`` DDP mesh, and *prove* it took.

    THIS FUNCTION IS THE CORRECTNESS GUARD, not a setup step. ttml's gradient
    synchronization begins (``core/distributed/distributed.cpp:56-59``)::

        void synchronize_gradients(const serialization::NamedParameters& parameters) {
            if (!autograd::ctx().is_parallelism_context_initialized()) {
                return;                                  // <-- silent early return

    So a run that opens a ``[1, 4]`` mesh and passes ``use_ddp=True`` to ``train()`` — which
    is otherwise a complete-looking DDP setup — but never reaches this function gets **four
    replicas that diverge from step one**: each chip trains on its own quarter of the batch,
    no gradient is ever reduced, nothing raises, the loss curve looks entirely plausible, and
    the checkpoint keeps replica 0. It would also be roughly 4x faster, which is precisely
    what makes it dangerous.

    Nothing upstream will tell us if this goes wrong, so the post-conditions are checked here
    and a mismatch is raised rather than logged. ``ParallelismContext``'s constructor assigns
    the DDP axis by *inspecting the open mesh*
    (``autograd/auto_context.cpp:157-218``) rather than taking it as an argument, so
    ``get_ddp_size()`` is a genuine read-back of what the hardware mesh turned out to be, not
    an echo of the ``ddp`` we passed in. Asserting it equals ``ddp`` therefore catches an
    opened mesh that is not the mesh we asked for, as well as a context that failed to
    register DDP at all.

    Called only when ``ddp > 1``: on a unit mesh there is no axis to reduce over, and
    ``ParallelismContext`` would reject the configuration anyway (a ``[1, 1]`` mesh is a line
    topology whose only axis has size 1, so no parallelism could be assigned to it).
    """
    import ttml

    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.initialize_parallelism_context(
        ttml.autograd.DistributedConfig(enable_ddp=True, enable_tp=False)
    )

    if not ctx.is_parallelism_context_initialized():
        raise RuntimeError(
            "initialize_parallelism_context returned without initialising the context; "
            "synchronize_gradients would silently early-return and the replicas would "
            "diverge from step 1"
        )
    pctx = ctx.get_parallelism_context()
    if not pctx.is_ddp_enabled():
        raise RuntimeError(
            "parallelism context is initialised but DDP is not enabled on it; gradients "
            "would not be reduced"
        )
    if pctx.get_ddp_size() != ddp:
        raise RuntimeError(
            f"parallelism context reports a DDP axis of {pctx.get_ddp_size()} devices, but "
            f"--ddp asked for {ddp}; the mesh that opened is not the mesh requested"
        )
    print_fn(
        f"  parallelism context: DDP over {pctx.get_ddp_size()} devices on mesh axis "
        f"{pctx.get_ddp_axis()} (gradients will be all-reduced and divided by "
        f"{pctx.get_ddp_size()})"
    )


def evaluate(model, val_ids: np.ndarray, cfg, batches: int = 10, use_ddp: bool = False) -> float:
    """Real validation loss over ``batches`` sampled windows.

    ttml's train() does not compute this — it appends the last training loss and labels it
    val_loss. We run the model in eval mode over held-out tokens and average properly.

    ``model.eval()`` only toggles dropout (0.0 in this config) — it does not disable
    gradient tracking. Without ``no_grad()``, every forward pass here would still build a
    full autograd graph that gets thrown away, wasting memory and compute and OOMing first
    at larger ``validation_batch_size``. ``no_grad`` also lives in ``ttml.common.utils``,
    alongside ``build_causal_mask``, so one import covers both.

    UNDER DDP (``use_ddp=True``) two things change, and both are required rather than
    optional:

    1. The validation batch is sharded across the mesh, exactly as the training batch is —
       ``get_batch_ttml``'s own ``use_ddp`` argument (``ttml/common/trainer.py:30-38``)
       selects ``shard_tensor_to_mesh_mapper(device, 0)``. Without it the host array is
       *replicated*, so every chip evaluates the same ``validation_batch_size`` windows and
       the "held-out sample" is a quarter the size it claims to be.
    2. ``loss.to_numpy()`` needs a composer. The loss is one scalar **per device**, i.e. a
       tensor distributed over the mesh, and reading it back without a composer raises
       ``Can't get a single buffer from host storage distributed over mesh``. Composing on
       dim 0 concatenates the four per-device scalars, and ``.mean()`` over them is the mean
       over the whole (sharded) batch because every shard has the same number of rows.

    Neither the mask nor the model needs mesh handling: DDP *replicates* weights, and a
    ``from_numpy`` with no mapper on a mesh device replicates too — which is exactly what
    ``ttml.common.trainer.train()`` relies on for its own causal mask.
    """
    import ttml
    import ttnn
    from ttml.common.trainer import get_batch_ttml
    from ttml.common.utils import build_causal_mask, no_grad

    composer = None
    if use_ddp:
        device = ttml.autograd.AutoContext.get_instance().get_device()
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)

    mask = ttml.autograd.Tensor.from_numpy(
        build_causal_mask(cfg.seq_len), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
    )
    model.eval()
    total = 0.0
    with no_grad():
        for _ in range(batches):
            x, y = get_batch_ttml(val_ids, cfg.seq_len, cfg.validation_batch_size, use_ddp)
            logits = model(x, mask)
            loss = ttml.ops.loss.cross_entropy_loss(logits, y, ttml.ops.ReduceType.MEAN)
            total += float(loss.to_numpy(composer=composer).mean())
            ttml.autograd.AutoContext.get_instance().reset_graph()
    model.train()
    return total / batches


#: Schedule shapes accepted by ``--lr-schedule``. ``constant`` is the default and is a true
#: no-op — see ``lr_at_step`` and ``run_training_loop`` for why that matters.
LR_SCHEDULES = ("constant", "cosine", "linear")


def lr_at_step(
    base_lr: float,
    step: float,
    total_steps: int,
    *,
    schedule: str = "constant",
    min_lr: float = 0.0,
    decay_start_frac: float = 0.0,
    warmup_frac: float = 0.0,
) -> float:
    """The learning rate a ``schedule`` prescribes at ``step`` of a ``total_steps`` run.

    Pure arithmetic — no optimizer, no device, no state. That is deliberate: it makes the
    whole schedule testable on a machine with no board (``tests/test_lr_schedule.py``), and
    it means the LR is a function of *position in the run* rather than of how many times
    something happened to be stepped. The latter matters here specifically, because the
    caller advances in **non-uniform chunks** (see ``run_training_loop``) and a step-counting
    scheduler object would silently mis-time itself on the short final chunk.

    WHY NOT REUSE ttml's SCHEDULERS. tt-train ships
    ``~/tt-metal/tt-train/sources/ttml/ttml/common/schedulers.py`` (``CosineAnnealing``,
    ``Step``, ``Linear``, ``Lambda``, ``Sequential``). They were the first thing checked and
    they do not fit this loop. Every one of them is a stateful object whose ``step()``
    advances an internal counter by exactly one and then calls ``optimizer.set_lr()`` —
    they are built for a loop that owns each individual training step. We do not own the
    individual steps: ttml's ``train()`` is a black box that runs ``cfg.steps`` steps
    internally and exposes no per-step hook (verified — ``ttml/common/trainer.py`` has no
    scheduler or LR callback of any kind). So the only place we can move the LR is *between*
    chunks, and those chunks are 500 steps here but 264 for the run's remainder. Driving a
    counter-based scheduler from that would either need one ``step()`` call per training step
    (500 redundant C++ ``set_lr`` calls per chunk, purely to advance a counter) or would let
    the counter drift out of correspondence with the real step number. A position function is
    both simpler and exactly right.

    THE SHAPE. ``decay_start_frac`` holds the LR flat at ``base_lr`` for that fraction of the
    run, then decays to ``min_lr`` across whatever is left. ``0.0`` (the default) decays over
    the whole run — the plain reading of "a cosine schedule". A non-zero value is the
    stable-then-decay ("decay tail") shape, which is what makes a clean A/B against an
    existing constant-LR run possible: the held portion reproduces that run exactly, so any
    divergence is attributable to the decay alone.

    ``constant`` ignores ``min_lr``/``decay_start_frac`` entirely and returns ``base_lr``.

    Args:
        base_lr: The peak/held LR — the optimizer's configured LR, unchanged from the config.
        step: Position in the run, in steps. Fractional positions are meaningful and
            expected — ``run_training_loop`` passes chunk *midpoints*. Clamped to
            ``[0, total_steps]``.
        total_steps: Length of the run the schedule spans. ``<= 0`` yields ``base_lr``.
        schedule: One of :data:`LR_SCHEDULES`.
        min_lr: LR at the end of the decay. Never returned before the run's final step.
        warmup_frac: Fraction of the run spent ramping linearly from 0 to ``base_lr``
            before anything else happens. 0 (the default) disables warmup entirely, so
            every pre-existing call site is unchanged. Applies to ``constant`` too.
        decay_start_frac: Fraction of the run held at ``base_lr`` before decay begins, in
            ``[0.0, 1.0]``. ``>= 1.0`` means "never decay" and yields ``base_lr`` throughout.

    Returns:
        The learning rate, as a float.

    Raises:
        ValueError: if ``schedule`` is not in :data:`LR_SCHEDULES`.
    """
    if schedule not in LR_SCHEDULES:
        raise ValueError(f"unknown lr schedule {schedule!r}; expected one of {LR_SCHEDULES}")
    if not 0.0 <= warmup_frac < 1.0:
        raise ValueError(f"warmup_frac must be in [0, 1), got {warmup_frac!r}")

    # Warmup is applied BEFORE the `constant` short-circuit below, deliberately.
    # `--lr-schedule constant --warmup-frac 0.02` has to mean "ramp up, then hold",
    # not "silently ignore the warmup" -- a flag that is quietly inert is the exact
    # shape of defect this project keeps finding in its own tooling.
    #
    # The ramp is linear from 0 to base_lr over the first `warmup_frac` of the run.
    # It exists because AdamW's second-moment estimate is near-meaningless for the
    # first handful of steps, so the least trustworthy updates are also the largest.
    # Adopted from tt-metal v0.77.0's stability set (#48716), which switches to a
    # `warmup_linear` scheduler -- but implemented HERE rather than by enabling
    # ttml's scheduler, because run_training_loop sets the LR itself once per chunk
    # and two authorities over one number is worse than either alone.
    if warmup_frac > 0.0 and total_steps > 0:
        warmup_steps = warmup_frac * total_steps
        if step < warmup_steps:
            return base_lr * (step / warmup_steps) if warmup_steps > 0 else base_lr

    if schedule == "constant" or total_steps <= 0 or decay_start_frac >= 1.0:
        return base_lr

    progress = min(max(step / total_steps, 0.0), 1.0)
    if progress <= decay_start_frac:
        return base_lr

    # Re-normalise the post-hold remainder onto [0, 1] so the decay always lands exactly on
    # min_lr at the run's final step, whatever fraction was held.
    t = (progress - decay_start_frac) / (1.0 - decay_start_frac)
    t = min(max(t, 0.0), 1.0)

    if schedule == "cosine":
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))
    # linear
    return base_lr + (min_lr - base_lr) * t


def _chunk_size(ran: int, remaining: int, save_every: int, val_every: int) -> int:
    """Steps to run in the next sub-chunk of ``train()``.

    Chosen so the chunk loop always stops exactly on whichever comes first: the next
    checkpoint boundary, the next validation boundary, or the run's end. ``save_every`` /
    ``val_every`` of 0 means "no such boundary" and does not constrain the chunk size.
    ``ran`` is steps already run *this invocation* (not the absolute step, which may carry
    a ``--resume`` offset) — both boundaries are periodic in invocation-local step count,
    matching the pre-existing ``--save-every`` behaviour.
    """
    size = remaining
    if save_every > 0:
        size = min(size, save_every - (ran % save_every))
    if val_every > 0:
        size = min(size, val_every - (ran % val_every))
    return size


def _at_boundary(ran: int, remaining: int, every: int) -> bool:
    """True if ``ran`` lands on a periodic boundary of ``every``, or this is the run's
    final chunk (``remaining == 0``).

    The ``remaining == 0`` clause preserves the pre-validation checkpoint behaviour: the
    original loop checkpointed unconditionally after every chunk, so the last chunk was
    always checkpointed even when ``--steps`` wasn't an exact multiple of ``--save-every``.
    Applying the same rule to validation means the loss curve always includes the run's
    final step, not just the periodic ones.
    """
    if every <= 0:
        return False
    return ran % every == 0 or remaining == 0


def run_training_loop(
    cfg,
    model,
    optimizer,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    *,
    save_every: int,
    val_every: int,
    start_step: int,
    val_log_path: Optional[Path],
    train_fn: Callable[..., Tuple[List[float], Any]],
    evaluate_fn: Callable[..., float],
    save_checkpoint_fn: Optional[Callable[[int], None]] = None,
    lr_fn: Optional[Callable[[float], float]] = None,
    use_ddp: bool = False,
    print_fn: Callable[..., None] = print,
) -> Tuple[List[float], List[Dict[str, Any]]]:
    """Run ``cfg.steps`` steps in chunks, checkpointing and validating at independent
    boundaries.

    ``train_fn`` / ``evaluate_fn`` are ttml's ``train()`` and this module's ``evaluate()`` in
    production, but are injected here so this loop is unit-testable without a device (see
    ``tests/test_run_validation.py``). ``save_checkpoint_fn(step)`` is called at each
    checkpoint boundary and should be ``None`` when ``save_every`` is 0.

    Mutates ``cfg.steps`` on every sub-chunk, same contract the pre-refactor loop in
    ``main()`` had (``train_fn`` reads ``cfg.steps`` to know how far to run).

    ``lr_fn``, when given, is the learning-rate schedule (in production a partial of
    :func:`lr_at_step`). It maps a position in the run, in steps, to a learning rate, and is
    applied by calling ``optimizer.set_lr()`` **once per chunk, before that chunk runs** —
    the only place the LR can be moved at all, since ttml's ``train()`` runs a whole chunk
    internally with no per-step hook. The position handed to ``lr_fn`` is the chunk's
    **midpoint** (``ran + cfg.steps / 2``), not its start or end: the LR is necessarily held
    constant across the chunk, and evaluating at the midpoint makes that staircase an
    unbiased approximation of the continuous schedule instead of one that lags (chunk start)
    or leads (chunk end) it by a chunk. When ``lr_fn`` is ``None`` — the default, and what
    ``--lr-schedule constant`` resolves to — ``set_lr`` is never called at all, so the
    optimizer keeps exactly the LR its config gave it and this loop is bit-identical to its
    pre-schedule behaviour. That is why the default is ``None`` rather than a
    constant-returning function: every run recorded before schedules existed stays
    reproducible, and the optimizer is not touched by a feature that isn't in use.

    ``use_ddp`` is forwarded verbatim to both ``train_fn`` and ``evaluate_fn``. It is a
    single flag rather than two because they must never disagree: ``train_fn`` reads it to
    decide whether to shard the batch **and whether to call ``synchronize_gradients``**
    (``ttml/common/trainer.py``), and ``evaluate_fn`` reads it to decide whether to shard the
    validation batch and compose the loss back. Passing it to one and not the other produces
    either a crash (uncomposed loss on a mesh) or a quietly wrong number (a validation set a
    quarter the intended size), depending on which way round the mistake goes. The caller in
    ``main()`` derives it from a single ``--ddp`` value, so there is one source of truth.

    Returns ``(all_losses, val_records)``. ``val_records`` is also the list appended, one
    JSON object per line, to ``val_log_path`` (skipped if ``val_log_path`` is ``None``) —
    each record is ``{"step": absolute_step, "train_loss": ..., "val_loss": ...}``, where
    ``val_loss`` comes from a real call to ``evaluate_fn`` (never copied from
    ``train_loss`` — that copy is exactly the ttml placeholder behaviour this module exists
    to avoid, see the module docstring). When ``lr_fn`` is given, each record carries an
    extra ``"lr"`` field recording the rate actually in effect over the chunk that just ran,
    so the curve says what it was trained at. The field is omitted entirely under the default
    constant behaviour, keeping those logs identical in shape to every previously-recorded run.
    """
    remaining = cfg.steps
    ran = 0
    step = start_step
    current_lr: Optional[float] = None
    all_losses: List[float] = []
    val_records: List[Dict[str, Any]] = []
    while remaining > 0:
        cfg.steps = _chunk_size(ran, remaining, save_every, val_every)
        if lr_fn is not None:
            current_lr = lr_fn(ran + cfg.steps / 2.0)
            optimizer.set_lr(current_lr)
        # (cfg, model, optim, train_ids, use_ddp, use_tp). use_tp stays False: tensor
        # parallelism shards the weights, which convert/ assumes are whole tensors.
        losses, _ = train_fn(cfg, model, optimizer, train_ids, use_ddp, False)
        all_losses.extend(losses)
        remaining -= cfg.steps
        ran += cfg.steps
        step += cfg.steps

        if save_checkpoint_fn is not None and _at_boundary(ran, remaining, save_every):
            save_checkpoint_fn(step)

        if _at_boundary(ran, remaining, val_every):
            train_loss = losses[-1] if losses else float("nan")
            val_loss = evaluate_fn(model, val_ids, cfg, use_ddp=use_ddp)
            lr_note = "" if current_lr is None else f" lr={current_lr:.3e}"
            print_fn(
                f"  step={step:>7} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f}{lr_note}"
            )
            record = {"step": step, "train_loss": train_loss, "val_loss": val_loss}
            if current_lr is not None:
                record["lr"] = current_lr
            val_records.append(record)
            if val_log_path is not None:
                val_log_path.parent.mkdir(parents=True, exist_ok=True)
                with val_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

    return all_losses, val_records


def _warn_if_stochastic_rounding_disabled(yaml_config: Dict[str, Any]) -> None:
    """Unconditional runtime guard for the frozen-gamma bug fixed in Task 1.

    ``build_yaml_config``'s ``stochastic_rounding`` default is ``False`` (kept for backward
    compatibility — see its docstring in ``train/config.py``), so simply omitting
    ``--config`` silently reruns the exact configuration that produced 13 permanently-frozen
    RMSNorm gammas, over a perfectly healthy-looking loss curve (the bug is invisible in
    training loss — see ``tests/test_training_config.py``). Previously the resolved
    optimizer was only printed when ``--config`` was passed, so an operator who forgot the
    flag got zero signal before committing to a run that can take tens of minutes.

    Runs on every invocation, ``--config`` or not, and always prints the resolved value —
    the point is that the operator sees what they're about to run either way, not only when
    something is wrong.
    """
    stochastic_rounding = yaml_config["training_config"]["optimizer"]["stochastic_rounding"]
    print(f"  stochastic_rounding: {stochastic_rounding}")
    if not stochastic_rounding:
        print(
            "WARNING: stochastic_rounding is disabled — RMSNorm gamma parameters will not "
            "learn. bfloat16 at 1.0 has a step size (ulp) of 0.0039, an order of magnitude "
            "larger than the ~3e-4 Adam updates those gammas receive, so every update "
            "rounds deterministically back to 1.0 and is discarded, every single time. Pass "
            "--config train/configs/nanollama3_bpe_v2.yaml (or otherwise set "
            "training_config.optimizer.stochastic_rounding: true) to fix this.",
            file=sys.stderr,
        )


def ttml_cxx_header_fields(size) -> Dict[str, Any]:
    """The four header facts that exist only as hardcoded defaults inside ttml's C++.

    These are not in any yaml (``nanollama3.yaml`` never sets them) and — apart from
    ``intermediate_dim`` — are not recoverable from the checkpoint's own tensors either,
    so they must be captured at write time or a later converter has to guess.
    ``weight_tying`` is the dangerous one: because it is on, a checkpoint has no
    ``llama/tok_emb/weight`` tensor at all (the embedding is tied to ``llama/fc/weight``),
    and a converter that doesn't know produces a model with a randomly-initialized
    embedding table while raising no error.

    **Why this is a function and not a dict literal at the call site.** It used to be a
    literal, with ``"intermediate_dim": 1024`` written out by hand. 1024 is the value
    ttml *derives* for the 384 model, so the constant was invisibly correct for the only
    size that had ever been trained — and silently wrong for the first ``--size 1024``
    run, which built a 2816-wide FFN while its checkpoints claimed 1024. Training did not
    care (ttml derives the width itself and never reads the header), but
    ``convert/to_hf.py`` copies the field straight into ``config.json``'s
    ``intermediate_size``, so the converted model failed to load at all. Deriving it from
    the size registry — the module that already reproduces ttml's rule and is pinned
    against the real converted model by ``tests/test_sizes.py`` — makes that class of
    drift impossible to reintroduce by hand.

    ``size`` is a :class:`train.sizes.ModelSize`.
    """
    return {
        # round_up(4 * embedding_dim * 2/3, 256); llama_block.cpp:15-23, reproduced by
        # ModelSize.intermediate_dim. Derived, never restated — see above.
        "intermediate_dim": size.intermediate_dim,
        # WeightTyingType::Enabled default; models/llama.hpp:35.
        "weight_tying": True,
        # RMSNormLayer default; modules/rms_norm_module.hpp:17.
        "rms_norm_eps": 1e-5,
        # All 50 model tensors are BFLOAT16 per this run's manifest.
        "weights_dtype": "bfloat16",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens-dir", default=str(ROOT / "artifacts" / "tokens"))
    p.add_argument("--steps", type=int, default=20,
                   help="Steps to run in this invocation. With --resume, this many "
                        "steps run past the checkpoint's step, not up to it — "
                        "'--resume latest --steps 100' trains to start_step + 100, "
                        "not to step 100.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=None,
                   help="Training sequence length in tokens. Defaults to the selected "
                        "--size's own max_sequence_length, which is the common case: "
                        "train at the context the architecture declares. It may also be "
                        "set SHORTER than that, which is how a matched-context control "
                        "run is done (e.g. --size 384 --seq-len 512 against a size "
                        "declaring 2048) -- the RoPE cache is simply read as a prefix, "
                        "which is exact. It may NOT exceed the size's "
                        "max_sequence_length: tiles past the end of the cache are "
                        "zero-filled on-device with no error. build_yaml_config enforces "
                        "that direction; see its docstring. Must be a multiple of 32.")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--size", default=DEFAULT_SIZE, choices=sorted(SIZES),
                   help=f"Model architecture to train, from train/sizes.py "
                        f"(default: {DEFAULT_SIZE}, the originally-trained model). "
                        f"Each size has its own vendored ttml config under "
                        f"train/configs/model/.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"RNG seed for weight init and batch shuffling (default: "
                        f"{DEFAULT_SEED}, the value hardcoded through v1-v4, so omitting "
                        f"this reproduces every committed measurement). Change it only to "
                        f"vary a run deliberately -- repeating a run under a different seed "
                        f"is what measures run-to-run variance, the noise floor every "
                        f"between-run comparison has to be read against.")
    p.add_argument("--arch", default="blackhole", choices=["blackhole", "wormhole_b0"])
    p.add_argument("--tt-metal-home", default=_default_tt_metal_home())
    p.add_argument("--model-impl", default="python", choices=["python", "cpp"],
                   help="Which of ttml's two Llama implementations to train. 'python' (the "
                        "default) is ttml.models.llama.Llama wrapped by train.model, which "
                        "drops the redundant causal mask and so runs SDPA in causal rather "
                        "than arbitrary-mask mode -- measured 1.41x faster per step at --size "
                        "384 and 1.15x at --size 1024, with identical parameter names and "
                        "checkpoints. 'cpp' is ttml's CppLlama via its own "
                        "TransformerModelFactory, which is what this project trained before "
                        "2026-08-16; its nanobind binding cannot accept a null mask, so it "
                        "always pays for the arbitrary-mask kernel. Kept as an A/B lever and "
                        "an escape hatch -- see train/model.py for the whole story.")
    p.add_argument("--ddp", type=int, default=1,
                   help="Number of chips to run data-parallel over, i.e. the width of the "
                        "[1, N] mesh this run opens (default: 1, single chip -- byte for "
                        "byte the behaviour of every run recorded before 2026-08-16). "
                        "batch_size is the TOTAL batch and is sharded across the mesh on "
                        "dim 0, and gradients are all-reduced AND divided by N, so --ddp 4 "
                        "at an unchanged --batch-size trains exactly the same effective "
                        "batch as --ddp 1 does: the step budget and the learning rate do "
                        "NOT move, only wall-clock does. --batch-size must be divisible by "
                        "N. The mesh is always [1, N] and never [2, 2] -- a 2-D mesh is a "
                        "hard TT_FATAL unless two parallelisms are enabled "
                        "(autograd/auto_context.cpp:198-204), and DDP is the only one this "
                        "project wants. N > 1 also initialises ttml's parallelism context, "
                        "without which gradient synchronization silently does nothing; see "
                        "_init_parallelism_context.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the resolved config and exit without opening a device.")
    p.add_argument("--save-every", type=int, default=0,
                   help="Checkpoint every N steps (0 disables checkpointing).")
    p.add_argument("--val-every", type=int, default=0,
                   help="Compute the real validation loss (this module's evaluate(), never "
                        "ttml's placeholder) every N steps and append one JSON line "
                        "{step, train_loss, val_loss} to <checkpoint-dir>/val_losses.jsonl "
                        "(0 disables periodic validation; the single end-of-run validation "
                        "loss printed at exit is unaffected either way). Independent of "
                        "--save-every: a validation boundary does not need to coincide with "
                        "a checkpoint boundary, or vice versa, and each still fires once "
                        "more at the run's final step if --steps isn't an exact multiple.")
    # Default deliberately None, resolved after --size is known (see resolve_checkpoint_dir).
    # It used to default to artifacts/checkpoints -- the protected baseline -- so a bare
    # `python train/run.py` wrote into irreplaceable evidence.
    p.add_argument("--checkpoint-dir", default=None,
                   help="Where checkpoints are read from and written to "
                        "(default: artifacts/<size>/checkpoints). Refuses to write to the "
                        "protected baseline directories regardless of what is passed.")
    p.add_argument("--resume", default=None,
                   help="Checkpoint path to resume from, or 'latest' to pick the newest "
                        "in --checkpoint-dir. --steps then counts steps run in this "
                        "invocation (additive past the checkpoint's step), not an "
                        "absolute target step.")
    p.add_argument("--config", default=None,
                   help="Optional training-recipe YAML (e.g. "
                        "train/configs/nanollama3_bpe_v2.yaml) whose "
                        "training_config.optimizer block replaces the default optimizer "
                        "assembled from CLI flags. This is how a fix like "
                        "stochastic_rounding gets opted into without a dedicated flag for "
                        "every future optimizer tweak; see "
                        "train.config.apply_optimizer_override.")
    p.add_argument("--lr-schedule", default="constant", choices=list(LR_SCHEDULES),
                   help="Learning-rate schedule shape (default: constant — the LR stays at "
                        "the optimizer's configured value for the whole run and set_lr() is "
                        "never called, which is exactly what every run before this flag "
                        "existed did, so those runs stay reproducible by simply not passing "
                        "it). 'cosine' and 'linear' decay from that same configured LR down "
                        "to --lr-min. The LR can only move between chunks (ttml's train() "
                        "has no per-step hook), so the schedule is a staircase evaluated at "
                        "each chunk's midpoint — with --val-every 500 that is a 500-step "
                        "step size. See lr_at_step.")
    p.add_argument("--lr-min", type=float, default=None,
                   help="Learning rate at the end of the decay (ignored by 'constant'). "
                        "Defaults to one tenth of the configured LR, the conventional 10x "
                        "decay. Reported explicitly at startup either way.")
    p.add_argument("--warmup-frac", type=float, default=0.0,
                   help="Fraction of the run spent ramping the LR linearly from 0 to its "
                        "base value before anything else. 0 (default) disables warmup, so "
                        "runs recorded before 2026-08-19 reproduce exactly. Applies to "
                        "--lr-schedule constant too. Adopted from tt-metal v0.77.0's "
                        "stability set (#48716), implemented host-side in lr_at_step "
                        "rather than by enabling ttml's scheduler, because this loop "
                        "already sets the LR once per chunk and two authorities over one "
                        "number is worse than either.")
    p.add_argument("--lr-decay-start-frac", type=float, default=0.0,
                   help="Fraction of the run to hold at the full LR before decay begins, in "
                        "[0.0, 1.0) (default: 0.0 — decay across the whole run). A non-zero "
                        "value gives a 'decay tail': the held portion reproduces a "
                        "constant-LR run of the same length exactly, so an A/B against one "
                        "isolates the decay's effect instead of confounding it with a "
                        "different LR trajectory from step 1.")
    args = p.parse_args()

    if not 0.0 <= args.warmup_frac < 1.0:
        print(f"error: --warmup-frac must be in [0, 1), got {args.warmup_frac}", file=sys.stderr)
        return 2
    if not 0.0 <= args.lr_decay_start_frac < 1.0:
        print(f"ERROR: --lr-decay-start-frac must be in [0.0, 1.0), got "
              f"{args.lr_decay_start_frac}", file=sys.stderr)
        return 1

    tokens = Path(args.tokens_dir)
    train_path, val_path = tokens / "train_ids.npy", tokens / "val_ids.npy"
    if not train_path.is_file():
        print(f"ERROR: {train_path} not found. Run train/tokenization.py first.",
              file=sys.stderr)
        return 1
    if not val_path.is_file():
        print(f"ERROR: {val_path} not found. Run train/tokenization.py first.", file=sys.stderr)
        return 1

    # The architecture comes from THIS repository, not from $TT_METAL_HOME. Before the
    # size registry this read tt-train's own nanollama3.yaml, which meant the architecture
    # could change under a tt-metal upgrade with no signal and there was no way to offer a
    # second size. `train/sizes.py` owns the mapping now; `tests/test_sizes.py` holds the
    # vendored copy to being a faithful copy of the upstream original.
    try:
        size = get_size(args.size)
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    model_config = size.config_path
    if not model_config.is_file():
        print(f"ERROR: model config for size {size.name} not found at {model_config}",
              file=sys.stderr)
        return 1
    print(f"  model size: {size.name} ({model_config.name})")
    # Which implementation of that architecture, and therefore which SDPA mask mode. Two
    # runs of the same size are not comparable on time without this line: 'python' drops
    # the redundant causal mask and runs SDPA in causal rather than arbitrary-mask mode,
    # worth ~1.41x on the whole step at --size 384. See train/model.py.
    print(f"  model impl: {args.model_impl}"
          + (" (ttml's Python Llama via train.model; SDPA causal, mask dropped)"
             if args.model_impl == "python"
             else " (ttml's CppLlama via TransformerModelFactory; SDPA arbitrary-mask)"))

    # --seq-len defaults to the size's own declared context. A fixed default here could
    # only ever be right for whichever sizes happened to share it -- and 384 moving to
    # 2048 while 1024 stayed at 512 is exactly that case. An explicit shorter value is
    # allowed (prefix read of the RoPE cache, exact); longer is rejected by
    # build_yaml_config below. Say which of the two happened, rather than always claiming
    # the value came from the size -- a run trained at a narrowed window must be legible
    # as such in its own log.
    if args.seq_len is None:
        args.seq_len = size.max_sequence_length
        print(f"  seq_len: {args.seq_len} (size's max_sequence_length)")
    elif args.seq_len < size.max_sequence_length:
        print(f"  seq_len: {args.seq_len} (NARROWED from the size's declared "
              f"max_sequence_length {size.max_sequence_length}; RoPE cache is read as a "
              f"prefix)")
    else:
        print(f"  seq_len: {args.seq_len} (explicit; equals the size's "
              f"max_sequence_length)")

    # Resolve the checkpoint directory now that the size is known, and guard it. The guard
    # applies to an explicitly-passed path too, not just the default: the point is that no
    # invocation can write into the baseline evidence, however it is invoked.
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(write_dir(size, "checkpoints"))
    else:
        try:
            assert_not_protected(args.checkpoint_dir)
        except ProtectedPathError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    print(f"  checkpoints: {args.checkpoint_dir}")

    # --ddp validation, before anything expensive. The only constraint that is really about
    # the *model* is batch divisibility: DDP replicates the weights and shards the batch, so
    # unlike tensor-parallel serving it imposes nothing on num_heads or num_groups (see
    # ModelSize.servable_mesh_widths, which is a TP/serving rule and does not apply here).
    if args.ddp < 1:
        print(f"ERROR: --ddp must be at least 1, got {args.ddp}", file=sys.stderr)
        return 1
    if args.batch_size % args.ddp:
        print(f"ERROR: --batch-size {args.batch_size} is not divisible by --ddp "
              f"{args.ddp}; batch_size is the TOTAL batch and is sharded across the mesh "
              f"on dim 0, so it must split evenly", file=sys.stderr)
        return 1
    use_ddp = args.ddp > 1
    if use_ddp:
        print(f"  ddp: {args.ddp} chips, mesh [1, {args.ddp}], "
              f"{args.batch_size // args.ddp} sequences/chip "
              f"(effective batch stays {args.batch_size}; step budget and LR unchanged)")

    yaml_config = build_yaml_config(
        str(ROOT / "artifacts" / "tokenizer"), str(model_config),
        seq_len=args.seq_len, max_sequence_length=size.max_sequence_length,
        batch_size=args.batch_size, max_steps=args.steps, eval_every=args.eval_every,
        seed=args.seed, ddp=args.ddp,
    )
    if args.config:
        apply_optimizer_override(yaml_config, args.config)
        print(f"  optimizer overridden from {args.config}: "
              f"{yaml_config['training_config']['optimizer']}")
    # Unconditional — runs whether or not --config was passed, so omitting --config no
    # longer means silently rerunning the frozen-gamma configuration with zero signal.
    _warn_if_stochastic_rounding_disabled(yaml_config)
    cfg = run_config_from_yaml(yaml_config)

    # Resolve the LR schedule now, before the device is opened, so --dry-run shows it and a
    # bad combination fails in the first second rather than 95 minutes in. base_lr is read
    # from the *resolved* optimizer block, so it already reflects --config: the schedule
    # decays from whatever LR this run is actually configured with, never a hardcoded guess.
    base_lr = float(yaml_config["training_config"]["optimizer"]["lr"])
    lr_min = base_lr * 0.1 if args.lr_min is None else args.lr_min
    lr_fn = None
    # `constant` normally means "never touch the LR", which is why set_lr is skipped
    # entirely -- but warmup is a change to the LR, so a constant schedule WITH warmup
    # still needs the hook. Getting this wrong is not theoretical: the first v0.77.0
    # training run (2026-08-19) was launched with --warmup-frac 0.02 against the
    # default constant schedule, this branch set lr_fn = None, and the warmup silently
    # did not happen for 6,000 steps. tests/test_lr_schedule.py had a passing test
    # named test_warmup_applies_to_constant_too -- it exercised lr_at_step, which
    # returned the right number that nobody asked for. The guard was one layer too low.
    if args.lr_schedule == "constant" and args.warmup_frac <= 0.0:
        print(f"  lr schedule: constant at {base_lr:.3e} (set_lr never called)")
    elif args.lr_schedule == "constant":
        warm_steps = int(args.warmup_frac * args.steps)
        def lr_fn(position: float) -> float:  # noqa: E306
            """Warmup, then flat. `constant` contributes the flat part."""
            return lr_at_step(base_lr, position, args.steps, schedule="constant",
                              warmup_frac=args.warmup_frac)
        print(f"  lr schedule: constant at {base_lr:.3e}, warmup 0 -> {base_lr:.3e} "
              f"over the first {args.warmup_frac:.1%} of {args.steps} steps "
              f"(~{warm_steps} steps); set_lr called once per chunk")
    else:
        held = args.lr_decay_start_frac
        def lr_fn(position: float) -> float:  # noqa: E306 — paired with the branch above
            """The schedule, bound to this run's shape. Passed to run_training_loop, which
            calls it once per chunk with that chunk's midpoint."""
            return lr_at_step(base_lr, position, args.steps, schedule=args.lr_schedule,
                              min_lr=lr_min, decay_start_frac=held,
                              warmup_frac=args.warmup_frac)
        print(f"  lr schedule: {args.lr_schedule} {base_lr:.3e} -> {lr_min:.3e}"
              + (f", held flat for the first {held:.0%} of {args.steps} steps "
                 f"(decay begins ~step {int(held * args.steps)})" if held else
                 f" across all {args.steps} steps"))

    # Read back out of the *resolved* config rather than echoing args.seed, so this line
    # is evidence that the flag reached what set_seed() will actually be handed.
    resolved_seed = yaml_config["training_config"]["seed"]
    print(f"  seed: {resolved_seed}"
          + ("" if resolved_seed == DEFAULT_SEED else f" (default is {DEFAULT_SEED})"))

    print(f"tt-tnt training — steps={cfg.steps} batch={cfg.batch_size} "
          f"seq_len={cfg.seq_len} arch={args.arch}")
    if args.dry_run:
        print("--dry-run set: not opening a device.")
        return 0

    _prepare_env(args.tt_metal_home, args.arch)

    # Must be set before ttml is imported and the fabric is armed -- enable_fabric() reads
    # TT_MESH_GRAPH_DESC_PATH at the moment it is called. See _mesh_graph_descriptor_path
    # for why a 4-chip run needs this named explicitly and an 8- or 32-chip one does not.
    try:
        mgd = _mesh_graph_descriptor_path(args.ddp)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if mgd is not None:
        if not mgd.is_file():
            print(f"ERROR: mesh graph descriptor not found at {mgd}; a --ddp {args.ddp} run "
                  f"needs one (enable_fabric ships defaults for 8 and 32 devices only) and "
                  f"hangs in fabric router sync without it", file=sys.stderr)
            return 1
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = str(mgd)
        print(f"  mesh graph descriptor: {mgd}")

    import ttml  # noqa: E402
    from ttml.common.model_factory import TransformerModelFactory  # noqa: E402
    from ttml.common.trainer import train  # noqa: E402
    from ttml.common.utils import create_optimizer, initialize_device, set_seed  # noqa: E402

    train_ids = np.load(train_path)
    val_ids = np.load(val_path)
    print(f"  train tokens={len(train_ids):,}  val tokens={len(val_ids):,}")

    # Nothing else checks that the token stream fits the model's vocabulary. The model's
    # embedding table is sized from the model config yaml (transformer_config.vocab_size),
    # not from train.config.VOCAB_SIZE — config.py never reads the yaml, it only asserts
    # its own constant against itself. If the two disagree, or if a token id from a
    # different tokenizer slipped in, an out-of-range embedding lookup produces silent
    # garbage or an on-device fault with no diagnostic. Catch it here, before the device
    # is even open.
    with model_config.open("r", encoding="utf-8") as f:
        model_yaml = yaml.safe_load(f)
    transformer_config = model_yaml["transformer_config"]
    model_vocab_size = transformer_config["vocab_size"]
    if model_vocab_size != VOCAB_SIZE:
        raise ValueError(
            f"model config declares vocab_size={model_vocab_size} but train.config.VOCAB_SIZE "
            f"is {VOCAB_SIZE}; the tokenizer and the model disagree"
        )
    if int(train_ids.max()) >= VOCAB_SIZE:
        raise ValueError(
            f"token id {int(train_ids.max())} exceeds vocab_size {VOCAB_SIZE}; these tokens "
            "were produced by a different tokenizer than the model config expects"
        )

    set_seed(yaml_config["training_config"]["seed"])
    try:
        initialize_device(yaml_config)
    except Exception:
        print(
            "ERROR: initialize_device failed to open the device. If the board timed out, "
            "run `tt-smi -r` to reset it and retry.",
            file=sys.stderr,
        )
        raise

    # Everything from here to the end of the function runs with the device open, so it
    # all belongs inside this try — model/optimizer construction included. If either
    # raises (bad config, on-device OOM) before train()/evaluate() even start, the device
    # must still be closed in the finally below, or teardown aborts in
    # MetalContext::destroy_all_instances.
    try:
        # Immediately after the mesh opens and BEFORE the model is built. The context is
        # what makes synchronize_gradients do anything at all (see the function's docstring
        # for the silent early return it exists to prevent), and it reads the live mesh, so
        # it cannot be initialised before open_device. It raises if the mesh that opened is
        # not the [1, --ddp] one asked for, which is why nothing downstream re-checks.
        if use_ddp:
            _init_parallelism_context(args.ddp)

        # Two implementations of the same architecture; see --model-impl's help and the
        # module docstring of train/model.py. They produce the same parameter names, the
        # same parameter count, and interchangeable checkpoints -- the Python one is
        # simply able to take the null mask that puts SDPA on its causal path.
        if args.model_impl == "python":
            model = tt_tnt_model.create_model(yaml_config, transformer_config)
        else:
            model = TransformerModelFactory(yaml_config).create_model()
        optimizer = create_optimizer(model, yaml_config)

        # ttml's train() sets the progress bar's val_loss to a copy of train_loss whenever
        # step % eval_every == 0 or step == 1 — it is not a real validation number. Tell the
        # operator before the bar starts printing it, not after they've already trusted it.
        print(
            "note: the progress bar's val_loss is ttml's placeholder (a copy of "
            "train_loss); the real validation loss is computed after training and "
            "printed below."
        )
        start_step = 0
        if args.resume:
            resume_path = (checkpoint.latest_checkpoint(Path(args.checkpoint_dir))
                           if args.resume == "latest" else Path(args.resume))
            if resume_path is None or not resume_path.is_file():
                raise FileNotFoundError(f"no checkpoint to resume from: {args.resume}")
            header = checkpoint.load(resume_path, model_params=model.parameters(),
                                     optimizer=optimizer)
            start_step = int(header["step"])
            # cfg.steps is still the --steps value here (the chunk loop below hasn't
            # touched it yet) — state the end step now, at the moment the operator is
            # actually looking, since --steps is additive past start_step, not absolute.
            # created_at is printed alongside the step because latest_checkpoint() picks
            # the highest-step file in --checkpoint-dir, not the most recently written one
            # — with one directory shared across runs those can silently differ, so the
            # operator needs a way to see which run's weights they actually got.
            print(f"  resumed from {resume_path} at step {start_step} "
                  f"(created_at={header.get('created_at', 'unknown')}); "
                  f"running {cfg.steps} more steps to step {start_step + cfg.steps}")

        def _save_checkpoint(step: int) -> None:
            """Checkpoint boundary callback for run_training_loop — everything below is
            unchanged from the pre-refactor inline checkpoint block."""
            path = checkpoint.checkpoint_path(Path(args.checkpoint_dir), step)
            checkpoint.save(
                path,
                header=checkpoint.build_header(
                    step, model_config_path=str(model_config),
                    tokenizer_dir=str(ROOT / "artifacts" / "tokenizer"),
                    corpus_tokens=int(len(train_ids) + len(val_ids)),
                    batch_size=args.batch_size,
                    # Explicit, not the build_header default: seq_len is now a CLI flag
                    # (--seq-len), so the header must record what THIS run actually used,
                    # not train.config.SEQ_LEN's current value. See build_header's
                    # docstring for why a wrong value here would be a silent lie that
                    # propagates into convert/to_hf.py's max_position_embeddings.
                    seq_len=cfg.seq_len,
                    extra={
                        "transformer_config": transformer_config,
                        # The ttml C++ defaults that no yaml carries, derived from the
                        # size registry rather than restated here — see
                        # ttml_cxx_header_fields for what they are and for the
                        # hardcoded-1024 bug that motivated pulling them out of this
                        # literal.
                        **ttml_cxx_header_fields(size),
                    },
                ),
                model_params=model.parameters(), optimizer=optimizer,
            )
            print(f"  checkpoint saved: {path}")

        # train() takes exactly (cfg, model, optim, train_ids, use_ddp, use_tp) — no val_ids.
        # ttml's train() has no checkpoint hook of its own, so we call it in chunks (of
        # whichever comes first: --save-every, --val-every, or the run's end) and save/
        # validate between chunks. The optimizer object persists across calls (it is the
        # same Python object each time), so AdamW's moments carry over — only train_losses
        # is per-call and must be accumulated here. See run_training_loop for the cadence
        # logic and why the real evaluate() (not ttml's placeholder val_losses) is used.
        all_losses, val_records = run_training_loop(
            cfg, model, optimizer, train_ids, val_ids,
            save_every=args.save_every, val_every=args.val_every, start_step=start_step,
            val_log_path=Path(args.checkpoint_dir) / "val_losses.jsonl",
            train_fn=train, evaluate_fn=evaluate,
            save_checkpoint_fn=_save_checkpoint if args.save_every > 0 else None,
            lr_fn=lr_fn, use_ddp=use_ddp,
        )
        if val_records:
            print(f"  periodic validation curve ({len(val_records)} entries) written to "
                  f"{Path(args.checkpoint_dir) / 'val_losses.jsonl'}")
        train_losses = all_losses
        val_loss = evaluate(model, val_ids, cfg, use_ddp=use_ddp)
        if train_losses:
            print(f"\nfirst train loss : {train_losses[0]:.4f}")
            print(f"last  train loss : {train_losses[-1]:.4f}")
        else:
            print("\nno training steps ran (--steps 0); no train loss to report.")
        print(f"real  val   loss : {val_loss:.4f}")
    finally:
        # Let ttml close the device — bypassing this triggers a teardown abort in
        # MetalContext::destroy_all_instances.
        ttml.autograd.AutoContext.get_instance().close_device()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
