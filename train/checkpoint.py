# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Checkpoint header schema and thin wrappers over ``ttml.checkpointing``.

ttml already does the hard part — it streams tensors to disk one at a time and writes
atomically (temp file then rename), so a crash mid-write leaves the previous checkpoint
intact. We add two things it deliberately leaves open:

1. **A validated header schema.** ttml's header is an opaque dict, so nothing checks it. A
   checkpoint whose header omits ``vocab_size`` cannot be converted later without guessing,
   and guessing is how a converted model silently mismatches its tokenizer.
2. **Path conventions**, so checkpoints sort by step and a resume can find the newest.

Everything else is a pass-through. Do not reimplement ttml's storage.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from train.config import SEQ_LEN, VOCAB_SIZE

#: Our header schema version, independent of ttml's own on-disk FORMAT_VERSION.
#: Bump when a field's meaning changes, not when one is added.
CHECKPOINT_FORMAT = 1

#: Fields every checkpoint header must carry. `extra` may not shadow any of these.
#:
#: `corpus_tokens`/`batch_size`/`tokens_seen` replace the old single `total_tokens` field
#: (checkpoint format 1, pre-fix): `total_tokens` recorded the size of the whole corpus
#: (train + val split), not how many tokens this checkpoint's training actually consumed —
#: a model card reading it would silently overstate training volume by ~2.6x for these
#: checkpoints (127.6M corpus vs 49.2M actually trained on at step 3000). `batch_size` makes
#: `tokens_seen = step * batch_size * seq_len` derivable from the header alone, without
#: guessing at a value nothing else records.
_REQUIRED = (
    "format", "step", "vocab_size", "seq_len",
    "model_config_path", "tokenizer_dir",
    "corpus_tokens", "batch_size", "tokens_seen", "created_at",
)


def build_header(
    step: int,
    *,
    model_config_path: str,
    tokenizer_dir: str,
    corpus_tokens: int,
    batch_size: int,
    seq_len: int = SEQ_LEN,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the header stored alongside a checkpoint's tensors.

    ``vocab_size`` is recorded from ``train.config`` rather than passed in: it must
    describe the model that produced these weights, and taking it from the single source
    of truth removes the chance of a caller recording something else.

    ``seq_len`` records the sequence length **actually used to train this checkpoint**.
    It defaults to ``train.config.SEQ_LEN`` purely for callers (tests, ad-hoc scripts)
    that don't care and don't want to plumb it explicitly — but ``seq_len`` is now a CLI
    flag (``train/run.py --seq-len``), so the module constant is no longer necessarily
    what any given run actually used. The real training call site always passes
    ``seq_len=cfg.seq_len`` explicitly (the resolved ``RunConfig`` value for *this* run),
    precisely so a header never silently records a value the run didn't use. Recording the
    wrong seq_len here would propagate into ``convert/to_hf.py``'s
    ``max_position_embeddings`` with no error anywhere along the way — exactly the kind of
    silent lie this schema exists to prevent for the other fields.

    ``corpus_tokens`` is the size of the corpus split the checkpoint was trained against
    (train + val token count) — provenance, not a training-volume claim. ``batch_size`` plus
    ``step`` and ``seq_len`` (already in the header) let us record the number that actually
    matters, ``tokens_seen``, without the caller having to compute or pass it separately.

    ``extra`` is also where a caller should put facts that exist only as hardcoded defaults
    in ttml's C++ (e.g. ``intermediate_dim``, ``weight_tying``, ``rms_norm_eps``) and are not
    recoverable from any yaml or from the checkpoint's own tensors later — see
    ``train/run.py``'s call site for why those three specifically must be captured here, at
    write time.
    """
    header: Dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "step": int(step),
        "vocab_size": VOCAB_SIZE,
        "seq_len": int(seq_len),
        "model_config_path": str(model_config_path),
        "tokenizer_dir": str(tokenizer_dir),
        "corpus_tokens": int(corpus_tokens),
        "batch_size": int(batch_size),
        "tokens_seen": int(step) * int(batch_size) * int(seq_len),
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    if extra:
        clashes = sorted(set(extra) & set(_REQUIRED))
        if clashes:
            raise ValueError(f"extra may not override schema field(s): {', '.join(clashes)}")
        header.update(extra)
    return header


def validate_header(header: Dict[str, Any]) -> None:
    """Raise ``ValueError`` if ``header`` is not a checkpoint header this code can read."""
    missing = [f for f in _REQUIRED if f not in header]
    if missing:
        raise ValueError(f"checkpoint header missing required field(s): {', '.join(missing)}")
    fmt = header["format"]
    if fmt > CHECKPOINT_FORMAT:
        raise ValueError(
            f"checkpoint header format {fmt} is newer than this code understands "
            f"({CHECKPOINT_FORMAT}); upgrade tt-tnt to read it"
        )


#: Filename prefix for checkpoints written by *this* code. Checkpoints already on disk from
#: before the tt-nanollama3 -> tt-tnt rename were written as ``nanollama3_step<N>.pkl`` and
#: are never renamed (they are evidence of a real run under the old name) — see
#: ``_LEGACY_GLOB`` below, which keeps them discoverable by ``latest_checkpoint`` alongside
#: anything newly written under the new prefix.
CHECKPOINT_PREFIX = "tt_tnt_step"

#: Glob for checkpoints written before the rename. Kept read-only: nothing in this codebase
#: ever writes a new file matching this pattern.
_LEGACY_GLOB = "nanollama3_step*.pkl"


def checkpoint_path(checkpoint_dir: Path, step: int) -> Path:
    """``<dir>/tt_tnt_step<step>.pkl``, zero-padded so paths sort by step.

    Without padding, ``step10`` sorts before ``step9`` and "newest checkpoint" becomes wrong.

    This prefix applies to checkpoints written from here on. Checkpoints written before the
    tt-nanollama3 -> tt-tnt rename are named ``nanollama3_step<N>.pkl`` and are untouched on
    disk; :func:`latest_checkpoint` still finds them (see its docstring).
    """
    return Path(checkpoint_dir) / f"{CHECKPOINT_PREFIX}{int(step):08d}.pkl"


def latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Highest-step checkpoint in ``checkpoint_dir``, or ``None`` if there are none.

    This is the highest **step**, not the most recently *written* file — with one directory
    shared across runs those can silently differ (an older run's step-5000 checkpoint
    outranks this run's fresh step-100 one, even though the step-100 file has the newer
    mtime). A caller that wants to know which run's weights this actually picked should
    inspect the returned checkpoint's header for ``created_at``, printed by ``--resume``.

    Looks for **both** the current ``tt_tnt_step*.pkl`` naming and the pre-rename
    ``nanollama3_step*.pkl`` naming, so a directory holding checkpoints from before the
    tt-nanollama3 -> tt-tnt rename keeps resolving correctly (e.g. ``--resume latest`` against
    ``artifacts/checkpoints/``, which holds only old-prefixed files). Sorted by the numeric
    step embedded after "step" in the filename, not by the prefix, so the two naming schemes
    interleave correctly by step rather than the new prefix always sorting after the old one.
    """
    paths = list(Path(checkpoint_dir).glob(f"{CHECKPOINT_PREFIX}*.pkl")) + list(
        Path(checkpoint_dir).glob(_LEGACY_GLOB)
    )
    if not paths:
        return None
    return max(paths, key=lambda p: int(p.stem.rsplit("step", 1)[-1]))


def _ddp_only_parallelism() -> Tuple[Optional[int], str]:
    """``(ddp_size, "")`` when ttml's live parallelism context is **DDP and nothing else**.

    Returns ``(None, reason)`` otherwise, and a caller that was about to excuse a ``Shard``
    placement must then refuse instead. This is the load-bearing half of
    :func:`assert_saveable_on_mesh`'s narrowing, so it fails *closed*: any failure to read the
    context at all (ttml not importable, no device, an API that moved) is reported as a reason,
    never as permission.

    Why DDP-only is exactly the right question to ask. Under data parallelism the model is
    **replicated** and only the batch is sharded — no parameter is ever legitimately split, so
    a ``Shard`` placement on one is necessarily the metadata defect described in
    :func:`assert_saveable_on_mesh`. Under tensor parallelism the opposite holds: parameters
    are split for real, and re-marking one ``Replicate`` would silently save a quarter of a
    model. ttml's own gradient reduction draws the same distinction from the same fact —
    ``core/distributed/distributed.cpp``'s ``is_sharded_on_axis`` treats a ``Shard`` placement
    as "FSDP already reduce-scattered this", i.e. as meaningful only when a
    parameter-sharding parallelism is switched on.

    ``ParallelismContext`` binds ``is_ddp_enabled``/``is_tp_enabled`` but not the
    context-parallel predicate, which does not weaken this: on a line mesh (the only shape a
    DDP-only run can open — ``autograd/auto_context.cpp:198-204`` ``TT_FATAL``s otherwise)
    *exactly one* parallelism may be enabled, so ``is_ddp_enabled()`` being true already
    excludes CP and TP there; and on a 2-D mesh every parallelism that shards parameters
    reports through ``is_tp_enabled()``.
    """
    try:
        import ttml

        ctx = ttml.autograd.AutoContext.get_instance()
        if not ctx.is_parallelism_context_initialized():
            return None, "no ttml parallelism context is initialised"
        pctx = ctx.get_parallelism_context()
        if not pctx.is_ddp_enabled():
            return None, "the live parallelism context does not have DDP enabled"
        if pctx.is_tp_enabled():
            return None, (
                f"tensor parallelism is enabled ({pctx.get_tp_size()} devices), and TP shards "
                f"parameters for real — a Shard placement here may be the truth"
            )
        return int(pctx.get_ddp_size()), ""
    except Exception as exc:  # noqa: BLE001 — any read failure must deny, not permit
        return None, f"the ttml parallelism context could not be read ({exc!r})"


def _ddp_axis_topologies(tensor, ddp_size: int):
    """``(value, as-found topology, all-Replicate topology)`` if ``tensor`` is distributed over
    *exactly* the DDP axis, else ``None``.

    The check is structural, not numeric. A tensor whose distribution shape is ``[ddp_size]``
    over ``ddp_size`` mesh coordinates is laid out along the data-parallel axis and no other,
    so — given :func:`_ddp_only_parallelism` has already established that DDP is the only
    parallelism in play — every placement it carries describes the DDP axis, where parameters
    are replicated by construction. Anything else (a wider distribution, a second axis, a
    coordinate count that disagrees with the axis size) is not the defect we have measured and
    is not excused.

    **Both** returned topologies are freshly constructed from values read out here, including
    the one that merely reproduces what was already there. ``Tensor.tensor_topology()`` is
    bound ``rv_policy::reference_internal`` (``pytensor.cpp:1600``), so the object it hands back
    is a view into the tensor's own attributes: hold it across an
    ``update_tensor_topology`` call and it no longer describes what it did when you took it.
    Rebuilding the "restore me" topology from copied placements and coordinates is what makes
    :func:`replicated_for_save`'s restore mean anything.
    """
    import ttml
    import ttnn

    # NATIVE, matching ttml.sharding.Sharding.from_tensor and ttml.checkpointing's own reads:
    # the topology we correct must be the one the saver will consult, not a precision-cast copy.
    value = tensor.get_value(ttml.autograd.PreferredPrecision.NATIVE)
    topology = value.tensor_topology()
    dist_shape = list(topology.distribution_shape())
    placements = list(topology.placements())
    coords = list(topology.mesh_coords())
    if dist_shape != [ddp_size] or len(coords) != ddp_size:
        return None
    as_found = ttnn.TensorTopology(
        distribution_shape=ttnn.MeshShape(dist_shape),
        placements=placements,
        mesh_coords=coords,
    )
    replicated = ttnn.TensorTopology(
        distribution_shape=ttnn.MeshShape(dist_shape),
        placements=[ttnn.PlacementReplicate() for _ in placements],
        mesh_coords=coords,
    )
    return value, as_found, replicated


def assert_saveable_on_mesh(model_params) -> None:
    """Refuse to write a checkpoint whose parameters are sharded for any reason we cannot name.

    THE DEFECT THIS EXISTS FOR (measured 2026-08-16, see ``.superpowers/ddp-bringup.md`` and
    ``.superpowers/ddp-checkpoint-fix.md``). Under ``--ddp N`` the weights are replicated: every
    chip holds the whole of every parameter. But the tensor's *topology metadata* does not stay
    replicated. Probed directly on a ``[1, 4]`` mesh:

    ===========================  ==========================  ====================
    when                         ``Sharding.placements``     ``gather()`` returns
    ===========================  ==========================  ====================
    freshly built model          ``[PlacementReplicate()]``  ``(1, 1, 1024, 1024)``
    after two DDP training steps ``[PlacementShard(0)]``     ``(4, 1, 1024, 1024)``
    ===========================  ==========================  ====================

    All 66 parameters are re-marked, on the first step, every run.
    ``ttml.checkpointing.save_checkpoint`` then does exactly what that metadata tells it to —
    ``Sharding.from_tensor(t).gather(t)`` concatenates the four "shards" — and writes a
    checkpoint in which every tensor carries four copies stacked on dim 0, with an extra
    leading dimension that ``convert/checkpoint_reader.py``, ``convert/hf_mapping.py`` and
    ``convert/ttml_forward.py`` (all of which assume whole ``[1, 1, out, in]`` tensors) cannot
    read. Measured: 1,475,602,288 bytes against 737,824,624 for the identical ``--ddp 1`` run.

    WHAT CHANGED, AND WHY THIS IS NOW A NARROWED GATE RATHER THAN A BLANKET ONE.
    ``ttnn.Tensor.update_tensor_topology`` is bound in Python (``pytensor.cpp:1611``) and
    ``ttnn.TensorTopology`` is constructible from Python (``distributed_nanobind.cpp:734``), so
    the false metadata can be corrected in this repo, at save time, without touching a byte of
    weight data — see :func:`replicated_for_save`. This function's job is therefore no longer
    "refuse whenever anything is sharded" but the sharper question **"is this sharding
    explainable?"**:

    * ttml's live parallelism context must be **DDP and only DDP**
      (:func:`_ddp_only_parallelism`). Under DDP the model is replicated and only the batch is
      split, so no parameter can legitimately be sharded. Under TP one genuinely can be, and
      the answer is still to refuse.
    * every offending tensor must be distributed over **exactly the DDP axis**
      (:func:`_ddp_axis_topologies`) — distribution shape ``[ddp_size]`` over ``ddp_size``
      coordinates, nothing wider and nothing two-dimensional.

    Both conditions fail closed. If a future parallelism really does shard parameters, the
    first condition alone stops it, and the refusal below is what an operator sees.

    Single-chip runs never reach the offender branch at all: with no mesh,
    ``Sharding.from_tensor`` reports no topology and ``is_fully_replicated`` is ``True`` by
    definition.

    Raises:
        RuntimeError: if any parameter is recorded as sharded for a reason this code cannot
            attribute to the measured DDP metadata defect.
    """
    from ttml.sharding import Sharding

    offenders = [(name, tensor) for name, tensor in model_params.items()
                 if not Sharding.from_tensor(tensor).is_fully_replicated]
    if not offenders:
        return

    ddp_size, reason = _ddp_only_parallelism()
    if ddp_size is None:
        raise RuntimeError(
            f"refusing to write a checkpoint: {len(offenders)} of {len(model_params)} tensors "
            f"are recorded as SHARDED on the mesh (e.g. {offenders[0][0]}), and this is not "
            f"the known data-parallel metadata defect — {reason}. ttml's saver would gather "
            f"each one as a concatenation of every shard, so writing would either produce an "
            f"oversized checkpoint convert/ cannot read or, if the shards are genuinely "
            f"different data, silently record only part of the model. Neither is written."
        )

    unexplained = [name for name, tensor in offenders
                   if _ddp_axis_topologies(tensor, ddp_size) is None]
    if unexplained:
        raise RuntimeError(
            f"refusing to write a checkpoint: {len(unexplained)} of {len(model_params)} tensors "
            f"are recorded as SHARDED over something other than the {ddp_size}-device data-"
            f"parallel axis (e.g. {unexplained[0]}). Data parallelism replicates every "
            f"parameter, so its spurious Shard(0) marking is safe to correct at save time; a "
            f"distribution this code cannot attribute to that axis may describe a real split "
            f"of a real tensor, and is refused rather than guessed at."
        )


def _optimizer_tensors(optimizer) -> Dict[str, Any]:
    """Every tensor in ``optimizer``'s state dict, keyed by a path-like name.

    Mirrors ``ttml.checkpointing._walk``'s traversal (a ``NamedParameters`` is a leaf; a
    ``dict`` recurses, as composite optimizers nest sub-state; scalars are skipped) so that
    the tensors this returns are exactly the tensors ``save_checkpoint`` will gather.

    Measured on a ``[1, 4]`` DDP run, **0 of 132 AdamW moment tensors are re-marked** — only
    the parameters are. This walks the optimizer state anyway: covering it costs one metadata
    read per tensor and means the guard cannot be silently outflanked if a future ttml change
    starts re-marking moments too.
    """
    import ttml

    def walk(node, prefix: str) -> Dict[str, Any]:
        if isinstance(node, ttml.NamedParameters):
            return {f"{prefix}{name}": tensor for name, tensor in node.items()}
        if isinstance(node, dict):
            found: Dict[str, Any] = {}
            for key, sub in node.items():
                found.update(walk(sub, f"{prefix}{key}/"))
            return found
        return {}

    return walk(optimizer.get_state_dict(), "optimizer/")


@contextlib.contextmanager
def replicated_for_save(tensors: Dict[str, Any]) -> Iterator[int]:
    """Correct the DDP metadata defect for the duration of a save, then put it back.

    Yields the number of tensors re-marked. Every tensor recorded as sharded — having already
    passed :func:`assert_saveable_on_mesh`, so the sharding is known to be the data-parallel
    re-mark and nothing else — has its topology replaced by an otherwise identical one whose
    placements are all ``Replicate``. ``ttml.checkpointing.save_checkpoint`` then reads that
    topology (``Sharding.from_tensor(t).gather(t)``) and takes a **single copy** of each
    tensor, which is what the ``--ddp 1`` path writes and what ``convert/`` expects.

    WHY THIS APPROACH AND NOT THE OTHER TWO. Three ways to make ``--ddp N`` write a correct
    checkpoint were available:

    (a) *this one* — correct the false metadata, save, restore. It moves **no data at all**:
        ``update_tensor_topology`` rewrites the ``TensorTopology`` on the tensor's shared
        ``TensorAttributes`` (``tensor.cpp:389``) and nothing else. The save then costs exactly
        what a ``--ddp 1`` save costs, because the composer takes one copy instead of four.
    (b) *extract one replica and write it ourselves* — would mean reimplementing ttml's
        streaming, atomic (temp-file-then-rename) storage, which this module's own docstring
        forbids for good reason, and would put a second checkpoint writer in the tree that
        could drift from the one every other path uses.
    (c) *assign a freshly-replicated tensor back into each parameter* — reads every parameter
        to host and writes it back to four devices, i.e. maximum data movement, and it mutates
        the live model rather than a piece of metadata about it.

    THE RESTORE IS NOT OPTIONAL, and not because ``Replicate`` would be wrong — it is in fact
    the truthful marking. It is because a checkpoint save happens *mid-run*
    (``--save-every``), and a save that leaves the training state different from how it found
    it is a save that can change the run's result. Restoring the original topology object makes
    that impossible to argue about: after ``save()`` returns, every tensor carries the exact
    topology it carried before, so a run with ``--save-every`` and a run without it are the
    same computation. (Measured: training continues normally through a re-mark/restore cycle.)

    A NOTE ON WHICH REPLICA IS WRITTEN, because it is not always a distinction without a
    difference. With ``stochastic_rounding: false`` the four replicas are bit-identical (max
    ``|replica0 - replica_i| = 0.0`` over all 66 tensors, 4 steps) and "one copy" is
    unambiguous. With ``stochastic_rounding: true`` — which is what
    ``train/configs/nanollama3_bpe_v2.yaml`` selects, and what this project's real runs use —
    each device rounds its own AdamW update independently, so the replicas perform independent
    random walks and **do** differ (measured 2.34e-2 on ``llama_block_5/mlp_norm/gamma`` after
    4 steps). What gets written is replica 0. That is a complete and coherent model — replica 0
    saw every all-reduced gradient and applied every update — but it is one of four, not the
    only one. See ``.superpowers/ddp-checkpoint-fix.md`` and upstream ask 4.
    """
    ddp_size, reason = _ddp_only_parallelism()
    restore: List[Tuple[Any, Any]] = []
    if ddp_size is not None:
        from ttml.sharding import Sharding

        for tensor in tensors.values():
            if Sharding.from_tensor(tensor).is_fully_replicated:
                continue
            found = _ddp_axis_topologies(tensor, ddp_size)
            if found is None:  # assert_saveable_on_mesh already refused this; belt and braces
                continue
            value, original, replicated = found
            value.update_tensor_topology(replicated)
            restore.append((value, original))
    try:
        yield len(restore)
    finally:
        for value, original in restore:
            value.update_tensor_topology(original)


def save(path: Path, *, header: Dict[str, Any], model_params, optimizer,
         display_progress: bool = False) -> None:
    """Write a checkpoint. Pass-through to ttml, which handles atomicity and streaming.

    The one thing this does not simply pass through is the mesh metadata — see
    :func:`assert_saveable_on_mesh` for the DDP defect it refuses to write past, and
    :func:`replicated_for_save` for how the explainable half of that defect is corrected for
    the duration of the write. The gate runs before ``validate_header`` so the cheaper, more
    specific failure comes first.

    The gate and the correction both cover the **optimizer's** tensors as well as the model's,
    because ``save_checkpoint`` gathers both through the same code path. No AdamW moment has
    ever been observed re-marked (0 of 132), but the guard is not narrowed to only look where
    the bug has already been seen.
    """
    from ttml.checkpointing import save_checkpoint

    tensors = {name: tensor for name, tensor in model_params.items()}
    if optimizer is not None:
        tensors.update(_optimizer_tensors(optimizer))
    assert_saveable_on_mesh(tensors)
    validate_header(header)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with replicated_for_save(tensors):
        save_checkpoint(str(path), header=header, model_params=model_params,
                        optimizer=optimizer, display_progress=display_progress)


def load(path: Path, *, model_params=None, optimizer=None,
         display_progress: bool = False) -> Dict[str, Any]:
    """Restore a checkpoint in place and return its validated header.

    Validates the header *before* calling ``load_checkpoint``, not after: a bad or
    future-format header should fail fast, without first mutating the live model/optimizer
    with a multi-second tensor load whose result we'd have to discard anyway. ``read_header``
    only reads record 0 (no tensor data), so this costs nothing extra on the success path.
    """
    from ttml.checkpointing import load_checkpoint, read_header

    validate_header(read_header(str(path)))
    return load_checkpoint(str(path), model_params=model_params, optimizer=optimizer,
                           display_progress=display_progress)


def peek(path: Path) -> Dict[str, Any]:
    """Read a checkpoint's header without touching its tensors."""
    from ttml.checkpointing import read_header

    header = read_header(str(path))
    validate_header(header)
    return header
