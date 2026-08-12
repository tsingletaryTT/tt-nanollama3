# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""vLLM entrypoint for tt-nanollama3 — stock Llama, plus one runtime patch.

WHAT THIS IS
------------
This module is the ``main_class`` the Tenstorrent vLLM plugin imports for this model
(``entrypoint.class`` in ``tt_kernel_manifest.json``). It adds **no model code**:
tt-nanollama3 is a standard HF Llama and the stock
``models.tt_transformers.tt.generator_vllm:LlamaForCausalLM`` computes it. This module
carries only what tt-metal gets wrong or defaults badly for a model this small:

1. a runtime **patch** to ``ModelArgs.find_grid`` (without it the model cannot run at all
   on a harvested Blackhole -- see below), and
2. a **precision default** of ``accuracy`` rather than ``performance`` (see
   ``DEFAULT_OPTIMIZATIONS``).

The point being demonstrated: **a model can carry the tt-metal change it needs, in its
own distribution bundle, without that change having to land upstream first.** Both travel
with the bundle, apply at import time in the serving process, and are inert everywhere
else.

KNOWN UNFIXED DEFECT -- READ BEFORE TRUSTING OUTPUT
---------------------------------------------------
Generation on the served path is **wrong**, and neither item above fixes it. Greedy
decoding collapses into a repetition loop (" girl named Lily. Lily. Lily. Lily.") where
the same weights on CPU produce coherent prose (" girl named Lily. She loved to play with
her toys...").

Localised, with precision held at ``accuracy``: given the **same** 13-token context, a
fresh prefill returns ``' She'`` (matching CPU) while arriving at that identical context
through 4 decode steps returns ``' Lily'``. Same server, same context, different answer
depending on the path taken -- so the error is in the **decode / KV-cache path**, not the
weights, the conversion, the ``find_grid`` patch, or precision.

Supporting measurement: teacher-forced (re-prefilling each prefix) per-step top-1
agreement with CPU is 23/25 = 92%, and every disagreement sits on a near-tie -- mean logit
gap 0.093 where they differ versus 2.299 where they agree. Prefill is sound.

Note that the tt_transformers PCC gate passed at 0.9940-0.9998 while this defect was
present: it exercises prefill far harder than long decode. **A green PCC is not evidence
of correct generation.**

THE BUG BEING PATCHED
---------------------
``ModelArgs.find_grid`` (``models/tt_transformers/tt/model_config.py:3205``) picks a core
grid from hardcoded per-architecture constants::

    max_rows = 8 if is_wormhole_b0() else 10
    max_cols = 8 if is_wormhole_b0() else 12      # 12 assumed for every Blackhole

It never asks the device how large its compute grid actually is. A **harvested** Blackhole
has fewer usable columns than the architectural maximum -- the p300c this was developed on
reports ``11x10``, not ``12x10``. For ``hidden_size=384`` (384/32 = 12 tiles) find_grid
returns ``(rows=1, cols=12)``, and RMSNorm then fails at the first decoder layer::

    TT_FATAL shard_spec_validation.cpp:34: device_range.contains(program_range)
    program_config grid (12x1) must be contained within device grid (11x10)

Given the *real* width the same search finds ``rows=2, cols=6``, which fits. So 384 is a
perfectly good dimension and the model is fine; only the helper's assumption is wrong.

Measured on this hardware, with the patch applied:
``models/tt_transformers/tests/test_model.py -k "full and performance"`` -> **2 passed**
(performance and accuracy), PCC **0.9940 - 0.9998** across 18 measurements, on
``blackhole``, mesh ``(1,1)``, seq 256, batch 1, paged attention.

SCOPE AND SAFETY
----------------
- Patches exactly one method, and only its ``max_rows``/``max_cols`` source. The search
  order, the "closest to 32 cores" heuristic, and the failure assertion are unchanged.
- **Falls back to the original implementation** whenever the device cannot be queried, so
  a mesh-less or unusual context behaves exactly as stock tt-metal does.
- Idempotent: re-importing will not double-wrap.
- Records the original on the module so a host can restore it (``restore_patches()``).
- Affects only the process that imports this module -- the vLLM worker. Nothing is written
  to the tt-metal installation on disk.

This is a **compatibility shim, not a fix.** The real fix belongs upstream: ``find_grid``
should read ``compute_with_storage_grid_size()`` rather than hardcoding per-arch constants.
When that lands in a released tt-metal, ``_patch_find_grid`` becomes a no-op that can be
deleted along with the ``platform.ttnn`` floor that pins this bundle below it.
"""

from __future__ import annotations

import logging
import os

from models.tt_transformers.tt.model_config import ModelArgs

logger = logging.getLogger(__name__)

#: Set once the patch is installed, holding the original unbound method so the
#: installation is reversible and detectably idempotent.
_ORIGINAL_FIND_GRID = None


def _device_grid(model_args):
    """(max_rows, max_cols) from the live mesh device, or None if unavailable.

    Returning None is the signal to defer to stock behaviour rather than guess.
    """
    mesh = getattr(model_args, "mesh_device", None)
    if mesh is None:
        return None
    try:
        grid = mesh.compute_with_storage_grid_size()
        return grid.y, grid.x
    except Exception as exc:  # pragma: no cover - diagnostic path
        logger.warning("tt-nanollama3: could not read device compute grid (%r); "
                       "deferring to stock find_grid", exc)
        return None


def _find_grid_from_device(self, N):
    """``ModelArgs.find_grid`` bounded by the device's real compute grid.

    Mirrors the upstream implementation exactly except for where ``max_rows``/``max_cols``
    come from. Any inability to read the device falls through to the original.
    """
    dims = _device_grid(self)
    if dims is None:
        return _ORIGINAL_FIND_GRID(self, N)

    max_rows, max_cols = dims
    max_cores = max_rows * max_cols

    target = 32
    possible_cores = [k for k in range(1, max_cores + 1) if N % k == 0]
    possible_cores.sort(key=lambda x: abs(x - target))

    for cores in possible_cores:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    logger.debug(
                        "tt-nanollama3: find_grid(%d) -> rows=%d cols=%d "
                        "(real device grid %dx%d)", N, rows, cols, max_cols, max_rows
                    )
                    return rows, cols

    raise AssertionError(
        f"Cannot find a grid configuration for {N} tiles that evenly divides into "
        f"{max_cores} cores of max size {max_rows}x{max_cols} (real device grid). "
        f"This is tt-nanollama3's find_grid shim, not stock tt-metal."
    )


def _patch_find_grid():
    """Install the shim. Idempotent; safe to call more than once."""
    global _ORIGINAL_FIND_GRID
    if _ORIGINAL_FIND_GRID is not None:
        return
    _ORIGINAL_FIND_GRID = ModelArgs.find_grid
    ModelArgs.find_grid = _find_grid_from_device
    logger.info(
        "tt-nanollama3: patched ModelArgs.find_grid to read the device's real compute "
        "grid (works around hardcoded max_cols=12 on harvested Blackhole)."
    )


def restore_patches():
    """Undo the patch. Provided so a host can leave the process as it found it."""
    global _ORIGINAL_FIND_GRID
    if _ORIGINAL_FIND_GRID is None:
        return
    ModelArgs.find_grid = _ORIGINAL_FIND_GRID
    _ORIGINAL_FIND_GRID = None


# Applied at import time: the plugin imports this module to resolve ``main_class``, which
# happens before any ModelArgs is constructed, so the shim is in place before first use.
_patch_find_grid()

from models.tt_transformers.tt.generator_vllm import (  # noqa: E402
    LlamaForCausalLM as _StockLlamaForCausalLM,
)

#: Precision profile. ``DecodersPrecision.performance`` -- the stock default -- serves the
#: MLP ``w1``/``w3`` projections as **BFLOAT4_B** and ``wqkv``/``wo``/``w2`` as BFLOAT8_B.
#: Those defaults are tuned for 8B-70B models, where 4-bit MLP weights are survivable
#: because there is enormous redundancy to absorb the error. This model has
#: ``hidden_size=384`` and ~22M parameters; there is no such headroom.
#:
#: Switching to ``accuracy`` moves ``wqkv``/``wo`` to BFLOAT16 and ``w1``/``w3`` from
#: BFLOAT4_B to BFLOAT8_B (verified in the serving log's tensor-cache dtypes).
#:
#: **This is a precision decision on its own merits, NOT a bug fix.** It was tried as a
#: hypothesis for the repetition-loop defect described below and **did not fix it** --
#: greedy output was materially unchanged under ``accuracy``. It is kept because serving a
#: 384-dim model's MLP at 4 bits is indefensible regardless, not because it repaired
#: anything. Do not cite it as the remedy for that defect.
#:
#: Overridable via ``TT_NANOLLAMA3_OPTIMIZATIONS`` for A/B testing without a rebuild.
DEFAULT_OPTIMIZATIONS = os.environ.get("TT_NANOLLAMA3_OPTIMIZATIONS", "accuracy")


def _build_capabilities():
    """Capability flags, all off by default, opt-in via ``TT_NANOLLAMA3_CAPS``.

    Built here rather than in the class body so the opt-in loop cannot leak or fail on an
    empty environment variable.
    """
    caps = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": False,
    }
    for name in os.environ.get("TT_NANOLLAMA3_CAPS", "").split(","):
        name = name.strip()
        if name:
            caps[name] = True
    return caps


_CAPABILITIES = _build_capabilities()


class LlamaForCausalLM(_StockLlamaForCausalLM):
    """Stock Llama, with a precision default appropriate to a 22M-parameter model.

    The only behavioural change is ``optimizations``: the base class defaults it to
    ``"performance"``, this subclass to ``"accuracy"`` (see ``DEFAULT_OPTIMIZATIONS``).
    Everything else -- prefill, decode, KV cache allocation -- is inherited untouched.

    This is the second half of what this bundle demonstrates: a model can ship not only the
    tt-metal *patch* it needs but also the tt-metal *configuration* it needs, without either
    having to become an upstream default that would be wrong for larger models.
    """

    #: Capability flags, narrowed from the stock class's blanket ``True``.
    #:
    #: tt-metal's own vLLM-integration guidance (``.agents/skills/vllm-integration/SKILL.md``
    #: on the ``agentic-research/fast-models-fast`` branch) is explicit that these must be
    #: *proof-backed*, not assumed:
    #:
    #:   "When supports_async_decode=True, sampling on device, tracing is enabled,
    #:    reset_batch=False, vLLM may build and submit decode step N+1 before sampled token N
    #:    has been applied to host scheduler state, so the inputs may be stale or wrong."
    #:   "...letting it default on can silently corrupt generation."
    #:   "Leave prefix caching False unless it is implemented and tested."
    #:
    #: and prescribes validating with a degenerate-output check for "doubled subwords or
    #: repeated control tokens" -- which is precisely the failure observed on this model
    #: (" girl named Lily. Lily. Lily. Lily."). None of these capabilities has been proven
    #: for tt-nanollama3, so none is claimed. Re-enable individually, with evidence, via
    #: TT_NANOLLAMA3_CAPS (comma-separated names) once each is actually tested.
    model_capabilities = _CAPABILITIES

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str = None,
    ):
        # None means "caller expressed no preference" -- apply ours. An explicit value from
        # the caller still wins, so this is a default, not an override.
        if optimizations is None:
            optimizations = DEFAULT_OPTIMIZATIONS
            logger.info(
                "tt-nanollama3: using optimizations=%r (stock default is 'performance', "
                "which serves MLP w1/w3 at BFLOAT4_B -- too coarse for a 384-dim model).",
                optimizations,
            )
        return super().initialize_vllm_model(
            hf_config,
            mesh_device,
            max_batch_size,
            max_seq_len,
            n_layers=n_layers,
            tt_data_parallel=tt_data_parallel,
            optimizations=optimizations,
        )


__all__ = ["LlamaForCausalLM", "restore_patches", "DEFAULT_OPTIMIZATIONS"]
