# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""vLLM entrypoint for tt-nanollama3 — stock Llama, plus one runtime patch.

WHAT THIS IS
------------
This module is the ``main_class`` the Tenstorrent vLLM plugin imports for this model
(``entrypoint.class`` in ``tt_kernel_manifest.json``). It deliberately adds **no model
code**: tt-nanollama3 is a standard HF Llama and the stock
``models.tt_transformers.tt.generator_vllm:LlamaForCausalLM`` serves it correctly. All
this module does is apply one narrowly-scoped patch to tt-metal *before* re-exporting
that class, then get out of the way.

The point being demonstrated: **a model can carry the tt-metal change it needs, in its
own distribution bundle, without that change having to land upstream first.** The patch
travels with the bundle, applies at import time in the serving process, and is inert
everywhere else.

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

# Re-export the stock class unchanged. tt-nanollama3 needs no bespoke model code -- the
# only reason this module exists is the patch above.
from models.tt_transformers.tt.generator_vllm import LlamaForCausalLM  # noqa: E402

__all__ = ["LlamaForCausalLM", "restore_patches"]
