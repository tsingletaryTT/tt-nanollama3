# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""vLLM entrypoint for tt-tnt — stock Llama, plus two runtime patches.

WHAT THIS IS
------------
This module is the ``main_class`` the Tenstorrent vLLM plugin imports for this model
(``entrypoint.class`` in ``tt_kernel_manifest.json``). It adds **no model code**:
tt-tnt is a standard HF Llama and the stock
``models.tt_transformers.tt.generator_vllm:LlamaForCausalLM`` computes it. This module
carries only what tt-metal gets wrong or defaults badly for a model this small:

1. a runtime **patch** to ``ModelArgs.find_grid`` (without it the model cannot run at all
   on a harvested Blackhole -- see "THE find_grid BUG" below),
2. a runtime **patch** to ``ModelArgs.weight_cache_path`` that scopes the converted-weight
   cache by a fingerprint of the *source* weights, so republishing a model under the same
   HF repo id cannot silently serve the previous weights (see "THE STALE-CACHE BUG"), and
3. a **precision default** of ``accuracy`` rather than ``performance`` (see
   ``DEFAULT_OPTIMIZATIONS``).

The point being demonstrated: **a model can carry the tt-metal change it needs, in its
own distribution bundle, without that change having to land upstream first.** All three
travel with the bundle, apply at import time in the serving process, and are inert
everywhere else.

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

THE find_grid BUG
-----------------
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

find_grid SCOPE AND SAFETY
--------------------------
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

THE STALE-CACHE BUG -- WHY WEIGHTS ARE FINGERPRINTED
----------------------------------------------------
``tt_transformers`` converts HF weights to ``.tensorbin`` once and reuses them forever,
keyed on the **HF repo id and nothing else**::

    model_config.py:577   self.CACHE_PATH = os.path.join("model_cache", HF_MODEL, self.device_name)
    model_config.py:3017  def weight_cache_path(self, dtype):   # -> CACHE_PATH / "tensor_cache_bfp8"
    core.py:719           if not cache_path.exists() or not cache_path.is_file():  # convert
                          ...                                                      # else: load it

That reuse decision (``ttnn/ttnn/operations/core.py:719``) is a bare *existence* check.
There is no revision, no content hash, no comparison against the source weights. Only a
deserialisation failure (``core.py:725``) ever triggers a re-conversion.

So a project that iterates checkpoints under a **stable repo id** -- which is exactly what
tt-tnt does -- gets the old model's weights under the new model's config, logged as an
ordinary warm start (``Loaded cache for model_cache/episod/tt-tnt/P150/...``). Observed
live: a retrained tt-tnt was republished to ``episod/tt-tnt``, the server came up clean,
reported the correct ``max_model_len: 2048``, and ran the **previous** model. The previous
model could not emit EOS by construction, so the headline measurement would have been a
confident "0% termination" and would have read as a real regression. It was caught only
because a human noticed the cache directory's mtime predated the publish.

**It does not fail. It lies.** That is the failure mode this patch exists to remove.

THE FIX
-------
``weight_cache_path`` is the single funnel every weight cache path flows through (all of
``attention.py``, ``mlp.py``, ``lm_head.py``, ``embedding.py``, and the vLLM KV-cache
allocator reach the cache only via ``model_args.weight_cache_path(dtype)`` /
``Generator.cache_path``). The patch appends **one directory component** derived from the
source weights::

    model_cache/episod/tt-tnt/P150/tensor_cache_bfp8/src-rev-a3c85ec799fe/...

The fingerprint is, in order of preference:

1. ``hf_config._commit_hash`` -- the HF commit sha. ``transformers`` records it on the
   config object when it resolves ``config.json`` out of the Hub cache
   (``configuration_utils.py:812``, via ``extract_commit_hash``), and ``ModelArgs.__init__``
   has already loaded that config (``model_config.py:616``) before any cache path is asked
   for. This is the authoritative answer and the one upstream should use.
2. For a **local** checkpoint directory (``HF_MODEL=/some/path``), where there is no commit
   sha: a sha256 over ``(name, size, mtime_ns)`` of ``config.json`` and every weight file.
   Sizes alone are useless here -- a retrain of the same architecture produces byte-identical
   file *sizes*, which is precisely today's bug -- so mtime carries the signal. That means a
   re-download of unchanged weights produces a *false miss*: one wasted conversion, loudly
   logged. False misses cost minutes; false hits cost a wrong published measurement.
3. Nothing. Then the path is left exactly as stock, and a **warning** is logged saying the
   guard is not in place.

WHY THIS SHAPE AND NOT ANOTHER
------------------------------
- *Validate the cache before use and re-convert on mismatch.* To know a cached tensor is
  wrong you must produce the right one -- i.e. pay the conversion the cache exists to avoid
  -- unless you store a side manifest, which is this fix with more moving parts and a
  weaker invariant. It also **overwrites**, so flipping back to the previous revision pays
  conversion again. Fingerprinting keeps every revision warm.
- *Refuse a cache older than the source weights (mtime).* The source here is a Hub repo id;
  there is no local file whose mtime means "publish time". HF blob mtimes are *download*
  times, and when the download and the conversion happen in one session their order is
  arbitrary -- the comparison can silently come out the wrong way. Reading a timestamp is
  what the human had to do today; it is not a thing to automate.
- *Disable the cache.* Correct, but it pays full conversion on every serve, so it will be
  switched back off the first busy afternoon -- restoring the bug. A fix that gets disabled
  is not a fix.
- Fingerprinting **self-heals**: a republish simply misses and re-converts. It never serves
  the wrong weights, and it is not silent in the other direction either -- see below.

NOT SILENT IN EITHER DIRECTION
------------------------------
The bug being fixed is silence, so the fix must not introduce a different silence:

- New fingerprint where sibling fingerprints already exist -> **WARNING** naming the old
  ones ("the source weights changed"). That is the sentence that was missing from the log.
- Un-fingerprinted ``.tensorbin`` files left over from before this patch -> **WARNING**
  that they are now unused (they are *not* deleted; that is a human's call).
- Fingerprint unavailable, or ``TT_TNT_CACHE_FINGERPRINT=0`` -> **WARNING** that the guard
  is off and stale weights are again possible.
- ``ModelArgs.weight_cache_path`` missing, or its first two parameters are no longer
  ``(self, dtype)`` -> the patch **declines to install** and says so at WARNING level,
  rather than crashing the serve or pretending it applied.
- Every fingerprinted directory gets a ``tt_tnt_cache_source.json`` stamp recording what it
  was built from, so ``ls`` answers the question that cost an afternoon.

A log line that is emitted but not *printed* is not a log line, so the module logger is
named under ``vllm`` -- see the comment on ``logger`` below. Without that, only the warnings
above reached the terminal and every confirming INFO line was dropped by the root logger.

weight_cache_path SCOPE AND SAFETY
----------------------------------
- Wraps one method; the returned path is the stock path plus one component, so every
  existing caller, ``mkdir(parents=True)`` included, keeps working unchanged.
- The vLLM empty-KV-cache tensors also live under ``cache_path``, so they are re-created
  per revision too. They are zeros -- cheap to regenerate, and the disk cost is bounded by
  the number of revisions actually served. Old revision directories are retained, not
  reclaimed; delete them by hand.
- Idempotent, reversible via ``restore_patches()``, and affects only the importing process.

Like ``find_grid`` this is a **shim, not a fix**. The real fix belongs upstream: the cache
key at ``model_config.py:577`` should include the resolved revision. When that lands,
``_patch_weight_cache_path`` becomes a no-op that can be deleted.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from models.tt_transformers.tt.model_config import ModelArgs

#: Named **under the ``vllm`` logger** on purpose, and this is load-bearing rather than
#: cosmetic. vLLM configures exactly one logger -- ``vllm`` -- with a stream handler at
#: ``VLLM_LOGGING_LEVEL`` (INFO by default) and ``propagate: False``
#: (``vllm/logger.py:41-71``), and leaves the **root** logger at its stock WARNING. A bare
#: ``getLogger(__name__)`` therefore propagates to a root that drops every INFO record, so
#: in a real serve only this module's *warnings* were ever visible: the lines that say the
#: guard is installed, that a cache is being reused, and that weights are being converted
#: for the first time all vanished. That is half of the "not silent in either direction"
#: guarantee below silently missing -- the reassuring half, which is the half you read when
#: deciding whether to trust a measurement. Every vLLM module gets its logging solely by
#: being named ``vllm.*`` (``init_logger`` is just ``getLogger(name)``, ``logger.py:204``),
#: so this is the mechanism upstream uses, not a trick played on it. Outside a vLLM process
#: the name is an ordinary unconfigured logger and behaves exactly as before.
logger = logging.getLogger(f"vllm.{__name__}")

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
        logger.warning("tt-tnt: could not read device compute grid (%r); "
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
                        "tt-tnt: find_grid(%d) -> rows=%d cols=%d "
                        "(real device grid %dx%d)", N, rows, cols, max_cols, max_rows
                    )
                    return rows, cols

    raise AssertionError(
        f"Cannot find a grid configuration for {N} tiles that evenly divides into "
        f"{max_cores} cores of max size {max_rows}x{max_cols} (real device grid). "
        f"This is tt-tnt's find_grid shim, not stock tt-metal."
    )


def _patch_find_grid():
    """Install the shim. Idempotent; safe to call more than once."""
    global _ORIGINAL_FIND_GRID
    if _ORIGINAL_FIND_GRID is not None:
        return
    _ORIGINAL_FIND_GRID = ModelArgs.find_grid
    ModelArgs.find_grid = _find_grid_from_device
    logger.info(
        "tt-tnt: patched ModelArgs.find_grid to read the device's real compute "
        "grid (works around hardcoded max_cols=12 on harvested Blackhole)."
    )


# ---------------------------------------------------------------------------
# Patch 2 -- scope the converted-weight cache by a fingerprint of the source
# weights, so a republish under the same repo id cannot be served from cache.
# See "THE STALE-CACHE BUG" in the module docstring for why this exists.
# ---------------------------------------------------------------------------

#: Set once the patch is installed, holding the original unbound method.
_ORIGINAL_WEIGHT_CACHE_PATH = None

#: Written into every fingerprinted cache directory so the directory can say what it was
#: built from. The whole point of this patch is that ``ls`` should answer the question that
#: previously required correlating a directory mtime against a Hub publish time.
CACHE_STAMP_NAME = "tt_tnt_cache_source.json"

#: Files whose identity defines "the model" for a *local* checkpoint directory. ``config.json``
#: is included so an architecture change is caught even if the weight files are untouched.
_LOCAL_SOURCE_GLOBS = ("config.json", "*.safetensors", "*.bin", "*.pth", "*.pt")

#: A commit sha as ``transformers`` records it. Anything not shaped like one (a branch name,
#: a sentinel) is rejected rather than baked into a path, and we fall through to the next probe.
_SHA_RE = re.compile(r"[0-9a-fA-F]{7,}\Z")

#: Warnings that must be said once, not once per weight tensor (there are hundreds).
_WARNED_ONCE = set()

#: Cache scopes already reported, keyed by resolved path -- same reason.
_ANNOUNCED_SCOPES = set()


def _warn_once(message, *args):
    """``logger.warning`` de-duplicated by format string.

    ``weight_cache_path`` is called once per weight tensor. A warning that repeated that
    often would be scrolled past, which is the same failure as not logging it at all.
    """
    if message in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(message)
    logger.warning(message, *args)


def fingerprinting_enabled():
    """Whether the guard is active. ``TT_TNT_CACHE_FINGERPRINT=0`` turns it off.

    Read at call time rather than import time so it can be flipped without a reimport.
    Turning it off is legitimate (e.g. to reproduce a run against a known cache) but is
    never silent -- see ``_weight_cache_path_fingerprinted``.
    """
    return os.environ.get("TT_TNT_CACHE_FINGERPRINT", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )


def _commit_fingerprint(model_args):
    """``("rev", <sha12>)`` from the HF commit sha, or None if there isn't one.

    ``transformers`` stamps ``_commit_hash`` onto the config object when it resolves
    ``config.json`` out of the Hub cache, and ``ModelArgs.__init__`` loads that config
    before anything asks for a cache path. This is the authoritative source identity.
    """
    hf_config = getattr(model_args, "hf_config", None)
    if hf_config is None:
        return None
    sha = getattr(hf_config, "_commit_hash", None)
    if isinstance(sha, str) and _SHA_RE.match(sha):
        return "rev", sha[:12].lower()
    return None


def _local_fingerprint(model_args):
    """``("files", <hash12>)`` over a local checkpoint directory, or None.

    Covers ``HF_MODEL=/some/path``, where there is no commit sha to lean on. ``mtime_ns`` is
    part of the digest deliberately: a retrained model of the same architecture has
    byte-identical file *sizes*, so size alone would reproduce the very bug being fixed.
    The cost is a false miss (one redundant conversion, logged) after a re-download.
    """
    ckpt_dir = getattr(model_args, "CKPT_DIR", None)
    if not isinstance(ckpt_dir, str) or not ckpt_dir:
        return None
    root = Path(ckpt_dir)
    if not root.is_dir():
        return None

    entries = []
    for pattern in _LOCAL_SOURCE_GLOBS:
        for path in root.glob(pattern):
            try:
                stat = path.stat()
            except OSError:  # pragma: no cover - vanished between glob and stat
                continue
            entries.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
    if not entries:
        return None

    digest = hashlib.sha256("\n".join(sorted(entries)).encode("utf-8")).hexdigest()
    return "files", digest[:12]


def source_fingerprint(model_args):
    """``(kind, value)`` identifying the weights behind ``model_args``, or None.

    None means "cannot tell" and is treated as a loud degradation, never as "assume fresh".
    A probe that raises is downgraded to None rather than being allowed to kill the serve:
    a wrong-but-loud cache path is worse than no serve, but a crash here would take out a
    server that stock tt-metal would have started.
    """
    for probe in (_commit_fingerprint, _local_fingerprint):
        try:
            result = probe(model_args)
        except Exception as exc:  # pragma: no cover - diagnostic path
            _warn_once(
                "tt-tnt: source fingerprint probe %s failed (%r); trying the next one.",
                probe.__name__,
                exc,
            )
            continue
        if result is not None:
            return result
    return None


def _write_cache_stamp(scoped, kind, value, source):
    """Record what a fingerprinted cache directory was built from.

    Best-effort by design: a read-only or full filesystem must degrade to a debug line, not
    to a failed serve. An existing stamp is left alone so it keeps its original timestamp.
    """
    stamp = scoped / CACHE_STAMP_NAME
    try:
        if stamp.exists():
            return
        scoped.mkdir(parents=True, exist_ok=True)
        stamp.write_text(
            json.dumps(
                {
                    "source": source,
                    "fingerprint_kind": kind,
                    "fingerprint": value,
                    "written_by": "tt_tnt_adapter",
                    "written_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError as exc:  # pragma: no cover - diagnostic path
        logger.debug("tt-tnt: could not write %s (%r)", stamp, exc)


def _announce_cache_scope(base, scoped, kind, value, source):
    """Say, exactly once per scope, whether this is a warm start or a changed model.

    This is the log line whose absence made the original bug invisible.
    """
    key = str(scoped)
    if key in _ANNOUNCED_SCOPES:
        return
    _ANNOUNCED_SCOPES.add(key)

    try:
        already_cached = scoped.is_dir() and any(scoped.glob("*.tensorbin"))
        siblings = (
            sorted(p.name for p in base.glob("src-*") if p.is_dir() and p != scoped)
            if base.is_dir()
            else []
        )
        legacy = sorted(p.name for p in base.glob("*.tensorbin")) if base.is_dir() else []
    except OSError as exc:  # pragma: no cover - diagnostic path
        logger.debug("tt-tnt: could not inspect %s (%r)", base, exc)
        return

    if already_cached:
        logger.info(
            "tt-tnt: reusing the converted-weight cache for %s (%s %s) at %s",
            source, kind, value, scoped,
        )
    elif siblings:
        # The whole point. Stock tt-metal logs this case as an ordinary cache hit.
        logger.warning(
            "tt-tnt: the source weights for %s changed -- no converted-weight cache for "
            "%s %s. Converting fresh weights. Previously cached revisions of the same repo "
            "id are still present (%s) and are NOT being used; delete them by hand if you "
            "want the space back.",
            source, kind, value, ", ".join(siblings),
        )
    else:
        logger.info(
            "tt-tnt: first conversion of %s (%s %s) -> %s", source, kind, value, scoped
        )

    if legacy:
        _warn_once(
            "tt-tnt: %s holds un-fingerprinted .tensorbin files from before this guard was "
            "installed. They are no longer read (that is the fix) and nothing here deletes "
            "them; remove them by hand when you are sure.",
            str(base),
        )

    _write_cache_stamp(scoped, kind, value, source)


def _weight_cache_path_fingerprinted(self, dtype, *args, **kwargs):
    """``ModelArgs.weight_cache_path`` with a source-fingerprint component appended.

    Every path that cannot produce a fingerprint falls back to the stock path *and says so*,
    so "the guard is not protecting you" is never a silent state.
    """
    base = _ORIGINAL_WEIGHT_CACHE_PATH(self, dtype, *args, **kwargs)
    source = getattr(self, "CKPT_DIR", "<unknown>")

    if not fingerprinting_enabled():
        _warn_once(
            "tt-tnt: TT_TNT_CACHE_FINGERPRINT is off -- the converted-weight cache is keyed "
            "on the repo id alone, so republished weights can be served from a stale cache."
        )
        return base

    fingerprint = source_fingerprint(self)
    if fingerprint is None:
        _warn_once(
            "tt-tnt: could not fingerprint the weights behind %r (no HF commit sha on the "
            "config and no readable local checkpoint directory). Falling back to the stock "
            "repo-id-keyed cache path -- if these weights were republished under an existing "
            "repo id, the PREVIOUS model may be served. Verify before trusting any output.",
            source,
        )
        return base

    kind, value = fingerprint
    try:
        base_path = Path(base)
        scoped = base_path / f"src-{kind}-{value}"
    except TypeError:  # pragma: no cover - upstream returned something un-path-like
        _warn_once(
            "tt-tnt: ModelArgs.weight_cache_path returned %r, which is not path-like; the "
            "stale-cache guard cannot scope it. Serving with the stock cache path.",
            base,
        )
        return base

    _announce_cache_scope(base_path, scoped, kind, value, source)
    return scoped


def _patch_weight_cache_path():
    """Install the cache-scoping shim. Idempotent; declines loudly if upstream has moved."""
    global _ORIGINAL_WEIGHT_CACHE_PATH
    if _ORIGINAL_WEIGHT_CACHE_PATH is not None:
        return

    original = getattr(ModelArgs, "weight_cache_path", None)
    if not callable(original):
        logger.warning(
            "tt-tnt: ModelArgs.weight_cache_path is missing or not callable (%r). The "
            "stale-weight-cache guard is NOT installed -- republished weights may be served "
            "from a stale cache. tt_transformers has moved; this shim needs updating.",
            original,
        )
        return

    # Shape check: we forward *args/**kwargs, so extra parameters are fine, but the first
    # two must still be (self, dtype) or we would be scoping something else entirely.
    try:
        params = list(inspect.signature(original).parameters)
    except (TypeError, ValueError):  # pragma: no cover - unintrospectable callable
        params = None
    if params is not None and params[:2] != ["self", "dtype"]:
        logger.warning(
            "tt-tnt: ModelArgs.weight_cache_path has an unexpected signature (%s). The "
            "stale-weight-cache guard is NOT installed -- republished weights may be served "
            "from a stale cache. tt_transformers has changed; this shim needs updating.",
            params,
        )
        return

    _ORIGINAL_WEIGHT_CACHE_PATH = original
    ModelArgs.weight_cache_path = _weight_cache_path_fingerprinted
    logger.info(
        "tt-tnt: scoped the tt_transformers weight cache by source fingerprint "
        "(works around a cache keyed on the HF repo id alone, which silently serves the "
        "previous model after a republish)."
    )


def restore_patches():
    """Undo the patches. Provided so a host can leave the process as it found it."""
    global _ORIGINAL_FIND_GRID, _ORIGINAL_WEIGHT_CACHE_PATH
    if _ORIGINAL_FIND_GRID is not None:
        ModelArgs.find_grid = _ORIGINAL_FIND_GRID
        _ORIGINAL_FIND_GRID = None
    if _ORIGINAL_WEIGHT_CACHE_PATH is not None:
        ModelArgs.weight_cache_path = _ORIGINAL_WEIGHT_CACHE_PATH
        _ORIGINAL_WEIGHT_CACHE_PATH = None


# Applied at import time: the plugin imports this module to resolve ``main_class``, which
# happens before any ModelArgs is constructed, so the shims are in place before first use.


# ---------------------------------------------------------------------------
# Check 3 -- refuse to serve quietly on a plugin that predates the decode fix.
#
# THE PROBLEM THIS SOLVES
# On vllm-tt-plugin builds older than c127c17, free-running decode degrades into
# repetition within a few tokens: measured local-repeat rate 0.222 against a CPU
# reference of 0.000, median agreement with CPU 4 tokens instead of 12. The cause
# was upstream, in `fix: return None from sample_tokens when no pending forward`
# (#26) and the device-state-slot fixes around it. See
# docs/measurements/decode-defect-resolved.json.
#
# WHY NOT A VERSION PIN
# tt-kernel manifests can carry a ``runtime.plugin_version`` range, and
# tt_kernel/metal.py resolves the installed plugin version for exactly that. It
# does not help here: vllm-tt-plugin has reported "0.1.0" for its entire history
# -- 0.0.0 -> 0.1.0 once, at b4325e0, and never since. The broken build and the
# fixed build report the same string, so a version range would be false comfort.
#
# WHAT IS ACTUALLY DETECTABLE
# `src/vllm_tt_plugin/engine.py` (798 lines) was deleted between the stale build
# and the fixed one. Its presence is a structural marker of a plugin that
# predates the fix, and needs no version string. This is a proxy rather than a
# direct test of the defect -- if upstream reinstates that module the check will
# report a false positive, which is why it warns rather than refuses.
# ---------------------------------------------------------------------------

#: Module deleted upstream between the last plugin that showed the decode defect
#: and the first that did not.
_STALE_PLUGIN_MARKER = "vllm_tt_plugin.engine"


def plugin_predates_decode_fix():
    """True when the installed plugin still carries the pre-fix module layout."""
    import importlib.util

    try:
        return importlib.util.find_spec(_STALE_PLUGIN_MARKER) is not None
    except (ImportError, ValueError):
        return False


def _check_plugin_freshness():
    if not plugin_predates_decode_fix():
        return
    _warn_once(
        "tt-tnt: this vllm-tt-plugin build appears to predate the decode fix "
        "(%s is still present, and it was removed upstream before c127c17). "
        "Free-running decode on such builds degrades into repetition within a "
        "few tokens -- measured local-repeat 0.222 against a CPU reference of "
        "0.000. Generation will look broken and the model is not at fault. "
        "Update the plugin. Note that its reported version is 0.1.0 either way, "
        "so a version check cannot tell you this.",
        _STALE_PLUGIN_MARKER,
    )


_patch_find_grid()
_patch_weight_cache_path()
_check_plugin_freshness()

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
#: **This helps measurably, and does not fix the defect.** Measured with
#: ``scripts/free_running_check.py`` over six prompts, 40 greedy tokens each, comparing
#: device output against the CPU reference token by token:
#:
#:     accuracy    : median agreement 4/40  (min 1, max 7)
#:     performance : median agreement 3/40  (min 0, max 5)
#:
#: So the precision profile is worth roughly one token of agreement — real, reproducible,
#: and nowhere near sufficient. An earlier revision of this comment said greedy output was
#: "materially unchanged"; that was based on eyeballing a single prompt and was too strong.
#: Keep ``accuracy`` because it is both better measured and indefensible to serve a 384-dim
#: model's MLP at 4 bits — but do not cite it as the remedy for the defect below.
#:
#: Overridable via ``TT_TNT_OPTIMIZATIONS`` for A/B testing without a rebuild.
DEFAULT_OPTIMIZATIONS = os.environ.get("TT_TNT_OPTIMIZATIONS", "accuracy")


def _build_capabilities():
    """Capability flags, all off by default, opt-in via ``TT_TNT_CAPS``.

    Built here rather than in the class body so the opt-in loop cannot leak or fail on an
    empty environment variable.
    """
    caps = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": False,
    }
    for name in os.environ.get("TT_TNT_CAPS", "").split(","):
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
    #: for tt-tnt, so none is claimed. Re-enable individually, with evidence, via
    #: TT_TNT_CAPS (comma-separated names) once each is actually tested.
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
                "tt-tnt: using optimizations=%r (stock default is 'performance', "
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


__all__ = [
    "LlamaForCausalLM",
    "restore_patches",
    "DEFAULT_OPTIMIZATIONS",
    "source_fingerprint",
    "fingerprinting_enabled",
    "CACHE_STAMP_NAME",
]
