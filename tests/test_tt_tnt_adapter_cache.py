# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Gates for the stale-converted-weight-cache guard in ``bundle/tt_tnt_adapter.py``.

WHY THESE EXIST
---------------
``tt_transformers`` caches converted weights at ``model_cache/<repo_id>/<device>/…`` and
decides to reuse them with a bare existence check
(``ttnn/ttnn/operations/core.py:719``). The key contains no revision and no content hash.
Republishing new weights to an **existing** repo id therefore hits the same path and the
old model is served, logged as an ordinary warm start.

That happened: a retrained tt-tnt was published to ``episod/tt-tnt``, the server came up
clean, reported the right ``max_model_len``, and ran the *previous* model. Because that
model could not emit EOS, the headline measurement would have been a confident "0%
termination" — a real-looking regression, entirely fictional. It was caught by a human
noticing a directory mtime.

So the property under test is not "the adapter has a function"; it is **two different
sets of source weights must never share a cache path, and every state in which that
guarantee does not hold must be audible in the log.**

HOW THEY RUN WITHOUT HARDWARE (AND WITHOUT tt-metal)
-----------------------------------------------------
``bundle/tt_tnt_adapter.py`` imports ``models.tt_transformers`` at module scope, which
pulls in ``ttnn``. These tests install a **fake** ``models.tt_transformers`` into
``sys.modules`` first, so the adapter loads against a stand-in ``ModelArgs`` that mirrors
the upstream shape (``model_cache_path``, ``CKPT_DIR``, ``hf_config``,
``weight_cache_path(self, dtype)``). Nothing here imports ttnn, opens a device, or reads a
real checkpoint — and nothing here is skipped, so a revert of the patch fails the suite
rather than quietly skipping it.

``test_upstream_cache_anchors_still_present`` is the one gate that needs the real
tt-metal, and it reads the *source text* rather than importing it (again: no ttnn, no
device). It skips with a reason when the tree is not on this machine.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import logging
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ADAPTER_PATH = Path(__file__).resolve().parents[1] / "bundle" / "tt_tnt_adapter.py"

#: The adapter is loaded under a private name so it cannot collide with a real
#: ``tt_tnt_adapter`` that a serving environment might already have imported.
_MODULE_NAME = "tt_tnt_adapter_under_test"

#: The adapter's logger, which is deliberately a **child of ``vllm``** so that vLLM's
#: logging configuration governs it (see the ``logger`` comment in the adapter). Derived
#: here the same way the adapter derives it, rather than hardcoded, so that if the two ever
#: disagree the log-content gates below fail loudly instead of watching an empty logger.
_LOGGER_NAME = f"vllm.{_MODULE_NAME}"

#: The fake package tree the adapter's imports resolve against.
_FAKE_MODULES = (
    "models",
    "models.tt_transformers",
    "models.tt_transformers.tt",
    "models.tt_transformers.tt.model_config",
    "models.tt_transformers.tt.generator_vllm",
)

#: Stand-in for a ``ttnn`` dtype. The patch must never inspect the dtype — it forwards it
#: to the original method untouched — so a plain string is a sufficient (and revealing)
#: substitute: a patch that tried to interpret it would fail here.
BFP8 = "bfp8"


class _StockLlamaForCausalLM:
    """Stand-in for ``generator_vllm.LlamaForCausalLM`` — only needs to be subclassable."""

    @classmethod
    def initialize_vllm_model(cls, *args, **kwargs):  # pragma: no cover - never called here
        return None


def make_model_args_cls(*, with_weight_cache_path=True, signature="standard"):
    """A fresh ``ModelArgs`` stand-in mirroring the upstream attribute surface.

    A *fresh class per test* matters: the adapter patches the class object, so a shared
    class would leak one test's patch into the next.

    ``with_weight_cache_path``/``signature`` exist to drive the "upstream moved" paths,
    where the patch must decline to install and say so rather than crash or pretend.
    """

    class FakeModelArgs:
        def __init__(self, cache_root, ckpt_dir="episod/tt-tnt", commit_hash=None):
            self.model_cache_path = Path(cache_root)
            self.CKPT_DIR = ckpt_dir
            self.hf_config = SimpleNamespace(_commit_hash=commit_hash)

        def find_grid(self, N):  # patched by the other shim; present so it can be
            return ("stock", N)

    if with_weight_cache_path:
        if signature == "standard":

            def weight_cache_path(self, dtype):
                return self.model_cache_path / f"tensor_cache_{dtype}"

        elif signature == "renamed":
            # Upstream renamed the parameter — we must not assume it still means dtype.
            def weight_cache_path(self, precision):
                return self.model_cache_path / f"tensor_cache_{precision}"

        else:  # pragma: no cover - guarded by the caller
            raise ValueError(signature)

        FakeModelArgs.weight_cache_path = weight_cache_path

    return FakeModelArgs


@contextlib.contextmanager
def loaded_adapter(model_args_cls):
    """Import ``bundle/tt_tnt_adapter.py`` against a fake tt_transformers, then unwind.

    Yields the loaded module. ``sys.modules`` is restored exactly, including entries that
    were absent before, so a real tt-metal on this machine is neither used nor disturbed.
    """
    saved = {name: sys.modules.get(name) for name in (*_FAKE_MODULES, _MODULE_NAME)}

    models = types.ModuleType("models")
    models.__path__ = []
    tt_transformers = types.ModuleType("models.tt_transformers")
    tt_transformers.__path__ = []
    tt = types.ModuleType("models.tt_transformers.tt")
    tt.__path__ = []
    model_config = types.ModuleType("models.tt_transformers.tt.model_config")
    model_config.ModelArgs = model_args_cls
    generator_vllm = types.ModuleType("models.tt_transformers.tt.generator_vllm")
    generator_vllm.LlamaForCausalLM = _StockLlamaForCausalLM

    sys.modules["models"] = models
    sys.modules["models.tt_transformers"] = tt_transformers
    sys.modules["models.tt_transformers.tt"] = tt
    sys.modules["models.tt_transformers.tt.model_config"] = model_config
    sys.modules["models.tt_transformers.tt.generator_vllm"] = generator_vllm

    try:
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, ADAPTER_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_NAME] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


@pytest.fixture
def adapter():
    """The adapter loaded against a standard fake ``ModelArgs``."""
    cls = make_model_args_cls()
    with loaded_adapter(cls) as module:
        yield SimpleNamespace(module=module, ModelArgs=cls)


@pytest.fixture(autouse=True)
def _clean_fingerprint_env(monkeypatch):
    """The guard is on unless a test says otherwise, whatever the ambient environment."""
    monkeypatch.delenv("TT_TNT_CACHE_FINGERPRINT", raising=False)


def _local_checkpoint(root, *, config='{"model_type": "llama"}', weights=b"weights-v1"):
    """A minimal local checkpoint directory: a config and one safetensors file."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(config, encoding="utf-8")
    (root / "model.safetensors").write_bytes(weights)
    return root


# ---------------------------------------------------------------------------
# The core regression: two revisions of one repo id must not share a cache path
# ---------------------------------------------------------------------------


def test_two_revisions_of_one_repo_id_get_different_cache_paths(adapter, tmp_path):
    """The exact incident: ``episod/tt-tnt`` republished, same repo id, new weights.

    Under stock tt_transformers both revisions resolve to the same directory and the
    second serve loads the first one's tensors. Here they must diverge.
    """
    v1 = adapter.ModelArgs(tmp_path / "cache", commit_hash="6d24cf7ed533d3219992f82ffa6f39c6ff0fbf3c")
    v3 = adapter.ModelArgs(tmp_path / "cache", commit_hash="a3c85ec799fe5e35b0cffd754b59b20cdb34866c")

    assert v1.weight_cache_path(BFP8) != v3.weight_cache_path(BFP8), (
        "two HF revisions of the same repo id share a converted-weight cache path — "
        "a republish will be served from the previous model's tensors"
    )


def test_cache_path_carries_the_commit_sha(adapter, tmp_path):
    """The revision must be visible *in the path*, not merely hashed into it.

    The recovery from the incident depended on a human being able to look at a directory
    and tell what it was. An opaque key would have failed that.
    """
    args = adapter.ModelArgs(tmp_path / "cache", commit_hash="a3c85ec799fe5e35b0cffd754b59b20cdb34866c")
    path = args.weight_cache_path(BFP8)

    assert path.name == "src-rev-a3c85ec799fe"
    assert path.parent.name == "tensor_cache_bfp8", (
        "the stock path must be preserved as the parent; only one component is appended"
    )


def test_same_revision_is_a_warm_start(adapter, tmp_path):
    """Correctness must not cost the cache. An unchanged model reuses its directory."""
    sha = "a3c85ec799fe5e35b0cffd754b59b20cdb34866c"
    first = adapter.ModelArgs(tmp_path / "cache", commit_hash=sha).weight_cache_path(BFP8)
    second = adapter.ModelArgs(tmp_path / "cache", commit_hash=sha).weight_cache_path(BFP8)

    assert first == second


def test_dtypes_still_get_separate_caches(adapter, tmp_path):
    """Scoping must not collapse the bf16/bfp8 split the stock path already makes."""
    args = adapter.ModelArgs(tmp_path / "cache", commit_hash="a3c85ec799fe5e35b0cffd754b59b20cdb34866c")

    assert args.weight_cache_path("bfp8") != args.weight_cache_path("bf16")


# ---------------------------------------------------------------------------
# Local checkpoint directories, where there is no commit sha to lean on
# ---------------------------------------------------------------------------


def test_local_checkpoint_retrain_changes_the_cache_path(adapter, tmp_path):
    """A retrain of the same architecture writes files of identical *size*.

    That is why the local fingerprint includes ``mtime_ns``: a size-only key would
    reproduce the bug exactly, since ``model.safetensors`` is the same length before and
    after a retrain.
    """
    ckpt = _local_checkpoint(tmp_path / "ckpt", weights=b"weights-v1")
    args = adapter.ModelArgs(tmp_path / "cache", ckpt_dir=str(ckpt))
    before = args.weight_cache_path(BFP8)

    # Same length, different content and mtime — a retrain.
    (ckpt / "model.safetensors").write_bytes(b"weights-v2")
    os.utime(ckpt / "model.safetensors", ns=(2_000_000_000_000_000_000, 2_000_000_000_000_000_000))
    after = adapter.ModelArgs(tmp_path / "cache", ckpt_dir=str(ckpt)).weight_cache_path(BFP8)

    assert len(b"weights-v1") == len(b"weights-v2"), "the premise of this test"
    assert before != after, (
        "a retrained local checkpoint of identical file size reused its cache path"
    )


def test_local_checkpoint_config_change_changes_the_cache_path(adapter, tmp_path):
    """An architecture edit alone must invalidate too — config.json is in the digest."""
    ckpt = _local_checkpoint(tmp_path / "ckpt")
    before = adapter.ModelArgs(tmp_path / "cache", ckpt_dir=str(ckpt)).weight_cache_path(BFP8)

    (ckpt / "config.json").write_text('{"model_type": "llama", "hidden_size": 768}', encoding="utf-8")
    after = adapter.ModelArgs(tmp_path / "cache", ckpt_dir=str(ckpt)).weight_cache_path(BFP8)

    assert before != after


def test_untouched_local_checkpoint_is_a_warm_start(adapter, tmp_path):
    ckpt = _local_checkpoint(tmp_path / "ckpt")
    first = adapter.ModelArgs(tmp_path / "cache", ckpt_dir=str(ckpt)).weight_cache_path(BFP8)
    second = adapter.ModelArgs(tmp_path / "cache", ckpt_dir=str(ckpt)).weight_cache_path(BFP8)

    assert first == second


def test_commit_sha_wins_over_the_local_scan(adapter, tmp_path):
    """When both are available the revision is authoritative — and cheaper."""
    ckpt = _local_checkpoint(tmp_path / "ckpt")
    args = adapter.ModelArgs(
        tmp_path / "cache", ckpt_dir=str(ckpt), commit_hash="a3c85ec799fe5e35b0cffd754b59b20cdb34866c"
    )

    assert args.weight_cache_path(BFP8).name == "src-rev-a3c85ec799fe"


def test_non_sha_revision_is_rejected_rather_than_baked_into_a_path(adapter, tmp_path):
    """``_commit_hash`` set to a branch name is not a content identity.

    Falling through to the local scan (or to the loud no-fingerprint path) is right;
    treating "main" as a revision would produce a key that never changes — the bug again.
    """
    ckpt = _local_checkpoint(tmp_path / "ckpt")
    args = adapter.ModelArgs(tmp_path / "cache", ckpt_dir=str(ckpt), commit_hash="main")

    assert args.weight_cache_path(BFP8).name.startswith("src-files-")


# ---------------------------------------------------------------------------
# Not silent in the other direction
# ---------------------------------------------------------------------------


def test_changed_source_is_a_warning_not_an_ordinary_cache_miss(adapter, tmp_path, caplog):
    """The log line whose absence made the original bug invisible.

    A first-ever conversion is unremarkable. A conversion that happens *next to* an
    existing cache for the same repo id means the model changed underneath us, and must
    be reported at WARNING with the old revision named.
    """
    cache = tmp_path / "cache"
    old = adapter.ModelArgs(cache, commit_hash="6d24cf7ed533d3219992f82ffa6f39c6ff0fbf3c")
    old_path = old.weight_cache_path(BFP8)
    old_path.mkdir(parents=True, exist_ok=True)
    (old_path / "tok_embeddings.weight_dtype_BFLOAT8_B_layout_TILE.tensorbin").write_bytes(b"x")

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        adapter.ModelArgs(cache, commit_hash="a3c85ec799fe5e35b0cffd754b59b20cdb34866c").weight_cache_path(BFP8)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "a republished model converted with no warning at all"
    text = "\n".join(r.getMessage() for r in warnings)
    assert "changed" in text
    assert "src-rev-6d24cf7ed533" in text, "the superseded revision must be named"


def test_first_conversion_does_not_cry_wolf(adapter, tmp_path, caplog):
    """A cold cache is normal. Warning here would train people to ignore the warning."""
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        adapter.ModelArgs(
            tmp_path / "cache", commit_hash="a3c85ec799fe5e35b0cffd754b59b20cdb34866c"
        ).weight_cache_path(BFP8)

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_confirming_lines_survive_vllms_real_logging_config(adapter, tmp_path):
    """The INFO half of the guarantee must reach the terminal in an actual serve.

    Every other log gate in this file opens the logger with ``caplog.at_level(INFO)``,
    which is precisely why they all passed while the real thing printed nothing: vLLM
    configures only the ``vllm`` logger (handler at INFO, ``propagate: False``) and leaves
    root at WARNING, so a bare ``getLogger(__name__)`` had its INFO records discarded. The
    warnings still showed, so the failure looked like success -- the adapter appeared to be
    logging, while the lines that say "this cache is being reused" and "converting fresh
    weights" were exactly the ones being dropped.

    So this gate does not raise any level. It rebuilds vLLM's configuration verbatim and
    asserts the message arrives through it. Renaming the adapter's logger out of the
    ``vllm`` hierarchy fails here and nowhere else.
    """
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    vllm_logger = logging.getLogger("vllm")
    handler = _Capture(level=logging.INFO)
    saved_level, saved_propagate = vllm_logger.level, vllm_logger.propagate
    vllm_logger.addHandler(handler)
    vllm_logger.setLevel(logging.INFO)
    vllm_logger.propagate = False  # as vLLM sets it
    try:
        adapter.ModelArgs(
            tmp_path / "cache", commit_hash="a3c85ec799fe5e35b0cffd754b59b20cdb34866c"
        ).weight_cache_path(BFP8)
    finally:
        vllm_logger.removeHandler(handler)
        vllm_logger.setLevel(saved_level)
        vllm_logger.propagate = saved_propagate

    infos = [r.getMessage() for r in records if r.levelno == logging.INFO]
    assert infos, (
        "no INFO record reached vLLM's handler -- the adapter's confirming lines are "
        "invisible in a real serve, which is the state this guard exists to prevent"
    )
    assert any("first conversion" in m for m in infos), infos


def test_unfingerprinted_leftovers_are_reported(adapter, tmp_path, caplog):
    """Caches converted before this guard existed are now dead. Say so; do not delete."""
    cache = tmp_path / "cache"
    stock = cache / f"tensor_cache_{BFP8}"
    stock.mkdir(parents=True)
    legacy = stock / "wo_dtype_BFLOAT8_B_layout_TILE.tensorbin"
    legacy.write_bytes(b"stale")

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        adapter.ModelArgs(cache, commit_hash="a3c85ec799fe5e35b0cffd754b59b20cdb34866c").weight_cache_path(BFP8)

    text = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "un-fingerprinted" in text
    assert legacy.exists(), "the guard must never delete a cache it did not create"


def test_missing_fingerprint_degrades_to_the_stock_path_loudly(adapter, tmp_path, caplog):
    """No sha, no local directory: the guard cannot help, and must not pretend it did."""
    args = adapter.ModelArgs(tmp_path / "cache", ckpt_dir="episod/tt-tnt", commit_hash=None)

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        path = args.weight_cache_path(BFP8)

    assert path == tmp_path / "cache" / f"tensor_cache_{BFP8}", "must fall back to stock"
    text = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "could not fingerprint" in text
    assert "PREVIOUS model may be served" in text


def test_opting_out_is_not_silent(adapter, tmp_path, caplog, monkeypatch):
    """``TT_TNT_CACHE_FINGERPRINT=0`` restores stock behaviour — and announces it."""
    monkeypatch.setenv("TT_TNT_CACHE_FINGERPRINT", "0")
    args = adapter.ModelArgs(tmp_path / "cache", commit_hash="a3c85ec799fe5e35b0cffd754b59b20cdb34866c")

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        path = args.weight_cache_path(BFP8)

    assert path == tmp_path / "cache" / f"tensor_cache_{BFP8}"
    text = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "TT_TNT_CACHE_FINGERPRINT is off" in text


def test_cache_directory_says_what_it_was_built_from(adapter, tmp_path):
    """``ls`` should answer the question that cost an afternoon."""
    sha = "a3c85ec799fe5e35b0cffd754b59b20cdb34866c"
    args = adapter.ModelArgs(tmp_path / "cache", ckpt_dir="episod/tt-tnt", commit_hash=sha)
    path = args.weight_cache_path(BFP8)

    stamp = json.loads((path / adapter.module.CACHE_STAMP_NAME).read_text(encoding="utf-8"))
    assert stamp["source"] == "episod/tt-tnt"
    assert stamp["fingerprint_kind"] == "rev"
    assert stamp["fingerprint"] == sha[:12]


# ---------------------------------------------------------------------------
# Degrade safely when upstream moves
# ---------------------------------------------------------------------------


def test_declines_to_patch_when_the_method_is_gone(tmp_path, caplog):
    """tt_transformers dropped ``weight_cache_path``: warn, do not crash the serve."""
    cls = make_model_args_cls(with_weight_cache_path=False)
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        with loaded_adapter(cls) as module:
            assert module._ORIGINAL_WEIGHT_CACHE_PATH is None
            assert not hasattr(cls, "weight_cache_path"), "nothing must be invented"

    text = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "is NOT installed" in text


def test_declines_to_patch_when_the_signature_changed(tmp_path, caplog):
    """A renamed second parameter means we can no longer be sure what we are scoping."""
    cls = make_model_args_cls(signature="renamed")
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        with loaded_adapter(cls) as module:
            assert module._ORIGINAL_WEIGHT_CACHE_PATH is None
            # Stock behaviour, untouched.
            args = cls(tmp_path / "cache")
            assert args.weight_cache_path(BFP8) == tmp_path / "cache" / f"tensor_cache_{BFP8}"

    text = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "unexpected signature" in text


def test_patch_is_idempotent_and_reversible(adapter, tmp_path):
    """Re-import must not double-wrap, and a host must be able to unwind cleanly."""
    module, cls = adapter.module, adapter.ModelArgs
    patched = cls.weight_cache_path

    module._patch_weight_cache_path()
    assert cls.weight_cache_path is patched, "double-wrapped"

    module.restore_patches()
    args = cls(tmp_path / "cache", commit_hash="a3c85ec799fe5e35b0cffd754b59b20cdb34866c")
    assert args.weight_cache_path(BFP8) == tmp_path / "cache" / f"tensor_cache_{BFP8}"
    assert cls.find_grid(args, 12) == ("stock", 12), "find_grid must be restored too"


# ---------------------------------------------------------------------------
# Upstream-drift canary (needs the tt-metal source; reads it, never imports it)
# ---------------------------------------------------------------------------


def _tt_metal_source_dir():
    """The tt-metal checkout, or None. Text only — this never imports ttnn."""
    for candidate in (os.environ.get("TT_METAL_HOME"), Path.home() / "tt-metal"):
        if not candidate:
            continue
        root = Path(candidate)
        if (root / "models" / "tt_transformers" / "tt" / "model_config.py").is_file():
            return root
    return None


def test_upstream_cache_anchors_still_present():
    """The three upstream facts this shim is built on must still be true.

    If any of them changes, the shim may be scoping the wrong thing (or the bug may have
    been fixed upstream and the shim can go). Either way a human should look — which is
    why this fails loudly rather than being folded into the patch's runtime guards.
    """
    root = _tt_metal_source_dir()
    if root is None:
        pytest.skip(
            "no tt-metal checkout found (set TT_METAL_HOME or clone to ~/tt-metal); the "
            "upstream-drift canary cannot run"
        )

    model_config = (root / "models" / "tt_transformers" / "tt" / "model_config.py").read_text(
        encoding="utf-8"
    )
    core = (root / "ttnn" / "ttnn" / "operations" / "core.py").read_text(encoding="utf-8")

    # 1. The cache key is still the repo id and the device name, with no revision.
    assert 'os.path.join("model_cache", HF_MODEL, self.device_name)' in model_config, (
        "the tt_transformers cache key changed — re-check whether this shim is still "
        "needed and still scoping the right thing"
    )
    # 2. weight_cache_path is still the single funnel this shim wraps.
    assert "def weight_cache_path(self, dtype):" in model_config, (
        "ModelArgs.weight_cache_path changed shape — the shim's install-time guard will "
        "decline, and the stale-cache bug returns"
    )
    # 3. The reuse decision is still a bare existence check, with no content validation.
    assert "if not cache_path.exists() or not cache_path.is_file():" in core, (
        "ttnn.as_tensor's cache reuse decision changed — if it now validates content, "
        "this shim may be redundant"
    )
