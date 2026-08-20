# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for train/enthusiasts.py — the Mixture of Enthusiasts wiring.

No hardware. `ttml` is faked, which is the point: the bug these were written for was a
HOST-side call on a device-distributed tensor, and a fake tensor reproduces it exactly
while a real one needs a 2-chip mesh.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.enthusiasts import (  # noqa: E402
    MoEHyperparams,
    _apply_die_gate,
    die_gate_weights,
    gate_recovers_region,
)


# --------------------------------------------------------------------------- fakes

class _FakeValue:
    pass


class _MeshTensor:
    """A parameter distributed over a mesh: shape/dtype are cheap, to_numpy RAISES.

    This is the real failure. Under `--ddp 2` the gate parameter lives across
    MeshShape([1, 2]) and pulling it to host raises

        TT_FATAL: Can't get a single buffer from host storage distributed over
        mesh shape MeshShape([1, 2])        (tensor_apis.cpp:631)

    so any host round-trip in the seeding path dies. Single-device runs never see it,
    which is why the seeded and frozen arms failed while dense and learned passed.
    """

    def __init__(self, shape):
        self._shape = tuple(shape)
        self.value = None
        self.requires_grad = True

    def get_shape(self):
        return self._shape

    @property
    def shape(self):
        return self._shape

    def to_numpy(self):
        raise RuntimeError(
            "TT_FATAL @ tensor_apis.cpp:631: buffers.size() == 1 — Can't get a single "
            "buffer from host storage distributed over mesh shape MeshShape([1, 2])")

    def set_value(self, v):
        self.value = v

    def set_requires_grad(self, flag):
        self.requires_grad = flag


class _Gate:
    def __init__(self, tensor):
        self._t = tensor

    def parameters(self):
        return {"llama/moe/gate/LinearLayer/weight": self._t}


class _MoE:
    def __init__(self, tensor):
        self.gate = _Gate(tensor)


@pytest.fixture
def fake_ttml(monkeypatch):
    """Minimal `ttml.autograd.Tensor.from_numpy(...).get_value()`."""
    captured = {}

    class _Built:
        def __init__(self, arr):
            captured["array"] = arr

        def get_value(self):
            return _FakeValue()

    tensor_cls = types.SimpleNamespace(from_numpy=lambda arr: _Built(arr))
    mod = types.ModuleType("ttml")
    mod.autograd = types.SimpleNamespace(Tensor=tensor_cls)
    monkeypatch.setitem(sys.modules, "ttml", mod)
    return captured


def _hp(experts=10, dim=64):
    return MoEHyperparams(dim=dim, n_routed_experts=experts, n_activated_experts=2,
                          n_shared_experts=1, moe_inter_dim=58)


# ------------------------------------------- the regression: no host round-trip

def test_die_gate_seeding_survives_a_mesh_distributed_parameter(fake_ttml):
    """The whole point. A gate whose to_numpy() raises must still be seedable.

    Before the fix `_apply_die_gate` called `tensor.to_numpy()` purely to read the shape
    and dtype, so it died on any multi-device run.
    """
    hp = _hp()
    rng = np.random.default_rng(0)
    embedding = rng.normal(size=(200, hp.dim)).astype(np.float32)
    per_token = rng.integers(0, hp.n_routed_experts, size=200)
    tensor = _MeshTensor((1, 1, hp.n_routed_experts, hp.dim))

    _apply_die_gate(_MoE(tensor), hp, embedding, per_token, freeze=False)

    assert tensor.value is not None, "the gate was never written"
    assert fake_ttml["array"].shape == (1, 1, hp.n_routed_experts, hp.dim)


def test_freeze_sets_requires_grad_false(fake_ttml):
    """The frozen arm differs from seeded ONLY here, so it needs its own assertion."""
    hp = _hp()
    rng = np.random.default_rng(1)
    tensor = _MeshTensor((1, 1, hp.n_routed_experts, hp.dim))
    _apply_die_gate(_MoE(tensor), hp,
                    rng.normal(size=(120, hp.dim)).astype(np.float32),
                    rng.integers(0, hp.n_routed_experts, size=120), freeze=True)
    assert tensor.requires_grad is False


def test_seeded_leaves_the_gate_trainable(fake_ttml):
    hp = _hp()
    rng = np.random.default_rng(2)
    tensor = _MeshTensor((1, 1, hp.n_routed_experts, hp.dim))
    _apply_die_gate(_MoE(tensor), hp,
                    rng.normal(size=(120, hp.dim)).astype(np.float32),
                    rng.integers(0, hp.n_routed_experts, size=120), freeze=False)
    assert tensor.requires_grad is True


def test_wrong_gate_shape_is_refused(fake_ttml):
    """A transposed gate scores by the wrong axis and STILL RUNS, so shape is checked."""
    hp = _hp()
    rng = np.random.default_rng(3)
    bad = _MeshTensor((1, 1, hp.dim, hp.n_routed_experts))   # transposed
    with pytest.raises(ValueError, match="unexpected gate weight shape"):
        _apply_die_gate(_MoE(bad), hp,
                        rng.normal(size=(120, hp.dim)).astype(np.float32),
                        rng.integers(0, hp.n_routed_experts, size=120), freeze=False)


# ------------------------------------------- the routing maths, previously untested

def test_die_gate_weights_shape_and_finiteness():
    rng = np.random.default_rng(4)
    emb = rng.normal(size=(500, 32)).astype(np.float32)
    owners = rng.integers(0, 7, size=500)
    W = die_gate_weights(emb, owners, 7)
    assert W.shape == (7, 32)
    assert np.isfinite(W).all()


def test_gate_recovers_region_beats_chance_on_separable_input():
    """A gate seeded from well-separated clusters must recover far more than 1/k.

    Guards the direction of the claim: the reported 61.2%-vs-10% figure is only
    meaningful if this function can tell a real signal from chance at all.
    """
    rng = np.random.default_rng(5)
    k, dim, n = 5, 16, 600
    centres = rng.normal(size=(k, dim)).astype(np.float32) * 6.0
    owners = rng.integers(0, k, size=n)
    emb = (centres[owners] + rng.normal(size=(n, dim)).astype(np.float32) * 0.1)
    assert gate_recovers_region(emb, owners, k) > 0.80


def test_gate_recovery_is_near_chance_on_noise():
    """The other direction: unseparable input must NOT look like recovery."""
    rng = np.random.default_rng(6)
    k, dim, n = 10, 16, 800
    emb = rng.normal(size=(n, dim)).astype(np.float32)
    owners = rng.integers(0, k, size=n)
    assert gate_recovers_region(emb, owners, k) < 0.35, "noise must not read as signal"
