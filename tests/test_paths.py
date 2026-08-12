# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for per-size artifact path resolution.

Two invariants carry real weight here:

1. **Writes can never reach the baseline.** ``artifacts/checkpoints`` and ``artifacts/hf``
   hold the only copies of the published weights and the checkpoint the parity gate pins
   against. Before this module, ``--checkpoint-dir`` *defaulted* to the former.

2. **The legacy fallback is scoped to one size.** The flat directories belong to the 384
   model. If the fallback applied to every size, asking for the untrained 1024 model's
   checkpoints would return the 384 baseline and the converter would build a confidently
   wrong model from it — no error, just the wrong weights. This was a real bug in the first
   draft of ``read_dir``, caught by reading its output; :func:`test_legacy_fallback_is_
   scoped_to_the_legacy_size_only` is what stops it coming back.
"""

from __future__ import annotations

import pytest

from train.paths import (
    LEGACY_SIZE,
    PER_SIZE_KINDS,
    SHARED_KINDS,
    ProtectedPathError,
    assert_not_protected,
    protected_dirs,
    read_dir,
    shared_dir,
    write_dir,
)
from train.sizes import SIZES


@pytest.fixture
def fake_root(tmp_path):
    """A repo-shaped tmp tree with the legacy flat layout populated."""
    (tmp_path / "artifacts" / "checkpoints").mkdir(parents=True)
    (tmp_path / "artifacts" / "hf").mkdir(parents=True)
    (tmp_path / "artifacts" / "tokenizer").mkdir(parents=True)
    return tmp_path


# --------------------------------------------------------------------------------------
# Invariant 1: writes never reach the baseline
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("size", sorted(SIZES))
@pytest.mark.parametrize("kind", sorted(PER_SIZE_KINDS))
def test_write_dir_is_always_per_size(size, kind, fake_root):
    """Even when the legacy directory exists and is populated."""
    target = write_dir(size, kind, root=fake_root)
    assert target == fake_root / "artifacts" / size / kind
    assert target not in [p for p in protected_dirs(fake_root)]


@pytest.mark.parametrize("kind", sorted(PER_SIZE_KINDS))
def test_assert_not_protected_rejects_the_baseline(kind, fake_root):
    with pytest.raises(ProtectedPathError) as excinfo:
        assert_not_protected(fake_root / "artifacts" / kind, root=fake_root)
    msg = str(excinfo.value)
    assert "protected baseline evidence" in msg
    assert "per-size directory" in msg, "the error must say what to do instead"


def test_assert_not_protected_sees_through_path_games(fake_root):
    """Resolution defeats ``..`` traversal — the guard compares resolved paths."""
    sneaky = fake_root / "artifacts" / "384" / ".." / "checkpoints"
    with pytest.raises(ProtectedPathError):
        assert_not_protected(sneaky, root=fake_root)


def test_assert_not_protected_allows_ordinary_paths(fake_root, tmp_path):
    for ok in (
        fake_root / "artifacts" / "384" / "checkpoints",
        fake_root / "artifacts" / "checkpoints-v2",
        tmp_path / "somewhere-else",
    ):
        assert assert_not_protected(ok, root=fake_root) == ok


# --------------------------------------------------------------------------------------
# Invariant 2: the legacy fallback is scoped
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(PER_SIZE_KINDS))
def test_read_dir_falls_back_to_legacy_for_the_legacy_size(kind, fake_root):
    """The baseline stays findable without being moved."""
    assert read_dir(LEGACY_SIZE, kind, root=fake_root) == fake_root / "artifacts" / kind


@pytest.mark.parametrize("kind", sorted(PER_SIZE_KINDS))
def test_legacy_fallback_is_scoped_to_the_legacy_size_only(kind, fake_root):
    """THE REGRESSION GATE.

    A non-legacy size must never resolve to the flat directory, even though it exists and
    the per-size one does not. Returning the baseline here would silently substitute one
    model's weights for another's.
    """
    other = next(s for s in SIZES if s != LEGACY_SIZE)
    resolved = read_dir(other, kind, root=fake_root)
    assert resolved == fake_root / "artifacts" / other / kind
    assert resolved != fake_root / "artifacts" / kind, (
        f"size {other} resolved to the {LEGACY_SIZE} baseline directory"
    )


@pytest.mark.parametrize("kind", sorted(PER_SIZE_KINDS))
def test_read_dir_prefers_per_size_when_it_exists(kind, fake_root):
    per_size = fake_root / "artifacts" / LEGACY_SIZE / kind
    per_size.mkdir(parents=True)
    assert read_dir(LEGACY_SIZE, kind, root=fake_root) == per_size


@pytest.mark.parametrize("kind", sorted(PER_SIZE_KINDS))
def test_read_dir_returns_the_per_size_path_when_nothing_exists(kind, tmp_path):
    """Error messages should point at where the artifact belongs, not at a legacy path."""
    assert read_dir("1024", kind, root=tmp_path) == tmp_path / "artifacts" / "1024" / kind


# --------------------------------------------------------------------------------------
# Shared kinds
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(SHARED_KINDS))
def test_shared_kinds_are_flat_and_size_independent(kind, fake_root):
    """One tokenizer serves every size — that is much of why they share a repo."""
    flat = fake_root / "artifacts" / kind
    assert shared_dir(kind, root=fake_root) == flat
    for size in SIZES:
        assert write_dir(size, kind, root=fake_root) == flat
        assert read_dir(size, kind, root=fake_root) == flat


def test_shared_dir_rejects_a_per_size_kind():
    with pytest.raises(ValueError, match="not a shared artifact kind"):
        shared_dir("checkpoints")


def test_write_dir_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown artifact kind"):
        write_dir("384", "not-a-kind")
