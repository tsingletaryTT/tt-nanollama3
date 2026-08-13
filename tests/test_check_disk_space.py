# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Preflight tests. The gate must be honest in both directions: it must refuse when space
is short, and it must not block when space is fine."""
from pathlib import Path

from scripts.check_disk_space import check_space, free_bytes


def test_free_bytes_is_positive_for_a_real_path(tmp_path: Path):
    assert free_bytes(tmp_path) > 0


def test_check_space_passes_when_requirement_is_tiny(tmp_path: Path):
    ok, msg = check_space(tmp_path, required_gb=0.000001)
    assert ok
    assert "free" in msg


def test_check_space_fails_when_requirement_is_absurd(tmp_path: Path):
    ok, msg = check_space(tmp_path, required_gb=10_000_000)
    assert not ok
    assert "need" in msg.lower()


def test_check_space_never_suggests_deleting_anything(tmp_path: Path):
    """The message must not invite the implementer to reclaim space on its own."""
    _, msg = check_space(tmp_path, required_gb=10_000_000)
    for forbidden in ("rm ", "delete", "prune", "clear the cache"):
        assert forbidden not in msg.lower()
