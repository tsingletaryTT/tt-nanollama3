# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Shared test helpers, and the one thing that keeps a green run honest.

``artifacts/`` is gitignored: checkpoints, converted HF directories, token arrays
and the 1.6 GB corpus all live there and none of it is in a clone. Tests that need
those inputs must therefore SKIP when they are absent, or a fresh checkout — CI
above all — reports failures that say nothing about the code.

But skipping has a failure mode of its own, and it was written down in this repo
before I met it (``tests/test_numpy_parity.py``):

    every decisive test being `skipif`-guarded on gitignored `artifacts/` means a
    CI run can report "all passed" while the load-bearing tests never executed

That is the real hazard. A green tick that covered 1037 of 1080 tests is only
useful if the 43 it skipped are known and few. So this module does two things:
it gives the skip a single shared spelling, and it makes the count impossible to
miss at the end of every run, locally and in CI alike.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Set at collection time by `needs_artifacts`, reported by the terminal summary.
_gated: set[str] = set()


def needs_artifacts(*relpaths: str, reason: str = ""):
    """Skip-if-missing marker for tests that need real, gitignored inputs.

    Prefer NOT using this. A test that can be made to run on a few synthetic bytes
    should be — `tests/test_lr_schedule.py`'s `tiny_tokens` fixture builds a
    4096-token directory precisely so two wiring tests need no corpus. Reach for
    this only when the real artifact IS the subject: `test_artifact_files_matches_
    real_hf_dir` cannot be satisfied by a stand-in, because agreeing with a
    stand-in is not the property under test.

    Args:
        relpaths: paths under the repo root that must all exist.
        reason: appended to the skip message, for anything non-obvious.
    """
    missing = [p for p in relpaths if not (ROOT / p).exists()]
    for p in relpaths:
        _gated.add(p)
    why = f"needs gitignored artifacts: {', '.join(missing)}"
    if reason:
        why = f"{why} ({reason})"
    return pytest.mark.skipif(bool(missing), reason=why)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Report how much of the suite did not run, and why.

    Printed unconditionally, including on a fully green run. The number is the
    whole point: "1080 passed" and "1037 passed, 43 skipped because this machine
    has no corpus" are different claims, and only one of them is what a CI badge
    silently means.
    """
    skipped = terminalreporter.stats.get("skipped", [])
    artifact_skips = [
        r for r in skipped
        if "gitignored artifacts" in str(getattr(r, "longrepr", ("", "", ""))[2]
                                         if isinstance(getattr(r, "longrepr", None), tuple)
                                         else getattr(r, "longrepr", ""))
    ]
    total = terminalreporter._numcollected if hasattr(terminalreporter, "_numcollected") else 0
    if artifact_skips:
        terminalreporter.write_sep("-", "artifact-gated tests")
        terminalreporter.write_line(
            f"{len(artifact_skips)} of {total} tests were SKIPPED for missing gitignored "
            f"artifacts/. A green run does not mean these passed — it means they did not run."
        )
    elif _gated:
        terminalreporter.write_sep("-", "artifact-gated tests")
        terminalreporter.write_line(
            f"0 skipped: all {len(_gated)} artifact paths this suite gates on are present."
        )
