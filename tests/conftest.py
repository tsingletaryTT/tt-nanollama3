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
    whole point: "1080 passed" and "1034 passed, 48 skipped because this machine
    has no corpus, no checkpoints and no tt-metal" are different claims, and only
    one of them is what a CI badge silently means.

    Reports the TOTAL skip count first, deliberately. An earlier version of this
    hook counted only the tests gated by `needs_artifacts` and announced "10 of
    1082 skipped" on a run where 48 had skipped -- understating by nearly five
    times, in the one place written to stop exactly that. Most skips in this suite
    predate the helper and phrase their own reasons; a summary that sees only its
    own marker is measuring itself.
    """
    skipped = terminalreporter.stats.get("skipped", [])
    total = getattr(terminalreporter, "_numcollected", 0)
    if not skipped:
        if _gated:
            terminalreporter.write_sep("-", "coverage of this run")
            terminalreporter.write_line(
                f"0 skipped: every input this suite gates on is present.")
        return

    def _reason(rep) -> str:
        lr = getattr(rep, "longrepr", "")
        return str(lr[2]) if isinstance(lr, tuple) and len(lr) > 2 else str(lr)

    reasons = [_reason(r) for r in skipped]
    gated = sum(1 for r in reasons if "gitignored artifacts" in r)

    # Group the rest by a coarse cause so the line is readable rather than a wall.
    def bucket(r: str) -> str:
        low = r.lower()
        if "gitignored artifacts" in low:
            return "artifacts (gated)"
        if any(k in low for k in ("checkpoint", "converted model", "tokenizer",
                                  "tokens", "safetensors", "not committed",
                                  "not in this checkout", "repository content",
                                  "config.json")):
            return "other local training artifacts"
        if "tt_metal_home" in low or "tt-metal" in low or "tt-kernel" in low:
            return "tt-metal / tt-kernel not available"
        if "lm-eval" in low:
            return "optional lm-eval venv"
        return "other"

    counts: dict[str, int] = {}
    for r in reasons:
        counts[bucket(r)] = counts.get(bucket(r), 0) + 1

    terminalreporter.write_sep("-", "coverage of this run")
    terminalreporter.write_line(
        f"{len(skipped)} of {total} tests did NOT run. A green result does not mean "
        f"they passed."
    )
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        terminalreporter.write_line(f"    {n:>4}  {name}")
    if gated != len(skipped):
        terminalreporter.write_line(
            f"    ({gated} of these use the needs_artifacts helper; the rest carry "
            f"their own skip reasons — run with -rs to read them)")
