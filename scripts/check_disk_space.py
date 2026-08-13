#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Refuse to start a download that could fill the volume.

This machine's root filesystem was at 98% (90 GB free) when this plan was written, with a
1.2 TB Hugging Face cache belonging to other work. Filling it would damage unrelated
projects.

This script REPORTS and EXITS. It never deletes, prunes, or relocates anything: reclaiming
space is a human decision, and an automated tool guessing which gigabytes are expendable is
exactly how someone's dataset disappears.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

#: Rough budget for the full discovery pipeline: filtered Gutenberg subsets, Simple
#: Wikipedia, the normalised text copies, and headroom. Streaming keeps the full 10.75 GB
#: dataset off local disk, so this is far below the raw dataset size.
DEFAULT_REQUIRED_GB = 45.0


def free_bytes(path: Path) -> int:
    """Bytes free on the filesystem holding ``path``."""
    return shutil.disk_usage(Path(path)).free


def check_space(path: Path, required_gb: float) -> tuple:
    """(ok, message). Never proposes reclaiming space."""
    free_gb = free_bytes(path) / 1e9
    total_gb = shutil.disk_usage(Path(path)).total / 1e9
    pct_used = 100.0 * (1.0 - free_gb / total_gb) if total_gb else 0.0
    if free_gb >= required_gb:
        return True, (f"{free_gb:,.1f} GB free of {total_gb:,.1f} GB "
                      f"({pct_used:.0f}% used); need {required_gb:,.1f} GB")
    return False, (f"INSUFFICIENT SPACE: {free_gb:,.1f} GB free of {total_gb:,.1f} GB "
                   f"({pct_used:.0f}% used), need {required_gb:,.1f} GB. "
                   f"Stopping. Report this rather than making room.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", type=Path, default=Path.cwd())
    p.add_argument("--required-gb", type=float, default=DEFAULT_REQUIRED_GB)
    args = p.parse_args()
    ok, msg = check_space(args.path, args.required_gb)
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
