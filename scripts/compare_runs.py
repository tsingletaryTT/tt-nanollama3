#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Compare two training runs' validation curves, with the guards today cost us.

WHY THIS EXISTS
===============
On 2026-08-19 the same comparison was done by hand three times and was wrong
three times. Not the model -- the *comparison*:

1. Two runs were compared against a baseline trained on a different corpus. The
   default ``--tokens-dir`` is ``artifacts/tokens``, the OLDEST of six token sets,
   and it was never passed. A 1.3-nat "regression" was nearly attributed to the
   optimizer, and a working config was nearly discarded.

2. A baseline lookup table was built by splitting a whitespace table with an
   empty middle column. The columns collapsed and one run's number was reported
   under the other run's name.

3. The baseline curve was printed with ``head -8`` and every subsequent
   conclusion was built on that truncation. Eleven points existed. The "five
   consecutive positive deltas" that made a tail effect look real stopped at step
   8000 because the DATA stopped at 8000; points 9 and 10 are negative.

Each of those was an instrument reporting confidently. This project applies real
measurement discipline to the model -- noise floors, withdrawn findings,
NOT INTERPRETABLE as a verdict -- and applied none of it to the tools measuring
the model. This script is the first correction of that.

WHAT IT REFUSES TO DO
=====================
* It never truncates. Both curves are read whole, and the count of matched points
  is printed as the headline, not buried.
* It never silently compares runs that are not comparable. If both directories
  carry ``run_manifest.json`` (written by ``train/run.py`` since 2026-08-19), the
  training inputs are cross-checked and a mismatch in ``tokens_dir``,
  ``train_tokens``, ``seq_len``, ``batch_size`` or ``ddp`` is a HARD FAILURE, not
  a warning. Passing ``--allow-mismatch`` states the intent explicitly.
* It never reports a delta as a finding without a floor. The floor is derived from
  the runs themselves -- consecutive validations of a single run 500 steps apart --
  and any mean difference inside it is stamped NOT INTERPRETABLE.
* A missing manifest is reported LOUDLY rather than treated as "probably fine".
  Runs from before the manifest existed cannot prove their own comparability, and
  that is exactly the situation that produced failure 1.

USAGE
-----
    python scripts/compare_runs.py artifacts/checkpoints-v077 \
                                   artifacts/checkpoints-1024-dialogue
    python scripts/compare_runs.py A B --json docs/measurements/out.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: Manifest fields that must agree for two curves to be comparable at all. Every
#: one of these changes the loss curve, so a difference makes a delta meaningless.
#: `tokens_dir` heads the list because it is the one that actually went wrong.
COMPARABILITY_KEYS = (
    "tokens_dir",
    "train_tokens",
    "val_tokens",
    "seq_len",
    "batch_size",
    "gradient_accumulation_steps",
    "ddp",
    "size",
)

#: Fields expected to DIFFER — they are usually the point of the comparison — and
#: are reported as the experiment's variables rather than flagged as problems.
EXPECTED_TO_DIFFER = ("optimizer", "optimizer_override_file", "lr_schedule",
                      "warmup_frac", "tt_metal", "seed")


def load_curve(d: Path) -> Dict[int, float]:
    """Every point in a run's val_losses.jsonl. No limit, no head, no sampling."""
    f = d / "val_losses.jsonl"
    if not f.exists():
        raise SystemExit(f"error: {f} does not exist; nothing to compare")
    out: Dict[int, float] = {}
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out[int(row["step"])] = float(row["val_loss"])
    if not out:
        raise SystemExit(f"error: {f} is empty")
    return out


def load_manifest(d: Path) -> Optional[dict]:
    f = d / "run_manifest.json"
    return json.loads(f.read_text()) if f.exists() else None


def wobble(curve: Dict[int, float]) -> List[float]:
    """Absolute change between consecutive validations of ONE run.

    This is the noise floor. If a single run's own curve moves by X between
    adjacent points, a difference between two runs smaller than X carries no
    information. It is a weaker floor than repeating a run under a new seed --
    that measures run-to-run variation, this measures within-run oscillation --
    and it is what is available from two curves alone. Stated so nobody mistakes
    it for the stronger thing.
    """
    steps = sorted(curve)
    return [abs(curve[b] - curve[a]) for a, b in zip(steps, steps[1:])]


def check_comparability(ma: Optional[dict], mb: Optional[dict],
                        allow: bool) -> Tuple[List[str], List[str]]:
    """Returns (blocking_mismatches, notes). Raises SystemExit unless *allow*."""
    notes: List[str] = []
    if ma is None or mb is None:
        which = [n for n, m in (("A", ma), ("B", mb)) if m is None]
        notes.append(
            f"NO run_manifest.json for {', '.join(which)}. Comparability CANNOT be "
            f"verified -- these runs predate the manifest (added 2026-08-19). This is "
            f"the exact situation that produced a 1.3-nat false regression: check the "
            f"tokens dir by hand before trusting any number below.")
        return [], notes

    bad = []
    for k in COMPARABILITY_KEYS:
        va, vb = ma.get(k), mb.get(k)
        if va != vb:
            bad.append(f"{k}: A={va!r} B={vb!r}")
    for k in EXPECTED_TO_DIFFER:
        va, vb = ma.get(k), mb.get(k)
        if va != vb:
            notes.append(f"variable under test -- {k}: A={va!r} B={vb!r}")
    if bad and not allow:
        print("REFUSING TO COMPARE: the runs differ in inputs that change the loss "
              "curve.\n", file=sys.stderr)
        for b in bad:
            print(f"  {b}", file=sys.stderr)
        print("\nA delta between these curves would measure the input difference, not "
              "the variable you are testing. Pass --allow-mismatch if the difference "
              "IS the experiment.", file=sys.stderr)
        raise SystemExit(2)
    return bad, notes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir_a", type=Path, help="candidate checkpoint dir")
    p.add_argument("dir_b", type=Path, help="baseline checkpoint dir")
    p.add_argument("--allow-mismatch", action="store_true",
                   help="compare even when manifests disagree on training inputs")
    p.add_argument("--paired", dest="paired", action="store_true", default=None,
                   help="force the paired test (runs share a seed, so data order and its "
                        "oscillation are common to both curves and cancel in the delta)")
    p.add_argument("--unpaired", dest="paired", action="store_false",
                   help="force the wobble-floor test (runs have different seeds)")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    a, b = load_curve(args.dir_a), load_curve(args.dir_b)
    ma, mb = load_manifest(args.dir_a), load_manifest(args.dir_b)
    mismatches, notes = check_comparability(ma, mb, args.allow_mismatch)

    matched = sorted(set(a) & set(b))
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))

    print(f"A (candidate) : {args.dir_a}   {len(a)} points, steps {min(a)}..{max(a)}")
    print(f"B (baseline)  : {args.dir_b}   {len(b)} points, steps {min(b)}..{max(b)}")
    print(f"matched       : {len(matched)} points")
    # Printed even when empty. Silence about unmatched steps is how a truncated
    # comparison looks identical to a complete one.
    print(f"only in A     : {only_a if only_a else 'none'}")
    print(f"only in B     : {only_b if only_b else 'none'}")
    for n in notes:
        print(f"\n  ! {n}")
    if mismatches:
        print("\n  ! COMPARING ANYWAY (--allow-mismatch): " + "; ".join(mismatches))
    if not matched:
        raise SystemExit("\nerror: no matched steps; the curves share no validation points")

    deltas = [a[s] - b[s] for s in matched]
    mean = statistics.mean(deltas)
    sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    se = sd / len(deltas) ** 0.5 if len(deltas) > 1 else 0.0
    wob = wobble(a) + wobble(b)
    floor = statistics.median(wob) if wob else 0.0

    print(f"\n{'step':>8}  {'A':>8}  {'B':>8}  {'delta':>9}")
    for s in matched:
        print(f"{s:>8}  {a[s]:>8.4f}  {b[s]:>8.4f}  {a[s]-b[s]:>+9.4f}")

    signs = "".join("+" if d > 0 else "-" for d in deltas)
    print(f"\nmean delta        {mean:+.4f}")
    print(f"sd / se           {sd:.4f} / {se:.4f}")
    print(f"signs             {signs}  ({signs.count('+')}+ / {signs.count('-')}-)")
    print(f"within-run floor  {floor:.4f}   (median consecutive move, both runs pooled)")

    # PAIRED vs UNPAIRED. This distinction was missing in the first version of this
    # script and it produced a wrong verdict on its first real use: mean -0.0482 at
    # se 0.0081 (six standard errors, 19 of 22 signs agreeing) was stamped NOT
    # INTERPRETABLE because |mean| fell under the within-run wobble floor.
    #
    # The wobble floor is right for runs with DIFFERENT seeds: their data order
    # differs, so each curve oscillates independently and a small mean difference
    # really can be an artefact of where the oscillations happened to land.
    #
    # It is far too conservative for runs that SHARE a seed. Identical data order
    # means the oscillation is common to both curves and subtracts out, so the
    # paired deltas are much tighter than either curve is on its own -- which is
    # exactly what se = 0.0081 against a 0.0762 wobble is telling you. Judging a
    # paired design by an unpaired floor discards most of its sensitivity.
    if args.paired is not None:
        paired = args.paired
        how = "forced on the command line"
    elif ma and mb and ma.get("seed") is not None and ma.get("seed") == mb.get("seed"):
        paired = True
        how = f"auto: both manifests report seed {ma.get('seed')}"
    else:
        paired = False
        how = ("auto: seeds could not be confirmed equal from manifests, so the "
               "conservative unpaired floor is used")
    print(f"design            {'PAIRED' if paired else 'UNPAIRED'}  ({how})")

    if paired:
        # se == 0 means every delta is identical -- either the curves are the same
        # (mean 0) or they are offset by a constant. Both are degenerate for a
        # t-like statistic, which would divide by zero, and both have an obvious
        # correct answer that must not be reached through the division.
        agree = signs.count('-' if mean < 0 else '+')
        if se == 0:
            verdict = ("NOT INTERPRETABLE -- the curves are identical" if mean == 0
                       else (("better (A lower)" if mean < 0 else "worse (A higher)") +
                             f" -- every one of {len(deltas)} deltas is exactly {mean:+.4f}; "
                             f"a constant offset with zero scatter"))
        elif abs(mean) <= 2 * se:
            verdict = "NOT INTERPRETABLE -- |mean| is within 2 standard errors of zero"
        else:
            verdict = (("better (A lower)" if mean < 0 else "worse (A higher)") +
                       f" -- |mean| is {abs(mean)/se:.1f} standard errors from zero, "
                       f"{agree}/{len(deltas)} signs agree")
    else:
        if abs(mean) <= floor:
            verdict = ("NOT INTERPRETABLE -- |mean| is inside the runs' own "
                       "point-to-point scatter")
        elif se == 0:
            verdict = (("better (A lower)" if mean < 0 else "worse (A higher)") +
                       f" -- constant offset of {mean:+.4f} with zero scatter")
        elif abs(mean) <= 2 * se:
            verdict = "NOT INTERPRETABLE -- |mean| is within 2 standard errors of zero"
        else:
            verdict = ("better (A lower)" if mean < 0 else "worse (A higher)") + \
                      " -- exceeds both the floor and 2 se"
    print(f"\nVERDICT: {verdict}")

    if args.json:
        args.json.write_text(json.dumps({
            "dir_a": str(args.dir_a), "dir_b": str(args.dir_b),
            "n_a": len(a), "n_b": len(b), "matched": len(matched),
            "only_in_a": only_a, "only_in_b": only_b,
            "points": [{"step": s, "a": a[s], "b": b[s], "delta": round(a[s]-b[s], 4)}
                       for s in matched],
            "mean_delta": round(mean, 4), "sd": round(sd, 4), "se": round(se, 4),
            "signs": signs, "within_run_floor": round(floor, 4),
            "design": "paired" if paired else "unpaired", "design_reason": how,
            "t_like": round(mean / se, 2) if se else None,  # None when deltas are constant
            "verdict": verdict, "manifest_notes": notes,
            "comparability_mismatches": mismatches,
        }, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
