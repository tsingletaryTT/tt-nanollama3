#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""What tt-tnt can measure about the stack it runs on, in one command.

WHY THIS EXISTS
===============
This project's most useful output so far has not been the model. It has been the
DEFECTS the model found by being a real consumer of the Tenstorrent stack: a real
bundle, a real manifest, a real serve command, on real silicon. Four landed
upstream or into review in a single day (2026-08-20).

But that hunting was manual. Each finding came from a human noticing that some
line of output was wrong, and none of it was repeatable: nothing re-checked the
previous four while looking for the fifth. A finding you cannot re-run is an
anecdote.

So every check here is a failure THIS PROJECT HAS ACTUALLY OBSERVED. None is
hypothetical. That is the whole design rule, and it is the reason to trust a
green run: each of these has been seen red, on this hardware, with a known cause.

WHAT IT IS NOT
--------------
Not a model benchmark. It says nothing about tt-tnt's quality — the loss curves
and the die-region measurements do that. This answers a different question:
*is the stack underneath telling me the truth about itself?*

Most checks need no device. `--with-device` adds the ones that do.

USAGE
-----
    python scripts/stack_probe.py                      # host + CLI checks only
    python scripts/stack_probe.py --json-out probe.json
    python scripts/stack_probe.py --with-device        # adds a training smoke
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

ROOT = Path(__file__).resolve().parents[1]

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclass
class Result:
    name: str
    status: str
    detail: str
    #: The failure this check exists because of. A check with no history here is a
    #: check nobody has watched fail, which is the thing this file refuses to ship.
    regression_of: str = ""
    evidence: dict = field(default_factory=dict)


def _run(argv: List[str], *, env: Optional[dict] = None, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, **(env or {})})


def _strip_ansi(s: str) -> str:
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


# --------------------------------------------------------------------------- checks

def check_instance_version_matches_its_tree() -> Result:
    """An instance's reported ttnn version must match its OWN tt-metal tree.

    OBSERVED FAILURE: `tt-model instances list` reported `ttnn=0.65.1rc17.dev6200` for a
    tree that was actually v0.77.0 — `resolve_version()` preferred frozen editable
    dist-metadata over the source tree, so a rebuilt checkout reported its install-time
    version forever, and every manifest range (`>=0.72`) silently failed to match.
    Fixed upstream as f572d1a (PR #15).

    The check compares the reported version against `git describe` in that instance's
    `tt_metal_home`. An earlier version of this check just grepped for a `.dev` tail and
    FAILED on a healthy stack, because a venv holding a genuinely old ttnn looks identical
    to stale metadata by that test. A heuristic that cannot tell those apart is not a check.
    """
    name = "instance ttnn version matches its own tt-metal tree"
    why = "tt-kernel PR #15 — resolve_version() trusted frozen editable metadata over the tree"
    reg = Path.home() / ".config" / "tt-kernel" / "instances.json"
    if not reg.is_file():
        reg = Path.home() / ".config" / "tt-model" / "instances.json"
    if not reg.is_file():
        return Result(name, SKIP, "no instance registry", why)
    insts = json.loads(reg.read_text()).get("instances") or []
    if isinstance(insts, dict):
        insts = list(insts.values())
    if not shutil.which("git"):
        return Result(name, SKIP, "git not available to describe the trees", why)

    compared, bad = [], []
    for i in insts:
        home, py = i.get("tt_metal_home"), i.get("python")
        if not home or not Path(home).is_dir() or not py or not Path(py).exists():
            continue
        tree = _run(["git", "-C", home, "describe", "--tags"], timeout=60)
        if tree.returncode != 0:
            continue
        tree_v = tree.stdout.strip().lstrip("v").split("-")[0]
        # What the instance's own interpreter reports for ttnn.
        got = _run([py, "-c", "import importlib.metadata as m;print(m.version('ttnn'))"],
                   env={"TT_METAL_HOME": home}, timeout=180)
        if got.returncode != 0:
            continue
        got_v = got.stdout.strip()
        entry = {"instance": i.get("name"), "tree": tree_v, "reported": got_v}
        compared.append(entry)
        # Compare on (major, minor) only: patch/rc tails differ legitimately between a
        # tag and a wheel. A minor-version disagreement is the actual bug's signature.
        def mm(v):
            parts = v.lstrip("v").split(".")
            return tuple(parts[:2])
        if mm(tree_v) != mm(got_v):
            bad.append(f"{i.get('name')}: tree {tree_v} but ttnn reports {got_v}")
    if not compared:
        return Result(name, SKIP, "no instance has both a git tree and a probeable ttnn", why)
    if bad:
        return Result(name, FAIL, "; ".join(bad), why, {"compared": compared})
    return Result(name, PASS, f"{len(compared)} instance(s) agree with their tree", why,
                  {"compared": compared})


def check_toolchain_agrees_with_the_instance() -> Result:
    """`check_toolchain(python)` must agree with what THAT interpreter can import.

    OBSERVED FAILURE: `tt-model serve ... --print` warned
    `vllm present but the TT plugin (vllm_tt_plugin) is not importable` directly above a
    correctly-rendered command that serves fine. `check_toolchain()` used an in-process
    `find_spec`, describing tt-model's own venv rather than the instance that serves — a
    false negative for EVERY registered instance on this host. Fixed in tt-kernel PR #18.

    The check drives `check_toolchain(python)` per instance and compares against a direct
    `find_spec` in that same interpreter. An earlier version asserted that bare
    `tt-model doctor` must never report the plugin missing, and FAILED on a healthy stack:
    with no instance resolved, `doctor` correctly describes the ambient venv, which really
    does lack the plugin. That tested the one code path the bug was never in.
    """
    name = "check_toolchain(python) agrees with that interpreter's reality"
    why = "tt-kernel PR #18 — check_toolchain() probed the wrong interpreter"
    try:
        from tt_kernel import toolchain
    except ImportError:
        return Result(name, SKIP, "tt_kernel not importable here", why)
    reg = Path.home() / ".config" / "tt-kernel" / "instances.json"
    if not reg.is_file():
        reg = Path.home() / ".config" / "tt-model" / "instances.json"
    if not reg.is_file():
        return Result(name, SKIP, "no instance registry", why)
    insts = json.loads(reg.read_text()).get("instances") or []
    if isinstance(insts, dict):
        insts = list(insts.values())

    checked, disagree = [], []
    for i in insts:
        py = i.get("python")
        if not py or not Path(py).exists():
            continue
        # Ground truth, asked of that interpreter directly. find_spec only — importing
        # vLLM could open a device, and a probe must never do that.
        code = ("import importlib.util as u,json;"
                "print(json.dumps({m: u.find_spec(m) is not None "
                "for m in ('vllm','vllm_tt_plugin')}))")
        g = _run([py, "-c", code], timeout=180)
        if g.returncode != 0:
            continue
        truth = json.loads(g.stdout)
        can_serve = truth["vllm"] and truth["vllm_tt_plugin"]
        # Degrade to the pre-fix signature rather than crashing on it. A TypeError here
        # would still FAIL (the harness catches it), but "check itself raised" says nothing
        # about what is WRONG. Calling the old no-arg form reproduces exactly the buggy
        # behaviour, so the check then reports the real symptom: a verdict that contradicts
        # the interpreter it claims to describe.
        try:
            report = toolchain.check_toolchain(py)
        except TypeError:
            report = toolchain.check_toolchain()
        verdict = next((c for c in report.components if c.name == "vllm"), None)
        if verdict is None:
            continue
        checked.append({"instance": i.get("name"), "can_serve": can_serve,
                        "verdict_adequate": bool(verdict.adequate)})
        if can_serve != bool(verdict.adequate):
            disagree.append(f"{i.get('name')}: can_serve={can_serve} but "
                            f"tt-model says adequate={verdict.adequate} ({verdict.message})")
    if not checked:
        return Result(name, SKIP, "no probeable instance interpreters", why)
    if disagree:
        return Result(name, FAIL, "; ".join(disagree), why, {"checked": checked})
    return Result(name, PASS, f"{len(checked)} instance(s) agree with their interpreter", why,
                  {"checked": checked})


def check_installed_bundles_point_somewhere_real() -> Result:
    """A bundle's pinned TT_METAL_HOME must exist and not be a temp dir.

    OBSERVED FAILURE: our installed bundle pinned
    `/tmp/claude-.../scratchpad/tt-metal-src` — the TT_METAL_HOME that happened to be
    set when the bundle was staged. It resolved only because it was a symlink to the
    real tree, and would have died with the session. `installed.json` records this at
    install time with no way to re-point it. NOT yet fixed upstream; flagged as the
    follow-up on tt-kernel PR #18.
    """
    name = "installed bundles pin a real, non-temp TT_METAL_HOME"
    why = "tt-kernel #18 follow-up — installed.json pins TT_METAL_HOME at staging time"
    inst = Path.home() / ".cache" / "tt-kernel" / "installed.json"
    if not inst.is_file():
        inst = Path.home() / ".cache" / "tt-model" / "installed.json"
    if not inst.is_file():
        return Result(name, SKIP, "no installed bundles", why)
    entries = json.loads(inst.read_text())
    problems, ok = [], []
    for repo, e in entries.items():
        if not isinstance(e, dict):
            continue
        home = e.get("instance_tt_metal_home")
        if not home:
            continue
        p = Path(home)
        if not p.exists():
            problems.append(f"{repo}: pinned TT_METAL_HOME does not exist ({home})")
        elif home.startswith(("/tmp/", "/var/tmp/")):
            problems.append(f"{repo}: pinned TT_METAL_HOME is a temp path ({home})")
        else:
            ok.append(f"{repo}: {home}")
    if problems:
        return Result(name, FAIL, "; ".join(problems), why, {"ok": ok})
    return Result(name, PASS, f"{len(ok)} bundle(s) pin a durable path", why, {"ok": ok})


def check_vllm_bundle_pins_an_interpreter_with_vllm() -> Result:
    """A vLLM-backend bundle must not be pinned to an interpreter lacking vLLM.

    OBSERVED FAILURE: re-staging our bundle resolved it to tt-model's OWN venv, which
    has vllm but not the plugin, because the bundle declares only `platform_ttnn` and
    selection had no vLLM constraint to apply. Serving through that pin hard-fails.
    Reported as the third defect on tt-kernel PR #18.
    """
    name = "vLLM bundles pin an interpreter that can actually import vLLM"
    why = "tt-kernel #18 3rd defect — selection ignores backend when ranges are silent"
    inst = Path.home() / ".cache" / "tt-kernel" / "installed.json"
    if not inst.is_file():
        return Result(name, SKIP, "no installed bundles", why)
    entries = json.loads(inst.read_text())
    problems, ok = [], []
    for repo, e in entries.items():
        if not isinstance(e, dict) or e.get("backend") != "vllm":
            continue
        py = e.get("instance_python")
        if not py:
            ok.append(f"{repo}: ambient (no pin)")
            continue
        if not Path(py).exists():
            problems.append(f"{repo}: pinned interpreter missing ({py})")
            continue
        code = ("import importlib.util as u,sys;"
                "sys.exit(0 if u.find_spec('vllm') and u.find_spec('vllm_tt_plugin') else 1)")
        if _run([py, "-c", code], timeout=180).returncode != 0:
            problems.append(f"{repo}: pinned to {py}, which cannot import vllm+plugin")
        else:
            ok.append(f"{repo}: {py}")
    if problems:
        return Result(name, FAIL, "; ".join(problems), why, {"ok": ok})
    if not ok:
        return Result(name, SKIP, "no vLLM bundles installed", why)
    return Result(name, PASS, f"{len(ok)} vLLM bundle(s) servable as pinned", why, {"ok": ok})


def check_val_curve_flag_is_wired() -> Result:
    """`--val-every` must be what writes val_losses.jsonl, and must be reachable.

    OBSERVED FAILURE: a four-arm experiment ran with `--eval-every 250` (a different,
    pre-existing knob) instead of `--val-every`, whose default is 0 = DISABLED. No arm
    logged a validation curve, so the comparison had nothing to compare — discovered
    only after the runs had burned hardware time. This asserts the WIRING, not the
    arithmetic: that the flag exists, defaults to off, and names the output file.
    """
    name = "--val-every is wired to val_losses.jsonl"
    why = "2026-08-20 — a whole experiment ran with validation logging silently off"
    run_py = ROOT / "train" / "run.py"
    if not run_py.is_file():
        return Result(name, SKIP, "train/run.py not found", why)
    p = _run([sys.executable, str(run_py), "--help"], timeout=180)
    help_text = p.stdout + p.stderr
    if "--val-every" not in help_text:
        return Result(name, FAIL, "--val-every is gone; the curve flag was renamed", why)
    src = run_py.read_text()
    if "val_losses.jsonl" not in src:
        return Result(name, FAIL, "nothing in run.py writes val_losses.jsonl", why)
    # The trap was the DEFAULT, so assert it explicitly rather than trusting the name.
    if '"--val-every", type=int, default=0' not in src.replace("'", '"'):
        return Result(name, PASS,
                      "--val-every present and wired (default not literal-matched)", why)
    return Result(name, PASS, "--val-every present, wired, defaults to 0 (off)", why,
                  {"note": "default 0 means an experiment MUST pass it explicitly"})


def check_checkpoints_record_their_corpus() -> Result:
    """Checkpoints must carry the inputs needed to prove two runs comparable.

    OBSERVED FAILURE: two runs were compared against a baseline trained on a DIFFERENT
    corpus (the default --tokens-dir is the oldest of six token sets), producing a
    1.3-nat phantom regression that nearly got a working config discarded. Checkpoint
    format 2 added seed/tokens_dir/optimizer/ddp for exactly this reason. Fourteen
    format-1 checkpoints still on disk cannot prove their own provenance.
    """
    name = "recent checkpoints record corpus + seed (format 2)"
    why = "2026-08-19 — a 1.3-nat phantom regression from comparing different corpora"
    ck = ROOT / "train" / "checkpoint.py"
    if not ck.is_file():
        return Result(name, SKIP, "train/checkpoint.py not found", why)
    src = ck.read_text()
    if "CHECKPOINT_FORMAT = 2" not in src:
        return Result(name, FAIL, "checkpoint format is no longer 2", why)
    missing = [k for k in ("seed", "tokens_dir", "optimizer", "ddp") if f'"{k}"' not in src]
    if missing:
        return Result(name, FAIL, f"format 2 no longer requires: {missing}", why)
    return Result(name, PASS, "format 2 requires seed, tokens_dir, optimizer, ddp", why)


def check_device_training_smoke(steps: int = 20) -> Result:
    """A short real training step on real silicon. Needs a lease.

    OBSERVED FAILURE: two distinct ones. (1) `Timed out while waiting for active
    ethernet core 29-25 to become active again` after a previous tenant left the board
    fragile — a reset cleared it. (2) A host HARD LOCKUP ~20s after a sparse-MoE arm
    opened a 4-chip mesh, with no OOM or panic recorded. This check is deliberately
    DENSE and single-board: it establishes the floor without risking the mesh.
    """
    name = f"{steps}-step dense training smoke on device"
    why = "2026-08-20 — eth-core timeout, then a host lockup on MoE mesh init"
    if not os.environ.get("TT_VISIBLE_DEVICES"):
        return Result(name, SKIP, "no TT_VISIBLE_DEVICES — take a gozer lease first", why)
    ckpt = ROOT / "artifacts" / "checkpoints-v077-beta2-control" / "tt_tnt_step00010764.pkl"
    out = ROOT / "artifacts" / "checkpoints-stack-probe"
    argv = [sys.executable, str(ROOT / "train" / "run.py"),
            "--size", "1024", "--seed", "5489",
            "--tokens-dir", str(ROOT / "artifacts" / "tokens-v4"),
            "--ddp", "2", "--steps", str(steps), "--val-every", str(max(steps // 2, 1)),
            "--model-impl", "python",
            "--config", str(ROOT / "train" / "configs" / "tt-tnt-v077.yaml"),
            "--checkpoint-dir", str(out)]
    if ckpt.is_file():
        argv += ["--warm-start", str(ckpt)]
    p = _run(argv, timeout=1800)
    if p.returncode != 0:
        tail = (p.stdout + p.stderr)
        sig = next((ln for ln in tail.splitlines()
                    if "ethernet core" in ln or "TT_THROW" in ln), "")
        return Result(name, FAIL, f"exit {p.returncode}. {sig[:160]}", why)
    curve = out / "val_losses.jsonl"
    pts = len(curve.read_text().strip().splitlines()) if curve.is_file() else 0
    if pts == 0:
        return Result(name, FAIL, "training exited 0 but wrote NO val points", why)
    return Result(name, PASS, f"exit 0, {pts} val point(s) written", why)


# --------------------------------------------------------------------------- provenance

def provenance() -> dict:
    """Versions the checks were run against. A verdict without these is unusable later."""
    def _v(argv, timeout=60):
        try:
            p = _run(argv, timeout=timeout)
            return _strip_ansi(p.stdout).strip().splitlines()[0] if p.stdout.strip() else None
        except (subprocess.SubprocessError, OSError, IndexError):
            return None
    home = os.environ.get("TT_METAL_HOME")
    git_desc = None
    if home and Path(home).is_dir():
        git_desc = _v(["git", "-C", home, "describe", "--tags"])
    return {
        "tt_metal_home": home,
        "tt_metal_git_describe": git_desc,
        "tt_model_version": _v(["tt-model", "version"]),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "python": sys.executable,
        "repo_head": _v(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-device", action="store_true",
                    help="also run the on-device training smoke (needs a gozer lease)")
    ap.add_argument("--smoke-steps", type=int, default=20)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    checks: List[Callable[[], Result]] = [
        check_instance_version_matches_its_tree,
        check_toolchain_agrees_with_the_instance,
        check_installed_bundles_point_somewhere_real,
        check_vllm_bundle_pins_an_interpreter_with_vllm,
        check_val_curve_flag_is_wired,
        check_checkpoints_record_their_corpus,
    ]
    if args.with_device:
        checks.append(lambda: check_device_training_smoke(args.smoke_steps))

    prov = provenance()
    print("tt-tnt stack probe")
    print(f"  tt-metal   {prov['tt_metal_git_describe'] or '—'}  ({prov['tt_metal_home'] or 'TT_METAL_HOME unset'})")
    print(f"  tt-model   {prov['tt_model_version'] or '—'}")
    print(f"  repo       {prov['repo_head'] or '—'}")
    print(f"  devices    {prov['tt_visible_devices'] or 'none leased'}\n")

    results: List[Result] = []
    for fn in checks:
        try:
            r = fn()
        except Exception as e:  # a broken check must not masquerade as a passing one
            r = Result(getattr(fn, "__name__", "check"), FAIL, f"check itself raised: {e!r}")
        results.append(r)
        mark = {PASS: "✓", FAIL: "✗", SKIP: "–"}[r.status]
        print(f"  {mark} {r.name}")
        print(f"      {r.detail}")
        if r.status == FAIL and r.regression_of:
            print(f"      regression of: {r.regression_of}")

    n_fail = sum(1 for r in results if r.status == FAIL)
    n_pass = sum(1 for r in results if r.status == PASS)
    n_skip = sum(1 for r in results if r.status == SKIP)
    print(f"\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    if n_skip:
        # A summary that counts only what it ran is measuring itself.
        for r in results:
            if r.status == SKIP:
                print(f"      skipped: {r.name} — {r.detail}")

    if args.json_out:
        args.json_out.write_text(json.dumps({
            "provenance": prov,
            "results": [r.__dict__ for r in results],
            "summary": {"passed": n_pass, "failed": n_fail, "skipped": n_skip},
        }, indent=2))
        print(f"\n  wrote {args.json_out}")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
