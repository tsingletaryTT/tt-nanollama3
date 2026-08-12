#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Publish (or re-publish) the tt-nanollama3 HF artifact to the Hugging Face Hub.

Uploads ``artifacts/hf/`` and applies ``docs/model-card.md`` as the model card to
``episod/tt-nanollama3``. Re-runnable by design, because it has to be run more than once:

* Once, to do the initial publish (default action, below).
* Every time ``tt-kernel push`` runs against this repo, because tt-kernel's ``tag_repo``
  (``src/tt_kernel/hub.py:56-66``) replaces the card's front matter with
  ``ModelCardData(tags=...)`` and *nothing else* -- it destroys ``license``,
  ``pipeline_tag``, ``library_name``, and ``datasets`` on every push. ``--restore-card``
  re-applies ``docs/model-card.md`` after that damage (see plan Task 4 Step 2).

Safety rules baked into this script, not left to the caller's discipline:

* The repo is created private and is NEVER flipped public here. There is no ``--public``
  flag. Flipping visibility is a separate, explicitly-confirmed action (plan Task 4 Step 5)
  and does not belong in a re-runnable publish script.
* Any action that writes to the Hub (initial publish, ``--restore-card``) requires ``--yes``.
  ``--dry-run`` never touches the Hub, regardless of ``--yes``.
* ``--verify`` is read-only: it round-trips the *published* copy through ``transformers``,
  not local state, so it actually proves what a downstream user would get.

A note on the "repo-level license" the packaging plan asks for: this script also calls
``huggingface_hub.metadata_update()`` right after repo creation, before any card exists, so
the license is set via a dedicated metadata API rather than living solely inside the prose
file this script uploads. Measured directly against a disposable scratch repo before writing
this: that call is NOT independent of card front matter under the hood -- the Hub stores
license only as part of the README's YAML block, so a `tag_repo`-style full front-matter
replacement wipes it exactly like everything else. The real defense is `--restore-card`
after every tt-kernel operation, not the order tags are set in. This script sets the license
early anyway (belt-and-suspenders, and it does make the repo well-formed before the full
card exists) but does not claim it survives what only ``--restore-card`` can fix.

Usage:

    python scripts/publish_to_hub.py --dry-run              # preview the initial publish
    python scripts/publish_to_hub.py --yes                  # do the initial publish
    python scripts/publish_to_hub.py --restore-card --dry-run
    python scripts/publish_to_hub.py --restore-card --yes   # re-apply the card after tt-kernel push
    python scripts/publish_to_hub.py --verify                # round-trip check against the Hub
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPO_ID_DEFAULT = "episod/tt-nanollama3"
LICENSE = "apache-2.0"
HF_DIR = ROOT / "artifacts" / "hf"
CARD_PATH = ROOT / "docs" / "model-card.md"

# What the round trip in --verify must find. These are not arbitrary: they are the
# properties the packaging plan calls out as the ones a serving stack would get wrong
# silently if the artifact were malformed (context length, weight tying, vocab, and the
# exact parameter count as a coarse "did the whole state dict actually load" check).
EXPECTED_MAX_POSITION_EMBEDDINGS = 256
EXPECTED_TIE_WORD_EMBEDDINGS = True
EXPECTED_VOCAB_SIZE = 32000
EXPECTED_PARAM_COUNT = 22_025_088
PROMPT = "Once upon a time, there was a little"


def _artifact_files() -> list[Path]:
    """Files ``upload_folder`` would send, in a stable order for printing and testing."""
    if not HF_DIR.is_dir():
        raise FileNotFoundError(f"{HF_DIR} does not exist -- run scripts/convert_checkpoint.py first")
    return sorted(p for p in HF_DIR.iterdir() if p.is_file())


def _print_upload_plan(repo_id: str) -> int:
    """Print the file list and total size that would be uploaded. Returns the total bytes."""
    files = _artifact_files()
    total = 0
    print(f"repo:    {repo_id} (private=True, license={LICENSE})")
    print(f"card:    {CARD_PATH.relative_to(ROOT)}")
    print("files:")
    for f in files:
        size = f.stat().st_size
        total += size
        print(f"  {f.name:30s} {size:>12,} B")
    print(f"total:   {total:,} B ({total / 1e6:.2f} MB)")
    return total


def _load_card_for_hub():
    """Load ``docs/model-card.md`` as a ``ModelCard`` fit to push to the Hub.

    ``docs/model-card.md`` intentionally leads with an HTML-comment explanation (SPDX
    headers, and a note on why the file exists) *before* the YAML front-matter fence, so a
    maintainer opening the file in an editor sees the explanation first. That trips a real
    gotcha: ``huggingface_hub.ModelCard``'s front-matter regex is anchored to the very start
    of the string (``^\\s*---``, no ``re.MULTILINE``), so ``ModelCard.load()`` on the raw
    file finds no metadata block -- and, worse, does not raise. It logs a warning and
    silently returns an EMPTY ``CardData`` (confirmed directly: ``card.data.license`` comes
    back ``None`` from the unmodified file). Pushing that would be worse than doing nothing.

    The fix: find the first front-matter fence and construct the card from that point
    onward, matching what the Hub actually needs (front matter must lead the README there
    too). Fail loudly, not silently, if that fence is missing or license didn't parse.
    """
    from huggingface_hub import ModelCard

    raw = CARD_PATH.read_text()
    stripped = raw.lstrip()
    if stripped.startswith("---"):
        content = stripped
    else:
        idx = raw.find("\n---")
        if idx == -1:
            raise ValueError(f"{CARD_PATH}: no YAML front-matter fence ('---') found; "
                              "refusing to push a card with no metadata")
        content = raw[idx + 1:]

    card = ModelCard(content)
    if card.data.license is None:
        raise ValueError(f"{CARD_PATH}: parsed card has no `license` in front matter after "
                          "stripping leading comments -- refusing to push what looks like an "
                          "empty card")
    return card


def _set_license(repo_id: str) -> None:
    """Set the repo license via a dedicated metadata call, not by hoping the card sticks."""
    from huggingface_hub import metadata_update

    metadata_update(repo_id, {"license": LICENSE}, repo_type="model", overwrite=True)


def _push_card(repo_id: str) -> None:
    card = _load_card_for_hub()
    card.push_to_hub(repo_id, repo_type="model")


def _report_card_state(repo_id: str) -> None:
    """Print what front-matter fields are actually present on the Hub right now.

    This is the check the packaging plan asks for after every tt-kernel operation:
    "verify front matter after every tt-kernel operation, and restore what was lost."
    """
    from huggingface_hub import ModelCard

    card = ModelCard.load(repo_id, repo_type="model")
    print("current card front matter on the Hub:")
    for field in ("license", "library_name", "pipeline_tag", "datasets", "tags"):
        print(f"  {field}: {getattr(card.data, field, None)!r}")


def cmd_publish(repo_id: str, dry_run: bool, yes: bool) -> int:
    """Create the repo (private), set the license, upload the artifact, apply the card."""
    _print_upload_plan(repo_id)

    if dry_run:
        print("[dry-run] no repo created, nothing uploaded, no card pushed.")
        return 0

    if not yes:
        print("refusing to publish without --yes (use --dry-run to preview safely)",
              file=sys.stderr)
        return 2

    from huggingface_hub import HfApi

    api = HfApi()

    print(f"creating (or reusing) PRIVATE repo {repo_id} ...")
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)

    print(f"setting repo-level license={LICENSE} ...")
    _set_license(repo_id)

    print(f"uploading {HF_DIR.relative_to(ROOT)} -> {repo_id} ...")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(HF_DIR),
        commit_message="Upload tt-nanollama3 HF artifact (config, weights, tokenizer)",
    )

    print(f"applying model card from {CARD_PATH.relative_to(ROOT)} ...")
    _push_card(repo_id)

    print("re-asserting repo-level license (belt-and-suspenders after the card push) ...")
    _set_license(repo_id)

    _report_card_state(repo_id)
    print("done. Repo remains PRIVATE -- visibility is never changed by this script.")
    return 0


def cmd_restore_card(repo_id: str, dry_run: bool, yes: bool) -> int:
    """Re-apply docs/model-card.md, for use after a tt-kernel push damages front matter."""
    print(f"repo:    {repo_id}")
    print(f"card:    {CARD_PATH.relative_to(ROOT)}")

    if dry_run:
        print(f"[dry-run] would push card from {CARD_PATH.relative_to(ROOT)} to {repo_id}, "
              f"then re-set license={LICENSE}. No changes made.")
        return 0

    if not yes:
        print("refusing to push without --yes (use --dry-run to preview safely)",
              file=sys.stderr)
        return 2

    _push_card(repo_id)
    _set_license(repo_id)
    _report_card_state(repo_id)
    print("card restored.")
    return 0


def cmd_verify(repo_id: str) -> int:
    """Round-trip verification from the Hub, not local state. Read-only."""
    from huggingface_hub import HfApi, ModelCard
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checks: list[bool] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}{('  ' + detail) if detail else ''}")
        checks.append(bool(cond))

    print(f"loading {repo_id} fresh from the Hub (not from artifacts/hf/) ...")
    tok = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(repo_id)
    model.eval()
    cfg = model.config

    check("max_position_embeddings == 256", cfg.max_position_embeddings == EXPECTED_MAX_POSITION_EMBEDDINGS,
          f"(got {cfg.max_position_embeddings})")
    check("tie_word_embeddings is True", cfg.tie_word_embeddings is EXPECTED_TIE_WORD_EMBEDDINGS,
          f"(got {cfg.tie_word_embeddings})")
    check("vocab_size == 32000", cfg.vocab_size == EXPECTED_VOCAB_SIZE,
          f"(got {cfg.vocab_size})")

    n_params = sum(p.numel() for p in model.parameters())
    check("parameter count == 22,025,088", n_params == EXPECTED_PARAM_COUNT,
          f"(got {n_params:,})")

    import torch

    ids = tok(PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=40, do_sample=True, temperature=0.8, top_p=0.95)
    text = tok.decode(out[0], skip_special_tokens=True)
    print(f"sample generation: {text!r}")
    # .strip() before comparing: the tokenizer's BPE decode adds a leading space before the
    # first token (a normal artifact of space-prefixed byte-level BPE, not a defect), so the
    # raw decoded string is " Once upon..." rather than "Once upon...". Stripping avoids a
    # false failure on that whitespace while still requiring the model to have reproduced
    # the prompt and appended new tokens after it.
    check("generation extended the prompt",
          len(text) > len(PROMPT) and text.strip().startswith(PROMPT[:10]))

    api = HfApi()
    info = api.model_info(repo_id)
    check("repo is private", info.private is True, f"(got {info.private})")

    card = ModelCard.load(repo_id, repo_type="model")
    check("card front matter has license == apache-2.0", getattr(card.data, "license", None) == LICENSE,
          f"(got {getattr(card.data, 'license', None)!r})")

    if not all(checks):
        print("one or more checks FAILED -- stopping. Do not re-upload blindly; diagnose first.",
              file=sys.stderr)
        return 1
    print("all checks passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=REPO_ID_DEFAULT,
                   help=f"Target model repo (default: {REPO_ID_DEFAULT}).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen; never contacts the Hub for writes.")
    p.add_argument("--yes", action="store_true",
                   help="Required to actually write to the Hub (publish or --restore-card).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--restore-card", action="store_true",
                       help="Only re-apply docs/model-card.md as the repo card (use after "
                            "`tt-kernel push` damages front matter). Skips repo creation "
                            "and weight upload.")
    mode.add_argument("--verify", action="store_true",
                       help="Read-only round-trip check: load the published model+tokenizer "
                            "fresh from the Hub via transformers and assert key fields.")
    args = p.parse_args(argv)

    if args.verify:
        if args.dry_run or args.yes or args.restore_card:
            print("--verify is read-only and takes no other flags", file=sys.stderr)
            return 2
        return cmd_verify(args.repo_id)

    if args.restore_card:
        return cmd_restore_card(args.repo_id, dry_run=args.dry_run, yes=args.yes)

    return cmd_publish(args.repo_id, dry_run=args.dry_run, yes=args.yes)


if __name__ == "__main__":
    raise SystemExit(main())
