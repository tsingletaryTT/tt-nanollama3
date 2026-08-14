#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Publish (or re-publish) the tt-tnt HF artifact to the Hugging Face Hub.

Uploads the current HF artifact directory (``HF_DIR``, see below -- it is *not*
``artifacts/hf`` any more) and applies ``docs/model-card.md`` as the model card to
``episod/tt-tnt``. Re-runnable by design, because it has to be run more than once:

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

  That flip has since happened, out-of-band, with explicit authorization: ``episod/tt-tnt``
  and ``episod/tt-tnt-corpus`` were both made public on 2026-08-14. This script's behavior
  did not change and does not need to -- ``create_repo(..., private=True, exist_ok=True)``
  only applies ``private=True`` when it actually creates a repo; per ``huggingface_hub``'s
  own docs, "this value is ignored if the repo already exists," so re-running the initial
  publish path against the now-public repo cannot silently flip it back. ``EXPECTED_PRIVATE``
  below records the current expectation for ``--verify`` rather than leaving it as a
  hardcoded assumption that would go stale the way this docstring almost did.
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

REPO_ID_DEFAULT = "episod/tt-tnt"
LICENSE = "apache-2.0"

#: The local HF artifact this script uploads -- i.e. the directory whose contents are
#: supposed to *be* ``episod/tt-tnt``.
#:
#: This was ``artifacts/hf`` up to and including the v2/256 publish. It is deliberately no
#: longer, and pointing it back would be a silent downgrade: ``artifacts/hf`` is the
#: protected, unregeneratable v2 baseline (``train/paths.py::PROTECTED_RELATIVE``) and
#: still holds ``max_position_embeddings: 256`` and the pre-blend tokenizer. The Hub now
#: holds tt-tnt-v1 (512 context, retrained tokenizer); re-running this script against
#: ``artifacts/hf`` would overwrite that with the older model, keeping the repo id and the
#: model card and changing only the weights. ``_assert_local_artifact_is_publishable``
#: below refuses that, so this constant is checked rather than merely believed.
HF_DIR = ROOT / "artifacts" / "hf-tt-tnt-v1"
CARD_PATH = ROOT / "docs" / "model-card.md"

# What the round trip in --verify must find, and (for the context length) what the local
# artifact must be before it may be uploaded at all. These are not arbitrary: they are the
# properties the packaging plan calls out as the ones a serving stack would get wrong
# silently if the artifact were malformed (context length, weight tying, vocab, and the
# exact parameter count as a coarse "did the whole state dict actually load" check).
#
# CONTEXT LENGTH, 256 -> 512 (2026-08-14). This constant was deliberately held at 256 for
# a while after ``train/sizes.py`` moved to 512, on the stated reasoning that it "describes
# the currently-published Hub artifact; bump only once a model is actually
# retrained/republished at 512". That has now happened: ``episod/tt-tnt`` commit
# ``ef0a9a91`` ("tt-tnt-v1: first run on the nine-source corpus (10,787 steps, seq_len 512,
# val 4.2203)") publishes 512-context weights, the Hub's config.json reads
# ``max_position_embeddings: 512``, and its model.safetensors sha256 (``dbc46211...``)
# matches ``artifacts/hf-tt-tnt-v1/model.safetensors`` exactly. The constant still
# describes the currently-published artifact; the artifact is what changed.
#
# Note this is not a loosening: the value moved but the rule did not, and it now guards in
# BOTH directions -- ``--verify`` fails if the Hub ever stops being 512, and
# ``_assert_local_artifact_is_publishable`` refuses to upload a local directory that is not
# 512, which is exactly the 256-over-512 downgrade the HF_DIR note above describes.
EXPECTED_MAX_POSITION_EMBEDDINGS = 512
EXPECTED_TIE_WORD_EMBEDDINGS = True
EXPECTED_VOCAB_SIZE = 32000
EXPECTED_PARAM_COUNT = 22_025_088
PROMPT = "Once upon a time, there was a little"

# PRIVACY, False since 2026-08-14. The repo was created private (as this script still does
# for any repo it has to create fresh) and was later flipped public out-of-band, with
# explicit authorization -- not through this script, which has no code path that can do
# that (see test_publish_to_hub.py::test_source_never_sets_private_false and the module
# docstring). ``--verify`` checks the Hub against this constant so the expectation lives in
# one edited place rather than as a hardcoded ``True`` that would silently start failing --
# or worse, stop meaning anything -- the day visibility legitimately changed again.
EXPECTED_PRIVATE = False


def _artifact_files() -> list[Path]:
    """Files ``upload_folder`` would send, in a stable order for printing and testing."""
    if not HF_DIR.is_dir():
        raise FileNotFoundError(f"{HF_DIR} does not exist -- run scripts/convert_checkpoint.py first")
    return sorted(p for p in HF_DIR.iterdir() if p.is_file())


def _assert_local_artifact_is_publishable() -> None:
    """Refuse to upload a local artifact whose context length isn't what we claim to ship.

    ``--verify`` checks the artifact *after* it is on the Hub, which is too late to prevent
    a bad publish -- it only tells you one happened. This is the same constant applied
    before the write, so ``EXPECTED_MAX_POSITION_EMBEDDINGS`` guards the upload rather than
    only describing it.

    The failure it exists to stop is concrete and has a live trigger: several local
    directories in ``artifacts/`` are loadable HF models of this architecture
    (``artifacts/hf``, ``artifacts/384/hf``, ``artifacts/hf-v2-scratch``), they differ only
    in weights and a config field, and all of them would upload perfectly happily. Pointing
    ``HF_DIR`` at the wrong one produces a Hub repo that still has the right name, the
    right card, and the right shape -- and half the trained context.
    """
    import json

    config_path = HF_DIR / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{config_path} does not exist -- {HF_DIR} is not an HF model dir")

    config = json.loads(config_path.read_text())
    actual = config.get("max_position_embeddings")
    if actual != EXPECTED_MAX_POSITION_EMBEDDINGS:
        raise ValueError(
            f"{config_path} has max_position_embeddings={actual!r}, but this script "
            f"publishes {EXPECTED_MAX_POSITION_EMBEDDINGS}-context weights. Refusing to "
            f"upload: this is how a shorter-context model silently replaces a longer one "
            f"under the same repo id. Point HF_DIR at the right artifact, or update "
            f"EXPECTED_MAX_POSITION_EMBEDDINGS if the published context really is changing."
        )


def _print_upload_plan(repo_id: str) -> int:
    """Print the file list and total size that would be uploaded. Returns the total bytes."""
    files = _artifact_files()
    total = 0
    print(f"repo:    {repo_id} (private=True if newly created -- per huggingface_hub, "
          f"ignored if it already exists, so this cannot silently re-privatize an existing "
          f"public repo; license={LICENSE})")
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
    # Before the plan is even printed, so a wrong HF_DIR is reported by --dry-run too --
    # the preview is worth nothing if it happily previews an upload that must not happen.
    _assert_local_artifact_is_publishable()
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

    print(f"creating (or reusing) repo {repo_id} (private=True only takes effect if this "
          f"call actually creates it) ...")
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)

    print(f"setting repo-level license={LICENSE} ...")
    _set_license(repo_id)

    print(f"uploading {HF_DIR.relative_to(ROOT)} -> {repo_id} ...")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(HF_DIR),
        commit_message="Upload tt-tnt HF artifact (config, weights, tokenizer)",
    )

    print(f"applying model card from {CARD_PATH.relative_to(ROOT)} ...")
    _push_card(repo_id)

    print("re-asserting repo-level license (belt-and-suspenders after the card push) ...")
    _set_license(repo_id)

    _report_card_state(repo_id)
    print("done. Visibility is never changed by this script, in either direction.")
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

    print(f"loading {repo_id} fresh from the Hub (not from {HF_DIR.name}/) ...")
    tok = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(repo_id)
    model.eval()
    cfg = model.config

    # Labels are interpolated from the constants, never spelled out. A hardcoded
    # "max_position_embeddings == 256" next to a comparison against a constant that says
    # 512 is a check that lies in its own output -- and this file had exactly that until
    # the 512 bump, which is precisely when a reader most needs the label to be true.
    check(f"max_position_embeddings == {EXPECTED_MAX_POSITION_EMBEDDINGS}",
          cfg.max_position_embeddings == EXPECTED_MAX_POSITION_EMBEDDINGS,
          f"(got {cfg.max_position_embeddings})")
    check(f"tie_word_embeddings is {EXPECTED_TIE_WORD_EMBEDDINGS}",
          cfg.tie_word_embeddings is EXPECTED_TIE_WORD_EMBEDDINGS,
          f"(got {cfg.tie_word_embeddings})")
    check(f"vocab_size == {EXPECTED_VOCAB_SIZE}", cfg.vocab_size == EXPECTED_VOCAB_SIZE,
          f"(got {cfg.vocab_size})")

    n_params = sum(p.numel() for p in model.parameters())
    check(f"parameter count == {EXPECTED_PARAM_COUNT:,}", n_params == EXPECTED_PARAM_COUNT,
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
    check(f"repo private == {EXPECTED_PRIVATE}", info.private is EXPECTED_PRIVATE,
          f"(got {info.private})")

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
