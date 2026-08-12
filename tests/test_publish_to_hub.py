# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""scripts/publish_to_hub.py: exercises everything that doesn't require the network.

Hub-dependent behavior (repo creation, upload, the --verify round trip) is covered by
running the script for real and reporting its output, per the packaging plan's rule that
Hub-dependent verification may be a script rather than a test. What *is* tested here, with
no network access, is the safety-load-bearing part: dry-run never reaches the Hub, writes
require --yes, the file listing matches the real artifact, and the card-loading fix for the
leading-HTML-comment gotcha actually parses front matter (rather than the silent-empty-card
failure mode this suite exists to catch).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Loaded by file path, matching this repo's convention (see test_backfill_checkpoint_headers.py):
# an unrelated project's own `scripts/__init__.py` earlier on sys.path would otherwise shadow
# a bare `import scripts.publish_to_hub`.
_SCRIPT_PATH = ROOT / "scripts" / "publish_to_hub.py"
_spec = importlib.util.spec_from_file_location("publish_to_hub", _SCRIPT_PATH)
publish_to_hub = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(publish_to_hub)


def test_source_never_sets_private_false():
    """There must be no code path in this script that can flip the repo public.

    This is the plan's hardest constraint for Task 2: the repo is created private and
    stays private until a separate, explicitly-confirmed action (Task 4 Step 5). Scanning
    the source rather than the runtime behavior catches a future edit that adds a
    `--public` flag or a `private=False` call before it ever runs, not just before it runs
    against the real repo.
    """
    source = _SCRIPT_PATH.read_text()
    assert "private=False" not in source
    assert "private = False" not in source
    # Checks the argparse wiring specifically, not the docstring's prose (which mentions
    # "no --public flag" by name as documentation of this very guarantee).
    assert 'add_argument("--public"' not in source
    with pytest.raises(SystemExit):
        publish_to_hub.main(["--public"])
    # And confirm the positive: repo creation really does pin private=True.
    assert "private=True" in source


def test_artifact_files_matches_real_artifacts_hf_dir():
    """The upload plan's file list must be exactly what's on disk in artifacts/hf/,
    non-recursively -- no silent drift between what this prints and what upload_folder
    actually sends."""
    files = publish_to_hub._artifact_files()
    expected = sorted(p.name for p in (ROOT / "artifacts" / "hf").iterdir() if p.is_file())
    assert [f.name for f in files] == expected
    assert len(files) > 0


def test_dry_run_publish_never_touches_the_hub(monkeypatch, capsys):
    """--dry-run must not import or call anything that reaches the network."""
    def _boom(*a, **k):
        raise AssertionError("dry-run must not contact the Hub")

    monkeypatch.setattr(publish_to_hub, "_push_card", _boom)
    monkeypatch.setattr(publish_to_hub, "_set_license", _boom)

    rc = publish_to_hub.cmd_publish("episod/tt-nanollama3", dry_run=True, yes=False)
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "model.safetensors" in out


def test_dry_run_restore_card_never_touches_the_hub(monkeypatch, capsys):
    def _boom(*a, **k):
        raise AssertionError("dry-run must not contact the Hub")

    monkeypatch.setattr(publish_to_hub, "_push_card", _boom)
    monkeypatch.setattr(publish_to_hub, "_set_license", _boom)
    monkeypatch.setattr(publish_to_hub, "_report_card_state", _boom)

    rc = publish_to_hub.cmd_restore_card("episod/tt-nanollama3", dry_run=True, yes=False)
    assert rc == 0
    assert "dry-run" in capsys.readouterr().out


def test_publish_without_yes_refuses_and_does_not_touch_the_hub(monkeypatch, capsys):
    def _boom(*a, **k):
        raise AssertionError("must not contact the Hub without --yes")

    monkeypatch.setattr(publish_to_hub, "_push_card", _boom)
    monkeypatch.setattr(publish_to_hub, "_set_license", _boom)

    rc = publish_to_hub.cmd_publish("episod/tt-nanollama3", dry_run=False, yes=False)
    assert rc != 0
    assert "--yes" in capsys.readouterr().err


def test_restore_card_without_yes_refuses_and_does_not_touch_the_hub(monkeypatch, capsys):
    def _boom(*a, **k):
        raise AssertionError("must not contact the Hub without --yes")

    monkeypatch.setattr(publish_to_hub, "_push_card", _boom)
    monkeypatch.setattr(publish_to_hub, "_set_license", _boom)

    rc = publish_to_hub.cmd_restore_card("episod/tt-nanollama3", dry_run=False, yes=False)
    assert rc != 0
    assert "--yes" in capsys.readouterr().err


def test_load_card_for_hub_parses_front_matter_despite_leading_html_comments():
    """The real bug this test exists to catch: docs/model-card.md leads with an HTML
    comment block *before* the '---' front-matter fence (deliberately, so a maintainer
    reading the file sees the explanation first). huggingface_hub.ModelCard's front-matter
    regex is anchored to the start of the string and does NOT raise when it finds nothing
    there -- it silently returns an empty CardData. `ModelCard.load(str(CARD_PATH))` on the
    raw file therefore parses to license=None, tags=None, etc. This test proves
    `_load_card_for_hub` strips the leading comment and actually recovers the metadata."""
    card = publish_to_hub._load_card_for_hub()
    assert card.data.license == "apache-2.0"
    assert card.data.pipeline_tag == "text-generation"
    assert card.data.library_name == "transformers"
    assert "roneneldan/TinyStories" in (card.data.datasets or [])
    assert card.content.startswith("---")


def test_load_card_for_hub_raises_on_missing_front_matter(tmp_path, monkeypatch):
    """A card with no front-matter fence must fail loudly, not push something empty."""
    bad_card = tmp_path / "model-card.md"
    bad_card.write_text("# No front matter here\n\nJust prose.\n")
    monkeypatch.setattr(publish_to_hub, "CARD_PATH", bad_card)
    with pytest.raises(ValueError, match="front-matter fence"):
        publish_to_hub._load_card_for_hub()


def test_verify_requires_no_other_flags(capsys):
    rc = publish_to_hub.main(["--verify", "--dry-run"])
    assert rc != 0
    assert "read-only" in capsys.readouterr().err
