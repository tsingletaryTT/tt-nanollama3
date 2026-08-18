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
import json
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

    This is the plan's hardest constraint for Task 2: the repo is created private, and
    flipping it public is a separate, explicitly-confirmed action (Task 4 Step 5) that does
    not belong in this file. That flip has since happened, out-of-band and with explicit
    authorization (see ``EXPECTED_PRIVATE`` in ``publish_to_hub.py``) -- which makes this
    guarantee more relevant, not less: nothing in here should be able to reverse that flip
    either. Scanning the source rather than the runtime behavior catches a future edit that
    adds a `--public` flag or a `private=False` call before it ever runs, not just before it
    runs against the real repo.
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


def test_artifact_files_matches_real_hf_dir():
    """The upload plan's file list must be exactly what's on disk in HF_DIR,
    non-recursively -- no silent drift between what this prints and what upload_folder
    actually sends.

    Reads ``publish_to_hub.HF_DIR`` rather than naming a directory: this test previously
    hardcoded ``artifacts/hf``, which meant it kept passing while agreeing with the script
    about a directory that had stopped being the published artifact.
    """
    files = publish_to_hub._artifact_files()
    expected = sorted(p.name for p in publish_to_hub.HF_DIR.iterdir() if p.is_file())
    assert [f.name for f in files] == expected
    assert len(files) > 0


def test_hf_dir_is_not_the_protected_v2_baseline():
    """HF_DIR must never point at ``artifacts/hf``.

    That directory is the protected, unregeneratable v2 baseline
    (``train/paths.py::PROTECTED_RELATIVE``) and still holds 256-context weights and the
    pre-blend tokenizer. Uploading it to ``episod/tt-tnt`` would replace the published
    512-context tt-tnt-v1 with an older model under the same repo id, same card, same
    shape -- a downgrade with nothing in the repo to signal it happened.
    """
    assert publish_to_hub.HF_DIR != ROOT / "artifacts" / "hf"
    assert publish_to_hub.HF_DIR.is_dir()


def test_local_artifact_context_length_matches_what_the_script_claims_to_publish():
    """The real artifact on disk must satisfy the pre-upload guard."""
    publish_to_hub._assert_local_artifact_is_publishable()
    config = json.loads((publish_to_hub.HF_DIR / "config.json").read_text())
    assert config["max_position_embeddings"] == publish_to_hub.EXPECTED_MAX_POSITION_EMBEDDINGS


def test_publish_refuses_an_artifact_with_the_wrong_context_length(tmp_path, monkeypatch):
    """The guard must actually fire, and must fire before anything reaches the Hub.

    Verified by pointing HF_DIR at a config that differs from the expected context in the
    one field that matters -- the 256-over-512 downgrade, reproduced.
    """
    def _boom(*a, **k):
        raise AssertionError("a refused publish must not contact the Hub")

    stale = tmp_path / "hf"
    stale.mkdir()
    (stale / "config.json").write_text(json.dumps({
        "max_position_embeddings": publish_to_hub.EXPECTED_MAX_POSITION_EMBEDDINGS // 2,
    }))
    monkeypatch.setattr(publish_to_hub, "HF_DIR", stale)
    monkeypatch.setattr(publish_to_hub, "_push_card", _boom)
    monkeypatch.setattr(publish_to_hub, "_set_license", _boom)

    with pytest.raises(ValueError, match="max_position_embeddings"):
        publish_to_hub.cmd_publish("episod/tt-tnt", dry_run=False, yes=True)
    # ...and --dry-run does not get a pass either: a preview that previews a forbidden
    # upload is worse than no preview.
    with pytest.raises(ValueError, match="max_position_embeddings"):
        publish_to_hub.cmd_publish("episod/tt-tnt", dry_run=True, yes=False)


def test_verify_labels_are_not_hardcoded_numbers():
    """``--verify``'s printed labels must interpolate the constants, not restate them.

    The bug this pins: the label spelled the number out while the comparison used a
    constant. Bumping the constant to 512 left the check correct and its own output wrong
    -- it reported passing a check named for 256 having just verified 512.

    Scans only the ``check(...)`` call lines, not the whole file: the prose above those
    calls necessarily quotes the old bad label to explain it, and a whole-file scan would
    fire on the explanation.
    """
    check_lines = [
        line for line in _SCRIPT_PATH.read_text().splitlines()
        if line.lstrip().startswith("check(")
    ]
    assert check_lines, "found no check(...) calls to inspect -- the scan is looking at nothing"
    for line in check_lines:
        assert "== 256" not in line
        assert "== 512" not in line
        assert "== 32000" not in line
        assert "22,025,088" not in line
    assert any("{EXPECTED_MAX_POSITION_EMBEDDINGS}" in line for line in check_lines)


def test_dry_run_publish_never_touches_the_hub(monkeypatch, capsys):
    """--dry-run must not import or call anything that reaches the network."""
    def _boom(*a, **k):
        raise AssertionError("dry-run must not contact the Hub")

    monkeypatch.setattr(publish_to_hub, "_push_card", _boom)
    monkeypatch.setattr(publish_to_hub, "_set_license", _boom)

    rc = publish_to_hub.cmd_publish("episod/tt-tnt", dry_run=True, yes=False)
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

    rc = publish_to_hub.cmd_restore_card("episod/tt-tnt", dry_run=True, yes=False)
    assert rc == 0
    assert "dry-run" in capsys.readouterr().out


def test_publish_without_yes_refuses_and_does_not_touch_the_hub(monkeypatch, capsys):
    def _boom(*a, **k):
        raise AssertionError("must not contact the Hub without --yes")

    monkeypatch.setattr(publish_to_hub, "_push_card", _boom)
    monkeypatch.setattr(publish_to_hub, "_set_license", _boom)

    rc = publish_to_hub.cmd_publish("episod/tt-tnt", dry_run=False, yes=False)
    assert rc != 0
    assert "--yes" in capsys.readouterr().err


def test_restore_card_without_yes_refuses_and_does_not_touch_the_hub(monkeypatch, capsys):
    def _boom(*a, **k):
        raise AssertionError("must not contact the Hub without --yes")

    monkeypatch.setattr(publish_to_hub, "_push_card", _boom)
    monkeypatch.setattr(publish_to_hub, "_set_license", _boom)

    rc = publish_to_hub.cmd_restore_card("episod/tt-tnt", dry_run=False, yes=False)
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


def test_upload_reads_the_same_directory_the_guard_validated(monkeypatch, tmp_path):
    """The bytes uploaded must come from the directory the assertions checked.

    This is the invariant that was missing when episod/tt-tnt-1024 was first
    published: ``_assert_local_artifact_is_publishable`` and the printed upload
    plan were routed through the per-repo target, but ``upload_folder`` still read
    the module-level ``HF_DIR``. The guard validated a 1024-dim artifact while the
    uploader sent the 384-dim one, and every existing test passed, because none of
    them related the two.
    """


    seen = {}

    class FakeApi:
        def create_repo(self, *a, **k):
            pass

        def upload_folder(self, *, repo_id, repo_type, folder_path, commit_message):
            seen["folder"] = Path(folder_path)

        def upload_file(self, *a, **k):
            pass

        def model_info(self, *a, **k):
            class I:
                private = False
            return I()

    # HfApi is imported inside the function, so patch it at its source module.
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: FakeApi())
    monkeypatch.setattr(publish_to_hub, "_set_license", lambda *a, **k: None)
    monkeypatch.setattr(publish_to_hub, "_push_card", lambda *a, **k: None)
    monkeypatch.setattr(publish_to_hub, "_assert_local_artifact_is_publishable",
                        lambda *a, **k: None)

    for repo_id in sorted(publish_to_hub.TARGETS):
        seen.clear()
        publish_to_hub.cmd_publish(repo_id, dry_run=False, yes=True)
        expected = publish_to_hub.target_for(repo_id)["hf_dir"]
        assert seen["folder"] == expected, (
            f"{repo_id}: uploaded {seen['folder']}, but its declared artifact is "
            f"{expected}. The uploader and the guard must read one directory."
        )
