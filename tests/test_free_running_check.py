# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""scripts/free_running_check.py: the parts that do not need a server or a device.

What is worth testing here is not the comparison arithmetic (it is six lines) but the
**reference-selection guard**, because that is where this script can produce numbers that
look like evidence and are not.

The script's whole output is a token-by-token comparison of device generation against a
local CPU reference. If the reference is a different model from the one being served, every
row still fills in, the medians still print, and the result is meaningless -- it measures
the gap between two models rather than the gap between two execution paths. That happened
for real: the default reference resolved to ``artifacts/384/hf`` (the v2 model, 256
context) while the Hub had been republished at 512, so the default silently pointed at the
wrong model.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from conftest import needs_artifacts

ROOT = Path(__file__).resolve().parent.parent

# Loaded by file path, matching this repo's convention (see test_publish_to_hub.py): an
# unrelated project's own `scripts/__init__.py` earlier on sys.path would otherwise shadow
# a bare `import scripts.free_running_check`.
_SCRIPT_PATH = ROOT / "scripts" / "free_running_check.py"
_spec = importlib.util.spec_from_file_location("free_running_check", _SCRIPT_PATH)
frc = importlib.util.module_from_spec(_spec)
sys.modules["free_running_check"] = frc
_spec.loader.exec_module(frc)


def _write_config(directory: Path, ctx: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"max_position_embeddings": ctx}))
    return directory


def _fake_models_endpoint(monkeypatch, payload):
    """Stand in for urlopen so the guard can be tested with no server running."""
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps(payload).encode()

    monkeypatch.setattr(frc.urllib.request, "urlopen", lambda *a, **k: _Resp())


def _load_publish_to_hub():
    """``scripts/publish_to_hub.py``, loaded by path for the same shadowing reason as above."""
    path = ROOT / "scripts" / "publish_to_hub.py"
    spec = importlib.util.spec_from_file_location("publish_to_hub_for_frc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@needs_artifacts("artifacts/hf-tt-tnt-v3")
def test_default_reference_is_the_artifact_that_publish_to_hub_uploads():
    """The default reference must be the same directory the publish script uploads.

    Stated against ``publish_to_hub.HF_DIR`` rather than a literal path on purpose. Both
    constants name "the currently-published artifact", and they have now gone stale
    together twice -- ``artifacts/384/hf`` (v2, 256) while the Hub served 512, then
    ``artifacts/hf-tt-tnt-v1`` (512) while the Hub served v3's 2048. Each time, the
    script kept producing a full table of agreement numbers that measured the gap between
    two *models* rather than between two *execution paths*.

    A literal assertion here would have to be edited at every republish and would pass
    the moment someone edited it, which is no gate at all. Tying the two constants
    together means the next republish cannot move one without the other.
    """
    default = frc._default_hf_dir()
    assert default == _load_publish_to_hub().HF_DIR, (
        "the CPU reference and the uploaded artifact must be the same directory"
    )
    assert default != ROOT / "artifacts" / "384" / "hf", "that is the v2/256 baseline"
    assert default != ROOT / "artifacts" / "hf", "that is the protected v2 baseline"
    config = json.loads((default / "config.json").read_text())
    assert config["max_position_embeddings"] == 2048


def test_guard_accepts_a_matching_reference(tmp_path, monkeypatch):
    _fake_models_endpoint(monkeypatch, {"data": [{"id": "m", "max_model_len": 512}]})
    frc._check_reference_matches_server(_write_config(tmp_path / "hf", 512), "http://x", "m", 1.0)


def test_guard_refuses_a_shorter_context_reference(tmp_path, monkeypatch):
    """The real failure: a 256-context reference against 512-context served weights."""
    _fake_models_endpoint(monkeypatch, {"data": [{"id": "m", "max_model_len": 512}]})
    with pytest.raises(frc.ReferenceMismatch, match="different models"):
        frc._check_reference_matches_server(
            _write_config(tmp_path / "hf", 256), "http://x", "m", 1.0
        )


def test_guard_refuses_when_the_server_does_not_serve_the_named_model(tmp_path, monkeypatch):
    _fake_models_endpoint(monkeypatch, {"data": [{"id": "other", "max_model_len": 512}]})
    with pytest.raises(frc.ReferenceMismatch, match="does not serve"):
        frc._check_reference_matches_server(
            _write_config(tmp_path / "hf", 512), "http://x", "m", 1.0
        )


def test_guard_refuses_a_directory_that_is_not_a_model(tmp_path):
    with pytest.raises(frc.ReferenceMismatch, match="not an HF model dir"):
        frc._check_reference_matches_server(tmp_path / "nope", "http://x", "m", 1.0)


def test_agreement_counts_leading_matches_only():
    """A later coincidental match must not be credited after a divergence."""
    assert frc.agreement(["a", "b", "c"], ["a", "b", "c"]) == 3
    assert frc.agreement(["a", "b", "c"], ["a", "x", "c"]) == 1
    assert frc.agreement(["a"], ["x"]) == 0
    assert frc.agreement([], []) == 0
