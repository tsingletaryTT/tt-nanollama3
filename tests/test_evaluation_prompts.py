# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The prompt set is FROZEN. These tests protect comparability across checkpoints."""
import hashlib
import json
from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[1] / "docs" / "evaluation_prompts.json"
REQUIRED_PROBES = {"target-voice", "stutter", "oracular", "agentic",
                   "grounding", "perpendicular", "coherence"}

#: Digest over the sorted (id, text) pairs. Rewriting every prompt's ``text`` to garbage
#: used to leave this suite green: ids, probes and count were pinned, and the CONTENT --
#: the only thing that makes two checkpoints' outputs comparable -- was not. A prompt set
#: whose ids are stable but whose text drifts is worse than an unfrozen one, because the
#: results still LOOK comparable.
#:
#: Changing a prompt is therefore a two-file edit: the prompt and this constant, in a
#: commit that says why comparability with earlier samples is being given up. Adding a
#: prompt with a NEW id is the supported move (see the note in the JSON) and also lands
#: here, which is intended -- a longer prompt set is still a different prompt set.
FROZEN_DIGEST = "33c221cb9d0379dc9957d2646016e38daf6d8e9acbfe30f0e905291bfe92b025"
FROZEN_COUNT = 15


def _digest(prompts) -> str:
    """SHA-256 over sorted (id, text) pairs, NUL-separated so no rearrangement of
    characters between the two fields can collide with another set."""
    h = hashlib.sha256()
    for pid, text in sorted((p["id"], p["text"]) for p in prompts):
        h.update(pid.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def test_prompt_file_parses():
    data = json.loads(PROMPTS.read_text())
    assert data["prompts"]


def test_ids_are_unique():
    ids = [p["id"] for p in json.loads(PROMPTS.read_text())["prompts"]]
    assert len(ids) == len(set(ids))


def test_every_required_probe_is_present():
    probes = {p["probe"] for p in json.loads(PROMPTS.read_text())["prompts"]}
    missing = REQUIRED_PROBES - probes
    assert not missing, f"prompt set is missing probes: {sorted(missing)}"


def test_stutter_probe_exists_because_this_model_has_a_repetition_defect():
    prompts = json.loads(PROMPTS.read_text())["prompts"]
    assert [p for p in prompts if p["probe"] == "stutter"], (
        "the stutter probe distinguishes 'learned Stein' from 'learned to repeat'"
    )


def test_no_prompt_is_empty_or_whitespace():
    for p in json.loads(PROMPTS.read_text())["prompts"]:
        assert p["text"].strip()


def test_prompt_text_is_frozen_not_just_the_ids():
    """THE GAP: rewriting every text field to garbage left this suite green."""
    prompts = json.loads(PROMPTS.read_text())["prompts"]
    assert len(prompts) == FROZEN_COUNT
    assert _digest(prompts) == FROZEN_DIGEST, (
        "the frozen prompt set changed. Checkpoint samples generated before this edit are "
        "no longer comparable with ones generated after it. If that is intended, update "
        "FROZEN_DIGEST and FROZEN_COUNT in the same commit and say why."
    )


def test_the_digest_actually_detects_a_rewritten_prompt():
    """A digest test that cannot fail is worse than no digest test."""
    prompts = json.loads(PROMPTS.read_text())["prompts"]
    tampered = [dict(p) for p in prompts]
    tampered[0]["text"] = tampered[0]["text"] + " and then everything was different"
    assert _digest(tampered) != FROZEN_DIGEST


def test_the_digest_does_not_depend_on_file_order():
    """Reordering the file is a cosmetic change; it must not read as a content change."""
    prompts = json.loads(PROMPTS.read_text())["prompts"]
    assert _digest(list(reversed(prompts))) == FROZEN_DIGEST
