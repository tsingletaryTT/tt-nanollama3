# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Prompt set B is FROZEN, exactly as set A is. These tests protect comparability.

Set B exists because power on set A is capped by the PROMPT count, not the sample count
(``docs/measurements/behaviour-tt-tnt-v1-vs-v3.md``, "Power, and why more samples will not help
much"). It is a SEPARATE set with its own ids, its own file and its own digest, reported beside
set A and never merged into it.

Every claim below is checked by PARSING the file, never by searching it for a substring: a
substring search passes on a file that says the right words in the wrong structure, which is the
failure mode that has already shipped twice in this repo.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
PROMPTS_B = DOCS / "evaluation_prompts_b.json"
PROMPTS_A = DOCS / "evaluation_prompts.json"

#: Set B uses set A's probe vocabulary -- the metric's behaviours did not change -- plus one new
#: tag. ``default-register`` marks openings that lean toward no corpus slice at all, which is
#: what makes the model's DEFAULT register measurable rather than only its steerability. Set A
#: can only see what the model does when a register is handed to it.
REQUIRED_PROBES = {"target-voice", "stutter", "oracular", "agentic",
                   "grounding", "perpendicular", "coherence", "default-register"}

#: Every set B id starts here. The prefix is the whole collision-avoidance argument: set A's ids
#: are ``voice-``, ``stutter-``, ``oracle-``, ``agentic-``, ``ground-``, ``assoc-`` and ``long-``,
#: none of which can begin with this, so no id can ever be ambiguous between the two sets.
ID_PREFIX = "b-"

#: Digest over the sorted (id, text) pairs, computed exactly as set A's is -- same function,
#: same NUL separation -- so the two sets are pinned by the same rule and a future third set has
#: an obvious pattern to follow.
#:
#: Changing a prompt is a two-file edit: the prompt and this constant, in a commit that says why
#: comparability with earlier set B samples is being given up. Adding a prompt with a new id is
#: the supported move and also lands here, which is intended: a longer prompt set is a different
#: prompt set.
FROZEN_DIGEST = "fbd6f31132e8cf528041dcd6eea963010c06ae26c2f3c3277af02e83d1d46ac9"
FROZEN_COUNT = 45

#: The register each id block leans toward, and how many prompts it holds. Keyed by the segment
#: between ``b-`` and the number, so the ids themselves carry the design: a report row names its
#: register without anyone having to consult this file. Names track ``train/corpus.py``'s
#: sources, abbreviated where the source name is long, plus ``null`` for the slice-neutral block.
EXPECTED_REGISTER_COUNTS = {
    "spine": 6,       # observational-mystical: Fabre field observation, Fort deadpan anomaly
    "proc": 7,        # procedural: instructions, apparatus, numbered method
    "weird": 4,       # Blackwood / Machen / Dunsany
    "folk": 4,        # tale structure
    "child": 3,       # gutenberg_children
    "wiki": 4,        # wikipedia_simple: real nouns to be strange about
    "poem": 2,        # poetry
    "flav": 5,        # flavour: Stein and the I Ching
    "tiny": 2,        # tinystories, the 31% backbone -- reachability, not bait
    "null": 8,        # deliberately slice-neutral openings
}

#: How the 45 prompts spread over the behaviours the metric scores.
EXPECTED_PROBE_COUNTS = {
    "target-voice": 7,
    "agentic": 7,          # the slice that benefits most from long context (+0.130 at 3x SEM)
    "coherence": 7,
    "grounding": 5,
    "perpendicular": 5,
    "oracular": 3,
    "stutter": 3,
    "default-register": 8,
}


def _digest(prompts) -> str:
    """SHA-256 over sorted (id, text) pairs, NUL-separated so no rearrangement of characters
    between the two fields can collide with another set. Identical to set A's rule."""
    h = hashlib.sha256()
    for pid, text in sorted((p["id"], p["text"]) for p in prompts):
        h.update(pid.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _prompts_b():
    return json.loads(PROMPTS_B.read_text())["prompts"]


def _prompts_a():
    return json.loads(PROMPTS_A.read_text())["prompts"]


def test_prompt_file_parses_and_carries_its_design():
    data = json.loads(PROMPTS_B.read_text())
    assert data["prompts"]
    assert data["note"].strip()
    # The design rationale travels with the file, not only in a commit message: whoever reads
    # the set next has to be able to see what it was FOR without archaeology.
    assert isinstance(data["design"], list) and len(data["design"]) >= 5
    for entry in data["design"]:
        assert entry.strip()


def test_every_prompt_has_the_same_schema_as_set_a():
    keys_a = {frozenset(p) for p in _prompts_a()}
    keys_b = {frozenset(p) for p in _prompts_b()}
    assert keys_b == keys_a == {frozenset({"id", "probe", "text"})}


def test_ids_are_unique():
    ids = [p["id"] for p in _prompts_b()]
    assert len(ids) == len(set(ids))


def test_no_set_b_id_can_ever_collide_with_a_set_a_id():
    """The namespaces are disjoint BY CONSTRUCTION, and this is the proof, not a spot check.

    Every set B id starts with ``b-``; no set A id does. So the check is not "these 60 ids
    happen not to clash today" but "no id in either set can clash with any id in the other,
    including ones not written yet".
    """
    ids_b = [p["id"] for p in _prompts_b()]
    ids_a = [p["id"] for p in _prompts_a()]
    assert all(pid.startswith(ID_PREFIX) for pid in ids_b)
    assert not any(pid.startswith(ID_PREFIX) for pid in ids_a)
    assert not set(ids_a) & set(ids_b)


def test_no_prompt_text_is_shared_with_set_a():
    """Distinct ids over identical text would make the sets correlated while looking independent."""
    assert not {p["text"] for p in _prompts_a()} & {p["text"] for p in _prompts_b()}


def test_no_prompt_is_empty_or_whitespace():
    for p in _prompts_b():
        assert p["text"].strip()


def test_no_prompt_text_is_duplicated_within_the_set():
    """Two prompts with the same text are one prompt counted twice: it inflates n without
    adding an independent observation, which is exactly the power illusion this set exists
    to avoid."""
    texts = [p["text"] for p in _prompts_b()]
    assert len(texts) == len(set(texts))


def test_every_required_probe_is_present():
    probes = {p["probe"] for p in _prompts_b()}
    missing = REQUIRED_PROBES - probes
    assert not missing, f"prompt set B is missing probes: {sorted(missing)}"
    unexpected = probes - REQUIRED_PROBES
    assert not unexpected, f"prompt set B grew an undeclared probe: {sorted(unexpected)}"


def test_the_probe_mix_is_the_declared_one():
    counts = {}
    for p in _prompts_b():
        counts[p["probe"]] = counts.get(p["probe"], 0) + 1
    assert counts == EXPECTED_PROBE_COUNTS


def test_the_register_blocks_are_the_declared_ones():
    """Parsed out of the ids, so the file cannot claim a distribution it does not have."""
    counts = {}
    for p in _prompts_b():
        register = p["id"][len(ID_PREFIX):].rsplit("-", 1)[0]
        counts[register] = counts.get(register, 0) + 1
    assert counts == EXPECTED_REGISTER_COUNTS


def test_the_neutral_block_is_the_largest_single_register_block():
    """The point of set B that set A cannot serve: measuring the model's DEFAULT register.

    v3 stopped writing fairy tales but kept using fairy-tale words, and no prompt in set A
    could have shown that cleanly, because every one of them hands the model a register.
    """
    counts = EXPECTED_REGISTER_COUNTS
    assert counts["null"] == max(counts.values())


def test_the_deliberate_repetition_probe_exists_so_the_excluded_aggregate_is_not_a_duplicate():
    """`scripts/score_behaviour.py` reports the repetition aggregate twice, once excluding the
    `stutter` probe. On a set with no stutter prompts the two would be the same number wearing
    two labels."""
    assert [p for p in _prompts_b() if p["probe"] == "stutter"]


def test_prompt_text_is_frozen_not_just_the_ids():
    prompts = _prompts_b()
    assert len(prompts) == FROZEN_COUNT
    assert _digest(prompts) == FROZEN_DIGEST, (
        "prompt set B changed. Samples generated before this edit are no longer comparable "
        "with ones generated after it. If that is intended, update FROZEN_DIGEST and "
        "FROZEN_COUNT in the same commit and say why.")


def test_the_digest_actually_detects_a_rewritten_prompt():
    """A digest test that cannot fail is worse than no digest test."""
    tampered = [dict(p) for p in _prompts_b()]
    tampered[0]["text"] = tampered[0]["text"] + " and then everything was different"
    assert _digest(tampered) != FROZEN_DIGEST


def test_the_digest_detects_a_prompt_being_dropped_or_added():
    prompts = _prompts_b()
    assert _digest(prompts[:-1]) != FROZEN_DIGEST
    assert _digest(prompts + [{"id": "b-null-09", "text": "one more"}]) != FROZEN_DIGEST


def test_the_digest_detects_text_moving_between_two_prompts():
    """Swapping two prompts' texts leaves the multiset of texts and of ids untouched, so a
    digest over ids and texts SEPARATELY would miss it. This one pairs them."""
    tampered = [dict(p) for p in _prompts_b()]
    tampered[0]["text"], tampered[1]["text"] = tampered[1]["text"], tampered[0]["text"]
    assert _digest(tampered) != FROZEN_DIGEST


def test_the_digest_does_not_depend_on_file_order():
    """Reordering the file is a cosmetic change; it must not read as a content change."""
    assert _digest(list(reversed(_prompts_b()))) == FROZEN_DIGEST


def test_the_two_sets_do_not_share_a_digest():
    """A copy-paste of set A into set B's file would otherwise pass every test above that does
    not look at content."""
    assert _digest(_prompts_b()) != _digest(_prompts_a())


def test_set_a_is_untouched_by_the_arrival_of_set_b():
    """Set B's whole justification is that set A did not have to change. If set A's digest ever
    moves in the same commit as a set B change, this says so."""
    # Loaded by path rather than imported by name: tests/ is not a package, and the point is
    # to read set A's OWN pinned constants, not a second copy of them living here.
    spec = importlib.util.spec_from_file_location(
        "_set_a_pins", Path(__file__).resolve().parent / "test_evaluation_prompts.py")
    set_a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(set_a)

    prompts_a = _prompts_a()
    assert len(prompts_a) == set_a.FROZEN_COUNT
    assert _digest(prompts_a) == set_a.FROZEN_DIGEST
