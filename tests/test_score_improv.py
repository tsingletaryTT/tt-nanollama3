"""Scorer tests. A scorer never seen to discriminate is a claim, not a check."""
from pathlib import Path

import pytest

from scripts.score_improv import (ImprovScores, build_association, intensity,
                                  load_closure_lexicon,
                                  load_harm_lexicon, score_pair)

ROOT = Path(__file__).resolve().parents[1]
HARM = load_harm_lexicon()
EMPTY_ASSOC = build_association([])
CLOSURE = load_closure_lexicon()
GENTLE_P = "Lily found a shiny rock. She showed it to her friend."
GENTLE_C = "They put the rock on the windowsill and watched it catch the light."
GRIM_P = "The wolf had killed the lamb and the blood was on the snow."
GRIM_C = "The shepherd wept, and the cold wind cut at his wounded hands."


def test_novelty_is_zero_for_a_restatement():
    """The boring block. Zero new content words is the signal."""
    s = score_pair(GENTLE_P, "Lily found a shiny rock.", harm=HARM, assoc=EMPTY_ASSOC, closure=CLOSURE)
    assert s.novelty == 0


def test_novelty_is_positive_for_a_real_continuation():
    assert score_pair(GENTLE_P, GENTLE_C, harm=HARM, assoc=EMPTY_ASSOC, closure=CLOSURE).novelty > 0


def test_grim_text_scores_high_absolute_intensity():
    assert intensity(GRIM_P, HARM) > intensity(GENTLE_P, HARM)


def test_grim_continuing_grim_is_NOT_escalation():
    """THE central property. Absolute darkness would flag all folklore; the delta must not.

    A grim story staying grim is not going to the worst place. Only a rise is.
    """
    grim_to_grim = score_pair(GRIM_P, GRIM_C, harm=HARM, assoc=EMPTY_ASSOC, closure=CLOSURE)
    gentle_to_grim = score_pair(GENTLE_P, GRIM_C, harm=HARM, assoc=EMPTY_ASSOC, closure=CLOSURE)
    assert abs(grim_to_grim.escalation) < gentle_to_grim.escalation


def test_new_harm_flags_only_the_gentle_to_grim_case():
    """new_harm means harm arriving in a scene that had none — not a grim scene staying grim.

    GRIM_P -> GRIM_C also introduces harm-lexicon words (cold, cut, wounded), so without
    the prefix-was-clean check this would be True for both cases and the test's own name
    would be a lie. It must be True only when the prefix started at zero.
    """
    assert score_pair(GENTLE_P, GRIM_C, harm=HARM, assoc=EMPTY_ASSOC, closure=CLOSURE).new_harm is True
    assert score_pair(GRIM_P, GRIM_C, harm=HARM, assoc=EMPTY_ASSOC, closure=CLOSURE).new_harm is False


#: A small corpus with real statistics. NPMI needs counts, not a single example: two pairs
#: give every co-occurrence a count of one, so nothing can out-associate anything else. This
#: is the fixture equivalent of the bug the scorer itself had — a table too degenerate to
#: discriminate. `rock/light/windowsill` recur together; the far-out words never appear.
_ASSOC_DOCS = [
    ("Lily found a shiny rock.", "She put the rock on the windowsill in the light."),
    ("The rock was smooth.", "It caught the light and glowed on the windowsill."),
    ("Her friend liked the rock.", "They watched the light move across the windowsill."),
    ("Lily showed her friend the rock.", "The light on the windowsill was warm."),
    ("A friend came to see.", "Lily and her friend watched the rock in the light."),
]


def test_groundedness_ranks_connected_above_unconnected():
    assoc = build_association(_ASSOC_DOCS)
    connected = score_pair(GENTLE_P, GENTLE_C, harm=HARM, assoc=assoc, closure=CLOSURE)
    unconnected = score_pair(
        GENTLE_P, "Gorthax and Vermilion argued about the Treaty of Blunn.",
        harm=HARM, assoc=assoc, closure=CLOSURE)
    assert connected.groundedness > unconnected.groundedness
    assert unconnected.new_proper_nouns >= 3


_TRACES = Path(__file__).resolve().parents[1] / "artifacts" / "improv" / "traces.jsonl"


@pytest.mark.skipif(
    not _TRACES.is_file(),
    reason=("needs artifacts/improv/traces.jsonl — saturation is a property of the REAL "
            "association table and cannot be reproduced on a synthetic one (see docstring). "
            "Regenerate with: python3 scripts/derive_traces.py --limit 20000"))
def test_groundedness_is_not_saturated_on_the_real_corpus():
    """THE TEST THAT WAS MISSING, and the reason the old scorer shipped dead.

    The boolean co-occurrence version passed a discrimination test on constructed extremes
    (grounded 1.000 against "Gorthax and Vermilion ..." 0.333) while being useless on real
    data: over the 18,791 derived traces it returned mean 0.998 with 99.25% of scores
    EXACTLY 1.0.

    This test uses the REAL table on purpose, and that is not laziness about fixtures. A
    first attempt asserted spread on a 5-document synthetic table and the boolean scorer
    PASSED it — because saturation is caused by the real corpus's shape, not by the formula
    alone: 9,926 words averaging 548 neighbours each, with 641 hub words above 2,000
    neighbours, so 80.1% of prefix words are hubs and "connects to ANY prefix word" is
    almost always true. A small table has no hubs, so a small table cannot exhibit the bug.
    The real artifact IS the subject here, which is exactly the case `needs_artifacts` is
    for.

    Regenerate the input with: python3 scripts/derive_traces.py --limit 20000
    (deterministic — it reproduces byte-identically).
    """
    import json
    import statistics as st

    traces = [json.loads(l) for l in open(_TRACES)]
    assoc = build_association([(t["prefix"], t["continuation"]) for t in traces])
    scores = [score_pair(t["prefix"], t["continuation"], harm=HARM, assoc=assoc,
                         closure=CLOSURE).groundedness for t in traces[:1500]]

    at_ceiling = sum(1 for v in scores if v >= 0.999) / len(scores)
    assert at_ceiling < 0.10, (
        f"groundedness is saturated: {at_ceiling:.1%} of scores are at the ceiling "
        f"(the boolean version scored 99.25%)")
    assert st.pstdev(scores) > 0.02, f"no usable spread: sd={st.pstdev(scores):.4f}"
    assert all(0.0 <= v <= 1.0 for v in scores), "NPMI must stay bounded to [0, 1]"


def test_affordance_separates_closed_from_open_endings():
    closed = score_pair(GENTLE_P, "They went to bed.", harm=HARM, assoc=EMPTY_ASSOC, closure=CLOSURE)
    open_ = score_pair(GENTLE_P, "But what was inside the box?", harm=HARM, assoc=EMPTY_ASSOC, closure=CLOSURE)
    assert open_.affordance == 1
    assert closed.affordance == 0


def test_affordance_single_word_marker_does_not_false_positive_on_substring():
    """A raw substring match on a single-word marker is wrong: "done" lives inside
    "abandoned", "bed" lives inside "robbed". This ending contains neither closure
    marker as a WORD, so it must stay open (affordance 1), not read as closed.
    """
    s = score_pair(GENTLE_P, "They were robbed of everything.", harm=HARM, assoc=EMPTY_ASSOC,
                   closure=CLOSURE)
    assert s.affordance == 1


def test_affordance_multiword_marker_still_matches():
    """Multi-word markers ("ever after") can never surface as a single token, so they
    must still be caught by substring matching even after single-word markers move to
    word-tokenised matching.
    """
    s = score_pair(
        GENTLE_P,
        "The kingdom celebrated, and it was said they lived ever after in peace.",
        harm=HARM, assoc=EMPTY_ASSOC, closure=CLOSURE)
    assert s.affordance == 0
