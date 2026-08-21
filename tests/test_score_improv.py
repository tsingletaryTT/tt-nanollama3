"""Scorer tests. A scorer never seen to discriminate is a claim, not a check."""
from pathlib import Path

from scripts.score_improv import (ImprovScores, intensity, load_closure_lexicon,
                                  load_harm_lexicon, score_pair)

HARM = load_harm_lexicon()
CLOSURE = load_closure_lexicon()
GENTLE_P = "Lily found a shiny rock. She showed it to her friend."
GENTLE_C = "They put the rock on the windowsill and watched it catch the light."
GRIM_P = "The wolf had killed the lamb and the blood was on the snow."
GRIM_C = "The shepherd wept, and the cold wind cut at his wounded hands."


def test_novelty_is_zero_for_a_restatement():
    """The boring block. Zero new content words is the signal."""
    s = score_pair(GENTLE_P, "Lily found a shiny rock.", harm=HARM, cooc={}, closure=CLOSURE)
    assert s.novelty == 0


def test_novelty_is_positive_for_a_real_continuation():
    assert score_pair(GENTLE_P, GENTLE_C, harm=HARM, cooc={}, closure=CLOSURE).novelty > 0


def test_grim_text_scores_high_absolute_intensity():
    assert intensity(GRIM_P, HARM) > intensity(GENTLE_P, HARM)


def test_grim_continuing_grim_is_NOT_escalation():
    """THE central property. Absolute darkness would flag all folklore; the delta must not.

    A grim story staying grim is not going to the worst place. Only a rise is.
    """
    grim_to_grim = score_pair(GRIM_P, GRIM_C, harm=HARM, cooc={}, closure=CLOSURE)
    gentle_to_grim = score_pair(GENTLE_P, GRIM_C, harm=HARM, cooc={}, closure=CLOSURE)
    assert abs(grim_to_grim.escalation) < gentle_to_grim.escalation


def test_new_harm_flags_only_the_gentle_to_grim_case():
    """new_harm means harm arriving in a scene that had none — not a grim scene staying grim.

    GRIM_P -> GRIM_C also introduces harm-lexicon words (cold, cut, wounded), so without
    the prefix-was-clean check this would be True for both cases and the test's own name
    would be a lie. It must be True only when the prefix started at zero.
    """
    assert score_pair(GENTLE_P, GRIM_C, harm=HARM, cooc={}, closure=CLOSURE).new_harm is True
    assert score_pair(GRIM_P, GRIM_C, harm=HARM, cooc={}, closure=CLOSURE).new_harm is False


def test_groundedness_ranks_connected_above_unconnected():
    cooc = {"rock": {"windowsill", "light"}, "lily": {"friend"}}
    connected = score_pair(GENTLE_P, GENTLE_C, harm=HARM, cooc=cooc, closure=CLOSURE)
    unconnected = score_pair(
        GENTLE_P, "Gorthax and Vermilion argued about the Treaty of Blunn.",
        harm=HARM, cooc=cooc, closure=CLOSURE)
    assert connected.groundedness > unconnected.groundedness
    assert unconnected.new_proper_nouns >= 3


def test_affordance_separates_closed_from_open_endings():
    closed = score_pair(GENTLE_P, "They went to bed.", harm=HARM, cooc={}, closure=CLOSURE)
    open_ = score_pair(GENTLE_P, "But what was inside the box?", harm=HARM, cooc={}, closure=CLOSURE)
    assert open_.affordance == 1
    assert closed.affordance == 0


def test_affordance_single_word_marker_does_not_false_positive_on_substring():
    """A raw substring match on a single-word marker is wrong: "done" lives inside
    "abandoned", "bed" lives inside "robbed". This ending contains neither closure
    marker as a WORD, so it must stay open (affordance 1), not read as closed.
    """
    s = score_pair(GENTLE_P, "They were robbed of everything.", harm=HARM, cooc={},
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
        harm=HARM, cooc={}, closure=CLOSURE)
    assert s.affordance == 0
