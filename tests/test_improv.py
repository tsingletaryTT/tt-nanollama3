from train.improv import Slots, parse_think, render_think, extract_slots


def _idf(words):
    """Uniform IDF: makes `add` selection depend only on absence from the prefix."""
    return {w: 1.0 for w in words}


def test_render_then_parse_round_trips():
    s = Slots(offer="the lantern went out", accept="the lantern is dark",
              add="moths", stakes="up", handback="her hands")
    assert parse_think(render_think(s)) == s


def test_parse_rejects_a_missing_slot():
    """Adherence scoring depends on this returning None, not a partial object."""
    broken = "<think>\noffer: a\naccept: b\nadd: c\nstakes: up\n</think>\n"
    assert parse_think(broken) is None


def test_parse_rejects_an_unknown_stakes_value():
    bad = ("<think>\noffer: a\naccept: b\nadd: c\nstakes: sideways\n"
           "handback: d\n</think>\n")
    assert parse_think(bad) is None


def test_extract_drops_when_nothing_carries_forward():
    """A continuation that carries nothing forward IS the boring block.

    Blocks must never become training exemplars, so extraction returns None.
    """
    prefix = "Lily found a needle. She showed it to her mother."
    continuation = "Elsewhere, a distant volcano erupted quietly."
    got = extract_slots(prefix, continuation, idf=_idf(["volcano"]),
                        intensity=lambda t: 0.0)
    assert got is None


def test_extract_fills_accept_from_the_carried_entity():
    prefix = "Lily found a needle. She showed the needle to her mother."
    continuation = "Her mother took the needle and sewed the button."
    got = extract_slots(prefix, continuation, idf=_idf(["button", "sewed"]),
                        intensity=lambda t: 0.0)
    assert got is not None
    assert "needle" in got.accept
    assert got.add != ""
    assert got.stakes == "level"


def test_stakes_reads_up_when_intensity_rises():
    prefix = "Lily found a needle."
    continuation = "The needle cut her hand and she cried."
    got = extract_slots(prefix, continuation, idf=_idf(["cut", "cried"]),
                        intensity=lambda t: 5.0 if "cut" in t else 0.0)
    assert got is not None and got.stakes == "up"
