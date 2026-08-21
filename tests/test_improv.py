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


def test_extract_drops_when_nothing_is_added():
    """A continuation that only repeats the prefix carries forward but adds nothing.

    That's the *other* way to be a boring block: `carried` must succeed here (every
    continuation word already appears in the prefix's last sentence, so the function
    gets past the carry-forward check) and only then does `fresh` come up empty. This
    exercises the `if not fresh: return None` branch specifically, distinct from the
    carry-forward drop covered above.
    """
    prefix = "Lily found a needle. She showed the needle to her mother."
    continuation = "Mother showed the needle."
    got = extract_slots(prefix, continuation, idf=_idf([]),
                        intensity=lambda t: 0.0)
    assert got is None


def test_extract_breaks_add_ties_alphabetically():
    """Slot extraction feeds training data, so identical input must always yield the
    same slots. `fresh_ranked` sorts a `set()` of strings, and Python randomizes
    string-hash order per process — with a plain `-idf` key, two words tied on IDF
    could come out in either order depending on process, and the corpus would be
    non-reproducible. `zebra` and `apple` are tied at the top IDF here; `near` is
    strictly lower so it can't be confused with the tie. The fixed key breaks the tie
    alphabetically, so `apple` (not `zebra`) must always be chosen — an outcome that's
    only guaranteed under the fixed sort key, not the old `-idf.get(w, 0.0)` one.
    """
    prefix = "Lily found a needle. She showed the needle to her mother."
    continuation = "Mother showed the needle near a zebra and an apple."
    got = extract_slots(prefix, continuation,
                        idf=_idf(["near"]) | {"zebra": 5.0, "apple": 5.0},
                        intensity=lambda t: 0.0)
    assert got is not None
    assert got.add == "apple"
