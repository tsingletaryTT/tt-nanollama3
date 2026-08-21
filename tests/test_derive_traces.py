import json

from scripts.derive_traces import (_sft_example_unaligned, build_sft_examples,
                                    derive_from_story)
from train.improv import Slots, render_think


STORY = ("Lily found a needle in her room. She knew it was sharp. "
         "She showed the needle to her mother. Her mother sewed the button. "
         "They were both happy with the shirt.")

PAD_TOKEN_ID = 999  # arbitrary; distinct from any id _Tok's checksum could plausibly emit


class _Tok:
    """A faithful (non-lossy) stand-in tokenizer: one id per whitespace-separated word.

    Deliberately NOT `ord(c) % 97 for c in s[:20]` (the original brief's mock) — that
    truncates to 20 chars, so any two completions of >=20 chars collapse to the same
    token count regardless of content, making length-based assertions unable to fail.

    Deliberately NOT `hash(w)` either — Python's string hash is randomized per process
    (PYTHONHASHSEED), and this project's determinism guarantee (Task 1) depends on
    nothing in the pipeline depending on hash randomization. `sum(ord(c) for c in w)` is
    a stable, process-independent function of the word.
    """

    def encode(self, s, add_special_tokens=True):
        return [sum(ord(c) for c in w) % 31_000 for w in s.split()]


def test_derive_returns_a_trace_with_all_slots():
    rec = derive_from_story(STORY, story_id=0, rng_seed=5489)
    assert rec is not None
    assert set(rec["slots"]) == {"offer", "accept", "add", "stakes", "handback"}
    assert rec["prefix"] and rec["continuation"]
    assert "<think>" in rec["think"]


def test_sft_example_masks_only_the_prompt():
    """The mask is the whole point: prompt positions carry -100, completion carries ids.

    Pinned against ttml's PRE-SHIFTED label convention (labels[t] == input_ids[t+1] --
    see `_sft_example_unaligned`'s docstring), not the same-position HF convention this
    test used to assert. Under shifting the supervised region starts one position EARLIER
    than the completion itself: the LAST prompt position's label is the FIRST completion
    token (that transition -- predict the start of the completion from the prompt -- is
    exactly the behaviour being trained, so it must be supervised, not masked), and the
    supervised region ends one position EARLIER than the completion's own last token,
    because that last token has no legitimate next token to predict (only padding follows
    it) and must be masked instead.

    A hollow version of this test (asserting only "something is masked, something is
    supervised") would still pass if the think-block were concatenated onto the PROMPT
    side instead of the completion side -- exactly the worst-case bug for this project.
    So this pins exact facts: (1) everything before the last prompt position is masked,
    no more, no less; (2) the boundary position (last prompt token) is supervised and its
    label is exactly the first completion token; (3) the think-block's own token sequence
    appears verbatim inside the supervised region, starting at that boundary; (4) the last
    completion position is masked (no next token to predict); and (5) the tile-alignment
    pad tail is masked too.
    """
    tok = _Tok()
    rec = derive_from_story(STORY, story_id=0, rng_seed=5489)
    ex = build_sft_examples([rec], tok, with_think=True, pad_token_id=PAD_TOKEN_ID)[0]
    input_ids, labels = ex["input_ids"], ex["labels"]
    assert len(input_ids) == len(labels)
    assert len(input_ids) % 32 == 0, "build_sft_examples must tile-align its output"

    prompt_ids = tok.encode(rec["prefix"])
    completion_ids = tok.encode(rec["think"] + rec["continuation"], add_special_tokens=False)
    boundary = len(prompt_ids) - 1  # last prompt position: first-supervised, per the shift

    assert labels[:boundary] == [-100] * boundary, (
        "everything strictly before the last prompt position must be masked")
    assert labels[boundary] == input_ids[boundary + 1] == completion_ids[0], (
        "the last prompt position's label must be the FIRST completion token -- this is "
        "the prompt-to-completion transition, and it must be supervised")

    # Supervised region: boundary .. boundary + len(completion_ids) - 2 (inclusive) --
    # one shorter than completion_ids because the completion's own last token has no
    # legitimate next token and is masked, not supervised (checked separately below).
    supervised_region = labels[boundary:boundary + len(completion_ids) - 1]
    assert all(v != -100 for v in supervised_region), (
        "nothing in the shifted supervised region should be masked")
    assert supervised_region == completion_ids[:-1], (
        "supervised label VALUES must be the completion tokens themselves, pre-shifted "
        "into position -- labels[boundary + j] == completion_ids[j] for the transition + "
        "all but the last completion token")

    last_completion_pos = boundary + len(completion_ids)
    assert labels[last_completion_pos] == -100, (
        "the completion's own last token has no next token to predict (only padding "
        "follows) and must be masked, not scored against a pad token")

    pad_region = labels[last_completion_pos + 1:]
    assert pad_region == [-100] * len(pad_region), (
        "the tile-alignment pad tail must be masked too, or it would pull the loss "
        "toward predicting pad_token_id")
    assert input_ids[last_completion_pos + 1:] == [PAD_TOKEN_ID] * len(pad_region), (
        "pad tail on input_ids must use pad_token_id")

    think_ids = tok.encode(rec["think"], add_special_tokens=False)
    assert supervised_region[:len(think_ids)] == think_ids, (
        "the think-block's tokens must land in the supervised region, not the prompt")


def test_no_think_arm_omits_the_block_but_keeps_the_continuation():
    """Assert the EXACT length delta, not just an inequality.

    `len(without) <= len(with_t)` is satisfied by equality, so it can't fail even if
    `with_think` is ignored entirely. The delta must equal precisely the think-block's
    own token count — that fails immediately if the flag is ignored, or if the block is
    concatenated in the wrong place, or duplicated.

    Asserted on `_sft_example_unaligned` (pre tile-alignment), not `build_sft_examples`'s
    public output: `build_sft_examples` pads each arm to the next multiple of 32
    independently, and that padding depends on each arm's own raw length mod 32 — so the
    *aligned* delta can differ from think_len by anywhere in [-31, +31] depending on the
    input, which would make this assertion either flaky or (worse) satisfiable by luck
    even with `with_think` ignored. The guarantee this test exists to pin — with_think
    adds exactly the think-block's own tokens, no more, no less — is a property of the
    unaligned construction; alignment is a separate, orthogonal concern covered by
    test_build_sft_examples_is_tile_aligned below.
    """
    tok = _Tok()
    rec = derive_from_story(STORY, story_id=0, rng_seed=5489)
    with_t = _sft_example_unaligned(rec, tok, with_think=True)
    without = _sft_example_unaligned(rec, tok, with_think=False)

    think_len = len(tok.encode(rec["think"], add_special_tokens=False))
    assert len(with_t["input_ids"]) - len(without["input_ids"]) == think_len


def _contains_subsequence(haystack: list, needle: list) -> bool:
    """True if `needle` appears as a contiguous run inside `haystack`."""
    n = len(needle)
    if n == 0:
        return True
    return any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


def test_build_sft_examples_with_think_flag_controls_think_tokens():
    """Content-based guard on `build_sft_examples`'s OWN `with_think` forwarding.

    `test_no_think_arm_omits_the_block_but_keeps_the_continuation` asserts the exact
    token-count delta, but it necessarily does so on `_sft_example_unaligned` (pre tile
    alignment) -- the aligned delta is not think_len-stable (see that test's docstring).
    That relocation left `build_sft_examples`'s own `with_think` forwarding completely
    unguarded: a `build_sft_examples` that hardcoded `with_think=True` internally,
    silently ignoring its caller's flag, would pass every other test in this file --
    `test_sft_example_masks_only_the_prompt` only ever calls with `with_think=True`, the
    delta test bypasses `build_sft_examples` entirely by calling
    `_sft_example_unaligned` directly, and `test_build_sft_examples_is_tile_aligned`
    checks length/shape only and never content, so it cannot tell the two arms apart.
    Task 5 calls the aligned `build_sft_examples`, so that regression could ship with
    every test in this file green.

    This test closes the gap directly against `build_sft_examples`'s own output, using
    containment rather than arithmetic so it is invariant to trailing tile-alignment pad:
    with `with_think=True`, the think-block's token subsequence must appear inside the
    SUPERVISED region of `labels` (positions where `labels != -100`); with
    `with_think=False`, that same subsequence must appear NOWHERE in `input_ids` or
    `labels`.

    LABEL-SHIFT NOTE: under the pre-shifted convention (see `_sft_example_unaligned`'s
    docstring) the supervised region's absolute positions move one earlier than they used
    to (the boundary token -- the last prompt position -- is now the first supervised
    position, not the first completion position). The ORDERED SEQUENCE OF SUPERVISED
    VALUES is unchanged by that shift -- `labels[boundary + j] == completion_ids[j]` for
    every valid `j`, exactly the same values in the same relative order as the old
    same-position convention produced, just written one position earlier -- so the
    position-agnostic containment check below remains valid without modification.
    Re-verified directly against the same mutation this test was built to catch
    (hardcoding `with_think=True` inside `build_sft_examples`, ignoring the caller's
    flag): the no-think arm still goes RED under the new labels, because a leaked
    think-block still shows up as a value-level subsequence match regardless of which
    position it starts at. The added position-anchored assertion below is strictly
    stronger than the containment check alone: it additionally pins that the supervised
    region *starts exactly at* the last prompt position, which a same-position-vs-shifted
    off-by-one regression would violate even if the value-level containment happened to
    still pass.
    """
    tok = _Tok()
    rec = derive_from_story(STORY, story_id=0, rng_seed=5489)
    think_ids = tok.encode(rec["think"], add_special_tokens=False)
    assert think_ids, "fixture think-block must tokenize to at least one id"

    with_t = build_sft_examples([rec], tok, with_think=True, pad_token_id=PAD_TOKEN_ID)[0]
    supervised = [v for v in with_t["labels"] if v != -100]
    assert _contains_subsequence(supervised, think_ids), (
        "with_think=True must place the think-block's tokens in the supervised region")

    # Position-anchored: the supervised region must begin exactly at the last prompt
    # position (boundary), and the think-block's tokens must be the first ones there.
    prompt_ids = tok.encode(rec["prefix"])
    boundary = len(prompt_ids) - 1
    assert with_t["labels"][boundary] != -100, (
        "the last prompt position must be the first supervised position")
    assert with_t["labels"][:boundary] == [-100] * boundary, (
        "nothing before the last prompt position should be supervised")
    assert with_t["labels"][boundary:boundary + len(think_ids)] == think_ids, (
        "the think-block's tokens must start exactly at the last prompt position")

    without = build_sft_examples([rec], tok, with_think=False, pad_token_id=PAD_TOKEN_ID)[0]
    assert not _contains_subsequence(without["input_ids"], think_ids), (
        "with_think=False must not leak the think-block into input_ids")
    assert not _contains_subsequence(without["labels"], think_ids), (
        "with_think=False must not leak the think-block into labels")


def _fake_trace(*, prefix: str, continuation: str) -> dict:
    """A trace dict with the same shape `derive_from_story` produces (prefix/think/
    continuation), built directly rather than via `derive_from_story` + `extract_slots`.

    Used only to get several DIFFERENT raw (pre-alignment) token lengths deterministically
    — `extract_slots` requires shared content words between prefix and continuation to
    return anything at all (see test_improv.py), which arbitrary short fixture text is not
    guaranteed to satisfy, and this test cares about exercising several remainders mod 32,
    not about `extract_slots`'s own selection logic (that is test_improv.py's job).
    """
    slots = Slots(offer="offer", accept="accept", add="add", stakes="level",
                  handback="handback")
    return {"prefix": prefix, "think": render_think(slots), "continuation": continuation}


def test_build_sft_examples_is_tile_aligned():
    """Every example build_sft_examples returns is a multiple of 32 tokens, both arms.

    This is the fix for the Task 2 SFTTrainer finding (task-2-report.md): ttml's SDPA
    backward kernel mismatches when a collated batch's sequence length is not a multiple
    of 32. sft_collate_fn pads a batch to its longest example, so tile-aligning every
    example here guarantees the batch max is aligned too. Uses several traces of
    DIFFERENT raw lengths (rather than one, repeated) so the raw pre-padding lengths land
    on different remainders mod 32 — a version that only right-pads to SOME fixed length
    regardless of input, or that is off by one tile boundary, would not reliably be caught
    by a single input whose raw length happens to already land on a multiple of 32.
    """
    tok = _Tok()
    recs = [
        _fake_trace(prefix="one two three", continuation="four five six seven"),
        _fake_trace(prefix=" ".join(f"w{i}" for i in range(20)),
                    continuation=" ".join(f"c{i}" for i in range(15))),
        _fake_trace(prefix="a", continuation="b"),
        derive_from_story(STORY, story_id=0, rng_seed=5489),
    ]
    raw_lengths = {len(_sft_example_unaligned(r, tok, with_think=True)["input_ids"]) % 32
                  for r in recs}
    assert len(raw_lengths) > 1, "fixture traces should span more than one remainder mod 32"

    for with_think in (True, False):
        examples = build_sft_examples(recs, tok, with_think=with_think,
                                      pad_token_id=PAD_TOKEN_ID)
        assert len(examples) == len(recs)
        for ex in examples:
            assert len(ex["input_ids"]) % 32 == 0
            assert len(ex["labels"]) == len(ex["input_ids"])


def test_labels_are_shifted_by_one_for_next_token_prediction():
    """The convention test whose absence let a whole plan ship with the wrong labels.

    ttml's `cross_entropy_loss` (reached via `SFTTrainer`/`sft_collate_fn`) never shifts
    internally -- it expects the CALLER to pre-shift, i.e. `labels[t]` must be the token
    the model should predict having seen `input_ids[0..t]`, which is `input_ids[t+1]`.
    This is the same convention `ttml.common.data.get_batch` uses for pretraining
    (`y = split_ids[i+1:i+seq_len+1]`) and the one `ttml.datasets.causal_lm_collate_fn`'s
    own docstring states outright ("already shifted by one for next-token prediction").

    Every other test in this file checks shape, containment, or length -- structural
    properties that are satisfied identically whether labels are shifted correctly or
    not. That is exactly how a same-position (HF-style) `_sft_example_unaligned` shipped
    and trained two SFT arms against an off-by-one target for a full task before being
    caught (see task-5-report.md's addendum: loss stuck near `ln(vocab_size)` (~11)
    instead of the warm-started checkpoint's own ~1.7-2.9). This test pins the actual
    arithmetic relationship directly, across both arms, so a regression back to
    same-position labels fails here immediately rather than needing a warm-started model
    and a full training run to notice.
    """
    tok = _Tok()
    rec = derive_from_story(STORY, story_id=0, rng_seed=5489)
    for with_think in (True, False):
        ex = build_sft_examples([rec], tok, with_think=with_think, pad_token_id=PAD_TOKEN_ID)[0]
        input_ids, labels = ex["input_ids"], ex["labels"]
        supervised_positions = [t for t, v in enumerate(labels) if v != -100]
        assert supervised_positions, (
            f"with_think={with_think}: fixture must have at least one supervised position")

        for t in supervised_positions:
            assert t + 1 < len(input_ids), (
                f"with_think={with_think}: position {t} is supervised but has no next "
                "token in input_ids to have predicted -- it should have been masked")
            assert labels[t] == input_ids[t + 1], (
                f"with_think={with_think}: labels[{t}]={labels[t]} != "
                f"input_ids[{t + 1}]={input_ids[t + 1]} -- labels must be pre-shifted by "
                "one for ttml's cross_entropy_loss convention (labels[t] == "
                "input_ids[t+1]), not same-position/HF-style (labels[t] == input_ids[t])")

        # The last supervised position must never be the sequence's final token -- if it
        # were, there would be no input_ids[t+1] to have been the (correct) target, which
        # would mean this position should have been masked instead of supervised.
        assert supervised_positions[-1] < len(input_ids) - 1, (
            f"with_think={with_think}: the last supervised position must leave at least "
            "one more token in input_ids to serve as its shifted target")
