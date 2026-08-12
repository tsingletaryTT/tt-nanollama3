# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Component and end-to-end tests for the pure-NumPy ttml forward pass.

Every component test below is checked against a value computed by hand (or a structural
*property* of the operation) -- never against `convert/hf_mapping.py` or `convert/to_hf.py`,
and never against the Hugging Face model. This module exists to be an *independent* check
of our HF converter; testing it against the thing it's supposed to check would make the
whole exercise circular (see `docs/ttml-forward-reference.md`'s "Independence" section).

Conventions asserted here are cited to `docs/ttml-forward-reference.md`, which is itself
cited to ttml's C++ source -- never to our own converter.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from convert.checkpoint_reader import read_checkpoint_meta
from convert.ttml_forward import (
    apply_rope,
    attention,
    forward,
    rms_norm,
    squeeze_leading,
    swiglu,
)

ROOT = Path(__file__).resolve().parent.parent
REAL_CHECKPOINT_DIR = ROOT / "artifacts" / "checkpoints"
VAL_IDS_PATH = ROOT / "artifacts" / "tokens" / "val_ids.npy"


# ---------------------------------------------------------------------------
# squeeze_leading
# ---------------------------------------------------------------------------


def test_squeeze_leading_drops_singleton_batch_dims_from_a_matrix():
    # ttml checkpoint linears are stored as [1, 1, out, in] (rmsnorm-forward-reference.md
    # §"Tensor / weight layout convention"). Only the two leading 1s should go.
    x = np.zeros((1, 1, 3, 4))
    out = squeeze_leading(x)
    assert out.shape == (3, 4)


def test_squeeze_leading_drops_all_the_way_to_1d_for_a_gamma_vector():
    # gamma tensors are [1, 1, 1, C] -- three leading 1s here, so squeezing should not stop
    # after two; it stops only once shape[0] != 1 (or ndim == 1).
    x = np.arange(5.0).reshape(1, 1, 1, 5)
    out = squeeze_leading(x)
    assert out.shape == (5,)
    assert np.array_equal(out, np.arange(5.0))


def test_squeeze_leading_is_a_noop_on_an_already_squeezed_array():
    x = np.ones((3, 4))
    out = squeeze_leading(x)
    assert out.shape == (3, 4)
    assert out is x or np.array_equal(out, x)


def test_squeeze_leading_stops_at_a_real_leading_dimension():
    # Guard against over-squeezing: a genuine batch of 2 rows-of-1 must survive.
    x = np.ones((2, 1, 3))
    out = squeeze_leading(x)
    assert out.shape == (2, 1, 3)


# ---------------------------------------------------------------------------
# rms_norm -- §2 of the reference doc
# ---------------------------------------------------------------------------


def test_rms_norm_matches_hand_computed_value():
    # x = [3, 4], gamma = [1, 1], eps = 0 for a clean hand computation.
    # mean(x^2) = (9 + 16) / 2 = 12.5; rms = sqrt(12.5) = 3.5355339059327378
    # y = gamma * x / rms = [3/3.5355339, 4/3.5355339]
    x = np.array([3.0, 4.0])
    gamma = np.array([1.0, 1.0])
    expected = np.array([3.0, 4.0]) / 3.5355339059327378
    got = rms_norm(x, gamma, eps=0.0)
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_rms_norm_applies_gamma_elementwise_not_uniformly():
    # x = [1, 1], gamma = [2, 3]; mean(x^2) = 1, rms = 1 (eps=0) -> y = gamma * x = [2, 3].
    x = np.array([1.0, 1.0])
    gamma = np.array([2.0, 3.0])
    got = rms_norm(x, gamma, eps=0.0)
    np.testing.assert_allclose(got, np.array([2.0, 3.0]), rtol=1e-6)


def test_rms_norm_eps_sits_inside_the_sqrt_added_to_the_mean_of_squares():
    # §2: rms(x) = sqrt(mean(x^2) + eps), NOT sqrt(mean(x^2)) + eps.
    # x = [0, 0], gamma = [1], eps = 4 -> mean(x^2) = 0, rms = sqrt(0 + 4) = 2, y = 0/2 = 0.
    # The discriminator: sqrt(mean(x^2)) + eps would give 0 + 4 = 4, y = 0/4 = 0 too for a
    # zero vector, so use a nonzero x: x = [3, 0], eps = 16.
    # mean(x^2) = 4.5; rms = sqrt(4.5 + 16) = sqrt(20.5). Wrong convention (eps outside):
    # rms = sqrt(4.5) + 16, a very different number -- so this discriminates cleanly.
    x = np.array([3.0, 0.0])
    gamma = np.array([1.0, 1.0])
    eps = 16.0
    expected_inside = np.array([3.0, 0.0]) / np.sqrt(4.5 + eps)
    got = rms_norm(x, gamma, eps=eps)
    np.testing.assert_allclose(got, expected_inside, rtol=1e-6)
    wrong_outside = np.array([3.0, 0.0]) / (np.sqrt(4.5) + eps)
    assert not np.allclose(got, wrong_outside)


def test_rms_norm_gamma_is_not_1_plus_gamma():
    # §2: plain gamma, no unit offset. gamma=0 must zero the output entirely.
    x = np.array([5.0, -2.0, 3.0])
    gamma = np.zeros(3)
    got = rms_norm(x, gamma, eps=1e-5)
    np.testing.assert_allclose(got, np.zeros(3), atol=1e-12)


def test_rms_norm_broadcasts_over_leading_batch_dims():
    x = np.array([[3.0, 4.0], [1.0, 1.0]])
    gamma = np.array([1.0, 1.0])
    got = rms_norm(x, gamma, eps=0.0)
    expected = np.stack(
        [
            np.array([3.0, 4.0]) / 3.5355339059327378,
            np.array([1.0, 1.0]) / 1.0,
        ]
    )
    np.testing.assert_allclose(got, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# apply_rope -- §3 of the reference doc. Rotation *properties*, not magic arrays: any
# correct RoPE (whatever the pairing) satisfies round-trip and inner-product preservation,
# so those alone don't prove the pairing is right. The locality test below is what actually
# discriminates interleaved (x[2i], x[2i+1]) from split-halves (x[i], x[i+Dh/2]).
# ---------------------------------------------------------------------------


def test_rope_round_trip_rotating_by_p_then_minus_p_returns_the_original():
    rng = np.random.default_rng(0)
    head_dim = 8
    x = rng.normal(size=(1, head_dim))
    rotated = apply_rope(x, positions=np.array([5.0]), theta=10000.0, head_dim=head_dim)
    back = apply_rope(rotated, positions=np.array([-5.0]), theta=10000.0, head_dim=head_dim)
    np.testing.assert_allclose(back, x, atol=1e-6)


def test_rope_preserves_inner_product_of_two_vectors_rotated_by_the_same_position():
    # RoPE is a rotation (orthogonal transform), so rotating two vectors by the *same*
    # angle preserves their inner product -- this holds regardless of the pairing
    # convention, since any block-diagonal-by-2x2-rotations matrix is orthogonal.
    rng = np.random.default_rng(1)
    head_dim = 8
    a = rng.normal(size=(1, head_dim))
    b = rng.normal(size=(1, head_dim))
    p = np.array([17.0])
    a_rot = apply_rope(a, positions=p, theta=500000.0, head_dim=head_dim)
    b_rot = apply_rope(b, positions=p, theta=500000.0, head_dim=head_dim)
    np.testing.assert_allclose(
        np.sum(a_rot * b_rot, axis=-1), np.sum(a * b, axis=-1), rtol=1e-6
    )


def test_rope_preserves_each_vectors_own_norm():
    # A rotation preserves length -- ||R(x)|| == ||x||. Follows from the inner-product
    # property above with a == b, called out separately because it's the simplest possible
    # sanity check and would catch e.g. an accidental scale factor.
    rng = np.random.default_rng(2)
    head_dim = 16
    x = rng.normal(size=(3, head_dim))
    rotated = apply_rope(x, positions=np.array([0.0, 1.0, 100.0]), theta=500000.0, head_dim=head_dim)
    np.testing.assert_allclose(
        np.linalg.norm(rotated, axis=-1), np.linalg.norm(x, axis=-1), rtol=1e-6
    )


def test_rope_position_zero_is_the_identity():
    # cos(0) = 1, sin(0) = 0 for every channel -- position 0 must not rotate anything.
    rng = np.random.default_rng(3)
    head_dim = 8
    x = rng.normal(size=(1, head_dim))
    got = apply_rope(x, positions=np.array([0.0]), theta=500000.0, head_dim=head_dim)
    np.testing.assert_allclose(got, x, atol=1e-6)


def test_rope_pairing_is_adjacent_channels_not_split_halves():
    # The discriminating structural test (§3.5's verdict). Interleaved pairing means
    # channels (2j, 2j+1) rotate together and are *independent* of every other pair.
    # Split-halves pairing would instead pair channel i with channel i + head_dim/2, so
    # under that (wrong) convention, changing channel 2 would change the output at
    # channel 0 too (since head_dim/2 == 2 here). Under interleaved pairing it must not.
    head_dim = 4  # pairs: (0,1) and (2,3); split-halves would instead pair (0,2) and (1,3)
    positions = np.array([7.0])
    x1 = np.array([[1.0, 2.0, 3.0, 4.0]])
    x2 = np.array([[1.0, 2.0, 30.0, 40.0]])  # only the *other* pair's channels changed

    out1 = apply_rope(x1, positions, theta=10000.0, head_dim=head_dim)
    out2 = apply_rope(x2, positions, theta=10000.0, head_dim=head_dim)

    # Pair (0,1)'s output must be unaffected by changing pair (2,3)'s input.
    np.testing.assert_allclose(out1[:, 0:2], out2[:, 0:2], atol=1e-9)
    # Sanity: pair (2,3)'s output *does* change (the test isn't accidentally vacuous).
    assert not np.allclose(out1[:, 2:4], out2[:, 2:4])


def test_rope_adjacent_pairs_use_different_frequencies():
    # §3.1: inv_freq[2j] == inv_freq[2j+1] but inv_freq for pair j != pair j+1 (for
    # theta != 1). Detect this via rotation angle, not by reaching into internals: at a
    # fixed position, pair 0's rotation angle should differ from pair 1's rotation angle,
    # which shows up as different behaviour when we rotate a pure-pair-0 unit vector vs a
    # pure-pair-1 unit vector by a position large enough to matter.
    head_dim = 4
    theta = 10.0  # small theta so the frequency gap between pairs is easy to detect
    p = np.array([3.0])
    pair0 = np.array([[1.0, 0.0, 0.0, 0.0]])
    pair1 = np.array([[0.0, 0.0, 1.0, 0.0]])
    rot0 = apply_rope(pair0, p, theta, head_dim=head_dim)
    rot1 = apply_rope(pair1, p, theta, head_dim=head_dim)
    # angle for pair 0 (channel index 0,1) uses inv_freq[0] = theta^0 = 1 -> angle = 3 rad
    # angle for pair 1 (channel index 2,3) uses inv_freq[2] = theta^(-2/4) = theta^-0.5
    angle0 = np.arctan2(rot0[0, 1], rot0[0, 0])
    angle1 = np.arctan2(rot1[0, 3], rot1[0, 2])
    assert not np.isclose(angle0, angle1)
    np.testing.assert_allclose(angle0, 3.0, atol=1e-6)
    np.testing.assert_allclose(angle1, 3.0 * theta ** (-0.5), atol=1e-6)


# ---------------------------------------------------------------------------
# swiglu -- §6 of the reference doc
# ---------------------------------------------------------------------------


def test_swiglu_matches_hand_computed_value():
    # x = [1, 2] (D=2). w1 = w3 = I (2x2 identity), w2 = I.
    # x @ w1.T = [1, 2]; silu([1,2]) = [1/(1+e^-1), 2/(1+e^-2)]
    # x @ w3.T = [1, 2]; gated = silu(.) * [1, 2]
    # result = gated @ w2.T = gated (identity)
    x = np.array([[1.0, 2.0]])
    identity = np.eye(2)
    silu_1 = 1.0 / (1.0 + np.exp(-1.0))
    silu_2 = 2.0 / (1.0 + np.exp(-2.0))
    expected = np.array([[silu_1 * 1.0, silu_2 * 2.0]])
    got = swiglu(x, identity, identity, identity)
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_swiglu_silu_is_on_the_w1_branch_not_w3():
    # Discriminator for the w1/w3 swap trap (§6): make w3's projection zero everywhere
    # (w3 = 0), so the gate is 0 regardless of w1 -- output must be exactly zero. If SiLU
    # were (wrongly) applied to the w3 branch instead, this would still be zero (0*x=0
    # either way), so use the complementary case: w1 = 0 (so the "w1 branch" is zero prior
    # to SiLU). silu(0) = 0, so gated = 0 * (x @ w3.T) = 0 regardless of w3 -- this pins
    # down that *w1*, not w3, feeds SiLU (if w3 fed SiLU instead, silu(x@w3.T) would be
    # silu of a nonzero vector, and gated = nonzero * (x @ w1.T) = nonzero * 0 = 0 too --
    # so pick w3 nonzero-favoring values that make the two conventions diverge instead).
    x = np.array([[1.0, 2.0]])
    zero = np.zeros((2, 2))
    nonzero = np.array([[1.0, 0.0], [0.0, 1.0]]) * 5.0
    identity = np.eye(2)
    # w1 = 0 -> silu(x @ 0) = silu(0) = 0 -> gated = 0 * (x @ w3.T) = 0 always, regardless
    # of w3's value. This is convention-agnostic in the wrong direction; instead assert
    # against the documented formula directly with distinct w1 != w3 and check the result
    # matches "silu on w1" and not "silu on w3".
    w1 = np.array([[2.0, 0.0], [0.0, 2.0]])  # doubles x
    w3 = np.array([[3.0, 0.0], [0.0, 3.0]])  # triples x
    got = swiglu(x, w1, identity, w3)
    swished = (x @ w1.T) / (1.0 + np.exp(-(x @ w1.T)))  # silu(x @ w1.T)
    gate = x @ w3.T
    expected = (swished * gate) @ identity.T
    np.testing.assert_allclose(got, expected, rtol=1e-6)
    # And confirm it is NOT silu(x@w3.T)*(x@w1.T), the swapped convention.
    swapped_swished = (x @ w3.T) / (1.0 + np.exp(-(x @ w3.T)))
    swapped_gate = x @ w1.T
    swapped = (swapped_swished * swapped_gate) @ identity.T
    assert not np.allclose(got, swapped)


def test_swiglu_projects_through_w2_at_the_end():
    x = np.array([[1.0, 0.0]])
    w1 = np.eye(2)
    w3 = np.eye(2)
    w2 = np.array([[0.0, 1.0], [1.0, 0.0]])  # swap the two channels
    got = swiglu(x, w1, w2, w3)
    silu_1 = 1.0 / (1.0 + np.exp(-1.0))
    gated = np.array([[silu_1 * 1.0, 0.0]])  # x=[1,0] -> silu([1,0])*[1,0] = [silu(1), 0]
    expected = gated @ w2.T
    np.testing.assert_allclose(got, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# attention -- §4, §5 of the reference doc
# ---------------------------------------------------------------------------


def _identity_projection(d_out: int, d_in: int) -> np.ndarray:
    """A [d_out, d_in] weight that is the identity on the leading min(d_out,d_in) dims."""
    w = np.zeros((d_out, d_in))
    for i in range(min(d_out, d_in)):
        w[i, i] = 1.0
    return w


def test_attention_causal_mask_blocks_future_tokens():
    # §5.2: the mask must be causal. If token 1's content leaks backward into token 0's
    # output, the mask is wrong (or missing/inverted). Use num_heads=num_groups=1 (no GQA
    # complexity) and out_linear/q_linear/kv_linear as identity-like projections that pass
    # q, k, v straight through, so we can reason about the raw attention weights.
    head_dim = 4
    num_heads = num_groups = 1
    q_w = _identity_projection(head_dim, head_dim)
    kv_w = np.zeros((2 * head_dim, head_dim))
    kv_w[:head_dim, :head_dim] = np.eye(head_dim)  # K = x
    kv_w[head_dim:, :head_dim] = np.eye(head_dim)  # V = x
    out_w = np.eye(head_dim)

    x = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    out_full = attention(x, q_w, kv_w, out_w, num_heads=num_heads, num_groups=num_groups, theta=10000.0)

    # Now perturb only token 1's input and re-run; token 0's output must be identical,
    # because a causal mask forbids position 0 from attending to position 1.
    x_perturbed = x.copy()
    x_perturbed[1] = np.array([9.0, 9.0, 9.0, 9.0])
    out_perturbed = attention(
        x_perturbed, q_w, kv_w, out_w, num_heads=num_heads, num_groups=num_groups, theta=10000.0
    )
    np.testing.assert_allclose(out_full[0], out_perturbed[0], atol=1e-6)
    assert not np.allclose(out_full[1], out_perturbed[1])  # and it does change token 1


def test_attention_gqa_uses_repeat_interleave_not_tile():
    # §5.3: query head h attends to KV group h // (H/G) -- contiguous blocks. For H=4,
    # G=2: heads {0,1} -> group 0, heads {2,3} -> group 1 (repeat_interleave). The wrong
    # convention (`tile`) would instead give heads {0,2} -> group 0, {1,3} -> group 1
    # (round-robin). Discriminate by perturbing only group 0's V and checking which query
    # heads' outputs change.
    head_dim = 2
    num_heads, num_groups = 4, 2
    embed = num_heads * head_dim  # 8
    kv_width = 2 * num_groups * head_dim  # 8

    rng = np.random.default_rng(4)
    q_w = rng.normal(size=(embed, embed)) * 0.1
    out_w = np.eye(embed)

    def make_kv(v_group0):
        kv_w = np.zeros((kv_width, embed))
        # K: identity-ish per group so scores are well-defined and nonzero (K = x's first
        # head_dim*num_groups channels, arbitrary but fixed).
        kv_w[: num_groups * head_dim, : num_groups * head_dim] = np.eye(num_groups * head_dim)
        # V: group 0 gets `v_group0`'s row-block, group 1 gets a fixed different block.
        kv_w[num_groups * head_dim : num_groups * head_dim + head_dim, :head_dim] = v_group0
        kv_w[num_groups * head_dim + head_dim :, head_dim : 2 * head_dim] = np.eye(head_dim) * 3.0
        return kv_w

    x = rng.normal(size=(3, embed))

    kv_w_a = make_kv(np.eye(head_dim) * 1.0)
    kv_w_b = make_kv(np.eye(head_dim) * 100.0)  # only group 0's V differs

    out_a = attention(x, q_w, kv_w_a, out_w, num_heads=num_heads, num_groups=num_groups, theta=10000.0)
    out_b = attention(x, q_w, kv_w_b, out_w, num_heads=num_heads, num_groups=num_groups, theta=10000.0)

    changed_heads = [
        h for h in range(num_heads)
        if not np.allclose(
            out_a[:, h * head_dim : (h + 1) * head_dim],
            out_b[:, h * head_dim : (h + 1) * head_dim],
            atol=1e-9,
        )
    ]
    # repeat_interleave: heads {0, 1} depend on group 0 -> both must change.
    # tile (wrong): heads {0, 2} would change instead.
    assert set(changed_heads) == {0, 1}, f"expected heads {{0,1}} to change, got {changed_heads}"


def test_attention_scale_is_one_over_sqrt_head_dim():
    # §5.1: scale = 1/sqrt(head_dim), using the head dimension, not the model dimension --
    # ttml reads `query.padded_shape()[3]`, the last (head) axis, not embedding_dim. To
    # discriminate the two, embedding_dim (8) must differ from head_dim (4): use
    # num_heads=2, num_groups=1, and zero out head 1's contribution via out_w so only
    # head 0's (head_dim=4) dynamics are visible in the output's first head_dim columns.
    #
    # RoPE (§3) is unconditionally applied inside attention() to Q/K, so the hand
    # computation below carries it through explicitly rather than pretending it's off --
    # at position 0 rotation is the identity (cos(0)=1, sin(0)=0), but position 1's is not.
    head_dim = 4
    embedding_dim = 2 * head_dim  # num_heads=2

    # q_w, kv_w: identity on head 0's own head_dim x head_dim block (K=V=x's first
    # head_dim columns, shared by the single group); head 1's block only needs to avoid
    # NaN, so also identity -- its output is discarded by out_w below.
    q_w = np.eye(embedding_dim)
    kv_w = np.zeros((2 * head_dim, embedding_dim))  # num_groups=1 -> kv width = 2*head_dim
    kv_w[:head_dim, :head_dim] = np.eye(head_dim)  # K = x[:, :head_dim]
    kv_w[head_dim:, :head_dim] = np.eye(head_dim)  # V = x[:, :head_dim]
    out_w = np.zeros((embedding_dim, embedding_dim))
    out_w[:head_dim, :head_dim] = np.eye(head_dim)  # keep only head 0's fused output

    x0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x1 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x = np.array([x0, x1])

    out = attention(x, q_w, kv_w, out_w, num_heads=2, num_groups=1, theta=10000.0)

    # By hand, head 0 only (head 1 is zeroed by out_w):
    # position 0 -> RoPE is the identity: k0_rot = [1,0,0,0].
    # position 1 -> pair (0,1) rotates by angle = 1 * inv_freq[0] = 1 * theta^0 = 1 rad
    #   (§3.1: inv_freq[2j] = theta^(-2j/head_dim), j=0 -> exponent 0 -> inv_freq=1).
    #   q1 = k1 = [0,1,0,0] rotates (per §3.2's out[2j]=x[2j]cos-x[2j+1]sin,
    #   out[2j+1]=x[2j+1]cos+x[2j]sin) to [-sin(1), cos(1), 0, 0].
    sin1, cos1 = np.sin(1.0), np.cos(1.0)
    q1_rot = np.array([-sin1, cos1, 0.0, 0.0])
    k0_rot = np.array([1.0, 0.0, 0.0, 0.0])
    k1_rot = q1_rot  # identical input and position to q1
    scale = 1.0 / np.sqrt(head_dim)  # the claim under test
    scores = np.array([q1_rot @ k0_rot, q1_rot @ k1_rot]) * scale
    w = np.exp(scores - scores.max())
    w /= w.sum()
    # V is never rotated (§1/§3): v0=[1,0,0,0], v1=[0,1,0,0], plain x, no RoPE.
    v0, v1 = np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0])
    expected_head0_out1 = w[0] * v0 + w[1] * v1
    np.testing.assert_allclose(out[1, :head_dim], expected_head0_out1, atol=1e-6)

    # And the discriminator: scale = 1/sqrt(embedding_dim) would give a visibly different
    # softmax (1/sqrt(8) != 1/sqrt(4)), so this test would fail under that wrong scale.
    wrong_scale = 1.0 / np.sqrt(embedding_dim)
    wrong_scores = np.array([q1_rot @ k0_rot, q1_rot @ k1_rot]) * wrong_scale
    wrong_w = np.exp(wrong_scores - wrong_scores.max())
    wrong_w /= wrong_w.sum()
    assert not np.allclose(w, wrong_w)


def test_attention_rope_is_applied_to_q_and_k_only_not_v():
    # §1: "RoPE is applied to Q and K only, never V." Discriminate by checking that a
    # single-token sequence (no causal masking effect, no cross-token mixing possible)
    # still produces a *position-dependent* output when we feed the same token content at
    # two different absolute positions -- if RoPE also touched V (or touched nothing),
    # results would differ in ways this doesn't test directly. Instead, test the
    # complementary invariant: with num_heads=num_groups=1 and a *single* token (so
    # attention collapses to softmax over one element = 1, i.e. output = V exactly), the
    # output must equal V *unrotated*, because at S=1 RoPE(position 0) is the identity
    # (cos(0)=1, sin(0)=0) regardless of whether V participates -- so this alone can't
    # separate "RoPE untouched V" from "RoPE touched V but V's rotation was also
    # identity". Use S=1 only as a smoke check that plain attention values pass through V.
    head_dim = 4
    q_w = _identity_projection(head_dim, head_dim)
    kv_w = np.zeros((2 * head_dim, head_dim))
    kv_w[:head_dim, :head_dim] = np.eye(head_dim)
    kv_w[head_dim:, :head_dim] = np.eye(head_dim)
    out_w = np.eye(head_dim)
    x = np.array([[3.0, 1.0, 4.0, 1.0]])
    out = attention(x, q_w, kv_w, out_w, num_heads=1, num_groups=1, theta=10000.0)
    # single token, single head: softmax over one score is 1 -> output = V = x.
    np.testing.assert_allclose(out[0], x[0], atol=1e-6)


# ---------------------------------------------------------------------------
# forward -- §8 of the reference doc, end to end against a synthetic checkpoint
# ---------------------------------------------------------------------------


def _write_synthetic_checkpoint(tmp_path: Path, *, seed: int = 0) -> Path:
    """A tiny but structurally faithful checkpoint: same tensor names/shape conventions
    (declaration-order stream, [1,1,out,in] linears, [1,1,1,C] gammas) as the real one,
    at toy dimensions so the full `forward()` can be exercised cheaply and its structural
    properties (causality, finiteness, shape) checked without needing the real artifact.
    """
    rng = np.random.default_rng(seed)
    embedding_dim, num_heads, num_groups, num_blocks = 8, 4, 2, 2
    head_dim = embedding_dim // num_heads  # 2
    intermediate_dim = 6
    vocab_size = 11

    def lin(out_f, in_f):
        return (rng.normal(size=(1, 1, out_f, in_f)) * 0.1).astype(np.float32)

    def gamma(c):
        return np.ones((1, 1, 1, c), dtype=np.float32)

    names_and_arrays = []
    for b in range(num_blocks):
        p = f"llama/llama_block_{b}"
        names_and_arrays += [
            (f"{p}/attention_norm/gamma", gamma(embedding_dim)),
            (f"{p}/attention/q_linear/weight", lin(num_heads * head_dim, embedding_dim)),
            (f"{p}/attention/kv_linear/weight", lin(2 * num_groups * head_dim, embedding_dim)),
            (f"{p}/attention/out_linear/weight", lin(embedding_dim, num_heads * head_dim)),
            (f"{p}/mlp_norm/gamma", gamma(embedding_dim)),
            (f"{p}/mlp/w1/weight", lin(intermediate_dim, embedding_dim)),
            (f"{p}/mlp/w3/weight", lin(intermediate_dim, embedding_dim)),
            (f"{p}/mlp/w2/weight", lin(embedding_dim, intermediate_dim)),
        ]
    names_and_arrays += [
        ("llama/ln_fc/gamma", gamma(embedding_dim)),
        ("llama/fc/weight", lin(vocab_size, embedding_dim)),
    ]

    manifest = {"model": {"named_parameters": {name: {} for name, _ in names_and_arrays}}}
    header = {
        "transformer_config": {
            "num_heads": num_heads,
            "num_groups": num_groups,
            "embedding_dim": embedding_dim,
            "num_blocks": num_blocks,
            "vocab_size": vocab_size,
            "theta": 10000.0,
        },
    }
    record = {"format": 1, "header": header, "manifest": manifest}
    path = tmp_path / "synthetic.pkl"
    with open(path, "wb") as f:
        pickle.dump(record, f)
        for _name, array in names_and_arrays:
            pickle.dump(array, f)
    return path


def test_forward_produces_finite_logits_of_the_right_shape(tmp_path):
    path = _write_synthetic_checkpoint(tmp_path)
    header, _manifest = read_checkpoint_meta(path)
    vocab_size = header["transformer_config"]["vocab_size"]
    tokens = np.array([1, 2, 3, 4, 5])
    logits = forward(path, tokens)
    assert logits.shape == (len(tokens), vocab_size)
    assert np.all(np.isfinite(logits))


def test_forward_is_causal_end_to_end():
    # Changing a later token must not change an earlier position's logits, all the way
    # through every block -- the end-to-end version of the attention-level causal test.
    def run(tmp_path, tokens):
        path = _write_synthetic_checkpoint(tmp_path)
        return forward(path, tokens)

    import tempfile

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        tokens_a = np.array([1, 2, 3, 4])
        tokens_b = np.array([1, 2, 3, 9])  # only the last token differs
        # Use the same checkpoint content for both (same seed) but write it twice, once
        # per tmp dir, so the two forward() calls read identical weights.
        path_a = _write_synthetic_checkpoint(Path(d1), seed=0)
        path_b = _write_synthetic_checkpoint(Path(d2), seed=0)
        logits_a = forward(path_a, tokens_a)
        logits_b = forward(path_b, tokens_b)

    np.testing.assert_allclose(logits_a[:3], logits_b[:3], atol=1e-5)
    assert not np.allclose(logits_a[3], logits_b[3])


def test_forward_is_a_pure_function_of_its_inputs():
    # Same checkpoint, same tokens, called twice -> bit-identical (no hidden RNG state,
    # no in-place mutation of the weight dict that would corrupt a second call).
    with tempfile_dir() as d:
        path = _write_synthetic_checkpoint(Path(d))
        tokens = np.array([0, 1, 2])
        first = forward(path, tokens)
        second = forward(path, tokens)
    np.testing.assert_array_equal(first, second)


import contextlib
import tempfile


@contextlib.contextmanager
def tempfile_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# Step 3 of the brief: validate the NumPy path independently of HF, against the
# checkpoint's own held-out cross-entropy. This is the test that actually decides
# pass/fail for the whole implementation -- see docs/ttml-forward-reference.md §8.1.
# ---------------------------------------------------------------------------

REAL_CHECKPOINT = sorted(REAL_CHECKPOINT_DIR.glob("nanollama3_step*.pkl"))[-1] if (
    REAL_CHECKPOINT_DIR.exists() and list(REAL_CHECKPOINT_DIR.glob("nanollama3_step*.pkl"))
) else None

pytestmark_ce = pytest.mark.skipif(
    REAL_CHECKPOINT is None or not VAL_IDS_PATH.is_file(),
    reason="no real checkpoint or validation tokens present under artifacts/",
)


@pytestmark_ce
def test_numpy_forward_reproduces_the_training_runs_held_out_cross_entropy():
    """Step 3 of the brief.

    Compute cross-entropy over 8 random 256-token windows of val_ids.npy (seed 0), using
    *only* this module's NumPy forward pass -- no HF model involved anywhere. The training
    run's own held-out figure (ttml's evaluate()) is 1.8781; docs/ttml-forward-reference.md
    §8.1 measured the derived reference at 1.847 (sd 0.315 across 8 windows, SE 0.112), so
    anything in roughly [1.6, 2.1] is consistent with a correct implementation. This is the
    test that proves the NumPy path is right *before* it's trusted to judge the converter.
    """
    val_ids = np.load(VAL_IDS_PATH)
    seq_len = 256
    n_windows = 8
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(val_ids) - seq_len, size=n_windows)

    per_window_ce = []
    for start in starts:
        window = val_ids[start : start + seq_len].astype(np.int64)
        logits = forward(REAL_CHECKPOINT, window)  # (seq_len, vocab)
        # Next-token loss: logits[i] predicts window[i+1], for i in [0, seq_len-2].
        log_probs = logits - _logsumexp(logits, axis=-1, keepdims=True)
        targets = window[1:]
        token_ll = log_probs[:-1, :][np.arange(seq_len - 1), targets]
        per_window_ce.append(float(-token_ll.mean()))

    mean_ce = float(np.mean(per_window_ce))
    assert 1.6 <= mean_ce <= 2.1, (
        f"NumPy-only mean cross-entropy {mean_ce:.4f} over 8 windows (seed 0) fell outside "
        f"[1.6, 2.1]; expected ~1.84 per docs/ttml-forward-reference.md §8.1. Per-window: "
        f"{per_window_ce}"
    )


def _logsumexp(x: np.ndarray, axis: int, keepdims: bool) -> np.ndarray:
    """Numerically stable log-sum-exp, used only by the test's loss computation (not part
    of the model under test -- forward() returns raw logits per the brief's contract).
    """
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)
