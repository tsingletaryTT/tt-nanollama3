# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for :mod:`train.model` — the Python-Llama wrapper that drops the redundant mask.

Everything here is pure host code: no device, no ttml import. The two things
:mod:`train.model` does that could silently corrupt a training run are both testable that
way, and both are tested here:

1. **Parameter renaming.** :func:`train.model.canonical_param_name` is the single point that
   keeps the Python model's checkpoints readable by ``convert/to_hf.py``,
   ``convert/ttml_forward.py``, and ``--resume``. If it renamed too little, new checkpoints
   would be unconvertible; if it renamed too much, it would corrupt names that were already
   correct. Both directions are covered, and the "already correct" direction is checked
   against a **real shipped checkpoint's actual manifest** rather than a hand-written list.

2. **Config translation.** :func:`train.model.llama_config_kwargs` moves our YAML's field
   names onto the Python ``LlamaConfig``'s HuggingFace-style ones. Getting ``num_groups``
   onto ``num_key_value_heads`` (rather than, say, ``num_attention_heads``) is the kind of
   mistake that trains a perfectly healthy model of the wrong architecture, so every field is
   checked against the size registry for every registered size.

What is *not* testable here, and where the evidence for it lives instead: that the renamed
names match the C++ model's on a live model, that an existing C++ checkpoint resumes into the
Python one, that ``mask=None`` is numerically equivalent to an explicit causal mask, and that
it is strictly causal. Those are on-device facts, measured and recorded in
``.superpowers/attention-mask-fix.md``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from convert.checkpoint_reader import read_checkpoint_meta, tensor_names
from train.model import CPP_ROOT_NAME, canonical_param_name, llama_config_kwargs
from train.sizes import SIZES, get_size

ROOT = Path(__file__).resolve().parent.parent

#: A checkpoint written by the C++ ``CppLlama``, i.e. one whose parameter names are the
#: canonical form by construction. Used to check that canonicalisation is the identity on
#: names that are already right — a hand-written list would only prove the function agrees
#: with this file's author, not with what ttml actually wrote.
CPP_CHECKPOINT = ROOT / "artifacts" / "checkpoints-tt-tnt-v5" / "tt_tnt_step00010764.pkl"

ALL_SIZES = sorted(SIZES)


# --------------------------------------------------------------------------------------
# canonical_param_name
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "python_name, cpp_name",
    [
        (
            "llama/blocks/0/attention/q_linear/weight",
            "llama/llama_block_0/attention/q_linear/weight",
        ),
        (
            "llama/blocks/5/attention/out_linear/weight",
            "llama/llama_block_5/attention/out_linear/weight",
        ),
        ("llama/blocks/3/attention_norm/gamma", "llama/llama_block_3/attention_norm/gamma"),
        ("llama/blocks/2/mlp_norm/gamma", "llama/llama_block_2/mlp_norm/gamma"),
        ("llama/blocks/4/mlp/w1/weight", "llama/llama_block_4/mlp/w1/weight"),
        ("llama/blocks/4/mlp/w2/weight", "llama/llama_block_4/mlp/w2/weight"),
        ("llama/blocks/4/mlp/w3/weight", "llama/llama_block_4/mlp/w3/weight"),
        # Double digits: the 1024 size has 8 blocks today, but nothing stops a future one
        # from having more, and a regex anchored on a single digit would quietly stop
        # renaming at block 10 while continuing to work for 0-9.
        ("llama/blocks/12/mlp/w1/weight", "llama/llama_block_12/mlp/w1/weight"),
    ],
)
def test_block_paths_are_rewritten_to_the_cpp_segment(python_name, cpp_name):
    """``blocks/<i>/`` (Python ``ModuleList``) becomes ``llama_block_<i>/`` (C++)."""
    assert canonical_param_name(python_name) == cpp_name


@pytest.mark.parametrize(
    "name",
    [
        "llama/fc/weight",
        "llama/ln_fc/gamma",
        "llama/tok_emb/weight",
    ],
)
def test_top_level_names_are_already_identical_and_pass_through(name):
    """Only the block segment differs between the two implementations; nothing else moves."""
    assert canonical_param_name(name) == name


def test_canonicalisation_is_the_identity_on_a_real_cpp_checkpoint():
    """Every name a C++-written checkpoint carries must survive canonicalisation untouched.

    This is the direction that would be catastrophic and silent: if the rewrite mangled names
    that were already canonical, ``--resume`` would still "work" (it would raise), but
    ``checkpoint.save`` would start writing a third naming scheme that neither
    ``convert/to_hf.py`` nor ``convert/ttml_forward.py`` can read.
    """
    if not CPP_CHECKPOINT.is_file():
        pytest.skip(f"{CPP_CHECKPOINT} not present")
    _, manifest = read_checkpoint_meta(CPP_CHECKPOINT)
    names = tensor_names(manifest)
    assert names, "checkpoint manifest carried no model tensors"
    unchanged = [n for n in names if canonical_param_name(n) == n]
    assert unchanged == names, (
        "canonical_param_name altered names that were already in C++ form: "
        f"{sorted(set(names) - set(unchanged))[:5]}"
    )


def test_canonicalisation_is_idempotent():
    """Applying it twice must equal applying it once — it is applied per-name, per-save."""
    name = "llama/blocks/7/attention/kv_linear/weight"
    once = canonical_param_name(name)
    assert canonical_param_name(once) == once


def test_a_differently_rooted_name_is_left_alone():
    """The rewrite is anchored on the root the model renames itself to.

    ``TtTntLlama.__init__`` calls ``create_name(CPP_ROOT_NAME)``, so by the time
    ``parameters()`` runs, every name starts with that root. A name that does not is not one
    of ours, and guessing at it would be worse than leaving it.
    """
    assert CPP_ROOT_NAME == "llama"
    assert canonical_param_name("gpt2/blocks/0/mlp/w1/weight") == "gpt2/blocks/0/mlp/w1/weight"


# --------------------------------------------------------------------------------------
# llama_config_kwargs
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("size_name", ALL_SIZES)
def test_every_registered_size_translates_field_for_field(size_name):
    """Each of our YAML's fields must land on the Python ``LlamaConfig``'s equivalent.

    Checked against ``train.sizes``' own registry rather than against the YAML, so this also
    fails if the two ever disagree about what the architecture is.
    """
    size = get_size(size_name)
    tc = size.to_transformer_config()
    kwargs = llama_config_kwargs(tc)

    assert kwargs["hidden_size"] == size.embedding_dim
    assert kwargs["num_hidden_layers"] == size.num_blocks
    assert kwargs["num_attention_heads"] == size.num_heads
    assert kwargs["num_key_value_heads"] == size.num_groups
    assert kwargs["vocab_size"] == size.vocab_size
    assert kwargs["max_position_embeddings"] == size.max_sequence_length
    assert kwargs["rope_theta"] == tc["theta"]


@pytest.mark.parametrize("size_name", ALL_SIZES)
def test_no_cpp_field_name_leaks_into_the_python_config(size_name):
    """A C++ field name reaching ``LlamaConfig(**kwargs)`` is a ``TypeError`` at model build.

    Better to catch the whole class of them here, on a machine with no device, than at the
    top of a training run that has already opened a board.
    """
    kwargs = llama_config_kwargs(get_size(size_name).to_transformer_config())
    cpp_only = {
        "embedding_dim",
        "num_blocks",
        "num_heads",
        "num_groups",
        "max_sequence_length",
        "theta",
        "dropout_prob",
        "intermediate_dim",
        "model_type",
        "runner_type",
    }
    assert not (cpp_only & set(kwargs)), f"C++ field names leaked: {sorted(cpp_only & set(kwargs))}"


def test_dropout_reaches_both_of_the_python_configs_two_dropout_fields():
    """Our YAML has one ``dropout_prob``; the Python config splits it in two.

    Dropping either half would leave that part of the block undropped — invisible at our
    ``dropout_prob: 0.0``, and wrong the moment anyone sets it.
    """
    kwargs = llama_config_kwargs(
        {
            "embedding_dim": 384,
            "num_blocks": 6,
            "num_heads": 6,
            "num_groups": 3,
            "vocab_size": 32000,
            "max_sequence_length": 2048,
            "dropout_prob": 0.1,
        }
    )
    assert kwargs["attention_dropout"] == 0.1
    assert kwargs["mlp_dropout"] == 0.1


def test_optional_fields_are_omitted_rather_than_defaulted_when_absent():
    """``theta`` and ``intermediate_dim`` absent means "let ttml derive it", not "pass None".

    ``intermediate_size=None`` happens to be the Python config's own default so it would
    survive, but ``rope_theta=None`` would not — and silently substituting our own default
    for ttml's would be exactly the kind of divergence ``train/sizes.py`` exists to prevent.
    """
    kwargs = llama_config_kwargs(
        {
            "embedding_dim": 384,
            "num_blocks": 6,
            "num_heads": 6,
            "num_groups": 3,
            "vocab_size": 32000,
            "max_sequence_length": 2048,
        }
    )
    assert "rope_theta" not in kwargs
    assert "intermediate_size" not in kwargs


def test_an_explicit_intermediate_dim_is_carried_across():
    kwargs = llama_config_kwargs(
        {
            "embedding_dim": 384,
            "num_blocks": 6,
            "num_heads": 6,
            "num_groups": 3,
            "vocab_size": 32000,
            "max_sequence_length": 2048,
            "intermediate_dim": 1024,
        }
    )
    assert kwargs["intermediate_size"] == 1024


def test_a_non_tile_aligned_vocab_is_rejected_rather_than_silently_diverging():
    """The one place the two implementations genuinely disagree, made loud.

    The C++ model rounds the vocabulary up to a tile and keeps the padded logit columns; the
    Python model rounds up and slices them back off. At a multiple of 32 both are no-ops and
    the models are identical, which is why this project (vocab 32000) is unaffected — but a
    future size that crossed the line would produce a different number of logits, a different
    loss, and no error at all.
    """
    with pytest.raises(ValueError, match="multiple of 32"):
        llama_config_kwargs(
            {
                "embedding_dim": 384,
                "num_blocks": 6,
                "num_heads": 6,
                "num_groups": 3,
                "vocab_size": 32001,
                "max_sequence_length": 2048,
            }
        )


def test_weight_tying_is_not_decided_here():
    """``llama_config_kwargs`` is pure translation; the tying default belongs to ``create_model``.

    Worth pinning, because the two implementations default it *oppositely* (C++ Enabled,
    Python Disabled) and this project's checkpoints are all tied. If tying ever leaked into
    this function it would need the enum, and therefore ttml, and therefore a device — which
    is precisely why it lives in ``create_model`` instead.
    """
    kwargs = llama_config_kwargs(get_size("384").to_transformer_config())
    assert "weight_tying" not in kwargs
    assert "runner_type" not in kwargs
