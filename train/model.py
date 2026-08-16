# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The Llama this project trains, and why it is ttml's *Python* Llama rather than its C++ one.

Summary: passing ``mask=None`` to ttml's SDPA makes the kernel run in causal mode instead of
arbitrary-mask mode, which is **1.41x faster on the whole training step** at this project's
384 shape and 1.15x at its 1024 shape (measured, see below). The C++ model cannot be handed a
null mask from Python; the Python model can. So we build the Python model, and drop the
(redundant) causal mask on the way in.

WHY THE MASK COSTS SO MUCH
--------------------------
``ttml::ops::scaled_dot_product_attention`` picks the kernel's mask mode purely from *whether
a mask object was passed*, never from what is in it
(``tt-train/sources/ttml/ops/scaled_dot_product_attention.cpp:249-255``)::

    ttml::metal::AttentionMaskType mask_type = ttml::metal::AttentionMaskType::Causal;
    if (mask.has_value() && mask.value()) {
        mask_tensor = mask.value()->get_value();
        mask_type = ttml::metal::AttentionMaskType::Arbitrary;
    }

Under ``Arbitrary`` the flash-attention program factory must assume every query row attends to
every key tile, so it (a) does the full S x S quadratic work instead of the triangular half and
(b) disables its load-balancing path, which is gated on ``Causal``
(``tt-train/sources/ttml/metal/ops/sdpa_fw/device/sdpa_fw_program_factory.cpp:305-316``)::

    const bool use_balanced_parallelism =
        (mask_type == AttentionMaskType::Causal) && (St % 2 == 0) && ...

A lower-triangular mask of ones is *exactly* what ``Causal`` already means, so passing it buys
nothing and costs roughly 2x the attention FLOPs. ttml's own example training script already
passes ``None`` for Llama (``tt-train/sources/examples/train/train.py:461-467``); it is
``ttml.common.trainer.train()`` — the loop this project reuses — that always builds and passes
one (``ttml/common/trainer.py:73-76, 102``).

WHY NOT JUST PASS ``None`` TO THE C++ MODEL
-------------------------------------------
Because it raises ``TypeError``. ``CppLlama.__call__`` is bound as
(``tt-train/sources/ttml/nanobind/nb_models.cpp:330-337``)::

    py_llama.def("__call__", [](models::llama::Llama& self,
                                const ttml::autograd::TensorPtr& tensor,
                                const ttml::autograd::TensorPtr& mask) { return self(tensor, mask); },
                 nb::arg("tensor"), nb::arg("mask"), ...);

``nb::arg("mask")`` with no ``.none()``, so nanobind refuses ``None`` — even though the C++
``Llama::operator()`` behind it takes ``std::optional<TensorPtr>`` and would accept
``std::nullopt`` happily (``models/llama.hpp:60-68``). Nor is there a back door: ``ModuleBase``'s
three ``operator()`` overloads all ``throw std::logic_error`` in the base
(``modules/module_base.cpp:114-128``) and only the two non-optional ones are bound at all
(``nanobind/nb_modules.cpp:78-86``), so calling the base binding does not reach Llama's override.
Fixing this upstream is a two-line change (take the mask as ``std::optional`` and declare it
``nb::arg("mask") = std::nullopt``, plus the same on the GPT-2 binding above it) — written up
with the diff in ``docs/upstream-tt-metal-asks.md`` — but it needs a tt-metal rebuild, which
we cannot do here.

The Python ``ttml.models.llama.Llama`` has no such problem: it is an ordinary Python class whose
``forward`` hands the mask straight to ``ttml.ops.attention.scaled_dot_product_attention``, and
*that* binding is declared ``nb::arg("mask") = std::nullopt``
(``nanobind/nb_ops.cpp:280-293``) — it takes ``None`` already.

IS THE PYTHON MODEL THE SAME MODEL?
-----------------------------------
Yes, on both axes that matter.

*Architecturally* it composes the identical fused ttml ops the C++ one does — ``swiglu``,
``rope``, ``grouped_heads_creation``, ``sdpa``, ``rmsnorm`` — with the same RMSNorm epsilon
(1e-5), the same SwiGLU intermediate-size derivation (8/3 of the embedding dim rounded up to
256), and the same RoPE parameters. :func:`create_model` below translates our YAML into its
config field-for-field and the resulting parameter count matches the C++ model exactly
(22,025,088 at the 384 shape; 122,962,944 at 1024 — both verified on device).

*In throughput* it is not a slower reference implementation, which was the thing worth checking
before believing any of this. Measured on a single Blackhole p300c, full training step
(forward + cross-entropy + backward + AdamW), s/1000 steps:

===================================  ==========  ==========
shape                                384/2048    1024/512
===================================  ==========  ==========
C++ ``CppLlama``, causal mask         521.7       896.9
Python ``Llama``, same causal mask    521.9       893.6
Python ``Llama``, ``mask=None``       374.6       777.0
**speedup**                           **1.39x**   **1.15x**
===================================  ==========  ==========

(384 shape: batch 16, seq 2048. 1024 shape: batch 64, seq 512. The two masked columns agree to
within 0.4%, i.e. the Python model costs nothing.) The 1024 shape gains less because its
sequence is 4x shorter and its hidden dimension 2.7x wider, so attention is a smaller share of
the step.

Those are an isolated benchmark of the training step. Measured instead end to end through
``train/run.py`` itself — 300 steps, seed 5489, ``artifacts/tokens-v3``,
``train/configs/nanollama3_bpe_v2.yaml``, read off the training loop's own progress bar —
``--model-impl cpp`` vs ``--model-impl python``: **503.3 -> 356.7 s/1000 steps (1.41x)** at the
384 shape and **890.0 -> 776.7 (1.15x)** at 1024. The recorded project baselines reproduce
(519.4 for v3 and 903 for 1024a, agreeing to 3% and 1.4%).

CHECKPOINT COMPATIBILITY
------------------------
The two implementations name their parameters differently, which would have broken checkpoint
resume, ``convert/to_hf.py``, ``convert/ttml_forward.py`` and their tests — all of which match
on literal C++ names such as ``llama/llama_block_0/attention/q_linear/weight``. The difference is
only in two path segments, and :class:`TtTntLlama` erases both, so *nothing downstream changes*:

- **root**: the Python base names the root after the class
  (``ttml/modules/module_base.py``: ``self.create_name(self.__class__.__name__)``), so it would
  be ``TtTntLlama/...``. Fixed by calling ``create_name("llama")`` — the same string
  ``models/llama.cpp:171`` uses.
- **block path**: the Python model holds its blocks in a ``ModuleList`` attribute called
  ``blocks``, which registers children under their index, giving ``blocks/0/``; C++ registers
  each block under a single ``llama_block_0`` segment. Fixed by rewriting the segment in
  :meth:`TtTntLlama.parameters`, which every consumer (the optimizer factory, checkpoint save,
  checkpoint load) goes through.

Everything below the block — ``attention/q_linear/weight``, ``mlp/w1/weight``,
``attention_norm/gamma``, ``fc/weight``, ``ln_fc/gamma`` — is already identical, as is the
weight-tying dedup that drops ``tok_emb/weight`` in favour of ``fc/weight``.

One trap worth naming: the C++ ``LlamaConfig`` defaults ``weight_tying`` to **Enabled**
(``models/llama.hpp:35``) while the Python ``LlamaConfig`` defaults it to **Disabled**. Our YAMLs
set no ``weight_tying`` key, so every checkpoint this project has ever written is tied, and
``train/run.py``'s header stamps ``weight_tying: True`` unconditionally. :func:`create_model`
therefore defaults to Enabled — matching the C++ default rather than the Python one. Getting this
wrong is silent: it produces an untied model, 12.3M extra parameters at the 384 shape, and a
``config.json`` claiming ``tie_word_embeddings: true`` over untied weights.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

#: Root module name the C++ ``models::llama::Llama`` gives itself (``models/llama.cpp:171``),
#: and therefore the prefix every existing checkpoint key carries.
CPP_ROOT_NAME = "llama"

#: ``llama/blocks/<i>/rest`` (Python ``ModuleList`` attribute + index) as written by the Python
#: model once its root has been renamed. Rewritten to the C++ ``llama/llama_block_<i>/rest``.
_PY_BLOCK_PATH = re.compile(r"^" + CPP_ROOT_NAME + r"/blocks/(\d+)/(.+)$")

#: How many distinct mask objects keep a cached causal verdict. See ``TtTntLlama.__init__``.
_MASK_CACHE_SLOTS = 2


def canonical_param_name(name: str) -> str:
    """Rewrite one Python-``Llama`` parameter name into the C++ name for the same tensor.

    The only difference is the block segment: the Python model reaches a block through a
    ``ModuleList`` attribute (``blocks``) plus an index, the C++ model registers it under one
    ``llama_block_<i>`` segment. Names that do not match that shape — ``llama/fc/weight``,
    ``llama/ln_fc/gamma``, and anything already in C++ form — pass through untouched, so this is
    safe to apply to a name from either implementation.

    >>> canonical_param_name("llama/blocks/3/mlp/w1/weight")
    'llama/llama_block_3/mlp/w1/weight'
    >>> canonical_param_name("llama/fc/weight")
    'llama/fc/weight'
    >>> canonical_param_name("llama/llama_block_3/mlp/w1/weight")
    'llama/llama_block_3/mlp/w1/weight'
    """
    return _PY_BLOCK_PATH.sub(rf"{CPP_ROOT_NAME}/llama_block_\1/\2", name)


def llama_config_kwargs(transformer_config: Dict[str, Any]) -> Dict[str, Any]:
    """Translate one of our ``transformer_config`` YAML blocks into Python ``LlamaConfig`` kwargs.

    Pure dict work, no ttml import, so the translation is testable without a device — which
    matters, because a silently-wrong translation here (``num_groups`` landing on the wrong
    field, a missing ``theta``) produces a model that trains perfectly well and is simply not
    the architecture the config describes.

    The two vocabularies of field names, side by side:

    ==================  =============================  ============================
    our YAML            C++ ``CppLlamaConfig``          Python ``LlamaConfig``
    ==================  =============================  ============================
    ``embedding_dim``   ``embedding_dim``               ``hidden_size``
    ``num_blocks``      ``num_blocks``                  ``num_hidden_layers``
    ``num_heads``       ``num_heads``                   ``num_attention_heads``
    ``num_groups``      ``num_groups``                  ``num_key_value_heads``
    ``max_sequence_l.`` ``max_sequence_length``         ``max_position_embeddings``
    ``theta``           ``theta``                       ``rope_theta``
    ``dropout_prob``    ``dropout_prob``                ``attention_dropout`` + ``mlp_dropout``
    ``intermediate_d.`` ``intermediate_dim``            ``intermediate_size``
    ==================  =============================  ============================

    Returns kwargs only; the caller adds the enum-valued ``weight_tying``/``runner_type``, which
    need ttml imported.

    Raises:
        ValueError: if ``vocab_size`` is not a multiple of 32. The two implementations round the
            vocabulary up to a tile differently — the C++ model widens the LM head and *keeps*
            the padded columns (``adjust_vocab_size`` in ``ttml/common/model_factory.py``), the
            Python model widens it and then slices them back off. When 32 divides the vocabulary
            both are no-ops and the models agree exactly; when it does not, they produce
            different numbers of logits and this translation would be a quiet lie. This project's
            vocabulary is 32000, so the guard never fires in practice — it exists so that a
            future size cannot cross the line unnoticed.
    """
    vocab_size = int(transformer_config["vocab_size"])
    if vocab_size % 32:
        raise ValueError(
            f"vocab_size {vocab_size} is not a multiple of 32; the C++ and Python Llama "
            "implementations pad the LM head differently in that case (C++ keeps the padded "
            "logit columns, Python slices them off), so they would not be equivalent"
        )

    dropout = float(transformer_config.get("dropout_prob", 0.0))
    kwargs: Dict[str, Any] = {
        "hidden_size": int(transformer_config["embedding_dim"]),
        "num_hidden_layers": int(transformer_config["num_blocks"]),
        "num_attention_heads": int(transformer_config["num_heads"]),
        "num_key_value_heads": int(transformer_config["num_groups"]),
        "vocab_size": vocab_size,
        "max_position_embeddings": int(transformer_config["max_sequence_length"]),
        "attention_dropout": dropout,
        "mlp_dropout": dropout,
    }
    if transformer_config.get("theta") is not None:
        kwargs["rope_theta"] = float(transformer_config["theta"])
    if transformer_config.get("intermediate_dim") is not None:
        kwargs["intermediate_size"] = int(transformer_config["intermediate_dim"])
    return kwargs


def _is_plain_causal(mask) -> bool:
    """Is ``mask`` exactly the lower-triangular all-ones mask ``build_causal_mask`` produces?

    ``ttml.common.utils.build_causal_mask(T)`` is ``np.tril(np.ones((1, 1, T, T)))`` — 1 means
    attend — which is precisely what ``AttentionMaskType::Causal`` computes without a mask
    tensor. This checks rather than assumes: the whole change rests on the two being the same
    thing, and a mask that is *not* plainly causal (a padding mask, a KV-cache mask, a sliding
    window) must keep its explicit tensor or the model would silently attend where it should not.

    The tensor is pulled to host to compare, which is why :class:`TtTntLlama` caches the verdict
    by object identity and does this once per distinct mask rather than once per step.

    ON A MESH (``--ddp N``) that host read needs a composer, and this is what made the Python
    model's 1.41x mask win and 4-chip DDP fail to compose on first attempt. ``mask.to_numpy()``
    with no composer raises ``TT_FATAL ... Can't get a single buffer from host storage
    distributed over mesh shape MeshShape([1, 4])``: ``ttml.common.trainer.train()`` builds the
    causal mask with a plain ``from_numpy`` and no mapper, which on a mesh device **replicates**
    it, so the tensor genuinely lives on four chips and there is no single buffer to hand back.
    The composer concatenates the replicas along dim 0 and we compare replica 0 — every replica
    is the same array by construction, and taking one copy is the same thing
    ``ttml.checkpointing``'s ``Sharding.gather`` does for replicated parameters.

    How many replicas there are is read off the tensor's own live topology via ttml's
    ``Sharding`` (``ttml/sharding.py``) rather than from the mesh device or a ``--ddp`` value
    threaded down here: ``Sharding.from_tensor`` already handles the unit-mesh and
    no-topology cases by returning ``dist_shape is None``, so the single-chip path stays
    exactly what it was — ``to_numpy()`` with no composer — and this function needs no
    argument it did not need before.

    Note the shape check runs first and on ``mask.shape()``, which is the **logical** (per-device)
    shape, not the composed one — so the ``[1, 1, T, T]`` test means the same thing on one chip
    and on four.
    """
    import numpy as np
    import ttml
    from ttml.sharding import Sharding

    shape = tuple(mask.shape())
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 1 or shape[2] != shape[3]:
        return False

    dist_shape = Sharding.from_tensor(mask).dist_shape
    replicas = 1 if dist_shape is None else int(np.prod(dist_shape))
    composer = None
    if replicas > 1:
        device = ttml.autograd.AutoContext.get_instance().get_device()
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)

    values = np.asarray(mask.to_numpy(composer=composer), dtype=np.float32)
    if composer is not None:
        # [replicas, 1, T, T] -> replica 0. shape[0] is 1 for every mask this can accept.
        values = values[: shape[0]]
    return bool(np.array_equal(values, np.tril(np.ones(shape, dtype=np.float32))))


def create_model(yaml_config: Dict[str, Any], transformer_config: Dict[str, Any]):
    """Build the model ``train/run.py`` trains, from the same config the C++ factory reads.

    This stands in for ``ttml.common.model_factory.TransformerModelFactory.create_model()`` on
    the Llama path only, and only because that factory reaches ``CppLlama``, which cannot be
    given a null mask (see this module's docstring). Everything it does is a translation of that
    factory's ``_create_llama``; no model logic is reimplemented.

    Args:
        yaml_config: the full assembled config, read for ``device_config`` (tensor parallelism
            must be off — the TP path is a different, distributed model).
        transformer_config: the ``transformer_config`` block of the model YAML.

    Returns:
        A :class:`TtTntLlama`.
    """
    import ttml

    device_config = yaml_config.get("device_config", {})
    if device_config.get("enable_tp"):
        raise ValueError(
            "train.model.create_model builds the single-device Llama; enable_tp needs "
            "ttml.models.distributed.llama, whose parameter layout and names differ"
        )

    kwargs = llama_config_kwargs(transformer_config)

    # The C++ default, not the Python one -- see this module's docstring. `weight_tying` is
    # absent from every YAML we ship, so this branch is what actually decides it.
    tying = str(transformer_config.get("weight_tying", "enabled")).lower()
    kwargs["weight_tying"] = (
        ttml.models.WeightTyingType.Enabled if "enabled" in tying else ttml.models.WeightTyingType.Disabled
    )

    runner = str(transformer_config.get("runner_type", "default")).strip().lower()
    kwargs["runner_type"] = (
        ttml.models.RunnerType.MemoryEfficient if runner == "memory_efficient" else ttml.models.RunnerType.Default
    )

    from ttml.models.llama import LlamaConfig

    return tt_tnt_llama_class()(LlamaConfig(**kwargs))


#: Memoised result of :func:`tt_tnt_llama_class`.
_TT_TNT_LLAMA_CLASS: Optional[type] = None


def tt_tnt_llama_class() -> type:
    """ttml's Python ``Llama``, subclassed with C++ parameter names and the causal mask dropped.

    Built on first use rather than at module scope so that importing ``train.model`` does not
    require ttml — the pure functions above (:func:`canonical_param_name`,
    :func:`llama_config_kwargs`) are host-testable on a machine with no device, and merely
    importing ttml opens the cluster.

    Two behaviours are added to ttml's ``Llama`` and nothing else:

    1. ``parameters()`` returns C++-style names, so checkpoints, the optimizer, and ``convert/``
       all see exactly what they saw before.
    2. ``forward()`` replaces a verified plain-causal mask with ``None``, which is what makes the
       SDPA kernel take its causal path.
    """
    global _TT_TNT_LLAMA_CLASS
    if _TT_TNT_LLAMA_CLASS is not None:
        return _TT_TNT_LLAMA_CLASS

    from ttml.models.llama import Llama

    class TtTntLlama(Llama):
        """ttml's Python ``Llama`` with C++ parameter names and the redundant causal mask dropped.

        See :mod:`train.model` for why this project trains the Python Llama rather than the C++
        one, and what each of the two overrides below is for.
        """

        def __init__(self, config) -> None:
            super().__init__(config)
            # ttml's AbstractModuleBase named the root after this class; rename it to the string
            # `models/llama.cpp:171` uses, so parameter names differ from the C++ model's in the
            # block segment only (which `parameters()` below fixes).
            self.create_name(CPP_ROOT_NAME)
            # Verdicts from `_is_plain_causal`, newest last, at most `_MASK_CACHE_SLOTS` of
            # them. ttml's train() builds one mask before the loop and passes that same object
            # every step, so caching turns a per-step 2048x2048 host round-trip into a single
            # one-off check. Two slots rather than one because run.py alternates a training
            # mask with evaluate()'s own; two rather than unbounded because each entry holds a
            # mask tensor alive on the device (8 MB at seq 2048) long after its caller is done
            # with it. Keyed on object *identity*, not equality: a different tensor object is
            # re-verified from scratch, which is the safe direction to be wrong in.
            object.__setattr__(self, "_causal_mask_verdicts", [])

        def parameters(self):
            """The live parameters, under the names the C++ ``Llama`` would have given them.

            Every consumer in this project goes through here — ``create_optimizer``,
            ``checkpoint.save``, ``checkpoint.load`` — so renaming at this one point is enough to
            keep checkpoints, HF conversion, and the NumPy parity reference working unchanged,
            and to let ``--resume`` read checkpoints written by the C++ model. The values are the
            same live ``Tensor`` objects, not copies, so ``load_checkpoint``'s in-place
            ``assign`` still lands on the model's real weights.
            """
            import ttml

            renamed = ttml.NamedParameters()
            for name, tensor in super().parameters().items():
                renamed[canonical_param_name(name)] = tensor
            return renamed

        def forward(self, input, mask=None, kv_cache=None, new_tokens=None):
            """Forward pass, dropping the mask when it is redundant with causal attention.

            A mask is dropped only when it is *verified* to be the plain lower-triangular causal
            mask (see :func:`_is_plain_causal`) and only on the no-KV-cache path. With a KV cache
            the mask is not square and ``forward_kv`` reads ``mask.shape()[-1]`` to size its cache
            slice, so it is load-bearing there and is always passed through.
            """
            if mask is not None and kv_cache is None:
                if self._verified_causal(mask):
                    mask = None
            return super().forward(input, mask, kv_cache, new_tokens)

        def _verified_causal(self, mask) -> bool:
            """Cached :func:`_is_plain_causal`. See ``__init__`` for the cache's shape."""
            cache = self._causal_mask_verdicts
            for cached_mask, verdict in cache:
                if cached_mask is mask:
                    return verdict
            verdict = _is_plain_causal(mask)
            cache.append((mask, verdict))
            del cache[:-_MASK_CACHE_SLOTS]
            return verdict

    _TT_TNT_LLAMA_CLASS = TtTntLlama
    return TtTntLlama
