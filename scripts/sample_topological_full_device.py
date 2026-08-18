#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Generation with BOTH the transformer and the sampler on Tenstorrent silicon.

What changed from sample_topological_device.py
----------------------------------------------
That script ran the sampler on 110 Tensix cores and the transformer in PyTorch on
the host. This one runs the forward pass on the device too, via
``tt_transformers``' own ``Generator``, so nothing in the token loop touches
PyTorch except the final argmax.

The seam is ``Generator.decode_forward(..., sampling_params=None)``. With no
sampling params the generator returns LOGITS rather than tokens, which is exactly
what a custom sampler needs -- the built-in path would otherwise do its own
top-k-of-32 and hand back a token, discarding the field this design samples from.

What is still on the host
-------------------------
The permutation of logits into per-core tiles, and the final argmax over the
masked field. Both are there for the same reason as before: ``reduce`` returns a
winning value and not its index, and the neighbourhood mask has to be applied at
that same point. Moving them on-device needs a gather and an argmax-with-index,
which is the next increment rather than this one.

Requires hardware, and the find_grid shim, on a harvested grid:

    gozer run --chips 1 --who "claude:tt-tnt" --reason "full-device topological generation" -- \
        python scripts/sample_topological_full_device.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", default="Once upon a time, there was a little")
    p.add_argument("--hf-model", default=str(ROOT / "artifacts" / "hf"))
    p.add_argument("--map", type=Path, default=ROOT / "artifacts" / "token_core_map.npz")
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--hops", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=20260818)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--check-against-cpu", action="store_true",
                   help="compare the device forward to a CPU reference at step 0")
    args = p.parse_args()

    # tt_transformers reads the model from the environment, not from an argument.
    os.environ.setdefault("HF_MODEL", args.hf_model)
    os.environ.setdefault("MESH_DEVICE", "P150")

    import torch
    import ttnn
    from models.tt_transformers.tt.common import PagedAttentionConfig, create_tt_model
    from models.tt_transformers.tt.generator import Generator
    from models.tt_transformers.tt.model_config import DecodersPrecision

    # The harvested-grid shim the vLLM adapter also installs. Without it the stock
    # find_grid picks a grid this die does not have and the program fails with
    # `not on_dispatch_core`.
    sys.path.insert(0, "/home/ttuser/tt-metal")
    from tt_tnt_patch_plugin import _find_grid_from_device  # noqa: F401  (installs on import)

    from scripts.sample_topological import TokenCoreMap, neighbourhood
    from scripts.sample_topological_device import DeviceSampler

    layout = TokenCoreMap.load(args.map)
    membership = [np.flatnonzero(layout.token_cell == c) for c in range(layout.n_cells)]

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        model_args, model, tt_kv_cache, _state = create_tt_model(
            mesh_device,
            instruct=False,
            max_batch_size=1,
            optimizations=lambda a: DecodersPrecision.performance(a.n_layers, a.model_name),
            max_seq_len=args.max_seq_len,
            paged_attention_config=None,
        )
        generator = Generator([model], [model_args], mesh_device,
                              tokenizer=model_args.tokenizer)
        tokenizer = model_args.tokenizer

        sampler = DeviceSampler(mesh_device, layout, membership,
                                args.temperature, args.seed)

        # This is a HF PreTrainedTokenizerFast, not a Llama-style tokenizer: no bos/eos kwargs.
        ids = tokenizer.encode(args.prompt)
        ids = torch.tensor([ids], dtype=torch.long)
        print(f"prompt: {args.prompt!r}  ({ids.shape[1]} tokens)")

        # Prefill on device, then decode one token at a time.
        logits = generator.prefill_forward_text(ids, page_table=None, kv_cache=tt_kv_cache,
                                                prompt_lens=torch.tensor([ids.shape[1]]))
        vocab = layout.token_cell.shape[0]

        # Compare the device forward against a CPU reference on the SAME input, once.
        # The sampler is already known good on host logits, so this separates "my decode
        # loop is wrong" from "the device logits are wrong" instead of guessing between them.
        if args.check_against_cpu:
            from transformers import AutoModelForCausalLM
            ref = AutoModelForCausalLM.from_pretrained(args.hf_model, torch_dtype="auto").eval()
            with torch.no_grad():
                ref_logits = ref(ids).logits[0, -1].float().numpy()
            dev0 = np.asarray(logits, dtype=np.float32).reshape(-1, 32000)[-1]
            import numpy.linalg as la
            cos = float(dev0 @ ref_logits / (la.norm(dev0) * la.norm(ref_logits) + 1e-9))
            print(f"[check] prefill logits vs CPU: cosine={cos:.6f} "
                  f"argmax dev={int(dev0.argmax())} cpu={int(ref_logits.argmax())}")
        print(f"[shape] prefill logits: type={type(logits).__name__} "
              f"shape={getattr(logits, 'shape', None)}  vocab={vocab} "
              f"padded={getattr(model_args, 'padded_vocab_size', '?')}")
        visited = []
        for step in range(args.max_new_tokens):
            arr = np.asarray(logits, dtype=np.float32)
            if step == 0:
                print(f"[shape] step0 array shape={arr.shape} size={arr.size}")
            # Take the LAST position's row, then the real (unpadded) vocab prefix.
            # tt_transformers pads the vocab, so slicing the tail of a flattened
            # tensor picks up padding and shifts every token id.
            row = arr.reshape(-1, arr.shape[-1])[-1][:vocab]
            token, cell = sampler.sample(row, step, hops=args.hops, direction=None)
            visited.append(cell)
            ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
            pos = torch.tensor([ids.shape[1] - 1])
            # decode_forward returns (logits, log_probs) -- a TUPLE -- while
            # prefill_forward_text returns a bare tensor. Assigning the tuple to `logits`
            # is why the first token was correct and every token after it was salad.
            logits, _log_probs = generator.decode_forward(
                torch.tensor([[token]]), pos, page_table=None, kv_cache=tt_kv_cache,
                enable_trace=False, read_from_device=True, sampling_params=None,
                reset_batch=(step == 0),
            )

        text = tokenizer.decode(ids[0].tolist())
        print(f"\nOUTPUT: {text}")
        print(f"cells visited: {visited}")
        print(f"distinct cores: {len(set(visited))}/{len(visited)}")
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
