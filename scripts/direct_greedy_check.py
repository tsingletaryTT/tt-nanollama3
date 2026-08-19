# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The missing arm of the decode bisection: greedy generation on device WITHOUT vLLM.

WHY THIS EXISTS
---------------
``docs/measurements/serving-on-v077.json`` records a device-vs-CPU looping gap on
``episod/tt-tnt-1024``: local repeat rate 0.1125 on device against 0.0125 on the
CPU reference, same weights, same prompts, same greedy settings. One prompt reaches
0.80 -- "The cat. The cat. The cat."

An A/B across plugin ``c127c17`` and ``bd150c7`` came out byte-identical, so the
plugin is ruled out. That leaves two suspects and no way to separate them from the
vLLM path alone:

    the vLLM serving layer   (scheduler, paging, batching, sampling glue)
    the ttnn decode path     (tt_transformers, the model, the kernels)

This script is the discriminator. It drives ``tt_transformers``' own ``Generator``
directly -- prefill then one-token-at-a-time decode, greedy argmax on the host --
with no vLLM anywhere. Same prompts and same metric as
``scripts/free_running_check.py`` so the numbers are directly comparable.

    direct loops too      -> the defect is in ttnn/tt_transformers
    direct is clean       -> the defect is in the vLLM layer

This is the same method that localised the ORIGINAL decode defect (see
``docs/measurements/decode-bisection-direct-vs-vllm.json``, where direct scored
0.0 against vLLM's 0.2222 and pointed straight at the plugin). It worked once;
it is worth running again rather than reasoning about it.

WHY GREEDY AND WHY HOST-SIDE ARGMAX
-----------------------------------
``sampling_params=None`` makes ``decode_forward`` return logits instead of
sampling on device, so the token choice is a plain host-side ``argmax``. That
removes the sampler from the comparison entirely: any looping seen here is the
model's own forward pass, not a draw.

USAGE
-----
Needs a gozer lease. Note that the direct path opens a 1x1 mesh, which works on a
P300 board where the vLLM path could not be pinned to one chip -- direct TTNN
device access can open a single enumerated chip.

    gozer run --chips 1 --who claude:decode-bisect --reason "direct greedy arm" -- \
      python scripts/direct_greedy_check.py --hf-model artifacts/hf-tt-tnt-1024-dialogue
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

#: The same six prompts free_running_check.py uses. Hard-coded rather than imported
#: so a change there cannot silently make the two arms incomparable.
PROMPTS = [
    "Once upon a time, there was a little",
    "One day, a small boy named Tom went to the park to",
    "The cat sat on the mat and looked at the",
    "Lily and her brother found a big red ball in the",
    "There was a dog who loved to run. Every morning he would",
    "The little girl opened the box and inside she found a",
]


def local_repeat(tokens: list[str], window: int = 4) -> float:
    """Fraction of tokens already seen in the preceding *window* tokens.

    The definition is copied from decode-defect-resolved.json verbatim, because a
    metric that drifts between runs cannot bisect anything.
    """
    if not tokens:
        return 0.0
    hits = sum(1 for i, t in enumerate(tokens) if t in tokens[max(0, i - window):i])
    return hits / len(tokens)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-model", default=str(ROOT / "artifacts" / "hf-tt-tnt-1024-dialogue"))
    p.add_argument("--tokens", type=int, default=40)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--paged", action="store_true",
                   help="enable paged attention, as vLLM necessarily uses. This is the "
                        "one structural difference left between the two arms once the "
                        "plugin has been ruled out by A/B: the default direct path runs "
                        "paged_attention_config=None, while vLLM cannot.")
    p.add_argument("--block-size", type=int, default=64, help="paged block size (vLLM uses 64)")
    p.add_argument("--max-blocks", type=int, default=512, help="paged max_num_blocks")
    p.add_argument("--json", type=Path, default=None)
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

    paged_cfg = None
    if args.paged:
        # Enough blocks to cover max_seq_len for a single sequence, which is all this
        # measurement runs. Under-allocating here would produce a DIFFERENT bug than
        # the one being chased.
        # Generous, not minimal. The first attempt sized this to exactly cover one
        # sequence (8 blocks) and paged_fill_cache threw "ShapeBase[] index out of
        # range" -- the op has its own constraints on max_num_blocks beyond
        # "enough for the sequence", so under-sizing produces a harness bug that
        # looks like a finding.
        paged_cfg = PagedAttentionConfig(block_size=args.block_size, max_num_blocks=args.max_blocks)

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    results = []
    try:
        model_args, model, tt_kv_cache, _state = create_tt_model(
            mesh_device,
            instruct=False,
            max_batch_size=1,
            optimizations=lambda a: DecodersPrecision.performance(a.n_layers, a.model_name),
            max_seq_len=args.max_seq_len,
            paged_attention_config=paged_cfg,
        )
        generator = Generator([model], [model_args], mesh_device, tokenizer=model_args.tokenizer)
        tokenizer = model_args.tokenizer
        vocab = int(getattr(model_args, "vocab_size", 32000))

        print(f"model    : {args.hf_model}")
        print(f"path     : direct tt_transformers Generator, 1x1 mesh, greedy, NO vLLM")
        print(f"paged    : {'ON block_size=' + str(args.block_size) if args.paged else 'OFF'}")
        print(f"tokens   : {args.tokens}\n")

        for prompt in PROMPTS:
            ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
            page_table = None
            if paged_cfg is not None:
                # Built exactly the way models/tt_transformers/demo/simple_text_demo.py
                # builds it (create_tt_page_table), including the block shuffle. A
                # hand-rolled identity table is not the thing vLLM exercises, and the
                # point of this arm is to reproduce vLLM's conditions, not simplify them.
                permutation = torch.randperm(args.max_blocks)
                page_table = torch.argsort(permutation).reshape(1, args.max_blocks)
            logits = generator.prefill_forward_text(
                ids, page_table=page_table, kv_cache=tt_kv_cache,
                prompt_lens=torch.tensor([ids.shape[1]]),
            )
            generated: list[str] = []
            for step in range(args.tokens):
                arr = np.asarray(logits, dtype=np.float32)
                # Last position's row, real (unpadded) vocab prefix only -- the padded
                # tail would shift every token id.
                row = arr.reshape(-1, arr.shape[-1])[-1][:vocab]
                token = int(row.argmax())
                generated.append(tokenizer.decode([token]))
                ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
                pos = torch.tensor([ids.shape[1] - 1])
                # decode_forward returns a TUPLE (logits, log_probs); prefill returns
                # a bare tensor. Unpacking matters -- see sample_topological_full_device.
                logits, _ = generator.decode_forward(
                    torch.tensor([[token]]), pos, page_table=page_table, kv_cache=tt_kv_cache,
                    enable_trace=False, read_from_device=True, sampling_params=None,
                    reset_batch=(step == 0),
                )
            rate = local_repeat(generated)
            results.append({"prompt": prompt, "repeat_rate": rate, "tokens": generated})
            print(f"  {rate:.3f}  {prompt[:44]:<46} {''.join(generated[:12]).strip()[:40]!r}")
    finally:
        ttnn.close_mesh_device(mesh_device)

    rates = [r["repeat_rate"] for r in results]
    median = statistics.median(rates)
    print(f"\ndirect local repeat: median {median:.4f}   per-prompt {[round(x, 3) for x in rates]}")
    print("\nfor comparison (docs/measurements/serving-on-v077.json, same model/prompts):")
    print("  cpu reference  0.0125")
    print("  vLLM device    0.1125")
    print("\nreading: if direct is near the CPU figure the defect is in the vLLM layer;")
    print("         if direct is near the vLLM figure it is in ttnn/tt_transformers.")

    if args.json:
        args.json.write_text(json.dumps(
            {"model": args.hf_model, "path": "direct tt_transformers, greedy, no vLLM",
             "paged": bool(args.paged), "block_size": args.block_size if args.paged else None,
             "tokens": args.tokens, "median_repeat": median,
             "per_prompt": rates, "results": results}, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
