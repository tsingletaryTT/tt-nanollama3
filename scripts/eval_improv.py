#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Stage-1 evaluation: the swap test FIRST, then schema adherence and failure rates.

THE SWAP TEST RUNS FIRST ON PURPOSE. If substituting another story's think-block barely
changes the continuation, the model learned to emit a plan and ignore it -- the thinking is
decorative and stage 1 has failed, regardless of how good the other numbers look. This
control exists because die-region gate seeding once worked as a classifier (61.2% against
10% chance) and bought nothing measurable in loss. Skill at producing an artefact is not
evidence the artefact is used.

CONTROLLER RULING -- OVERRIDES THE ORIGINAL TASK-6 BRIEF'S `paired_verdict`
============================================================================
The brief's own `paired_verdict` computed ``se = sd / sqrt(n)`` and then
``t = mean/se if se > 0 else 0.0``. For a PERFECT constant separation -- every delta
identical and non-zero -- ``sd`` is 0, so ``se`` is 0, so ``t`` collapses to 0 and the
verdict comes back NOT INTERPRETABLE. That is backwards: identical non-zero deltas across
many paired points are the STRONGEST possible signal, not the weakest. Measured directly:
``paired_verdict([1.0]*10, [2.0]*10)`` under the brief's own formula gives mean=-1.0,
sd=0.0, t=0.0, verdict NOT INTERPRETABLE -- and the brief's own
``test_a_clear_separation_is_reported`` cannot pass as written.

The fix adds an explicit zero-scatter guard:
  - sd == 0 and mean != 0  -> every paired point moved the same, non-zero amount. That is
    the perfect-separation case. Treated as significant: t is reported as ``inf`` (not
    fudged to some finite value above the threshold), and the verdict is decided by the
    sign of ``mean`` exactly as the normal branch does.
  - sd == 0 and mean == 0  -> the two series are IDENTICAL. That genuinely carries no
    information about which arm is better, and stays NOT INTERPRETABLE.

Do not "simplify" this guard away by going back to ``t = mean/se if se > 0 else 0.0``. That
version silently mis-scores every zero-scatter case as noise, and it already produced one
wrong verdict in this repo (``scripts/compare_runs.py``, same day this file was written) on
a real paired result before it was caught.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import shutil
import statistics as st
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.derive_traces import build_idf, build_sft_examples  # noqa: E402
from scripts.score_improv import (  # noqa: E402
    intensity,
    load_closure_lexicon,
    load_harm_lexicon,
    score_pair,
)
from train.improv import (  # noqa: E402
    content_words,
    extract_slots,
    parse_think,
    render_think,
    split_sentences,
)

#: Four scorers plus schema adherence = five tests. Uncorrected 0.05 would read a null
#: sitting at 2.0 standard errors as a real effect.
BONFERRONI_ALPHA = 0.01
#: Two-sided normal critical value at alpha = 0.01.
CRITICAL_T = 2.576

#: FIX 2 (code review, task-6-report.md) -- each scorer's "better" direction is NOT
#: uniform. `escalation` and `new_harm` are lower-is-better (fewer/weaker
#: escalations, fewer newly-introduced harm mentions). `groundedness` and `affordance`
#: are HIGHER-is-better (more of the continuation's content is grounded in what
#: precedes it / more of it names something the preceding context afforded). The
#: original implementation labelled `mean < 0` as "think better" unconditionally --
#: correct for the first two, INVERTED for the last two. Every scorer's direction is
#: named explicitly here so a label can never again come from a hidden assumption.
SCORER_DIRECTIONS = {
    "escalation": "lower",
    "new_harm": "lower",
    "groundedness": "higher",
    "affordance": "higher",
}

STORY_SEP = "</s>"
TRACES_PATH = ROOT / "artifacts" / "improv" / "traces.jsonl"
CORPUS_PATH = ROOT / "artifacts" / "corpus" / "tinystories.txt"
CKPT_THINK = ROOT / "artifacts" / "improv" / "ckpt-think" / "step_3000.pkl"
CKPT_NOTHINK = ROOT / "artifacts" / "improv" / "ckpt-nothink" / "step_3000.pkl"
MANIFEST_THINK = ROOT / "artifacts" / "improv" / "ckpt-think" / "train_manifest.json"
MANIFEST_NOTHINK = ROOT / "artifacts" / "improv" / "ckpt-nothink" / "train_manifest.json"
#: FIX 3(e) (task-6-report.md): the derivation drop rate is only ever written to
#: this gitignored file (artifacts/ is not committed) -- the spec requires it be
#: reported WITH the results, so main() embeds its contents into the JSON below.
DERIVE_MANIFEST_PATH = ROOT / "artifacts" / "improv" / "derive_manifest.json"
#: The dense pretraining checkpoint both arms warm-started from. Used here ONLY to recover
#: an architecture header (vocab_size, seq_len, transformer_config, ...) for HF conversion
#: -- both SFT arms are dense, identical shape, so this checkpoint's header describes them
#: exactly. Its own WEIGHTS are never used for anything in this file.
WARM_START_CKPT = (ROOT / "artifacts" / "checkpoints-v077-beta2-control"
                   / "tt_tnt_step00010764.pkl")
TOKENIZER_DIR = ROOT / "artifacts" / "hf-tt-tnt-1024-dialogue"


def paired_verdict(a: Sequence[float], b: Sequence[float], direction: str) -> Dict[str, object]:
    """Paired comparison of two equal-length score series.

    See the CONTROLLER RULING in this module's docstring for why a zero-scatter,
    non-zero-mean delta is treated as a perfect (significant) separation rather than as
    noise -- that is the one deliberate departure from the original task-6 brief.

    ``direction`` (required, no default -- see FIX 2 in task-6-report.md and
    ``SCORER_DIRECTIONS`` above) says which sign of ``mean_delta = a - b`` (think minus
    no-think) favours the think arm for THIS scorer:

      - "lower":  think favoured when mean < 0 (think's scores are lower/better).
      - "higher": think favoured when mean > 0 (think's scores are higher/better).

    There is no scorer-agnostic default on purpose: the original bug was exactly a
    silent, uniform "mean < 0 means think better" applied to all four scorers, which is
    correct for `escalation`/`new_harm` (lower is better) and backwards for
    `groundedness`/`affordance` (higher is better). Forcing every call site to state its
    scorer's direction makes that assumption visible instead of implicit.
    """
    if direction not in ("lower", "higher"):
        raise ValueError(f"direction must be 'lower' or 'higher', got {direction!r}")
    if len(a) != len(b) or not a:
        raise ValueError(f"paired series must be equal-length and non-empty: {len(a)}, {len(b)}")
    deltas = [x - y for x, y in zip(a, b)]
    mean = st.fmean(deltas)
    sd = st.pstdev(deltas)
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    zero = len(deltas) - pos - neg

    if sd == 0.0:
        # Zero scatter. Two sub-cases, per the CONTROLLER RULING above:
        if mean == 0.0:
            # Every paired point is IDENTICAL between arms. No information here at all --
            # this is the genuinely uninterpretable zero-scatter case.
            se = 0.0
            t = 0.0
        else:
            # Every paired point moved by the exact same non-zero amount. This is the
            # STRONGEST possible signal a paired comparison can produce, not the weakest
            # -- report it as infinitely significant rather than folding it into the same
            # t=0 bucket as "no information", which is what the brief's original formula
            # did (and which cannot pass test_a_clear_separation_is_reported).
            se = 0.0
            t = math.inf
    else:
        se = sd / (len(deltas) ** 0.5)
        t = abs(mean / se)

    if t <= CRITICAL_T:
        verdict = "NOT INTERPRETABLE"
    else:
        think_favoured = (mean < 0) if direction == "lower" else (mean > 0)
        verdict = "think better" if think_favoured else "no-think better"
    return {"mean_delta": round(mean, 4), "sd": round(sd, 4),
            "se": round(se, 4) if math.isfinite(se) else se,
            "t": round(t, 3) if math.isfinite(t) else t,
            "direction": direction,
            # FIX 3(a) (task-6-report.md): ties were previously folded into signs_neg,
            # which made a scorer saturated at a single constant value (e.g.
            # `groundedness` at 200/200 identical pairs) misleadingly report
            # `signs_neg: 200`, as if every pair favoured no-think, when in fact NONE of
            # them differed at all. Reporting `signs_zero` separately makes that visible.
            "signs_pos": pos, "signs_neg": neg, "signs_zero": zero,
            "n": len(deltas), "critical_t": CRITICAL_T, "verdict": verdict}


def swap_verdict(divergence_positions: Sequence[Optional[int]], n: int) -> Dict[str, object]:
    """Did substituting another story's think-block change the continuation?

    `divergence_positions[i]` is the token index where the swapped generation first differs
    from the original, or None if it never differs.
    """
    changed = [p for p in divergence_positions if p is not None]
    frac = len(changed) / max(n, 1)
    return {"n": n, "n_changed": len(changed), "fraction_changed": round(frac, 4),
            "median_first_divergence": (st.median(changed) if changed else None),
            "thinking_is_load_bearing": frac >= 0.5,
            "note": ("Below 0.5 the think-block is decorative: the model emits a plan and "
                     "writes independently of it. Stage 1 has failed in that case, and "
                     "that is the correct thing to report.")}


# ---------------------------------------------------------------------------------------
# Checkpoint -> HF conversion, entirely on the CPU.
#
# SFTTrainer (ttml.trainers.SFTTrainer, driven by scripts/train_improv.py) does NOT write
# checkpoints through train/checkpoint.py's save()/ttml.checkpointing header+manifest
# format -- scripts/convert_checkpoint.py's convert_checkpoint() (which reads that format)
# fails on a step_*.pkl from this run with "record 0 is not a ttml checkpoint". Verified by
# hand: a step_*.pkl here is a plain ``pickle.dump({"step": int, "model_state": {ttml_name:
# numpy array}})`` -- no header, no manifest.
#
# Rather than reimplementing the tensor-name mapping (kv-split, RoPE row permutation, tied
# embedding handling) a second time, this reuses the exact primitives convert/to_hf.py
# uses (`convert.hf_mapping.map_name/permute_rope_qk/split_kv/squeeze_leading` and
# `convert.to_hf.build_config`), only substituting the tensor SOURCE: a plain dict already
# in memory instead of `convert.checkpoint_reader.read_tensors(ckpt)`. The architecture
# header itself is read off the untouched dense warm-start checkpoint both arms started
# from (`WARM_START_CKPT`) -- both SFT checkpoints are dense models of that exact shape;
# only the WEIGHTS changed during SFT, never the architecture -- so that header's
# vocab_size/seq_len/transformer_config/etc. describe both step_3000.pkl files exactly.
# ---------------------------------------------------------------------------------------


def sft_checkpoint_to_hf(step_pkl: Path, *, warm_start_ckpt: Path, tokenizer_dir: Path,
                         out_dir: Path) -> Dict[str, Any]:
    """Turn an SFTTrainer ``{"step", "model_state"}`` checkpoint into a loadable HF model
    directory. No ttml/ttnn import, no device -- pure numpy/pickle/safetensors, like
    convert/to_hf.py itself. Returns the config dict written to ``out_dir/config.json``.
    """
    import numpy as np
    from safetensors.numpy import save_file
    from transformers import GenerationConfig

    from convert.checkpoint_reader import read_checkpoint_meta
    from convert.hf_mapping import map_name, permute_rope_qk, split_kv, squeeze_leading
    from convert.to_hf import build_config

    header, _manifest = read_checkpoint_meta(warm_start_ckpt)
    config = build_config(header, tokenizer_dir=tokenizer_dir)

    with step_pkl.open("rb") as fh:
        ckpt = pickle.load(fh)
    model_state: Dict[str, np.ndarray] = ckpt["model_state"]

    # The warm-start header's weights_dtype (bfloat16) describes THAT checkpoint, not
    # necessarily this one -- SFTTrainer's own save wrote plain float32 arrays here.
    # Recording the actual on-disk dtype keeps config.json honest about what it ships.
    any_tensor = next(iter(model_state.values()))
    config["torch_dtype"] = str(np.asarray(any_tensor).dtype)

    tc = header["transformer_config"]
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    weight_tying = bool(header["weight_tying"])

    out: Dict[str, np.ndarray] = {}
    unmapped: List[str] = []
    for name, tensor in model_state.items():
        target = map_name(name, weight_tying=weight_tying)
        if target is None:
            unmapped.append(name)
            continue
        if name.endswith("attention/kv_linear/weight"):
            k, v = split_kv(tensor, num_groups=int(tc["num_groups"]), head_dim=head_dim)
            out[target[0]] = permute_rope_qk(k, num_heads=int(tc["num_groups"]),
                                             head_dim=head_dim)
            out[target[1]] = v
        elif name.endswith("attention/q_linear/weight"):
            arr = squeeze_leading(tensor)
            out[target] = permute_rope_qk(arr, num_heads=config["num_attention_heads"],
                                          head_dim=head_dim)
        elif isinstance(target, tuple):
            # Tied embedding: only the embed_tokens destination is written; transformers
            # reconstructs lm_head.weight from it at load time (see convert/to_hf.py).
            out[target[0]] = squeeze_leading(tensor)
        else:
            out[target] = squeeze_leading(tensor)

    if unmapped:
        raise ValueError(
            f"{len(unmapped)} tensor(s) in {step_pkl} had no HF mapping and would have "
            f"been silently dropped: {sorted(unmapped)}. Fix hf_mapping.map_name."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_dir / "model.safetensors"))
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    GenerationConfig(bos_token_id=config["bos_token_id"], eos_token_id=config["eos_token_id"],
                      pad_token_id=config["pad_token_id"]).save_pretrained(str(out_dir))
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        src = tokenizer_dir / f
        if src.is_file():
            shutil.copy2(src, out_dir / f)
    return config


# ---------------------------------------------------------------------------------------
# Held-out openings.
# ---------------------------------------------------------------------------------------


def load_trace_story_ids(traces_path: Path) -> set:
    return {json.loads(line)["story_id"] for line in traces_path.open()}


def _iter_corpus_stories(corpus_path: Path, limit_raw: int):
    """Yield (story_id, text) exactly as scripts/derive_traces.py's main() numbers them:
    story_id is the index into the list of NON-EMPTY, stripped ``</s>``-separated records
    -- i.e. the same numbering ``traces.jsonl``'s own ``story_id`` field uses.
    """
    text = corpus_path.read_text(errors="ignore")
    story_id = -1
    for raw in text.split(STORY_SEP)[:limit_raw]:
        s = raw.strip()
        if not s:
            continue
        story_id += 1
        yield story_id, s


def select_heldout_openings(corpus_path: Path, trace_ids: set, *, n: int, seed: int,
                            harm: frozenset, idf_probe_limit: int = 400_000
                            ) -> Tuple[List[Dict[str, Any]], int]:
    """Pick ``n`` story openings whose ``story_id`` is absent from ``trace_ids``.

    Uses the SAME cut-point recipe ``derive_from_story`` uses (``random.Random(seed +
    story_id)``, ``k = randint(2, len(sents) - 2)``) so the held-out openings are drawn
    from the same distribution of cut points the training data was, and additionally
    requires ``extract_slots`` to succeed on the story's own TRUE continuation (the same
    drop rule training applied) -- so every held-out opening carries a well-formed,
    ground-truth extractive think-block available to force into context later (see
    `main()`'s swap test and paired-rates sections for why a FORCED block is needed).

    Returns ``(openings, n_scanned)`` -- ``n_scanned`` for the report, so "how many
    candidates were looked at to find n usable ones" is not silently lost.
    """
    idf = build_idf([s for _, s in _iter_corpus_stories(corpus_path, idf_probe_limit)])
    out: List[Dict[str, Any]] = []
    scanned = 0
    for story_id, story in _iter_corpus_stories(corpus_path, idf_probe_limit):
        scanned += 1
        if story_id in trace_ids:
            continue
        sents = split_sentences(story)
        if len(sents) < 4:
            continue
        rng = random.Random(seed + story_id)
        k = rng.randint(2, len(sents) - 2)
        prefix = " ".join(sents[:k])
        continuation = " ".join(sents[k:k + 2])
        slots = extract_slots(prefix, continuation, idf=idf,
                              intensity=lambda t: intensity(t, harm))
        if slots is None:
            continue
        out.append({"story_id": story_id, "prefix": prefix, "true_continuation": continuation,
                    "think_slots": slots.as_dict(), "think_block": render_think(slots)})
        if len(out) >= n:
            break
    return out, scanned


def compute_truncation_counts(traces_path: Path, tokenizer_dir: Path, pad_token_id: int,
                              max_seq_len: int = 512) -> Dict[str, Any]:
    """FIX 3(f) (task-6-report.md): how many training examples per arm exceed
    ``max_seq_len`` tokens after tile alignment -- ``sft_collate_fn`` silently truncates
    these to ``max_seq_len`` (it does not raise, skip, or warn), and that truncation
    appeared in no drop table anywhere in this plan's artifacts before this fix. Uses the
    exact same ``build_sft_examples`` construction training itself used, over the exact
    training traces, so these counts are a fact about what actually happened, not an
    estimate.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    traces = [json.loads(line) for line in traces_path.open()]
    counts: Dict[str, int] = {}
    for arm in ("think", "nothink"):
        examples = build_sft_examples(traces, tok, with_think=(arm == "think"),
                                      pad_token_id=pad_token_id)
        counts[arm] = sum(1 for e in examples if len(e["input_ids"]) > max_seq_len)
    n = len(traces)
    asymmetry_pct = round(abs(counts["think"] - counts["nothink"]) / n * 100, 4)
    return {
        "max_seq_len": max_seq_len,
        "n_examples": n,
        "think_exceeding_max_seq_len": counts["think"],
        "nothink_exceeding_max_seq_len": counts["nothink"],
        "arm_asymmetry_pct_of_n": asymmetry_pct,
        "note": (f"sft_collate_fn silently truncates any example longer than "
                f"max_seq_len={max_seq_len} to that length -- it does not raise, skip, "
                f"or log. {counts['think']} think-arm and {counts['nothink']} "
                f"nothink-arm examples (of {n}) are affected. The arm asymmetry "
                f"({asymmetry_pct}% of n) is negligible and not a source of bias between "
                f"arms on its own, but the truncation itself was previously invisible: "
                f"it appears in no drop table in this plan's artifacts."),
    }


def build_association(traces_path: Path) -> Dict[str, object]:
    """NPMI counts for `groundedness`, built from the derived traces.

    Replaces the earlier boolean co-occurrence table, which saturated: on this corpus 80.1%
    of prefix words are hub words with >2000 neighbours each, so "does any fresh word
    co-occur with any prefix word" was true almost always (mean 0.998, 99.25% exactly 1.0).
    """
    from scripts.score_improv import build_association as _build
    pairs: List[Tuple[str, str]] = []
    with traces_path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            pairs.append((rec["prefix"], rec["continuation"]))
    return _build(pairs)


# ---------------------------------------------------------------------------------------
# Generation.
# ---------------------------------------------------------------------------------------


def load_hf(hf_dir: Path):
    import torch  # noqa: F401  -- surfaced import error early and clearly if missing
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(hf_dir))
    tok.padding_side = "left"  # required for correct batched decoder-only generation
    model = AutoModelForCausalLM.from_pretrained(str(hf_dir), torch_dtype="auto").eval()
    return tok, model


def generate_batched(tok, model, prompts: List[str], *, max_new_tokens: int,
                     do_sample: bool, temperature: Optional[float] = None,
                     batch_size: int = 40, seed: Optional[int] = None) -> List[str]:
    """Greedy or T-sampled completions for every prompt, batched for CPU throughput.

    Returns only the NEW tokens' text (skip_special_tokens=True strips the pad/eos tail
    that batched generation leaves on shorter sequences).
    """
    import torch

    if seed is not None:
        torch.manual_seed(seed)
    kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        kwargs.update(temperature=temperature, top_p=1.0, top_k=0)

    texts: List[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True)
        with torch.no_grad():
            got = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                                 **kwargs)
        prompt_len = enc.input_ids.shape[1]
        for i in range(len(chunk)):
            texts.append(tok.decode(got[i][prompt_len:], skip_special_tokens=True))
    return texts


def generate_batched_from_ids(tok, model, id_lists: List[List[int]], *, max_new_tokens: int,
                              do_sample: bool, temperature: Optional[float] = None,
                              batch_size: int = 40, seed: Optional[int] = None) -> List[str]:
    """Same as `generate_batched`, but takes PRE-TOKENIZED prompt id lists rather than
    strings.

    REQUIRED whenever a forced prompt is built by concatenating two pieces that were each
    tokenized SEPARATELY -- exactly how scripts/derive_traces.py's
    `_sft_example_unaligned` builds every training example:

        p_ids = tok.encode(prompt)
        c_ids = tok.encode(completion, add_special_tokens=False)
        input_ids = p_ids + c_ids

    Tokenizing the JOINED RAW STRING instead (``tok(prefix + think_block)``) re-merges the
    boundary through the tokenizer's BPE and can silently produce a DIFFERENT token at the
    seam than training ever saw: training's boundary token before ``<think>`` is ``Ġ<``
    (id 19691, glued to the preceding space, seen 18,791 times at that exact position),
    but re-tokenizing the joined string instead produces a bare ``<`` (id 31, never seen
    there) -- an out-of-distribution seam at exactly the point being forced. Found in code
    review; see task-6-report.md's FINDING 1 addendum for the verified before/after
    numbers. Concatenating separately-encoded id lists (this function) is the only way to
    reproduce training's tokenization exactly -- a decode/re-encode round trip through text
    would reintroduce the same class of bug.
    """
    import torch

    if seed is not None:
        torch.manual_seed(seed)
    kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        kwargs.update(temperature=temperature, top_p=1.0, top_k=0)

    pad_id = tok.pad_token_id

    texts: List[str] = []
    for start in range(0, len(id_lists), batch_size):
        chunk = id_lists[start:start + batch_size]
        max_len = max(len(ids) for ids in chunk)
        input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(chunk), max_len), dtype=torch.long)
        for i, ids in enumerate(chunk):
            # Left-pad -- matches tok.padding_side="left" set in load_hf(), required for
            # correct batched decoder-only generation (every sequence's real content must
            # end at the same right-hand column so the next-token position lines up).
            input_ids[i, max_len - len(ids):] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, max_len - len(ids):] = 1
        with torch.no_grad():
            got = model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        for i in range(len(chunk)):
            texts.append(tok.decode(got[i][max_len:], skip_special_tokens=True))
    return texts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs" / "measurements" / "improv-stage1.json")
    ap.add_argument("--n-heldout", type=int, default=200)
    ap.add_argument("--n-swap", type=int, default=50)
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=5489,
                    help="Same seed as scripts/train_improv.py, for cut-point parity with "
                         "the training distribution -- these are DIFFERENT story_ids, so "
                         "this does not reproduce any training draw.")
    ap.add_argument("--adherence-max-new-tokens", type=int, default=64)
    ap.add_argument("--continuation-max-new-tokens", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--work-dir", type=Path, default=ROOT / "artifacts" / "improv" / "hf-eval")
    args = ap.parse_args()

    harm = load_harm_lexicon()
    closure = load_closure_lexicon()

    print("[1/7] converting SFT checkpoints -> HF model directories (CPU only) ...")
    hf_think_dir = args.work_dir / "think"
    hf_nothink_dir = args.work_dir / "nothink"
    cfg_think = sft_checkpoint_to_hf(CKPT_THINK, warm_start_ckpt=WARM_START_CKPT,
                                     tokenizer_dir=TOKENIZER_DIR, out_dir=hf_think_dir)
    cfg_nothink = sft_checkpoint_to_hf(CKPT_NOTHINK, warm_start_ckpt=WARM_START_CKPT,
                                       tokenizer_dir=TOKENIZER_DIR, out_dir=hf_nothink_dir)

    print("[2/7] loading both arms into transformers (CPU) ...")
    tok_think, model_think = load_hf(hf_think_dir)
    tok_nothink, model_nothink = load_hf(hf_nothink_dir)

    print("[3/7] selecting held-out openings (story_id absent from traces.jsonl) ...")
    trace_ids = load_trace_story_ids(TRACES_PATH)
    openings, n_scanned = select_heldout_openings(
        CORPUS_PATH, trace_ids, n=args.n_heldout, seed=args.seed, harm=harm)
    held_out_ids = {o["story_id"] for o in openings}
    overlap = held_out_ids & trace_ids
    if overlap:
        raise RuntimeError(f"held-out set overlaps traces.jsonl story_id(s): {sorted(overlap)}")
    print(f"  found {len(openings)}/{args.n_heldout} held-out openings "
          f"(scanned {n_scanned} corpus records; overlap with traces.jsonl: 0)")

    assoc = build_association(TRACES_PATH)

    # -------------------------------------------------------------------------------
    # HEADLINE 1: schema adherence. ORGANIC generation -- a bare prefix, nothing forced --
    # both greedy and temperature=0.8 x num_samples, THINK arm only for the headline
    # number, NOTHINK arm too as a negative control (it never saw a think-block during
    # training, so its adherence should be ~0; if it isn't, the parser or pipeline has a
    # bug worth knowing about before trusting the think arm's number at all).
    # -------------------------------------------------------------------------------
    print("[4/7] schema adherence: organic (unforced) generation from bare prefixes ...")
    prefixes = [o["prefix"] for o in openings]

    def adherence_for(tok, model) -> Dict[str, Any]:
        greedy = generate_batched(tok, model, prefixes,
                                  max_new_tokens=args.adherence_max_new_tokens,
                                  do_sample=False, batch_size=args.batch_size)
        sampled: List[str] = []
        for s in range(args.num_samples):
            sampled += generate_batched(tok, model, prefixes,
                                        max_new_tokens=args.adherence_max_new_tokens,
                                        do_sample=True, temperature=args.temperature,
                                        batch_size=args.batch_size, seed=args.seed + s + 1)
        all_gen = greedy + sampled
        parsed_ok = [parse_think(t) is not None for t in all_gen]
        return {
            "n_greedy": len(greedy), "n_sampled": len(sampled),
            "n_total": len(all_gen),
            "n_parsed": sum(parsed_ok),
            "rate": round(sum(parsed_ok) / len(all_gen), 4),
            "greedy_rate": round(sum(parse_think(t) is not None for t in greedy)
                                 / len(greedy), 4),
            "example_generations": all_gen[:3],
        }

    adherence_think = adherence_for(tok_think, model_think)
    adherence_nothink = adherence_for(tok_nothink, model_nothink)
    print(f"  think arm adherence   : {adherence_think['rate']:.1%} "
          f"({adherence_think['n_parsed']}/{adherence_think['n_total']})")
    print(f"  nothink arm adherence : {adherence_nothink['rate']:.1%} "
          f"({adherence_nothink['n_parsed']}/{adherence_nothink['n_total']}) "
          f"[negative control -- never trained on a think-block]")

    # -------------------------------------------------------------------------------
    # HEADLINE 2 setup: paired failure-mode rates. Organic adherence above already shows
    # the think arm essentially never chooses to emit a think-block on its own (teacher-
    # forced per-token loss looks fine because template tokens dominate the average; the
    # ENTRY decision and the slot CONTENT are the tokens that are actually still poorly
    # predicted -- see task-6-report.md's per-position loss breakdown for the verbatim
    # training example that pins this down). So the paired rates and the swap test below
    # FORCE a think-block into the think arm's context (the story's own ground-truth,
    # extractive block -- the same construction that built its training data) rather than
    # waiting for one to be generated. This is a deliberate, separate measurement from
    # adherence: "does the CONTENT of a think-block change what follows, once one is
    # present" is a different question from "does the model choose to produce one", and
    # the brief's own instruction to report the two headlines separately is exactly why
    # this split is the right way to keep both questions honestly answered rather than
    # letting a near-zero adherence rate silently starve the rates/swap measurements of
    # data.
    # -------------------------------------------------------------------------------
    print("[5/7] paired failure-mode rates: forced think-block vs. no-think baseline ...")
    # Tokenize prefix and think-block SEPARATELY and concatenate ids -- exactly how
    # scripts/derive_traces.py's _sft_example_unaligned builds training examples
    # (tok.encode(prompt) + tok.encode(completion, add_special_tokens=False)). Joining the
    # raw STRINGS first and tokenizing the concatenation (the original version of this
    # line) puts a different, unseen token at the seam -- see generate_batched_from_ids's
    # docstring and task-6-report.md's FINDING 1 addendum.
    think_prompt_ids = [tok_think.encode(o["prefix"])
                        + tok_think.encode(o["think_block"], add_special_tokens=False)
                        for o in openings]
    nothink_prompts = list(prefixes)

    think_continuations = generate_batched_from_ids(
        tok_think, model_think, think_prompt_ids,
        max_new_tokens=args.continuation_max_new_tokens, do_sample=False,
        batch_size=args.batch_size)
    nothink_continuations = generate_batched(
        tok_nothink, model_nothink, nothink_prompts,
        max_new_tokens=args.continuation_max_new_tokens, do_sample=False,
        batch_size=args.batch_size)

    scores_think = [score_pair(o["prefix"], c, harm=harm, assoc=assoc, closure=closure)
                    for o, c in zip(openings, think_continuations)]
    scores_nothink = [score_pair(o["prefix"], c, harm=harm, assoc=assoc, closure=closure)
                      for o, c in zip(openings, nothink_continuations)]

    scorer_series = {
        "escalation": ([s.escalation for s in scores_think],
                      [s.escalation for s in scores_nothink]),
        "new_harm": ([float(s.new_harm) for s in scores_think],
                    [float(s.new_harm) for s in scores_nothink]),
        "groundedness": ([s.groundedness for s in scores_think],
                        [s.groundedness for s in scores_nothink]),
        "affordance": ([float(s.affordance) for s in scores_think],
                      [float(s.affordance) for s in scores_nothink]),
    }
    rates = {name: paired_verdict(a, b, SCORER_DIRECTIONS[name])
             for name, (a, b) in scorer_series.items()}
    n_favouring_think = sum(1 for v in rates.values()
                            if v["verdict"] == "think better")
    for name, v in rates.items():
        print(f"  {name:14} mean_delta={v['mean_delta']:+.4f} t={v['t']} "
              f"verdict={v['verdict']}")

    # FIX 3(b) (task-6-report.md): groundedness is measured against this run's actual
    # scores (not hardcoded from a prior run) -- how saturated is it, really, on THIS
    # co-occurrence table and THESE continuations? Computed over both arms pooled (400
    # scores) since saturation is a property of the metric/table, not of either arm.
    _all_groundedness = ([s.groundedness for s in scores_think]
                         + [s.groundedness for s in scores_nothink])
    groundedness_saturation = {
        "mean": round(st.fmean(_all_groundedness), 4),
        "pct_exactly_1_0": round(
            sum(1 for v in _all_groundedness if v == 1.0) / len(_all_groundedness) * 100, 2),
        "n": len(_all_groundedness),
    }

    # Auxiliary (not part of the 5-test Bonferroni family, reported for context only).
    aux = {
        "novelty": {"think_mean": round(st.fmean(s.novelty for s in scores_think), 3),
                   "nothink_mean": round(st.fmean(s.novelty for s in scores_nothink), 3)},
        "new_proper_nouns": {
            "think_mean": round(st.fmean(s.new_proper_nouns for s in scores_think), 3),
            "nothink_mean": round(st.fmean(s.new_proper_nouns for s in scores_nothink), 3)},
    }

    # -------------------------------------------------------------------------------
    # THE SWAP TEST. Reported FIRST in the output JSON's narrative position even though
    # it's computed here, after the model objects are already loaded -- see module
    # docstring for why it is the control that can fail the whole stage on its own.
    #
    # For each of n_swap held-out openings, generate the continuation twice from the THINK
    # arm, holding the prefix fixed and varying only WHICH story's think-block precedes it:
    #   own      : prefix_i + think_block_i  (the story's own ground-truth block)
    #   swapped  : prefix_i + think_block_j  (another story's block, j = i's neighbour)
    # If the continuation is token-identical in both conditions, the think-block's content
    # was not used -- decorative. Both conditions are greedy (deterministic), so any
    # difference is attributable to the swapped content, not sampling noise.
    # -------------------------------------------------------------------------------
    print("[6/7] the swap test (runs first in the narrative -- computed here since the "
          "model is already loaded) ...")
    n_swap = min(args.n_swap, len(openings))
    swap_subset = openings[:n_swap]
    # Same training-style id concatenation as the rates section above -- both conditions
    # carry the SAME boundary construction, so this fix mostly changes the absolute
    # divergence-token positions, not the own-vs-swapped comparison's fairness (it was
    # already internally fair: both sides shared the identical string-joined seam before
    # this fix too). Re-run anyway now that the seam matches training.
    own_prompt_ids = [tok_think.encode(o["prefix"])
                      + tok_think.encode(o["think_block"], add_special_tokens=False)
                      for o in swap_subset]
    swapped_prompt_ids = [tok_think.encode(o["prefix"])
                          + tok_think.encode(swap_subset[(i + 1) % n_swap]["think_block"],
                                            add_special_tokens=False)
                          for i, o in enumerate(swap_subset)]

    own_continuations = generate_batched_from_ids(
        tok_think, model_think, own_prompt_ids,
        max_new_tokens=args.continuation_max_new_tokens, do_sample=False,
        batch_size=args.batch_size)
    swapped_continuations = generate_batched_from_ids(
        tok_think, model_think, swapped_prompt_ids,
        max_new_tokens=args.continuation_max_new_tokens, do_sample=False,
        batch_size=args.batch_size)

    divergence_positions: List[Optional[int]] = []
    for own_text, swapped_text in zip(own_continuations, swapped_continuations):
        own_toks = tok_think.encode(own_text, add_special_tokens=False)
        swap_toks = tok_think.encode(swapped_text, add_special_tokens=False)
        pos = None
        for i, (a, b) in enumerate(zip(own_toks, swap_toks)):
            if a != b:
                pos = i
                break
        else:
            if len(own_toks) != len(swap_toks):
                pos = min(len(own_toks), len(swap_toks))
        divergence_positions.append(pos)

    swap = swap_verdict(divergence_positions, n_swap)
    # FIX 3(c) (task-6-report.md): the forced-block qualifier used to live ONLY in
    # generation_settings, far from where a reader of the swap test itself would look.
    # It is the single most important caveat on this specific number (100% "changed" is
    # a FORCED-context result, not evidence the model uses a think-block it chose to
    # produce itself), so it is attached directly to the swap_test block now.
    swap["forced_think_note"] = (
        "Both the 'own' and 'swapped' conditions FORCE the story's own (or another "
        "story's) extractive ground-truth think-block into the think arm's context "
        "before generating -- this measures whether think-block CONTENT changes "
        "generation once one is present, not whether the model organically chooses to "
        "produce one (see the top-level 'adherence' section for that, separately)."
    )
    print(f"  swap test: {swap['n_changed']}/{swap['n']} continuations changed "
          f"({swap['fraction_changed']:.1%}); "
          f"thinking_is_load_bearing={swap['thinking_is_load_bearing']}")

    # -------------------------------------------------------------------------------
    # FIX 3(e)/(f) (task-6-report.md): drop rate and truncation counts, computed/loaded
    # here so they can be embedded in the report rather than left to live only in a
    # gitignored artifact (derive_manifest.json) or nowhere at all (truncation).
    # -------------------------------------------------------------------------------
    derive_manifest = (json.loads(DERIVE_MANIFEST_PATH.read_text())
                       if DERIVE_MANIFEST_PATH.is_file() else None)
    if derive_manifest is not None and "corpus" in derive_manifest:
        # Same repo-relative-path treatment as _manifest_for_report below, and for the
        # same reason (FIX 3(g)): an absolute worktree path stops existing once the
        # worktree is removed.
        _corpus_path = Path(derive_manifest["corpus"])
        if _corpus_path.is_absolute():
            try:
                derive_manifest["corpus"] = str(_corpus_path.relative_to(ROOT))
            except ValueError:
                pass
    truncation = compute_truncation_counts(TRACES_PATH, TOKENIZER_DIR,
                                           pad_token_id=tok_think.pad_token_id or 0)
    print(f"  derivation drop rate: "
          f"{derive_manifest['drop_rate']:.2%} ({derive_manifest['drops_by_rule']})"
          if derive_manifest else "  derivation drop rate: derive_manifest.json not found")
    print(f"  truncation (>{truncation['max_seq_len']} tok): "
          f"think={truncation['think_exceeding_max_seq_len']} "
          f"nothink={truncation['nothink_exceeding_max_seq_len']}")

    # -------------------------------------------------------------------------------
    # Assemble the report.
    # -------------------------------------------------------------------------------
    print("[7/7] writing report ...")
    success = {
        "adherence_at_least_0_80": adherence_think["rate"] >= 0.80,
        "at_least_2_of_4_scorers_favour_think_at_alpha_0_01": n_favouring_think >= 2,
        "swap_test_shows_continuations_do_change": bool(swap["thinking_is_load_bearing"]),
    }
    success["all_criteria_met"] = all(success.values())

    # Persisted as a top-level `verdict` field (not just printed) so the artifact carries
    # the named field the plan's Produces line calls for -- the substance already lived in
    # `success_criteria`, but the label itself must be in the file too.
    verdict_line = ("DECORATIVE" if not swap["thinking_is_load_bearing"]
                    else ("STAGE 1 SUCCESS" if success["all_criteria_met"]
                          else "PARTIAL -- see success_criteria"))

    # FIX 3(d)/(g) (task-6-report.md): each manifest gets an explicit loss_note (the
    # two arms' losses are NOT comparable -- the think arm's completion supervises the
    # think-block template as well as the continuation, the nothink arm's supervises
    # only the continuation, so a side-by-side loss_end comparison is not a "which arm
    # trained better" signal), and `traces` is rewritten repo-relative -- the raw
    # manifest embeds an absolute path into THIS worktree, which stops existing the
    # moment the worktree is removed.
    LOSS_NOTE = (
        "The two arms' losses are NOT directly comparable: the think arm's completion "
        "is `think_block + continuation` (supervises the think-block template AND the "
        "slot content, in addition to the continuation), while the nothink arm's "
        "completion is the continuation alone. A lower/decreasing think-arm loss does "
        "not mean 'the think arm learned the continuation better than the nothink arm' "
        "-- the two arms are not scored against the same target distribution, so this "
        "loss_end is not a legitimate comparison point between arms on its own."
    )

    def _manifest_for_report(path: Path) -> Dict[str, Any]:
        m = dict(json.loads(path.read_text()))
        traces_path = Path(m["traces"])
        if traces_path.is_absolute():
            try:
                m["traces"] = str(traces_path.relative_to(ROOT))
            except ValueError:
                pass  # not under ROOT; leave as-is rather than guess
        # The warm-start checkpoint path is NESTED, and an earlier pass rewrote only the
        # top-level `traces` key — so this one kept leaking an absolute worktree path into a
        # committed artifact, which stops resolving once the worktree is removed. Rewrite it
        # the same way rather than assuming one relative-path fix covers the whole manifest.
        ws = m.get("warm_start_summary")
        if isinstance(ws, dict) and ws.get("checkpoint"):
            ck = Path(ws["checkpoint"])
            if ck.is_absolute():
                try:
                    ws["checkpoint"] = str(ck.relative_to(ROOT))
                except ValueError:
                    pass  # not under ROOT; leave rather than guess
        m["loss_note"] = LOSS_NOTE
        return m

    report = {
        "verdict": verdict_line,
        # FIX 3(b) (task-6-report.md): plain statement of what this measurement can and
        # cannot support, next to the numbers rather than scattered across comments only
        # a source-reader would find.
        "limitations": {
            "swap_test_and_rates_are_forced_not_organic": (
                "The swap test and the paired failure-mode rates FORCE the think-block "
                "into the prompt; they do not measure whether the model spontaneously "
                "emits one. See 'adherence' for the organic measurement, and "
                "swap_test.forced_think_note for the same caveat attached directly to "
                "that number."
            ),
            "single_training_run_per_arm": (
                "Each arm is a SINGLE SFT run (one seed, one training trajectory). The "
                "paired t-tests in 'rates' capture item variance over the 200 held-out "
                "examples ONLY -- they say nothing about run-to-run variance (a second "
                "training run of the same arm/seed/recipe could land on different "
                "scorer means for reasons unrelated to thinking vs. not-thinking). "
                "Treat 'rates' as a single paired sample, not as evidence the effect (or "
                "null) would replicate across retraining."
            ),
            "groundedness_is_saturated_and_cannot_discriminate": {
                **groundedness_saturation,
                "explanation": (
                    "groundedness is saturated on the production co-occurrence table: "
                    f"mean {groundedness_saturation['mean']}, "
                    f"{groundedness_saturation['pct_exactly_1_0']}% of the "
                    f"{groundedness_saturation['n']} pooled think+nothink scores are "
                    "exactly 1.0. A metric with almost no variance left to explain is "
                    "structurally unable to discriminate between the two arms "
                    "regardless of any real underlying difference. So 'rates' reporting "
                    "0/4 scorers favouring think is really 0 of 3 LIVE scorers "
                    "(escalation, new_harm, affordance) plus one metric "
                    "(groundedness) that could not have returned a different answer "
                    "either way -- redesigning groundedness is out of scope for this "
                    "evaluation pass and is tracked as a follow-up, not attempted here."
                ),
            },
        },
        "swap_test": swap,
        "swap_test_detail": {
            "n": n_swap,
            "divergence_positions": divergence_positions,
            "note": "position is a TOKEN index into the generated continuation; None means "
                    "the swapped continuation was identical to the own-think-block one for "
                    "the full generation window.",
        },
        "adherence": {"think": adherence_think, "nothink_negative_control": adherence_nothink},
        "rates": rates,
        "rates_auxiliary_not_in_bonferroni_family": aux,
        "bonferroni": {"alpha": BONFERRONI_ALPHA, "critical_t": CRITICAL_T,
                      "n_tests": 5, "n_scorers_favouring_think": n_favouring_think},
        "success_criteria": success,
        "held_out": {
            "n_requested": args.n_heldout, "n_found": len(openings),
            "n_corpus_records_scanned": n_scanned,
            "story_ids": sorted(held_out_ids),
            "overlap_with_traces_jsonl_story_id": sorted(held_out_ids & trace_ids),
            "verification": "held_out story_id set intersected against every story_id in "
                            "traces.jsonl; overlap is asserted empty before this file is "
                            "written (main() raises otherwise).",
        },
        "manifests_used": {
            "think": _manifest_for_report(MANIFEST_THINK),
            "nothink": _manifest_for_report(MANIFEST_NOTHINK),
        },
        # FIX 3(e) (task-6-report.md): embedded verbatim from the gitignored
        # derive_manifest.json so the drop rate is reported WITH the results, per spec,
        # rather than surviving only in an artifact this repo never commits.
        "derivation": (derive_manifest if derive_manifest is not None else {
            "error": f"{DERIVE_MANIFEST_PATH} not found -- drop rate could not be "
                     "embedded. Regenerate via scripts/derive_traces.py.",
        }),
        # FIX 3(f) (task-6-report.md): sft_collate_fn silently truncates examples longer
        # than max_seq_len; this previously appeared in no drop table anywhere.
        "truncation": truncation,
        "hf_conversion": {"think_config": cfg_think, "nothink_config": cfg_nothink,
                          "warm_start_header_source": str(
                              WARM_START_CKPT.relative_to(ROOT))},
        "generation_settings": {
            "adherence_max_new_tokens": args.adherence_max_new_tokens,
            "continuation_max_new_tokens": args.continuation_max_new_tokens,
            "num_samples": args.num_samples, "temperature": args.temperature,
            "seed": args.seed,
            "forced_think_note": "See swap_test.forced_think_note and "
                                 "limitations.swap_test_and_rates_are_forced_not_organic "
                                 "-- rates and the swap test FORCE the story's own "
                                 "extractive ground-truth think-block into the think "
                                 "arm's context rather than waiting for organic "
                                 "emission; see the comment above [5/7] in "
                                 "scripts/eval_improv.py for why.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")

    print(f"\nVERDICT: {verdict_line}")
    print(f"  organic think-arm adherence : {adherence_think['rate']:.1%}")
    print(f"  swap test load-bearing      : {swap['thinking_is_load_bearing']} "
          f"({swap['fraction_changed']:.1%} changed)")
    print(f"  scorers favouring think     : {n_favouring_think}/4 at alpha={BONFERRONI_ALPHA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
