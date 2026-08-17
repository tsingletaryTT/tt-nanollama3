<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Serving tt-tnt with tt-kernel and the Tenstorrent vLLM plugin

The headline of this project is that the model is packaged with
[tt-kernel](https://github.com/tenstorrent/tt-kernel-package-manager) and served through the
Tenstorrent vLLM plugin. This file is the procedure: what the bundle is, the commands that
actually brought a server up, and the traps that cost hours to find.

It is a **how-to**. [`tt-kernel-conformance.md`](tt-kernel-conformance.md) is the neighbouring
*findings report* — what a v4 manifest can express, which fields are read and which are
silently ignored — and it is the right file for "is tt-kernel's schema doing what it says".
This one assumes you want a server. Where the two overlap, this file states the operational
consequence and links there for the analysis.

Everything below was run on the box this model was developed on: a TT-QuietBox 2, four
Blackhole chips on two dual-chip p300c cards, serving on **one** of them. Paths under
`/home/ttuser/` are that box's; the rule each one illustrates is not.

> **Read [§7](#7-known-limitation-on-device-generation-is-degenerate) before you serve this
> for generation.** The packaging path works and the server answers. The *output* on device
> is worse than the same weights produce on CPU, and the cause is not known.

---

## 1. What the bundle is

Two artifacts, and nothing else:

| Path | What it is |
|---|---|
| [`manifests/tt_kernel_manifest-384.json`](../manifests/tt_kernel_manifest-384.json) | The v4 manifest — what tt-kernel renders into a launch command |
| [`bundle/tt_tnt_adapter.py`](../bundle/tt_tnt_adapter.py) | The `main_class` the plugin imports, carrying two runtime patches and one precision default |

`tt-kernel push` renders the manifest into a `vllm_metadata.json` and lays it beside the
adapter in the bundle folder. That rendered file is what `tt-kernel serve` actually reads:

```json
{
  "arch": "LlamaForCausalLM",
  "main_class": "tt_tnt_adapter:LlamaForCausalLM",
  "hf_weights": "episod/tt-tnt",
  "launch": {
    "default": {
      "command": ["python3", "server_example_tt.py", "--model", "episod/tt-tnt",
                  "--max_model_len", "2048", "--max_num_seqs", "8"],
      "env": {"VLLM_USE_V1": "1", "MESH_DEVICE": "P150", "HF_MODEL": "episod/tt-tnt"}
    }
  }
}
```

### 1.1 The manifest fields that matter

Of the 13 fields our manifest sets, four decide whether the server comes up and what it
serves.

**`entrypoint.class` — `"tt_tnt_adapter:LlamaForCausalLM"`**

`module:Class`, resolved *inside the bundle folder*. This is the field that makes the whole
"a model carries its own runtime change" property work: the plugin imports our module out of
`EXTRA_MODELS_DIR`, our module patches tt-metal at import time, and the patches are in place
before any `ModelArgs` is constructed. Confirmed in the serving log as
`platform.py:499 Registered TT model TTLlamaForCausalLM -> tt_tnt_adapter:LlamaForCausalLM
(from EXTRA_MODELS_DIR/episod__tt-tnt)`. `entrypoint.arch_name` (`LlamaForCausalLM`) is the
architecture the class registers under.

**`mesh` — `{"devices": 1, "topology": "1x1"}`**

`mesh.devices` reaches `device_count` in the published bundle's metadata, so search and any
consumer reasoning about topology see it. It does **not** reach the launch command: the mesh
the plugin actually opens comes only from `env.MESH_DEVICE`. `mesh.topology` has no consumer
at all. Those are two independent channels describing the same thing, and nothing
cross-checks them — see conformance findings [1 and 3](tt-kernel-conformance.md#findings).
Keep them consistent by hand; `tests/test_manifests.py` is our own gate on it.

**`resources.max_model_len` — `2048`**

Composed into the launch command as `--max_model_len` and reported on the wire at
`/v1/models`. Two constraints, both learned the hard way:

- It must be a **power of two**. `capped_warmup_seq_len`
  (`model_config.py:1150-1152`) requires it; a non-power-of-two hard-fails at startup with an
  unrelated-looking message.
- It must match **what the published weights were actually trained to**, which is not
  necessarily what the size registry currently targets. Declaring 512 against a 2048-context
  model does not error — it silently discards three quarters of the trained context.
  `tests/test_manifests.py::_PUBLISHED_CONTEXT` is the gate, and it distinguishes a published
  fact from a placeholder.

`resources.max_num_seqs` (8) is the concurrent-sequence cap and is composed the same way.

**`env` — `{"VLLM_USE_V1", "MESH_DEVICE", "HF_MODEL"}`**

- `MESH_DEVICE: "P150"` is a **lookup key for a mesh grid shape, not a board type**, and the
  name is a red herring on a box with no P150 in it. The plugin's
  `_MESH_GRID_PRESETS` (`vllm_tt_plugin/utils/dp_discovery.py`) maps `P150 -> (1, 1)`; nothing
  on the vLLM serving path attaches board semantics to the string. `(1, 1)` is the *only*
  shape this model can serve at: `num_key_value_heads: 3`, and `model_config.py:687-691`
  asserts both `n_heads` (6) and `n_kv_heads` (3) divide the mesh width, which admits widths
  {1, 3} — and no preset offers 3.
- `HF_MODEL` is **required and undocumented**. tt-kernel composes `--model <repo>` from
  `weights.repo`, but `tt_transformers` reads the model id from the `HF_MODEL` environment
  variable and raises `ValueError: Please set HF_MODEL to a HuggingFace name ...` if it is
  unset. Every v4 vLLM bundle needs it duplicated into `manifest.env`, and nothing validates
  that you did (conformance finding 2).

The remaining fields are compatibility and catalog metadata: `platform.ttnn` (`>=0.65,<0.80`)
is the real compatibility declaration and is what instance selection resolves against;
`runtime.kind: "vllm"`; `target: "p150"` is the machine key the launch entry is selected by;
`weights.repo`; `name` and `description`.

### 1.2 The two runtime patches the model depends on

`bundle/tt_tnt_adapter.py` adds **no model code** — tt-tnt is a standard HF Llama and the
stock `models.tt_transformers.tt.generator_vllm:LlamaForCausalLM` computes it. What the
adapter carries is what tt-metal gets wrong or defaults badly for a model this small. Both
patches describe themselves in-file as *shims, not fixes*, each with the upstream change that
would make it a deletable no-op.

**Patch 1 — `ModelArgs.find_grid`, for a harvested grid. Without it the model does not run
at all.** Upstream picks a core grid from hardcoded per-architecture constants
(`max_cols = 8 if is_wormhole_b0() else 12`) and never asks the device how large its compute
grid actually is. A harvested Blackhole has fewer usable columns than the architectural
maximum — the p300c this was developed on reports **11×10, not 12×10** — so for
`hidden_size=384` (12 tiles) `find_grid` returns a 12-column program config and RMSNorm fails
at the first decoder layer with `TT_FATAL shard_spec_validation.cpp:34 ... program_config grid
(12x1) must be contained within device grid (11x10)`. Given the *real* width the same
unmodified search finds `rows=2, cols=6`, which fits. The patch changes only where
`max_rows`/`max_cols` come from, falls back to the original implementation whenever the device
cannot be queried, and is idempotent and reversible (`restore_patches()`).

*To delete it:* upstream `find_grid` should read `compute_with_storage_grid_size()` rather
than hardcoding per-arch constants. Once that is in a released tt-metal, `_patch_find_grid`
becomes a no-op — remove it and raise the `platform.ttnn` floor in the manifest to the first
release that carries the fix, since the floor is what currently pins this bundle below it.

**Patch 2 — `ModelArgs.weight_cache_path`, fingerprinting the converted-weight cache.
Without it a republished model can be served as the previous one.** `tt_transformers`
converts HF weights to `.tensorbin` once and reuses them forever, keyed on the HF repo id and
nothing else (`model_config.py:577`), and the reuse decision (`ttnn/ttnn/operations/core.py:719`)
is a bare *existence* check — no revision, no content hash, no comparison against the source.
The patch appends one directory component derived from the source weights:

```
model_cache/episod/tt-tnt/P150/tensor_cache_bfp8/src-rev-a3c85ec799fe/...
```

The fingerprint is the HF commit sha (`hf_config._commit_hash`) where there is one; a sha256
over `(name, size, mtime_ns)` of `config.json` and every weight file for a local checkpoint
directory; and otherwise nothing, in which case the path is left exactly as stock **and a
warning says the guard is not in place**. `TT_TNT_CACHE_FINGERPRINT=0` turns it off (also
loudly). See [§4.6](#46-a-converted-weight-cache-keyed-on-the-repo-id-serves-the-previous-model)
for the incident this exists to prevent and what its log lines mean.

*To delete it:* the cache key at `model_config.py:577` should include the resolved revision.
Once that lands, `_patch_weight_cache_path` becomes a no-op and can go.

**And one default, not a patch.** The adapter subclasses the stock class solely to default
`optimizations` to `"accuracy"` rather than `"performance"`. The stock default serves MLP
`w1`/`w3` at BFLOAT4_B, which is tuned for 8B–70B models; this one is 384-wide and 22M
parameters. Measured over six prompts × 40 greedy tokens against the CPU reference, it is
worth about one token of agreement (median 4/40 against 3/40) — real, reproducible, and
**not** a remedy for §7. Overridable via `TT_TNT_OPTIMIZATIONS`.

The adapter also narrows the stock class's blanket-`True` capability flags to all-`False`
(`supports_prefix_caching`, `supports_async_decode`, `supports_sample_on_device`), because
none of them has been proven for this model. The plugin logs that async scheduling was
requested and **disabled**; that is the flags working, not a fault. Re-enable individually,
with evidence, via `TT_TNT_CAPS`.

---

## 2. The environment that can serve it

`tt-kernel serve` launches the bundle under a registered **tt-metal instance**. On this box:

```bash
tt-kernel instances list
```

```
[active] active: ttnn=0.65.1rc17.dev6200 vllm=— plugin=—  (/home/ttuser/.tenstorrent-venv/bin/python3)
[registry] metal-0.75-vllm: ttnn=0.75.0 vllm=0.24.0+empty plugin=0.1.0  (/home/ttuser/venv-vllm-latestmetal/bin/python3)
[registry] metal-src-vllm: ttnn=0.65.1rc17.dev6200 vllm=0.24.0+empty plugin=0.1.0  (/home/ttuser/venv-vllm-standalone/bin/python3)
```

**`metal-src-vllm` is the one that works** — a venv against the tt-metal *source tree* at
`/home/ttuser/tt-metal`. `metal-0.75-vllm` looks newer and better and cannot serve at all;
[§4.3](#43-the-0750-wheel-cannot-serve-on-this-box-sfpi-7610-against-a-pinned-7670) is why.

One registration detail worth knowing before you add your own: `all_instances()` dedupes by
`(os.path.realpath(python), tt_metal_home)`, and `realpath()` of a venv's `bin/python3`
resolves to the **base interpreter** — under `uv`, every venv on this box resolves to the same
`cpython-3.12.12` binary. Two venvs sharing a base interpreter *and* a `TT_METAL_HOME`
therefore collapse to one entry and the loser is dropped silently, with `instances add`
reporting success and `instances list` then not showing it. `metal-src-vllm` is registered
with its `--tt-metal-home` pointing at a **symlink** to `/home/ttuser/tt-metal` so that its key
differs from `active`'s. If that symlink ever disappears the instance breaks; a durable
symlink in a stable location is the better home for it.

---

## 3. Pull and serve

One command, from the right directory (see [§4.1](#41-the-launch-command-is-cwd-dependent)):

```bash
cd /home/ttuser/vllm-tt-plugin-standalone/examples
tt-kernel serve episod/tt-tnt --force --instance metal-src-vllm
```

`serve` pulls the bundle folder if it is absent, points `EXTRA_MODELS_DIR` at it, and launches
the OpenAI-compatible server with the bundle's launch command. **Repeat invocations skip the
pull** — which is a trap, not a convenience; see [§4.5](#45-tt-kernel-serve-prefers-the-cached-bundle-to-the-hub).

`--force` is needed on **every** invocation, not just the first. `serve` correctly selects
`metal-src-vllm` and prints its versions, and then the compatibility gate reads the *active*
environment rather than the selected instance and refuses to install, citing a ttnn and a
missing plugin that are not what would run. Both warnings below are that gate misreading the
environment; they are expected here and are not about the instance being launched:

```
! tt-metal: version 0.65.1rc17.dev6200 is older than required 0.72.0 — upgrade
! vllm: vllm present but the TT plugin (vllm_tt_plugin) is not importable — ...
```

Nothing about that is good — it trains the operator to ignore the flag that exists to surface
real problems — but it is where things stand.

**Before you launch, print what will launch.** `--print` composes the exact env and command
and returns without opening a device or a server:

```bash
tt-kernel serve episod/tt-tnt --print --local-only --instance metal-src-vllm
```

```
[vLLM: LlamaForCausalLM via default; instance=metal-src-vllm; EXTRA_MODELS_DIR=/home/ttuser/.cache/tt-kernel/bundles]
  OpenAI endpoint (once up): http://localhost:8000
TT_METAL_HOME=<tt-metal source tree> PYTHONPATH=<same> LD_LIBRARY_PATH=<same>/build/lib \
EXTRA_MODELS_DIR=/home/ttuser/.cache/tt-kernel/bundles VLLM_USE_V1=1 MESH_DEVICE=P150 \
HF_MODEL=episod/tt-tnt /home/ttuser/venv-vllm-standalone/bin/python3 server_example_tt.py \
  --model episod/tt-tnt --max_model_len 2048 --max_num_seqs 8
```

Read `--max_model_len` in that output before every serve you intend to measure. It is the
cheapest check in this document and it catches [§4.5](#45-tt-kernel-serve-prefers-the-cached-bundle-to-the-hub)
outright.

---

## 4. The traps

Every one of these was found by hitting it. None of them announces itself.

### 4.1 The launch command is cwd-dependent

The rendered launch command names `server_example_tt.py` by **bare relative path**, so
`tt-kernel serve` only works from a directory that happens to contain the plugin's example
script. Run it from the model repo and it dies with

```
can't open file '/home/ttuser/code/tt-tnt/server_example_tt.py'
```

— an error naming *your* cwd, with nothing pointing at the bundle. Nothing documents the
requirement and nothing checks it. Serve from the plugin's `examples/` directory.

There is a second consequence that is easier to miss. `tt_transformers` builds its cache path
as a **relative** `model_cache/<repo id>/<MESH_DEVICE>/...`, so the converted-weight cache
lands under whatever directory you launched from. On this box that is
`/home/ttuser/vllm-tt-plugin-standalone/examples/model_cache/episod/tt-tnt/P150/`. Serving the
same model from two different directories gives you two different caches, and the one you
inspect may not be the one that was used.

### 4.2 `TT_VISIBLE_DEVICES` must not be exported

For single-device serving, leave `TT_VISIBLE_DEVICES` **unset**. The lease tooling actively
encourages exporting it with every chip you hold; doing that breaks this serve in two
different ways, and neither error names the variable:

```
TT_VISIBLE_DEVICES=<4 chips>  ->  mesh grid (1, 4)  ->  AssertionError: n_heads must be
                                                        divisible by num_devices: 6 % 4
TT_VISIBLE_DEVICES=<1 chip>   ->  TT_FATAL tt_cluster.cpp:281
                                  is_custom_fabric_mesh_graph_desc_path_specified()
```

The mechanism is `_resolve_mesh_grid` (`vllm_tt_plugin/worker.py:110-134`): `MESH_DEVICE=P150`
resolves to `(1, 1)`, but **a non-empty `TT_VISIBLE_DEVICES` overrides it** to
`(1, visible_count)`. Four visible chips therefore demand a 4-way tensor split of 6 heads and
3 KV groups. Exposing exactly one chip avoids that and breaks auto-discovery instead, because
a p300c board is two linked chips and splitting one off needs a custom mesh-graph descriptor.

With the variable unset, a p300c auto-discovers as `(1, 1)` — confirmed in the log as
`worker.py:840 Attempting to open mesh device with grid shape (1, 1)` — while UMD still opens
all four chips at cluster level (`Opening local chip ids/PCIe ids: {0, 1, 2, 3}`). If you are
holding a lease, that is still inside it; but note that it is a cluster-level open, so do not
run this alongside another tenant on the same box.

### 4.3 The 0.75.0 wheel cannot serve on this box: sfpi 7.61.0 against a pinned 7.67.0

`venv-vllm-latestmetal` carries ttnn **0.75.0** and looks like the obvious environment to
serve from. It cannot JIT its own dispatch kernels here. The host's
`/opt/tenstorrent/sfpi` is **7.61.0** (its own `README.md` says so) while that wheel's
`ttnn/tt_metal/sfpi-version` requires **7.67.0**, so every kernel compile fails with

```
'clamp' is not a member of 'sfpi'      (likewise 'min', 'max')
```

and the engine dies at `TT_THROW ... program.cpp:232`. Nothing in that message mentions sfpi
versions.

Serve from the **tt-metal source tree** instead. `/home/ttuser/tt-metal` ships its own
matching `runtime/sfpi` — 7.66.0 per `runtime/sfpi-version.cmake` — and compiles cleanly. That
is what the `metal-src-vllm` instance points at, and it is the environment every successful
serve recorded here used.

This is not a tt-tnt problem and its real fix is elsewhere: upgrade `/opt/tenstorrent/sfpi` to
7.67.0. Until then the newest wheel is unusable on this host.

### 4.4 `tt-kernel push` can change repository visibility as a side effect

On tt-kernel `main`, `--private/--public` is a plain boolean defaulting to `False`, and
`hub.set_visibility(repo_id, private=private)` runs on **every** push — not only when the repo
is created (`cli.py:275-277` on the dispatch path, `cli.py:497-499` on the vLLM path). So a
push with no visibility argument does not mean "leave it alone"; it **asserts public**. A push
with `--private` against a public repo asserts private. There is no confirmation and no diff.

This is fixed in [PR #12](https://github.com/tenstorrent/tt-kernel-package-manager/pull/12),
which makes the flag tri-state so an unset value leaves an existing repo's visibility alone.
**Until that lands, be explicit and verify on both sides.** Both of this project's Hub repos —
[`episod/tt-tnt`](https://huggingface.co/episod/tt-tnt) and
[`episod/tt-tnt-corpus`](https://huggingface.co/datasets/episod/tt-tnt-corpus) — are public and
must stay public, so every push here passes `--public`:

```bash
tt-kernel push episod/tt-tnt --public --backend vllm \
  --manifest manifests/tt_kernel_manifest-384.json \
  --bundle-dir <clean staging copy of bundle/> \
  --tt-metal-version 0.65.1 --arch blackhole
```

`python scripts/publish_to_hub.py --verify` checks the published repo's visibility against
`EXPECTED_PRIVATE = False` among its other round-trip checks; run it before and after.

Two smaller notes on `push` in the same area. `--tt-metal-version` should name the ttnn the
bundle was genuinely packaged and served against (`0.65.1` here), not the newest one
available; the manifest's `platform.ttnn` range is the real compatibility declaration.
And push from a **clean staging copy** of `bundle/` — the bundle-dir index on tt-kernel `main`
has no exclusion list, so an early push shipped
`vllm_bundle/__pycache__/tt_tnt_adapter.cpython-312.pyc`: stale bytecode published as part of a
model bundle. `bundle/__pycache__/` exists in this repo right now, so this is not hypothetical.
An exclusion list exists on an unmerged tt-kernel branch; until it lands, stage by hand.

### 4.5 `tt-kernel serve` prefers the cached bundle to the Hub

`serve` documents that "repeat invocations skip the pull and go straight to launch". The
consequence is not documented: **a manifest fix pushed to the Hub does not reach a machine that
already has the bundle.** `_ensure_vllm_pulled` returns the recorded local entry whenever its
`bundle_path` is a directory, before any Hub lookup, and `--force` does not change that —
`--force` only relaxes compatibility gating *if* a pull happens. Filed as
[issue #13](https://github.com/tenstorrent/tt-kernel-package-manager/issues/13).

It has already nearly produced a wrong measurement here. After the model was retrained and
republished at a 2048 context, `~/.cache/tt-kernel/bundles/episod__tt-tnt/vllm_metadata.json`
still held the **512** launch command from the previous push, so the server would have launched
at `--max_model_len 512` while the Hub said 2048.

**Detect it:**

```bash
tt-kernel serve episod/tt-tnt --print --local-only --instance metal-src-vllm   # read --max_model_len
cat ~/.cache/tt-kernel/bundles/episod__tt-tnt/vllm_metadata.json               # compare to the manifest you pushed
md5sum ~/.cache/tt-kernel/bundles/episod__tt-tnt/tt_tnt_adapter.py <repo>/bundle/tt_tnt_adapter.py
```

That last line matters independently: the **adapter** in the cached bundle is the one that
runs, so a local copy can be newer or older than the Hub's. On this box the two md5s match.
The Hub bundle, as of the last recorded check, still carries the **pre-fingerprint** adapter:
the cache fix was verified here but distributing it needs another `push`, which has not been
done. If you pull this bundle fresh on another machine, check the adapter you got against this
repo's `bundle/tt_tnt_adapter.py` before trusting §4.6's guarantees.

**Avoid it:** `tt-kernel rm episod/tt-tnt` removes the installed bundle folder and its index
entry, so the next `serve` pulls again. (On the runs recorded here the folder was *moved
aside* rather than removed, which is the same effect and keeps the old copy for comparison —
worth preferring when you are in the middle of a measurement.)

### 4.6 A converted-weight cache keyed on the repo id serves the previous model

This is the one that nearly produced a confident false measurement, and it is the reason the
adapter carries patch 2.

The first serve after a retrain came up **clean**: the adapter registered, the mesh opened at
`(1, 1)`, `/v1/models` reported the correct `max_model_len: 2048` — and it was running the
**previous model's weights**. `tt_transformers` had converted those weights once, cached them
at `model_cache/episod/tt-tnt/P150/tensor_cache_bfp8/`, and the reuse decision is a bare
existence check on a path keyed by repo id. The republished weights hit the same path and the
cache won. The log line reads

```
Loaded cache for model_cache/episod/tt-tnt/P150/...
```

which is indistinguishable from an ordinary warm start. Nothing said "these are a different
revision's weights". It was caught only because a human noticed the cache directory's mtime
predated the publish. Had it not been, the headline finding of that session would have been a
confident **0% EOS termination** — because the previous model could not emit an end-of-document
token by construction — and it would have read as a real regression.

**It does not fail. It lies.**

#### What the adapter's log lines mean

With patch 2 installed, four lines carry the whole story. Two are the reassuring half and are
emitted at INFO — which is why the adapter's logger is named `vllm.<module>`: vLLM configures
only the `vllm` logger and leaves root at `WARNING`, so a bare `getLogger(__name__)` had every
INFO record dropped and only the warnings ever reached the terminal.

| Line | Level | What it means |
|---|---|---|
| `scoped the tt_transformers weight cache by source fingerprint` | INFO | The guard is installed. Its **absence** is the thing to notice. |
| `reusing the converted-weight cache for episod/tt-tnt (rev a3c85ec799fe) at .../src-rev-a3c85ec799fe` | INFO | Legitimate warm start — same source revision as last time. |
| `the source weights for episod/tt-tnt changed -- no converted-weight cache for rev <sha12>. Converting fresh weights. Previously cached revisions ... are NOT being used` | WARNING | A republish was detected; these weights are being converted now. |
| `... holds un-fingerprinted .tensorbin files from before this guard was installed. They are no longer read (that is the fix) and nothing here deletes them` | WARNING | Pre-guard leftovers, now dead weight. Deleting them is your call. |

Two further warnings say the guard is *not* protecting you: one when no fingerprint could be
derived (falls back to the stock path), one when `TT_TNT_CACHE_FINGERPRINT=0`. Both are worth
treating as a stop condition if you are about to record a number.

#### What a stale cache looks like on disk

Without the guard: one flat `tensor_cache_bfp8/` full of `.tensorbin` files whose mtimes
predate the publish, and no way to tell which weights produced them. With it, `ls` answers the
question that cost an afternoon:

```
tensor_cache_bfp8/
  *.tensorbin                 59 files   <- pre-guard leftovers, no longer read
  src-rev-a3c85ec799fe/       59 files + tt_tnt_cache_source.json
  src-rev-745d708fca12/       59 files + tt_tnt_cache_source.json
  src-rev-9583e97d0f94/       59 files + tt_tnt_cache_source.json
```

```json
{
  "source": "episod/tt-tnt",
  "fingerprint_kind": "rev",
  "fingerprint": "745d708fca12",
  "written_by": "tt_tnt_adapter",
  "written_at": "2026-08-15T02:23:06.400997+00:00"
}
```

The fingerprint is the **repo revision, not the weight content**, so any commit to the repo —
a card fix, a bundle push — mints a new revision and costs a full re-conversion even when
`model.safetensors` is untouched. That is the deliberate trade: false misses cost minutes,
false hits cost a wrong published measurement. At 22M parameters a re-conversion is under a
second; on a 70B model, fixing a README would not be. Old revision directories are retained,
not reclaimed — delete them by hand.

---

## 5. Verify it actually works

### 5.1 The server answers

```bash
curl -s localhost:8000/v1/models
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "episod/tt-tnt",
  "prompt": "Once upon a time, there was a little",
  "max_tokens": 32, "temperature": 0
}'
```

`/v1/models` must report the `max_model_len` your manifest declares — that is the manifest
live on the wire, and it is the check that the bundle you served is the bundle you pushed.

### 5.2 The log lines worth reading

Four, in order of appearance:

- `platform.py:499 Registered TT model TTLlamaForCausalLM -> tt_tnt_adapter:LlamaForCausalLM
  (from EXTRA_MODELS_DIR/episod__tt-tnt)` — the bundle's adapter, not a built-in, is in play.
- `worker.py:840 Attempting to open mesh device with grid shape (1, 1)` — [§4.2](#42-tt_visible_devices-must-not-be-exported)
  did not bite.
- the adapter's cache lines — [§4.6](#46-a-converted-weight-cache-keyed-on-the-repo-id-serves-the-previous-model).
- `tt-tnt: using optimizations='accuracy'` and MLP weights appearing as `BFLOAT8_B` in the
  tensor-cache names (`feed_forward.w1_sharded_dtype_BFLOAT8_B`) — the precision default took
  effect. For contrast the KV cache allocates at `BFLOAT16`
  (`empty_kcache_paged_attention(2056, 3, 64, 64)_t0_dtype_BFLOAT16_layout_TILE`), which is the
  highest precision the stack offers.

You should also see async scheduling requested and **disabled**, and prefix caching disabled.
That is the adapter's capability flags doing their job.

### 5.3 That you are serving the weights you think you are

Three independent checks, cheapest first:

1. **The fingerprint in the log names a revision.** Compare that `sha12` against the Hub
   repo's current commit sha. If the log says `reusing` and the sha is not the one you just
   published, you are serving the previous model.
2. **The cache directory on disk.** `ls model_cache/<repo id>/<MESH_DEVICE>/tensor_cache_bfp8/`
   should hold a `src-rev-<sha12>/` for the revision you expect, with a
   `tt_tnt_cache_source.json` saying so. Remember the cache is relative to the cwd you served
   from ([§4.1](#41-the-launch-command-is-cwd-dependent)).
3. **A CPU cross-check that refuses to run on a mismatch.**

   ```bash
   python scripts/free_running_check.py --tokens 40
   ```

   It defaults its CPU reference to `artifacts/hf-tt-tnt-v3` — the artifact
   `scripts/publish_to_hub.py` uploads — and `_check_reference_matches_server()` compares that
   reference's `max_position_embeddings` against the served model's `max_model_len` from
   `/v1/models`, **exiting 3** when they disagree, however `--hf-dir` was chosen. This is what
   it printed the day it was first pointed at the wrong reference, when the served context was
   still 512:

   ```
   ERROR: CPU reference artifacts/384/hf has max_position_embeddings=256 but the server
   serves 'episod/tt-tnt' at max_model_len=512. These are different models, so a
   token-by-token comparison measures the difference between them, not the decode path.
   ```

   It is coarse — it cannot tell two models of equal context apart — but it catches the
   failure that actually happened, and it caught it again on the very next publish.
   `--allow-reference-mismatch` keeps deliberate cross-model comparison possible.

Checks 1 and 2 are the ones that answer "which weights", and they only exist because of
patch 2. On stock tt-metal there is no answer to that question in the server's output at all.

---

## 6. Republishing: the order that works

Two of the traps above only fire on a republish, so the sequence matters. This is the one that
was run:

```bash
python scripts/publish_to_hub.py --dry-run     # pre-flight guard, incl. the context length
python scripts/publish_to_hub.py --yes         # weights + card
tt-kernel push episod/tt-tnt --public --backend vllm \
  --manifest manifests/tt_kernel_manifest-384.json \
  --bundle-dir <clean staging copy of bundle/> \
  --tt-metal-version 0.65.1 --arch blackhole
python scripts/publish_to_hub.py --restore-card --yes
python scripts/publish_to_hub.py --verify
```

`--restore-card` is not optional and it is not a nicety. `tt-kernel`'s `tag_repo`
(`hub.py:56-66`) replaces the model card's front matter wholesale with
`ModelCardData(tags=...)` on **every** push, discarding `license`, `library_name`,
`pipeline_tag`, `datasets` and `base_model`. The prose body survives; the metadata does not.
Observed both times: all four were `None` after the push and all four were back after the
restore. [`docs/model-card.md`](model-card.md) is the source of truth that gets re-applied.

Then, before serving the new revision: clear or move aside the cached bundle
([§4.5](#45-tt-kernel-serve-prefers-the-cached-bundle-to-the-hub)) and expect the adapter to
announce a fresh conversion ([§4.6](#46-a-converted-weight-cache-keyed-on-the-repo-id-serves-the-previous-model)).
If the first serve after a republish says `reusing`, stop and find out why.

---

## 7. Known limitation: on-device generation is degenerate

**The packaging path works. The generation does not.** Anyone serving this model for
generation should know this before they start, and nothing above should be read as "the model
serves correctly" — only that the *packaging* path works.

The same weights, served through the plugin on a Blackhole chip, produce worse output than
they produce on CPU **at identical settings**. Greedy decoding collapses into repetition loops
where CPU produces coherent prose. Sampling (t = 0.8, top-p 0.95) removes the hard loops and
shows what is underneath: agrammatical run-ons with no clause boundaries, and malformed
non-words — *"Invisers"*, *"o'Splains"*, *"megathering"*, *"d'Bule"* — that appear nowhere in
the CPU output. Over the same 29 completions per side, the local-repeat rate is **0.161 on
device against 0.104 on CPU (1.55×)**; the metrics understate it, because they do not capture
grammaticality, which is where the gap is widest. The full output is recorded in
[`docs/measurements/samples-tt-tnt-v3-ondevice-t0.8.md`](measurements/samples-tt-tnt-v3-ondevice-t0.8.md).
The recorded verdict is *not usable*.

Two more angles on the same defect:

- **Free-running greedy agreement with CPU**, six prompts × 40 tokens: median **4/40**
  (min 0, max 5).
- **EOS termination** on the frozen 15-prompt set: **2/15 on device against 5/15 on CPU**
  greedy, 7/30 against 11/30 sampled. A path that degenerates into repetition is a path that
  does not reach a natural ending.

**Prefill is sound.** Across both prompt sets, 20 of 21 prompts agree with CPU on the *first*
generated token — the one that comes purely out of prefill with no decode step. That localises
the fault to the decode / KV-cache path.

**Nine hypotheses have been refuted** and the cause is not known: the weights, the conversion,
the `find_grid` patch, MLP quantization, prefix caching, async decode, on-device sampling, the
KV-cache dtype (it is already `BFLOAT16`, the highest the stack offers), and the possibility
that the whole thing was an artifact of greedy decoding. The defect has also survived a change
of tokenizer, corpus, corpus revision, context length, weights and tt-metal build. The surface
still worth attacking is the paged-attention decode control path — page-table / slot-mapping
and position handling per decode step.

One thing to carry away if you are gating a serving path of your own: the `tt_transformers`
PCC check passed at **0.9940–0.9998** throughout, because it exercises prefill far harder than
long decode. **A green PCC is not evidence of correct generation.**

---

## 8. Where the rest of it is written down

- [`docs/tt-kernel-conformance.md`](tt-kernel-conformance.md) — what a v4 manifest can express,
  which fields are read, and what our bundle found. The analysis behind §1.
- [`bundle/tt_tnt_adapter.py`](../bundle/tt_tnt_adapter.py) — both patches, each with a section
  on scope, safety, why this shape and not another, and what deletes it.
- [`docs/upstream-tt-metal-asks.md`](upstream-tt-metal-asks.md) — the tt-metal/tt-train asks
  this project cannot fix in its own tree, each with a reproduction and a note on whether it
  blocks anything.
- [`docs/model-card.md`](model-card.md) — the source of truth for the Hub card, and the
  model's own limitations.
- `tests/test_manifests.py` — the gates on the manifest: context length against the published
  weights, the power-of-two requirement, and the `HF_MODEL` rule.
