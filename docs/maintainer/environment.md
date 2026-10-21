# Maintainer options and environment variables

Under [deterministic decisions](deterministic-decisions.md), no environment variable may change a
decision: everything that steers a route, a tuner, the decode pacing or a debug aid is an explicit
option, and the server records the options it ran with (`server_start` in the request log). The
variables that remain only write diagnostics, inject faults for tests, or locate files and keys;
none of them changes what the engine computes.

## Options

`ninferno` and `ninferno-serve` take all of these. `ninferno-perplexity` takes all but
`--decode-steps`, `--decode-wait` and `--decode-route`, which it refuses: it scores and runs no
decode rounds.
`ninferno spec-tune` takes `--decode-wait`, `--cuda-sync`, `--route-table`, `--route-dir` and
`--drafter-registry`, and records them in the profile's settings (and an `--out`'s, which
`--resume` compares) by content, never by a local path: a route table or a drafter registry as
the SHA-256 of its bytes, a route directory as the SHA-256 over the files the engine reads from it
(every `*.json` table in it and `tuner/dense_tune.json`, symlinks followed), sorted by relative
path, each as the line `<relative path>\0<hex SHA-256 of its bytes>\n`, so `--resume` refuses cells
measured with other tables. The family tools
`ninferno-gemma4` and `ninferno-dense` take `--cuda-sync`, `--route-table`, `--route-dir`,
`--dense-tune`, `--tune-cache`, `--debug-poison` and `--debug-poison-skip` (`ninferno-gemma4` also
`--drafter-registry`), check them before anything is converted or loaded, and apply them to the
process as an Engine does: the dense tuner is on for the loaded model whatever `--dense-tune` says,
so `off` reads the release's `routes/tuner/dense_tune.json` as `ninferno` and `ninferno-serve` do.
They refuse `--moe-tune`: the MoE schedule tuner runs in an Engine's model load (every family's),
which neither tool goes through. `ninferno-gemma4 spec-tune` names in its profile's settings the
`--cuda-sync`, `--dense-tune`, `--route-table` and `--route-dir` it ran with, when not the
release's. `EngineOptions::pacing` and `EngineOptions::maintainer` (`include/ninferno/types.h`)
carry them for the Engine API.

The maintainer options and `--cuda-sync` are process-wide. An Engine built while no other Engine
of the process is alive applies its own values afresh: the route tables return to the default
lookup in `--route-dir` (an earlier explicit table is dropped) before its own `--route-table`
loads, the tuners forget what an earlier engine tuned or read, and the debug aids take its values.
An Engine built while others are alive must ask for the same values; otherwise it is refused,
since it would change their decisions.

The [decision fingerprint](deterministic-decisions.md#fingerprint) hashes all of them with the
other options (`options.pacing.*`, `options.maintainer.*`; an option naming a file or directory
enters as `set`, the files themselves by content among the sources), and a maintainer option
given is taint (`taint.maintainer`, by name). `--tune-cache` enters the fingerprint and is taint only
while a tuner is on (it is read only then), and a `--drafter-registry` file in place of the compiled-in registry is taint too
(`drafter_registry`). The route pins enter as `options.maintainer.prefill_route`,
`options.maintainer.decode_route` and `options.maintainer.moe_path` by their spelling, and each one
not `auto` is taint by name. The request log's `server_start` record carries them
(`engine.decode_pacing`, `engine.maintainer` with `drafter_registry` and the three pins;
`--decode-steps` as `auto` or its integer).

### Decode pacing

Documented with [decode round pacing](../cli.md#decode-round-pacing) and
[CUDA synchronization](../cli.md#cuda-synchronization). They change when tokens are published and
how the host waits, never which tokens.

| Option | Values | Default |
|---|---|---|
| `--decode-steps` | `auto`, or `1..32` | `auto` |
| `--decode-wait` | `auto`, `spin`, `block` | `auto` (`spin` under `ninferno spec-tune`, which takes `--decode-wait` too) |
| `--cuda-sync` | `spin`, `blocking`, `yield`, `auto` | `spin` |

### Route tables and lab tuners

These are lab tools. A user's engine times nothing and reads no state an earlier run left: routes
come from the release's `routes/tuner/dense_tune.json`, then the GPU class's route table, else the
heuristic.

| Option | Values | Effect |
|---|---|---|
| `--route-table FILE` | file path | This route table alone, instead of the GPU class's own. A table that cannot be read or a malformed one is refused before the device is touched; one for another GPU class fails startup when it loads. |
| `--route-dir DIR` | directory | The directory of per-GPU-class route tables (`DIR/*.json`) and of the release's dense-tuner measurements (`DIR/tuner/dense_tune.json`). A directory that cannot be listed, or one of those files that cannot be resolved (with the file system's reason), is a dangling symlink or cannot be read, is refused before the device is touched. Default: `routes/` beside the binary, then `share/ninferno/routes/` above it, then the source tree the binary was built from (only a build with `NINFERNO_SOURCE_TREE_PATHS=ON`, the default; a release has none). |
| `--dense-tune` | `off` (default), `on`, `retune` | The dense GEMM tuner. `on` times the candidate routes of a problem the table does not cover the first time the engine computes it outside graph capture (the warm-up's eager prefill) and stores the choice per GPU class and problem; `on` and `retune` also read the cache file (`retune` ignores its choices, then merges its new ones into it). Route tables are built offline ([route tuning helpers](../../bench/README.md#route-tuning-helpers)). |
| `--moe-tune` | `off` (default), `on`, `retune` | The MoE schedule tuner an Engine runs at model load (the family tools, which load no model through an Engine, refuse it), with the same meaning per MoE layer shape; `tools/moe_tune_to_table.py` turns its cache into a route table's MoE bands. |
| `--tune-cache DIR` | directory | Where the tuners keep `dense_tune.json` and `moe_tune.json`, read only while a tuner is on. Default: NInferno's cache directory, `$XDG_CACHE_HOME/ninferno`, else `~/.cache/ninferno` (`%LOCALAPPDATA%\NInferno` on Windows). |

### Route pins

Test options for A/B runs that check the automatic selection: each pins one kernel family, never a
tuner, and times nothing. `auto` everywhere is the release's selection, byte for byte
(`ninferno_selector_golden_test` runs with no pin; `ninferno_route_pin_selector_test` checks that
clearing a pin restores every route).

| Option | Values | Band | Families |
|---|---|---|---|
| `--prefill-route` | `auto`, `a16`, `w4a8`, `i8`, `mmq`, `w4a4` | calls of 17 tokens or more | `a16`: what the automatic selection takes for an A16Only weight (register MMA, tiled or staged tiles, the route tables' A16 entries); `w4a8`: the W4A8 heuristic's int8 tile (with its tuned prefill band); `i8`: the Q4/Q5/Q6 G64 int8 tiles; `mmq`: ggml's int8 matrix kernel in 512-token slices; `w4a4`: NVFP4 activations on NVFP4 weights |
| `--decode-route` | `auto`, `a16`, `r8`, `vec`, `mmq` | calls of 1..16 tokens | `a16`: the automatic selection for an A16Only weight; `r8`: the int8 R8 kernel at 1 token (where the activation fits a CTA) and 2..4 tokens (the pre-quantized form); `vec`: ggml's dp4a vector kernel, 1..8 tokens (16 without tensor mma.sync); `mmq`: ggml's matrix kernel, 5..16 tokens |
| `--moe-path` | `auto`, `fused`, `generic` | each MoE layer at load | `fused`: Qwen 3.5/3.6 SparseMoe (top-8 experts it reads) or Gemma 4's NVFP4 MoE (calibrated NVFP4 experts, sm_120a); `generic`: the generic MoE Op, which reads every expert format |

A pinned family takes every call of its band that it can compute, and a call it cannot compute
keeps the automatic route. A pin never changes the precision the user chose: the int8 families
need a policy that allows int8 activations (the artifact's, or `--activations int8`), `w4a4` one
that allows four bits. The one exception is the automatic promotion of the official Q4/Q5/Q6 G64
weights recorded A16Only to int8 at prefill on the cards where it pays: `--prefill-route w4a8`
or `i8` promotes them on every card (that is the decision under test), `a16` on none.

Where the pins reach:

- the generic projection path (`ops/generic`): every format and shape a registered shape table does
  not own, all ggml block formats among them; a pin stands before the route tables' and the dense
  tuner's choices;
- the registered shapes' hand-off gates: R8 decode (`--decode-route r8` keeps the rule's 1-token
  hand-off; any other decode pin keeps the call on the shape's own BF16 kernel), the int8 prefill
  tiers of Q8 G32 and Q4/Q5/Q6 G64, and the NVFP4 W4A8 prefill (`a16` keeps the plan's A16 route);
- the NVFP4 and FP8 shape tables of `linear`: `a16` keeps BF16 activations at the pinned widths,
  and `--prefill-route w4a4` takes A4 from 17 tokens;
- the MoE path choice of the Qwen 3.5/3.6 and Gemma 4 families.

Where they do not: the fused NVFP4 plans of the registered Qwen shapes (attention and GDN input
projections, `linear_add`, `linear_swiglu`) keep their own A4 or A16 choice; the routed experts'
grouped GEMMs keep their schedules (`--moe-tune` and the route tables' MoE bands); a GPU without
tensor mma.sync (Volta) keeps its tiled route for calls of 17 tokens or more of a format with a
tiled codec; the invariant routes of `--activations exact` take no pin at all; TMA and other
schedule-level choices inside a shape table stay the table's.

Refusals, each naming what can run instead:

1. Before the device: a prefill or decode pin with `--activations exact`; `--prefill-route w4a4`
   with `--activations int8`.
2. Once the device is bound, before any weight is read: a family this build and GPU hold no kernel
   of (`w4a4` outside the sm_120a builds, `mmq` below sm_75, a family whose device features are
   missing), naming the families that run here.
3. At load: `--moe-path fused` on a MoE layer its fused kernels cannot run (top-k, format, the
   activation bits), naming the reason.
4. At the end of startup (after the kernel warm-up and the served Program's decode graphs): a pin
   whose band had calls none of which it could take, naming the families those calls could take;
   `--moe-path` on a model with no MoE layer that chooses. A band no startup call reached (no
   warm-up because every size was given explicitly, and no decode graphs) is reported as not
   checked.

The startup log prints one `routes | pinned |` line per pin with the families its band met. The
serve bench tools take the three options (`tools/bench/run_serve_corpus.py`,
`run_serve_concurrency.py`), pass them to `ninferno-serve` and check them in its `server_start`
record.

### Speculation data

| Option | Takes it | Effect |
|---|---|---|
| `--drafter-registry FILE` | `ninferno`, `ninferno-serve`, `ninferno-convert`, `ninferno spec-tune`, `ninferno spec-reselect`, `ninferno-gemma4` | A drafter registry JSON to use instead of the compiled-in one. |
| `--corpus FILE` | `ninferno spec-tune`, `ninferno-gemma4` | The prompt corpus to measure on, instead of the compiled-in one. |

The token-frequency rankings a drafter conversion names are found in `tools/spec/rankings` of the
working tree, then among the rankings installed beside the binary, then in the tree it was built
from; `ninferno-convert --ranking FILE` names the target's own.

### Debug aids

| Option | Values | Effect |
|---|---|---|
| `--debug-prefill-cuts P1,P2,...` | absolute positions | Qwen 3.5: ends a prefill call at each of these positions, the way a restored or continued context meets the cache off the chunk grid. |
| `--debug-poison` | `0` (default), `1`/`nan`, a byte `1..255` | Overwrites pooled device memory with that byte whenever it is released (`1` and `nan` mean `0xFF`: NaN in every float format, -1 in indices), so a reader that skips initialization sees poison rather than the previous owner's data. |
| `--debug-poison-skip SITE,...` | `kv`, `ingress`, `state`, `weights` | Leaves these sites unpoisoned, for bisecting which poisoned memory changed a result. |

### Removed variables

These variables used to steer a decision. No program reads them to decide anything any more; every
product binary (`ninferno` with `spec-tune` and `spec-reselect`, `ninferno-serve`,
`ninferno-perplexity`, `ninferno-gemma4`, `ninferno-dense`, `ninferno-convert`) warns once at
startup, after its arguments parse (`--help` and `--version` stay quiet), for each one still set,
naming its replacement (`src/core/removed_environment.h`, the only place in `src/` and `apps/`
that may quote these names; `ninferno_decision_fingerprint_test` fails otherwise).

| Variable | Replacement |
|---|---|
| `NINFERNO_CUDA_SYNC` | `--cuda-sync` |
| `NINFERNO_DECODE_STEPS` | `--decode-steps` |
| `NINFERNO_DECODE_WAIT` | `--decode-wait` |
| `NINFERNO_ROUTE_TABLE` | `--route-table` |
| `NINFERNO_ROUTE_DIR` | `--route-dir` |
| `NINFERNO_DENSE_TUNE`, `NINFERNO_MOE_TUNE` | `--dense-tune`, `--moe-tune` |
| `NINFERNO_DENSE_TUNE_CACHE`, `NINFERNO_MOE_TUNE_CACHE` | `--tune-cache` (one directory for both tuners) |
| `NINFERNO_QWEN_PREFILL_CUTS` | `--debug-prefill-cuts` |
| `NINFERNO_DEBUG_POISON`, `NINFERNO_DEBUG_POISON_SKIP` | `--debug-poison`, `--debug-poison-skip` |
| `NINFERNO_DRAFTER_REGISTRY` | `--drafter-registry` |
| `NINFERNO_SPEC_CORPUS` | `--corpus` (`ninferno spec-tune`, `ninferno-gemma4`); elsewhere the compiled-in corpus |
| `NINFERNO_RANKING_DIR` | none: rankings are found in `tools/spec/rankings`, then beside the binary, then in the source tree; `ninferno-convert --ranking FILE` names one |
| `NINFERNO_MMQ_VERIFY` | none (an A/B knob of the IQ formats' verify rule) |

### Outside the engine

Two variables steer choices outside the engine and its products, and stay variables:

| Variable | Read by | Effect |
|---|---|---|
| `NINFERNO_FORCE_WSL` | the Windows launcher and installer only (`tools/release/windows`) | `1` installs and runs the WSL engine even where a native Windows engine exists ([Windows](../windows.md)). The engine it starts reads none of it. |
| `NINFERNO_BENCH_PREFILL_ROUTES` | the op benches only (`bench/ops`) | `1` runs the benched op under the prefill route profile for the rest of the process. |

## Environment variables

None of these changes a decision, and none enters the fingerprint. Each is read once and costs
nothing when unset. A new variable read in `src/` or `apps/` fails `ninferno_decision_fingerprint_test`
until it is listed in `src/runtime/engine/decision_fingerprint.cpp`, which lists only variables
that steer nothing: a choice belongs in an option.

The user-facing ones are documented where they belong: `NINFERNO_API_KEY` in
[Serving](../serving.md), `NINFERNO_UI_KEY`, `NINFERNO_CATALOG_URL`, `NINFERNO_UPDATER`,
`NINFERNO_UPDATE_URL` and `NINFERNO_UPDATE_CHECK` with the desktop UI and the release tools, and
`NINFERNO_PDFIUM` (where the PDF library is) with document input. Test programs read
`NINFERNO_TEST_*`, `NINFERNO_DET_*` and `NINFERNO_PROBE_*` and map them to the options above.

### Loading

| Variable | Values | Effect |
|---|---|---|
| `NINFERNO_ARTIFACT_DIRECT_IO` | `1`/`true`/`yes`/`on`, `0`/`false`/`no`/`off` | Forces `O_DIRECT` or buffered reads of artifact payloads. Unset or any other value: chosen automatically. The bytes read are the same either way. |

### Numerics investigation and route recording

These write tensors or records to disk; the dumps synchronize the device after each step and are
never for timing. Graph-captured passes are skipped.

| Variable | Values | Effect |
|---|---|---|
| `NINFERNO_QWEN_DUMP` | directory | Qwen3.5 prefill: appends each layer's intermediates to `<dir>/L<layer>.<kind>.bin`. |
| `NINFERNO_QWEN_DUMP_ROWS` | `lo:hi` | Only the token columns whose absolute position is in `[lo, hi)`. |
| `NINFERNO_QWEN_DUMP_KINDS` | `a,b,...` | Only these intermediate kinds. Default: all. |
| `NINFERNO_GEMMA4_DUMP` | directory | Gemma 4: appends each layer's intermediates to `<dir>/L<layer>.<kind>.bin`. |
| `NINFERNO_GEMMA4_DUMP_ROWS` | `lo:hi` | Only these absolute positions. Default: all. |
| `NINFERNO_DENSE_DUMP` | directory | Dense decoders, batch-1 eager calls: writes `q`, `attn`, `ao`, `mlp` and `x` per layer, the embedding, the final-norm output and the last column's logits to `<dir>/c<call>_p<position>_t<tokens>.L<layer>.<kind>.bin`. |
| `NINFERNO_DENSE_DUMP_ROWS` | `lo:hi` | Only these absolute positions. Default: all. |
| `NINFERNO_SCORE_LOGITS` | file path | `ninferno-perplexity` on Qwen3.5: appends each scored column's logits over the public vocabulary as FP32, for comparing distributions with a reference. |
| `NINFERNO_SCORE_ARGMAX` | file path | The same, the argmax id per scored column, for top-1 agreement. |
| `NINFERNO_ROUTE_RECORD` | file path | Appends every distinct problem that reaches the generic GEMM path, in `ninferno_route_tune --shapes` format, so a model's shapes come from running it (see [bench/README.md](../../bench/README.md)). |

### Fault injection

For the tests of failure handling. Creating the named file while the server runs triggers the
fault.

| Variable | Values | Effect |
|---|---|---|
| `NINFERNO_DEBUG_FREEZE_FILE` | file path | While the file exists, the executor blocks as on a hung device wait. |
| `NINFERNO_DEBUG_TRAP_FILE` | file path | Once the file appears, the next check launches a kernel that traps, which poisons the CUDA context as a failed device-side assert does. Fires once. |
| `NINFERNO_DEBUG_CUDA_ERROR_FILE` | file path | When the file exists, the next check removes it and raises a recoverable CUDA error (`cudaErrorMemoryAllocation`). Each new file is one error. |
| `NINFERNO_DEBUG_NAN_FILE` | file path | See the [serving reference](../reference/serving.md). |
