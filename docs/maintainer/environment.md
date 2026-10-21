# Maintainer options and environment variables

Under [deterministic decisions](deterministic-decisions.md), no environment variable may change a
decision: everything that steers a route, a tuner, the decode pacing or a debug aid is an explicit
option, and the server records the options it ran with (`server_start` in the request log). The
variables that remain only write diagnostics, inject faults for tests, or locate files and keys;
none of them changes what the engine computes.

## Options

`ninferno` and `ninferno-serve` take all of these; `EngineOptions::pacing` and
`EngineOptions::maintainer` (`include/ninferno/types.h`) carry them for the Engine API. The route
table, the tuners and the debug aids are process-wide: the engine applies them when it is
constructed.

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
| `--route-table FILE` | file path | This route table alone, instead of the GPU class's own. A table for another GPU class, or a malformed one, fails startup. |
| `--route-dir DIR` | directory | The directory of per-GPU-class route tables. Default: `routes/` beside the binary, then `share/ninferno/routes/` above it, then the source tree the binary was built from (only a build with `NINFERNO_SOURCE_TREE_PATHS=ON`, the default; a release has none). |
| `--dense-tune` | `off` (default), `on`, `retune` | The dense GEMM tuner. `on` times the candidate routes of a problem the table does not cover the first time the engine computes it outside graph capture (the warm-up's eager prefill) and stores the choice per GPU class and problem; `on` and `retune` also read the cache file (`retune` ignores its choices, then merges its new ones into it). Route tables are built offline ([route tuning helpers](../../bench/README.md#route-tuning-helpers)). |
| `--moe-tune` | `off` (default), `on`, `retune` | The MoE schedule tuner, with the same meaning per MoE layer shape; `tools/moe_tune_to_table.py` turns its cache into a route table's MoE bands. |
| `--tune-cache DIR` | directory | Where the tuners keep `dense_tune.json` and `moe_tune.json`, read only while a tuner is on. Default: NInferno's cache directory, `$XDG_CACHE_HOME/ninferno`, else `~/.cache/ninferno` (`%LOCALAPPDATA%\NInferno` on Windows). |

### Speculation data

| Option | Takes it | Effect |
|---|---|---|
| `--drafter-registry FILE` | `ninferno`, `ninferno-serve`, `ninferno-convert` | A drafter registry JSON to use instead of the compiled-in one. |
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

Removed with no replacement: `NINFERNO_MMQ_VERIFY` (an A/B knob of the IQ formats' verify rule).

## Environment variables

None of these changes a decision. Each is read once and costs nothing when unset.

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
