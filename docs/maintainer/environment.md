# Environment variables for development

The server and the CLI take their configuration from command-line options. The variables below
are for working on NInferno itself: tests, numerics investigations, kernel tuning and fault
injection. Each one is read once, costs nothing when unset, and none of them is needed to serve a
model.

The user-facing variables are documented with the options they belong to:
`NINFERNO_API_KEY` in [Serving](../serving.md), `NINFERNO_CUDA_SYNC`, `NINFERNO_DECODE_STEPS` and
`NINFERNO_DECODE_WAIT` in the [CLI reference](../cli.md), and `NINFERNO_DEBUG_NAN_FILE` in the
[serving reference](../reference/serving.md).

## Tuning caches and route tables

| Variable | Values | Effect |
|---|---|---|
| `NINFERNO_DENSE_TUNE` | `off` (default), `on`, `retune` | The dense GEMM tuner. Off, the engine starts with no timing and reads no cache an earlier run wrote: routes come from the release's `routes/tuner/dense_tune.json`, then the GPU class's route table, else the heuristic. `on` times the candidate routes of a problem the table does not cover the first time the engine computes it outside graph capture (the warm-up's eager prefill) and stores the choice per GPU class and problem; `on` also reads the cache file first; `retune` ignores the stored choices; any other value is refused. Route tables are built offline ([route tuning helpers](../../bench/README.md#route-tuning-helpers)). |
| `NINFERNO_DENSE_TUNE_CACHE` | file path | Where the dense tuner's choices live. Default `dense_tune.json` in NInferno's cache directory: `$XDG_CACHE_HOME/ninferno`, else `~/.cache/ninferno` (`%LOCALAPPDATA%\NInferno` on Windows). |
| `NINFERNO_MOE_TUNE` | `off` (default), `on`, `retune` | The MoE schedule tuner, with the same meaning per MoE layer shape; `tools/moe_tune_to_table.py` turns its cache into a route table's MoE bands. |
| `NINFERNO_MOE_TUNE_CACHE` | file path | Where the MoE tuner's choices live. Default `moe_tune.json` in NInferno's cache directory (as above). |
| `NINFERNO_ROUTE_DIR` | directory | The directory of per-GPU-class route tables. Default: `routes/` beside the binary, then `share/ninferno/routes/` above it, then the source tree the binary was built from (only a build with `NINFERNO_SOURCE_TREE_PATHS=ON`, the default; a release has none). |
| `NINFERNO_MMQ_VERIFY` | unset (default), `0` | `0` (any value starting with it) keeps the IQ formats' 9..16-token verify rounds of several sequences off ggml's mul_mat_q, on the route the table or heuristic gives them. For an A/B of that rule. |
| `NINFERNO_ROUTE_TABLE` | file path | One route table to load instead of the GPU class's own. A table that fails to load is reported at startup. |
| `NINFERNO_ROUTE_RECORD` | file path | Appends every distinct problem that reaches the generic GEMM path, in `ninferno_route_tune --shapes` format, so a model's shapes come from running it (see [bench/README.md](../../bench/README.md)). |

## Speculation data

| Variable | Values | Effect |
|---|---|---|
| `NINFERNO_DRAFTER_REGISTRY` | file path | A drafter registry JSON to use instead of the compiled-in one. |
| `NINFERNO_SPEC_CORPUS` | file path | The prompt corpus `ninferno spec-tune` measures on, instead of the compiled-in one. |
| `NINFERNO_RANKING_DIR` | directory | Searched first for the token-frequency rankings a drafter conversion names; then the source tree given to `ninferno-convert` and the tree the binary was built from (`tools/spec/rankings`; only a build with `NINFERNO_SOURCE_TREE_PATHS=ON`). |

## Loading

| Variable | Values | Effect |
|---|---|---|
| `NINFERNO_ARTIFACT_DIRECT_IO` | `1`/`true`/`yes`/`on`, `0`/`false`/`no`/`off` | Forces `O_DIRECT` or buffered reads of artifact payloads. Unset or any other value: chosen automatically. |

## Numerics investigation

These write tensors to disk and synchronize the device after each step: they localize a numeric
difference between two paths and are never for timing. Graph-captured passes are skipped.

| Variable | Values | Effect |
|---|---|---|
| `NINFERNO_QWEN_DUMP` | directory | Qwen3.5 prefill: appends each layer's intermediates to `<dir>/L<layer>.<kind>.bin`. |
| `NINFERNO_QWEN_DUMP_ROWS` | `lo:hi` | Only the token columns whose absolute position is in `[lo, hi)`. |
| `NINFERNO_QWEN_DUMP_KINDS` | `a,b,...` | Only these intermediate kinds. Default: all. |
| `NINFERNO_QWEN_PREFILL_CUTS` | `p1,p2,...` | Ends a prefill call at each of these absolute positions, the way a restored or continued context meets the cache off the chunk grid. |
| `NINFERNO_GEMMA4_DUMP` | directory | Gemma 4: appends each layer's intermediates to `<dir>/L<layer>.<kind>.bin`. |
| `NINFERNO_GEMMA4_DUMP_ROWS` | `lo:hi` | Only these absolute positions. Default: all. |
| `NINFERNO_DENSE_DUMP` | directory | Dense decoders, batch-1 eager calls: writes `q`, `attn`, `ao`, `mlp` and `x` per layer, the embedding, the final-norm output and the last column's logits to `<dir>/c<call>_p<position>_t<tokens>.L<layer>.<kind>.bin`. |
| `NINFERNO_DENSE_DUMP_ROWS` | `lo:hi` | Only these absolute positions. Default: all. |
| `NINFERNO_SCORE_LOGITS` | file path | `ninferno-perplexity` on Qwen3.5: appends each scored column's logits over the public vocabulary as FP32, for comparing distributions with a reference. |
| `NINFERNO_SCORE_ARGMAX` | file path | The same, the argmax id per scored column, for top-1 agreement. |

## Fault injection

For the tests of failure handling. With the file-triggered variables, creating the named file
while the server runs triggers the fault.

| Variable | Values | Effect |
|---|---|---|
| `NINFERNO_DEBUG_POISON` | `0` (default), `1`/`nan`, a byte `1..255` | Overwrites pooled device memory with that byte whenever it is released (`1` and `nan` mean `0xFF`: NaN in every float format, -1 in indices), so a reader that skips initialization sees poison rather than the previous owner's data. |
| `NINFERNO_DEBUG_POISON_SKIP` | `kv,ingress,state,weights` | Leaves these sites unpoisoned, for bisecting which poisoned memory changed a result. |
| `NINFERNO_DEBUG_FREEZE_FILE` | file path | While the file exists, the executor blocks as on a hung device wait. |
| `NINFERNO_DEBUG_TRAP_FILE` | file path | Once the file appears, the next check launches a kernel that traps, which poisons the CUDA context as a failed device-side assert does. Fires once. |
| `NINFERNO_DEBUG_CUDA_ERROR_FILE` | file path | When the file exists, the next check removes it and raises a recoverable CUDA error (`cudaErrorMemoryAllocation`). Each new file is one error. |
