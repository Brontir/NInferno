# Benchmarks

The executables under `bench/` measure NInferno at three levels: the complete public
`ninferno::Engine` route (`ninferno_bench`), single public Ops (`ninferno_<op>_bench`), and model
round schedules (`ninferno_qwen3_5_*_round_bench`). Use `ninferno_bench` to compare end-to-end
prefill and decode throughput on your GPU, the Op benchmarks to qualify or tune a kernel route, and
`ninferno_context_cost_bench` to calibrate context-cache cost presets. Correctness is covered by the
test suites ([tests/README.md](../tests/README.md)); published results live in
[docs/performance](../docs/performance/README.md); Op development rules are in
[op-development.md](../docs/maintainer/op-development.md).

Benchmarks of a running server (TTFT, corpus, concurrency) are driven from
[tools/bench](../tools/bench/README.md).

## Quick start

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNINFERNO_BUILD_BENCHMARKS=ON
cmake --build build -j --target ninferno_bench
./build/bench/ninferno_bench --weights models/qwen3_6_27b.ninferno -p 512,2048 -n 128 -r 5
```

`--weights` takes a `.ninferno` artifact converted from a Hugging Face checkpoint with
`ninferno-convert --fetch org/name --out models/name.ninferno` (see the
[conversion guide](../docs/weight-conversion.md)). The default token corpus is
`bench/fixtures/bench_corpus.ids`; regenerate it with
[`make_bench_corpus.py`](../tools/bench/README.md#corpus-baker).

## Build

All executables are written to `build/bench/`. Benchmarks are off in the default build; the `dev`
configure preset enables them together with products and tests
(see [Build system](../docs/maintainer/build-system.md)). `CMakeLists.txt` includes the
registrations of `ops/`, `inference/`, `context_cost/` and `models/qwen3_5/`; the Op registration
helper is `cmake/NInfernoBenchmarks.cmake`. Op benchmarks of the NVFP4 family are skipped in builds
without NVFP4 support. A benchmark-only configuration does not need a Python interpreter.

## Product benchmark: `ninferno_bench`

`ninferno_bench` slices exact token counts from the corpus, calls `Engine::prepare_tokens()` and then
`Engine::generate()` once per repetition. It has no private prefill or decode loop and does not call
model internals. Requests use raw output, ignore model-default stops and disable prefix reuse, so the
requested token count is exact. With CUDA Graph decode, one ordinary request primes the graph before
the warmups.

```text
ninferno_bench --weights <artifact.ninferno> [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--weights PATH` | required | `.ninferno` artifact. |
| `--corpus PATH` | `bench/fixtures/bench_corpus.ids` | Whitespace-separated token ids; must be at least as long as the largest prompt. |
| `-p`, `--n-prompt LIST` | none | Prefill tests `pp{P}` (comma list; the option may repeat). |
| `-n`, `--n-gen LIST` | none | Decode tests `tg{G}` after a one-token seed. |
| `-pg`, `--prompt-gen P,G;P,G` | none | Combined tests `pp{P}+tg{G}`. |
| `-r`, `--repetitions N` | 5 | Measured repetitions. |
| `--warmup N` | 1 | Discarded repetitions. |
| `--max-ctx N` | sized from the tests | Context capacity; rejected when too small for the selected tests. |
| `--prefill-chunk N` | 1024 | Prefill chunk; multiple of 128. |
| `--kv-dtype T` | `bf16` | `bf16`, `int8`, `fp8`, `nvfp4` or `k8v4`. |
| `--activations M` | `fast` | Quantized-linear activation mode: `exact`, `fast` or `int8`. |
| `--spec B` | none | Speculative backend: `mtp`, `dflash` or `dflash2`. |
| `--draft-tokens N` | none | Draft window; required with `--spec` (1..15; a model may allow fewer, e.g. MTP of Qwen3.5 up to 5). |
| `--lm-head-draft` | off | Use the optimized proposal head; requires `--spec`. |
| `--device N` | 0 | CUDA device ordinal. |
| `--no-cuda-graph` | graph on | Eager decode. |
| `--profile-measured` | off | Bracket the single measured repetition with `cudaProfilerStart/Stop`; needs exactly one test and `-r 1`. |
| `-o`, `--output F` | `table` | `table`, `json` or `csv`. |
| `--output-file PATH` | stdout | Write the report to a file. |

With none of `-p`, `-n`, `-pg` the matrix is `pp512` and `tg128`.

| Test | Meaning |
|---|---|
| `pp{P}` | Prepare `P` tokens, request one output token; `prefill t/s = P / prefill_seconds`. |
| `tg{G}` | Prepare a one-token seed outside the timed phase, request `G+1` tokens. The begin-round token belongs to prefill, so exactly `G` tokens are decoded. |
| `pp{P}+tg{G}` | Same `G+1` convention after a `P`-token prefill; both rates come from one generation call. |

```bash
# DFlash2 companion with the optimized head
./build/bench/ninferno_bench --weights models/qwen3_8_27b_nvfp4.ninferno \
  -pg '2048,128' --spec dflash2 --draft-tokens 7 --lm-head-draft \
  --max-ctx 4096 --kv-dtype bf16 --warmup 1 -r 3

# Nsight Systems capture of one measured repetition
nsys profile -c cudaProfilerApi ./build/bench/ninferno_bench --weights model.ninferno \
  -pg '2048,128' -r 1 --warmup 1 --profile-measured
```

MTP alone reserves a lookahead KV margin of `2 * draft_tokens`. The run-matrix orchestrator
[`run_ninferno_bench_matrix.py`](../tools/bench/README.md#ninferno-performance-matrix) drives this
binary over a preset grid.

### Report

Table, JSON and CSV reports carry the architecture, artifact, Engine configuration, load summary,
memory capacity, KV payload, workspace peak, phase throughput and speculative statistics. JSON
`schema_version` is 15:

| Section | Content |
|---|---|
| `load` | Architecture, public name, actual formats, prefill signature, load/upload time, file/H2D/staging bytes, Device/Host object counts. |
| `memory` | Weight, sequence and unified-workspace arenas, optional (non-additive) Vision layout, planned context, KV storage, CUDA Graph allowance, KV payload. |
| per repetition `timings` | Prepare, Vision, prefill, decode and total seconds. |
| per repetition `speculative` | Window, rounds, drafted/accepted tokens, fallbacks, per-position acceptance. |

Each test also reports `workspace_peak_bytes` (planned phase markers, including Graph replay) and
`workspace_allocator_peak_bytes` (host-side arena activity); they differ because replay reuses
captured addresses. `decode_output_tok_s` counts the requested `G` outputs; `decode_engine_tok_s`
uses the Program's round statistics and therefore includes a final partially committed round. Reports
also record the command line and machine information.

## Context-cost calibration

`ninferno_context_cost_bench` measures the static coefficients that compare context-cache
materialization alternatives. It is an offline tool, not a startup benchmark or runtime autotuner.

| Option | Default | Meaning |
|---|---|---|
| `--suite S` | `all` | `transfer` (machine D2H/H2D/D2D roofline with the production paged-KV copy primitives, no model), `prefill` (loads an artifact and measures Text/Vision recomputation under canonical BF16/no-spec), or `all` (transfer, then prefill). |
| `--artifact PATH` | none | Required by `prefill` and `all`. |
| `--corpus PATH` | `bench/fixtures/bench_corpus.ids` | Prefill token corpus. |
| `--device N` | 0 | CUDA device. |
| `--max-context N` | 8192 | Prefill measurement context. |
| `--prefill-chunk N` | 1024 | Prefill measurement chunk. |
| `--transfer-warmup N` | 2 | Warmups per transfer point. |
| `--transfer-reps N` | 9 | Samples per transfer point. |
| `--prefill-reps N` | 5 | Samples per prefill point. |
| `--json PATH` | stdout | Complete measurement report (every repetition, median/MAD, coefficients, training and held-out predictions). |
| `--preset-out PATH` | none | Atomically update the measured preset components; must differ from `--json`. |

```bash
./build/bench/ninferno_context_cost_bench --suite transfer \
  --json profiles/bench/context_cost_transfer.json --preset-out profiles/bench/context_cost_presets.json
./build/bench/ninferno_context_cost_bench --suite prefill --artifact models/qwen3_6_27b.ninferno \
  --json profiles/bench/prefill_cost.json --preset-out profiles/bench/context_cost_presets.json
```

The fitted models are:

```text
transfer = max(batch_ns + copy_operations * operation_ns, payload_bytes * ns_per_byte)
prefill  = chunks * chunk_ns + suffix_tokens * token_ns + attention_pairs * attention_pair_ns
         + vision_items * vision_item_ns + vision_patches * vision_patch_ns
```

`payload_bytes` is the copied payload without Host arena padding; `copy_operations` is the number of
physical `cudaMemcpy*` calls implied by the page geometry. For a suffix `S` after prefix `B`,
`attention_pairs = B*S + S*(S+1)/2`; `chunks` is the sum of `ceil(segment_tokens/prefill_chunk)` over
the prefill schedule's capture/rewrite segments. Resource kind, KV dtype and speculative backend do
not select coefficients. Both results are keyed by the measuring device's GPU class id
(`DeviceClass::id`, `gb202:170sm`), the prefill result also by the actual Text/Vision
`prefill_signature`.

A fit is accepted when training and held-out p95 relative error are at most 35% (transfer) or 15%
(prefill) and materially different points are never predicted in reverse order; exact predicted ties
are allowed. A rejected fit still writes the report, exits with status 3 and leaves the preset
unchanged. `--preset-out` replaces only the component its suite measured (transfer: the node's three
directions; prefill: one prefill-signature entry) and the file is parsed by the runtime's strict
loader before it is published.

Defaults compiled into the binary live in
[`context_cost_defaults.cpp`](../src/runtime/engine/context_cache/context_cost_defaults.cpp). A preset
file is selected at run time with `EngineOptions.context_cost.preset_path` or
`ninferno-serve --context-cost-presets FILE`, with no rebuild. Resolution starts from generic
numerical coefficients, then applies the compiled machine of the first key of the device's class
chain that has one (else the one of the nearest measured board within
`measured::kCarryDistance`, as measured), then the external entry of the first key of the chain
that has one; a malformed explicit file is an error and a missing entry keeps the preceding layer.
A compiled machine records the board it was measured on (`core/measured_boards.h`). The `--json`
report is provenance only, never a runtime input.

## Op benchmarks

### Conventions

Every `ninferno_<op>_bench` calls one public Op (production dispatch picks the route; there are no
private launchers, route forcing or candidate controls) and times it with CUDA events.

- Inputs are reproducible pseudorandom values with distinct operand seeds, finite, with
  format-correct codes and scales. Control inputs (masks, positions, routing) keep their defined
  meaning; model benchmarks use real artifact weights.
- Setup, restoration of in-place operands, workspace queries and graph capture are outside the timed
  interval. Each timed graph or eager interval holds one Op call.
- Cold cache: an L2 eviction read (256 MiB unless `--flush-mib` says otherwise) completes before the
  timed interval. Warm cache reuses buffers.
- Graph bundles (`--graph-calls N`) capture N serial calls and report time per call; a cold bundle
  flushes only before its first call. Bundles cannot be combined with `--profile`.
- `--profile` brackets exactly one public call with CUDA profiler start/stop so that `ncu`/`nsys`
  capture it (use `--profile-from-start off`); it needs one exact point.
- Bandwidth and FLOP figures are semantic useful-work models (weights once, activations, outputs),
  not DRAM traffic; use a profiler for physical traffic. Percentages use per-GPU-class reference
  peaks from `bench/ops/ninferno_bench_common.h` (measured classes `gb202:170sm` and `gb203:84sm`);
  on other GPUs bandwidth is derived from the device and Tensor Core percentages are reported as
  unknown.
- `NINFERNO_BENCH_PREFILL_ROUTES=1` runs the Op under the prefill route profile.

Options shared by most Op benchmarks (check `--help` of a tool for the exact set and defaults; the
default warmup and repeat counts differ per tool):

| Option | Meaning |
|---|---|
| `--tokens LIST`, `--t-sweep LIST`, `--sweep A:B[:STEP]` | Matrix columns T to measure. `--sweep` is an inclusive range. |
| `--execution eager\|graph\|both` | Time the public call directly or one graph replay. |
| `--cache cold\|warm\|both` | Cache state of the timed interval. |
| `--graph-calls N` | Calls per graph bundle (1..64 where supported). |
| `--warmup N`, `--repeat N` | Discarded and measured samples. |
| `--flush-mib N` | L2 eviction size. |
| `--csv-out PATH` | Write rows as CSV (median, min and p95 latency plus work rates). |
| `--profile` | One profiled public call. |

Compare implementations only with the same fixture, seed, cache policy and timing mode; the
retained fixtures replace older constant or periodic inputs.

### Linear

`ninferno_linear_bench` measures only the public pure `linear()` contract.

| Option | Default | Meaning |
|---|---|---|
| `--qtype Q` | required for a point | `q4`, `q5`, `q6`, `q8`, `bf16`, `nvfp4`, `fp8`, `fp8r32`, or any other format by its route-table name (`q4k`, `iq4xs`, `mxfp4`, ...). |
| `--policy P` | `a16` | Activation-compute permission: `a16`, `a8` (FP8) or `a4` (NVFP4). The production resolver picks the route within it. |
| `--n N`, `--k K` | required | Output rows and reduction extent. |
| `--t T` / `--sweep A:B[:STEP]` | one required | Single T or continuous sweep (one packed weight, one maximum-T allocation). |
| `--suite S` | none | `qwen3_6_27b`, `qwen3_6_35b_a3b`, `gemma4_mtp`, `gemma4_dflash` or `all`; compact surveys, not combinable with an explicit point or `--profile`. |
| `--execution M` | `eager` | `eager` or `graph`. |
| `--graph-calls N` | 1 | 1..64; needs `--execution graph`, no `--profile`. |
| `--warmup N` / `--repeat N` | 3 / 20 | Samples per point. |
| `--flush-mib N` | 256 | L2 eviction. |
| `--csv-out PATH` | none | Rows incl. `execution` and Graph `graph_nodes`; not with `--profile`. |
| `--profile` | off | Needs one exact `--t`. |

```bash
./build/bench/ninferno_linear_bench --qtype q4 --policy a16 --n 4096 --k 5120 --t 8
./build/bench/ninferno_linear_bench --qtype nvfp4 --policy a4 --n 14336 --k 5120 --t 1024
./build/bench/ninferno_linear_bench --qtype q4 --n 4096 --k 5120 --sweep 1:128:1 --csv-out q4.csv
./build/bench/ninferno_linear_bench --suite qwen3_6_27b
ncu --profile-from-start off --set full ./build/bench/ninferno_linear_bench \
  --qtype q4 --policy a16 --n 4096 --k 5120 --t 8 --profile
```

When a point group includes `T=1`, `T1_lin_x` is the calculated scaling reference
`T * median(T=1) / median(T)`; no sequential `T=1` launches are issued. Effective bandwidth counts
the encoded weight planes once plus one BF16 activation read and one BF16 output write; FLOPs are
`2*N*K*T`. Workload priorities, dispatch trade-offs and the final curve tables are in
[Linear tuning](../docs/maintainer/linear-tuning.md).

### Benchmark index

Targets built under `build/bench/`. Rows list the tool, what it measures and its distinctive
options (shared options above apply unless noted). Default `--warmup/--repeat` are in the last
column where the source fixes them.

| Target | Measures | Distinctive options | W / R |
|---|---|---|---|
| `ninferno_linear_bench` | Pure `linear()` | see above | 3 / 20 |
| `ninferno_linear_pair_bench` | `linear_pair()` over adjacent Q8 `[1024,K]` K/V row views of a `[6144,K]` parent | `--k 2048\|5120` | 5 / 30 |
| `ninferno_linear_topk_bench` | Projection plus stable top-16 for the Q8 and FP8 full heads and the mapped Q4 optimized head, replayed in a Graph after an L2 flush | `--profile all\|q8-full\|fp8-full\|q4-optimized`, `--columns U,...` (default sweep U=1..120) | 8 / 60 |
| `ninferno_q4_linear_swiglu_bench` | Q4 `[34816,5120] -> [17408,T]` fused SwiGLU | `--t-sweep` | 5 / 30 |
| `ninferno_q8_linear_swiglu_bench` | Q8 fused SwiGLU | `--problem companion` (`[12288,2048]`) or `dflash2` (`[34816,5120]`) | 5 / 30 |
| `ninferno_nvfp4_linear_swiglu_bench` | NVFP4 fused SwiGLU incl. activation quantization | `--policy` | 5 / 30 |
| `ninferno_fp8_linear_swiglu_bench` | Row-scaled FP8 fused SwiGLU | `--policy a8\|a16` | 5 / 30 |
| `ninferno_q5_linear_add_bench` | Q5 LinearAdd `[5120,6144]`, `[5120,17408]` | `--k 6144\|17408` | 5 / 50 |
| `ninferno_q8_linear_add_bench` | Q8 LinearAdd `[2048,4096]`, `[2048,6144]` | `--k`, `--production-only` | 5 / 30 |
| `ninferno_bf16_linear_add_bench` | BF16 `[5120,6144]` with in-place residual | `--sweep`, `--t-sweep`, `--route production`, `--flush-bytes` | 3 / 20 |
| `ninferno_fp8_linear_add_bench` | Row-scaled FP8 LinearAdd | `--k 6144\|17408`, `--policy` | 5 / 30 |
| `ninferno_nvfp4_linear_add_bench` | NVFP4 LinearAdd | `--k`, `--n`, `--policy` | 5 / 30 |
| `ninferno_linear_dynamic_grouped_conv_add_bench` | Linear plus dynamic grouped convolution add | `--width 2..16`, `--k 4096\|17408`, `--batch 1..8` | 8 / 40 |
| `ninferno_dynamic_grouped_conv_prepare_bench` | BF16 `rmsnorm_dynamic_grouped_conv_prepare` for every `B=1..8` (RMSNorm, `[1280,5120]` projection, two-tap grouped conv, both outputs) | `--width`, `--batch` | 8 / 40 |
| `ninferno_attn_input_proj_bench` | Every registered `attn_input_proj()` shape: 27B Q4/Q5 two-parent; 35B Q8 Q/K/gate/V and companion Q/K/V; DFlash2 Q8 `[6144,5120]`; 27B BF16, NVFP4, FP8 single-parent | `--format`, `--nvfp4-policy`, `--fp8-policy` | 5 / 30 |
| `ninferno_gdn_input_proj_bench` | Every registered `gdn_input_proj` form (27B Q4/Q5, 35B Q8, 27B NVFP4/FP8) | `--format all\|...`, `--nvfp4-policy`, `--fp8-policy` | 5 / 30 |
| `ninferno_gdn_input_proj_conv_snapshot_bench` | `gdn_input_proj_conv_snapshot` / `_record` for exact `B=1..8` (Q4/Q5, NVFP4, FP8, Q8) | `--form snapshot\|record\|both`, `--format`, `--batch`, `--valid-columns`, `--sweep` (default T=1..6; Record starts at T=2) | 10 / 100 |
| `ninferno_gdn_gating_proj_bench` | `gdn_norm_gating_proj` (`--op norm`) or `gdn_gating_proj` (`--op control`) | `--geometry 27b\|35b`, `--weights parent\|split`, `--op` | 10 / 61 |
| `ninferno_gdn_gating_bench` | `gdn_gating` at the 27B gate shape | `--decode`, `--prefill` | n/a |
| `ninferno_gated_delta_net_bench` | BF16 Gated DeltaNet (state/head dim 128) | `--running`, `--batch-update`, `--force-chunked`, `--recurrent-only`, `--chunked-only`, `--breakdown`, `--qk-norm composed\|fused`, `--value-heads`, `--qk-heads`, `--sweep`, `--csv` | 20 / 100 |
| `ninferno_gdn_replay_bench` | ReplaySSM recurrent-Record and all-layer Fold | `--profile 27b\|35b\|all`, `--component fold\|recurrent\|all`, `--batch`, `--commits`, `--valid`, `--width` | 10 / 50 |
| `ninferno_causal_softmax_attention_bench` | Causal cache entries: append-and-attend and cached-only; D256 H24/KV4 and H16/KV2; BF16, INT8-G64, FP8-row256, NVFP4-G16, K8V4 | `--entry append\|cached\|both`, `--geometry`, `--kv-dtype`, `--batch`, `--context`, `--mapping`, `--row-contexts`, `--valid-columns`, `--table-rows`, `--envelope-max` | 5 / 30 |
| `ninferno_context_softmax_attention_bench` | Read-only context-plus-query attention, Q32/KV8/D128 BF16 | `--tokens`, `--context` | 5 / 50 |
| `ninferno_sliding_window_attention_bench` | Q32/KV8/D128 symmetric sliding window over a 2048- or 4096-slot cyclic cache | `--window 2048\|4096`, `--batches`, `--envelope-max` | 5 / 50 |
| `ninferno_packed_softmax_attention_bench` | Dense attention: uniform plain-segment and packed (cumulative lengths) entries | `--entry uniform\|packed\|both`, `--segments`, `--length`, `--segment-lengths` | 3 / 30 |
| `ninferno_gemma4_attention_bench` | Gemma 4 attention (26B/31B, sliding D256 and full D512) over head-major slots | `--mode decode\|prefill\|all`, `--mask causal\|block\|both`, `--geometry`, `--decode-tokens`, `--contexts`, `--prefill-tokens`, `--bases`, `--envelope` | 3 / n/a |
| `ninferno_kv_cache_append_bench` | KV append: `--mode full` (D256 KV4/KV2, five KV types) and `--mode prefix` (BF16 D128/KV8 paged or cyclic) | `--full-geometry`, `--kv-dtype`, `--layout`, `--cyclic-capacity 2048\|4096`, `--counts`, `--max-count` | 5 / 50 |
| `ninferno_prepare_ragged_prefix_bench` | Gather/zero-fill with positions and counts (D=25600 DFlash2, 16384 DFlash) | `--rows`, `--widths`, `--batches`, `--counts full\|one\|ragged\|zero` | 10 / 100 |
| `ninferno_prepare_masked_block_bench` | I32 anchor/mask transform, every `B=2..16` | `--block-sizes` | 20 / 101 |
| `ninferno_context_kv_materialize_bench` | Five-layer DFlash2 context K/V materialization (capacity 2048; `W=1..16`, `B=1..8`, prefill `W` up to 2048) | `--widths`, `--batches`, `--counts` | 10 / 100 |
| `ninferno_scatter_bf16_batch_bench` | Five `scatter_bf16_batch` calls into a `[5*D,W,8]` pool | `--rows` (default 5120), `--widths`, `--batches`, `--counts` | 10 / 100 |
| `ninferno_scatter_bench`, `ninferno_vision_pos_embed_bench` | Scatter, Vision position embedding | `--d`, `--vision`, `--patch`, `--control`, `--profile-once` | n/a |
| `ninferno_embedding_bench` | Quantized embedding profiles | `--format q6-d5120\|q8-d5120\|q8-d2048\|fp8-d5120`, `--id-pattern normal\|masked`, `--block-width` | 10 / 100 |
| `ninferno_rmsnorm_bench` | Public RMSNorm | `--kind dflash2_hidden\|hidden27\|target_q27\|target_k27` (more in `--help`) | 10 / 100 |
| `ninferno_rmsnorm_pack_tail_bench`, `ninferno_rmsnorm_rope_bench`, `ninferno_rope_bench`, `ninferno_l2norm_bench`, `ninferno_layer_norm_bench` | Norm and rotary Ops | `--form`, `--route split\|fused`, `--widths`, `--batches`, `--geometry`, `--text`, `--vision`, `--control` | varies |
| `ninferno_residual_add_bench`, `ninferno_sigmoid_mul_bench`, `ninferno_silu_mul_bench`, `ninferno_gelu_bench`, `ninferno_add_bias_bench`, `ninferno_position_bench` | Pointwise Ops over the registered small, established and maximum Vision shapes | `--tokens`, `--patches`, `--d`, `--columns`, `--mode tanh\|exact`, `--control` (same topology, minimal bitwise work), `--candidate-block` | varies |
| `ninferno_causal_conv1d_silu_bench` | Causal conv1d + SiLU, one entry per run | `--decode`, `--prefill`, `--distinct`, `--snapshot`, `--split`, `--legacy-stage`, `--cache warm\|cold` (default warm), `--channels`, `--tokens`, `--batch`, `--valid-columns`, `--slots`, `--initial-slot` | n/a |
| `ninferno_mtp_pack_bench` | Bit-exact MTP pack or attention-split | `--op pack\|split`, `--d 2048\|5120`, `--tokens` | n/a |
| `ninferno_proposal_remap_bench` | Proposal token-id remap | `--tokens` | n/a |
| `ninferno_argmax_bench` | Public `argmax` on the full vocabulary and the 131072-row shortlist | `--shape full\|shortlist`, `--cols` | n/a |
| `ninferno_sampling_select_bench` | Sampling, MTP window selection and DFlash2 sparse-q acceptance | `--matrix`, `--sample`, `--mtp`, `--dflash2`, `--mode greedy\|stochastic`, `--batch`, `--top-k`, `--mtp-k`, `--drafts`, `--extent`, `--mixed`, `--general`, `--no-counts`, `--reject-at`, `--graph-calls` | 8 / 60 |
| `ninferno_candidate_selector_bench` | Conditional candidate selector, `K=1..15`, `B=1..8`, 16 candidates, rank 256, BF16 codebooks `[256,248320]`; greedy, stochastic and mixed | `--steps`, `--batch`, `--flush-mib` | 8 / 50 |
| `ninferno_sparse_moe_bench` | 35B routed-plus-shared post-mixer Op | `--codec q4-q5\|q4-q6\|q8-q8\|nvfp4\|all`, `--distribution trace-like\|independent\|same`, `--seed`, `--sweep` | 5 / 50 |
| `ninferno_moe_bench` | Generic MoE Op, any geometry and expert format | see [below](#generic-moe-and-gemma-4-moe) | repeat 20 |
| `ninferno_gemma4_moe_bench` | Gemma 4 26B-A4B MoE, route by route | `--arms`, `--passes`, `--duplicate` | 3 / 20 |

Execution defaults differ: the linear-family benchmarks run `eager` unless noted, most others default
to `graph` and `cold`. Run a tool with `--help` for its defaults.

### Notes on selected benchmarks

- **Attention.** The four softmax-attention executables keep fixture setup, workspace queries, L2
  conditioning and graph capture outside the timed interval and expose no route labels. The causal
  benchmark takes uniform profiles (`--batch`, `--context`) or one exact mixed profile
  (`--row-contexts`, `--valid-columns`, `--table-rows`, each with exactly `B` entries; cached-only
  stays `B=1`). `--envelope-max N` fixes a `[1,N]` execution envelope while contexts vary; CSV rows
  record both bounds. Persistent K+V bytes per D256 vector are 516 (FP8), 288 (NVFP4) and 402
  (K8V4: 258-byte K plus 144-byte V). Compute roofline fractions from the ordinary latency `t`, not
  profiler replay time: `unique_kv_bytes / (t * peak_bandwidth)` and
  `(qk_flops / peak_qk + pv_flops / peak_pv) / t`. Use the Gemma 4 benchmark's `--envelope N` to
  hold its capture envelope above the measured context; its decode rows (`T <= 16`) report K/V bytes
  of the union key range per second, and it times two identical graphs interleaved so the second
  serves as a null arm.
- **Gated DeltaNet.** `--running` measures the running-state entry; the prefill dispatcher uses
  recurrence below 16 tokens and a two-kernel chunked route above. `--force-chunked` and
  `--recurrent-only` call the production launchers with fused Q/K normalization to find the
  crossover; `--chunked-only` forces the chunked route with pre-normalized Q/K; `--breakdown` adds
  isolated `chunked.prepare` and `chunked.recurrence` timings (`stage_share_pct` partitions the sum
  of isolated medians, each flushed separately). `logical_bytes` counts each tensor and state
  transfer once; `tensor_io_request_bytes` counts CTA I/O requests; neither is DRAM traffic.
- **Embedding.** `--id-pattern masked` generates an anchor followed by masks at the `--block-width`
  period (248070 for D=5120, 248077 for Q8/D=2048); use `T = B*W` for complete proposal blocks.
  `--graph-calls 32 --execution graph --cache warm` measures repeated calls in one graph.
- **Sparse MoE.** Graph replay with timing event nodes is the authoritative mode. `trace-like` is
  the primary single-sequence verification distribution; `independent` and `same` bound zero and
  complete expert overlap. `unique_weight_gbps` counts router, shared and selected-expert weights
  once (a lower bound on traffic). `--codec nvfp4` needs an `sm_120a` build.
- **Causal conv1d.** `--split` selects the split-output entry whose partition follows the channel
  extent (`--channels 8192` gives 2048/2048/4096, `10240` gives 2048/2048/6144). `--legacy-stage`
  times the packed convolution plus three `extract_bf16_columns` calls that the split entry
  replaces.
- **Sampling.** The G4 DFlash2 mode reports the raw-greedy or general route, Graph node count,
  workspace and whole-Op time; `--mixed` cycles raw greedy, penalty-bearing greedy and stochastic
  rows, `--general` forces the conservative envelope, `--reject-at J` rejects one proposal,
  `--extent 0` checks the bonus-only case, `--drafts` defaults to the checkpoint recommendation of
  seven.
- **Pointwise `--control`.** Keeps the kernel topology and payload but replaces the mathematical
  operation with minimal bitwise work, to separate memory from math cost.

### Generic MoE and Gemma 4 MoE

```text
ninferno_moe_bench --format F [--down-format F] --experts E --topk K --hidden H --inter I
                   [--act swiglu|geglu|clamped|relu2] [--layout concatenated|interleaved]
                   [--tokens 1,2,4,...] [--plans auto|sweep] [--prefill] [--repeat 20]
                   [--passes 3] [--flush-mib 256] [--graph 0|1] [--stages 0|1] [--a8 0|1]
                   [--frag 0|1]
```

Defaults: `--format mxfp4`, 32 experts, top-k 4, hidden 2880, inter 2880, `swiglu`, `concatenated`,
tokens `1,2,4,8,16,32,64,128,512,2048`, `--plans auto`. Formats: `mxfp4 nvfp4 bf16 fp8 q4 q5 q6 q8`
plus ggml blocks (`q4_k q5_k q6_k q3_k q2_k iq4_xs iq3_xxs iq3_s iq2_xxs iq2_xs iq2_s q4_0 q4_1 q5_0
q5_1 iq4_nl`). `--down-format` gives the down bank its own format. `--plans sweep` times every
(gate/up, down) schedule pair; `--graph 1` times graph replays; `--stages 1` also times the gate/up
and down launches alone; `--a8 1` admits R8 schedules; `--frag 1` stores ggml banks in the
fragment layout the converter uses. Each row ends with a digest of the plan's output: equal digests
mean bit-identical results.

`ninferno_gemma4_moe_bench [--tokens ...] [--passes 5] [--repeat 20] [--warmup 3] [--arms
public,decode,small,a4e,a4t,a4n,a16] [--warm] [--calibrate] [--duplicate N]` runs each arm as a
graph, interleaved pass by pass, with one arm repeated under a second name as a noise-floor control;
`--duplicate N` repeats every N-th token over the next N-1 to concentrate routing.

## Route tuning helpers

| Target | Purpose |
|---|---|
| `ninferno_route_tune` | Offline autotuner of the generic projection path. `--shapes FILE` (one `<format> <compute> <epilogue> <n> <k>` line per problem; the engine records such a file when run with `NINFERNO_ROUTE_RECORD=FILE`) `[--out routes/<key>.json] [--key chip\|variant\|arch] [--samples 15] [--prefill-chunk 1024] [--prefill-only 1] [--points T,...] [--routes-out FILE.csv]`. Writes a per-GPU-class `ninferno-route-table/1` file. |
| `ninferno_linear_route_sweep_bench` | For each T, times the dispatcher's route and every route the table would pick at a larger width, and checks their outputs. `--qtype --n --k [--policy] [--t LIST] [--as LIST] [--quantized both] [--w4a8-schedules] [--random-weight] [--no-class-table] [--warmup 2] [--repeat 10] [--csv-out]`; `--qtype` also takes the route table's format names (q4k, iq4xs, ...); `--w4a8-schedules` (with `--policy a8`) times every admitted W4A8 schedule and names the dispatcher's route; feed the CSVs to `tools/route_as_table.py` ([tools](../tools/README.md)). |
| `ninferno_r8_schedule_bench` | Every compiled R8 decode schedule (T=1) next to the dispatcher's pick, with a byte-equality check. `[--shapes FILE] [--problem FORMAT,EPILOGUE,N,K ...] [--samples 15] [--launches 60]`; `tools/bench/r8_decode_table.py` turns repeated sweeps into a table header. |
| `ninferno_nvfp4_a16_mma_sweep_bench` | A16 tensor-core schedule sweep on the Qwen3.5-27B NVFP4 shapes; CSV on stdout. `[--shape n34816_k5120\|...\|all] [--tokens LIST] [--warmup N] [--repeat N]`. |

## Model round benchmarks and quality probe

| Target | Purpose |
|---|---|
| `ninferno_qwen3_5_mtp_round_bench` | Native MTP proposal and verification round through the production Program over a real artifact; reports round latency and licensed-token counts. |
| `ninferno_qwen3_5_dflash_round_bench` | Consecutive steady DFlash rounds (context append, six-layer proposal, verify/accept, host publication) on a 35B artifact; reports GPU and wall latency, acceptance (overall and per position), mean licensed length and published tokens/s. Run separate invocations per `K`, head and context. |
| `ninferno_quality_probe` | Token-level outputs for comparison with an independent reference (for example Hugging Face transformers). `<artifact> score --tokens T.bin --first-target F --out O.bin [--context N] [--kv-dtype T]` writes FP32 teacher-forced log-probabilities (set `NINFERNO_SCORE_ARGMAX=FILE` to also append argmax ids); `<artifact> generate --tokens T.bin --max-new N --out O.bin [--context N] [--spec mtp\|dflash\|dflash2 --draft-tokens K]` writes a greedy continuation of the raw prompt ids. Token files are little-endian int32. |

| Option | MTP round default | DFlash round default | Meaning |
|---|---|---|---|
| `--artifact PATH` | `out/qwen3_6_27b.ninferno` | `out/qwen3_6_35b_a3b.ninferno` | `.ninferno` artifact. |
| `--device N` | 0 | 0 | CUDA device. |
| `--warmup N` | 2 | 2 | Warmup rounds. |
| `--reps N` | 10 | 10 | Measured rounds. |
| `--draft-tokens K` | 5 | 15 | Draft window. |
| `--context N` | n/a | 128 | Context length before the measured rounds. |
| `--batch N` | n/a | 1 | Batch size. |
| `--proposal-head H` | `optimized` | `optimized` | `full` or `optimized`. |
| `--no-cuda-graph` | graph on | graph on | Eager execution. |

```bash
./build/bench/ninferno_qwen3_5_mtp_round_bench --artifact models/qwen3_6_27b.ninferno
./build/bench/ninferno_qwen3_5_dflash_round_bench --artifact models/qwen3_6_35b_a3b.ninferno \
  --context 4096 --draft-tokens 15 --proposal-head optimized
```

Raw reports and profiler captures stay local under `profiles/bench`, `profiles/ncu` and
`profiles/nsys`.
