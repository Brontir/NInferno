# Linear benchmark contract and preset suites

## Scope

This document is the current authority for `bench/ops/linear_bench.cu`. It defines the commands, timing contract, metrics, preset suites, and extension rules
of the pure Linear benchmark. The benchmark does not change the production Linear route,
nor does it re-select any route winner.

The benchmark measures only:

```text
ninferno::ops::linear(x, w, out, policy, workspace, stream)
```

This is a long-lived public benchmark. The measured calls of single, sweep, suite, and profile must all
go through the public entry point above; the fixture may build inputs according to public weight formats, but the benchmark must not include
Linear-private launcher/plan/dispatch headers, must not call `ninferno::ops::detail`, and must not provide
candidate, kernel, or route forcing options.

Q4/Q5/Q6/Q8 LinearAdd, LinearSwiGLU, LinearPair, and other fused Ops do not belong to this
benchmark. They continue to be measured independently by their own benchmarks.

Q4/Q5/Q6/Q8 and BF16 use the existing A16 route. The following NVFP4 exact problems support both the A16
and A4 policies and serve as a permanent development surface, not added to the model suites:

```text
[14336,5120]  attention input
[16384,5120]  GDN input
[34816,5120]  MLP gate/up development parent
[ 5120,6144]  attention/GDN output
[ 5120,17408] MLP down
```

`--policy a4` measures the complete public call; the production resolver selects an already qualified A16 or A4 route from the exact geometry and T.
The benchmark does not duplicate or infer this private selection.
The tuning range, the 512/1024 large-T anchors, the key points, and the final report format are all defined in
[Linear tuning and reporting specification](linear-tuning.md).

## 1. Use cases

This benchmark serves four concrete needs.

### 1.1 Single-point performance

Explicitly specify the weight type, `N`, `K`, and `T` to measure one production Linear point:

```bash
./build/bench/ninferno_linear_bench \
  --qtype q4 --policy a16 \
  --n 4096 --k 5120 --t 8
```

The numeric `(N,K)` is the primary entry. The benchmark does not duplicate the full shape admission table of the production selector;
unsupported points are rejected by the public `linear()` and its format selector.

The command for a BF16 decode exact point is:

```bash
./build/bench/ninferno_linear_bench \
  --qtype bf16 --policy a16 \
  --n 14336 --k 5120 --t 1
```

The permanent A16 decode point for NVFP4 is:

```bash
./build/bench/ninferno_linear_bench \
  --qtype nvfp4 --policy a16 \
  --n 14336 --k 5120 --t 1
```

Its main W4A4 MMA point is:

```bash
./build/bench/ninferno_linear_bench \
  --qtype nvfp4 --policy a4 \
  --n 14336 --k 5120 --t 1024
```

The other four permanent NVFP4 problems use the same numeric entry, for example:

```bash
./build/bench/ninferno_linear_bench \
  --qtype nvfp4 --policy a4 \
  --n 16384 --k 5120 --t 1024

./build/bench/ninferno_linear_bench \
  --qtype nvfp4 --policy a4 \
  --n 34816 --k 5120 --t 1024

./build/bench/ninferno_linear_bench \
  --qtype nvfp4 --policy a4 \
  --n 5120 --k 6144 --t 1024

./build/bench/ninferno_linear_bench \
  --qtype nvfp4 --policy a4 \
  --n 5120 --k 17408 --t 1024
```

The workspace is allocated before the timed region according to the public capacity query; all launches and traffic of activation quantization and
GEMM are inside one timed `linear()`. A result that counts only the GEMM after pre-quantization is not
a production metric of this benchmark.

FP8 uses the same entry, for example:

```bash
./build/bench/ninferno_linear_bench \
  --qtype fp8 --policy a8 \
  --n 14336 --k 5120 --t 1024
```

`--execution graph` measures the complete Op inside a CUDA Graph; `--graph-calls N` places `1..64` calls in one timed graph
and reports the time per call. The default is eager. Both timing modes call the same public Op.

### 1.2 NCU single point

Profile mode must accept exactly one exact point:

```bash
ncu --profile-from-start off ... \
  ./build/bench/ninferno_linear_bench \
  --qtype q4 --policy a16 \
  --n 4096 --k 5120 --t 8 --profile
```

The program completes allocation, payload initialization, warmup, and L2 flush outside the capture, then:

```text
cudaProfilerStart()
    one public linear() call
    stream synchronize
cudaProfilerStop()
```

NCU can therefore see the one or more kernel launches actually issued by that host launcher, without simultaneously
capturing weight initialization, the copy roofline probe, the Tensor Core probe, warmup, or repeated measurements.

The following distinctions must still be kept:

- a benchmark point is one public Linear call;
- what the selector returns is a host launcher;
- a host launcher may issue one or more kernel launches;
- what NCU shows is a concrete kernel template instance;
- the benchmark does not mislabel a schedule, template instance, or host launcher as a kernel.

### 1.3 Small-T sweep

Fix `(qtype,N,K,policy)` and scan a T range continuously:

```bash
./build/bench/ninferno_linear_bench \
  --qtype q4 --policy a16 \
  --n 4096 --k 5120 \
  --sweep 1:32:1 \
  --csv-out profiles/bench/q4_4096x5120_t1_32.csv
```

The Vision step domain can be expressed directly as:

```bash
--sweep 4:512:4
```

Each T outputs the change in median latency relative to the adjacent point. The sweep is used to find unreasonable latency steps, observe existing
production route seams, and provide leads for separate route measurement tasks; each point still calls only the public
Linear. It does not compare forced candidates, does not set automatic performance gates, and does not duplicate the selector,
crossover, or candidate-legality tables in the benchmark.

### 1.4 Typical model suites

To avoid hand-writing `(qtype,N,K,T)` each time, the benchmark registers a small number of representative 27B/35B points. One
suite invocation runs all points in order:

```bash
./build/bench/ninferno_linear_bench --suite qwen3_6_27b
./build/bench/ninferno_linear_bench --suite qwen3_6_35b_a3b
./build/bench/ninferno_linear_bench --suite all
```

Running without arguments still prints usage and does not implicitly start a heavy suite. A suite is an explicit convenience, not a
correctness matrix, a complete artifact inventory, or a second production admission registry.

## 2. Typical suites

### 2.1 Default T sets

Suite entries use two fixed sampling classes:

| T class | Default T | Purpose |
|---|---|---|
| `Continuous` | `1,16,128,1024` | Observe the small-T bandwidth phase, the route transition, and the large-T compute phase together |
| `VisionStep4` | `4,128,1024` | A conservative common sampling that is legal for all current Vision geometries |

These points are not copies of production route boundaries, and make no attempt to cover every selector case. The full
continuous seam check is done by the explicit `--sweep`.

For each `(qtype,N,K)` the suite generates the weight only once and allocates the activation/output once, sized to that entry's maximum T. Different T values reuse the same set of allocations and do not rebuild the large weights.

### 2.2 `qwen3_6_27b`

The 27B core suite registers only the main Text/MTP/head geometries that the current model actually uses through pure public Linear:

| Label | QType | `(N,K)` | T class | Actual role |
|---|---|---:|---|---|
| `27b.output_head` | Q6 | `(248320,5120)` | Continuous | full output head |
| `27b.draft_head` | Q4 | `(131072,5120)` | Continuous | optimized proposal head |
| `27b.gdn_output_gate` | Q5 | `(6144,5120)` | Continuous | GDN Z row view |
| `27b.mtp_input` | Q8 | `(5120,10240)` | Continuous | MTP input projection |
| `27b.mtp_attention` | Q8 | `(14336,5120)` | Continuous | packed MTP Q/K/gate/V projection |
| `27b.mtp_gate_up` | Q8 | `(34816,5120)` | Continuous | MTP gate/up |
| `27b.mtp_down` | Q8 | `(5120,17408)` | Continuous | MTP down |

The 27B Vision suite registers the seven geometries actually called through public Linear:

| Label | QType | `(N,K)` | T class | Actual role |
|---|---|---:|---|---|
| `27b.vision_patch` | Q6 | `(1152,1536)` | VisionStep4 | patch projection |
| `27b.vision_qkv` | Q4 | `(3456,1152)` | VisionStep4 | attention QKV |
| `27b.vision_attn_out` | Q5 | `(1152,1152)` | VisionStep4 | attention output |
| `27b.vision_fc1` | Q4 | `(4304,1152)` | VisionStep4 | MLP expansion |
| `27b.vision_fc2` | Q5 | `(1152,4304)` | VisionStep4 | MLP contraction |
| `27b.vision_merger_fc1` | Q8 | `(4608,4608)` | VisionStep4 | merger expansion |
| `27b.vision_merger_fc2` | Q8 | `(5120,4608)` | VisionStep4 | merger output |

The packed/fused parents of base Text Attention, GDN input, and MLP are not added to the
pure Linear suite merely because they exist in the artifact. They are currently owned by separate-semantics Ops such as Attention/GDN/LinearSwiGLU/LinearAdd,
and should be measured in the corresponding Op benchmarks.

### 2.3 `qwen3_6_35b_a3b`

The 35B-A3B core suite registers the main head, MTP, and
DFlash geometries that are genuinely used through pure public Linear:

| Label | QType | `(N,K)` | T class | Actual role |
|---|---|---:|---|---|
| `35b.output_head` | Q6 | `(248320,2048)` | Continuous | full output head |
| `35b.draft_head` | Q4 | `(131072,2048)` | Continuous | optimized proposal head |
| `35b.mtp_projection` | Q8 | `(2048,4096)` | Continuous | MTP input/output projection geometry |
| `35b.dflash_feature` | Q8 | `(2048,16384)` | Continuous | DFlash conditioning projection |

The 35B-A3B Vision backbone shares the first six geometries with 27B, and the merger output goes into the 2048 hidden:

| Label | QType | `(N,K)` | T class | Actual role |
|---|---|---:|---|---|
| `35b.vision_patch` | Q6 | `(1152,1536)` | VisionStep4 | patch projection |
| `35b.vision_qkv` | Q4 | `(3456,1152)` | VisionStep4 | attention QKV |
| `35b.vision_attn_out` | Q5 | `(1152,1152)` | VisionStep4 | attention output |
| `35b.vision_fc1` | Q4 | `(4304,1152)` | VisionStep4 | MLP expansion |
| `35b.vision_fc2` | Q5 | `(1152,4304)` | VisionStep4 | MLP contraction |
| `35b.vision_merger_fc1` | Q8 | `(4608,4608)` | VisionStep4 | merger expansion |
| `35b.vision_merger_fc2` | Q8 | `(2048,4608)` | VisionStep4 | merger output |

The 35B routed expert `[262144,2048]` gate/up and `[524288,512]` down are not added to the pure Linear
suite. They belong to the execution contract of `sparse_moe`, and are currently not geometries registered with the pure Linear selector.
The Attention/GDN/DFlash layer projections likewise remain the responsibility of their fused Op
benchmarks.

### 2.4 `all`

`all` is the union of the two model suites. Points with exactly the same
`(qtype,policy,N,K,T)` are executed only once; a shared Vision point is not
timed twice just because both targets use it. The result may keep multiple role labels, but there is only one timing row.

## 3. Request pipeline

```text
CLI
 |
 +-- exact point: qtype + policy + N + K + T
 |
 +-- sweep:       qtype + policy + N + K + T range
 |
 +-- suite:       curated target entries + entry-local default T class
 |
 v
expand to BenchPoint(qtype, policy, N, K, T, labels)
 |
 v
group points by (qtype, policy, N, K)
 |
 v
allocate and initialize one packed weight
allocate x/out for max T in the group
 |
 v
for each T:
    bind [K,T] and [N,T] Tensor views
    warm up
    flush L2 outside timed interval
    time public ops::linear()
 |
 v
derive model bytes, useful FLOPs, fixed-spec ratios and measured-read ratio
 |
 v
compact table and optional CSV
```

Suite expansion only provides convenience points; execution still goes through exactly the same fixture, public API,
timer, and metric computation as an explicit single point, with no suite-specific dispatch.

## 4. Theoretical traffic

Let:

```text
K_pad = align_up(K, 128)
groups = N * K_pad / group_size
```

The format weight bytes are the sum of the storage planes the kernel needs to consume, excluding gaps between planes that exist only for start
address alignment and are never read:

| QType | group | bytes/group | weight bytes |
|---|---:|---|---:|
| q4_g64_fp16 | 64 | `32 code + 2 scale` | `groups * 34` |
| q5_g64_fp16 | 64 | `32 low + 8 high + 2 scale` | `groups * 42` |
| q6_g64_fp16 | 64 | `32 low + 16 high + 2 scale` | `groups * 50` |
| q8_g32_fp16 | 32 | `32 code + 2 scale` | `groups * 34` |
| NVFP4 | 16 | `8 code + 1 E4M3 scale` | `groups * 9` |
| BF16 | — | direct BF16 | `2 * N * K` |
| fp8_e4m3fn_row_bf16 | one row | `K code + 2 scale` | `N * (K + 2)` |

The theoretical minimum traffic of one Linear is:

```text
model_bytes = weight_bytes + 2*K*T + 2*N*T
effective_GB/s = model_bytes / seconds / 1e9
dram_spec_pct = effective_GB/s / dram_gbs * 100
read_ceiling_pct = effective_GB/s / read_gbs * 100
```

This is the model floor obtained from the public representation, not the physical DRAM traffic observed by a profiler. It does not count:

- cache-line transaction amplification;
- route-private rereads of the same weight tile;
- intermediate traffic of workspace, split/reduce, or composite launchers;
- the reduction in DRAM traffic due to cache hits.

These questions need NCU to answer. The benchmark no longer derives
`weight_replay_lower_bound_bytes` from the launcher column tile.

The percentages divide by the running GPU's roofs (`bench::device_roofs()` in
`bench/ops/ninferno_bench_common.h`): `DRAM_%` by the memory clock times the bus width, and
`READ_%` by the pure-read ceiling `tools/hbm_bandwidth_probe.cu` measured on that GPU class (93 %
of the DRAM figure on a class without a measurement). `READ_%` is the achievable utilization of a
read-dominated Linear; it does not change the `model_bytes` convention of one logical weight read
pass. The copy probe reads and writes at once and is not the ceiling for this kind of GEMV.

## 5. Math workload and route-neutral metrics

All weight types use the same mathematical GEMM workload:

```text
useful_flops = 2*N*K*T
useful_TFLOP/s = useful_flops / seconds / 1e12
```

Dequantization, bit decode, padding, repeated split-K work, or tile rounding are not added to
`useful_flops`, and `executed_tflops` is not derived from the private schedule. This ensures that different implementations are compared against the same
mathematical workload.

`AllowA4` is a permission, not an actual activation-compute profile; different exact
geometries and T under the same policy may select different routes. The uniform fixed-spec reference is the memory floor; only points explicitly associated with an actual
MMA path output `tensor_profile`, `tensor_peak_tflops`, and `tensor_peak_pct`, and the rest are left blank.
These fields must not be inferred from the activation policy alone:

```text
memory_floor_us = model_bytes / dram_gbs
memory_floor_pct = memory_floor_us / median_us
```

## 6. Timing contract

Ordinary single-point, sweep, and suite runs use the same cold-cache timing contract:

1. Allocation, payload initialization, and the first public call are not timed;
2. Before each sample, L2 is evicted with a fixed 256 MiB buffer;
3. The flush completes before the CUDA event start;
4. The events bracket exactly one complete public Linear call;
5. The primary result uses the median, and min and p95 are also recorded;
6. Default warmup is `3` and repeat is `20`, with explicit override allowed;
7. No separate warm-cache measurement is run for each point.

If an independent hot-cache question does arise later, a clearly named opt-in mode should be added; the doubled cold/warm
work must not again become the default behavior of every command.

## 7. Output

The console header always prints, for example on an RTX 5090:

```text
# gpu=NVIDIA GeForce RTX 5090 sm=120 roofs=gb202:170sm (GB202 table, probe read)
# dram_spec_gbs=1792.0 sustained_read_gbs=1674.5 cache=cold
```

A single result row keeps:

```text
label qtype policy N K T median_us min_us p95_us
effective_GB/s DRAM_% READ_%
useful_TFLOP/s tensor_profile tensor_peak_pct memory_floor_pct t1_linear_extrapolation
```

The sweep additionally outputs the `delta_%` relative to the adjacent T. The CSV may add the weight/x/out byte breakdown, warmup,
repeat, and environment versions, but does not restore the following fields:

- measured stream-copy ceiling;
- measured Tensor Core ceiling;
- candidate;
- kernel variant;
- inferred weight replay;
- inferred executed FLOPs;
- duplicated launcher tile metadata.

The benchmark does not need to claim which kernel was selected. A single-point NCU run can see the real kernel instance, and the production
selector source is the sole executable authority for the host launcher route.

## 8. Registration rules

### 8.1 New production shape

Adding a new production Linear shape does not require modifying the benchmark:

1. The selector and the public numerical test complete registration;
2. It can immediately be measured with the numeric `--n/--k/--t/--qtype`;
3. It is added to a model suite only when it becomes a high-frequency or architecturally representative geometry of 27B/35B.

A suite entry contains only:

```text
label, qtype, policy, N, K, T class
```

It must not carry launcher, schedule, kernel, tile, Full/Predicated, workspace, or route-boundary
metadata.

### 8.2 New route

Adding or replacing a host launcher does not modify the benchmark. Single, sweep, and suite always call the public
Linear, and therefore naturally measure the production route currently returned by the selector.

Candidate selection follows the recommended workflow defined in [`op-development.md`](op-development.md#7-performance-evidence):

1. Within the development task scope of that format or route, write a temporary benchmark/sweep, calling the private
   launcher directly when necessary;
2. Under the candidates' common legal domain, identical inputs, and cache/timing conditions, determine a fixed winner or crossover;
3. Write the decision only into the production selector;
4. Re-verify correctness and final performance through the public Linear;
5. Delete the temporary sweep, candidate forcing, and private entry points retained only for comparison.

Generic candidate forcing, private legal domains, crossover mirrors, launcher catalogs, or the
plan layer must not be reintroduced into the long-lived pure Linear benchmark. The public `--sweep` only observes the overall behavior of the final selector
and does not take on candidate selection.

### 8.3 New weight/activation compute type

A new benchmark type is added only when all of the following are done:

1. The public policy corresponds to a truly reachable production path;
2. The packed-weight fixture supports that persistent format;
3. The model-byte formula is explicit;
4. The public numerical suite has been qualified against the routes actually reachable under that policy.

A policy is only a permission, and the long-lived benchmark does not pass the permission itself off as low-precision execution. All current preset suites
explicitly use `A16Only`; NVFP4 AllowA4 is kept as explicit points of numeric geometries.

The benchmark does not take on numerical correctness; each weight/activation-compute profile continues to be covered by the public
Linear conformance suite and the unified CPU FP64 GEMM oracle.
