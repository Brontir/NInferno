# Kernel proposals

Every new kernel, or new variant of a kernel, starts here: one entry per proposal, written before
the code. This is the register of #619 (decision K4): a kernel is built only where a counted law
explains the current kernel's time, and its gain is predicted from facts before any micro-bench.
The [gap map](gap-map.md) names the regimes; this page holds what is proposed for them and what
the measurements then said.

## The rule

1. **Regime.** Name it the way the gap map does: the op, the format, the T band, the tile, the
   boards. Cite the gap-map rows or committed measurements that show the gap.
2. **Law.** The counted law that explains the current kernel's time in that regime: which term
   binds (DRAM bytes, tensor-pipe cycles, issue, latency, launch), counted from the kernel-facts
   snapshots (`src/ops/generic/data/kernel_facts/`) and the chip facts
   (`src/core/data/gpu_facts.json`, `src/core/data/isa_facts.json`). A gap no counted law explains
   yet is no proposal yet: the entry records the law that is missing, and the variant waits for it.
3. **Prediction, frozen.** The variant's time per regime cell, from the same facts. Write it as a
   range with its sources, before the micro-bench or A/B, and do not change it after a run.
4. **Acceptance and refutation, frozen.** These are fixed before the A/B:
   - what counts as the predicted gain (for example, at least half of the lower edge);
   - what refutes it;
   - the no-regression bound outside the regime;
   - the byte rule (`exact` bit-identical, or the KLD bound).
5. **A/B.** The A/B runs ABBA on the boards named, with its baseline, workload, GPU and method
   stated. Report predicted against measured, cell by cell. A cell under half its
   predicted gain, or over the upper edge, is a finding: it names the term that was wrong, from ncu
   on that cell, and the law is fixed. A constant is never fitted to close the gap.
6. **Decision.** Record it on the entry: kept (with its price in the cost model), narrowed to the
   cells where it held, or removed. The entry stays, with its numbers, either way.

## Entry template

```
### <id>. <name> (#issue, #PR)
- Regime: <op, format, T band, tile, boards; gap-map rows>
- Weight: <its share of the benchmark mix, where the mix is known for it>
- Law: <the binding term of today's kernel and its count, with the facts it reads; its residual>
- Variant: <what changes in the kernel>
- Prediction (frozen <date>): <time change per cell and arch, with sources>
- Prototype: <the cheap first measurement, as a command>
- Cost: <what this variant costs beyond the cost every new route has (below)>
- Acceptance / refutation (frozen): <bounds>
- A/B: <boards, protocol, predicted vs measured, or "not run">
- Decision: <kept / narrowed / removed / open>
```

Every new route, whatever its family, costs (#619 section 4):
- a conformance point in `test_generic_linear` or the family's `test_*_a16`;
- a `test_route_classes` summation class, and `same_summation` / `route_is_token_invariant`
  entries where the arithmetic differs by tile;
- the selector goldens, regenerated on three stands per arch set;
- `--build-manifest` and `test_prepared_rules`;
- `decision_inputs` where it reads a device fact.

A new family also costs a `Family` enum entry, which every route table parses, and a
`route_resources` branch.

## Register

Entries A-F and M come from the candidates table of #619 (section 4), G from #849, N from #856,
P from #850's plan. A, G, N and P have frozen predictions so far.

### A. Pre-summed activation groups for the min-term formats (#619 steps 5-6, #709)
- Regime: the register MMA on the min-term formats, T >= 17; Gemma heads and SwiGLU. The formats
  are those codec_traits marks `min` (Q4_K, Q5_K, Q4_1, Q5_1) or `half_min` (Q2_K).
- Law: the ones-MMA doubles an m1 tile's tensor work: 16.7 against 8.4 cycles per counted MMA on
  the RTX 5090 (#501).
- Variant: a plane of T x K/G FP32 sums (G the format's group: 32, and 16 for Q2_K), written once
  per call by an activation pass (as the W4A8 quantizer is). Every tile reads the plane in place of the ones-MMA, which keeps one summation
  order at every T and tile.
- Prediction (frozen in #709's body before any run):
  - prefill, narrow tiles at T 24-191: −5..−30 % by board;
  - heads and SwiGLU tiles at T 32-256: −10..−50 %;
  - the pre-pass costs +1.7-2.5 µs per call;
  - the speed profile and `exact` are unchanged.

  A different variant, all-ALU sums, measured −5..−12 % (heads) and −8..−18 % (SwiGLU) at
  T 17-256 in an earlier ABBA, and lost at T <= 16 (+4..23 %).
- Prototype: `ninferno_route_tune` ABBA on the 5090 and 3060, Gemma head and SwiGLU at T 17-64.
- Acceptance / refutation (frozen): #709's body, items 1-5.
- A/B (12.11, #709): ABBA of master against the branch, `ninferno_route_tune --prefill-only 1
  --rotate 1 --routes-out` at chunks 16, 34 and 48 (T 8-256), three passes (6 + 6 runs a cell),
  medians; the prefill profile's admitted cells only (RTX 5090: Q2_K store and add; RTX 3060:
  Q4_K add), shapes 4096x4096, 4096x12288 and 3840x15360. The timer ticks 1.024 µs, so the
  RTX 5090's cells were also timed per kernel with nsys and read with ncu. Against the frozen
  protocol: no Ada board ran, only the prefill profile was run (the speed profile reads no plane),
  and 3840x15360 (Gemma 4 12B's down projection) is not in the frozen shape list.
  - RTX 3060, Q4_K add: the fastest route −7..−20 % at T 17-256 against the predicted −10..−30 %;
    item 1 holds at T >= 17 in every cell. Its frozen "T = 8 within ±1 %" clause fails: T = 8
    moved −10..−15 %, because the prefill profile reads the plane below 17 tokens too. Item 2
    fails: the m2.n2 plane instance runs +3..+10 % over its twin in 11 of 24 cells.
  - RTX 5090, Q2_K store and add: the fastest route 0..−16 % at T 8-17 and 128-256, +8..−12 %
    at T 24-32 (mostly under half of the predicted −8 %), +4..+14 % at T 48 (over the 1.03 bound),
    0..+3 % at T 64. Items 1 and 2 fail: m1.n1 runs +3..+18 % over its twin at T 48-256 (+8 % at
    T 24 on the store), m1.n8 and m2.n1 +3..+9 %.
  - Finding, the term that was wrong: the law took the ones-MMA's tensor work as binding. On the
    RTX 5090 (ncu, Q2_K store 4096x4096, T 24-128) tensor-active is 14-41 % on master; the plane
    halves it, but long-scoreboard stalls per issue rise (m1.n1 1.0-1.7 to 2.3-2.4, m1.n8 2.1-2.4
    to 3.5-5.9): the plane's group loads lengthen the K loop's load chain, and the narrow tiles'
    kernel gets slower (nsys, Q2_K add 4096x12288: m1.n1 +6..+20 % at T 48-128). The pre-pass
    costs 0.8 µs at T 8 to 2.9 µs at T 128 (K 12288): it grows with T·K/G, below the predicted
    1.7-2.5 µs at small T and above it from T 96.
  - Items 4-5 (e2e TTFT, KLD) were not run: items 1-2 already refute the prediction.
- Decision (12.11): off by default, the code kept. `rmma_group_sums` answers false unless
  `set_group_sums_plane_enabled` turns it on: the plane's tests, and `ninferno_route_tune
  --group-sums-plane 1` for its next A/B (no environment variable decides it, principle 3). The
  prefill profile needs one answer per format, epilogue and device, so the plane cannot be
  narrowed by width; the RTX 3060's Q4_K add cell held per cell but failed item 2, and no fact
  separates it from the RTX 5090's refuted cells. The plane comes back with a law for the narrow
  tiles' load chain (step 6 prices it), and a new A/B.

### B. Blackwell MoE expert kernel at 4-8 rows (#619 step 11)
- Regime: the routed experts on GB20x at 4 and 8 rows. They run at 0.74 and 0.63 of the DRAM-bound
  speed, against Ada's 0.91 and 0.81 (DECISIONS 65). The 5070 Ti IQ3_XXS estimate is 7-13 % slow
  at 2-6 rows.
- Law: none yet. The gap map lists MoE under "No law yet". The loss may come from the expert grid
  or from the GB20x tensor rate rather than from the kernel.
- Prototype: `moe_bench` (`r8vs.sh`) and nsys on the 5090 and 5070 Ti at C = 4 and 8.
- Decision: law first, then decide.

### C. Turing int8 tile: dense `w4a8_kernel` and MoE `moe_w4a8` (#619 steps 7-8)
- Regime: int8 prefill on sm_75: 30-34 SM clocks per K, against 12 on Ada. MMQ Q4_K reaches 22 % of
  peak there, against 37 % on Ada.
- Law (partial): Turing's int8 MMA is m8n8k16 only, half the shape of sm_80's m16n8k32, so the facts
  predict about 2x the MMA issues. The measured 2.5-2.8x leaves 30-40 % unexplained.
- Prototype: ncu of `w4a8_kernel` at K 4096 on the 2080 Ti (stalls, issue, ldmatrix count).
- Decision: open. The study (step 7) writes a counted-instruction prediction before the kernel
  (step 8).

### D. Gemma 4 int8 KV plane for sm_75/80/86 (#619 steps 9-10)
- Regime: Gemma 4 attention below sm_89 (`kGemma4Fp8Hardware = arch >= 89`). The KV stays BF16
  there, which means twice the bytes and half the context.
- Law: long-context decode is bytes-bound on Ampere (1.0-1.16 of DRAM time). On Turing it is
  SM-bound, so the gain there is capacity, not speed.
- Variant: an int8 plane through the FP8 plane's "codes widened in registers" mechanism
  (`fp8_row.cuh`).
- Prototype: `gemma4_attention_bench` on the 3060 at 24K, then `mdecl.sh`.
- Decision: open. It is a B1 byte class, so the choice must be a fact of the arch.

### E. Narrow-T (17-64) register-MMA load chain: a deeper ring or split loads (#619 step 12)
- Regime: the register MMA at T 17-64. On four cards the winner is latency-bound at 2-4 warps per
  SMSP, with DRAM at 28-50 % and tensor at 25-63 %.
- Law: the bytes one SM keeps in flight against its DRAM latency (Little's law, #690). The
  per-architecture rate and bytes-in-flight facts that `tools/bench/warp_inflight` measures count
  it. #735 (open, draft) is to wire them into the price.
- Risk: the winning schedules sit at 245-255 registers, so a deeper ring spills.
- Prototype: `rmma_kernel` m2.n4.w4x1 with ring depth 3, `route_tune --rotate` on the 4090 and
  3060 at T 24 and 32.
- Decision: prototype after A; adopt only with the law.

### F. Gated gate/up twin at T = 3-4 (vec kernel)
- Regime: gated launches at C = 4. The 3060 still loses 3-4 % there; the 5060 Ti is at parity.
- Law: the vec kernel's bounded twin excludes gated launches (`kVecHasBoundedTwin`): they spill
  under 72 registers. A twin needs the gate and up rows on separate warps. The gain is small, on one
  class (SMs x clock / bandwidth ~0.17).
- Prototype: `tools/bench/run_521_gateup.sh` (#606, not run).
- Decision: run the kit first. Build only if the loss is above 3 % and #606's rule holds on the
  3090.

### G. NVFP4 KV decode attention: register decode on warp-private key slabs (#849)
- Regime: causal attention over the NVFP4 G16 KV cache at decode, the grouped calls whose query
  rows fit one n8 tile (W = 1 on every geometry, W = 2 at four query heads per KV head), sm_80
  and later. The census (#618, `ninferno_kv_decode_attention_bench --codec nvfp4`, the slope
  between 7843 and 21961 keys) reads the KV at 2.2-2.8 B per SM clock: 64 % of the DRAM line on
  the RTX 5090, 57 % on the 5070 Ti, about 80 % on the 4070 Ti and 4090, 30-34 % on the 3090 and
  3060. INT8's kernel, with the same warps, keys and occupancy, reads at 90-92 %.
- Weight: the KV term of a Qwen3.5-9B decode step, 1152 B per token per full-attention layer:
  0.8 % of a 4.1 ms step at 22K keys, about 6 % at 128K, times the streams.
- Law of today's kernel (`nvfp4/grouped_mma.cuh`, one CTA of 8 warps per 32-key tile, 2 CTAs per
  SM, 9216 B a tile). Counted from `cuobjdump -sass` of the CUDA 13.2 build
  (`nvfp4_kv_grouped_mma_kernel<CausalGeometry<256,16,4>, T = 1>`), cycles from `isa_facts.json`:
  - four `__syncthreads` per tile;
  - K decode: two 16-value groups per thread. A group costs 166 instructions on sm_86 (the
    bit-form E2M1, the E4M3 scale in software, two 16-byte stores of FP16), 125 on sm_89 (the
    scale by cvt) and 50-78 on sm_120a (its hardware E2M1 cvt). That is 2650, 2000 and 800-1250
    warp instructions a tile, mostly on the 2-clock integer pipe: 1000-2650 and 400-800 clocks
    with the other CTA. V decode costs the same.
  - QK: 64 HMMA m16n8k16 on warp 0 alone (`ProducerWarps = RowTiles = 1`), 2048 clocks of one
    partition's tensor pipe (32 clocks each), while seven warps wait at the barrier after it;
  - PV: 8 HMMA a warp, 512 clocks;
  - the next tile's `cp.async` goes out only after the V decode, so 200-1150 clocks of DRAM
    latency (540 sm_86, 690 sm_89, 837 sm_120a, w8) show every tile;
  - sum: 5300-8900 clocks a tile on sm_86/89, 4200-5700 on sm_120a, so 2.1-3.5 and 3.2-4.4
    B/SM-clk. Measured: 2.2 on the 3090, 2.6 on the 4070 Ti, 2.8 on the 5090. ncu on the three
    generations has the barrier as the main stall (46 % of samples on the 4070 Ti), and DRAM bytes
    equal the KV bytes (3090).
  - The residual on sm_120a: when the QK warps of the two resident CTAs share a partition, their
    2048-clock sections serialize, which gives 2.5-3.0. The rate does not move with the
    generation, because the serial QK, the latency and the barriers are 60-75 % of the tile.
- Variant: `nvfp4/slab_mma.cuh`, one CTA of 8 warps per SM. Every warp owns a 16-key slab of
  every 128 keys, with a two-stage `cp.async` ring of its own (2 x 8 x 4608 B = 72 KiB), waited
  with `cp.async.wait_group` and `__syncwarp`: there is no CTA barrier in the loop. S^T = K Q^T,
  with the 16 keys as M and the query rows as one n8 tile, so Q's B fragments stay in registers.
  K and V go from `ldmatrix` of the raw code tile straight into FP16 MMA fragments: in sm_100+
  family-specific builds (`NINFERNO_ARCH_FAMILY_SPECIFIC`) one `cvt.rn.f16x2.e2m1x2` a pair;
  elsewhere the bit form at 2^-14 of the value, restored exactly in FP32. Then one HMUL2 by the group's E4M3 scale per pair. The V^T and P^T fragments
  are transposed by `movmatrix`. Each warp keeps its own online softmax, and the 8 states merge
  once per CTA. The plan's wave partition counts one CTA per SM (`slab_rule.h`). The aligned
  launches of the invariant profile, RowTiles >= 2, rows 9-16, and sm_70/75 keep today's kernel.
- Counts of the variant (SASS, CUDA 13.2; one slab per warp; the loop is identical in every
  instance):
  - static loop: 1238 instructions on sm_86, 1171 on sm_89, 921 on sm_120a. That includes 64 FMUL
    of the softmax rescale, which a warp skips when no row's maximum moved, and about 12
    divergence stubs that never execute. So about 1160, 1095 and 853 issue a slab.
  - per slab: 32 HMMA, 66 MOVM, 8 LDSM, 9 LDGSTS, 10 LDS, 6 SHFL;
  - integer pipe: about 540 (sm_86), 470 (sm_89) and 320 (sm_120a). LOP3, SHF, PRMT and IADD3
    are the decode. ptxas moves the left shifts to IMAD.SHL on the FMA pipe: 263 IMAD on
    sm_86/89, plus 160 / 128 HMUL2;
  - sm_120a: 162 F2FP (128 E2M1 pairs, 32 scales, 2 P);
  - registers: 255 on sm_86/89, 226-238 on sm_120a, no local memory, 72 KiB dynamic and 4.5 KiB
    static shared memory, 1 CTA per SM.
- Capacity of the variant per SM and 128 keys (36864 B; two slabs per partition):
  - issue: 2 x 1160 = 2320 clocks on sm_86 (15.9 B/SM-clk) and 2 x 1095 = 2190 on sm_89 (16.8).
    sm_120a: 1706 clocks (21.6); with F2FP at 8 clocks a warp instruction (no fact yet), 2592
    (14.2);
  - integer pipe: 2 x 540 x 2.02 = 2180 clocks (sm_86) and 1900 (sm_89), under the issue term;
  - tensor: 2 x 32 x 32 = 2048 clocks (18.0) on every arch;
  - MIO: about 100 a slab. movmatrix has no measured rate; at 4 clocks it would still be under
    the issue term;
  - latency: two 8-long HMMA chains (264 clocks) and 37-74 KiB in flight per SM, against
    Little's need of 4-11 KiB (line x DRAM latency, w8 to lanes-chasing).
  - The binding term is DRAM on all six boards. The margin min(capacity) / line is 2.2 on the
    3060, 2.4 on the 3090, 5.2 on the 4070 Ti, 5.4 on the 4090, 2.7-3.4 on the 5070 Ti and 3.2-4.1
    on the 5090. The fixed part per CTA (the Q prologue, the first load, the last slab of an
    uneven split, the merge) does not grow with the context, and the census slope drops it.
- Prediction (frozen 13.11.2026, before any run): the census share of the DRAM line, from
  `--codec nvfp4 --context 7843,21961`, eager and graph:

  | card | today | predicted | KV term at 21961 keys |
  |---|---|---|---|
  | RTX 5090 | 64 % | 88-93 % | 130 µs -> 90-95 µs (call 207 -> 165-172 µs; the fixed 77 µs unchanged) |
  | RTX 5070 Ti | 57 % | 85-92 % | x 0.62-0.67 |
  | RTX 4090 | ~80 % | 88-93 % | x 0.86-0.91 |
  | RTX 4070 Ti | 80 % | 88-93 % | x 0.86-0.91 |
  | RTX 3090 | 30-34 % | 80-90 % | x 0.36-0.42 |
  | RTX 3060 | 30-34 % | 78-88 % | x 0.37-0.44 |

  Ampere's lower edge comes from its issue margin of only 2.2-2.4x (the bit-form decode). INT8's
  90-92 % is the ceiling the census allows. W = 2 at four query heads per KV head (8 rows, MTP
  verify) reads the same bytes per key at the same capacity, so it gets the same shares (the bench
  runs W = 1 only; the e2e MTP decode measures it). The RTX 2080 Ti (sm_75) and Volta are
  unchanged.
- Prototype: `ninferno_kv_decode_attention_bench --codec nvfp4 --context 7843,21961` (`--execution
  eager` and `graph`), ABBA master against the branch. `--activations exact` is the control, since
  aligned launches keep today's kernel. ncu `--set full` of `nvfp4_kv_slab_mma_kernel` on one card
  per generation: long scoreboard should lead, as INT8's does, with `alu` and `fma` pipe use below
  the issue term.
- Cost: one kernel file, its codec header and its occupancy rule. The plan's split partition
  follows the schedule's CTAs per SM. There are no route tables (the attention plan has none), no
  family and no selector change.
- Numerics: Q in FP16, and K and V the codec's FP16 values times an exact power of two, with FP32
  accumulation: the products are the same. The summation order changes: the k permutation inside
  an MMA, a softmax per warp over 16-key slabs, the 8-way merge. `kAttentionNvfp4Criterion` is
  unchanged. The invariant (`exact`) profile keeps today's kernel bit for bit.
- Acceptance / refutation (frozen):
  - accepted: every card at or above 85 % of its predicted share's lower edge;
  - refuted: any card under 75 % of the DRAM line, or the RTX 5090 KV term at or above 100 µs at
    21961 keys;
  - no regression: RowTiles >= 2, prefill, `exact` and sm_70/75 run today's kernel, and the
    attention tests pass with the criterion unchanged;
  - byte rule: the criterion, plus a KLD run of Qwen3.5-9B with NVFP4 KV against master on one
    card, which must be noise.
- A/B (13.11, PR #868 at 0ad1c881): `kv_decode_attention_bench`, Qwen3.5-9B geometry, 8 layers,
  slope over 7843 and 21961 keys, ABBA master against the branch. The controls (INT8, K8V4) did
  not move.

  | card | master | branch | predicted | verdict |
  |---|---|---|---|---|
  | RTX 3060 | 31.4 %, 36.2 % | 94.7 %, 94.4 % (x 2.8) | 78-88 % | above the upper edge |
  | RTX 4090 | 75.1 %, 75.0 % | 80.8 %, 81.1 % (x 1.08) | 88-93 % | below the range |
  | RTX 5090 | 64.7 %, 64.5 % | 89.1 %, 88.6 % (x 1.38) | 88-93 % | inside |
  | RTX 3090 | 36.4 %, 36.6 % | 85.4 %, 84.8 % (x 2.33) | 80-90 % | inside |
  | RTX 4070 Ti | 78.3 %, 78.1 % | 96.4 %, 96.6 % (x 1.23) | 88-93 % | above the upper edge |
  | RTX 5070 Ti | 58.6 %, 57.6 % | 95.5 %, 94.5 % (x 1.64) | 85-92 % | above the upper edge |

  The 5090, 3090, 4070 Ti and 5070 Ti rows are the 13.11 afternoon ABBA (kvab2, branch merged with
  master). The fixed part per call rose on two boards (RTX 3090 15.7 -> 19.2 us, RTX 5070 Ti
  10.0 -> 11.8 us) and fell on the RTX 5090 (11.8 -> 11.1 us); the slope gain outweighs it from
  about 1K keys up. The RTX 5090's KV term at 21961 keys is 27.0 us a call (frozen bound 100 us).

  - The RTX 4090 passes the frozen acceptance (at least 74.8 %, 85 % of 88 %). It gains +6
    points against a predicted +13 or more: under half the predicted gain, so it is a finding.
  - ncu on the RTX 4090 (21961 keys, SM clock 2.75 GHz):
    - one wave of 128 CTAs, one per SM, 2 warps per scheduler;
    - issue slots busy 13.9 %, eligible warps 0.20 per scheduler;
    - DRAM 70.6 % of the call;
    - 11.7 cycles per issued instruction, 7.6 of them short-scoreboard waits;
    - SM active 82 % of elapsed, so the SMs sit idle 18 % of the call;
    - no local memory.

    The slope rate is 2.3 B/SM-clk on the RTX 4090 against 6.8 on the RTX 3060.
  - Missing term 1, the shared-memory dependency chain with 2 warps per scheduler. Counted, and
    refuted as the binding term:
    - Method: one warp's in-order issue of one slab, from the loop's SASS control codes (stall
      counts, scoreboard waits), with `isa_facts` latencies for the variable-latency ops: LDS and
      LDSM at `shared_latency` (24.1 clk sm_89, 23.1 sm_86), MOVM the same (no fact), SHFL 27,
      HMMA 32.9, F2FP 17. The cp.async wait is taken as satisfied.
    - Result: 2493 clk a slab on sm_89 (1101 issued), 2574 on sm_86 (1168) and 2298 on sm_120a
      (857). About 730 of those are scoreboard waits, mostly the decode's first IMAD after the
      ldmatrix and the row-max shuffles. With MOVM at 64 clk instead, sm_89 takes 2639.
    - A partition's round of 128 keys (two warps' slabs) therefore takes max(chain,
      2 x issue) = 2.5-2.6 k clk. That is 14-15 B/SM-clk.
    - RTX 4090 at 2.75 GHz: its DRAM line is 2.86 B/SM-clk, a round of 12.9 k clk, so two warps
      cover the chain five times over.
    - RTX 3060: its line is 7.2 B/SM-clk, a round of 5.1 k clk, a margin of only 2x. The chain
      would bind the RTX 3060 first, yet the RTX 3060 reads at 94.5 %.
    - The chain predicts DRAM-bound on the RTX 4090 (88-93 %), not 81 %. The 7.6 short-scoreboard
      cycles per instruction are mostly warps waiting for data, not the chain itself. ncu's
      source view is needed to split them between the cp.async wait (`DEPBAR`) and the
      LDSM/LDS/MOVM results.
  - Missing term 2, the one-wave tail:
    - The plan runs one CTA per SM in a single wave (32 splits over 4 KV heads on 128 SMs), and
      a CTA that finishes early leaves its SM idle until the slowest one ends.
    - What the RTX 4090 shows: ncu's 18 % idle SM time with DRAM at 70.6 / 0.82 = 86 % while
      SMs are active. The 81 % census share is 0.82 x 0.96-0.99.
    - Splits differ by one slab (42-43). Rounding to whole 128-key rounds is about constant per
      CTA, and the slope drops it. So the spread is not the work: it is the time to stream a
      CTA's bytes, which varies between SMs under a saturated DRAM. It grows with the call and
      stays in the slope.
    - The RTX 3060 runs 7 splits per KV head on 28 SMs and loses about 1 %. The law has no fact
      for the spread on a 128-SM part. It is recorded as a term to count, not as a fitted
      constant.
  - Next candidate, G2: two waves of smaller splits at the same kernel. Not built in #868.
    - Variant: `causal_wave_partition` budgets 2 x SMs CTAs for the slab instances: two waves at
      one CTA per SM, each split half today's keys. A CTA that finishes early takes a second-wave
      split, so an SM idles only for the spread of half a CTA's bytes.
    - Cost: the per-CTA fixed part (Q prologue, first load, merge) doubles in count. It stays in
      the census intercept, not the slope, and the merge reads twice the partials.
    - Prediction (frozen 13.11.2026):
      - the RTX 4090 census share goes from 81 % to 86-90 % (idle 18 % -> 8-10 %);
      - the RTX 3060 stays at 93-95 %;
      - the RTX 5090 and 5070 Ti, at 170 and 70 SMs, get the same share of the gain as their
        idle fraction in a first ncu of #868 shows;
      - the call's fixed part rises by under 2 us per 8 layers.
    - Acceptance: RTX 4090 at 85 % or more with ncu idle under 10 %.
    - Refutation: idle unchanged (the spread is not the cause), or the RTX 3060 under 92 %.
- Decision (13.11): every card is at or above 85 % of its predicted lower edge and none is under 75 %
  of the DRAM line, so the frozen acceptance holds on all six boards. The band itself is refuted on
  four of them: under it on the RTX 4090, over it on the RTX 3060, 4070 Ti and 5070 Ti (the band
  assumed a DRAM efficiency below the 94-96 % INT8 and K8V4 reach on those cards). The ncu source
  view of the RTX 4090's short-scoreboard waits stays open.
- Byte rule, KLD (13.11, RTX 5090, Qwen3.5-9B, NVFP4 KV, ninferno-ppl-1m-v1 quick, context 4096,
  stride 2048, 261,167 tokens):
  - The branch against master: mean KLD 0.0043, top-1 97.67 %, PPL ratio 0.9999. Master against a
    second master run gives 0 (the decode is deterministic), so these are the slab's other FP32
    order, not run noise.
  - Each against the exact BF16-KV reference: master 0.010934 (top-1 96.92 %, PPL 1.0038), the
    branch 0.010923 (96.92 %, 1.0037). The branch is -0.1 % of master's, inside ±15 %. Both are
    within KLD_TOLERANCE (0.15) of the recorded NVFP4 KV term, 0.00982 (choice_constants.h).
  - The NVFP4 KV error is unchanged, so the byte rule holds.
- Decision (13.11): kept where it is dispatched (NVFP4 KV, at most 8 query rows a tile
  (`kNvfp4KvSlabMaxRows`), sm_80+, not `exact`); every frozen acceptance item holds. Width-1
  grouped append chunks, and width-2 chunks at G = 4, fit in 8 rows and so also take the slab; the
  identity test covers them.

### M. MMQ column widths per arch and format (#619 step 13, DECISIONS 74)
- Regime: MMQ on short prompts. Ada and Turing are done (#597, #717). Q2_K and the IQ formats still
  keep the fewest tiles.
- Prototype: the forced-J grid on the 4090 and 2080 Ti.
- Decision: open. These are data rows.

### N. BF16 KV width-1 decode attention without register spills (#856)
- Regime: `bf16_kv_grouped_mma_kernel`, the GroupedDecode instance (width 1, 16 packed query rows,
  32-key tiles, 2 KV warps), D = 256, every listed head geometry; decode with a BF16 KV cache.
  The #618 census put it at 53 % of DRAM on the RTX 3090, 50 % on the RTX 3060 and 42 % on the
  RTX 2080 Ti (#853), against 92 % on the RTX 4090.
- Law: register spills, and where their traffic goes. `__launch_bounds__(128, 2)` caps a lane at
  255 registers, the ISA maximum. Across the key loop a lane kept Q's fragments (16 K steps x 4 =
  64 registers) and O's accumulators (32 column blocks x 4 = 128); the tile's scores, P, K/V
  fragments, softmax state and addresses take the rest. ptxas (CUDA 13.2, the decode call:
  unbatched, unmasked, append) spilled Q to local memory on sm_86/89 (392 B a lane: per key
  tile 408 B of LDL and 120 B of STL a lane, 33.8 KB a CTA against the tile's 32 KB of K/V), on
  sm_75 (120 B; 112 B LDL and 68 B STL a tile) and in 8 of the 32 variants on sm_80 (496 B);
  sm_90/100/120a did not spill. A spilled line reaches DRAM when the device traffic between its
  store and its reload, one key tile on every resident CTA, exceeds the L2: RTX 3090
  82 x 2 x 65.8 KB = 10.8 MB > 6 MB, RTX 3060 3.7 MB > 2.25 MB, but RTX 4070 Ti 7.9 MB < 48 MB and
  RTX 4090 16.8 MB < 72 MB. The ncu run through the #862 bench agrees: the RTX 3090 read 1.50x the
  KV bytes and wrote 19 MB a call, the RTX 4070 Ti read 1.00x. The RTX 2080 Ti holds one CTA an
  SM (2 x 33.2 KB > 64 KB), and its window (68 x 43.5 KB = 3.0 MB) fits its 5.5 MB L2: there the
  spill is latency on a serial load-then-compute tile, not DRAM bytes.
- Variant: two D warps per key slice (128 threads; each scores the slice's keys and accumulates
  half of D: O 64 registers), and Q in shared memory (rows 0-7 only, 4 KiB in the MMA's A form,
  read per key tile). The packed rows are one token's G::GroupSize <= 8 heads, so a lane's second
  MMA row (gid + 8) holds no query; it enters the MMAs as zeros and its sums are dropped. The K/V
  tiles, the plan and the split count are unchanged, and so are the CTAs an SM holds (2 on
  sm_86/89/120, 1 on sm_75; sm_80/90 register-bound at 2-3, above the plan's 2 CTAs an SM).
  The other candidates, each compiled: a lower register cap (`MinBlocks` 3, 168) forces spills
  under a 192-register live set; D warps alone still spilled 136 B on sm_80/86/89; Q in shared
  memory with the dropped rows alone still spilled 64 B on sm_86/89 and 40-96 B on sm_75, since
  the m16n8k16 accumulator is a 4-register tuple and dropping its rows frees no registers; O in
  shared memory adds a 32 KB read-modify-write a tile. After (ptxas, all 32 GroupedDecode
  variants): no stack, no LDL/STL on sm_75/80/86/89/90/100/120a; 154-158 registers on sm_75,
  122-223 on sm_80/86/89, 127-132 on sm_90, 168-184 on sm_100/120a. Every other kernel of the
  translation unit compiles to the same SASS. The kept rows' bits are those of master: the same
  operands, MMAs and order (an MMA row's sums read only that row of A).
- Prediction (frozen 2026-11-13), Qwen3.5-9B, 21,961 keys, one layer (89.95 MB of KV), the call's
  KV term; DRAM from `gpu_facts.json`:

  | card | DRAM GB/s | master | predicted | gain | source |
  |---|---|---|---|---|---|
  | RTX 3090 | 936 | 192 us | 107-120 us | 1.6-1.8x | 154.6 -> 90 MB at ncu's 805 GB/s |
  | RTX 3060 | 360 | 503 us | 272-293 us | 1.72-1.85x | INT8's 91.8 % share .. the 3090's byte ratio |
  | RTX 2080 Ti | 616 | 351 us | 280-335 us | 1.05-1.25x | DRAM bytes unchanged; tile compute 0.75-0.85x |
  | RTX 4070 Ti | 504 | 203 us | 197-206 us | 0.99-1.03x | DRAM bytes unchanged (spill in L2) |
  | RTX 4090 | 1008 | 97 us | 94-99 us | 0.98-1.03x | no spill reached DRAM |
  | RTX 5070 Ti | 896 | 118 us | 115-122 us | 0.97-1.03x | no spill before |
  | RTX 5090 | 1792 | not run | +-3 % | 0.97-1.03x | no spill before |

  The 2080 Ti's 1.49x of #853 (INT8's share) is not predicted by this change: with one CTA an SM
  a tile's load and compute are serial; it needs a second CTA an SM or a second K/V stage.
- Prototype: `ninferno_kv_decode_attention_bench --codec bf16` (#862) on each card, master
  against the branch; ncu of the 3090 call for its DRAM bytes.
- Cost: none beyond the instance: no plan, split or price change. Once measured, the ga10x BF16
  rows of `kKvSmRate` (#861) become a read factor of about 1.0.
- Acceptance / refutation (frozen): the bench's read rate (the slope), ABBA, master against the
  branch.
  - RTX 3090 and RTX 3060: at least 1.35x. Refuted below it, or when ncu reads more than 1.05x the
    KV bytes a call.
  - RTX 2080 Ti: at least 1.025x. Over 1.25x is a finding: the tile's compute term was wrong.
  - RTX 4070 Ti, 4090, 5070 Ti, 5090: within 0.97-1.03x.
  - Bytes: `ninferno_softmax_attention_bf16_test` passes, and
    `ninferno_bf16_decode_identity_test` finds the output, the split partials and the appended
    cache byte-identical to the former instance (one D warp, compiled in the test; its SASS is
    master's).
- A/B: not run.
- Decision: open.

### P. R8 decode schedule of the wide-K Q6_K down projection (#850 candidate (a))
- Regime: the R8 decode kernel (`r8_kernel`, A8) at T = 1, store and add epilogues: Qwen3.5-9B's
  down projection, Q6_K 4096x12288. On the boards with native BF16 MMA (sm_80 and up) `rules_r8`'s
  K rule (K >= 8192) picks `w8r2`; on the RTX 2080 Ti its Turing Q6_K rule picks `w4r2`. No R8
  decode table has this problem. The #850 census (RTX 4090 ncu, master 3f20b638c): 48.7 us for
  41.3 MB (849 GB/s, 84 % of DRAM), 256 CTAs of 256 threads at 209 registers, 16 % occupancy; RTX
  5090 nsys: 29.3 us. Both sit 7-8 % above the line the Q4_K calls of the same step define.
- Weight: one call a layer, 32 a 9B step; 1-2.5 % of the C = 1 step (#850 plan, section 4).
- Law (#850 plan, sections 1.2-1.5): per call, t = t_node + t_ramp + L_first + B / BW_eff +
  min(R - 1, 3) b_round + t_partial + t_tail + t_imb. On the RTX 4090 it holds to 0.6 us on six
  of the step's eight calls with no fitted constant. At `w8r2` the registers (209 x 256 threads)
  leave one CTA a SM, so the 256 CTAs are two a SM on the 4090 (no dynamic balancing), 1.5 on the
  5090 (a second round on half the SMs) and 3.1 on the 3090 (a last round of 10 CTAs). The residual
  of this call, 3.6 us on the 4090 and 2.5 on the 5090 (8-9 % of the stream on both), is named
  t_imb: a hypothesis, not yet a fact. ncu `sm__cycles_active.max` against `.avg` on this call
  decides it (#850 PR-1). Bytes in flight do not bind (x8-50 margin, x2.3 on the 5090).
- Variant: no kernel. The compiled schedule `w8r1` on sm_80 and up: 120 registers, two CTAs a SM,
  512 CTAs of 80 KB. On Turing `w4r1`: 126 registers, four CTAs a SM, 1024 CTAs of 40 KB. Both keep
  the bytes in flight a SM, halve (quarter) the work quantum an SM can be left with, and put 16
  warps on a SM. Every R8 schedule sums a row in one order (lane l takes block groups l, l + 32,
  ...; generic_r8.cuh), so the change is bit-identical: `check_r8_schedules_agree` holds it at
  K = 12288 in `test_generic_linear`, and the sweep's `same=1` holds it on every row.
- Prediction (frozen 13.11, before any sweep; change per call, us, candidate minus the rules'
  schedule). The bands are the plan's, re-derived below; they are the acceptance basis:

  | card | today | candidate | band | this entry's re-derivation |
  |---|---|---|---|---|
  | RTX 4090 | 48.7 measured (`w8r2`) | `w8r1` | -2.0 .. -4.0 | -1.7 .. -3.5 |
  | RTX 5090 | 29.3 measured (`w8r2`) | `w8r1` | -0.8 .. -2.2 | -1.0 .. -2.3 |
  | RTX 3090 | 56.4 law (`w8r2`) | `w8r1` | -3 .. -7 | -3.0 .. -4.9 |
  | RTX 3060 | 128.7 law (`w8r2`) | `w8r1` | -1 .. -4 | -2.3 .. -3.5 |
  | RTX 2080 Ti | 78.2 law (`w4r2`) | `w4r1` | -1 .. -4 | -1.4 .. -2.8 |
  | RTX 4070 Ti | not priced by the plan (`w8r2`) | `w8r1` | -0.5 .. -2.0 | (this entry's) |
  | RTX 5070 Ti | not priced by the plan (`w8r2`) | `w8r1` | -0.5 .. -2.0 | (this entry's) |

  Where this entry differs from the plan:
  - A `w8r1` CTA has half a `w8r2` CTA's bytes at half the SM's share, so it lasts as long. What
    halves is the work an SM can be left with at the end, so t_imb halves; it need not vanish. The
    central estimate is therefore the band's lower edge: 4090 -1.7 (half of 3.6, less 0.08 us of
    ramp for the 512 CTAs), 5090 -1.0 (half of 2.5, less 0.2 us of ramp).
  - The plan's 3090 and 3060 laws count the last partial round in full, although its bytes are
    already in B / BW_eff. Only the round's time beyond its bytes' share is extra. 3090: 10 CTAs of
    161 KB at 10 x 54 GB/s take 3.0 us, of which 1.8 are in the stream, so +1.2. 3060: 4 CTAs, 3.0
    against 1.9, so +1.1. At `w8r1` the last round's CTAs pull more than the DRAM gives (3090: 20 x
    54 GB/s = 1.08 TB/s > 896 GB/s; 3060: 8 x 54 > 344), so its extra is 0, not the plan's 1.5
    and 1.4. With t_imb half to all: 3090 -3.0 .. -4.9, 3060 -2.3 .. -3.5. A 3090 beyond -5 would
    say t_imb is larger than the 4090's share carried over.
  - RTX 2080 Ti: the last round of `w4r2` (104 CTAs on 136 slots) already pulls more than the DRAM,
    so there is no partial term. Only t_imb moves: 2.8 to 1.4, so -1.4 .. -2.8.
  - RTX 4070 Ti (60 SMs, 504 GB/s, 482 effective: a 85.7 us stream) and RTX 5070 Ti (70 SMs,
    896 GB/s, 840 effective at the 5090's stream share: 49.2 us): `w8r2` gives 4.3 and 3.7 CTAs a
    SM. Their last rounds (16 and 46 CTAs) pull more than the DRAM, so there is no partial term.
    t_imb of a multi-round grid: between the 3060's 2 % of the stream (nine rounds) and the 4090's
    8 % (two), halved by `w8r1`. Hence -0.5 .. -2.0 on both. These two cards check the rule
    below more than they test a gain.
  - `w4r1` on sm_80 and up: a quarter of the grain, but twice `w8r1`'s activation prologue (3 blocks
    a thread against 1.5) and four times its activation reads from L2 (25 MB a call on the 4090).
    It is predicted within 1 us of `w8r1`; the summary prints it, and it is not the candidate.
  - Q4_K and Q5_K 4096x12288 (the same call in other mixes): the same direction where `w8r2` holds
    one CTA a SM. No snapshot has their `w8r2` registers (kernel_facts has no r8 rows yet), so no
    magnitude is frozen. An entry there lands on its sweep's margin and is reported as outside this
    prediction.
  - The rest of the step (the context sweep) is predicted unchanged: the rules' `w4r1` calls sit on
    the Q4_K line (plan section 0). A schedule 0.5 % faster there is a finding for another entry, not
    merged here.
  - Outside, noted: at K >= 8192 the K rule also gives SwiGLU calls `w8r2`, whose x,2 SwiGLU
    kernels spill (255 registers + 296-344 B; the plan's "never chosen for SwiGLU" holds for the
    9B, whose SwiGLU K is 4096, not for the goldens' 2560x17408 and 14336x8192 SwiGLU rows).
- Prototype: `tools/bench/run_850_r8_sweep.sh <build>/bench/ninferno_r8_schedule_bench` on each idle
  card: three sweeps (`--samples 15 --launches 60`) of the down projections and of the rest of the
  step. `tools/bench/r8_sweep_summary.py` prints every cell against this entry.
- Cost: no new route or summation class, and nothing beyond the table. The schedules exist and are
  in `test_generic_linear`'s every-schedule checks.
- Acceptance / refutation (frozen 13.11):
  1. Gain: on each card, the candidate's median minus the rules' on Q6_K 4096x12288 store is at or
     below half the band's lower edge. The edges in us: 4090 -1.0, 5090 -0.4, 3090 -1.5,
     3060 -0.5, 2080 Ti -0.5, 4070 Ti -0.25, 5070 Ti -0.25.
  2. Refutation: a card above its edge refutes the prediction there. A card beyond its band's upper
     edge is a finding. Either names the term from that card's ncu of the call (no t_imb: the law
     loses the term).
  3. Bytes: `exact`, `same=1` on every R8 row of every sweep.
  4. No regression outside: the merge writes an entry only where its schedule is faster in every
     sweep, so no merged problem gets slower. The 9B step does not get slower on any card: plain
     decode, C = 1, 22K, `--spec none`, six repeats, ABBA against master (#850 PR-2).
- Selection (decided before the sweeps): the change lands as R8 decode table entries, through
  `tools/bench/r8_decode_table.py --merge` of the down sweeps. They are per board, measured and
  bit-identical. The K rule itself moves to `w8r1` for Q4_K, Q5_K and Q6_K on boards with
  `mma_bf16_native` only if the sweeps show, on at least four sm_80+ boards and on none against,
  `w8r1` never slower than `w8r2` on any K >= 8192 K-quant row that is not a SwiGLU call (the
  4096x12288, 2560x9216, 5120x17408 and 28672x8192 rows of both sweeps), with no row slower by more
  than 0.5 % on any board. Why not the rule now: the committed tables show `w8r1` over `w8r2` at
  K >= 8192 on six boards, but those gains do not follow the grain law. The Ampere boards gain the
  most (Q6_K 5120x17408: -8 % RTX 3090, -10 % A10, -6 % A100). The coarsest grid, the RTX
  5090's 1.9 CTAs a SM, gains 2.8 %. And the 5090 table has no entry for Q4_K 5120x17408, so
  `w8r1` was not 0.5 % faster there. A residual the law does not explain is no rule.
- A/B (13.11, PR #874): `tools/bench/run_850_r8_sweep.sh` on each idle card, master's
  `ninferno_r8_schedule_bench`, three sweeps of the down and the context problems
  (`--samples 15 --launches 60`, every schedule interleaved in each sample), medians over the
  sweeps; `same=1` on every R8 row of every sweep (exact). The Q6_K 4096x12288 cell, candidate
  minus the rules' schedule, store (add), us:

  | card | rules | candidate | measured | band | verdict |
  |---|---|---|---|---|---|
  | RTX 3090 | `w8r2` 59.12 | `w8r1` 53.82 | -5.30 (-4.97) | -3 .. -7 | accepted (beyond this entry's -4.9) |
  | RTX 4070 Ti | `w8r2` 91.13 | `w8r1` 89.65 | -1.48 (-1.49) | -0.5 .. -2.0 | accepted |
  | RTX 3060 | `w8r2` 140.07 | `w8r1` 128.07 | -12.00 (-12.45) | -1 .. -4 | finding: three times the upper edge |
  | RTX 4090 | `w8r2` 47.04 | `w8r1` 46.79 | -0.25 (-0.27) | -2.0 .. -4.0 | refuted (edge -1.0) |
  | RTX 5070 Ti | `w8r2` 53.53 | `w8r1` 53.78 | +0.25 (+0.47) | -0.5 .. -2.0 | refuted, slower |
  | RTX 5090 | `w8r2` 29.12 | `w8r1` 30.15 | +1.03 (+1.08) | -0.8 .. -2.2 | refuted, slower |
  | RTX 2080 Ti | `w4r2` 80.00 | `w4r1` 83.35 | +3.35 (+3.53) | -1 .. -4 | refuted, slower |

  The rule's criterion is not met: `w8r1` is never slower than `w8r2` at K >= 8192 on the RTX 4090
  and 4070 Ti only; the RTX 3090, 3060, 2080 Ti, 5070 Ti and 5090 each have a row slower by more
  than 0.5 % (5090: Q4_K 4096x12288 +5.5 %; 3060: Q5_K 4096x12288 +7.0 %).

  Where the law missed. Each 4096x12288 call against the card's own Q4_K line in the same protocol
  (from the context sweep's Q4_K 4096x4096 and 8192x4096 at `w4r1`; us above the line, `w4r1` /
  `w4r2` / `w8r1` / `w8r2`):

  | card | Q4_K | Q5_K | Q6_K |
  |---|---|---|---|
  | RTX 4090 | +1.4 / +1.8 / +0.5 / +1.1 | +2.3 / +2.9 / +1.9 / +2.3 | +3.1 / +2.9 / +2.7 / +2.9 |
  | RTX 5090 | +3.2 / +3.1 / +2.9 / +1.8 | +2.7 / +3.3 / +2.4 / +1.5 | +4.6 / +3.8 / +4.1 / +3.1 |
  | RTX 3090 | +3.9 / +6.6 / +4.7 / +4.2 | +7.1 / +6.4 / +7.1 / +4.4 | +7.1 / +8.7 / +5.6 / +10.9 |
  | RTX 3060 | +5.1 / +6.3 / +5.0 / +4.6 | +7.3 / +9.3 / +11.5 / +4.0 | +6.3 / +16.8 / +5.6 / +17.6 |
  | RTX 2080 Ti | +7.1 / +4.1 / +2.8 / +1.7 | +6.3 / +3.4 / +3.9 / +2.0 | +7.0 / +3.7 / +5.9 / +4.6 |
  | RTX 4070 Ti | +1.2 / +1.9 / +0.5 / +2.9 | +1.5 / +2.7 / +1.7 / +2.7 | +2.7 / +3.2 / +2.4 / +3.8 |
  | RTX 5070 Ti | +2.5 / +3.6 / +1.7 / +2.4 | +3.2 / +3.8 / +2.6 / +2.6 | +3.8 / +4.6 / +4.5 / +4.3 |

  - The term the prediction rested on, t_imb (the grain of a one-CTA-a-SM grid), is not what the
    residual is. On the RTX 4090 the Q6_K call sits 2.7-3.1 us above the line under all four
    schedules, whose grains differ by four; a grain term would move with them. The residual is
    the Q6_K codec's at K = 12288 (a lane's ring holds three groups there; at K = 4096, one group,
    no ring, the Q6_K 8192x4096 call sits 0.7-0.9 us above the line). The census's 3.6 us was read
    from eager launches (48.7 us ncu, 50.1 bench); in the sweep's graph protocol `w8r2` is 47.04.
    What ncu checks: on Q6_K and Q4_K 4096x12288 at one schedule,
    `l1tex__t_requests_pipe_lsu_mem_global_op_ld`,
    `smsp__warp_issue_stalled_lg_throttle_per_warp_active`, `long_scoreboard` and
    `sm__cycles_active.max / .avg` (the last one the t_imb hypothesis directly). The plan's
    candidate (c1), the Q6_K group-level Raw (4 of 12 loads a group duplicated), is the variant
    this residual points at.
  - The RTX 3060's -12 us is not imbalance either: imbalance does not depend on the codec, and on
    the same card Q4_K gains nothing and Q5_K loses 7.5 us at `w8r1`. What moves is the codec
    against the rows a warp: Q6_K's two-row schedules (`w8r2`, `w4r2`; 209 registers, eight warps
    a SM) sit 11-12 us above its one-row ones, while Q5_K's two-row `w8r2` is its best by 6-7 us.
    The 3090 shows the same pattern smaller (Q6_K two-row +3..+5 us, Q5_K `w8r2` best). So on
    sm_86 the term is per-warp issue or latency of the codec's decode at the schedule's warps a
    SM, at the highest DRAM bytes per SM clock of the boards (7.2 and 6.7 B/clk against the 4090's
    3.1). ncu on the RTX 3060, Q6_K and Q5_K 4096x12288 at `w8r2` and `w8r1`: issue-active,
    the stall reasons (long scoreboard, MIO / LG throttle, math pipe), achieved warps, local
    memory traffic (a spill the register count does not show).
  - Blackwell and Turing: more CTAs a SM cost at K = 12288. On the RTX 5090, `w8r2` (one CTA a
    SM, eight warps) is the lowest residual on all three codecs, so occupancy is not the bottleneck
    there; its four schedules order by the activation prologue's work per SM (`w8r2` < `w4r2` ~
    `w8r1` < `w4r1`: +0, +0.75, +1.03, +1.57 us on Q6_K). The law priced that prologue as hidden
    (plan section 1.3, 0 .. 0.3 us); on sm_120 it sits after `griddepcontrol.wait`, ahead of slot 1.
    On the RTX 2080 Ti the two-row schedules are the lowest on every codec and `w4r1` is the
    highest (+6.3 .. +7.1), Turing's quarter-rate F2I in the same prologue. ncu on the 5090:
    the instructions and cycles before the first `BAR.SYNC` against the slot-0 bytes' arrival, and
    the DRAM idle time per round, at `w8r2` against `w8r1`; on the 2080 Ti, the XU pipe in the
    prologue at `w4r1`.
  - The RTX 4070 Ti, the one card with the predicted gain inside its band (-1.5), and the 3090
    (-5.3) fit the sign of the law; neither tells t_imb apart from the per-warp term above.
  Acceptance item 4's 9B step ABBA was not run; it waits (the entries change only the 32 down
  calls of a step). Its kit is `tools/bench/run_850_step_ab.sh`, run on every card with an entry
  (RTX 4090, 3090, 3060, 4070 Ti, 5070 Ti); a card whose step is slower than master beyond the
  noise loses its entries.
- Decision (13.11): narrowed to table entries, the K rule unchanged. `r8_decode_table.py --merge`
  of the down sweeps wrote, at 4096x12288 store (an add call takes it):
  - `ad102:128sm` (RTX 4090): Q4_K `w8r1` (31.90 -> 31.23), Q5_K `w8r1` (39.48 -> 39.13), Q6_K
    `w8r1` (47.04 -> 46.79);
  - `ga102:82sm` (RTX 3090): Q4_K `w4r1` (38.08 -> 37.73), Q6_K `w8r1` (59.12 -> 53.82);
  - `ga106:28sm` (RTX 3060, a new table): Q6_K `w8r1` (140.07 -> 128.07);
  - `ad104:60sm` (RTX 4070 Ti, a new table): Q4_K `w8r1` (63.15 -> 60.77), Q5_K `w4r1`
    (76.03 -> 74.85), Q6_K `w8r1` (91.13 -> 89.65);
  - `gb203:70sm` (RTX 5070 Ti, a new table): Q4_K `w8r1` (36.71 -> 36.03), Q6_K `w4r1`
    (53.53 -> 53.07; the candidate `w8r1` was slower there);
  - `gb202:170sm` (RTX 5090) and `tu102:68sm` (RTX 2080 Ti): none; the rules' `w8r2` / `w4r2`
    were the fastest.
  Every entry is faster than the rules' schedule in every sweep, and where the add call was swept
  (Q4_K and Q6_K) it is faster in the median with the entry's schedule too. The Q5_K entries rest
  on store alone: a 9B mix with a Q5_K down projection runs it through `ops::linear_add`, which
  takes the store entry, so the sweep kit now also times Q5_K 4096x12288 add, and a Q5_K entry
  stays only where its schedule also wins on the add call (the summary's add check). The law is
  not refitted: its t_imb term stays a hypothesis this A/B did not confirm, and the three terms above
  wait for their ncu facts before any kernel (c1 for the Q6_K ring, the prologue's price on
  sm_120 and sm_75).
- Carried to unmeasured boards (kR8CarryDistance 0.5, same feature set): the RTX 3060 non-LHR
  (0x2503, 0.0, the 3060 table), the 4070 SUPER (0.10, the 4070 Ti's), the 4080 Laptop
  (0x27A0 / 0x27E0, 0.27, the 4070 Ti's) and the 5090 Laptop (0x2C18 / 0x2C58, 0.44, the 5070
  Ti's). The last sits near a bound that is itself `Basis::Unknown`, its runtime memory clock
  decides whether it lands inside it, and the 5070 Ti entries it takes win by 0.9 % (Q6_K `w4r1`)
  and 1.9 % (Q4_K); the sweeps show the schedule order flipping between 70 and 170 SMs on sm_120.
  The schedules are bit-identical, so a wrong carry costs speed, never bits; a laptop sweep
  settles it. The RTX 5080 (0.57) takes nothing.

### Q. Volta BF16 KV width-1 decode on a key-major kernel (#877, #893)
- Regime: the BF16 KV width-1 decode (GroupedDecode, D = 256, every listed head geometry) on
  sm_70 builds: Quadro GV100. At #866 it read the cache at about 175 GB/s (20 % of the 871 GB/s
  DRAM line; 228 / 558 µs a call at 7843 / 21961 keys, Qwen3.5-9B geometry, 8 layers), against
  672 GB/s (77 %) for the INT8 key-major kernel on the same card and run (#877).
- Weight: GV100's whole BF16-KV decode attention term. At 21961 keys that is about 4.5 ms of an
  8-layer step at today's rate.
- Law: issue on the MMA emulation. sm_70 has no BF16 MMA and none of the m16n8k16 shapes the grouped
  kernel issues; each product is `mma_bf16_fma_call` / `mma_f16_fma_call` (`__noinline__`), 552
  SASS instructions and 32 shuffle-helper calls for 16 query rows, of which a decode fills G
  (4 here) (CUDA 12.8 SASS of `CausalGeometry<256,16,4>`; the kernel 13,536 instructions, 484
  CALLs). 175 GB/s at GV100's ~464 G warp instructions a second is ~86 k instructions per 32 KB of
  K/V; a 32-key tile per D warp takes 64-96 emulated products, 40-86 k at 600-900 each.
- Variant: S on CUDA cores with one key a lane: the key's BF16 row widened to FP32 by a shift, G
  FMAs a dimension against Q in shared memory (FP32), no cross-lane product. O^T += V^T P^T on
  Volta's native `mma.m8n8k4` (f16 operands, FP32 accumulation; V is stored FP16 and P rounded
  to FP16 as before), with `int8_kv_key_major_kt_kernel`'s PV fragments. Four warps own every
  fourth 32-key step of the split and merge their softmax states in the CTA. The grouped
  kernel's partition, partials and merge are kept. Counted: about 10 B a warp instruction, so
  ~4.6 TB/s of issue against the 871 GB/s line: DRAM-bound.
- Prediction (frozen 13.11 on #877, before any code): the BF16 slope read rate on GV100 at
  0.85-1.05x of INT8's measured 672 GB/s (570-705 GB/s, 65-81 % of DRAM), and the fixed part a
  call within ±10 µs of INT8's (about 30 µs today).
- Prototype: `CODECS="bf16 int8" kvab2.sh gv100 perf/877-bf16-kv-volta-key-major k877k`
  (`ninferno_kv_decode_attention_bench --codec bf16 --context 7843,21961 --layers 8`).
- Cost: not a linear route, so no selector goldens or route classes. It is built for sm_70 only
  and dispatched in `causal_cache/bf16/launch.cu`. ptxas 0.15 s an instance, 190-206 registers,
  no spills (the forceinline experiment on the emulation did not finish in 1 h 27 min, #877).
- Acceptance / refutation (frozen): kept at a slope rate of at least 0.75x INT8's; refuted below
  that. No other arch builds it. Bytes: not `exact` (other FP32 order).
  `ninferno_bf16_decode_identity_test` holds it to the shipped schedule:
  - the appended cache and the set of written partials byte for byte;
  - numerators within 2^-9 of their row's largest, the log2 maximum within 1e-5, the sum within
    2^-10, the output within one BF16 step;
  - its own bits with `--self`.

  The FP64 oracle of `ninferno_softmax_attention_test --kv-dtype bf16` also applies.
- A/B (13.11, Quadro GV100, `kv_decode_attention_bench` ABBA, master b2a836ef2 against the
  branch merged with it, Qwen3.5-9B geometry, 8 layers, INT8 the control):

  | arm | 7843 keys, µs a call | 21961 keys, µs a call | slope read rate | fixed part |
  |---|---|---|---|---|
  | BF16 master, round 1 | 280.2 | 632.6 | 164 GB/s | 84 µs |
  | BF16 master, round 2 | 228.5 | 557.8 | 176 GB/s | 46 µs |
  | BF16 key-major, both rounds | 70.7 | 166.3 | 605 GB/s | 17.6 µs |
  | INT8, all four | 54.6 | 99.0 | 672 GB/s | 30 µs |

  - Slope: 605 GB/s is 0.90x INT8's, inside the frozen 0.85-1.05x (69 % of DRAM). Held.
  - A call at 21961 keys takes 166 µs against 558-633: x3.35-3.8, about 3.1 ms less a step over 8
    layers.
  - Fixed part: 17.6 µs, under the frozen INT8 ± 10 µs (20-40). A finding on the favourable side.
    The term that was wrong: the prediction took INT8's fixed part as the floor, but INT8's carries
    its Q codec (the Hadamard and per-group quantization of every row) and its scale staging. The
    BF16 kernel has neither: it widens Q to FP32 once and starts its keys. Recorded, not fitted.
- Decision: kept on sm_70 (the slope acceptance, at least 0.75x INT8's, is met).
