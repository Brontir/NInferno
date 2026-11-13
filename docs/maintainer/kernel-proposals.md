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

The entries below come from the candidates table of #619 (section 4). Only A and N have a frozen
prediction so far.

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
