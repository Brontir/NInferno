#pragma once

// The register MMA's terms counted from its kernel (#613 PR-7a): what one iteration of the K loop
// issues, read off the SASS into the kernel-facts snapshot (kernel_facts::LoopCounts, #650), turned
// into seconds on a chip. No CUDA, no measurement: the counts are the kernel's, the rates and
// latencies the chip's published facts. The price (generic_cost.cpp) reads them in place of two
// fitted constants under route_cost::kRmmaCountedLaw:
//   - the tensor term from the loop's MMAs, the min-term ones (activations against 1.0, mma_step)
//     included: (sets x m_tiles + 1) / (sets x m_tiles) of the weight MMAs where the tile does not
//     add the sums in FP32 (#598), read, not reasoned. On the lab sweeps this term is a floor the
//     measured time never goes under (0 of 45 000 rows), and the best Q4_K routes run at 1.06-1.14
//     of it: the ones MMAs cost tensor time like any other;
//   - the spill term from the loop's LDL / STL count in place of kSpillAccessesPerStep (2 accesses
//     of the whole stack frame per 128-K step, Fitted). A spilled register is reloaded right before
//     its use at 255 registers, so the warp waits the L1 hit latency (kL1HitLatencyCycles), hidden
//     only by the other warps its scheduler holds (resident warps / 4): local x latency / warps per
//     scheduler, not bytes over the L1 data path, which is 10-15x too small (3090, Q4_K SwiGLU
//     15360x3840 at 96 tokens: m2.n4 spills 114 LDL/STL an iteration and measures 290 us over
//     m1.n4; bytes over 128 B/clk say 20 us, the latency form 300 us).
// The issue term (every warp instruction of the loop over the SM's four schedulers, and the INT32
// and FP32 pipes' cycles) is computed beside them for the replica and the tests: on the measured
// Ampere, Ada and Blackwell sweeps it never exceeds the bytes-plus-tensor sum, so the fill that
// kRmmaSaturatingWarps stands for is a latency effect (the narrow-T chain, PR-7b), not an issue
// bound; on Turing it binds on most rows (the BF16 operands' conversion is 192 FP32 instructions
// per iteration against 8 MMAs), where the heavy form is not priced anyway.
// What the counted terms do not fix (tools/bench/rmma_counted_replica.py, --law): the price adds
// bytes and tensor time, while the measured kernels overlap them (the best routes sit at 1.1-1.4
// of the tensor floor). Raising an m1 route's tensor term by its ones share then moves picks to m2
// routes that the sum prices lower but that measure slower; the remaining regressions of the
// counted law against today's (worst x1.3-2.2 per board) are all of that form, plus the token tiles
// of one weight block re-reading it from DRAM when they are not co-resident (blockIdx.x runs the
// row blocks first). Both need the max form of the price, #613's one cost model, not a term here.

#include "ops/generic/kernel_facts.h"

#include <string_view>

namespace ninferno::ops::detail::generic::rmma_counted {

// ISA facts of the SM: four schedulers, each issuing one warp instruction a clock; 16 INT32 lanes
// per scheduler (two clocks per warp instruction); 16 FP32 lanes on Volta and Turing, 32 from
// Ampere on (half of them shared with INT32).
inline constexpr double kSchedulersPerSm = 4.0;

[[nodiscard]] constexpr double int32_cycles_per_warp_instruction(int /*major*/) noexcept {
    return 2.0;
}

[[nodiscard]] constexpr double fp32_cycles_per_warp_instruction(int major) noexcept {
    return major <= 7 ? 2.0 : 1.0;
}

// Whether the counts price the kernel: one K loop found, its K per iteration known, tensor MMAs
// in it (the FMA-emulated Volta build has none).
[[nodiscard]] constexpr bool usable(const kernel_facts::LoopCounts& l) noexcept {
    return l.loops == 1 && l.k_iter > 0 && l.mma > 0 && l.mma_ones < l.mma;
}

// The floating-point operations of one MMA of the loop's shape: 2 x m x n x k (m16n8k16 4096,
// Turing's m16n8k8 2048, m16n8k4 1024); the m16n8k16 figure for a shape not listed.
[[nodiscard]] inline double mma_flops(const kernel_facts::LoopCounts& l) noexcept {
    const std::string_view s = l.mma_shape;
    if (s.find(".1688.") != std::string_view::npos) return 2048.0;
    if (s.find(".1684.") != std::string_view::npos) return 1024.0;
    if (s.find(".884.") != std::string_view::npos) return 512.0;
    return 4096.0;
}

// The loop iterations one warp runs for a CTA's K range `kc` split over `warps_k` warps.
[[nodiscard]] constexpr double iterations(const kernel_facts::LoopCounts& l, double kc,
                                          double warps_k) noexcept {
    return l.k_iter > 0 ? (kc / warps_k) / l.k_iter : 0.0;
}

// The tensor work over the weight MMAs alone: the min-term MMAs' share, 1 without them.
[[nodiscard]] constexpr double ones_factor(const kernel_facts::LoopCounts& l) noexcept {
    return l.mma > l.mma_ones ? static_cast<double>(l.mma) / (l.mma - l.mma_ones) : 1.0;
}

struct Terms {
    double warp_iterations = 0; // of the busiest SM: waves x warps per CTA x iterations per warp
    double mma             = 0; // seconds: every MMA of the loop over the tensor rate
    double spill           = 0; // seconds: the loop's LDL / STL waits at the L1 latency, shared
                                // by the warps a scheduler holds
    double issue_slots = 0;     // seconds: every instruction over the schedulers' issue rate
    double int_pipe    = 0;     // seconds: the INT32 pipe
    double fp_pipe     = 0;     // seconds: the FP32 pipe
    double issue       = 0;     // the longest of the three
};

// The counted terms of `waves` CTAs of `warps_per_cta` warps on one SM, each warp over `kc` K
// values split `warps_k` ways; `tensor_flops_per_s` the SM's tensor rate (the generation row's
// rate x the reference clock), `resident_warps` the warps resident on the SM at once (the spill
// latency is shared by the quarter of them a scheduler holds), `l1_hit_latency_cycles` that
// latency, `clock_hz` the reference clock.
[[nodiscard]] inline Terms terms(const kernel_facts::LoopCounts& l, double waves,
                                 double warps_per_cta, double kc, double warps_k,
                                 double tensor_flops_per_s, double resident_warps,
                                 double l1_hit_latency_cycles, double clock_hz,
                                 int major) noexcept {
    Terms t;
    if (!usable(l) || clock_hz <= 0 || tensor_flops_per_s <= 0) return t;
    t.warp_iterations = waves * warps_per_cta * iterations(l, kc, warps_k);
    t.mma             = t.warp_iterations * l.mma * mma_flops(l) / tensor_flops_per_s;
    const double per_scheduler =
        resident_warps / kSchedulersPerSm > 1.0 ? resident_warps / kSchedulersPerSm : 1.0;
    t.spill = t.warp_iterations * l.local * l1_hit_latency_cycles / per_scheduler / clock_hz;
    const double per_slot = 1.0 / (kSchedulersPerSm * clock_hz);
    t.issue_slots         = t.warp_iterations * l.total * per_slot;
    t.int_pipe =
        t.warp_iterations * l.int_alu * int32_cycles_per_warp_instruction(major) * per_slot;
    t.fp_pipe = t.warp_iterations * l.fp_alu * fp32_cycles_per_warp_instruction(major) * per_slot;
    t.issue   = t.issue_slots > t.int_pipe ? (t.issue_slots > t.fp_pipe ? t.issue_slots : t.fp_pipe)
                                           : (t.int_pipe > t.fp_pipe ? t.int_pipe : t.fp_pipe);
    return t;
}

} // namespace ninferno::ops::detail::generic::rmma_counted
