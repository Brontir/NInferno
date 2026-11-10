#pragma once

// The register MMA's terms counted from its kernel (#613 PR-7a): what one iteration of the K loop
// issues, read off the SASS into the kernel-facts snapshot (kernel_facts::LoopCounts, #650), turned
// into seconds on a chip. No CUDA, no measurement: the counts are the kernel's, the rates the
// chip's published facts. The price (generic_cost.cpp) reads them in place of two fitted constants
// where NINFERNO_RMMA_COUNTED_TERMS is set:
//   - the tensor term from the loop's MMAs, the min-term ones (activations against 1.0, mma_step)
//     included: (sets x m_tiles + 1) / (sets x m_tiles) of the weight MMAs where the tile does not
//     add the sums in FP32 (#598), read, not reasoned;
//   - the spill term from the loop's LDL / STL bytes in place of kSpillAccessesPerStep (2 accesses
//     of the whole stack frame per 128-K step, Fitted).
// The issue term (every warp instruction of the loop over the SM's four schedulers, and the INT32
// and FP32 pipes' cycles) is computed beside them for the replica and the tests: on the measured
// Ampere, Ada and Blackwell sweeps it never exceeds the bytes-plus-tensor sum, so the fill that
// kRmmaSaturatingWarps stands for is a latency effect (the narrow-T chain, PR-7b), not an issue
// bound; on Turing it binds on most rows (the BF16 operands' conversion is 192 FP32 instructions
// per iteration against 8 MMAs), where the heavy form is not priced anyway.

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
    double spill           = 0; // seconds: the loop's LDL / STL bytes over the L1 rate
    double issue_slots     = 0; // seconds: every instruction over the schedulers' issue rate
    double int_pipe        = 0; // seconds: the INT32 pipe
    double fp_pipe         = 0; // seconds: the FP32 pipe
    double issue           = 0; // the longest of the three
};

// The counted terms of `waves` CTAs of `warps_per_cta` warps on one SM, each warp over `kc` K
// values split `warps_k` ways; `tensor_flops_per_s` the SM's tensor rate (the generation row's
// rate x the reference clock), `l1_bytes_per_s` its L1 rate, `clock_hz` the reference clock.
[[nodiscard]] inline Terms terms(const kernel_facts::LoopCounts& l, double waves,
                                 double warps_per_cta, double kc, double warps_k,
                                 double tensor_flops_per_s, double l1_bytes_per_s, double clock_hz,
                                 int major) noexcept {
    Terms t;
    if (!usable(l) || clock_hz <= 0 || tensor_flops_per_s <= 0 || l1_bytes_per_s <= 0) return t;
    t.warp_iterations     = waves * warps_per_cta * iterations(l, kc, warps_k);
    t.mma                 = t.warp_iterations * l.mma * mma_flops(l) / tensor_flops_per_s;
    t.spill               = t.warp_iterations * static_cast<double>(l.local_bytes) / l1_bytes_per_s;
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
