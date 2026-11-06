#pragma once

// The generation row: what an SM of one card generation does per clock, and the per-generation
// constants the route cost model was fitted with. A card is everything its query reports (SMs,
// clocks, bus, shared memory, residency: DeviceClass) plus the row of its compute capability; no
// code outside this table compares a major or minor version to learn a rate (the generic
// selector's rules measured per generation compare them to pick a route, not a rate). A new
// generation is one row.
//
// Rates are the cost model's tensor rate per SM and clock (the 2 x rows x cols x K of a tile over
// rate x clock is its MMA time), with FP32 accumulation, the unit route_tune's fit used:
//   bf16  the BF16 MMA. Ampere, Ada and Blackwell consumer and pro boards 512, the full-rate HBM
//         parts 2048; where the BF16 MMA is emulated (Turing's f16 k8 pairs, Volta's FP32 FMAs,
//         DeviceFeatures::mma_bf16_native false) the fitted aggregate 1536.
//   s8    the int8 MMA, 0 where the card has none (Volta, whose build composes it from dp4a).
// fp32acc_full_rate: FP32-accumulate HMMA at the tensor core's full rate (sm_80, sm_90,
// sm_100/103), where int8 activations gain less than their quantization pass costs; the GeForce,
// workstation and inference parts (sm_75, sm_86, sm_89, sm_110, sm_120/121) run it at half rate,
// a quarter of int8. Measured data does not carry between the two kinds (board_distance).
// The fitted constants (ninferno_route_tune timings of the RTX 4090, 5090, 4060 Ti, L4, A10 and
// A100, 18.10; the full-rate rows have one board, the A100):
//   w4a8_eff_scale  the W4A8 tiles' fraction of the consumer fit's efficiency (A100 0.46)
//   a16_cost_scale  the A16 model's factor against the A8 one (1.11; A100 1.64: with
//                   the busy-SM fill it moves 0.89 -> 1.11, which also prices the dequant of the
//                   K-quant and codebook formats the A16 side pays: 5070 Ti ISTA T 17..32 1.20 -> 1.04)
//   mmq_candidate   ggml's matrix kernel is a cost-model candidate (not on the full-rate row,
//                   where its microbenchmark win and the 24K prefill A/B, -15 %, disagree)
//   attention_chain the Gemma 4 chained prefill attention tile runs near the plain tile's speed in
//                   full waves, so chaining a mostly idle last wave pays
//                   (ops/softmax_attention/gemma4/launch.cu chain_pays): measured on gb20x (RTX
//                   5090 31B T=4096 -2.2 %, RTX 5080 26B at 16K keys +5.5 %); on Ada the chained
//                   tile runs 25-35 % slower even in whole waves (RTX 4090, 4080 SUPER); the other
//                   generations are unmeasured and keep the plain launch

//
// What a lent row carries (#300): generation_row answers every compute capability, lending the
// newest (or the oldest) row to one the table does not know, so that the code runs there. Beside
// the device's capability, which is not in this table, its fields are of four kinds, and only the
// rates are lent:
//   capability   what the device can execute: not in this table (DeviceFeatures, the query)
//   rate         bf16, s8: the analytic tensor rate a cost model ranks routes with (lent: an
//                assumed rate, the ranking said to rest on it)
//   kind         fp32acc_full_rate: which measured kind of tensor core the generation is; a lent row
//                has none (fp32acc_full_rate_of: nullopt), and what rests on the kind (int8
//                prefill pays, i8 replaces the q4a8 kernels, board distances) is not carried
//   fitted       w4a8_eff_scale, a16_cost_scale: route_tune fits of the boards above; not carried
//                (fitted_route_scales: nullopt), the comparisons they price not made
//   preference   mmq_candidate, attention_chain: a measured A/B; not carried
//                (mmq_candidate_measured, attention_chain_measured)
// A generation newer than the table therefore runs the routes every board runs (A16 prefill, the
// plain attention launch) until a row of its own is added.

#include <iterator>
#include <optional>

namespace ninferno {

struct GenerationRow {
    int major    = 0;
    int minor_lo = 0;
    int minor_hi = 0;
    const char* name = "";
    double bf16  = 0;
    double s8    = 0;
    bool fp32acc_full_rate = false;
    double w4a8_eff_scale  = 1.0;
    double a16_cost_scale  = 1.0;
    bool mmq_candidate     = false;
    bool attention_chain   = false;
};

inline constexpr GenerationRow kGenerations[] = {
    //  major minor     name       bf16    s8      full   w4a8  a16   mmq    chain
    {7, 0, 4, "volta", 1536.0, 0.0, false, 1.0, 1.11, true},
    {7, 5, 9, "turing", 1536.0, 2048.0, false, 1.0, 1.11, true},
    {8, 0, 0, "ga100", 2048.0, 4096.0, true, 0.46, 1.64, false},
    {8, 1, 8, "ga10x", 512.0, 2048.0, false, 1.0, 1.11, true},
    {8, 9, 9, "ada", 512.0, 2048.0, false, 1.0, 1.11, true},
    {9, 0, 9, "hopper", 2048.0, 4096.0, true, 0.46, 1.64, false},
    {10, 0, 9, "gb100", 2048.0, 4096.0, true, 0.46, 1.64, false},
    {11, 0, 9, "thor", 512.0, 2048.0, false, 1.0, 1.11, true},
    {12, 0, 9, "gb20x", 512.0, 2048.0, false, 1.0, 1.11, true, true},
};

// Whether the table has a row of its own for compute capability major.minor (generation_row lends
// one otherwise): a lent row runs the code, never carries its generation's evidence (#300).
[[nodiscard]] constexpr bool generation_known(int major, int minor) noexcept {
    for (const GenerationRow& row : kGenerations) {
        if (row.major == major && minor >= row.minor_lo && minor <= row.minor_hi) return true;
    }
    return false;
}

// The row of compute capability major.minor: a later generation than the table knows takes its
// last row (the newest consumer one), an earlier one its first.
[[nodiscard]] constexpr const GenerationRow& generation_row(int major, int minor) noexcept {
    for (const GenerationRow& row : kGenerations) {
        if (row.major == major && minor >= row.minor_lo && minor <= row.minor_hi) return row;
    }
    return major > kGenerations[std::size(kGenerations) - 1].major
               ? kGenerations[std::size(kGenerations) - 1]
               : kGenerations[0];
}

// The generation's measured tensor-core kind (FP32-accumulate HMMA at full rate or not), nullopt
// on a lent row (#300): the kind is the generation's, the kinds' measurements its boards'.
[[nodiscard]] constexpr std::optional<bool> fp32acc_full_rate_of(int major, int minor) noexcept {
    if (!generation_known(major, minor)) return std::nullopt;
    return generation_row(major, minor).fp32acc_full_rate;
}

// The fitted route-cost scales of the generation's own row, nullopt on a lent row (#300).
struct FittedRouteScales {
    double w4a8_eff = 1.0;
    double a16_cost = 1.0;
};
[[nodiscard]] constexpr std::optional<FittedRouteScales> fitted_route_scales(int major,
                                                                             int minor) noexcept {
    if (!generation_known(major, minor)) return std::nullopt;
    const GenerationRow& row = generation_row(major, minor);
    return FittedRouteScales{row.w4a8_eff_scale, row.a16_cost_scale};
}

// Whether the Gemma 4 chained prefill attention tile is the measured preference on compute
// capability major.minor (#300): its generation's own row says so. A lent row runs the code but
// does not carry this measured preference: a generation newer than the table keeps the plain
// launch, which every board runs, until it is measured.
[[nodiscard]] constexpr bool attention_chain_measured(int major, int minor) noexcept {
    return generation_known(major, minor) && generation_row(major, minor).attention_chain;
}

// Whether ggml's matrix kernel is a cost-model candidate on compute capability major.minor
// (mmq_candidate, a measured preference: the full-rate row's 24K prefill A/B lost): only on a
// generation of the table's own; a lent row does not carry it (#300).
[[nodiscard]] constexpr bool mmq_candidate_measured(int major, int minor) noexcept {
    return generation_known(major, minor) && generation_row(major, minor).mmq_candidate;
}

// The compute capabilities the generic selector's measured rules name (generic_gemm.cpp). A
// generation row spans minors no board of which was measured (12.1, GB10, lies in gb20x's row),
// so a rule measured on one capability's boards keeps that capability exactly, as written.
// sm_75 .. sm_79: Turing's tensor cores, no cp.async (ring_swiglu_four_warps).
[[nodiscard]] constexpr bool turing_cc(int major, int minor) noexcept {
    return major == 7 && minor >= 5;
}
// sm_80 .. sm_88: Ampere, the A100 (sm_80) and the GA10x boards (sm_86) alike (w4a8_ws_wins).
[[nodiscard]] constexpr bool ampere_cc(int major, int minor) noexcept {
    return major == 8 && minor < 9;
}
// sm_89: the RTX 40 boards (w4a8_ws_extra, w4a8_ws_wins, w4a8_wide_q4k).
[[nodiscard]] constexpr bool ada_cc(int major, int minor) noexcept {
    return major == 8 && minor == 9;
}
// sm_120: the RTX 50 boards (w4a8_ws_extra, w4a8_ws_wins).
[[nodiscard]] constexpr bool gb20x_cc(int major, int minor) noexcept {
    return major == 12 && minor == 0;
}
// Major 12, the consumer Blackwell SM (the R8 rules' SM-count edge between the RTX 5080 and 5090).
[[nodiscard]] constexpr bool consumer_blackwell_major(int major) noexcept {
    return major == 12;
}

} // namespace ninferno
