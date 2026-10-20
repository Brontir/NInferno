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

#include <iterator>

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
};

inline constexpr GenerationRow kGenerations[] = {
    //  major minor     name       bf16    s8      full   w4a8  a16   mmq
    {7, 0, 4, "volta", 1536.0, 0.0, false, 1.0, 1.11, true},
    {7, 5, 9, "turing", 1536.0, 2048.0, false, 1.0, 1.11, true},
    {8, 0, 0, "ga100", 2048.0, 4096.0, true, 0.46, 1.64, false},
    {8, 1, 8, "ga10x", 512.0, 2048.0, false, 1.0, 1.11, true},
    {8, 9, 9, "ada", 512.0, 2048.0, false, 1.0, 1.11, true},
    {9, 0, 9, "hopper", 2048.0, 4096.0, true, 0.46, 1.64, false},
    {10, 0, 9, "gb100", 2048.0, 4096.0, true, 0.46, 1.64, false},
    {11, 0, 9, "thor", 512.0, 2048.0, false, 1.0, 1.11, true},
    {12, 0, 9, "gb20x", 512.0, 2048.0, false, 1.0, 1.11, true},
};

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

} // namespace ninferno
