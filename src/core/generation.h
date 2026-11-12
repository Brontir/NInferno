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
//         parts 2048. Turing (#781) 512 from the spec: its build rounds BF16 operands to f16 and
//         issues mma.sync m16n8k8 .f32.f16.f16.f32 (two per k16, ops/common/mma.cuh), the f16
//         MMA with FP32 accumulation; an SM's 8 tensor cores do 64 FMA a clock each, 8 x 64 x 2 =
//         1024 FLOP with f16 accumulation, and the GeForce Turing boards accumulate FP32 at half
//         that, 512 (the Turing whitepaper's RTX 2080 Ti: 107.6 TFLOPS f16-accumulate, 53.8 with
//         FP32 accumulation, at 68 SMs and the 1545 MHz reference boost). The f16 conversion is
//         not in this rate. OPEN (#781): the Tesla T4, the Quadro RTX and the TITAN RTX run FP32
//         accumulation at the full 1024, a fact of the board, not of the compute capability; the
//         facts have no per-board tensor rate (gpu_facts.json carries mma_slow, the one per-board
//         MMA fact) and every reader of the kind (fp32acc_full_rate_of) is keyed by generation,
//         so those boards are priced at 512 here (TITAN RTX is segment geforce in gpu_facts.json,
//         so not even the segment separates them). Turing's A8 prices (W4A8, MMQ, priced since
//         #781) carry efficiencies fitted on other generations (route_cost_constants.h
//         kW4a8Efficiency, kMmqEfficiency; w4a8_eff_scale 1.0 below): 0.58 .. 0.75 of the
//         measured time for MMQ and 0.70 .. 0.74 for W4A8 on the RTX 2080 Ti A5 sweep. OPEN: the
//         missing fact, MMQ's dequantize on 64 CUDA-core lanes a SM against 128 on Ampere and Ada.
//         Volta (FP32 FMAs, mma_bf16_native false) keeps the fitted aggregate 1536.
//   s8    the int8 MMA, 0 where the card has none (Volta, whose build composes it from dp4a).
//   fp32_fma  not a tensor rate: the FP32 fused multiply-adds an SM issues per clock (the CUDA C++
//         Programming Guide's arithmetic instruction throughput table, "32-bit floating-point
//         add, multiply, multiply-add"): 64 on Volta, Turing and GA100, 128 on GA10x, Ada, Hopper
//         and the Blackwell generations. A rate (lent as bf16 and s8 are).
// fp32acc_full_rate: FP32-accumulate HMMA at the tensor core's full rate (sm_80, sm_90,
// sm_100/103), where int8 activations gain less than their quantization pass costs; the GeForce,
// workstation and inference parts (sm_75, sm_86, sm_89, sm_110, sm_120/121) run it at half rate,
// a quarter of int8. Measured data does not carry between the two kinds (board_distance).
// The fitted constants (ninferno_route_tune timings of the RTX 4090, 5090, 4060 Ti, L4, A10 and
// A100, 18.10; the full-rate rows have one board, the A100):
//   w4a8_eff_scale  the W4A8 tiles' fraction of the consumer fit's efficiency (A100 0.46)
//   a16_cost_scale  the A16 model's factor against the A8 one (1.11; A100 1.64: with
//                   the busy-SM fill it moves 0.89 -> 1.11, which also prices the dequant of the
//                   K-quant and codebook formats the A16 side pays: 5070 Ti ISTA T 17..32 1.20
//                   -> 1.04)
//   mmq_candidate   ggml's matrix kernel is a cost-model candidate (not on the full-rate row,
//                   where its microbenchmark win and the 24K prefill A/B, -15 %, disagree)
//   attention_chain the Gemma 4 chained prefill attention tile runs near the plain tile's speed in
//                   full waves, so chaining a mostly idle last wave pays
//                   (ops/softmax_attention/gemma4/launch.cu chain_pays): measured on gb20x (RTX
//                   5090 31B T=4096 -2.2 %, RTX 5080 26B at 16K keys +5.5 %); on Ada the chained
//                   tile runs 25-35 % slower even in whole waves (RTX 4090, 4080 SUPER); the other
//                   generations are unmeasured and keep the plain launch

//   mmq_work_tiles  ggml's matrix kernel also instantiates the 80- and 96-column tiles and takes
//                   the column tile of least work, ceil(columns / J) x (0.43 + 0.57 J / 128),
//                   instead of the fewest tiles of 16, 32, 64, 128 (core/mmq_columns.h, DECISIONS
//                   74). The rule: a tile is a fixed part (its 128 weight rows' decode, the same
//                   for every J) plus a part in J (the MMA, the y tile's loads), and narrower tiles
//                   pay while the part in J is large enough that the padding they save outweighs
//                   the fixed part they repeat. On GA10x the int8 MMA alone is 4096 of a Q4_K
//                   tile's 10600 SM-clocks a 256-value K step (1024 MMAs of 4096 MACs at 1024 MACs
//                   a clock). The fixed share is a measurement of the generation (mmq_tile_fixed:
//                   the grid regret over the widths is least at 0.15-0.25 on the RTX 3060 and
//                   3090, 0.05-0.15 on the RTX 5090; DECISIONS 74), a generation with a higher
//                   MMA rate (ga100: s8 4096) has a smaller part in J and a larger fixed share, so
//                   fewer widths pay. A generation joins by its own grid: Ada did (0.20, the RTX
//                   4070 Ti and 4090). Turing joined by its grid (the RTX 2080 Ti, DECISIONS 78)
//                   with a launch law instead of a fixed share (mmq_launch_law): on its 68 SMs the
//                   width decides whether a launch runs a tile a CTA or stream-k with a fixup,
//                   which the tile count does not see. A measured preference, not carried by a
//                   lent row.

//
// What a lent row carries (#300): generation_row answers every compute capability, lending the
// newest (or the oldest) row to one the table does not know, so that the code runs there. Beside
// the device's capability, which is not in this table, its fields are of four kinds, and only the
// rates are lent:
//   capability   what the device can execute: not in this table (DeviceFeatures, the query)
//   rate         bf16, s8 (the analytic tensor rate a cost model ranks routes with), fp32_fma
//                (lent: an assumed rate, the ranking said to rest on it)
//   kind         fp32acc_full_rate: which measured kind of tensor core the generation is; a lent
//                row has none (fp32acc_full_rate_of: nullopt), and what rests on the kind (int8
//                prefill pays, i8 replaces the q4a8 kernels, board distances) is not carried
//   fitted       w4a8_eff_scale, a16_cost_scale, mmq_tile_fixed: route_tune fits of the boards
//                above (mmq_tile_fixed: DECISIONS 74's grids); not carried (fitted_route_scales:
//                nullopt, the comparisons they price not made; mmq_tile_fixed_of: 0.43)
//   preference   mmq_candidate, attention_chain, mmq_work_tiles: a measured A/B; not carried
//                (mmq_candidate_measured, attention_chain_measured)
// A generation newer than the table therefore runs the routes every board runs (A16 prefill, the
// plain attention launch) until a row of its own is added.

#include <iterator>
#include <optional>

namespace ninferno {

// The launch law of ggml's matrix kernel, for a generation whose column tiles are priced by the
// launch they run (core/mmq_columns.h launch_seconds; DECISIONS 78) instead of their tile count. A
// launch takes its longest CTA's units times a unit's cost (a unit: one 128-row x J-column tile
// over 256 K values), plus, where it splits tiles (stream-k with a fixup), the fixup's bytes at the
// DRAM rate and one dependent launch.
//   unit_clocks     a unit's SM-clocks at the board's reference clock, in a launch of a tile a CTA,
//                   by J = 16, 32, 64, 80, 96, 128 (core/mmq_columns.h kTilesWork)
//   stream_k_unit   a stream-k CTA's unit against a tile-a-CTA one at the same J
//   launch_seconds  what the fixup costs beyond its bytes at the DRAM rate: its kernel's fixed time
//                   and its dependent launch (a generation without PDL)
struct MmqLaunchLaw {
    double unit_clocks[6] = {};
    double stream_k_unit  = 1.0;
    double launch_seconds = 0.0;
};

// Turing: MEASURED on one card, the RTX 2080 Ti (68 SMs), and carried to the generation as a
// transfer: no second Turing card is measured (DECISIONS 78). tools/bench/mmq_launch_microbench.cu,
// 11.11, Q4_K, 4096 x 4096 and 4096 x 8192 products, the medians of two passes of 9 x 1000 calls.
// unit_clocks: the time of one unit (16 more units, k 4096 -> 8192, over 16) in a launch of a
// tile a CTA on 64 CTAs (two column tiles, the 4096-row products' launch from 65 columns on),
// 3.40 4.56 6.28 7.23 8.50 11.60 us, at the 1545 MHz reference clock. stream_k_unit: the same on
// 68 stream-k CTAs over one column tile, the mean ratio over the six widths, 1.18 (1.10 .. 1.25).
// launch_seconds: nsys, the fixup kernel's time beyond its bytes at 616 GB/s and the launch gap it
// adds, 5.0 us (J 128) and 1.6 us (J 64 at 121 columns), their mean: one number for a 3x spread,
// so a fitted summary of two measurements (it does not decide the 121-column flip, 126 against
// 100 us). Not derived: the SASS issue counts of a unit are linear in J
// (IMMA 4 J, I2F 4 J + 10 a thread on sm_75), the measured unit about three times their issue bound
// (2.8 at J 128 to 3.5 at J 16, at the 1.95 GHz the board ran).
inline constexpr MmqLaunchLaw kTuringMmqLaw{{5256, 7047, 9698, 11170, 13134, 17920}, 1.18, 3.3e-6};

struct GenerationRow {
    int major    = 0;
    int minor_lo = 0;
    int minor_hi = 0;
    const char* name = "";
    double bf16  = 0;
    double s8    = 0;
    double fp32_fma        = 0; // FP32 fused multiply-adds an SM issues per clock (see above)
    bool fp32acc_full_rate = false;
    double w4a8_eff_scale  = 1.0;
    double a16_cost_scale  = 1.0;
    bool mmq_candidate     = false;
    bool attention_chain   = false;
    bool mmq_work_tiles    = false;
    double mmq_tile_fixed  = 0.43; // with mmq_work_tiles: the fixed share of a column tile's step
    // With mmq_work_tiles: the launch law that prices the widths instead of mmq_tile_fixed (null:
    // the tile law, ceil(columns / J) x step_share(J, mmq_tile_fixed)).
    const MmqLaunchLaw* mmq_launch_law = nullptr;
};

inline constexpr GenerationRow kGenerations[] = {
    //  major minor     name       bf16    s8     fp32   full   w4a8  a16   mmq    chain
    {7, 0, 4, "volta", 1536.0, 0.0, 64.0, false, 1.0, 1.11, true},
    {7, 5, 9, "turing", 512.0, 2048.0, 64.0, false, 1.0, 1.11, true, false, true, 0.43,
     &kTuringMmqLaw},
    {8, 0, 0, "ga100", 2048.0, 4096.0, 64.0, true, 0.46, 1.64, false},
    {8, 1, 8, "ga10x", 512.0, 2048.0, 128.0, false, 1.0, 1.11, true, false, true, 0.20},
    {8, 9, 9, "ada", 512.0, 2048.0, 128.0, false, 1.0, 1.11, true, false, true, 0.20},
    {9, 0, 9, "hopper", 2048.0, 4096.0, 128.0, true, 0.46, 1.64, false},
    {10, 0, 9, "gb100", 2048.0, 4096.0, 128.0, true, 0.46, 1.64, false},
    {11, 0, 9, "thor", 512.0, 2048.0, 128.0, false, 1.0, 1.11, true},
    {12, 0, 9, "gb20x", 512.0, 2048.0, 128.0, false, 1.0, 1.11, true, true, true, 0.10},
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

// Whether ggml's matrix kernel takes the column tile of least work on compute capability
// major.minor (mmq_work_tiles, a measured preference): only on a generation of the table's own.
[[nodiscard]] constexpr bool mmq_work_tiles_measured(int major, int minor) noexcept {
    return generation_known(major, minor) && generation_row(major, minor).mmq_work_tiles;
}

// The fixed share of a column tile's K step on that generation (mmq_tile_fixed): the part that does
// not scale with the tile's columns; DECISIONS 69's 0.43 where the row has none of its own.
[[nodiscard]] constexpr double mmq_tile_fixed_of(int major, int minor) noexcept {
    return mmq_work_tiles_measured(major, minor) ? generation_row(major, minor).mmq_tile_fixed
                                                 : 0.43;
}

// The launch law the widths are priced by on that generation (mmq_launch_law), null where it takes
// the tile law or keeps the fewest tiles: only on a generation of the table's own.
[[nodiscard]] constexpr const MmqLaunchLaw* mmq_launch_law_of(int major, int minor) noexcept {
    return mmq_work_tiles_measured(major, minor) ? generation_row(major, minor).mmq_launch_law
                                                 : nullptr;
}

// The compute capabilities the generic selector's measured rules name (generic_select.cpp). A
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

// Whether the register-MMA cost terms of the K-quant and codebook decoders (generic_cost.cpp:
// the L2 bytes, the real occupancy from the kernel's registers, the spill bytes, the 8-warp fill)
// are measured on compute capability major.minor. Only a generation with two or more lab cards
// that confirm them takes them (AGENTS.md: a lab measurement enters as a fact or as a constant
// fitted on two or more cards of a generation); every other part keeps the old price.
//   sm_86  GA10x Ampere:       RTX 3060 + RTX 3090, the final constants fitted on both together
//   sm_120 consumer Blackwell: RTX 5090 + RTX 5070 Ti, the final constants fitted on both together
// The second card of each pair was first measured out of sample (#499: RTX 3090 1.136 -> 1.091);
// after the refit neither is a held-out board for the shipped constants, and a third board of the
// generation takes them as a transfer, not a validated fit (#301).
// Not sm_89 (Ada): the out-of-sample RTX 4060 Ti got worse (production worst 1.36 -> 1.85), a
// defect of the model (#501), so Ada comes back with its fix. Not sm_121 (GB10), the datacenter
// parts (sm_80, sm_90, sm_100) or sm_87 / sm_88, nor Turing and Volta, whose BF16 MMA is emulated
// and which got worse with the terms.
[[nodiscard]] constexpr bool rmma_terms_measured(int major, int minor) noexcept {
    return (major == 8 && minor == 6) || (major == 12 && minor == 0);
}

} // namespace ninferno
