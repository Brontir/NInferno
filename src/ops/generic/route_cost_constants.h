#pragma once

// The constants of the route cost model (generic_cost.cpp price_routes, a8_candidates), each
// tagged with where its value comes from:
//   Fact     a property of the chip or of CUDA (limits, ISA), the same on every board it names;
//   Counted  read off our own kernels' code (tile shapes, K per step, threads, shared bytes);
//   Fitted   chosen to make the model match ninferno_route_tune timings (cycles of an exposed
//            chain, fractions of a peak, warps to saturate): the board-held-out fits the
//            comments at their use describe. These are what the model does not derive yet.
// A new number in the cost model belongs here with its tag, so the fitted ones stay listed in one
// place (kRouteCostConstants, test_route_cost_constants) instead of hiding inside expressions.
// The board-row scales (core/generation.h: w4a8_eff_scale, a16_cost_scale) are tagged there.

#include "core/holdout.h"

#include <array>
#include <span>
#include <cstddef>
#include <string_view>

namespace ninferno::ops::detail::generic::route_cost {

enum class Provenance { Fact, Counted, Fitted };

template <std::size_t N>
struct Constant {
    std::array<double, N> values;
    Provenance from;
    std::string_view name;
    // Required where from == Fitted (test_route_cost_constants fails on an Unset one).
    Holdout holdout{};

    constexpr double operator[](std::size_t i) const { return values[i]; }

    constexpr double value() const
        requires(N == 1)
    {
        return values[0];
    }
};

using Scalar = Constant<1>;

// --- Holdout records of the fitted constants (core/holdout.h) ---
// Each Fitted constant below ends with one of these. h_* are not constants of the model.
// The A16 tiles: one leave-one-board-out refit of the model's constants.
inline constexpr Holdout h_a16_refit = Holdout::checked(
    "ninferno_route_tune timings of seven boards (each tile in the engine, L2 flushed; BF16, Q4 "
    "and "
    "Q8, T >= 17); the record does not name all seven",
    "each of the seven boards in turn: the constants refit with that board held out",
    "geomean time over the best tile",
    "1.095 of the best (RTX 4090 1.083, RTX 5090 1.068) against 1.110 for the F-lite rule and "
    "1.137 "
    "without the chain and reduction terms; the worst case is not recorded",
    "generic_cost.cpp price_routes comment",
    "one aggregate for the constants of the A16 model; the record gives no per-constant error; "
    "measured at 0.1.0 (54ef5203), before #499 re-priced the K-quant register MMA on sm_86 / "
    "sm_120");
// The same refit, for the two constants whose flatness was also scanned.
inline constexpr Holdout h_a16_refit_flat = Holdout::checked(
    "ninferno_route_tune timings of seven boards (each tile in the engine, L2 flushed; BF16, Q4 "
    "and "
    "Q8, T >= 17); the record does not name all seven",
    "each of the seven boards in turn: the constants refit with that board held out",
    "geomean time over the best tile",
    "1.095 of the best (RTX 4090 1.083, RTX 5090 1.068) against 1.110 for the F-lite rule and "
    "1.137 "
    "without the chain and reduction terms; the worst case is not recorded",
    "generic_cost.cpp price_routes comment",
    "the optimum is flat: this constant at 0.5..1.5x (register-MMA step) or 0.5..2x (pipeline "
    "stage) of its fit costs under 0.2 %; measured at 0.1.0 (54ef5203), before #499 re-priced the "
    "K-quant register MMA on sm_86 / sm_120");
// kMinWarpFill is used by the A16 model and by the W4A8 and I8 tiles, so it rests on both refits.
// Like its siblings it is named by neither comment, only covered as one of "the constants".
inline constexpr Holdout h_min_warp_fill = Holdout::checked(
    "A16: ninferno_route_tune timings of seven boards (BF16, Q4, Q8, T >= 17); A8: --a8 timings of "
    "RTX 4090, RTX 5090, RTX 4060 Ti, L4, A10, A100",
    "each board in turn: the constants refit with that board held out (A100 in sample only)",
    "A16: geomean time over the best tile; A8: time over the best route of either class",
    "A16 1.095 of the best (4090 1.083, 5090 1.068); A8 1.06..1.09 on the consumer boards, A100 "
    "1.13 in sample; no error for this floor on its own",
    "generic_cost.cpp price_routes comment; generic_gemm.cpp route_tile comment",
    "the A16 figures were measured at 0.1.0 (54ef5203), before #499 re-priced the K-quant register "
    "MMA on sm_86 / sm_120");
// The A8 tiles (W4A8, I8, ggml MMQ) and their activation quantizer.
inline constexpr Holdout h_a8_refit = Holdout::checked(
    "ninferno_route_tune --a8 and --a8 --epilogue timings of RTX 4090, RTX 5090, RTX 4060 Ti, L4, "
    "A10, A100 (six formats, n 48..28672, T 17..2048, quantizer in the sample; SwiGLU and "
    "AddResidual on the A100)",
    "each board in turn: the constants refit with that board held out (the minor-0 row, A100, has "
    "one board and no held-out value)",
    "time over the best route of either class (A16 or A8)",
    "1.06..1.09 of the best on the consumer boards; A100 1.13 in sample only",
    "generic_gemm.cpp route_tile comment",
    "one aggregate for the A8 constants; the record gives no per-constant error");
// The register MMA of the K-quant and codebook decoders (#499, #301).
inline constexpr Holdout h_rmma_heavy_warps = Holdout::none(
    "RTX 3060 + RTX 3090 and RTX 5090 + RTX 5070 Ti, cost-model replica",
    "6 and 8 warps give the same regret on both generations' cards; two warps for each of the four "
    "schedulers is the least that keeps them fed, the step latency that would derive it is not "
    "measured",
    "#499 Constants table");
inline constexpr Holdout h_l2_ga10x = Holdout::none(
    "RTX 3060 + RTX 3090 together (12/16/24/32/48 tried; replica geomean regret 3060 1.068, 3090 "
    "1.098; master 1.165, 1.174)",
    "after the refit on both cards neither is held out; a third GA10x board takes the value as a "
    "transfer",
    "#499 Constants table and gate; core/generation.h rmma_terms_measured",
    "the RTX 3090 was out of sample only for the first version of the terms: production geomean "
    "1.136 -> 1.091 (90 cases, 2 passes x 7 samples, worst 1.83 -> 1.83)");
inline constexpr Holdout h_l2_gb20x = Holdout::none(
    "RTX 5090 + RTX 5070 Ti together (replica geomean regret 5090 1.047, 5070 Ti 1.041; master "
    "1.135, 1.080; 16 gives 1.048 / 1.043, 32 gives 1.064 / 1.050)",
    "after the refit on both cards neither is held out; the 24 comes from the replica and was not "
    "re-timed on the cards",
    "#499 Constants table and gate; core/generation.h rmma_terms_measured",
    "the RTX 5070 Ti was held out for the first version of the terms only");
inline constexpr Holdout h_spill_accesses = Holdout::none(
    "a modelling assumption (one store and one load of a spilled byte a step), not a fit to data",
    "the spill term as a whole (with or without) moves the regret by under 0.5 % where it applies; "
    "no scan of this value was run",
    "#499 Constants table");

// --- A16 tiles (price_routes) ---
// Register MMA: the exposed load latency of one 16-K step per warp, in cycles.
inline constexpr Scalar kRmmaStepCycles{
    {150.0}, Provenance::Fitted, "rmma_step_cycles", h_a16_refit_flat};
// Register MMA: K per warp step.
inline constexpr Scalar kRmmaStepK{{16.0}, Provenance::Counted, "rmma_step_k"};
// Register MMA: resident warps per SM at which the tile reaches its roofline.
inline constexpr Scalar kRmmaSaturatingWarps{
    {11.0}, Provenance::Fitted, "rmma_saturating_warps", h_a16_refit};
// Register MMA of the K-quant and codebook decoders (#301): resident warps per SM at which the
// tile reaches its roofline. Two warps for each of the four sub-partition schedulers is the least
// that can keep every scheduler fed; the step latency that would derive it is not measured here, so
// it is fitted. The regret of both fitted generations is the same at 6 and at 8 warps (RTX 3060 +
// RTX 3090, RTX 5090 + RTX 5070 Ti, cost-model replica). Used only where two or more lab cards
// confirm the register-MMA terms (rmma_terms_measured: sm_86 and sm_120).
inline constexpr Scalar kRmmaHeavySaturatingWarps{
    {8.0}, Provenance::Fitted, "rmma_heavy_saturating_warps", h_rmma_heavy_warps};
// Register MMA, same decoders: bytes per SM clock that one SM sustains from the L2 while every SM
// reads (the 64 B/clk port of its L1 is the ceiling; route_tune timings read 15..30). No published
// number backs it, so it is fitted, one value per generation on both of its cards together
// (cost-model replica, mean geomean regret of the two cards over 12/16/24/32/48):
//   sm_86  (GA10x):  RTX 3060 + RTX 3090 -> 32 (3060 1.068, 3090 1.098; master 1.165, 1.174)
//   sm_120 (GB20x):  RTX 5090 + RTX 5070 Ti -> 24 (1.047, 1.041; master 1.135, 1.080); the 24 comes
//                    from the replica and was not re-timed on the cards
inline constexpr Scalar kL2BytesPerSmClockGa10x{
    {32.0}, Provenance::Fitted, "l2_bytes_per_sm_clock_ga10x", h_l2_ga10x};
inline constexpr Scalar kL2BytesPerSmClockGb20x{
    {24.0}, Provenance::Fitted, "l2_bytes_per_sm_clock_gb20x", h_l2_gb20x};
// Register MMA, same decoders: K per step of their decoders (four lanes of one 32-value run), and
// what a spilled register costs per step: one store and one load of its local-memory byte through
// the L1, whose data path is 128 B per clock and SM.
inline constexpr Scalar kRmmaHeavyStepK{{128.0}, Provenance::Counted, "rmma_heavy_step_k"};
// The spill count is a modelling assumption (a spilled value is stored once and loaded once a
// step; the compiler may keep it longer), so it is tagged Fitted, not Counted: it moved the regret
// by under 0.5 % where it applies.
inline constexpr Scalar kSpillAccessesPerStep{
    {2.0}, Provenance::Fitted, "spill_accesses_per_step", h_spill_accesses};
inline constexpr Scalar kL1BytesPerSmClock{{128.0}, Provenance::Fact, "l1_bytes_per_sm_clock"};
// The register file's allocation unit per warp (256 registers): a CUDA fact of every part the
// model names.
inline constexpr Scalar kRegAllocPerWarp{{256.0}, Provenance::Fact, "reg_alloc_per_warp"};
// K split over a cluster: cycles per FP32 element of the partial sums.
inline constexpr Scalar kClusterReduceCycles{
    {30.0}, Provenance::Fitted, "cluster_reduce_cycles", h_a16_refit};
// K split over a CTA's warps: cycles per element and extra warp, through shared memory.
inline constexpr Scalar kWarpReduceCycles{
    {2.75}, Provenance::Fitted, "warp_reduce_cycles", h_a16_refit};
// Pipelined tiles (tiled, staged): one stage's barrier and cp.async wait, in cycles.
inline constexpr Scalar kPipelinedStageCycles{
    {500.0}, Provenance::Fitted, "pipelined_stage_cycles", h_a16_refit_flat};
// Pipelined tiles: K per pipeline stage.
inline constexpr Scalar kPipelinedStageK{{64.0}, Provenance::Counted, "pipelined_stage_k"};
// Pipelined tiles: resident warps per SM at which the tile reaches its roofline.
inline constexpr Scalar kPipelinedSaturatingWarps{
    {4.0}, Provenance::Fitted, "pipelined_saturating_warps", h_a16_refit};
// Pipelined tiles: fraction of the tensor rate (staged, tiled).
inline constexpr Scalar kStagedTensorFraction{
    {0.9}, Provenance::Fitted, "staged_tensor_fraction", h_a16_refit};
inline constexpr Scalar kTiledTensorFraction{
    {0.8}, Provenance::Fitted, "tiled_tensor_fraction", h_a16_refit};
// Pipelined tiles: shared bytes per stage and row or token of the tile (64 K of BF16).
inline constexpr Scalar kPipelinedStageBytesPerRow{
    {128.0}, Provenance::Counted, "pipelined_stage_bytes_per_row"};
// Shared memory one CTA may take: the opt-in cap of sm_86 / sm_89 / sm_12x (99 KiB), which the
// model takes on every part (sm_80 allows 163 KiB, sm_90 227 KiB, sm_75 64 KiB): a modelling
// choice read off those parts, not a fact of every chip.
inline constexpr Scalar kMaxTileSharedBytes{
    {99.0 * 1024.0}, Provenance::Counted, "max_tile_shared_bytes"};
// Every tile: the floor of the warp slowdown factor (a lone warp runs at most 20x slow).
inline constexpr Scalar kMinWarpFill{{0.05}, Provenance::Fitted, "min_warp_fill", h_min_warp_fill};

// --- A8 tiles (a8_candidates) ---
// The activation quantizer: its launch, and its bytes per value (read BF16, write the code and
// its share of the scale and sum planes).
inline constexpr Scalar kQuantLaunchSeconds{
    {1.7e-6}, Provenance::Fitted, "quant_launch_seconds", h_a8_refit};
inline constexpr Scalar kQuantBytesPerValue{
    {4.3}, Provenance::Fitted, "quant_bytes_per_value", h_a8_refit};
// W4A8 tiles: fraction of the int8 peak by kW4a8Schedules index (32x128, 64x128, 128x128,
// 128x64, 64x64, 128x64 with producer warps).
inline constexpr Constant<6> kW4a8Efficiency{
    {0.31, 0.28, 0.24, 0.47, 0.37, 0.98}, Provenance::Fitted, "w4a8_efficiency", h_a8_refit};
// W4A8 tiles: shared bytes per stage and row or token (128 K of codes, scales and sums).
inline constexpr Scalar kW4a8StageBytesPerRow{
    {160.0}, Provenance::Counted, "w4a8_stage_bytes_per_row"};
// W4A8 tiles: K per step, and the exposed chain of one step (with / without producer warps).
inline constexpr Scalar kW4a8StepK{{128.0}, Provenance::Counted, "w4a8_step_k"};
inline constexpr Scalar kW4a8StepCycles{
    {1390.0}, Provenance::Fitted, "w4a8_step_cycles", h_a8_refit};
inline constexpr Scalar kW4a8ProducerStepCycles{
    {1690.0}, Provenance::Fitted, "w4a8_producer_step_cycles", h_a8_refit};
// I8 tiles: fraction of the int8 peak by kI8Schedules index (256x128, 128x128, 128x64).
inline constexpr Constant<3> kI8Efficiency{
    {0.53, 0.46, 0.42}, Provenance::Fitted, "i8_efficiency", h_a8_refit};
// I8 tiles: K per pipeline stage (one activation scale group) and its barrier and copy wait.
inline constexpr Scalar kI8StageK{{64.0}, Provenance::Counted, "i8_stage_k"};
inline constexpr Scalar kI8StageCycles{{250.0}, Provenance::Fitted, "i8_stage_cycles", h_a8_refit};
// I8 tiles: resident warps per SM at which the tile reaches its roofline.
inline constexpr Scalar kI8SaturatingWarps{
    {8.0}, Provenance::Fitted, "i8_saturating_warps", h_a8_refit};
// ggml MMQ: tile rows and the token columns' granularity (ggml_mmq), its fraction of the int8
// peak, the K of one exposed chain step and that step's cycles, and a slice's fixed cost.
inline constexpr Scalar kMmqTileRows{{128.0}, Provenance::Counted, "mmq_tile_rows"};
inline constexpr Scalar kMmqColumnStep{{8.0}, Provenance::Counted, "mmq_column_step"};
inline constexpr Scalar kMmqEfficiency{{0.44}, Provenance::Fitted, "mmq_efficiency", h_a8_refit};
inline constexpr Scalar kMmqChainK{{862.0}, Provenance::Fitted, "mmq_chain_k", h_a8_refit};
inline constexpr Scalar kMmqChainCycles{
    {4075.0}, Provenance::Fitted, "mmq_chain_cycles", h_a8_refit};
inline constexpr Scalar kMmqSliceSeconds{
    {18.6e-6}, Provenance::Fitted, "mmq_slice_seconds", h_a8_refit};

// What a constant measures, for the range its values must lie in: a fraction of a peak or of a
// fill is in (0, 1]; every other kind is positive (BytesPerClock: bytes an SM moves a clock).
enum class Kind { Fraction, Cycles, Count, Bytes, BytesPerClock, Seconds };

struct Listed {
    std::string_view name;
    Provenance from;
    Kind kind;
    std::span<const double> values;
    const Holdout& holdout;
};

template <std::size_t N>
constexpr Listed listed(const Constant<N>& c, Kind kind) {
    return {c.name, c.from, kind, std::span<const double>(c.values), c.holdout};
}

// Every constant above with its kind: the one descriptor test_route_cost_constants checks (names
// unique, each value in its kind's range) and its `--list` table of what is fitted. A constant
// belongs here when it is defined above.
inline constexpr Listed kRouteCostConstants[] = {
    listed(kRmmaStepCycles, Kind::Cycles),
    listed(kRmmaStepK, Kind::Count),
    listed(kRmmaSaturatingWarps, Kind::Count),
    listed(kRmmaHeavySaturatingWarps, Kind::Count),
    listed(kL2BytesPerSmClockGa10x, Kind::BytesPerClock),
    listed(kL2BytesPerSmClockGb20x, Kind::BytesPerClock),
    listed(kRmmaHeavyStepK, Kind::Count),
    listed(kSpillAccessesPerStep, Kind::Count),
    listed(kL1BytesPerSmClock, Kind::BytesPerClock),
    listed(kRegAllocPerWarp, Kind::Count),
    listed(kClusterReduceCycles, Kind::Cycles),
    listed(kWarpReduceCycles, Kind::Cycles),
    listed(kPipelinedStageCycles, Kind::Cycles),
    listed(kPipelinedStageK, Kind::Count),
    listed(kPipelinedSaturatingWarps, Kind::Count),
    listed(kStagedTensorFraction, Kind::Fraction),
    listed(kTiledTensorFraction, Kind::Fraction),
    listed(kPipelinedStageBytesPerRow, Kind::Bytes),
    listed(kMaxTileSharedBytes, Kind::Bytes),
    listed(kMinWarpFill, Kind::Fraction),
    listed(kQuantLaunchSeconds, Kind::Seconds),
    listed(kQuantBytesPerValue, Kind::Bytes),
    listed(kW4a8Efficiency, Kind::Fraction),
    listed(kW4a8StageBytesPerRow, Kind::Bytes),
    listed(kW4a8StepK, Kind::Count),
    listed(kW4a8StepCycles, Kind::Cycles),
    listed(kW4a8ProducerStepCycles, Kind::Cycles),
    listed(kI8Efficiency, Kind::Fraction),
    listed(kI8StageK, Kind::Count),
    listed(kI8StageCycles, Kind::Cycles),
    listed(kI8SaturatingWarps, Kind::Count),
    listed(kMmqTileRows, Kind::Count),
    listed(kMmqColumnStep, Kind::Count),
    listed(kMmqEfficiency, Kind::Fraction),
    listed(kMmqChainK, Kind::Count),
    listed(kMmqChainCycles, Kind::Cycles),
    listed(kMmqSliceSeconds, Kind::Seconds),
};

} // namespace ninferno::ops::detail::generic::route_cost
