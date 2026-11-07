#pragma once

// The constants of the route cost model (generic_gemm.cpp price_routes, a8_candidates), each
// tagged with where its value comes from:
//   Fact     a property of the chip or of CUDA (limits, ISA), the same on every board it names;
//   Counted  read off our own kernels' code (tile shapes, K per step, threads, shared bytes);
//   Fitted   chosen to make the model match ninferno_route_tune timings (cycles of an exposed
//            chain, fractions of a peak, warps to saturate): the board-held-out fits the
//            comments at their use describe. These are what the model does not derive yet.
// A new number in the cost model belongs here with its tag, so the fitted ones stay listed in one
// place (kRouteCostConstants, test_route_cost_constants) instead of hiding inside expressions.
// The board-row scales (core/generation.h: w4a8_eff_scale, a16_cost_scale) are tagged there.

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

    constexpr double operator[](std::size_t i) const { return values[i]; }

    constexpr double value() const
        requires(N == 1)
    {
        return values[0];
    }
};

using Scalar = Constant<1>;

// --- A16 tiles (price_routes) ---
// Register MMA: the exposed load latency of one 16-K step per warp, in cycles.
inline constexpr Scalar kRmmaStepCycles{{150.0}, Provenance::Fitted, "rmma_step_cycles"};
// Register MMA: K per warp step.
inline constexpr Scalar kRmmaStepK{{16.0}, Provenance::Counted, "rmma_step_k"};
// Register MMA: resident warps per SM at which the tile reaches its roofline.
inline constexpr Scalar kRmmaSaturatingWarps{{11.0}, Provenance::Fitted, "rmma_saturating_warps"};
// Register MMA of the K-quant and codebook decoders (#301): resident warps per SM at which the
// tile reaches its roofline, two for each of the four sub-partition schedulers.
inline constexpr Scalar kRmmaHeavySaturatingWarps{
    {8.0}, Provenance::Fact, "rmma_heavy_saturating_warps"};
// Register MMA, same decoders: bytes per SM clock that one SM sustains from the L2 while every SM
// reads. Half of the 64 B/clk its L1 takes from the L2; a regression of route_tune timings of the
// RTX 3060, 4090 and 5090 read 15..30, and the model's regret is flat across 16..32. No published
// number backs it, so it is fitted.
inline constexpr Scalar kL2BytesPerSmClock{{32.0}, Provenance::Fitted, "l2_bytes_per_sm_clock"};
// Register MMA, same decoders: registers per thread of a tile, 96 + 28 MT NT + 32 (MT - 1), the
// medians of ptxas (sm_89, sm_86, sm_120) over the K-quant and i-quant decoders.
inline constexpr Scalar kRmmaRegsBase{{96.0}, Provenance::Counted, "rmma_regs_base"};
inline constexpr Scalar kRmmaRegsPerTile{{28.0}, Provenance::Counted, "rmma_regs_per_tile"};
inline constexpr Scalar kRmmaRegsPerExtraRowTile{
    {32.0}, Provenance::Counted, "rmma_regs_per_extra_row_tile"};
// Registers one thread may hold (the ISA's 255), and the register file's allocation unit per warp
// (256 registers): CUDA facts of every part the model names.
inline constexpr Scalar kMaxRegsPerThread{{255.0}, Provenance::Fact, "max_regs_per_thread"};
inline constexpr Scalar kRegAllocPerWarp{{256.0}, Provenance::Fact, "reg_alloc_per_warp"};
// K split over a cluster: cycles per FP32 element of the partial sums.
inline constexpr Scalar kClusterReduceCycles{{30.0}, Provenance::Fitted, "cluster_reduce_cycles"};
// K split over a CTA's warps: cycles per element and extra warp, through shared memory.
inline constexpr Scalar kWarpReduceCycles{{2.75}, Provenance::Fitted, "warp_reduce_cycles"};
// Pipelined tiles (tiled, staged): one stage's barrier and cp.async wait, in cycles.
inline constexpr Scalar kPipelinedStageCycles{
    {500.0}, Provenance::Fitted, "pipelined_stage_cycles"};
// Pipelined tiles: K per pipeline stage.
inline constexpr Scalar kPipelinedStageK{{64.0}, Provenance::Counted, "pipelined_stage_k"};
// Pipelined tiles: resident warps per SM at which the tile reaches its roofline.
inline constexpr Scalar kPipelinedSaturatingWarps{
    {4.0}, Provenance::Fitted, "pipelined_saturating_warps"};
// Pipelined tiles: fraction of the tensor rate (staged, tiled).
inline constexpr Scalar kStagedTensorFraction{{0.9}, Provenance::Fitted, "staged_tensor_fraction"};
inline constexpr Scalar kTiledTensorFraction{{0.8}, Provenance::Fitted, "tiled_tensor_fraction"};
// Pipelined tiles: shared bytes per stage and row or token of the tile (64 K of BF16).
inline constexpr Scalar kPipelinedStageBytesPerRow{
    {128.0}, Provenance::Counted, "pipelined_stage_bytes_per_row"};
// Shared memory one CTA may take: the opt-in cap of sm_86 / sm_89 / sm_12x (99 KiB), which the
// model takes on every part (sm_80 allows 163 KiB, sm_90 227 KiB, sm_75 64 KiB): a modelling
// choice read off those parts, not a fact of every chip.
inline constexpr Scalar kMaxTileSharedBytes{
    {99.0 * 1024.0}, Provenance::Counted, "max_tile_shared_bytes"};
// Every tile: the floor of the warp slowdown factor (a lone warp runs at most 20x slow).
inline constexpr Scalar kMinWarpFill{{0.05}, Provenance::Fitted, "min_warp_fill"};

// --- A8 tiles (a8_candidates) ---
// The activation quantizer: its launch, and its bytes per value (read BF16, write the code and
// its share of the scale and sum planes).
inline constexpr Scalar kQuantLaunchSeconds{{1.7e-6}, Provenance::Fitted, "quant_launch_seconds"};
inline constexpr Scalar kQuantBytesPerValue{{4.3}, Provenance::Fitted, "quant_bytes_per_value"};
// W4A8 tiles: fraction of the int8 peak by kW4a8Schedules index (32x128, 64x128, 128x128,
// 128x64, 64x64, 128x64 with producer warps).
inline constexpr Constant<6> kW4a8Efficiency{
    {0.31, 0.28, 0.24, 0.47, 0.37, 0.98}, Provenance::Fitted, "w4a8_efficiency"};
// W4A8 tiles: shared bytes per stage and row or token (128 K of codes, scales and sums).
inline constexpr Scalar kW4a8StageBytesPerRow{
    {160.0}, Provenance::Counted, "w4a8_stage_bytes_per_row"};
// W4A8 tiles: K per step, and the exposed chain of one step (with / without producer warps).
inline constexpr Scalar kW4a8StepK{{128.0}, Provenance::Counted, "w4a8_step_k"};
inline constexpr Scalar kW4a8StepCycles{{1390.0}, Provenance::Fitted, "w4a8_step_cycles"};
inline constexpr Scalar kW4a8ProducerStepCycles{
    {1690.0}, Provenance::Fitted, "w4a8_producer_step_cycles"};
// I8 tiles: fraction of the int8 peak by kI8Schedules index (256x128, 128x128, 128x64).
inline constexpr Constant<3> kI8Efficiency{{0.53, 0.46, 0.42}, Provenance::Fitted, "i8_efficiency"};
// I8 tiles: K per pipeline stage (one activation scale group) and its barrier and copy wait.
inline constexpr Scalar kI8StageK{{64.0}, Provenance::Counted, "i8_stage_k"};
inline constexpr Scalar kI8StageCycles{{250.0}, Provenance::Fitted, "i8_stage_cycles"};
// I8 tiles: resident warps per SM at which the tile reaches its roofline.
inline constexpr Scalar kI8SaturatingWarps{{8.0}, Provenance::Fitted, "i8_saturating_warps"};
// ggml MMQ: tile rows and the token columns' granularity (ggml_mmq), its fraction of the int8
// peak, the K of one exposed chain step and that step's cycles, and a slice's fixed cost.
inline constexpr Scalar kMmqTileRows{{128.0}, Provenance::Counted, "mmq_tile_rows"};
inline constexpr Scalar kMmqColumnStep{{8.0}, Provenance::Counted, "mmq_column_step"};
inline constexpr Scalar kMmqEfficiency{{0.44}, Provenance::Fitted, "mmq_efficiency"};
inline constexpr Scalar kMmqChainK{{862.0}, Provenance::Fitted, "mmq_chain_k"};
inline constexpr Scalar kMmqChainCycles{{4075.0}, Provenance::Fitted, "mmq_chain_cycles"};
inline constexpr Scalar kMmqSliceSeconds{{18.6e-6}, Provenance::Fitted, "mmq_slice_seconds"};

// What a constant measures, for the range its values must lie in: a fraction of a peak or of a
// fill is in (0, 1]; every other kind is positive.
enum class Kind { Fraction, Cycles, Count, Bytes, Seconds };

struct Listed {
    std::string_view name;
    Provenance from;
    Kind kind;
    std::span<const double> values;
};

template <std::size_t N>
constexpr Listed listed(const Constant<N>& c, Kind kind) {
    return {c.name, c.from, kind, std::span<const double>(c.values)};
}

// Every constant above with its kind: the one descriptor test_route_cost_constants checks (names
// unique, each value in its kind's range) and its `--list` table of what is fitted. A constant
// belongs here when it is defined above.
inline constexpr Listed kRouteCostConstants[] = {
    listed(kRmmaStepCycles, Kind::Cycles),
    listed(kRmmaStepK, Kind::Count),
    listed(kRmmaSaturatingWarps, Kind::Count),
    listed(kRmmaHeavySaturatingWarps, Kind::Count),
    listed(kL2BytesPerSmClock, Kind::Bytes),
    listed(kRmmaRegsBase, Kind::Count),
    listed(kRmmaRegsPerTile, Kind::Count),
    listed(kRmmaRegsPerExtraRowTile, Kind::Count),
    listed(kMaxRegsPerThread, Kind::Count),
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
