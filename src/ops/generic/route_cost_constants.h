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
// A Counted constant of a kernel is read from its family's record (family_facts.h, #619 step 3),
// which the kernel's template is static_asserted against: the value is not retyped here.

#include "core/cost_constants.h"
#include "ops/generic/family_facts.h"

#include <array>
#include <optional>
#include <stdexcept>
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
// Register MMA: K per warp step (one m16n8k16 MMA's).
inline constexpr Scalar kRmmaStepK{
    {static_cast<double>(kRegisterMmaFacts.mma_k)}, Provenance::Counted, "rmma_step_k"};
// Register MMA: resident warps per SM at which the tile reaches its roofline.
inline constexpr Scalar kRmmaSaturatingWarps{{11.0}, Provenance::Fitted, "rmma_saturating_warps"};
// Register MMA of the K-quant and codebook decoders (#301): resident warps per SM at which the
// tile reaches its roofline. Two warps for each of the four sub-partition schedulers is the least
// that can keep every scheduler fed; the step latency that would derive it is not measured here, so
// it is fitted. The regret of both fitted generations is the same at 6 and at 8 warps (RTX 3060 +
// RTX 3090, RTX 5090 + RTX 5070 Ti, cost-model replica). Used only where two or more lab cards
// confirm the register-MMA terms (rmma_terms_measured: sm_86 and sm_120).
inline constexpr Scalar kRmmaHeavySaturatingWarps{
    {8.0}, Provenance::Fitted, "rmma_heavy_saturating_warps"};
// Register MMA, same decoders: bytes per SM clock that one SM sustains from the L2 while every SM
// reads (the 64 B/clk port of its L1 is the ceiling; route_tune timings read 15..30). No published
// number backs it, so it is fitted, one value per generation on both of its cards together
// (cost-model replica, mean geomean regret of the two cards over 12/16/24/32/48):
//   sm_86  (GA10x):  RTX 3060 + RTX 3090 -> 32 (3060 1.068, 3090 1.098; master 1.165, 1.174)
//   sm_120 (GB20x):  RTX 5090 + RTX 5070 Ti -> 24 (1.047, 1.041; master 1.135, 1.080); the 24 comes
//                    from the replica and was not re-timed on the cards
inline constexpr Scalar kL2BytesPerSmClockGa10x{
    {32.0}, Provenance::Fitted, "l2_bytes_per_sm_clock_ga10x"};
inline constexpr Scalar kL2BytesPerSmClockGb20x{
    {24.0}, Provenance::Fitted, "l2_bytes_per_sm_clock_gb20x"};
// Register MMA, same decoders: K per step of their decoders (four lanes of one 32-value run), and
// what a spilled register costs per step: one store and one load of its local-memory byte through
// the L1, whose data path is 128 B per clock and SM.
inline constexpr Scalar kRmmaHeavyStepK{
    {static_cast<double>(kRegisterMmaFacts.step_k)}, Provenance::Counted, "rmma_heavy_step_k"};
// The spill count is a modelling assumption (a spilled value is stored once and loaded once a
// step; the compiler may keep it longer), so it is tagged Fitted, not Counted: it moved the regret
// by under 0.5 % where it applies.
inline constexpr Scalar kSpillAccessesPerStep{{2.0}, Provenance::Fitted, "spill_accesses_per_step"};
inline constexpr Scalar kL1BytesPerSmClock{{128.0}, Provenance::Fact, "l1_bytes_per_sm_clock"};
// Which register-MMA law the binary prices (rmma_counted_terms.h, #613 PR-7a): false, the fitted
// spill and weight-only tensor terms above (what the selector goldens hold); true, the kernel's
// counted K loop. One law a binary, no environment switch; a test flips it for its process through
// rmma_counted_law_for_test (generic_gemm.h). It ships false until the A/B of the counted law.
inline constexpr bool kRmmaCountedLaw = false;
// The L1 hit latency a warp waits on a spilled register's reload (LDL); the counted law divides it
// by the warps a scheduler holds, which overlap each other's waits. One value for every
// architecture: 32 clocks, Turing's in the published microbenchmarks (33 on Ampere, 34 on Ada and
// Blackwell there, 3-6 % of this term). It stands in for the per-architecture measured latency (the
// ISA probe's row, #613), which the counted law is to read before it prices anything: the law is
// off (kRmmaCountedLaw), so no price depends on this value. A fact of the SM, not of our sweeps.
inline constexpr Scalar kL1HitLatencyCycles{{32.0}, Provenance::Fact, "l1_hit_latency_cycles"};
// The register file's allocation unit per warp (256 registers): a CUDA fact of every part the
// model names.
inline constexpr Scalar kRegAllocPerWarp{{256.0}, Provenance::Fact, "reg_alloc_per_warp"};
// K split over a cluster: cycles per FP32 element of the partial sums.
inline constexpr Scalar kClusterReduceCycles{{30.0}, Provenance::Fitted, "cluster_reduce_cycles"};
// K split over a CTA's warps: cycles per element and extra warp, through shared memory.
inline constexpr Scalar kWarpReduceCycles{{2.75}, Provenance::Fitted, "warp_reduce_cycles"};
// Pipelined tiles (tiled, staged): one stage's barrier and cp.async wait, in cycles.
inline constexpr Scalar kPipelinedStageCycles{
    {500.0}, Provenance::Fitted, "pipelined_stage_cycles"};
// Pipelined tiles: K per pipeline stage (one value for both tiles).
inline constexpr Scalar kPipelinedStageK{
    {static_cast<double>(kStagedMmaFacts.stage_k)}, Provenance::Counted, "pipelined_stage_k"};
static_assert(kTiledMmaFacts.stage_k == kStagedMmaFacts.stage_k);
// Pipelined tiles: resident warps per SM at which the tile reaches its roofline.
inline constexpr Scalar kPipelinedSaturatingWarps{
    {4.0}, Provenance::Fitted, "pipelined_saturating_warps"};
// Pipelined tiles: fraction of the tensor rate (staged, tiled).
inline constexpr Scalar kStagedTensorFraction{{0.9}, Provenance::Fitted, "staged_tensor_fraction"};
inline constexpr Scalar kTiledTensorFraction{{0.8}, Provenance::Fitted, "tiled_tensor_fraction"};
// Pipelined tiles: shared bytes per stage and row or token of the tile (64 K of BF16): the staged
// tile's, and the tiled one's per token. The price takes it for the tiled tile's weight rows too,
// which stage the codec's raw chunks (TiledShared::raw) and decode into one BF16 buffer.
inline constexpr Scalar kPipelinedStageBytesPerRow{
    {static_cast<double>(kStagedMmaFacts.stage_row_bytes)},
    Provenance::Counted,
    "pipelined_stage_bytes_per_row"};
static_assert(kStagedMmaFacts.stage_token_bytes == kStagedMmaFacts.stage_row_bytes &&
              kTiledMmaFacts.stage_token_bytes == kStagedMmaFacts.stage_row_bytes);
// Shared memory one CTA may take: the opt-in cap of sm_86 / sm_89 / sm_12x (99 KiB), which the
// model takes on every part (sm_80 allows 163 KiB, sm_90 227 KiB, sm_75 64 KiB): a modelling
// choice read off those parts, not a fact of every chip. test_route_cost_constants holds it equal
// to those parts' chip column of the archs table (core/arch_registry.h smem_per_block_optin).
inline constexpr Scalar kMaxTileSharedBytes{
    {99.0 * 1024.0}, Provenance::Counted, "max_tile_shared_bytes"};
// Every tile: the floor of the warp slowdown factor (a lone warp runs at most 20x slow).
inline constexpr Scalar kMinWarpFill{{0.05}, Provenance::Fitted, "min_warp_fill"};

// --- A8 tiles (a8_candidates) ---
// The activation quantizer: its launch (a kernel with its scale and sum planes, not the bare graph
// node of ninferno::launch_node_seconds, 1.5e-6 on Ada; the two stay separate facts, #617 B), and
// its bytes per value (read BF16, write the code and its share of the scale and sum planes).
// The graph launch node (core/cost_constants.h launch_node_seconds, the one launch constant): Ada's
// per-launch t0 from the census of three cards, read by the speculation round cost. Listed here
// for its provenance.
inline constexpr Scalar kLaunchNodeAda{
    {launch_node_seconds("ada")}, Provenance::Fitted, "launch_node_seconds_ada"};
// Both stay the price while the completeness terms are off (kCompletenessTerms below): they were
// fitted (h_a8_refit) on route_tune timings that hold the quantizer, the output write and the
// launches together, so the fit absorbed those.
inline constexpr Scalar kQuantLaunchSeconds{{1.7e-6}, Provenance::Fitted, "quant_launch_seconds"};
inline constexpr Scalar kQuantBytesPerValue{{4.3}, Provenance::Fitted, "quant_bytes_per_value"};
// The completeness terms (#613 PR-8a, kernel_roofline.h Completeness): the output write, the
// residual read of AddResidual, the A8 quantizer by the counts below with its launch, and the
// kernel's own launch from the architecture's launch facts (core/launch_facts.h). false: computed
// (price_tool prints them), never offered, and the price is golden-equal to the one before; true:
// the selector adds them. One switch a binary, as kRmmaCountedLaw; a test flips it for its process
// (completeness_terms_for_test, generic_gemm.h). It ships false: the A8 efficiencies and the A16
// scale were fitted with these costs inside the timings, so switching it on alone would count
// them twice. It flips per family with that family's counted law (PR-5c, PR-6, PR-7b; #613 PR-8d).
inline constexpr bool kCompletenessTerms = false;
// The activation quantizers' DRAM bytes per value, counted from their kernels and static_asserted
// there against the block each thread writes:
//   W4A8 (generic_w4a8.cu w4a8_quantize_kernel): a thread a 32-value block reads 64 B of BF16 and
//     writes 32 B of codes, a 4 B scale and a 4 B sum: 104 B / 32 = 3.25 B a value of K; a block
//     of the row's padding past K (the row is w4a8_ld(K) values, whole 128s) reads nothing and
//     writes its zeros: 40 B / 32 = 1.25 B a value;
//   I8 (generic_i8.cu i8_quantize_kernel): a thread a 64-value group reads 128 B and writes 64 B of
//     codes, a 4 B scale and a 4 B sum: 200 B / 64 = 3.125 B a value of a real token; a padding
//     token (tokens padded to 256, i8_padded_tokens) writes its zeros: 72 B / 64 = 1.125 B;
//   ggml MMQ (ggml_mmq.cu quantize_kernel): 2 B of BF16 read a value, one block_q8_1_mmq of 144 B
//     written a 128 values: 2 + 1.125 = 3.125 B a value, per weight type's grid.
inline constexpr Scalar kQuantW4a8BytesPerValue{
    {3.25}, Provenance::Counted, "quant_w4a8_bytes_per_value"};
inline constexpr Scalar kQuantW4a8PadBytesPerValue{
    {1.25}, Provenance::Counted, "quant_w4a8_pad_bytes_per_value"};
inline constexpr Scalar kQuantI8BytesPerValue{
    {3.125}, Provenance::Counted, "quant_i8_bytes_per_value"};
inline constexpr Scalar kQuantI8PadBytesPerValue{
    {1.125}, Provenance::Counted, "quant_i8_pad_bytes_per_value"};
inline constexpr Scalar kQuantMmqBytesPerValue{
    {3.125}, Provenance::Counted, "quant_mmq_bytes_per_value"};
// The quantizers' CTAs: W4A8 and I8 a thread a block or group in CTAs of 256 threads; MMQ a CTA of
// 128 threads a column and 512 values of K (ggml_mmq.cu quantize_activation). Their dispatch ramp
// reads these grids (launch_facts.h ramp_seconds).
inline constexpr Scalar kQuantThreadsPerCta{{256.0}, Provenance::Counted, "quant_threads_per_cta"};
inline constexpr Scalar kQuantMmqValuesPerCta{
    {512.0}, Provenance::Counted, "quant_mmq_values_per_cta"};
// W4A8 tiles: fraction of the int8 peak by kW4a8Schedules index (32x128, 64x128, 128x128,
// 128x64, 64x64, 128x64 with producer warps).
inline constexpr Constant<6> kW4a8Efficiency{
    {0.31, 0.28, 0.24, 0.47, 0.37, 0.98}, Provenance::Fitted, "w4a8_efficiency"};
// W4A8 tiles: shared bytes per stage and row or token (128 K of codes, scales and sums).
inline constexpr Scalar kW4a8StageBytesPerRow{{static_cast<double>(kW4a8MmaFacts.stage_row_bytes)},
                                              Provenance::Counted,
                                              "w4a8_stage_bytes_per_row"};
static_assert(kW4a8MmaFacts.stage_token_bytes == kW4a8MmaFacts.stage_row_bytes);
// W4A8 tiles: K per step, and the exposed chain of one step (with / without producer warps).
inline constexpr Scalar kW4a8StepK{
    {static_cast<double>(kW4a8MmaFacts.step_k)}, Provenance::Counted, "w4a8_step_k"};
inline constexpr Scalar kW4a8StepCycles{{1390.0}, Provenance::Fitted, "w4a8_step_cycles"};
inline constexpr Scalar kW4a8ProducerStepCycles{
    {1690.0}, Provenance::Fitted, "w4a8_producer_step_cycles"};
// I8 tiles: fraction of the int8 peak by kI8Schedules index (256x128, 128x128, 128x64).
inline constexpr Constant<3> kI8Efficiency{{0.53, 0.46, 0.42}, Provenance::Fitted, "i8_efficiency"};
// I8 tiles: K per pipeline stage (one activation scale group) and its barrier and copy wait.
inline constexpr Scalar kI8StageK{
    {static_cast<double>(kI8MmaFacts.stage_k)}, Provenance::Counted, "i8_stage_k"};
inline constexpr Scalar kI8StageCycles{{250.0}, Provenance::Fitted, "i8_stage_cycles"};
// I8 tiles: resident warps per SM at which the tile reaches its roofline.
inline constexpr Scalar kI8SaturatingWarps{{8.0}, Provenance::Fitted, "i8_saturating_warps"};
// ggml MMQ: tile rows and the token columns' granularity (ggml_mmq), its fraction of the int8
// peak, the K of one exposed chain step and that step's cycles, and a slice's fixed cost.
inline constexpr Scalar kMmqTileRows{
    {static_cast<double>(kGgmlMmqFacts.tile_rows)}, Provenance::Counted, "mmq_tile_rows"};
inline constexpr Scalar kMmqColumnStep{
    {static_cast<double>(kGgmlMmqFacts.column_step)}, Provenance::Counted, "mmq_column_step"};
inline constexpr Scalar kMmqEfficiency{{0.44}, Provenance::Fitted, "mmq_efficiency"};
inline constexpr Scalar kMmqChainK{{862.0}, Provenance::Fitted, "mmq_chain_k"};
inline constexpr Scalar kMmqChainCycles{{4075.0}, Provenance::Fitted, "mmq_chain_cycles"};
inline constexpr Scalar kMmqSliceSeconds{{18.6e-6}, Provenance::Fitted, "mmq_slice_seconds"};

// What a constant measures, for the range its values must lie in: a fraction of a peak or of a
// fill is in (0, 1]; every other kind is positive (BytesPerClock: bytes an SM moves a clock).
enum class Kind { Fraction, Cycles, Count, Bytes, BytesPerClock, Seconds };

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
    listed(kL2BytesPerSmClockGa10x, Kind::BytesPerClock),
    listed(kL2BytesPerSmClockGb20x, Kind::BytesPerClock),
    listed(kRmmaHeavyStepK, Kind::Count),
    listed(kSpillAccessesPerStep, Kind::Count),
    listed(kL1BytesPerSmClock, Kind::BytesPerClock),
    listed(kL1HitLatencyCycles, Kind::Cycles),
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
    listed(kLaunchNodeAda, Kind::Seconds),
    listed(kQuantLaunchSeconds, Kind::Seconds),
    listed(kQuantBytesPerValue, Kind::Bytes),
    listed(kQuantW4a8BytesPerValue, Kind::Bytes),
    listed(kQuantW4a8PadBytesPerValue, Kind::Bytes),
    listed(kQuantI8BytesPerValue, Kind::Bytes),
    listed(kQuantI8PadBytesPerValue, Kind::Bytes),
    listed(kQuantMmqBytesPerValue, Kind::Bytes),
    listed(kQuantThreadsPerCta, Kind::Count),
    listed(kQuantMmqValuesPerCta, Kind::Count),
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

// A Fitted constant whose calibration measurement is not taken yet: it has no value (a Constant
// cannot hold "unknown"), and the code that reads it keeps its default while `value` is empty.
// test_route_cost_constants checks every entry is Fitted and empty; a calibrated one moves into
// kRouteCostConstants with its Holdout record.
struct PendingFit {
    std::string_view name;
    Provenance from;
    std::optional<double> value;
};

inline constexpr PendingFit kPendingFits[] = {
    // The fused text qk-norm + RoPE's per-warp chain excess over the three separate kernels, in
    // SM cycles (sm_wave_rules.h launch_amortized_tokens). Calibrated by the crossover bench
    // (bench/ops/rmsnorm_rope_bench.cu --crossover, #617 G).
    {"fused_text_excess_cycles", Provenance::Fitted, std::nullopt},
};

// The pending fit named `name` (a name not in kPendingFits is a bug: the lookup throws, which fails
// a constant evaluation at compile time).
constexpr const PendingFit& pending_fit(std::string_view name) {
    for (const auto& fit : kPendingFits) {
        if (fit.name == name) return fit;
    }
    throw std::logic_error("unknown pending fit");
}

} // namespace ninferno::ops::detail::generic::route_cost
