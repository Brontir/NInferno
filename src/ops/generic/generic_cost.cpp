// The route cost model (refactor B1): price_routes prices every admitted route of a call of 17
// tokens or more by the roofline of its tiles on the device's facts (the A16 tiles through
// kernel_roofline.h, the A8 ones in a8_candidates), cost_route takes the cheapest. Its constants
// and their provenance: route_cost_constants.h.
#include "core/mmq_columns.h"
#include "core/device_class.h"
#include "core/generation.h"
#include "core/weight.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_ggml_internal.h"
#include "ops/generic/generic_schedules.h"
#include "ops/generic/generic_select_internal.h"
#include "ops/generic/ggml_mmq/ggml_mmq.h"
#include "ops/generic/kernel_roofline.h"
#include "ops/generic/route_cost_constants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

namespace ninferno::ops::detail::generic {

namespace {

template <class Offer>
void a8_candidates(const Problem& problem, const Weight& weight, std::int32_t tokens,
                   const DeviceClass& d, const Offer& offer) {
    // BF16 operands in the MMA itself (sm_80 on) imply tensor mma.sync (device_features_for).
    if (!d.features.mma_bf16_native || !allows_a8(problem.policy)) return;
    // A device without a roofline in its facts (an unknown id whose driver reports no clock or bus
    // width) has none here: the A16 routes stand alone. The clock and the bandwidth are the
    // board's reference facts (DeviceClass::facts, B2), never the driver's clocks.
    if (d.sms <= 0 || !d.facts.has_roofline()) return;
    // The A8 routes' cost rests on the generation's fitted W4A8 efficiency and A16 factor; a lent
    // row carries neither (#300), so a generation the table does not know keeps the A16 routes.
    const auto fitted = fitted_route_scales(d.major, d.minor);
    if (!fitted) return;
    const GenerationRow& g = generation_row(d.major, d.minor);
    const double clk     = d.facts.reference_clock_hz();
    const double bw      = d.facts.dram_bytes_per_s();
    const double sms     = d.sms;
    const double n = problem.n, k = problem.k, t = tokens;
    const double bpw     = static_cast<double>(weight.payload_bytes) / (n * k);
    const double peak_sm = g.s8 * clk;
    const double scale   = fitted->w4a8_eff;
    const double bw_sm   = bw / sms;
    const double max_thr = std::max(d.max_threads_per_sm, 1);
    const double max_cta = std::max(d.max_blocks_per_sm, 1);
    const double smem_sm = static_cast<double>(d.smem_per_sm);
    const auto cdiv      = [](double a, double b) { return std::ceil(a / b); };
    const auto quant     = [&](double cols) {
        return route_cost::kQuantLaunchSeconds.value() +
               cols * k * route_cost::kQuantBytesPerValue.value() / bw;
    };
    // Fraction of the int8 peak by kW4a8Schedules index: 32x128, 64x128, 128x128, 128x64, 64x64,
    // 128x64 with producer warps.
    constexpr auto kW4a8Eff = route_cost::kW4a8Efficiency;
    if (allows_w4a8(weight, problem.policy)) {
        for (int i = 0; i < w4a8_schedule_count(); ++i) {
            const auto s         = w4a8_schedule_info(i);
            const Route r{Family::W4a8Mma, static_cast<std::uint8_t>(i)};
            const auto tile      = *route_tile(r, problem.epilogue, tokens);
            const double rows_w  = s.block_n, cols = tile.tokens;
            const double rows_o  = tile.out_rows;
            const double threads =
                s.producers > 0 ? 384.0 : rows_w == 64.0 && cols == 64.0 ? 128.0 : 256.0;
            const double smem = (s.producers > 0 ? 3.0 : 2.0) * (rows_w + cols) *
                                route_cost::kW4a8StageBytesPerRow.value();
            const double occ     = std::max(
                1.0, std::min({std::floor(max_thr / threads), max_cta, std::floor(smem_sm / smem)}));
            const double ctas  = cdiv(n, rows_o) * cdiv(t, cols);
            const double q     = std::ceil(ctas / sms);
            const double warps = std::min(1.0, ctas / sms) * std::min(q, occ) * threads / 32;
            const double mem   = std::max(q * rows_o * k * bpw / bw_sm, n * k * bpw / bw);
            const double mma   = q * 2.0 * rows_w * cols * k / peak_sm;
            const double cap   = s.producers > 0 ? route_cost::kW4a8ProducerStepCycles.value()
                                                 : route_cost::kW4a8StepCycles.value();
            const double chain =
                std::ceil(q / occ) * std::ceil(k / route_cost::kW4a8StepK.value()) * cap / clk;
            const double eff   = kW4a8Eff[std::min(i, 5)] * scale;
            offer(r, quant(t) +
                         std::max(mem, mma / eff) /
                             std::clamp(warps, route_cost::kMinWarpFill.value(), 1.0) +
                         chain);
        }
    }
    // The I8 tiles (generic_i8.cu): one activation scale per 64 values and the code as the MMA's
    // unsigned operand, about six instructions per MMA against the W4A8 tiles' twelve, at kI8Eff
    // of the int8 peak; one group per pipeline stage (250 cycles of barrier and copy wait). Tokens
    // padded to 256 in the quantizer.
    if (allows_i8(weight, problem.policy)) {
        const double tp = i8_padded_tokens(tokens);
        for (int i = 0; i < i8_schedule_count(); ++i) {
            const auto s         = i8_schedule_info(i);
            const Route r{Family::I8Mma, static_cast<std::uint8_t>(i)};
            const auto tile      = *route_tile(r, problem.epilogue, tokens);
            const double rows_w  = s.block_n, cols = tile.tokens;
            const double rows_o  = tile.out_rows;
            const double occ     = std::max(1.0, std::min({std::floor(max_thr / s.threads), max_cta,
                                                           std::floor(smem_sm / s.smem_bytes)}));
            const double ctas    = cdiv(n, rows_o) * cdiv(t, cols);
            const double q       = std::ceil(ctas / sms);
            const double warps   = std::min(1.0, ctas / sms) * std::min(q, occ) * s.threads / 32;
            const double mem     = std::max(q * rows_o * k * bpw / bw_sm, n * k * bpw / bw);
            const double mma     = q * 2.0 * rows_w * cols * k / peak_sm;
            const double chain = std::ceil(q / occ) * std::ceil(k / route_cost::kI8StageK.value()) *
                                 route_cost::kI8StageCycles.value() / clk;
            const double eff     = route_cost::kI8Efficiency[i] * scale;
            offer(r, quant(tp) +
                         std::max(mem, mma / eff) /
                             std::clamp(warps / route_cost::kI8SaturatingWarps.value(),
                                        route_cost::kMinWarpFill.value(), 1.0) +
                         chain);
        }
    }
    if (mmq_candidate_measured(d.major, d.minor) && ggml_mmq_format(problem.qtype) &&
        problem.n % ggml_mmq::kRowMultiple == 0 && problem.k % ggml_mmq::kKMultiple == 0) {
        const double slices = std::ceil(t / kMmqPrefillSlice);
        const double tt     = std::min<double>(t, kMmqPrefillSlice);
        const double sets   = problem.epilogue == Epilogue::SwiGlu ? 2.0 : 1.0;
        const double rows_t = route_cost::kMmqTileRows.value();
        const double step   = route_cost::kMmqColumnStep.value();
        // The column tile: the kernel's own table where the generation and the format take the
        // work-based one (core/mmq_columns.h), else 8-column steps up to a tile, the shape the
        // efficiency was fitted on.
        // Never wider than the tile the device's shared memory holds for the format (Turing's
        // 64 KiB: Q2_K's 128-column tile does not fit, the kernel runs 64).
        const bool format   = mmq_columns::work_tiles_format(problem.qtype);
        const int widest    = mmq_columns::widest_tile(problem.qtype, d.major, d.minor);
        const double cols =
            mmq_columns::uses_work_table(d.major, d.minor, format)
                ? static_cast<double>(
                      mmq_columns::pick(static_cast<int>(tt), d.major, d.minor, true, widest))
                : std::min({rows_t, static_cast<double>(widest), cdiv(tt, step) * step});
        const double tiles  = cdiv(n * sets, rows_t) * cdiv(tt, cols);
        const double mem    = n * k * bpw / bw;
        const double mma    = tiles / sms * 2.0 * rows_t * cols * k / peak_sm;
        const double chain =
            std::max(1.0, tiles * std::ceil(k / route_cost::kMmqChainK.value()) / sms) *
            route_cost::kMmqChainCycles.value() / clk;
        offer(Route{Family::GgmlMmq, 0},
              slices * (std::max(mem, mma / (route_cost::kMmqEfficiency.value() * scale)) + chain +
                        quant(tt) + route_cost::kMmqSliceSeconds.value()));
    }
}

} // namespace

namespace {

// What the compiled register-MMA kernel of a schedule keeps per thread, from the committed
// kernel-facts snapshot of the SASS the device runs (rmma_resources: the launch described for
// this call's rows and tokens, never run; ops/generic/kernel_facts.h, #612): its registers decide
// how many CTAs one SM's register file holds, its stack bytes are the spill. The answer is the
// release engine's, the same on every machine whatever its nvcc or driver (#556). Nothing is
// cached here, and the selector keeps no memo of its own (select_route runs on every launch, and
// at 17 tokens or more outside the prefill profile prices every schedule): what is cached is the
// snapshot read, once per kernel instance and device (launch_rmma_kernel, bounded by the template
// instances), so a priced call is a describe-mode dispatch, not a table search per schedule. A
// device that runs PTX the driver compiled, a schedule the build does not hold for the weight, or
// an instance the snapshot lacks has none and is priced by threads alone.
struct RmmaKept {
    int registers;
    double local_bytes;
};

std::optional<RmmaKept> rmma_kept(int schedule, const Weight& weight, Epilogue epilogue,
                                  std::int32_t tokens, std::int32_t rows) {
    std::optional<KernelResources> r;
    try {
        r = rmma_resources(schedule, weight, epilogue, tokens, rows);
    } catch (const std::exception&) {}
    if (!r) return std::nullopt;
    note_rmma_fact_read(RmmaFactRead{weight.qtype, schedule, epilogue, *r}); // never throws
    return RmmaKept{r->registers, static_cast<double>(r->local_bytes)};
}

} // namespace

// The speed-profile route of a call of 17 tokens or more from a cost model of every tile the
// build holds for the weight (register MMA, tiled, staged): the kernel roofline
// (kernel_roofline.h: a tile's bytes and MMA work on the busiest SM, its warp fill, its chain of
// K steps, its split-K reduction), the register MMA in series, the pipelined tiles (staged,
// tiled) overlapped at 0.9 / 0.8 of the tensor rate. The device numbers come from its query (SMs,
// threads, CTAs and shared memory per SM) and its facts (reference clock, bus width x memory
// clock: DeviceClass::facts); the tensor rate per SM and clock from the generation row
// (core/generation.h: 512 on the consumer and pro boards, 2048 on the full-rate HBM parts, 1536
// where the BF16 MMA is emulated, which also takes no pipelined tile and no 8-tile token strip).
// ninferno_route_tune's timings of seven boards (each tile in the engine, L2 flushed; BF16, Q4
// and Q8, T >= 17), the constants refit with the board held out: 1.095 of the best (geomean;
// RTX 4090 1.083, RTX 5090 1.068) against 1.110 for the F-lite rule (heuristic_rmma) and 1.137
// without the chain and reduction terms; the optimum of every constant is flat (the register-MMA
// step's and the pipeline stage's cycles, route_cost_constants.h, at 0.5..1.5x and 0.5..2x of
// their fit cost under 0.2 %).
// price_routes hands `take` every admitted route it prices with its time (the A16 scale applied);
// false where the facts have no roofline (nothing priced).
bool price_routes(const Problem& problem, const Weight& weight, std::int32_t tokens,
                  const PricedRoute& take) {
    namespace kr = kernel_roofline;
    DeviceClass d;
    try {
        d = device_class();
    } catch (const std::exception&) { return false; }
    // The reference clock and bandwidth of the board's facts (B2: the PCI-ID table, else the
    // driver's values on the 2^(1/16) grid), never the driver's clocks.
    // The register file is a fact of the device query: a board without it is not priced.
    if (d.sms <= 0 || d.registers_per_sm <= 0 || !d.facts.has_roofline() || problem.n <= 0 ||
        problem.k <= 0 || weight.payload_bytes == 0) {
        return false;
    }
    const bool emulated       = !d.features.mma_bf16_native;
    const kr::ChipFacts chip  = kr::chip_facts_of(d);
    const kr::Calibration cal = kr::calibration_of(d.major, d.minor);
    const double n = problem.n, k = problem.k;
    // SwiGLU: the weight has 2 n rows (gate and up), every output row multiplies both; `bpw` is the
    // payload over the n output rows, so the bytes of a CTA stay rows x K x bpw.
    const kr::Call call{n, k, static_cast<double>(tokens),
                        problem.epilogue == Epilogue::SwiGlu ? 2.0 : 1.0,
                        static_cast<double>(weight.payload_bytes) / (n * k)};
    // The register-MMA heavy-decoder terms (the L2 bytes, the real occupancy from the kernel's
    // registers, the spill bytes, the 8-warp fill) are verified for the decoders with
    // per-sub-block scales (the K-quants and the codebook i-quants, has_subblock_scales) on the
    // generations where two or more lab cards confirm them (rmma_terms_measured: sm_86 and
    // consumer Blackwell sm_120). Every other part keeps its old price: Turing and Volta (emulated
    // BF16 MMA) got worse with the terms (Quadro GV100 geomean 1.108 -> 1.155, worst 1.57 ->
    // 1.80), and Ada (the 4060 Ti got worse, #501) and the parts with a single card wait. OPEN:
    // the plain decoders (BF16, Q8_0, Q4_0 .. IQ4_NL, TQ1_0 / TQ2_0, the group-scaled, FP8 and FP4
    // formats) want their own measurement.
    const bool heavy_decoder =
        has_subblock_scales(problem.qtype) && rmma_terms_measured(d.major, d.minor);
    // The A16 model reads 11 % slow against the A8 one (fitted with it, 0.81..0.89 with any board
    // held out); on the minor-0 row (A100) it reads 39 % fast (one board); a constant factor leaves
    // the order of the A16 tiles as it was. 1 on a lent row, where no A8 candidate is offered
    // (a8_candidates).
    const double kA16Scale =
        fitted_route_scales(d.major, d.minor).value_or(FittedRouteScales{}).a16_cost;
    const auto offer = [&](Route r, double time) {
        if (!route_admits(r, weight, problem.policy)) return;
        if (r.family == Family::RegisterMma || r.family == Family::TiledMma ||
            r.family == Family::StagedMma) {
            time *= kA16Scale;
        }
        take(r, time);
    };
    for (int i = 0; i < rmma_schedule_count(); ++i) {
        const auto s = rmma_schedule_info(i);
        // Cluster schedules exist in the full (sm_12x) build only (rmma_schedule_launchable); a
        // substituted board without clusters is refused by route_admits (route_requires).
        if (!rmma_schedule_launchable(i)) continue;
        if (emulated && s.n_tiles == 8) continue;
        const auto kept = heavy_decoder ? rmma_kept(i, weight, problem.epilogue, tokens, problem.n)
                                        : std::optional<RmmaKept>{};
        const Route r{Family::RegisterMma, static_cast<std::uint8_t>(i)};
        const auto tile = *route_tile(r, problem.epilogue, tokens);
        kr::RegisterMmaKernel kernel;
        kernel.rows    = tile.out_rows;
        kernel.cols    = tile.tokens;
        kernel.cluster = tile.k_split;
        kernel.threads = 32.0 * s.warps_m * s.warps_k;
        kernel.warps_k = s.warps_k;
        kernel.heavy   = heavy_decoder;
        if (kept) {
            kernel.registers   = kept->registers;
            kernel.local_bytes = kept->local_bytes;
        }
        offer(r, kr::price_register_mma(chip, cal, call, kernel).total);
    }
    a8_candidates(problem, weight, tokens, d, offer);
    if (emulated) return true;
    for (int i = 0; i < tiled_schedule_count(); ++i) {
        const auto s = tiled_schedule_info(i);
        const Route r{Family::TiledMma, static_cast<std::uint8_t>(i)};
        const auto tile = *route_tile(r, problem.epilogue, tokens);
        kr::PipelinedKernel kernel;
        kernel.rows            = tile.out_rows;
        kernel.cols            = tile.tokens;
        kernel.threads         = 256;
        kernel.stages          = 3;
        kernel.stage_rows      = s.block_n;
        kernel.stage_cols      = s.block_m;
        kernel.tensor_fraction = route_cost::kTiledTensorFraction.value();
        offer(r, kr::price_pipelined(chip, cal, call, kernel).total);
    }
    for (int i = 0; i < staged_schedule_count(); ++i) {
        const auto s = staged_schedule_info(i);
        const Route r{Family::StagedMma, static_cast<std::uint8_t>(i)};
        const auto tile = *route_tile(r, problem.epilogue, tokens);
        kr::PipelinedKernel kernel;
        kernel.rows            = tile.out_rows;
        kernel.cols            = tile.tokens;
        kernel.threads         = 32.0 * s.warps_m * s.warps_n;
        kernel.stages          = s.stages;
        kernel.stage_rows      = s.rows;
        kernel.stage_cols      = s.tokens;
        kernel.tensor_fraction = route_cost::kStagedTensorFraction.value();
        offer(r, kr::price_pipelined(chip, cal, call, kernel).total);
    }
    return true;
}

std::optional<Route> cost_route(const Problem& problem, const Weight& weight, std::int32_t tokens) {
    std::optional<Route> best;
    double best_time = 0;
    price_routes(problem, weight, tokens, [&](Route r, double time) {
        if (!best || time < best_time) {
            best      = r;
            best_time = time;
        }
    });
    return best;
}


} // namespace ninferno::ops::detail::generic
