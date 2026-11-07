// The route cost model (#refactor B1): price_routes prices every admitted route of a call of 17
// tokens or more by the roofline of its tiles on the device's facts (the A16 tiles here, the A8
// ones in a8_candidates), cost_route takes the cheapest. Its constants and their provenance:
// route_cost_constants.h.
#include "ops/generic/generic_gemm.h"
#include "core/per_device.h"
#include "ops/common/arch_limits.h"
#include "ops/common/native_nvfp4.h"

#include "core/device_class.h"
#include "core/generation.h"
#include "core/measured_boards.h"
#include "core/tensor.h"
#include "core/layout.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/common/route_probe.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_select_internal.h"
#include "ops/generic/route_cost_constants.h"
#include "ops/generic/generic_ggml_internal.h"
#include "ops/generic/generic_schedules.h"
#include "ops/generic/ggml_mmq/ggml_mmq.h"
#include "ops/generic/r8_decode_table.h"
#include "ops/generic/route_table.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/route_pins.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <functional>
#include <array>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <tuple>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

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
    const double clk       = d.facts.reference_clock_hz();
    const double bw        = d.facts.dram_bytes_per_s();
    const double sms       = d.sms;
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
            const auto s = w4a8_schedule_info(i);
            const Route r{Family::W4a8Mma, static_cast<std::uint8_t>(i)};
            const auto tile     = *route_tile(r, problem.epilogue, tokens);
            const double rows_w = s.block_n, cols = tile.tokens;
            const double rows_o  = tile.out_rows;
            const double threads = s.producers > 0                  ? 384.0
                                   : rows_w == 64.0 && cols == 64.0 ? 128.0
                                                                    : 256.0;
            const double smem    = (s.producers > 0 ? 3.0 : 2.0) * (rows_w + cols) *
                                route_cost::kW4a8StageBytesPerRow.value();
            const double occ   = std::max(1.0, std::min({std::floor(max_thr / threads), max_cta,
                                                         std::floor(smem_sm / smem)}));
            const double ctas  = cdiv(n, rows_o) * cdiv(t, cols);
            const double q     = std::ceil(ctas / sms);
            const double warps = std::min(1.0, ctas / sms) * std::min(q, occ) * threads / 32;
            const double mem   = std::max(q * rows_o * k * bpw / bw_sm, n * k * bpw / bw);
            const double mma   = q * 2.0 * rows_w * cols * k / peak_sm;
            const double cap   = s.producers > 0 ? route_cost::kW4a8ProducerStepCycles.value()
                                                 : route_cost::kW4a8StepCycles.value();
            const double chain =
                std::ceil(q / occ) * std::ceil(k / route_cost::kW4a8StepK.value()) * cap / clk;
            const double eff = kW4a8Eff[std::min(i, 5)] * scale;
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
            const auto s = i8_schedule_info(i);
            const Route r{Family::I8Mma, static_cast<std::uint8_t>(i)};
            const auto tile     = *route_tile(r, problem.epilogue, tokens);
            const double rows_w = s.block_n, cols = tile.tokens;
            const double rows_o = tile.out_rows;
            const double occ    = std::max(1.0, std::min({std::floor(max_thr / s.threads), max_cta,
                                                          std::floor(smem_sm / s.smem_bytes)}));
            const double ctas   = cdiv(n, rows_o) * cdiv(t, cols);
            const double q      = std::ceil(ctas / sms);
            const double warps  = std::min(1.0, ctas / sms) * std::min(q, occ) * s.threads / 32;
            const double mem    = std::max(q * rows_o * k * bpw / bw_sm, n * k * bpw / bw);
            const double mma    = q * 2.0 * rows_w * cols * k / peak_sm;
            const double chain = std::ceil(q / occ) * std::ceil(k / route_cost::kI8StageK.value()) *
                                 route_cost::kI8StageCycles.value() / clk;
            const double eff = route_cost::kI8Efficiency[i] * scale;
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
        const double cols   = std::min(rows_t, cdiv(tt, step) * step);
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

// The speed-profile route of a call of 17 tokens or more from a cost model of every tile the
// build holds for the weight (register MMA, tiled, staged): a tile's time on the busiest SM is its
// weight bytes over that SM's share of the DRAM bandwidth (or the whole matrix over the whole
// bandwidth, whichever is longer) and its MMA work over the SM's tensor rate; the register MMA
// runs them in series, the pipelined tiles (staged, tiled) overlap them at 0.9 / 0.8 of the
// tensor rate; either is slowed in proportion when the SM holds fewer than 16 (register MMA) or 4
// (pipelined) warps, and pays its warps' dependent chain of K steps per residency and the sum of
// its split K (over a CTA's warps or a cluster). The register MMA of the K-quant and codebook
// decoders (`heavy_decoder`) is priced by its bytes instead: the matrix once over the DRAM bus,
// what its CTAs pull through the SM's L2 port (their weights, and the token tile of x once per
// residency round), and a residency bounded by the registers its tile keeps (#301: the 16-row
// tiles of K split warps look cheap by the DRAM share alone and move 3..4 times the x of the
// 64-row tiles; their 32-token accumulators leave room for two CTAs, not twelve). The device
// numbers come from its query (SMs, threads, CTAs and shared memory per SM) and its facts
// (reference clock, bus width x memory clock: DeviceClass::facts); the tensor rate per SM and clock
// from the generation row (core/generation.h: 512 on the consumer and pro boards, 2048 on the
// full-rate HBM parts, 1536 where the BF16 MMA is emulated, which also takes no pipelined tile and
// no 8-tile token strip). ninferno_route_tune's timings of seven boards (each tile in the engine,
// L2 flushed; BF16, Q4 and Q8, T >= 17), the constants refit with the board held out: 1.095 of the
// best (geomean; RTX 4090 1.083, RTX 5090 1.068) against 1.110 for the F-lite rule (heuristic_rmma)
// and 1.137 without the chain and reduction terms; the optimum of every constant is flat (the
// register-MMA step's and the pipeline stage's cycles, route_cost_constants.h, at 0.5..1.5x and
// 0.5..2x of their fit cost under 0.2 %). nullopt where the facts have no roofline. price_routes
// hands `take` every admitted route it prices with its time (the A16 scale applied); false where
// the facts have no roofline (nothing priced).
bool price_routes(const Problem& problem, const Weight& weight, std::int32_t tokens,
                  const PricedRoute& take) {
    DeviceClass d;
    try {
        d = device_class();
    } catch (const std::exception&) { return false; }
    // The reference clock and bandwidth of the board's facts (B2: the PCI-ID table, else the
    // driver's values on the 2^(1/16) grid), never the driver's clocks.
    if (d.sms <= 0 || !d.facts.has_roofline() || problem.n <= 0 || problem.k <= 0 ||
        weight.payload_bytes == 0) {
        return false;
    }
    const GenerationRow& g = generation_row(d.major, d.minor);
    const bool emulated    = !d.features.mma_bf16_native;
    const double clk       = d.facts.reference_clock_hz();
    const double bw        = d.facts.dram_bytes_per_s();
    const double sms       = d.sms;
    const double peak_sm   = g.bf16 * clk;
    const double bw_sm     = bw / sms;
    // What one SM reads from the L2 a second.
    const double l2_sm = route_cost::kL2BytesPerSmClock.value() * clk;
    const double n = problem.n, k = problem.k;
    const double bpw     = static_cast<double>(weight.payload_bytes) / (n * k);
    const double max_thr = std::max(d.max_threads_per_sm, 1);
    const double max_cta = std::max(d.max_blocks_per_sm, 1);
    const double smem_sm = static_cast<double>(d.smem_per_sm);
    const auto cdiv      = [](double a, double b) { return std::ceil(a / b); };
    // SwiGLU: the weight has 2 n rows (gate and up), every output row multiplies both; `bpw` is the
    // payload over the n output rows, so the bytes of a CTA stay rows x K x bpw.
    const double sets = problem.epilogue == Epilogue::SwiGlu ? 2.0 : 1.0;
    // The register-MMA terms below are verified for the decoders that unpack scales and codes
    // (the K-quants and the codebook i-quants), on the RTX 3060, 4090 and 5090; the others keep
    // their old price (with the terms they lost 5..15 % of regret). OPEN: this list is a judgement
    // from the per-format regret, not derived from a decoder property; the plain decoders (BF16,
    // Q8_0, Q4_0 .. IQ4_NL, the group-scaled, FP8 and FP4 formats) want their own measurement.
    const bool heavy_decoder = [&] {
        switch (problem.qtype) {
        case QType::GGML_Q4_K:
        case QType::GGML_Q5_K:
        case QType::GGML_Q6_K:
        case QType::GGML_Q3_K:
        case QType::GGML_Q2_K:
        case QType::GGML_IQ4_XS:
        case QType::GGML_IQ3_XXS:
        case QType::GGML_IQ3_S:
        case QType::GGML_IQ2_XXS:
        case QType::GGML_IQ2_XS:
        case QType::GGML_IQ2_S:
        case QType::GGML_IQ1_S:
        case QType::GGML_IQ1_M:
            return true;
        default:
            return false;
        }
    }();
    // Registers a thread of such a kernel keeps (an estimate from the tile: accumulators, the
    // weight ring and the decoded fragments), and so the CTAs of `threads` threads that one SM's
    // register file holds. A 16-row, 32-token tile of a K-quant (229 registers) holds two
    // 128-thread CTAs, not the twelve the thread limit allows.
    const double regs_sm      = std::max(d.registers_per_sm, 1);
    const auto rmma_occupancy = [&](const RmmaScheduleInfo& s, double threads) {
        const double thread_limit = std::min(std::floor(max_thr / threads), max_cta);
        if (!heavy_decoder) return thread_limit;
        const double regs =
            std::min(route_cost::kMaxRegsPerThread.value(),
                     route_cost::kRmmaRegsBase.value() +
                         route_cost::kRmmaRegsPerTile.value() * s.m_tiles * s.n_tiles +
                         route_cost::kRmmaRegsPerExtraRowTile.value() * (s.m_tiles - 1));
        // Registers are allocated per warp, in units of kRegAllocPerWarp.
        const double unit    = route_cost::kRegAllocPerWarp.value();
        const double per_cta = std::ceil(regs * 32.0 / unit) * unit * (threads / 32.0);
        return std::max(1.0, std::min(thread_limit, std::floor(regs_sm / per_cta)));
    };
    // Time of `ctas` CTAs of `rows` output rows x `cols` (K over `cluster` CTAs) of `threads`
    // threads, at most `occ` resident per SM.
    const auto cost = [&](double rows, double cols, double threads, double cluster, double occ,
                          double warps_k, bool pipelined, double eff) {
        const double tiles = cdiv(tokens, cols);
        const double ctas  = cdiv(n, rows) * tiles * cluster;
        const double kc    = k / cluster;
        const double q     = std::ceil(ctas / sms);
        // The warps resident on a busy SM: SMs left idle by a small grid do not slow the busiest
        // one, whose bytes are already in `mem` (the fleet-wide fill charged calls of fewer CTAs
        // than SMs up to 20x).
        const double warps = std::min(q, std::max(occ, 1.0)) * threads / 32;
        const double mem   = std::max(q * rows * kc * bpw / bw_sm, n * k * bpw / bw);
        const double mma   = q * 2.0 * rows * sets * cols * kc / peak_sm;
        // The CTAs the busiest SM runs one residency after another each pay their warps'
        // dependent chain: the exposed load latency of a K step (register MMA: 16 K per warp, 150
        // cycles) or of a pipeline stage (64 K, 500 cycles: the barrier and the cp.async wait).
        const double rounds = std::ceil(q / std::max(occ, 1.0));
        if (pipelined) {
            const double chain = rounds * std::ceil(kc / route_cost::kPipelinedStageK.value()) *
                                 route_cost::kPipelinedStageCycles.value() / clk;
            return std::max(mem, mma / eff) /
                       std::clamp(warps / route_cost::kPipelinedSaturatingWarps.value(),
                                  route_cost::kMinWarpFill.value(), 1.0) +
                   chain;
        }
        // The register MMA's bytes (heavy decoders). The matrix crosses the DRAM bus once whatever
        // its token tiles (the CTAs of a weight row block after the first read it from the L2);
        // every CTA pulls its weights and its token tile of x through its SM's L2 port. The CTAs
        // resident together on an SM read the same token tile at the same K, so a residency round
        // fetches each tile once for all of them: min(tiles, occ) copies a round.
        const double dram       = n * k * bpw / bw;
        const double l2_w       = q * kc * rows * bpw / l2_sm;
        const double copies     = rounds * std::min({tiles, std::max(occ, 1.0), q});
        const double l2_x       = copies * kc * (tokens / tiles) * 2.0 / l2_sm;
        const double bytes      = heavy_decoder ? dram + l2_w + l2_x : mem;
        const double fill_warps = heavy_decoder ? route_cost::kRmmaHeavySaturatingWarps.value()
                                                : route_cost::kRmmaSaturatingWarps.value();
        // K split over a cluster (30 cycles per FP32 element) and over the CTA's K warps (2.75
        // cycles per element and extra warp, through shared memory).
        const double reduce = q * (cluster - 1) * rows * sets * cols *
                                  route_cost::kClusterReduceCycles.value() / clk +
                              rounds * (warps_k - 1.0) * rows * sets * cols *
                                  route_cost::kWarpReduceCycles.value() / clk;
        const double chain = rounds * std::ceil(kc / (route_cost::kRmmaStepK.value() * warps_k)) *
                             route_cost::kRmmaStepCycles.value() / clk;
        return (bytes + mma + reduce) /
                   std::clamp(warps / fill_warps, route_cost::kMinWarpFill.value(), 1.0) +
               chain;
    };
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
        const double threads = 32.0 * s.warps_m * s.warps_k;
        const double occ     = rmma_occupancy(s, threads);
        const Route r{Family::RegisterMma, static_cast<std::uint8_t>(i)};
        const auto tile = *route_tile(r, problem.epilogue, tokens);
        offer(r, cost(tile.out_rows, tile.tokens, threads, tile.k_split, occ, s.warps_k, false, 1));
    }
    a8_candidates(problem, weight, tokens, d, offer);
    if (emulated) return true;
    const auto pipelined_occ = [&](double threads, double stages, double rows, double cols) {
        const double tile =
            std::min(stages * (rows + cols) * route_cost::kPipelinedStageBytesPerRow.value(),
                     route_cost::kMaxTileSharedBytes.value());
        return std::min({std::floor(max_thr / threads), max_cta, std::floor(smem_sm / tile)});
    };
    for (int i = 0; i < tiled_schedule_count(); ++i) {
        const auto s = tiled_schedule_info(i);
        const Route r{Family::TiledMma, static_cast<std::uint8_t>(i)};
        const auto tile = *route_tile(r, problem.epilogue, tokens);
        offer(r,
              cost(tile.out_rows, tile.tokens, 256, 1, pipelined_occ(256, 3, s.block_n, s.block_m),
                   1.0, true, route_cost::kTiledTensorFraction.value()));
    }
    for (int i = 0; i < staged_schedule_count(); ++i) {
        const auto s         = staged_schedule_info(i);
        const double threads = 32.0 * s.warps_m * s.warps_n;
        const Route r{Family::StagedMma, static_cast<std::uint8_t>(i)};
        const auto tile = *route_tile(r, problem.epilogue, tokens);
        offer(r, cost(tile.out_rows, tile.tokens, threads, 1,
                      pipelined_occ(threads, s.stages, s.rows, s.tokens), 1.0, true,
                      route_cost::kStagedTensorFraction.value()));
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
