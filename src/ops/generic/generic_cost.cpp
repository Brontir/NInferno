// The route cost model (refactor B1): price_routes prices every admitted route of a call of 17
// tokens or more by the roofline of its tiles on the device's facts (the A16 tiles through
// kernel_roofline.h, the A8 ones in a8_candidates), cost_route takes the cheapest. Its constants
// and their provenance: route_cost_constants.h. A slice of a fused parent and a row view are
// priced by their own rows (bytes_per_weight).
#include "core/mmq_columns.h"
#include "core/device_class.h"
#include "core/generation.h"
#include "core/weight.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_ggml_internal.h"
#include "ops/generic/generic_i8_schedules.h"
#include "ops/generic/generic_schedules.h"
#include "ops/generic/generic_select_internal.h"
#include "ops/generic/ggml_mmq/ggml_mmq.h"
#include "ops/generic/kernel_roofline.h"
#include "ops/generic/route_cost_constants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>

namespace ninferno::ops::detail::generic {

double bytes_per_weight(const Weight& weight, const Problem& problem) {
    const double sets = problem.epilogue == Epilogue::SwiGlu ? 2.0 : 1.0;
    // A row view's bytes are a standalone weight's of its shape, every other weight's its payload,
    // over the rows the weight holds and never fewer than the call reads.
    const std::uint64_t bytes = weight.priced_bytes ? weight.priced_bytes : weight.payload_bytes;
    const double stored =
        std::max(static_cast<double>(weight.n), static_cast<double>(call_weight_rows(problem)));
    return static_cast<double>(bytes) / (stored * problem.k) * sets;
}

namespace {

// FUSION-SITE: activation_quantizer_launch
template <class Offer>
void a8_candidates(const Problem& problem, const Weight& weight, std::int32_t tokens,
                   const DeviceClass& d, const Offer& offer) {
    // The A8 tiles multiply on the int8 MMA: tensor mma.sync (sm_75 on: Turing's build composes
    // the s8 m16n8k32 from its m8n8k16, ops/common/mma.cuh), not on a chip that runs mma.sync on
    // its FP16 units (mma_slow: TU116 / TU117), and an int8 rate in the generation row (Volta has
    // none: its build composes the int8 MMA from dp4a). Whether the BF16 MMA is native says
    // nothing about the int8 one (#781: Turing's W4A8 and ggml MMQ went unpriced, and the
    // register MMA was taken at up to 4.7x the fastest route's time). The I8 tiles stay refused
    // there (route_requires: ldmatrix and cp.async, sm_80 on). Turing's A8 prices carry
    // efficiencies fitted on other generations (kW4a8Efficiency, kMmqEfficiency, its row's
    // w4a8_eff_scale 1.0): on the RTX 2080 Ti A5 sweep they read 0.58 .. 0.75 of the measured
    // time for ggml MMQ and 0.70 .. 0.74 for W4A8 (band geomeans), in the right order. OPEN: the
    // chip fact they lack, MMQ's dequantize on 64 CUDA-core lanes a SM on Turing against 128 on
    // Ampere and Ada (choice_constants.h kPrefillEta's Turing row).
    if (!d.features.mma_tensor || d.features.mma_slow || !allows_a8(problem.policy)) return;
    if (generation_row(d.major, d.minor).s8 <= 0) return;
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
    const double bpw     = bytes_per_weight(weight, problem);
    const double peak_sm = g.s8 * clk;
    const double scale   = fitted->w4a8_eff;
    const double bw_sm   = bw / sms;
    const double max_thr = std::max(d.max_threads_per_sm, 1);
    const double max_cta = std::max(d.max_blocks_per_sm, 1);
    const double smem_sm = static_cast<double>(d.smem_per_sm);
    const auto cdiv      = [](double a, double b) { return std::ceil(a / b); };
    // The DRAM time of a tile grid (kernel_roofline.h dram_time, issue #690): the shared bus or the
    // busiest SM's rounds at the rate its own warps reach, with the shared-largest carve-out (the
    // A8 tiles stage through shared memory); on an architecture without an SM rate row the legacy
    // law, the busiest SM's share of the bus under a fleet-wide fill. b_w: the I8 tiles' cp.async
    // ring keeps stages - 1 K steps of the CTA's weight rows in flight while it multiplies one,
    // over its warps; the W4A8 tiles load one K step through registers (below).
    namespace kr             = kernel_roofline;
    const kr::ChipFacts chip = kr::chip_facts_of(d);
    const bool legacy        = kr::legacy_dram(chip);
    const auto quant         = [&](double cols) {
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
            // Two stages, three with producer warps (generic_w4a8.cuh kStages).
            const double stages = s.producers > 0 ? 3.0 : 2.0;
            const double smem =
                stages * (rows_w + cols) * route_cost::kW4a8StageBytesPerRow.value();
            const double occ     = std::max(
                1.0, std::min({std::floor(max_thr / threads), max_cta, std::floor(smem_sm / smem)}));
            const double ctas = cdiv(n, rows_o) * cdiv(t, cols);
            const double q    = std::ceil(ctas / sms);
            const double mma  = q * 2.0 * rows_w * cols * k / peak_sm;
            const double cap  = s.producers > 0 ? route_cost::kW4a8ProducerStepCycles.value()
                                                : route_cost::kW4a8StepCycles.value();
            const double chain =
                std::ceil(q / occ) * std::ceil(k / route_cost::kW4a8StepK.value()) * cap / clk;
            const double eff = kW4a8Eff[std::min(i, 5)] * scale;
            if (legacy) {
                const double warps = std::min(1.0, ctas / sms) * std::min(q, occ) * threads / 32;
                const double mem   = std::max(q * rows_o * k * bpw / bw_sm, n * k * bpw / bw);
                offer(r, quant(t) +
                             std::max(mem, mma / eff) /
                                 std::clamp(warps, route_cost::kMinWarpFill.value(), 1.0) +
                             chain);
                continue;
            }
            // The fill reads the busy SM's own warps (no fleet-wide factor) and slows the MMA only.
            const double fill =
                std::clamp(std::min(q, occ) * threads / 32, route_cost::kMinWarpFill.value(), 1.0);
            // The weights come through registers (W4a8Weights: one buffer, the next K step loaded
            // while the current one is stored or multiplied), so one K step of the CTA's rows is in
            // flight whatever the stages, over the warps that load it: every warp of the classic
            // tile, the producer warps of the ws tile (generic_w4a8.cuh w4a8_kernel and
            // w4a8_ws_kernel). The ring's stages hold decoded data in shared memory, not loads.
            const double load_warps = s.producers > 0 ? s.producers : threads / 32;
            const double in_flight  = rows_o * route_cost::kW4a8StepK.value() * bpw / load_warps;
            const double mem =
                kr::dram_time(chip, ctas, occ, load_warps, rows_o * k * bpw, in_flight, true);
            offer(r, quant(t) + std::max(mem, mma / eff / fill) + chain);
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
            // The stages and shared memory the format's tile compiles with (generic_i8_schedules.h,
            // which generic_i8.cu's kI8Stages reads): Q5 / Q6 add their high plane to every
            // stage, and Q6 at 256x128 runs three stages, not the table's four (#761). The block
            // limit is the board's architecture's build limit, not this tree's: the base goldens
            // price sm_80/90/100 boards in an sm_86/89 build too, and their engines compile four.
            const int bits   = i8_bits(problem.qtype);
            const int stages = i8_stages(s.block_m, s.block_n, bits,
                                         shared_bytes_per_block_for_arch(d.major * 10 + d.minor));
            const double smem =
                static_cast<double>(i8_smem_bytes(s.block_m, s.block_n, bits, stages));
            const double occ     = std::max(1.0, std::min({std::floor(max_thr / s.threads), max_cta,
                                                           std::floor(smem_sm / smem)}));
            const double ctas    = cdiv(n, rows_o) * cdiv(t, cols);
            const double q       = std::ceil(ctas / sms);
            const double mma     = q * 2.0 * rows_w * cols * k / peak_sm;
            const double chain = std::ceil(q / occ) * std::ceil(k / route_cost::kI8StageK.value()) *
                                 route_cost::kI8StageCycles.value() / clk;
            const double eff     = route_cost::kI8Efficiency[i] * scale;
            if (legacy) {
                const double warps = std::min(1.0, ctas / sms) * std::min(q, occ) * s.threads / 32;
                const double mem   = std::max(q * rows_o * k * bpw / bw_sm, n * k * bpw / bw);
                offer(r, quant(tp) +
                             std::max(mem, mma / eff) /
                                 std::clamp(warps / route_cost::kI8SaturatingWarps.value(),
                                            route_cost::kMinWarpFill.value(), 1.0) +
                             chain);
                continue;
            }
            const double fill      = std::clamp(std::min(q, occ) * s.threads / 32 /
                                                    route_cost::kI8SaturatingWarps.value(),
                                                route_cost::kMinWarpFill.value(), 1.0);
            const double in_flight =
                (stages - 1.0) * rows_o * route_cost::kI8StageK.value() * bpw / (s.threads / 32.0);
            const double mem =
                kr::dram_time(chip, ctas, occ, s.threads / 32.0, rows_o * k * bpw, in_flight, true);
            offer(r, quant(tp) + std::max(mem, mma / eff / fill) + chain);
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
        // A generation whose widths a launch law prices (Turing, DECISIONS 78) takes the width
        // that law picks for the product's grid on this card (the facts below).
        const bool format   = mmq_columns::work_tiles_format(problem.qtype);
        const int widest    = mmq_columns::widest_tile(problem.qtype, d.major, d.minor);
        const mmq_columns::LaunchFacts facts{static_cast<int>(n), static_cast<int>(k), d.sms, clk,
                                             bw};
        const double cols =
            mmq_columns::uses_work_table(d.major, d.minor, format)
                ? static_cast<double>(mmq_columns::pick(static_cast<int>(tt), d.major, d.minor,
                                                        true, widest, facts))
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
// snapshot read, once per kernel instance and device (rmma_resources keeps a table of the
// instances read, keyed by integers, generic_rmma_describe.cpp; the launch's describe mode keeps
// one per template instance, bounded by the instances), so after the first call for an instance a
// priced call is a few hash lookups, not a string built and a snapshot searched per schedule. A
// device that runs PTX the driver compiled, a schedule the build does not hold for the weight, or
// an instance the snapshot lacks has none and is priced by threads alone.
struct RmmaKept {
    int registers;
    double local_bytes;
    double static_smem;
    std::optional<kernel_facts::LoopCounts> loop; // the K loop's counts (#613), where known
};

// `counted_law`: the price reads the kernel's K loop (Calibration::rmma_counted_law), so the loop
// is kept; off, it is not (nothing reads it).
std::optional<RmmaKept> rmma_kept(int schedule, const Weight& weight, Epilogue epilogue,
                                  std::int32_t tokens, std::int32_t rows, bool counted_law) {
    std::optional<KernelResources> r;
    try {
        r = rmma_resources(schedule, weight, epilogue, tokens, rows);
    } catch (const std::exception&) {}
    if (!r) return std::nullopt;
    note_rmma_fact_read(RmmaFactRead{weight.qtype, schedule, epilogue, *r}); // never throws
    return RmmaKept{r->registers, static_cast<double>(r->local_bytes),
                    static_cast<double>(r->static_smem),
                    counted_law ? std::move(r->loop) : std::nullopt};
}

// The counted register-MMA terms (rmma_counted_terms.h, #613 PR-7a) in place of the fitted spill
// and the weight-only tensor term: route_cost::kRmmaCountedLaw says which law this binary prices
// (false until the A/B of the counted law; off, the price is golden-equal to the fitted one), and a
// test overrides it for its process. price_routes hands it to price_register_mma
// (Calibration::rmma_counted_law) with the kernel's loop (RegisterMmaKernel::loop).
std::optional<bool> g_rmma_counted_law_override;

} // namespace

bool rmma_counted_law() noexcept {
    return g_rmma_counted_law_override.value_or(route_cost::kRmmaCountedLaw);
}

void rmma_counted_law_for_test(std::optional<bool> law) noexcept {
    g_rmma_counted_law_override = law;
}

bool a16_route_priced(Route route, bool bf16_mma_native) {
    if (bf16_mma_native) return true;
    switch (route.family) {
    case Family::RegisterMma:
        return route.schedule < rmma_schedule_count() &&
               rmma_schedule_info(route.schedule).n_tiles != 8;
    case Family::TiledMma:
    case Family::StagedMma:
        return false;
    default:
        return true;
    }
}

// The speed-profile route of a call of 17 tokens or more from a cost model of every tile the
// build holds for the weight (register MMA, tiled, staged): the kernel roofline
// (kernel_roofline.h: a tile's bytes and MMA work on the busiest SM, its warp fill, its chain of
// K steps, its split-K reduction), the register MMA in series, the pipelined tiles (staged,
// tiled) overlapped at 0.9 / 0.8 of the tensor rate. The device numbers come from its query (SMs,
// threads, CTAs and shared memory per SM) and its facts (reference clock, bus width x memory
// clock: DeviceClass::facts); the tensor rate per SM and clock from the generation row
// (core/generation.h: 512 on the consumer and pro boards, Turing's GeForce boards included, 2048
// on the full-rate HBM parts, Volta's fitted 1536; where the BF16 MMA is emulated (Turing, Volta)
// the model takes no pipelined tile and no 8-tile token strip).
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
    // The binary's register-MMA law, or a test's override of it (rmma_counted_law).
    const kr::Calibration cal = [&] {
        kr::Calibration c  = kr::calibration_of(d.major, d.minor);
        c.rmma_counted_law = rmma_counted_law();
        return c;
    }();
    const double n = problem.n, k = problem.k;
    // SwiGLU: every output row multiplies its gate and its up row; `bpw` (bytes_per_weight) holds
    // both rows' bytes, so the bytes of a CTA stay rows x K x bpw.
    const kr::Call call{n, k, static_cast<double>(tokens),
                        problem.epilogue == Epilogue::SwiGlu ? 2.0 : 1.0,
                        bytes_per_weight(weight, problem)};
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
        const Route r{Family::RegisterMma, static_cast<std::uint8_t>(i)};
        if (!a16_route_priced(r, !emulated)) continue;
        auto kept       = heavy_decoder ? rmma_kept(i, weight, problem.epilogue, tokens, problem.n,
                                                    cal.rmma_counted_law)
                                        : std::optional<RmmaKept>{};
        const auto tile = *route_tile(r, problem.epilogue, tokens);
        kr::RegisterMmaKernel kernel;
        kernel.rows    = tile.out_rows;
        kernel.cols    = tile.tokens;
        kernel.cluster = tile.k_split;
        kernel.threads = 32.0 * s.warps_m * s.warps_k;
        kernel.warps_k = s.warps_k;
        kernel.m_tiles      = s.m_tiles;
        kernel.k_step       = static_cast<double>(rmma_step_values(problem.qtype, problem.k));
        kernel.heavy   = heavy_decoder;
        kernel.shared_bytes = kr::register_mma_static_shared_bytes(
            rmma_geometry_decoder(problem.qtype), s.warps_m, s.warps_k, s.m_tiles, s.n_tiles,
            s.cluster, problem.epilogue == Epilogue::SwiGlu, d.major * 10 + d.minor);
        if (kept) {
            kernel.registers    = kept->registers;
            kernel.local_bytes  = kept->local_bytes;
            kernel.shared_bytes = kept->static_smem;
            kernel.loop         = std::move(kept->loop); // kept under the counted law only
        }
        offer(r, kr::price_register_mma(chip, cal, call, kernel).total);
    }
    a8_candidates(problem, weight, tokens, d, offer);
    for (int i = 0; i < tiled_schedule_count(); ++i) {
        const auto s = tiled_schedule_info(i);
        const Route r{Family::TiledMma, static_cast<std::uint8_t>(i)};
        if (!a16_route_priced(r, !emulated)) continue;
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
        if (!a16_route_priced(r, !emulated)) continue;
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
