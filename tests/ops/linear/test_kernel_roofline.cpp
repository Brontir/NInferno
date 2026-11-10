// The kernel roofline (#613 PR-4, ops/generic/kernel_roofline.h) against the formula it replaced,
// host only. Two checks:
//  1. The register-MMA and pipelined prices equal, bit for bit, the lambdas price_routes held
//     before the move (copied below verbatim, parameterized by what they captured), over the
//     selector golden boards, every schedule, shapes, widths, SwiGLU and plain epilogues, several
//     bytes per value, both register-MMA forms, and compiled facts present and absent.
//  2. The selector's own priced routes (route_query.h priced_routes_query) are the roofline's
//     terms times the A16 scale for the inputs the selector passes (the route's tile, the
//     described weight's bytes, the compiled facts it read): the client's wiring.
#include "core/device_class.h"
#include "core/generation.h"
#include "core/weight.h"
#include "ninferno/ops/linear.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/kernel_roofline.h"
#include "ops/generic/route_cost_constants.h"
#include "ops/generic/route_query.h"
#include "ops/generic/route_table.h"
#include "ops/generic/selector_facts.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <optional>
#include <string>
#include <vector>

namespace g  = ninferno::ops::detail::generic;
namespace kr = g::kernel_roofline;
using ninferno::DeviceClass;
using ninferno::QType;
using ninferno::ops::LinearPolicy;

namespace {

int failures = 0;

// ---- The old formula, verbatim (generic_cost.cpp price_routes before #613 PR-4) ----

struct RmmaKept {
    int registers;
    double local_bytes;
};

struct OldModel {
    const DeviceClass& d;
    double n, k;
    std::int32_t tokens;
    double sets;
    double bpw;
    bool heavy_decoder;

    double clk, bw, sms, peak_sm, bw_sm, l2_sm, max_thr, max_cta, smem_sm, regs_sm;

    OldModel(const DeviceClass& dev, double n_, double k_, std::int32_t tokens_, double sets_,
             double bpw_, bool heavy)
        : d(dev), n(n_), k(k_), tokens(tokens_), sets(sets_), bpw(bpw_), heavy_decoder(heavy) {
        namespace route_cost               = g::route_cost;
        const ninferno::GenerationRow& gen = ninferno::generation_row(d.major, d.minor);
        clk                                = d.facts.reference_clock_hz();
        bw                                 = d.facts.dram_bytes_per_s();
        sms                                = d.sms;
        peak_sm                            = gen.bf16 * clk;
        bw_sm                              = bw / sms;
        l2_sm = (ninferno::gb20x_cc(d.major, d.minor) ? route_cost::kL2BytesPerSmClockGb20x
                                                      : route_cost::kL2BytesPerSmClockGa10x)
                    .value() *
                clk;
        max_thr = std::max(d.max_threads_per_sm, 1);
        max_cta = std::max(d.max_blocks_per_sm, 1);
        smem_sm = static_cast<double>(d.smem_per_sm);
        regs_sm = d.registers_per_sm;
    }

    static double cdiv(double a, double b) { return std::ceil(a / b); }

    double rmma_occupancy(double threads, const std::optional<RmmaKept>& kept) const {
        namespace route_cost = g::route_cost;
        const double limit   = std::min(std::floor(max_thr / threads), max_cta);
        if (!kept) return limit;
        const double unit    = route_cost::kRegAllocPerWarp.value();
        const double per_cta = std::ceil(kept->registers * 32.0 / unit) * unit * (threads / 32.0);
        return std::max(1.0, std::min(limit, std::floor(regs_sm / per_cta)));
    }

    double cost(double rows, double cols, double threads, double cluster, double occ,
                double warps_k, bool pipelined, double eff, double local_bytes = 0.0) const {
        namespace route_cost = g::route_cost;
        const double tiles   = cdiv(tokens, cols);
        const double gx      = cdiv(n, rows);
        const double ctas    = gx * tiles * cluster;
        const double kc      = k / cluster;
        const double q       = std::ceil(ctas / sms);
        const double warps   = std::min(q, std::max(occ, 1.0)) * threads / 32;
        const double mem     = std::max(q * rows * kc * bpw / bw_sm, n * k * bpw / bw);
        const double mma     = q * 2.0 * rows * sets * cols * kc / peak_sm;
        const double rounds  = std::ceil(q / std::max(occ, 1.0));
        if (pipelined) {
            const double chain = rounds * std::ceil(kc / route_cost::kPipelinedStageK.value()) *
                                 route_cost::kPipelinedStageCycles.value() / clk;
            return std::max(mem, mma / eff) /
                       std::clamp(warps / route_cost::kPipelinedSaturatingWarps.value(),
                                  route_cost::kMinWarpFill.value(), 1.0) +
                   chain;
        }
        const double dram = n * k * bpw / bw;
        const double l2_w = q * kc * rows * bpw / l2_sm;
        const double per_round =
            std::min({tiles, std::max(occ, 1.0), q, std::max(1.0, std::ceil(occ * sms / gx))});
        const double l2_x  = rounds * per_round * kc * (tokens / tiles) * 2.0 / l2_sm;
        const double spill = q * threads * local_bytes * route_cost::kSpillAccessesPerStep.value() *
                             std::ceil(kc / (route_cost::kRmmaHeavyStepK.value() * warps_k)) /
                             (route_cost::kL1BytesPerSmClock.value() * clk);
        const double bytes      = heavy_decoder ? dram + l2_w + l2_x + spill : mem;
        const double fill_warps = heavy_decoder ? route_cost::kRmmaHeavySaturatingWarps.value()
                                                : route_cost::kRmmaSaturatingWarps.value();
        const double reduce     = q * (cluster - 1) * rows * sets * cols *
                                  route_cost::kClusterReduceCycles.value() / clk +
                              rounds * (warps_k - 1.0) * rows * sets * cols *
                                  route_cost::kWarpReduceCycles.value() / clk;
        const double chain = rounds * std::ceil(kc / (route_cost::kRmmaStepK.value() * warps_k)) *
                             route_cost::kRmmaStepCycles.value() / clk;
        return (bytes + mma + reduce) /
                   std::clamp(warps / fill_warps, route_cost::kMinWarpFill.value(), 1.0) +
               chain;
    }

    double pipelined_occ(double threads, double stages, double rows, double cols) const {
        namespace route_cost = g::route_cost;
        const double tile =
            std::min(stages * (rows + cols) * route_cost::kPipelinedStageBytesPerRow.value(),
                     route_cost::kMaxTileSharedBytes.value());
        return std::min({std::floor(max_thr / threads), max_cta, std::floor(smem_sm / tile)});
    }
};

// ---- Check 1: the formula ----

struct Shape {
    std::int32_t n, k;
};

constexpr Shape kShapes[]        = {{256, 2048},   {1024, 4096},  {2048, 1024}, {4096, 4096},
                                    {17408, 5120}, {5120, 17408}, {28672, 8192}};
constexpr std::int32_t kTokens[] = {17, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 2048};
// Q4_K, IQ3_S, Q6_K, Q8 (34 bytes a 32), BF16.
constexpr double kBytesPerValue[]     = {144.0 / 256, 110.0 / 256, 210.0 / 256, 34.0 / 32, 2.0};
const std::optional<RmmaKept> kKept[] = {std::nullopt, RmmaKept{229, 0.0}, RmmaKept{128, 64.0},
                                         RmmaKept{96, 512.0}, RmmaKept{255, 1024.0}};

bool priceable(const DeviceClass& d) {
    return d.sms > 0 && d.registers_per_sm > 0 && d.facts.has_roofline();
}

void check_formula() {
    long cases = 0, bad = 0;
    for (const auto& board : g::kFactsBoards) {
        const DeviceClass d = g::board_device_class(board);
        if (!priceable(d)) {
            std::printf("%s: no roofline, skipped\n", board.name);
            continue;
        }
        const kr::ChipFacts chip  = kr::chip_facts_of(d);
        const kr::Calibration cal = kr::calibration_of(d.major, d.minor);
        for (const Shape shape : kShapes) {
            for (const std::int32_t t : kTokens) {
                for (const g::Epilogue e : {g::Epilogue::Store, g::Epilogue::SwiGlu}) {
                    const double sets = e == g::Epilogue::SwiGlu ? 2.0 : 1.0;
                    for (const double bpw : kBytesPerValue) {
                        const kr::Call call{static_cast<double>(shape.n),
                                            static_cast<double>(shape.k), static_cast<double>(t),
                                            sets, bpw};
                        for (const bool heavy : {false, true}) {
                            const OldModel old(d, shape.n, shape.k, t, sets, bpw, heavy);
                            for (int i = 0; i < g::rmma_schedule_count(); ++i) {
                                const auto s = g::rmma_schedule_info(i);
                                const g::Route r{g::Family::RegisterMma,
                                                 static_cast<std::uint8_t>(i)};
                                const auto tile      = *g::route_tile(r, e, t);
                                const double threads = 32.0 * s.warps_m * s.warps_k;
                                for (const auto& kept : kKept) {
                                    const double occ  = old.rmma_occupancy(threads, kept);
                                    const double want = old.cost(
                                        tile.out_rows, tile.tokens, threads, tile.k_split, occ,
                                        s.warps_k, false, 1, kept ? kept->local_bytes : 0.0);
                                    kr::RegisterMmaKernel kernel;
                                    kernel.rows    = tile.out_rows;
                                    kernel.cols    = tile.tokens;
                                    kernel.cluster = tile.k_split;
                                    kernel.threads = threads;
                                    kernel.warps_k = s.warps_k;
                                    kernel.heavy   = heavy;
                                    if (kept) {
                                        kernel.registers   = kept->registers;
                                        kernel.local_bytes = kept->local_bytes;
                                    }
                                    const auto terms =
                                        kr::price_register_mma(chip, cal, call, kernel);
                                    ++cases;
                                    if (terms.total != want || terms.grid.occupancy != occ) {
                                        if (++bad <= 10) {
                                            std::fprintf(stderr,
                                                         "FAIL %s rmma:%d %dx%d T=%d sets %.0f bpw "
                                                         "%.4f heavy %d regs %d: old %.17g new "
                                                         "%.17g\n",
                                                         board.name, i, shape.n, shape.k, t, sets,
                                                         bpw, heavy, kept ? kept->registers : -1,
                                                         want, terms.total);
                                        }
                                    }
                                }
                            }
                            if (heavy) continue; // the pipelined price has one form
                            for (int i = 0; i < g::tiled_schedule_count(); ++i) {
                                const auto s = g::tiled_schedule_info(i);
                                const g::Route r{g::Family::TiledMma, static_cast<std::uint8_t>(i)};
                                const auto tile = *g::route_tile(r, e, t);
                                const double want =
                                    old.cost(tile.out_rows, tile.tokens, 256, 1,
                                             old.pipelined_occ(256, 3, s.block_n, s.block_m), 1.0,
                                             true, g::route_cost::kTiledTensorFraction.value());
                                kr::PipelinedKernel kernel;
                                kernel.rows       = tile.out_rows;
                                kernel.cols       = tile.tokens;
                                kernel.threads    = 256;
                                kernel.stages     = 3;
                                kernel.stage_rows = s.block_n;
                                kernel.stage_cols = s.block_m;
                                kernel.tensor_fraction =
                                    g::route_cost::kTiledTensorFraction.value();
                                const auto terms = kr::price_pipelined(chip, cal, call, kernel);
                                ++cases;
                                if (terms.total != want) {
                                    if (++bad <= 10) {
                                        std::fprintf(
                                            stderr,
                                            "FAIL %s tiled:%d %dx%d T=%d sets %.0f bpw %.4f: "
                                            "old %.17g new %.17g\n",
                                            board.name, i, shape.n, shape.k, t, sets, bpw, want,
                                            terms.total);
                                    }
                                }
                            }
                            for (int i = 0; i < g::staged_schedule_count(); ++i) {
                                const auto s = g::staged_schedule_info(i);
                                const g::Route r{g::Family::StagedMma,
                                                 static_cast<std::uint8_t>(i)};
                                const auto tile      = *g::route_tile(r, e, t);
                                const double threads = 32.0 * s.warps_m * s.warps_n;
                                const double want    = old.cost(
                                    tile.out_rows, tile.tokens, threads, 1,
                                    old.pipelined_occ(threads, s.stages, s.rows, s.tokens), 1.0,
                                    true, g::route_cost::kStagedTensorFraction.value());
                                kr::PipelinedKernel kernel;
                                kernel.rows       = tile.out_rows;
                                kernel.cols       = tile.tokens;
                                kernel.threads    = threads;
                                kernel.stages     = s.stages;
                                kernel.stage_rows = s.rows;
                                kernel.stage_cols = s.tokens;
                                kernel.tensor_fraction =
                                    g::route_cost::kStagedTensorFraction.value();
                                const auto terms = kr::price_pipelined(chip, cal, call, kernel);
                                ++cases;
                                if (terms.total != want) {
                                    if (++bad <= 10) {
                                        std::fprintf(stderr,
                                                     "FAIL %s staged:%d %dx%d T=%d sets %.0f bpw "
                                                     "%.4f: old %.17g new %.17g\n",
                                                     board.name, i, shape.n, shape.k, t, sets, bpw,
                                                     want, terms.total);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::printf("formula: %ld cases, %ld differ\n", cases, bad);
    if (bad > 0 || cases == 0) ++failures;
}

// ---- Check 2: the selector's wiring ----

constexpr QType kFormats[]             = {QType::GGML_Q4_K,   QType::GGML_Q6_K,   QType::GGML_IQ3_S,
                                          QType::Q4_G64_FP16, QType::Q8_G32_FP16, QType::BF16,
                                          QType::GGML_Q4_0,   QType::GGML_IQ4_XS, QType::MXFP4};
constexpr Shape kWiringShapes[]        = {{1024, 4096}, {4096, 4096}, {17408, 5120}, {5120, 17408}};
constexpr std::int32_t kWiringTokens[] = {17, 32, 64, 256, 1024};

void check_wiring() {
    long priced_rmma = 0, priced_pipelined = 0, bad = 0;
    for (const auto& board : g::kFactsBoards) {
        const DeviceClass d = g::board_device_class(board);
        if (!priceable(d)) continue;
        const kr::ChipFacts chip  = kr::chip_facts_of(d);
        const kr::Calibration cal = kr::calibration_of(d.major, d.minor);
        const double a16          = ninferno::fitted_route_scales(d.major, d.minor)
                               .value_or(ninferno::FittedRouteScales{})
                               .a16_cost;
        for (const QType q : kFormats) {
            const bool heavy =
                ninferno::has_subblock_scales(q) && ninferno::rmma_terms_measured(d.major, d.minor);
            for (const Shape shape : kWiringShapes) {
                for (const g::Epilogue e : {g::Epilogue::Store, g::Epilogue::SwiGlu}) {
                    const double sets     = e == g::Epilogue::SwiGlu ? 2.0 : 1.0;
                    std::uint64_t payload = 0;
                    try {
                        payload = g::described_weight_bytes(q, e, shape.n, shape.k);
                    } catch (const std::exception&) {
                        continue; // a shape the format's layout does not take
                    }
                    const double n = shape.n, k = shape.k;
                    for (const std::int32_t t : kWiringTokens) {
                        std::vector<g::RmmaFactRead> reads;
                        std::vector<g::PricedRouteQuery> priced;
                        try {
                            const g::ScopedRmmaFactsSink sink(reads);
                            priced = g::priced_routes_query(d, q, LinearPolicy::A16Only, e, shape.n,
                                                            shape.k, t);
                        } catch (const std::exception&) {
                            continue; // a call the selector refuses on this board
                        }
                        const kr::Call call{n, k, static_cast<double>(t), sets,
                                            static_cast<double>(payload) / (n * k)};
                        for (const auto& p : priced) {
                            const g::Route r = p.route;
                            const auto tile  = g::route_tile(r, e, t);
                            double want      = 0;
                            if (r.family == g::Family::RegisterMma) {
                                const auto s = g::rmma_schedule_info(r.schedule);
                                kr::RegisterMmaKernel kernel;
                                kernel.rows    = tile->out_rows;
                                kernel.cols    = tile->tokens;
                                kernel.cluster = tile->k_split;
                                kernel.threads = 32.0 * s.warps_m * s.warps_k;
                                kernel.warps_k = s.warps_k;
                                kernel.heavy   = heavy;
                                // The facts the selector read for this schedule (the last read
                                // wins, as the price's own lookup did): none where it read none.
                                for (const auto& read : reads) {
                                    if (read.schedule == r.schedule && read.qtype == q &&
                                        read.epilogue == e) {
                                        kernel.registers = read.facts.registers;
                                        kernel.local_bytes =
                                            static_cast<double>(read.facts.local_bytes);
                                    }
                                }
                                want = kr::price_register_mma(chip, cal, call, kernel).total * a16;
                                ++priced_rmma;
                            } else if (r.family == g::Family::TiledMma) {
                                const auto s = g::tiled_schedule_info(r.schedule);
                                kr::PipelinedKernel kernel;
                                kernel.rows       = tile->out_rows;
                                kernel.cols       = tile->tokens;
                                kernel.threads    = 256;
                                kernel.stages     = 3;
                                kernel.stage_rows = s.block_n;
                                kernel.stage_cols = s.block_m;
                                kernel.tensor_fraction =
                                    g::route_cost::kTiledTensorFraction.value();
                                want = kr::price_pipelined(chip, cal, call, kernel).total * a16;
                                ++priced_pipelined;
                            } else if (r.family == g::Family::StagedMma) {
                                const auto s = g::staged_schedule_info(r.schedule);
                                kr::PipelinedKernel kernel;
                                kernel.rows       = tile->out_rows;
                                kernel.cols       = tile->tokens;
                                kernel.threads    = 32.0 * s.warps_m * s.warps_n;
                                kernel.stages     = s.stages;
                                kernel.stage_rows = s.rows;
                                kernel.stage_cols = s.tokens;
                                kernel.tensor_fraction =
                                    g::route_cost::kStagedTensorFraction.value();
                                want = kr::price_pipelined(chip, cal, call, kernel).total * a16;
                                ++priced_pipelined;
                            } else {
                                continue; // the A8 tiles are not moved yet
                            }
                            if (p.seconds != want) {
                                if (++bad <= 10) {
                                    std::fprintf(stderr,
                                                 "FAIL %s %s %dx%d T=%d %s %s:%d: selector %.17g "
                                                 "roofline %.17g\n",
                                                 board.name, g::qtype_name(q), shape.n, shape.k, t,
                                                 g::epilogue_name(e),
                                                 r.family == g::Family::RegisterMma ? "rmma"
                                                 : r.family == g::Family::TiledMma  ? "tiled"
                                                                                    : "staged",
                                                 r.schedule, p.seconds, want);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::printf("wiring: %ld register-MMA and %ld pipelined prices, %ld differ\n", priced_rmma,
                priced_pipelined, bad);
    // Every priceable board offers register-MMA routes at these widths: a run that priced none
    // tested nothing.
    if (bad > 0 || priced_rmma == 0) ++failures;
}

} // namespace

int main() {
    check_formula();
    check_wiring();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
