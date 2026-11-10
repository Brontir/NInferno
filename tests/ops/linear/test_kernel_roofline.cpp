// The kernel roofline (#613 PR-4, ops/generic/kernel_roofline.h) against the formula it replaced,
// host only. Two checks:
//  1. The selector's priced routes equal, to a relative 1e-12, the fixed golden table computed
//     from the code before the extraction (kernel_roofline_golden.inc): the golden boards, three
//     formats, two shapes, two widths, plain and SwiGLU epilogues, every route offered.
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

struct Shape {
    std::int32_t n, k;
};

// ---- Check 1: golden prices from the formula before #613 PR-4 ----
//
// kernel_roofline_golden.inc holds, for a set of calls (board, format, shape, width, epilogue),
// every route the selector offered with the time it priced, printed (%.17g) by the selector's
// price_routes before the roofline was extracted. Nothing of the old formula is copied here: an
// extraction error shows as a price that moved.

struct GoldenCall {
    int board, format;
    std::int32_t n, k, tokens;
    g::Epilogue epilogue;
    long first;
    std::size_t count;
};

struct GoldenRoute {
    int family, schedule;
    double seconds;
};

constexpr QType kGoldenFormats[] = {QType::GGML_Q4_K, QType::GGML_Q6_K, QType::BF16};

#include "kernel_roofline_golden.inc"

constexpr double kGoldenTolerance = 1e-12;

void check_golden() {
    long prices = 0, bad = 0;
    for (const GoldenCall& c : kGoldenCalls) {
        const DeviceClass d = g::board_device_class(g::kFactsBoards[c.board]);
        const auto priced   = g::priced_routes_query(d, kGoldenFormats[c.format],
                                                     LinearPolicy::A16Only, c.epilogue, c.n, c.k,
                                                     c.tokens);
        if (priced.size() != c.count) {
            if (++bad <= 10) {
                std::fprintf(stderr, "FAIL %s format %d %dx%d T=%d: %zu routes, golden %zu\n",
                             g::kFactsBoards[c.board].name, c.format, c.n, c.k, c.tokens,
                             priced.size(), c.count);
            }
            continue;
        }
        for (std::size_t i = 0; i < c.count; ++i) {
            const GoldenRoute& want = kGoldenRoutes[c.first + static_cast<long>(i)];
            const auto& got         = priced[i];
            ++prices;
            const bool same = static_cast<int>(got.route.family) == want.family &&
                              static_cast<int>(got.route.schedule) == want.schedule &&
                              std::fabs(got.seconds - want.seconds) <=
                                  kGoldenTolerance * want.seconds;
            if (!same && ++bad <= 10) {
                std::fprintf(stderr,
                             "FAIL %s format %d %dx%d T=%d route %d:%d: golden %.17g now %.17g\n",
                             g::kFactsBoards[c.board].name, c.format, c.n, c.k, c.tokens,
                             want.family, want.schedule, want.seconds, got.seconds);
            }
        }
    }
    std::printf("golden: %zu calls, %ld route prices, %ld differ\n",
                sizeof(kGoldenCalls) / sizeof(kGoldenCalls[0]), prices, bad);
    if (bad > 0 || prices == 0) ++failures;
}

// ---- Check 2: the selector's wiring ----

constexpr QType kFormats[]           = {QType::GGML_Q4_K,   QType::GGML_Q6_K,   QType::GGML_IQ3_S,
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
    check_golden();
    check_wiring();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
