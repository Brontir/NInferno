// The kernel roofline (#613 PR-4, ops/generic/kernel_roofline.h) against the formula it replaced,
// host only. Two checks:
//  1. The selector's priced routes equal, to a relative 1e-12, the fixed golden table computed
//     from the code before the extraction (kernel_roofline_golden.inc): the golden boards, two
//     formats, four shapes, five widths, plain and SwiGLU epilogues, every register-MMA, tiled and
//     staged route offered. Routes are matched by (family, schedule); a route a build gates in or
//     out (cluster schedules, shared memory) is checked against its gate, so one file serves every
//     build set.
//  2. The selector's own priced routes (route_query.h priced_routes_query) are the roofline's
//     terms times the A16 scale for the inputs the selector passes (the route's tile, the
//     described weight's bytes, the compiled facts it read): the client's wiring.
#include "core/device_class.h"
#include "core/generation.h"
#include "core/weight.h"
#include "ninferno/ops/linear.h"
#include "ops/common/arch_limits.h"
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

bool priceable(const DeviceClass& d) {
    return d.sms > 0 && d.registers_per_sm > 0 && d.facts.has_roofline();
}

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

constexpr QType kGoldenFormats[] = {QType::GGML_Q4_K, QType::BF16};

#include "kernel_roofline_golden.inc"

constexpr double kGoldenTolerance = 1e-12;

constexpr bool is_golden_family(g::Family f) {
    return f == g::Family::RegisterMma || f == g::Family::TiledMma || f == g::Family::StagedMma;
}

// The table was made in the base build (no NVFP4, 99 KiB of shared memory per block). Which routes
// a build offers differs by build set, so routes are matched by (family, schedule), never by
// position: the prices of the routes both sides offer must agree, and a route on one side only
// must be one this build gates out or in:
//  - missing now: a register-MMA schedule the build cannot launch, or a staged schedule whose
//    shared stages do not fit this build's block (sm75: 64 KiB);
//  - extra now: a register-MMA cluster schedule (the full build, rmma_schedule_launchable) or a
//    staged schedule that fits only because this build's block is past 99 KiB (sm80: 163 KiB).
// A route that is a tiled one, or that is gated nowhere, is a failure.
bool gate_explains(bool extra, g::Route r) {
    constexpr std::size_t kBaseShared = 99 * 1024;
    switch (r.family) {
    case g::Family::RegisterMma:
        return extra ? g::rmma_schedule_info(r.schedule).cluster > 1
                     : !g::rmma_schedule_launchable(r.schedule);
    case g::Family::StagedMma:
        return extra ? g::staged_fits(r.schedule) &&
                           ninferno::ops::kMaxSharedBytesPerBlock > kBaseShared
                     : !g::staged_fits(r.schedule);
    default:
        return false;
    }
}

void check_golden() {
    long prices = 0, gated_out = 0, gated_in = 0, bad = 0;
    for (const GoldenCall& c : kGoldenCalls) {
        const DeviceClass d = g::board_device_class(g::kFactsBoards[c.board]);
        const auto priced   = g::priced_routes_query(
            d, kGoldenFormats[c.format], LinearPolicy::A16Only, c.epilogue, c.n, c.k, c.tokens);
        const auto fail = [&](const char* what, int family, int schedule, double want, double got) {
            if (++bad <= 10) {
                std::fprintf(
                    stderr,
                    "FAIL %s format %d %dx%d T=%d route %d:%d: %s (golden %.17g now %.17g)\n",
                    g::kFactsBoards[c.board].name, c.format, c.n, c.k, c.tokens, family, schedule,
                    what, want, got);
            }
        };
        std::vector<bool> seen(c.count, false);
        for (const auto& got : priced) {
            if (!is_golden_family(got.route.family)) continue;
            const int family   = static_cast<int>(got.route.family);
            const int schedule = static_cast<int>(got.route.schedule);
            std::size_t at     = c.count;
            for (std::size_t i = 0; i < c.count; ++i) {
                const GoldenRoute& want = kGoldenRoutes[c.first + static_cast<long>(i)];
                if (want.family == family && want.schedule == schedule) at = i;
            }
            if (at == c.count) {
                if (gate_explains(true, got.route)) {
                    ++gated_in;
                } else {
                    fail("offered now, not in the golden, and no gate explains it", family,
                         schedule, 0, got.seconds);
                }
                continue;
            }
            seen[at]                = true;
            const GoldenRoute& want = kGoldenRoutes[c.first + static_cast<long>(at)];
            ++prices;
            if (!(std::fabs(got.seconds - want.seconds) <= kGoldenTolerance * want.seconds)) {
                fail("price moved", family, schedule, want.seconds, got.seconds);
            }
        }
        for (std::size_t i = 0; i < c.count; ++i) {
            if (seen[i]) continue;
            const GoldenRoute& want = kGoldenRoutes[c.first + static_cast<long>(i)];
            const g::Route r{static_cast<g::Family>(want.family),
                             static_cast<std::uint8_t>(want.schedule)};
            if (gate_explains(false, r)) {
                ++gated_out;
            } else {
                fail("in the golden, not offered now, and no gate explains it", want.family,
                     want.schedule, want.seconds, 0);
            }
        }
    }
    std::printf("golden: %zu calls, %ld route prices equal-checked, %ld routes gated out and %ld "
                "gated in by this build, %ld differ\n",
                sizeof(kGoldenCalls) / sizeof(kGoldenCalls[0]), prices, gated_out, gated_in, bad);
    if (bad > 0 || prices == 0) ++failures;
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
    check_golden();
    check_wiring();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
