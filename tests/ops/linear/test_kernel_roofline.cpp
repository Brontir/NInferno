// The kernel roofline (#613 PR-4, ops/generic/kernel_roofline.h) against the formula it replaced,
// host only. Two checks:
//  1. The selector's priced routes equal, exactly, the fixed golden table computed from the code
//     before the extraction (kernel_roofline_golden.inc): the golden boards, two
//     formats, four shapes, five widths, plain and SwiGLU epilogues, every register-MMA, tiled and
//     staged route offered. Routes are matched by (family, schedule); a route a build gates in or
//     out (cluster schedules, shared memory) is checked against its gate, so one file serves every
//     build set.
//  2. The selector's own priced routes (route_query.h priced_routes_query) are the roofline's
//     terms times the A16 scale for the inputs the selector passes (the route's tile, the
//     described weight's bytes, the compiled facts it read): the client's wiring.
#include "core/device_class.h"
#include "core/format_facts.h"
#include "core/generation.h"
#include "core/weight.h"
#include "ninferno/ops/linear.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/kernel_roofline.h"
#include "ops/generic/price_tool.h"
#include "ops/generic/route_cost_constants.h"
#include "ops/generic/route_query.h"
#include "ops/generic/route_table.h"
#include "ops/generic/selector_facts.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <exception>
#include <optional>
#include <string>
#include <utility>
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
// extraction error shows as a price that moved. Prices are compared exactly: %.17g round-trips a
// double, and the price is IEEE arithmetic with floor, ceil, min and max only (no library
// function whose last bit could differ between hosts), and no multiply-add is fused: the files
// that compute it (generic_cost.cpp, price_tool.cpp) are compiled with -ffp-contract=off
// (src/ops/generic/sources.cmake), as this test is (ninferno_op_oracle_options). So the same
// terms in the same order give the same bits in every build set, and the merge of the two
// builds' tables (merge_kernel_roofline_golden.py) demands the same bits too.

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

constexpr bool is_golden_family(g::Family f) {
    return f == g::Family::RegisterMma || f == g::Family::TiledMma || f == g::Family::StagedMma;
}

// The table is the UNION of every route any build set offers, each with the price the pre-refactor
// selector gave it: the full sm_120a build (NVFP4, so the cluster K-split register-MMA schedules
// launch) and the sm_80 build (163 KiB of shared memory per block, so the deep staged schedules
// 10-12 fit) merged by merge_kernel_roofline_golden.py, which also requires every route the two
// price to agree. Which routes a build offers differs by build set, so routes are matched by
// (family, schedule), never by position:
//  - a route this build offers must be in the table, with the same price: a route the table has no
//    price for is a FAILURE (nothing is waived as "gated in");
//  - a route in the table this build does not offer must be one it gates out: a register-MMA
//    schedule it cannot launch (a base or sm70/sm75 build has no cluster schedules), or a staged
//    schedule whose shared stages do not fit its block (sm70: 96 KiB, sm75: 64 KiB, the 99 KiB
//    builds the deep ones). Any other absence, and any tiled one, is a failure.
bool gated_out_here(g::Route r) {
    switch (r.family) {
    case g::Family::RegisterMma:
        return !g::rmma_schedule_launchable(r.schedule);
    case g::Family::StagedMma:
        return !g::staged_fits(r.schedule);
    default:
        return false;
    }
}

void check_golden() {
    // The rows name their board by an index into kFactsBoards: a reordered or grown list would
    // price another board against the table.
    constexpr std::size_t kBoards = sizeof(kGoldenBoards) / sizeof(kGoldenBoards[0]);
    if (std::size(g::kFactsBoards) != kBoards) {
        std::fprintf(stderr, "FAIL: kFactsBoards has %zu boards, the golden was made over %zu\n",
                     std::size(g::kFactsBoards), kBoards);
        ++failures;
        return;
    }
    for (std::size_t i = 0; i < kBoards; ++i) {
        if (std::strcmp(g::kFactsBoards[i].name, kGoldenBoards[i]) != 0) {
            std::fprintf(stderr,
                         "FAIL: kFactsBoards[%zu] is %s, the golden's board %zu is %s "
                         "(the list moved: regenerate the golden)\n",
                         i, g::kFactsBoards[i].name, i, kGoldenBoards[i]);
            ++failures;
            return;
        }
    }
    long prices = 0, gated_out = 0, bad = 0;
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
                fail("offered now, and the golden has no price for it", family, schedule, 0,
                     got.seconds);
                continue;
            }
            seen[at]                = true;
            const GoldenRoute& want = kGoldenRoutes[c.first + static_cast<long>(at)];
            ++prices;
            if (got.seconds != want.seconds) {
                fail("price moved", family, schedule, want.seconds, got.seconds);
            }
        }
        for (std::size_t i = 0; i < c.count; ++i) {
            if (seen[i]) continue;
            const GoldenRoute& want = kGoldenRoutes[c.first + static_cast<long>(i)];
            const g::Route r{static_cast<g::Family>(want.family),
                             static_cast<std::uint8_t>(want.schedule)};
            if (gated_out_here(r)) {
                ++gated_out;
            } else {
                fail("in the golden, not offered now, and this build does not gate it out",
                     want.family, want.schedule, want.seconds, 0);
            }
        }
    }
    std::printf("golden: %zu calls, %ld route prices compared, %ld routes gated out by this build, "
                "%ld differ\n",
                sizeof(kGoldenCalls) / sizeof(kGoldenCalls[0]), prices, gated_out, bad);
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

// ---- Check 3: the price tool's format names ----
// Every name the usage text prints is one `--format` takes (the parser and the help read one list),
// maps back to the format it names, and the name the help gave for Q4 groupwise is the one that
// worked before the fix ("q4_g64" is not a route-table name: the table calls it "q4").
void check_price_formats() {
    const std::string usage = ninferno::tools::price_usage_text();
    const auto names        = ninferno::tools::price_format_names();
    long bad                = 0;
    for (const std::string& name : names) {
        const auto found = ninferno::tools::price_find_format(name);
        if (!found || g::qtype_name(*found) != name) {
            std::fprintf(stderr, "FAIL: price --format %s is not taken\n", name.c_str());
            ++bad;
        }
        if (usage.find(name) == std::string::npos) {
            std::fprintf(stderr, "FAIL: the usage text does not print %s\n", name.c_str());
            ++bad;
        }
    }
    const auto q4 = ninferno::tools::price_find_format("q4");
    if (!q4 || *q4 != QType::Q4_G64_FP16) {
        std::fprintf(stderr, "FAIL: price --format q4 is not Q4 groupwise\n");
        ++bad;
    }
    std::printf("price formats: %zu names, %ld differ\n", names.size(), bad);
    if (bad > 0 || names.empty()) ++failures;
}

// ---- Check 4: the price tool prices only the routes the selector admits ----
// `price --route F:N` exits 0 (a price printed) exactly where the selector admits the route for
// the call on this build and board (route_query.h route_admitted_query: route_admits, the
// family's whole rule), else 2 with no price: every register-MMA, tiled and staged route, for
// BF16 and Q4_K, on 99 KiB boards (RTX 4090, RTX 5090) and a 227 KiB one with clusters (H100
// SXM). Explicit cases pin the rule's refusals by name, so a query that drifted from the
// selector would not pass by agreeing with itself.
int price_exit(const std::string& board, const std::string& format, const std::string& k,
               const std::string& route) {
    std::vector<std::string> args = {"price", "--board", board, "--format", format,
                                     "--n",   "4096",    "--k", k,          "--t",
                                     "64",    "--route", route};
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    std::fflush(stdout);
    const int rc = ninferno::tools::price_main(static_cast<int>(argv.size()), argv.data());
    std::fflush(stdout);
    return rc;
}

const g::FactsBoard* facts_board(const char* name) {
    for (const auto& b : g::kFactsBoards) {
        if (std::strcmp(b.name, name) == 0) return &b;
    }
    return nullptr;
}

std::string board_id(const g::FactsBoard& board) {
    char id[16];
    std::snprintf(id, sizeof id, "0x%04X", static_cast<unsigned>(board.pci_device_id));
    return id;
}

void check_price_admitted() {
    long bad = 0, priced = 0, refused = 0, named = 0;
    const auto expect = [&](const g::FactsBoard& board, QType q, std::int32_t k, g::Route r,
                            bool admitted, const char* why) {
        const std::string route = std::string(g::family_name(r.family)) + ":" +
                                  std::to_string(static_cast<int>(r.schedule));
        const int rc   = price_exit(board_id(board), g::qtype_name(q), std::to_string(k), route);
        const int want = admitted ? 0 : 2;
        if (rc != want) {
            std::fprintf(stderr,
                         "FAIL: price --board %s --format %s --k %d --route %s exits %d, "
                         "want %d (%s)\n",
                         board.name, g::qtype_name(q), k, route.c_str(), rc, want, why);
            ++bad;
        }
        ++(admitted ? priced : refused);
    };
    // Admitted by the selector and priced by price_routes on the board (a16_route_priced: where
    // the BF16 MMA is emulated, no pipelined tile and no 8-tile strip; the golden check holds
    // price_routes to the old rule on the emulated golden boards).
    const auto query = [](const g::FactsBoard& board, QType q, std::int32_t k, g::Route r) {
        const DeviceClass d = g::board_device_class(board);
        return g::route_admitted_query(d, r, q, LinearPolicy::A16Only, g::Epilogue::Store, 4096,
                                       k) &&
               g::a16_route_priced(r, d.features.mma_bf16_native);
    };
    const g::FactsBoard* ada    = facts_board("rtx4090");
    const g::FactsBoard* hopper = facts_board("h100-sxm");
    const g::FactsBoard* gb202  = facts_board("rtx5090");
    const g::FactsBoard* turing = facts_board("rtx2080ti");
    if (!ada || !hopper || !gb202 || !turing) {
        std::fprintf(stderr, "FAIL: a facts board of check 4 is missing\n");
        ++failures;
        return;
    }
    // Every route of the three families, against the selector's own rules.
    for (const g::FactsBoard* board : {ada, gb202, hopper, turing}) {
        for (const QType q : {QType::BF16, QType::GGML_Q4_K}) {
            const std::pair<g::Family, int> families[] = {
                {g::Family::RegisterMma, g::rmma_schedule_count()},
                {g::Family::TiledMma, g::tiled_schedule_count()},
                {g::Family::StagedMma, g::staged_schedule_count()}};
            for (const auto& [family, count] : families) {
                for (int i = 0; i < count; ++i) {
                    const g::Route r{family, static_cast<std::uint8_t>(i)};
                    expect(*board, q, 4096, r, query(*board, q, 4096, r), "route_admitted_query");
                }
            }
        }
    }
    // The refusals by name (each priced and exited 0 before the tool read the selector's rule).
    const auto pin = [&](const g::FactsBoard& board, QType q, std::int32_t k, g::Route r,
                         bool admitted, const char* why) {
        ++named;
        if (query(board, q, k, r) != admitted) {
            std::fprintf(stderr, "FAIL: route_admitted_query disagrees with the pinned case (%s)\n",
                         why);
            ++bad;
        }
        expect(board, q, k, r, admitted, why);
    };
    using F = g::Family;
    pin(*ada, QType::GGML_Q4_K, 4096, {F::StagedMma, 0}, false, "staged reads BF16 only");
    // rmma:11 holds 2x8 MMA tiles per warp: no decoder of 32-value runs (Q4_K) is compiled for 16
    // tiles (rmma_tile_compiled). rmma:19 is 1x1 tiles split over 16 warps in K: compiled.
    pin(*ada, QType::GGML_Q4_K, 4096, {F::RegisterMma, 11}, false,
        "no 2x8-tile register-MMA decoder for Q4_K");
    pin(*ada, QType::GGML_Q4_K, 4096, {F::RegisterMma, 19}, true,
        "1x1 tiles over 16 K warps, compiled for Q4_K");
    pin(*ada, QType::BF16, 4096, {F::RegisterMma, 16}, false,
        "cluster schedule, no clusters on Ada");
    pin(*hopper, QType::BF16, 4096, {F::RegisterMma, 16}, g::rmma_schedule_launchable(16),
        "cluster schedule on a board with clusters, where this build compiles it");
    pin(*ada, QType::BF16, 4000, {F::StagedMma, 0}, false, "K not whole 64-value steps");
    pin(*ada, QType::BF16, 4096, {F::TiledMma, 0}, false, "the tiled kernel has no BF16 tile");
    // Emulated BF16 MMA (Turing): price_routes prices no pipelined tile and no 8-tile strip
    // (rmma:10 holds 1x8 tiles), but the 1x1 rmma:19 it prices.
    pin(*turing, QType::GGML_Q4_K, 4096, {F::TiledMma, 0}, false,
        "no pipelined tile where the BF16 MMA is emulated");
    pin(*turing, QType::BF16, 4096, {F::RegisterMma, 10}, false,
        "no 8-tile strip where the BF16 MMA is emulated");
    pin(*turing, QType::BF16, 4096, {F::RegisterMma, 19}, true,
        "a 1x1 schedule where the BF16 MMA is emulated");
    // Tiled schedules whose tiles do not fit this build's block for a format (tiled_fits).
    for (const QType q : kFormats) {
        const auto codec = ninferno::format_facts(q).codec;
        if (!codec || !g::tiled_supports(*codec)) continue;
        try {
            (void)g::described_weight_bytes(q, g::Epilogue::Store, 4096, 4096);
        } catch (const std::exception&) {
            continue; // a shape the format's layout does not take
        }
        for (int i = 0; i < g::tiled_schedule_count(); ++i) {
            if (!g::tiled_fits(i, *codec)) {
                pin(*ada, q, 4096, {F::TiledMma, static_cast<std::uint8_t>(i)}, false,
                    "tiled tiles past this build's block");
            }
        }
    }
    std::printf("price admitted: %ld routes priced, %ld refused (%ld pinned cases), %ld differ\n",
                priced, refused, named, bad);
    if (bad > 0 || priced == 0) ++failures;
}

// ---- Check 5: the price tool's command line ----
// A malformed value is a usage error (2), never the prefix std::atoi would take: a route index
// with trailing text or none, a number with trailing text, a non-number, a non-finite one. And
// every board spelling the usage text advertises resolves (a whole price table, exit 0).
int price_args(std::vector<std::string> args) {
    args.insert(args.begin(), "price");
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    std::fflush(stdout);
    const int rc = ninferno::tools::price_main(static_cast<int>(argv.size()), argv.data());
    std::fflush(stdout);
    return rc;
}

void check_price_arguments() {
    long bad = 0, cases = 0;
    const auto call = [](std::vector<std::string> extra, const std::string& board = "rtx4090") {
        std::vector<std::string> args = {"--board", board, "--format", "bf16", "--n",
                                         "4096",    "--k", "4096",     "--t",  "64"};
        args.insert(args.end(), extra.begin(), extra.end());
        return args;
    };
    const auto expect = [&](const std::vector<std::string>& args, int want, const char* what) {
        ++cases;
        const int rc = price_args(args);
        if (rc != want) {
            std::fprintf(stderr, "FAIL: price %s exits %d, want %d\n", what, rc, want);
            ++bad;
        }
    };
    expect(call({"--route", "rmma:bogus"}), 2, "--route rmma:bogus");
    expect(call({"--route", "rmma:1x"}), 2, "--route rmma:1x");
    expect(call({"--route", "rmma:"}), 2, "--route rmma:");
    expect(call({"--route", "rmma:256"}), 2, "--route rmma:256");
    expect(call({"--route", "rmma:-1"}), 2, "--route rmma:-1");
    expect(call({"--route", "rmma:99"}), 2, "--route rmma:99 (past the family's count)");
    expect(call({"--n", "4096x"}), 2, "--n 4096x");
    expect(call({"--k", ""}), 2, "--k ''");
    expect(call({"--t", "0"}), 2, "--t 0");
    expect(call({"--registers", "abc"}), 2, "--registers abc");
    expect(call({"--registers", "300"}), 2, "--registers 300");
    expect(call({"--local-bytes", "nan"}), 2, "--local-bytes nan");
    expect(call({"--local-bytes", "inf"}), 2, "--local-bytes inf");
    expect(call({"--local-bytes", "-4"}), 2, "--local-bytes -4");
    expect(call({"--local-bytes", "8x"}), 2, "--local-bytes 8x");
    // The board spellings the usage text prints.
    const std::string usage = ninferno::tools::price_usage_text();
    for (const char* board : {"rtx4090", "l4", "a100pcie40gb", "0x2684"}) {
        if (usage.find(board) == std::string::npos) {
            std::fprintf(stderr, "FAIL: the usage text no longer prints %s\n", board);
            ++bad;
        }
        const std::string what = std::string("--board ") + board;
        expect(call({}, board), 0, what.c_str());
    }
    std::printf("price arguments: %ld cases, %ld differ\n", cases, bad);
    if (bad > 0) ++failures;
}

int main() {
    check_golden();
    check_wiring();
    check_price_formats();
    check_price_admitted();
    check_price_arguments();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
