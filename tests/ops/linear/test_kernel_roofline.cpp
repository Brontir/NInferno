// The kernel roofline (#613 PR-4, ops/generic/kernel_roofline.h) against the formula it replaced,
// host only. Two checks:
//  1. The selector's priced routes equal, exactly, the fixed golden table computed from the code
//     before the extraction (kernel_roofline_golden.inc): the golden boards, two
//     formats, four shapes, five widths, plain and SwiGLU epilogues, every register-MMA, tiled and
//     staged route offered. Routes are matched by (family, schedule); a route a build gates in or
//     out (cluster schedules, shared memory) is checked against its gate, so one file serves every
//     build set. On a board the DRAM law of #690 prices (its architecture has an SM rate row,
//     core/sm_rate_facts.h), the roofline's legacy law is compared instead (check 1's comment).
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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <exception>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef _WIN32
#    include <io.h>
#    define dup    _dup
#    define dup2   _dup2
#    define close  _close
#    define fileno _fileno
#else
#    include <unistd.h>
#endif

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

// The roofline's price of an A16 route of the call on `chip` (the device's facts, or a variant of
// them), times the board's A16 scale, from the inputs the selector passes: the route's tile, the
// described weight's bytes, and the compiled facts the selector read for the route (`reads`, the
// last read wins as the price's own lookup did; none where it read none). nullopt for an A8
// route (a8_candidates prices those, not the roofline's tiles) or a shape the format's layout
// does not take.
std::optional<double> roofline_price(const DeviceClass& d, const kr::ChipFacts& chip, QType q,
                                     g::Epilogue e, std::int32_t n_rows, std::int32_t k_cols,
                                     std::int32_t t, g::Route r,
                                     const std::vector<g::RmmaFactRead>& reads) {
    const kr::Calibration cal = kr::calibration_of(d.major, d.minor);
    const double a16          = ninferno::fitted_route_scales(d.major, d.minor)
                           .value_or(ninferno::FittedRouteScales{})
                           .a16_cost;
    const bool heavy =
        ninferno::has_subblock_scales(q) && ninferno::rmma_terms_measured(d.major, d.minor);
    std::uint64_t payload = 0;
    try {
        payload = g::described_weight_bytes(q, e, n_rows, k_cols);
    } catch (const std::exception&) { return std::nullopt; }
    const double n = n_rows, k = k_cols;
    const kr::Call call{n, k, static_cast<double>(t), e == g::Epilogue::SwiGlu ? 2.0 : 1.0,
                        static_cast<double>(payload) / (n * k)};
    const auto tile = g::route_tile(r, e, t);
    if (r.family == g::Family::RegisterMma) {
        const auto s = g::rmma_schedule_info(r.schedule);
        kr::RegisterMmaKernel kernel;
        kernel.rows         = tile->out_rows;
        kernel.cols         = tile->tokens;
        kernel.cluster      = tile->k_split;
        kernel.threads      = 32.0 * s.warps_m * s.warps_k;
        kernel.warps_k      = s.warps_k;
        kernel.m_tiles      = s.m_tiles;
        kernel.k_step       = static_cast<double>(g::rmma_step_values(q, k_cols));
        kernel.heavy        = heavy;
        kernel.shared_bytes = kr::register_mma_schedule_shared_bytes(
            s.warps_m, s.warps_k, static_cast<int>(call.sets), s.m_tiles, s.n_tiles, s.cluster);
        for (const auto& read : reads) {
            if (read.schedule == r.schedule && read.qtype == q && read.epilogue == e) {
                kernel.registers    = read.facts.registers;
                kernel.local_bytes  = static_cast<double>(read.facts.local_bytes);
                kernel.shared_bytes = static_cast<double>(read.facts.static_smem);
            }
        }
        return kr::price_register_mma(chip, cal, call, kernel).total * a16;
    }
    if (r.family == g::Family::TiledMma) {
        const auto s = g::tiled_schedule_info(r.schedule);
        kr::PipelinedKernel kernel;
        kernel.rows            = tile->out_rows;
        kernel.cols            = tile->tokens;
        kernel.threads         = 256;
        kernel.stages          = 3;
        kernel.stage_rows      = s.block_n;
        kernel.stage_cols      = s.block_m;
        kernel.tensor_fraction = g::route_cost::kTiledTensorFraction.value();
        return kr::price_pipelined(chip, cal, call, kernel).total * a16;
    }
    if (r.family == g::Family::StagedMma) {
        const auto s = g::staged_schedule_info(r.schedule);
        kr::PipelinedKernel kernel;
        kernel.rows            = tile->out_rows;
        kernel.cols            = tile->tokens;
        kernel.threads         = 32.0 * s.warps_m * s.warps_n;
        kernel.stages          = s.stages;
        kernel.stage_rows      = s.rows;
        kernel.stage_cols      = s.tokens;
        kernel.tensor_fraction = g::route_cost::kStagedTensorFraction.value();
        return kr::price_pipelined(chip, cal, call, kernel).total * a16;
    }
    return std::nullopt;
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
    long prices = 0, gated_out = 0, bad = 0, legacy_law = 0;
    for (const GoldenCall& c : kGoldenCalls) {
        const DeviceClass d = g::board_device_class(g::kFactsBoards[c.board]);
        const QType q       = kGoldenFormats[c.format];
        std::vector<g::RmmaFactRead> reads;
        std::vector<g::PricedRouteQuery> priced;
        {
            const g::ScopedRmmaFactsSink sink(reads);
            priced =
                g::priced_routes_query(d, q, LinearPolicy::A16Only, c.epilogue, c.n, c.k, c.tokens);
        }
        // A board whose architecture has an SM rate row is priced by the DRAM law of #690, which
        // the golden predates: its routes are held to the golden through the roofline with the
        // legacy law (the chip facts without the SM rate facts, kernel_roofline.h legacy_dram),
        // whose wiring to the selector check 2 holds. Every other board's selector price is the
        // golden's itself.
        kr::ChipFacts legacy_chip    = kr::chip_facts_of(d);
        const bool priced_by_new_law = !kr::legacy_dram(legacy_chip);
        legacy_chip.dram_latency_s.reset();
        legacy_chip.sm_dram_bytes_per_clock_free.reset();
        legacy_chip.sm_dram_bytes_per_clock_placed.reset();
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
            double seconds = got.seconds;
            if (priced_by_new_law) {
                ++legacy_law;
                seconds = roofline_price(d, legacy_chip, q, c.epilogue, c.n, c.k, c.tokens,
                                         got.route, reads)
                              .value_or(0.0);
            }
            if (seconds != want.seconds) {
                fail(priced_by_new_law ? "legacy-law price moved" : "price moved", family, schedule,
                     want.seconds, seconds);
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
    std::printf("golden: %zu calls, %ld route prices compared (%ld of them through the legacy "
                "law of a board the #690 law prices), %ld routes gated out by this build, "
                "%ld differ\n",
                sizeof(kGoldenCalls) / sizeof(kGoldenCalls[0]), prices, legacy_law, gated_out, bad);
    // Both laws must be compared: a board of each kind is in the golden.
    if (bad > 0 || prices == 0 || legacy_law == 0 || legacy_law == prices) ++failures;
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
        const kr::ChipFacts chip = kr::chip_facts_of(d);
        for (const QType q : kFormats) {
            for (const Shape shape : kWiringShapes) {
                for (const g::Epilogue e : {g::Epilogue::Store, g::Epilogue::SwiGlu}) {
                    try {
                        (void)g::described_weight_bytes(q, e, shape.n, shape.k);
                    } catch (const std::exception&) {
                        continue; // a shape the format's layout does not take
                    }
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
                        for (const auto& p : priced) {
                            const g::Route r = p.route;
                            const auto priced_here =
                                roofline_price(d, chip, q, e, shape.n, shape.k, t, r, reads);
                            if (!priced_here) continue; // the A8 tiles are not moved yet
                            const double want = *priced_here;
                            ++(r.family == g::Family::RegisterMma ? priced_rmma : priced_pipelined);
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
        return rc;
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
    // Every route of the three families, against the selector's own rules; and, independent of
    // both predicates, the routes the tool prices at this width (64 tokens: the selector's
    // pricing domain) are exactly the A16 routes the selector itself priced (priced_routes_query).
    long set_differ = 0;
    for (const g::FactsBoard* board : {ada, gb202, hopper, turing}) {
        for (const QType q : {QType::BF16, QType::GGML_Q4_K}) {
            const auto selector =
                g::priced_routes_query(g::board_device_class(*board), q, LinearPolicy::A16Only,
                                       g::Epilogue::Store, 4096, 4096, 64);
            const std::pair<g::Family, int> families[] = {
                {g::Family::RegisterMma, g::rmma_schedule_count()},
                {g::Family::TiledMma, g::tiled_schedule_count()},
                {g::Family::StagedMma, g::staged_schedule_count()}};
            for (const auto& [family, count] : families) {
                for (int i = 0; i < count; ++i) {
                    const g::Route r{family, static_cast<std::uint8_t>(i)};
                    const int rc = expect(*board, q, 4096, r, query(*board, q, 4096, r),
                                          "route_admitted_query");
                    const bool offered =
                        std::any_of(selector.begin(), selector.end(),
                                    [&](const g::PricedRouteQuery& p) { return p.route == r; });
                    if (offered != (rc == 0)) {
                        std::fprintf(
                            stderr, "FAIL: %s %s %s:%d: the tool %s it, the selector %s it\n",
                            board->name, g::qtype_name(q),
                            std::string(g::family_name(r.family)).c_str(), i,
                            rc == 0 ? "prices" : "refuses", offered ? "priced" : "did not price");
                        ++set_differ;
                    }
                }
            }
        }
    }
    bad += set_differ;
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

// ---- Check 6: under --policy a8 the table holds every route the selector priced ----
// The A8 tiles (a8_candidates) have no roofline terms yet; their rows carry the selector's price,
// so its cheapest route is never hidden. The table's text is read back from the tool's stdout.
std::string price_stdout(std::vector<std::string> args, int& rc) {
    std::fflush(stdout);
    std::FILE* sink = std::tmpfile();
    if (sink == nullptr) return {};
    const int out   = fileno(stdout);
    const int saved = dup(out);
    dup2(fileno(sink), out);
    rc = price_args(std::move(args));
    std::fflush(stdout);
    dup2(saved, out);
    close(saved);
    std::string text;
    std::rewind(sink);
    char buf[4096];
    for (std::size_t got; (got = std::fread(buf, 1, sizeof buf, sink)) > 0;) text.append(buf, got);
    std::fclose(sink);
    return text;
}

void check_price_a8_rows() {
    long bad = 0, a8 = 0, rows = 0;
    const g::FactsBoard* board = facts_board("rtx4090");
    if (board == nullptr) {
        std::fprintf(stderr, "FAIL: no facts board rtx4090\n");
        ++failures;
        return;
    }
    const auto selector =
        g::priced_routes_query(g::board_device_class(*board), QType::GGML_Q4_K,
                               LinearPolicy::AllowA8, g::Epilogue::Store, 4096, 4096, 64);
    int rc                 = -1;
    const std::string text = price_stdout({"--board", board_id(*board), "--format", "q4_k", "--n",
                                           "4096", "--k", "4096", "--t", "64", "--policy", "a8"},
                                          rc);
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: price --policy a8 exits %d\n", rc);
        ++bad;
    }
    for (const auto& p : selector) {
        const bool a16 = p.route.family == g::Family::RegisterMma ||
                         p.route.family == g::Family::TiledMma ||
                         p.route.family == g::Family::StagedMma;
        if (!a16) ++a8;
        const std::string label = a16 ? std::string(g::family_name(p.route.family)) + ":" +
                                            std::to_string(static_cast<int>(p.route.schedule))
                                      : g::route_name(p.route);
        if (text.find("\n" + label + " ") == std::string::npos) {
            std::fprintf(stderr, "FAIL: price --policy a8 prints no row for %s\n", label.c_str());
            ++bad;
        } else {
            ++rows;
        }
    }
    std::printf("price a8 rows: %zu routes priced by the selector (%ld A8), %ld printed, %ld "
                "differ\n",
                selector.size(), a8, rows, bad);
    // An A8 board with no A8 candidate tests nothing.
    if (bad > 0 || a8 == 0) ++failures;
}

// The bytes a warp of the probe's two-slot rings keeps in flight by count: 2 slots x 2 (ring2) or
// 4 (ring4) 16-byte loads a lane x 32 lanes.
constexpr double kRing2Bytes = 2 * 2 * 32 * 16, kRing4Bytes = 2 * 4 * 32 * 16;

// ---- Check 7: the DRAM time of #690 (dram_time) ----
// Over a grid of CTA counts, occupancies, warps and SM counts on an RTX 3090's facts with its
// SM rate row: (a) the time never rises with the SM count; (b) a grid of fewer CTAs than SMs costs
// the same on any larger chip (idle SMs are not billed: the charge of #690); (c) the time is at
// least every CTA's bytes over the bus; (d) a round of one CTA moves at its warps' rate,
// W x b_w / L0, while that is under P. Revert check: the legacy law (the busiest SM's share
// bw / SMs) fails (a) and (b) on the same grid, so the check does not pass vacuously.
void check_dram_time() {
    long bad = 0, legacy_rises = 0, cases = 0;
    const auto it =
        std::find_if(std::begin(g::kFactsBoards), std::end(g::kFactsBoards),
                     [](const g::FactsBoard& b) { return std::strcmp(b.name, "rtx3090") == 0; });
    if (it == std::end(g::kFactsBoards)) {
        std::fprintf(stderr, "FAIL: dram_time: no rtx3090 board\n");
        ++failures;
        return;
    }
    const DeviceClass d      = g::board_device_class(*it);
    const kr::ChipFacts base = kr::chip_facts_of(d);
    if (kr::legacy_dram(base)) {
        std::fprintf(stderr, "FAIL: dram_time: the RTX 3090 has no SM rate facts (row sm_86)\n");
        ++failures;
        return;
    }
    const double bytes   = 64.0 * 4096 * 0.5625; // a 64-row CTA of a 4.5-bit weight, K 4096
    const double b_w     = kRing2Bytes; // a BF16 m1 Store call's ring (under the 3090's b_w_max)
    const auto legacy_of = [&](const kr::ChipFacts& c, double ctas) {
        return std::ceil(ctas / c.sms) * bytes / (c.dram_bytes_per_s / c.sms);
    };
    for (const double ctas : {1.0, 7.0, 40.0, 82.0, 100.0, 164.0, 300.0, 1000.0, 5000.0}) {
        for (const double occ : {1.0, 2.0, 4.0, 12.0}) {
            for (const double wpc : {1.0, 4.0, 8.0}) {
                kr::ChipFacts less = base;
                for (const double sms : {41.0, 82.0, 103.0, 170.0}) {
                    kr::ChipFacts more = base;
                    more.sms           = sms;
                    if (more.sms <= less.sms) continue;
                    ++cases;
                    const double a = kr::dram_time(less, ctas, occ, wpc, bytes, b_w, false);
                    const double b = kr::dram_time(more, ctas, occ, wpc, bytes, b_w, false);
                    if (b > a * (1 + 1e-12)) ++bad;
                    if (ctas <= less.sms && b != a) ++bad;
                    if (b < ctas * bytes / more.dram_bytes_per_s) ++bad;
                    if (legacy_of(more, ctas) > legacy_of(less, ctas) * (1 + 1e-12)) {
                        ++legacy_rises;
                    }
                    less = more;
                }
            }
        }
    }
    // (d): one CTA of one warp on the 3090: 2048 B in flight over the measured L0, under P (the
    // free carve-out's, and the placed one's for a shared-memory tile).
    const double rate = b_w / *base.dram_latency_s;
    for (const bool placed : {false, true}) {
        const double p =
            (placed ? *base.sm_dram_bytes_per_clock_placed : *base.sm_dram_bytes_per_clock_free) *
            base.clock_hz;
        const double one = kr::dram_time(base, 1.0, 1.0, 1.0, bytes, b_w, placed);
        if (!(rate < p) ||
            std::abs(one - std::max(bytes / rate, bytes / base.dram_bytes_per_s)) > one * 1e-12) {
            ++bad;
        }
        // Saturated: 48 warps' worth of flight is over P, which then is the rate.
        const double sat = kr::dram_time(base, 1.0, 1.0, 48.0, bytes, b_w, placed);
        if (std::abs(sat - std::max(bytes / p, bytes / base.dram_bytes_per_s)) > sat * 1e-12) {
            ++bad;
        }
    }
    std::printf("dram_time: %ld cases, %ld differ; the legacy law rises in %ld of them\n", cases,
                bad, legacy_rises);
    if (bad > 0 || cases == 0 || legacy_rises == 0) ++failures;
}

// ---- Check 8: the law against the probe's own rows ----
// For every architecture of core/sm_rate_facts.h, dram_time prices one CTA of W warps on one SM,
// each warp keeping the counted bytes of one of the probe's two-slot rings in flight (ring2: 2048
// B, ring4: 4096 B, both under the row's b_w_max), and that rate is held to the probe's measured
// ring rows (both carve-outs, every W, at each row's own SM clock). P, L0 and b_w_max come from
// the table the rows sit in, so the check is about the counted b_w and the law's form (ring4 at
// W = 1 equals the row where its b_w_max is ring4's own reading: sm_75, sm_86, sm_89, and nearly on
// sm_70 and sm_120a). The law is Little's with the unloaded L0 and a hard corner at P, an upper
// bound of the rate, so:
//  - it is never below a measured row (to 1 %, kNeverBelow);
//  - at W = 1 it is at most kAtOneWarp above it: ring2's counted 2048 B against the 1767-1907 B the
//    rows give (x1.07-1.16); a b_w counted twice over or a P in another unit fails here;
//  - on every row at most kOnTheCurve above it. The worst reading, x1.16-1.34 at W = 8-16, is a gap
//    of the model, not noise: the measured curve rounds off into P (a knee of queueing near the
//    port's limit) where the law has a corner. It is recorded on #690; the law is optimistic there.
// The bounds are this check's tolerances, the same as warp_inflight.py's (LAW_*), not constants of
// the price. The other cards' runs of an architecture (its transfer checks) are held to the same
// bounds under the row's facts and their own SM clocks: the verdict must be the table's
// (`within_bounds`, `refuted`), and a refuted row gives no SM rate facts (fill_sm_rate_facts). The
// RTX 5070 Ti refutes the sm_120a row: the law is x0.88-0.89 of its rows at W = 1 (its warps fetch
// 12 % faster than the RTX 5090's) and x0.92 at its placed P (23.8 B/clk against 21.9).
// Last, the boards of each law, written out: the legacy law on sm_80, sm_90, sm_100, sm_121 (no
// row) and on sm_120a (refuted), the law of #690 on sm_70, sm_75, sm_86, sm_89.
constexpr double kNeverBelow = 0.99, kAtOneWarp = 1.20, kOnTheCurve = 1.40;

struct LawOverRows {
    long rows = 0, outside = 0;
    double lowest = 1e9, at_one = 0, highest = 0;
};

// The law of `row` over `rings`, both rings, both carve-outs. `report`: name each row outside the
// bounds (a reference card's rows must all be inside; a transfer check's may refute).
LawOverRows law_over(const ninferno::SmRateRow& row, const ninferno::SmRateRings& rings,
                     const char* who, bool report) {
    LawOverRows out;
    kr::ChipFacts chip;
    chip.sms                            = row.sms;
    chip.dram_latency_s                 = row.dram_stream_latency_ns * 1e-9;
    chip.sm_dram_bytes_per_clock_free   = row.bytes_per_clock_free;
    chip.sm_dram_bytes_per_clock_placed = row.bytes_per_clock_placed;
    chip.sm_max_warp_bytes              = row.max_warp_bytes_in_flight;
    // One CTA on one SM: the bus is not the bound the probe measured.
    chip.dram_bytes_per_s = 1e18;
    const double bytes    = 1 << 20;
    const std::tuple<const char*, double, bool, const std::vector<ninferno::SmRateRingRow>*> all[] =
        {{"ring2", kRing2Bytes, false, &rings.ring2_free},
         {"ring2", kRing2Bytes, true, &rings.ring2_placed},
         {"ring4", kRing4Bytes, false, &rings.ring4_free},
         {"ring4", kRing4Bytes, true, &rings.ring4_placed}};
    for (const auto& [variant, counted, placed, measured] : all) {
        for (const ninferno::SmRateRingRow& r : *measured) {
            ++out.rows;
            chip.clock_hz      = r.sm_mhz * 1e6;
            const double law   = bytes / kr::dram_time(chip, 1.0, 1.0, r.w, bytes, counted, placed);
            const double ratio = law / (r.gbps * 1e9);
            out.lowest         = std::min(out.lowest, ratio);
            out.highest        = std::max(out.highest, ratio);
            if (r.w == 1) out.at_one = std::max(out.at_one, ratio);
            if (ratio < kNeverBelow || ratio > kOnTheCurve || (r.w == 1 && ratio > kAtOneWarp)) {
                ++out.outside;
                if (report) {
                    std::fprintf(stderr,
                                 "FAIL: sm_%s %s %s %s W=%g: the law %.2f GB/s, the probe %.2f "
                                 "(x%.3f)\n",
                                 row.arch.c_str(), who, variant, placed ? "placed" : "free", r.w,
                                 law / 1e9, r.gbps, ratio);
                }
            }
        }
    }
    return out;
}

void check_law_against_probe() {
    long rows = 0, bad = 0;
    for (const ninferno::SmRateRow& row : ninferno::sm_rate_facts_table()) {
        const LawOverRows own = law_over(row, row.rings, row.card.c_str(), true);
        rows += own.rows;
        bad += own.outside;
        std::printf("law vs probe: sm_%s (%s, b_w_max %.0f B): x%.3f..%.3f, x%.3f at W = 1\n",
                    row.arch.c_str(), row.card.c_str(), row.max_warp_bytes_in_flight, own.lowest,
                    own.highest, own.at_one);
        bool outside = false;
        for (const ninferno::SmRateTransfer& t : row.transfers) {
            const LawOverRows other = law_over(row, t.rings, t.card.c_str(), false);
            const bool within       = other.outside == 0;
            outside                 = outside || !within;
            std::printf("law vs probe: sm_%s row against %s: x%.3f..%.3f, x%.3f at W = 1: %s\n",
                        row.arch.c_str(), t.card.c_str(), other.lowest, other.highest, other.at_one,
                        within ? "within the bounds" : "outside: refutes the row");
            if (within != t.within_bounds) {
                std::fprintf(stderr, "FAIL: sm_%s against %s: the table's verdict differs\n",
                             row.arch.c_str(), t.card.c_str());
                ++bad;
            }
        }
        kr::ChipFacts filled;
        kr::fill_sm_rate_facts(filled, row.major, row.minor);
        if (outside != row.transfer_refuted || kr::legacy_dram(filled) != row.transfer_refuted) {
            std::fprintf(stderr, "FAIL: sm_%s: refuted %d, the checks say %d, facts %s\n",
                         row.arch.c_str(), row.transfer_refuted, outside,
                         kr::legacy_dram(filled) ? "absent" : "filled");
            ++bad;
        }
    }
    // The boards of each law.
    const std::pair<const char*, bool> boards[] = {
        {"a100", true},       {"h100-sxm", true},   {"b300", true},
        {"gb10-spark", true}, {"rtx5090", true},    {"rtx5080", true},
        {"rtx5070ti", true},  {"rtx5060ti", true},  {"unknown-sm120-48sm", true},
        {"v100", false},      {"rtx2080ti", false}, {"rtx3090", false},
        {"a10", false},       {"rtx4090", false},   {"rtx4060ti", false},
        {"l4", false}};
    for (const auto& [name, legacy] : boards) {
        const auto it =
            std::find_if(std::begin(g::kFactsBoards), std::end(g::kFactsBoards),
                         [&](const g::FactsBoard& b) { return std::strcmp(b.name, name) == 0; });
        if (it == std::end(g::kFactsBoards)) {
            std::fprintf(stderr, "FAIL: no %s board\n", name);
            ++bad;
            continue;
        }
        if (kr::legacy_dram(kr::chip_facts_of(g::board_device_class(*it))) != legacy) {
            std::fprintf(stderr, "FAIL: %s is priced by the %s DRAM law, not the %s one\n", name,
                         legacy ? "#690" : "legacy", legacy ? "legacy" : "#690");
            ++bad;
        }
    }
    std::printf("law vs probe: %ld rows of the reference cards, %ld outside the bounds or "
                "miscalled\n",
                rows, bad);
    if (bad > 0 || rows == 0) ++failures;
}

int main() {
    check_golden();
    check_wiring();
    check_dram_time();
    check_law_against_probe();
    check_price_formats();
    check_price_admitted();
    check_price_arguments();
    check_price_a8_rows();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
