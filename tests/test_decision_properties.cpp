// Property tests for the projection selector's decisions, set 1 (roadmap #620 step 6; the
// specification is Roadmap 0's report 08, Q3). Every property runs over the board rows of
// core/data/gpu_facts.json through the same substitution the selector goldens use
// (selector_facts.h board_device_class, ScopedDeviceClass), so it needs no GPU to run. The
// register-MMA price reads the registers and spill of the compiled kernels from the committed
// kernel-facts snapshot of the SASS (generic_cost.cpp rmma_kept -> rmma_resources), not from the
// running GPU: the counts below are the same with a GPU visible and without one. They are not the
// same in every build: what P1 prices is the routes the build holds (see `Build` below).
//
//   P1  price monotone: for every priced route of a call of 17 tokens or more, a board with 1.25x
//       the SMs, or 1.25x the DRAM bandwidth, all else equal, prices it no slower (to 1e-9). The
//       price (price_routes), not the choice.
//   P3  fields the selector does not read: the board's labels (names, PCI address, UUID, ordinal),
//       the table row's name, sources and note, and (each on its own) its TDP, nominal VRAM, PCIe
//       generation and segment, and the memory size the driver reports, move no route. Every
//       variant also takes a slot of its own in the per-class caches (see `rebuilt`). Boards of the
//       table that differ in their PCI id alone (SKU twins) choose alike. The subject is what
//       route_query answers over the grid below: the gates select_route asks (the W4A8 workspace
//       gate on 5 rows, the R8 route table on 22). Not reached: the cached int8-prefill gate
//       (generic_gemm.cpp, asked by q8_prefill_a8 under the prefill scope, not by select_route) and
//       the route the W4A8 path executes (w4a8_executed_route). Every variant moves global_bytes
//       (the cache slot), so a gate that read it would fail under several variant names at once.
//       Decisions only: a non-fact that scales every route's price alike (the ordinal or the TDP
//       as a common factor) flips no decision and is invisible here; one that scales some routes
//       and not others is caught.
//   P6  clock invariance over every row: the clocks the driver reports (a factory overclock, a
//       vendor's lower boost, none at all) move no route of a board the table knows
//       (test_selector_golden.cpp run_clock_invariance does this for the golden boards only).
//       A row whose facts read the driver on purpose (a laptop without a published boost floor:
//       MachineFacts::driver_read) is counted and left out.
//       The decision only: a clock that leaks into a price and flips no route stays invisible here.
//
// P8 (B1 under `exact` reads no continuous fact) is in test_plan_dry_run.cpp, where the plan
// fixtures are.
//
// A property that fails on today's code stays a test: its failures are grouped by (property, key)
// and compared with kKnown below, the expected failures, each with the function that causes it.
// A failure outside kKnown fails the test; a kKnown entry that no longer fails is reported
// (XPASS) and should be removed. Every group prints its count and its worst case.

#include "core/device_class.h"
#include "core/machine_facts.h"
#include "core/units.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "ops/common/arch_limits.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_select_internal.h"
#include "ops/generic/route_query.h"
#include "ops/generic/route_table.h"
#include "ops/generic/selector_facts.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using namespace ninferno;
namespace generic = ninferno::ops::detail::generic;
using ops::LinearPolicy;
using generic::Epilogue;

// ---- Expected failures -------------------------------------------------------------------------

// A waived group's recorded window, for one build (`Build` below).
//
// The share of the cases this property checked for the route's family that the waiver covers, as
// [min_share, max_share]. A share, not an absolute count, so that a table with more rows does not
// trip it by size alone. Both ends are RECORDED from this test's output (the "record" line it
// prints for every waived group), not fitted:
//   max_share  the share measured, rounded up to a tenth of a percent. A group that grows past it
//              fails: the waiver may not hide a law that got worse.
//   min_share  the one-row floor measured, rounded down to a hundredth of a percent: the lowest
//              share one edit of one board row of the table can leave, (c - c_max) / (F + F_max),
//              where c is the group's cases, F the family's, c_max the most cases of the group one
//              row holds and F_max the most cases of the family one row holds (removing or changing
//              that row takes at most c_max of the group away; the row changed or added brings at
//              most F_max cases, as large as the largest row of today). A share between min_share
//              and the recorded one passes and prints "re-record kKnown"; one below min_share is
//              more than one row of the table explains (a change of the law, or of several rows)
//              and fails, so that the ceiling cannot be left loose above an improved law: the
//              room a regression can hide in under a stale ceiling is at most one row's worth.
// The worst price ratio the group covers, as [min_ratio, max_ratio]; 0 in both: no ratio check.
//   max_ratio  the worst measured, rounded up to 0.01; worse fails.
//   min_ratio  the runner-up row's worst (the worst ratio of the group without the row that holds
//              the worst case), rounded down to 0.01: an edit of one row leaves the worst at least
//              there. Between it and the recorded worst: passes, prints "re-record kKnown".
// The "within" groups have no ratio check: their membership is by each row's own SM ratio (the
// test below), so no larger ratio can occur in them.
struct Window {
    double min_share;
    double max_share;
    double min_ratio;
    double max_ratio;
};

// What P1 prices is the routes the build holds, and two of the build's own facts change it (the
// selector's admission reads them, not the substituted board): whether it compiles the NVFP4
// kernels and the thread-block cluster splits of the register MMA (CMakeLists.txt
// NINFERNO_FULL_CUDA_ARCHS: 120a, 121a, 120f; without them generic_gemm.cpp
// rmma_schedule_launchable admits no cluster schedule and w4a4_schedule_count is 0), and the
// shared memory one block may opt into (ops/common/arch_limits.h kMaxSharedBytesPerBlock, which
// admits the W4A8 and I8 tiles: w4a8_fits, i8_fits). An sm_120a build prices the cluster
// schedules of the register MMA on every board with clusters, cases an sm_89 build never sees, so
// the shares differ by build and are recorded per build. Windows are recorded for the 99 KiB
// builds, with the NVFP4 kernels (sm_120a) and without them (sm_89; an sm_86 build holds the same
// routes). Another build (sm_70, sm_75, sm_80) has no recorded window: the groups are
// still matched (a new group or an XPASS fails), the shares are printed and not checked.
struct Build {
    bool nvfp4_kernels;
    std::size_t block_shared_bytes;
};
#ifdef NINFERNO_WITHOUT_NVFP4
constexpr Build kBuild{false, ops::kMaxSharedBytesPerBlock};
#else
constexpr Build kBuild{true, ops::kMaxSharedBytesPerBlock};
#endif
constexpr bool kBuildRecorded = kBuild.block_shared_bytes == 99 * 1024;

struct Known {
    const char* property;
    const char* key;
    const char*
        cause; // which charge of the price law explains the group (the three charges, below)
    Window without_nvfp4; // recorded on an sm_89 build
    Window with_nvfp4;    // recorded on an sm_120a build
};

const Window* window_of(const Known& k) {
    if (!kBuildRecorded) return nullptr;
    return kBuild.nvfp4_kernels ? &k.with_nvfp4 : &k.without_nvfp4;
}

// The price of a route on a board with more SMs is higher, by up to x1.52, for the families below
// (found by this test, roadmap #620 step 6; the defect of the law is issue #690). price_routes
// (ops/generic/generic_cost.cpp) charges for idle SMs in three places:
//   (a) the memory term charges the busiest SM q = ceil(CTAs / sms) CTAs of bytes at bw / sms
//       (`cost` lambda and the A8 loops): bw / sms is that SM's share only when every SM pulls at
//       once, so q x sms CTA slots are billed and a grid whose last wave leaves SMs idle is
//       overcharged when SMs are added (50 -> 63 SMs for 100 CTAs: q stays 2, q x sms goes 100 ->
//       126); the DRAM floor n k bpw / bw never binds;
//   (b) the A8 loops (W4A8, I8) keep the fleet-wide fill min(1, CTAs / sms) in the warp fill, which
//       falls when SMs are added;
//   (c) the bus-bound memory time is divided by the busy SM's warp fill (the register-MMA and
//       pipelined paths), so a fill that drops with more SMs raises it.
// Which charge explains which group below comes from experiments on a scratch copy of the law
// (each charge removed alone and together; made by the review of this PR on a build without the
// NVFP4 kernels, not run by this test; the shares in the causes are that build's):
// with all three removed 0.83 % of the SM-scale cases still fail (register MMA only, worst x1.336)
// against 33.5 % today. The cause of that residual is in the plan comment of issue #690
// (https://github.com/Brontir/NInferno/issues/690#issuecomment-6773348851): the split-K reduce is
// priced under the per-CTA fill, and the shared activation read loses a co-resident CTA when the
// busiest SM's CTA count grows.
//
// Each failing case is sorted by what can explain it. "within the SM ratio": the price rose by no
// more than sms' / sms, the most q x sms (the bytes charged to the busiest SM) can grow. "beyond
// the SM ratio": the warp-fill term adds to it. A waived group may not grow past max_share of the
// family's cases nor past max_ratio, nor fall below its one-row floors min_share / min_ratio
// (`Window`); a group that stops failing is an XPASS that fails the test. So a change that makes
// the price rise with the SMs in more cases or by more (a bw / sqrt(sms) share, say) fails, and so
// does a fix of the law that shrinks a group by more than one board row can: the window is then
// re-recorded from the "record" line the test prints. Bandwidth x1.25
// fails nowhere. The tolerance is 1e-9: the price is deterministic. (The test cannot see a route's
// CTA count, so it cannot tell a rise a wave count explains from one it does not, case by case,
// and the shares cannot be replaced by a predicate.)
constexpr Known kKnown[] = {
    {"P1 sms x1.25",
     "rmma within the SM ratio",
     "charge (a): removed alone the group falls from 35.0 % to 2.6 % of the family's cases, with "
     "all "
     "three removed to 0.99 % (issue #690)",
     {0.341, 0.351, 0, 0},
     {0.3337, 0.344, 0, 0}},
    {"P1 sms x1.25",
     "rmma beyond the SM ratio",
     "charge (c) with (a): removing (c) alone leaves 20 of the 675 cases (x1.336), removing (a) "
     "alone grows the group to 13 156 (x1.90): the two charges change together (issue #690)",
     {0.0007, 0.001, 1.4, 1.43},
     {0.0007, 0.001, 1.4, 1.43}},
    {"P1 sms x1.25",
     "staged within the SM ratio",
     "charge (a): removed alone the group disappears (issue #690)",
     {0.4184, 0.436, 0, 0},
     {0.4184, 0.436, 0, 0}},
    {"P1 sms x1.25",
     "w4a8 within the SM ratio",
     "charges (a) and (b): removing (b) alone leaves 14.0 % of the family's cases, all three none "
     "(issue #690)",
     {0.2282, 0.239, 0, 0},
     {0.2287, 0.239, 0, 0}},
    {"P1 sms x1.25",
     "w4a8 beyond the SM ratio",
     "charges (a) and (b) together: removing either alone makes the group disappear (issue #690)",
     {0.0223, 0.025, 1.52, 1.53},
     {0.0226, 0.025, 1.52, 1.53}},
    {"P1 sms x1.25",
     "i8 within the SM ratio",
     "charge (b): removed alone the group falls from 39.1 % to 0.72 % of the family's cases "
     "(issue #690)",
     {0.3781, 0.391, 0, 0},
     {0.3781, 0.391, 0, 0}},
};

// ---- Findings ----------------------------------------------------------------------------------

struct Group {
    int count    = 0;
    double worst = 0; // property-specific severity (a price ratio); 0 where none
    std::string example;
    std::map<std::string, int> rows;         // the group's cases per board row
    std::map<std::string, double> row_worst; // the worst severity per board row
};

std::map<std::pair<std::string, std::string>, Group> groups;
int cases_checked = 0;

void note(const std::string& property, const std::string& key, const std::string& row,
          const std::string& example, double severity = 0) {
    Group& g = groups[{property, key}];
    ++g.count;
    ++g.rows[row];
    double& row_worst = g.row_worst[row];
    row_worst         = std::max(row_worst, severity);
    if (g.example.empty() || severity > g.worst) {
        g.example = example;
        g.worst   = std::max(g.worst, severity);
    }
}

// ---- The rows ----------------------------------------------------------------------------------

struct Row {
    std::string name;
    std::string label; // the name and the PCI id: unique where names repeat (laptop boards)
    generic::FactsBoard board;
    const GpuFactsRow* source;
};

std::vector<Row> rows_of_table() {
    std::vector<Row> rows;
    const auto table = gpu_facts_table();
    rows.reserve(table.size()); // `board.name` points into the element's own string: no moves
    for (const GpuFactsRow& r : table) {
        if (r.sms <= 0 || r.boost_mhz <= 0 || r.bus_bits <= 0 || r.gbps <= 0) { continue; }
        Row row;
        row.name = r.name;
        std::ostringstream label;
        label << r.name << " [0x" << std::hex << std::uppercase << r.pci_device_id << ']';
        row.label  = label.str();
        row.source = &r;
        rows.push_back(std::move(row));
        Row& placed = rows.back();
        // The numbers as the CUDA query reports a stock board (memory clock: half the data rate).
        placed.board =
            generic::FactsBoard{placed.name.c_str(),
                                r.pci_device_id,
                                r.major,
                                r.minor,
                                r.sms,
                                static_cast<std::size_t>(std::lround(r.l2_mib)),
                                static_cast<std::size_t>(std::max(1L, std::lround(r.vram_gib))),
                                r.boost_mhz,
                                static_cast<int>(std::lround(r.gbps * 500.0)) + 1,
                                r.bus_bits};
    }
    return rows;
}

// ---- Problems ----------------------------------------------------------------------------------

struct Shape {
    std::int32_t n, k;
};

constexpr QType kFormats[] = {
    QType::BF16,      QType::Q4_G64_FP16, QType::Q8_G32_FP16, QType::NVFP4,     QType::MXFP4,
    QType::GGML_Q4_K, QType::GGML_Q6_K,   QType::GGML_IQ2_S,  QType::GGML_Q4_0, QType::GGML_IQ4_XS};
constexpr Shape kShapes[] = {{1024, 4096}, {17408, 5120}};
// Every width of the decode, verify and prefill bands the selector tells apart.
constexpr std::int32_t kTokens[]      = {1, 2, 4, 8, 12, 16, 17, 32, 64, 256, 1024};
constexpr std::int32_t kPriceTokens[] = {17, 32, 64, 256, 1024};

std::vector<LinearPolicy> policies_of(QType q) {
    std::vector<LinearPolicy> out = {LinearPolicy::A16Only, LinearPolicy::AllowA8};
    if (q == QType::NVFP4) { out.push_back(LinearPolicy::AllowA4); }
    return out;
}

// What route_query answers for the whole grid, as text: a route name, "none" where the generic
// path reads no such weight, "throw: <what>" where the selector refuses. Two boards that decide
// alike give the same text.
std::string decisions_of(const DeviceClass& d) {
    std::ostringstream out;
    for (const QType q : kFormats) {
        for (const Shape s : kShapes) {
            for (const LinearPolicy policy : policies_of(q)) {
                for (const Epilogue e : {Epilogue::Store, Epilogue::SwiGlu}) {
                    const std::int32_t rows = e == Epilogue::SwiGlu ? s.n / 2 : s.n;
                    for (const std::int32_t t : kTokens) {
                        for (const bool prefill : {false, true}) {
                            if (prefill && t < 256) { continue; }
                            out << static_cast<int>(q) << '/' << static_cast<int>(policy) << '/'
                                << static_cast<int>(e) << '/' << s.n << 'x' << s.k << "/T" << t
                                << (prefill ? "p" : "") << '=';
                            try {
                                const auto r =
                                    generic::route_query(d, q, policy, e, rows, s.k, t, prefill);
                                out << (r ? generic::route_name(*r) : std::string("none"));
                            } catch (const std::exception& error) {
                                out << "throw: " << error.what();
                            }
                            out << '\n';
                        }
                    }
                }
            }
        }
    }
    return out.str();
}

// The first line two decision texts differ at, "<problem> was <a>, is <b>".
std::string first_difference(const std::string& a, const std::string& b) {
    std::istringstream la(a), lb(b);
    std::string x, y;
    while (std::getline(la, x) && std::getline(lb, y)) {
        if (x != y) { return x + "  ->  " + y; }
    }
    return "(length differs)";
}

// The lines of a decision text that name a route: a property over a text of refusals is vacuous.
std::size_t routes_in(const std::string& dump) {
    std::istringstream in(dump);
    std::size_t n = 0;
    for (std::string line; std::getline(in, line);) {
        n += line.find("=none") == std::string::npos && line.find("throw:") == std::string::npos;
    }
    return n;
}

std::size_t differing_lines(const std::string& a, const std::string& b) {
    std::istringstream la(a), lb(b);
    std::string x, y;
    std::size_t n = 0;
    while (std::getline(la, x) && std::getline(lb, y)) { n += x != y ? 1 : 0; }
    return n;
}

// ---- P1: the price is monotone in SMs and bandwidth ---------------------------------------------

// The weight as the converter lays it out, its bytes never read. A copy of route_query.cpp's
// describe_weight (private there; price_routes needs the Weight, route_query returns only the
// route).
struct DescribedWeight {
    WeightParent parent;
    WeightView view;
    Weight weight;
    DescribedWeight()                                  = default;
    DescribedWeight(const DescribedWeight&)            = delete;
    DescribedWeight& operator=(const DescribedWeight&) = delete;
};

void describe(DescribedWeight& out, QType qtype, Epilogue epilogue, std::int32_t n,
              std::int32_t k) {
    const std::uint64_t own = epilogue == Epilogue::SwiGlu ? 2ULL * static_cast<std::uint64_t>(n)
                                                           : static_cast<std::uint64_t>(n);
    const bool nvfp4        = qtype == QType::NVFP4;
    out.parent.data         = reinterpret_cast<const std::byte*>(std::uintptr_t{1} << 32);
    out.parent.weight_scale_divisor = nvfp4 ? 1.0F : 0.0F;
    const std::uint64_t shape[2]    = {own, static_cast<std::uint64_t>(k)};
    out.parent.geometry             = weight_geometry(qtype, default_weight_layout(qtype), shape);
    out.view.shape                  = {own, static_cast<std::uint64_t>(k)};
    out.view.parts = {WeightRegion{&out.parent, 0, own * static_cast<std::uint64_t>(k)}};
    out.weight     = native_weight(out.view, nvfp4 ? 1.0F : 0.0F);
}

std::string family_name(int family) {
    return std::string(generic::family_name(static_cast<generic::Family>(family)));
}

// The cases P1 checked on the SM scale, per route family: the denominator of a waiver's share.
std::map<std::string, int> sms_cases_by_family;
// The same per family and board row: the most one row holds bounds what an edit of one row of the
// table can move a share by (Window min_share).
std::map<std::string, std::map<std::string, int>> sms_cases_by_family_row;
int sms_cases_priced = 0, routes_skipped_unpriced = 0;
int sms_cases_fell = 0, sms_cases_equal = 0; // the price fell / stayed (to 1e-9) with the SMs

using Prices = std::map<std::pair<int, int>, double>; // (family, schedule) -> seconds

// The time of every route the cost model prices for the call on `d`; false where the board has no
// roofline (nothing priced).
bool prices_of(const DeviceClass& d, const generic::Problem& problem, const Weight& weight,
               std::int32_t tokens, Prices& out) {
    const ScopedDeviceClass scoped(d);
    out.clear();
    return generic::price_routes(problem, weight, tokens, [&](generic::Route r, double seconds) {
        out[{static_cast<int>(r.family), static_cast<int>(r.schedule)}] = seconds;
    });
}

constexpr double kResolution = 1e-9; // the price is deterministic: no resolution is needed
constexpr double kScale      = 1.25;

DeviceClass with_sms(DeviceClass d) {
    d.sms = static_cast<int>(std::ceil(d.sms * kScale));
    return d;
}

DeviceClass with_bandwidth(DeviceClass d) {
    d.memory_clock_khz       = static_cast<int>(std::lround(d.memory_clock_khz * kScale));
    d.facts.memory_clock_khz = static_cast<int>(std::lround(d.facts.memory_clock_khz * kScale));
    return d;
}

void property_price_monotone(const std::vector<Row>& rows) {
    std::size_t unpriced_rows = 0;
    for (const Row& row : rows) {
        const DeviceClass base = generic::board_device_class(row.board);
        if (!base.facts.has_roofline()) {
            ++unpriced_rows;
            continue;
        }
        const DeviceClass more_sms = with_sms(base);
        const DeviceClass more_bw  = with_bandwidth(base);
        for (const QType q : kFormats) {
            for (const Shape s : kShapes) {
                for (const LinearPolicy policy : policies_of(q)) {
                    for (const Epilogue e : {Epilogue::Store, Epilogue::SwiGlu}) {
                        const std::int32_t rows_out = e == Epilogue::SwiGlu ? s.n / 2 : s.n;
                        DescribedWeight w;
                        describe(w, q, e, rows_out, s.k);
                        if (!generic::supports(w.weight)) { continue; }
                        const generic::Problem problem{q, policy, rows_out, s.k, e};
                        for (const std::int32_t t : kPriceTokens) {
                            Prices stock, sms, bw;
                            if (!prices_of(base, problem, w.weight, t, stock)) { continue; }
                            prices_of(more_sms, problem, w.weight, t, sms);
                            prices_of(more_bw, problem, w.weight, t, bw);
                            for (const auto& [route, seconds] : stock) {
                                const generic::Route r{static_cast<generic::Family>(route.first),
                                                       static_cast<std::uint8_t>(route.second)};
                                const std::string where =
                                    row.name + " " + ops::detail::generic::qtype_name(q) + " " +
                                    std::to_string(s.n) + "x" + std::to_string(s.k) + " e" +
                                    std::to_string(static_cast<int>(e)) + " p" +
                                    std::to_string(static_cast<int>(policy)) + " T" +
                                    std::to_string(t) + " " + generic::route_name(r);
                                const std::pair<const char*, const Prices*> scaled[] = {
                                    {"P1 sms x1.25", &sms}, {"P1 bandwidth x1.25", &bw}};
                                for (const auto& [property, prices] : scaled) {
                                    const auto it = prices->find(route);
                                    if (it == prices->end()) {
                                        // Priced on the stock board, not on the scaled one: no
                                        // case, but counted and printed.
                                        ++routes_skipped_unpriced;
                                        continue;
                                    }
                                    ++cases_checked;
                                    if (prices == &sms) {
                                        ++sms_cases_priced;
                                        ++sms_cases_by_family[family_name(route.first)];
                                        ++sms_cases_by_family_row[family_name(route.first)]
                                                                 [row.label];
                                        if (it->second < seconds * (1.0 - kResolution)) {
                                            ++sms_cases_fell;
                                        } else if (it->second <= seconds * (1.0 + kResolution)) {
                                            ++sms_cases_equal;
                                        }
                                    }
                                    if (it->second <= seconds * (1.0 + kResolution)) { continue; }
                                    std::string key = family_name(route.first);
                                    if (prices == &sms) {
                                        // A grid of fewer CTAs than SMs (q = 1) is charged
                                        // q x sms of bytes: its memory term rises with the SM
                                        // ratio, and no wave count raises it further. What rises
                                        // beyond that is the warp-fill term.
                                        key += it->second <= seconds *
                                                                 (more_sms.sms / double(base.sms)) *
                                                                 (1.0 + kResolution)
                                                   ? " within the SM ratio"
                                                   : " beyond the SM ratio";
                                    }
                                    std::ostringstream text;
                                    text << where << ": " << seconds << " s -> " << it->second
                                         << " s";
                                    note(property, key, row.label, text.str(),
                                         it->second / seconds);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout << "P1: " << rows.size() - unpriced_rows << " priced rows, " << unpriced_rows
              << " without a roofline, " << sms_cases_priced << " SM-scale cases and "
              << cases_checked - sms_cases_priced << " bandwidth-scale cases, "
              << routes_skipped_unpriced << " routes priced on stock and not on the scaled board\n";
    for (const auto& [family, n] : sms_cases_by_family) {
        std::cout << "    SM-scale cases of " << family << ": " << n << '\n';
    }
    std::cout << "    SM-scale cases: price fell " << sms_cases_fell << ", unchanged "
              << sms_cases_equal << ", rose " << sms_cases_priced - sms_cases_fell - sms_cases_equal
              << " (the cases the waivers below cover)\n";
    if (sms_cases_priced == 0) {
        note("vacuous", "P1 priced no route", "(all rows)", "no route was priced on the SM scale");
    }
    if (routes_skipped_unpriced != 0) {
        // A route priced on the stock board and not on a scaled one is a case P1 no longer
        // verifies for that route (a fact-dependent admission, say): a failure, not a skip.
        note("vacuous", "P1 route priced on the stock board only", "(all rows)",
             std::to_string(routes_skipped_unpriced) + " routes");
    }
}

// ---- P3: fields the selector does not read move nothing
// ------------------------------------------------

// The board as board_device_class builds it (the production builder), with `edit` applied to the
// driver-side fields and the facts derived again from machine_facts over `table`: every derived
// field (mma_slow, the facts) follows from the edited inputs, none is edited on a built class.
//
// Every variant also takes a slot of its own in anything the selector caches per device class
// (the per-class gates select_route asks, in generic_gemm.cpp, are computed once per class key, and
// the key does not
// hold a name, an ordinal, a PCI address, a UUID or the table row): `slot` changes the reported
// memory size, the one key field this test pins as neutral (the "reported memory size" variant),
// by a different amount for each variant. Without it a read of a label made inside a cached gate
// would be answered from the stock board's slot and never be seen.
DeviceClass rebuilt(const Row& row, const std::vector<GpuFactsRow>& table, int slot,
                    const std::function<void(DeviceClass&)>& edit) {
    DeviceClass d = generic::board_device_class(row.board);
    edit(d);
    d.global_bytes = d.global_bytes * 2 + 4096 * static_cast<std::size_t>(slot);
    d.facts        = machine_facts(driver_report_of(d), table);
    return d;
}

// One variant of the board and its matched control: the control is built with the same `slot` (so
// the same reported memory size and cache key) and no edit, so that the variant and the control
// differ in the field under test alone. A variant compared with the stock board only also differs
// in the reported memory size, which `rebuilt` changes for the cache: a route that read a ratio of
// the edited fact and the reported memory would not move.
struct Variant {
    std::string what;
    DeviceClass board;
    DeviceClass control;
    // False for the variant that is the control itself (the reported memory size, which `rebuilt`
    // changes by the slot alone): comparing it with its own control would be a tautology.
    bool has_control = true;
};

// The same board with fields the projection selector does not read changed. (Segment, TDP and the
// PCIe generation are facts a B2 decision may read, machine_facts.h; the selector reads none of
// them, and this is what the property pins: each has a variant of its own, so a failure names it.)
std::vector<Variant> relabelled(const Row& row) {
    const std::span<const GpuFactsRow> embedded = gpu_facts_table();
    const std::vector<GpuFactsRow> stock(embedded.begin(), embedded.end());
    std::vector<Variant> out;
    const auto add = [&](const char* what, const std::vector<GpuFactsRow>& table, int slot,
                         const std::function<void(DeviceClass&)>& edit, bool has_control = true) {
        out.push_back({what, rebuilt(row, table, slot, edit),
                       rebuilt(row, stock, slot, [](DeviceClass&) {}), has_control});
    };
    add("reported memory size", stock, 1, [](DeviceClass&) {}, false);
    add("names", stock, 2, [](DeviceClass& d) { d.name = "another name"; });
    add("ordinal, PCI address, UUID", stock, 3, [](DeviceClass& d) {
        d.device              = 3;
        d.identity.pci_domain = 1;
        d.identity.pci_bus    = 0x41;
        d.identity.uuid.fill(0xAB);
    });
    // The table row's fields, through the loader's own derivation of the facts.
    const auto edited_row = [&](const std::function<void(GpuFactsRow&)>& edit) {
        std::vector<GpuFactsRow> table = stock;
        for (GpuFactsRow& r : table) {
            if (r.pci_device_id == row.source->pci_device_id) { edit(r); }
        }
        return table;
    };
    const auto none = [](DeviceClass&) {};
    add("the table row's name, note, sources", edited_row([](GpuFactsRow& r) {
            r.name    = "renamed in the table";
            r.note    = "a different note";
            r.sources = {"https://example.invalid/other"};
        }),
        4, none);
    add("the table row's tdp", edited_row([](GpuFactsRow& r) { r.tdp_w = r.tdp_w * 2 + 1; }), 5,
        none);
    add("the table row's nominal vram",
        edited_row([](GpuFactsRow& r) { r.vram_gib = r.vram_gib * 2 + 1; }), 6, none);
    add("the table row's pcie generation",
        edited_row([](GpuFactsRow& r) { r.pcie_gen = r.pcie_gen == 5 ? 3 : 5; }), 7, none);
    add("the table row's segment", edited_row([](GpuFactsRow& r) {
            r.segment = r.segment == Segment::Datacenter ? Segment::GeForce : Segment::Datacenter;
        }),
        8, none);
    return out;
}

void property_non_fact_fields(const std::vector<Row>& rows) {
    std::size_t routes_compared = 0;
    for (const Row& row : rows) {
        const DeviceClass d      = generic::board_device_class(row.board);
        const std::string stock  = decisions_of(d);
        const std::size_t routes = routes_in(stock);
        if (routes == 0) {
            note("vacuous", "no route chosen", row.name,
                 row.name + ": every decision is a refusal");
        }
        for (const Variant& v : relabelled(row)) {
            // Against the stock board, and against the matched control (same cache slot, no edit).
            const std::string control = v.has_control ? decisions_of(v.control) : std::string();
            const std::string changed = decisions_of(v.board);
            std::vector<std::pair<const char*, const std::string*>> against = {{"", &stock}};
            if (v.has_control) { against.emplace_back(" against its matched control", &control); }
            for (const auto& [suffix, reference] : against) {
                ++cases_checked;
                routes_compared += routes;
                if (changed == *reference) { continue; }
                const std::string what = v.what + suffix;
                note("P3 unread fields", what, row.name,
                     row.name + " (" + what +
                         "): " + std::to_string(differing_lines(*reference, changed)) +
                         " decisions move, first " + first_difference(*reference, changed));
            }
        }
    }
    // SKU twins: boards whose rows differ in nothing but the PCI id and the labels.
    std::map<std::string, std::vector<const Row*>> twins;
    for (const Row& row : rows) {
        const GpuFactsRow& r = *row.source;
        std::ostringstream key;
        key << r.chip << '/' << r.major << r.minor << '/' << r.sms << '/' << r.boost_mhz << '/'
            << r.boost_min_mhz << '/' << r.bus_bits << '/' << r.gbps << '/' << r.l2_mib << '/'
            << r.mma_slow << '/' << static_cast<int>(r.form);
        twins[key.str()].push_back(&row);
    }
    std::size_t pairs = 0;
    for (const auto& [key, group] : twins) {
        for (std::size_t i = 1; i < group.size(); ++i) {
            ++pairs;
            ++cases_checked;
            const std::string a = decisions_of(generic::board_device_class(group[0]->board));
            const std::string b = decisions_of(generic::board_device_class(group[i]->board));
            if (a == b) { continue; }
            note("P3 unread fields", "SKU twins (PCI id alone differs)",
                 group[0]->name + " / " + group[i]->name,
                 group[0]->name + " vs " + group[i]->name + ": " +
                     std::to_string(differing_lines(a, b)) + " decisions move, first " +
                     first_difference(a, b));
        }
    }
    if (routes_compared == 0) { note("vacuous", "P3 compared no route", "(all rows)", "none"); }
    std::cout << "P3: " << rows.size() << " rows relabelled, " << routes_compared
              << " route decisions compared, " << pairs << " SKU twin pairs\n";
}

// ---- P6: the driver's clocks move no route of a board the table knows ---------------------------

void property_clock_invariance(const std::vector<Row>& rows) {
    // The variants of test_selector_golden.cpp run_clock_invariance: a factory overclock, a
    // memory clock inside the grid step, both raised, and a driver that reports no clock at all.
    constexpr struct {
        double sm, memory;
    } kVariants[] = {{0.85, 1.0}, {1.0, 0.985}, {1.08, 1.07}, {0.0, 1.0}, {1.0, 0.0}, {0.0, 0.0}};

    std::size_t driver_read_rows = 0;
    std::vector<std::string> exempt;
    for (const Row& row : rows) {
        const DeviceClass stock_class = generic::board_device_class(row.board);
        if (stock_class.facts.driver_read) {
            ++driver_read_rows;
            exempt.push_back(row.name);
            continue; // priced from the driver's value on purpose (machine_facts.h)
        }
        const std::string stock = decisions_of(stock_class);
        if (routes_in(stock) == 0) {
            note("vacuous", "no route chosen", row.name,
                 row.name + ": every decision is a refusal");
        }
        for (const auto v : kVariants) {
            ++cases_checked;
            const DeviceClass moved   = generic::board_device_class(row.board, v.sm, v.memory);
            const std::string changed = decisions_of(moved);
            if (changed == stock) { continue; }
            std::ostringstream key;
            key << "SM x" << v.sm << ", memory x" << v.memory;
            note("P6 clock invariance", key.str(), row.name,
                 row.name + " (" + key.str() +
                     "): " + std::to_string(differing_lines(stock, changed)) +
                     " decisions move, first " + first_difference(stock, changed));
        }
    }
    std::cout << "P6: " << rows.size() - driver_read_rows << " rows checked under "
              << std::size(kVariants) << " clock variants, " << driver_read_rows
              << " rows read the driver's clock on purpose and are left out";
    for (const std::string& name : exempt) { std::cout << "\n    exempt: " << name; }
    std::cout << '\n';
}

// ---- The waivers' windows ----------------------------------------------------------------------

// A waived P1 group against its window (`Window`): prints its share, its one-row floors and the
// "record" line kKnown is re-recorded from; returns 1 where the group fails its window.
int check_window(const Known& known, const std::string& key, const Group& g) {
    const std::string family = key.substr(0, key.find(' '));
    const int family_cases   = sms_cases_by_family[family];
    const double share       = family_cases > 0 ? g.count / static_cast<double>(family_cases) : 1.0;
    int row_max              = 0; // the most of the group's cases one row holds
    for (const auto& [row, n] : g.rows) { row_max = std::max(row_max, n); }
    int family_row_max = 0; // the most of the family's cases one row holds
    for (const auto& [row, n] : sms_cases_by_family_row[family]) {
        family_row_max = std::max(family_row_max, n);
    }
    const double floor_share =
        family_cases > 0 ? (g.count - row_max) / static_cast<double>(family_cases + family_row_max)
                         : 0.0;
    // The worst ratio without the row that holds the worst case.
    double runner_up = 0;
    bool worst_seen  = false;
    for (const auto& [row, worst] : g.row_worst) {
        if (!worst_seen && worst == g.worst) {
            worst_seen = true;
            continue;
        }
        runner_up = std::max(runner_up, worst);
    }
    const bool ratio_group = key.find("beyond") != std::string::npos;
    std::cout << "    share of the family's " << family_cases
              << " SM-scale cases: " << 100.0 * share << " %, one-row floor " << 100.0 * floor_share
              << " % (" << row_max << " cases on one row, " << family_row_max
              << " of the family's on one row)";
    if (ratio_group) { std::cout << ", runner-up row's worst x" << runner_up; }
    std::cout << "\n    record: {" << std::floor(floor_share * 1e4) / 1e4 << ", "
              << std::ceil(share * 1e3) / 1e3 << ", "
              << (ratio_group ? std::floor(runner_up * 100) / 100 : 0.0) << ", "
              << (ratio_group ? std::ceil(g.worst * 100) / 100 : 0.0) << "}\n";

    const Window* w = window_of(known);
    if (w == nullptr) {
        std::cout << "    not checked: no window is recorded for this build\n";
        return 0;
    }
    const bool ratio_checked = w->max_ratio > 0;
    std::cout << "    window: share " << 100.0 * w->min_share << " .. " << 100.0 * w->max_share
              << " %";
    if (ratio_checked) {
        std::cout << ", worst ratio x" << w->min_ratio << " .. x" << w->max_ratio;
    }
    std::cout << '\n';
    if (share > w->max_share || (ratio_checked && g.worst > w->max_ratio * (1.0 + kResolution))) {
        std::cout << "    WAIVER EXCEEDED: the law got worse, or the board table changed; if only "
                     "core/data/gpu_facts.json changed, re-record the window from the record "
                     "line above, otherwise fix the law (issue #690)\n";
        return 1;
    }
    if (share < w->min_share || (ratio_checked && g.worst < w->min_ratio)) {
        std::cout << "    WAIVER BELOW ITS ONE-ROW FLOOR: the group shrank by more than an edit of "
                     "one board row explains (the law improved, or several rows changed): "
                     "re-record kKnown from the record line above\n";
        return 1;
    }
    // More than the rounding step under the ceiling: below the share recorded.
    if (share < w->max_share - 1e-3 || (ratio_checked && g.worst < w->max_ratio - 1e-2)) {
        std::cout << "    shrank below its recorded share, within one row of the table: passes; "
                     "re-record kKnown from the record line above\n";
    }
    return 0;
}

} // namespace

int main() {
    try {
        const std::vector<Row> rows = rows_of_table();
        std::cout << rows.size() << " board rows of gpu_facts.json (" << gpu_facts_table().size()
                  << " in the table)\n";
        std::cout << "build sm_" << NINFERNO_BUILD_ARCH_LABEL << ": NVFP4 and cluster kernels "
                  << (kBuild.nvfp4_kernels ? "compiled" : "not compiled")
                  << ", block shared memory " << kBuild.block_shared_bytes << " B: "
                  << (kBuildRecorded ? "the waiver windows of this build are checked"
                                     : "no waiver window is recorded for this build")
                  << '\n';
        property_price_monotone(rows);
        property_non_fact_fields(rows);
        property_clock_invariance(rows);

        int unexpected = 0;
        for (const auto& [id, g] : groups) {
            const Known* known = nullptr;
            for (const Known& k : kKnown) {
                if (id.first == k.property && id.second == k.key) { known = &k; }
            }
            std::cout << (known ? "XFAIL " : "FAIL  ") << id.first << " [" << id.second
                      << "]: " << g.count << " case(s) on " << g.rows.size() << " row(s)";
            if (g.worst > 0) { std::cout << ", worst x" << g.worst; }
            std::cout << "\n    e.g. " << g.example << '\n';
            if (known) {
                std::cout << "    cause: " << known->cause << '\n';
                unexpected += check_window(*known, id.second, g);
            } else {
                ++unexpected;
            }
        }
        for (const Known& k : kKnown) {
            if (groups.find({k.property, k.key}) == groups.end()) {
                std::cout << "XPASS " << k.property << " [" << k.key
                          << "] no longer fails: remove it from kKnown\n";
                ++unexpected;
            }
        }
        std::cout << cases_checked << " cases checked, " << groups.size() << " failing group(s), "
                  << unexpected << " unexpected\n";
        if (unexpected != 0) { return 1; }
        std::cout << "decision properties OK\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "decision properties: " << error.what() << '\n';
        return 1;
    }
}
