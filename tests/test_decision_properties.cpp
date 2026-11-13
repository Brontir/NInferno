// Property tests for the projection selector's decisions, set 1 (roadmap #620 step 6; the
// specification is Roadmap 0's report 08, Q3). Every property runs over the board rows of
// core/data/gpu_facts.json through the same substitution the selector goldens use
// (selector_facts.h board_device_class, ScopedDeviceClass), so it needs no GPU to run. The
// register-MMA price reads the registers and spill of the compiled kernels from the committed
// kernel-facts snapshot of the SASS (generic_cost.cpp rmma_kept -> rmma_resources), not from the
// running GPU: the counts below are the same with a GPU visible and without one. They are not the
// same in every build: what P1 prices is the routes the build holds (a substituted board is priced
// with the routes of the build that runs the test: the cluster schedules and the NVFP4 kernels of
// a full sm_120a build, the shared memory a block may opt into).
//
//   P1  price monotone: for every priced route of a call of 17 tokens or more, a board with 1.25x
//       the SMs, or 1.25x the DRAM bandwidth, all else equal, prices it no slower (to 1e-9). The
//       price (price_routes), not the choice. Three kinds of SM-scale case are told apart by a
//       predicate and printed with their counts, not failed (explain_rmma_rise for the last two):
//         - a board whose architecture has no SM rate row (core/sm_rate_facts.h: sm_80, sm_90,
//           sm_100 today), or a row another card of it refutes (sm_120a), keeps the legacy DRAM
//           law (kernel_roofline.h legacy_dram), which bills idle SMs (issue #690) until a probe
//           row that holds exists for the architecture: its SM scale is not compared, its rises
//           are counted and printed;
//         - a register-MMA route of the heavy form whose rise is the shared activation read alone:
//           the CTAs one SM runs at once read a token tile once for all of them, and with more
//           SMs the busiest one holds fewer of them, so that read is spread over fewer warps (the
//           price without that term does not rise);
//         - a register-MMA route of the heavy form whose larger board runs another compiled
//           instance of the schedule (the launcher's latency and wide variants are chosen by the
//           board's DRAM bytes per SM and cycle): two kernels, not one priced twice.
//       And on both scales, ggml's matrix kernel on a generation whose launch law picks its column
//       width (Turing, DECISIONS 78: core/mmq_columns.h pick) where the larger board picks another
//       width: two launches, not one priced twice. The price is the A8 fit's tile count at the
//       law's width, not the law's own time, so it can rise there (#786: up to x1.04 on the
//       Tesla T4). OPEN: pricing that launch by the law itself was tried and moved the RTX 2080
//       Ti's A8 regret from +0.3 % to +14 .. +31 %, against the W4A8 tiles still priced at 0.7 of
//       their time; counted and printed, not failed.
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
// Failures are grouped by (property, key); every group fails the test and prints its count, its
// rows and its worst case. (The waivers #639 recorded for the idle-SM charges of the price are
// gone with the DRAM law of issue #690.)

#include "core/device_class.h"
#include "core/format_facts.h"
#include "core/generation.h"
#include "core/machine_facts.h"
#include "core/mmq_columns.h"
#include "core/units.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "ops/common/arch_limits.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_ggml_internal.h"
#include "ops/generic/generic_select_internal.h"
#include "ops/generic/kernel_roofline.h"
#include "ops/generic/route_query.h"
#include "ops/generic/route_table.h"
#include "ops/generic/selector_facts.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <iterator>
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
                                r.l2_mib,
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
// Every width of the decode, verify and prefill bands the selector tells apart: both sides of each
// width bound it switches at (held below), and widths inside the bands.
// Besides the named bounds: 3 / 4 (rules_r8's IQ2-class formats from 4), 64 / 65 (heuristic_w4a8's
// 64x64 tile up to 64 and the prefill narrow band, then a choice by the SM count), 8 / 9
// (kMmqVerifyMinTokens, format_rules.h), 255 / 256 (kVoltaTileMaxTokens, local to
// generic_select.cpp).
constexpr std::int32_t kTokens[] = {1,  2,  3,  4,   5,   8,   9,   12,  16,  17,  32,
                                    33, 64, 65, 127, 128, 255, 256, 512, 513, 1024};
// The widths the price is checked at: from the first the cost model ranks (17), both sides of
// each bound there.
constexpr std::int32_t kPriceTokens[] = {17, 32, 33, 64, 65, 127, 128, 255, 256, 512, 513, 1024};

template <std::size_t N>
constexpr bool has(const std::int32_t (&grid)[N], std::int32_t t) {
    return std::find(std::begin(grid), std::end(grid), t) != std::end(grid);
}

// The selector's width bounds (#719): a "from B on" bound needs B - 1 and B in the grid, an "up to
// B" bound B and B + 1, so P3 / P6 compare decisions on both sides of each.
template <std::size_t N>
constexpr bool from_bound(const std::int32_t (&grid)[N], std::int32_t bound) {
    return has(grid, bound - 1) && has(grid, bound);
}

template <std::size_t N>
constexpr bool upto_bound(const std::int32_t (&grid)[N], std::int32_t bound) {
    return has(grid, bound) && has(grid, bound + 1);
}

static_assert(upto_bound(kTokens, generic::kR8MaxTokens) &&
                  upto_bound(kTokens, generic::kR8VerifyTokens) &&
                  upto_bound(kTokens, generic::kMmqMaxTokens) &&
                  upto_bound(kTokens, generic::kVoltaHmmaMaxTokens) && upto_bound(kTokens, 64) &&
                  from_bound(kTokens, 4) && from_bound(kTokens, generic::kMmqMinTokens) &&
                  upto_bound(kTokens, generic::kGgmlVecMaxTokens) &&
                  upto_bound(kTokens, generic::kGgmlMmqMaxTokens) &&
                  upto_bound(kTokens, generic::kDenseTuneMaxTokens) &&
                  upto_bound(kTokens, generic::kPdlPrefetchMaxTokens) &&
                  from_bound(kTokens, generic::kPrefillBandMinTokens) &&
                  from_bound(kTokens, generic::kRmmaMidMinTokens) &&
                  from_bound(kTokens, generic::kTiledSpeedMinTokens) &&
                  from_bound(kTokens, generic::kW4a8WideMinTokens) &&
                  from_bound(kTokens, generic::kStagedSpeedMinTokens) &&
                  upto_bound(kTokens, generic::kStreamMaxTokens),
              "a width bound of the selector lacks a width on one side in the grid");
static_assert(from_bound(kPriceTokens, generic::kRmmaMidMinTokens) &&
                  from_bound(kPriceTokens, generic::kTiledSpeedMinTokens) &&
                  from_bound(kPriceTokens, generic::kW4a8WideMinTokens) &&
                  from_bound(kPriceTokens, generic::kStagedSpeedMinTokens) &&
                  upto_bound(kPriceTokens, generic::kStreamMaxTokens),
              "a width bound the price is ranked across lacks a width on one side");

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
                            // The prefill profile from its first width: the selector's prefill
                            // band switches inside 17..255 too (#719).
                            if (prefill && t < generic::kPrefillBandMinTokens) { continue; }
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

// The cases P1 checked on the SM scale, per route family.
std::map<std::string, int> sms_cases_by_family;
int sms_cases_priced = 0, routes_skipped_unpriced = 0;
int sms_cases_fell = 0, sms_cases_equal = 0; // the price fell / stayed (to 1e-9) with the SMs
// The SM-scale cases the predicates of the header tell apart: on a board of the legacy DRAM law
// (not compared), another compiled instance on the larger board and the shared activation reads
// (compared, the rise explained), with the worst such rises.
int sms_cases_legacy = 0, sms_cases_shared_read = 0, sms_cases_instance = 0;
double shared_read_worst = 1, instance_worst = 1;
int mmq_width_cases    = 0; // either scale: another MMQ width on the larger board (header)
double mmq_width_worst = 1;
// The legacy boards' SM-scale cases are counted, not compared: how many rose, and the worst.
int sms_cases_legacy_rose = 0;
double legacy_worst       = 1;
std::vector<std::string> legacy_rows;

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

namespace kr = generic::kernel_roofline;

// The roofline terms of a register-MMA route on `d` from the inputs price_routes passes (the
// route's tile, the weight's bytes, the compiled facts of the heavy decoders); nullopt where their
// total with the A16 scale differs from `selector`, the price P1 compared (to 1e-12), so that a
// predicate over the terms is never about another price.
struct RmmaPriced {
    kr::RegisterMmaTerms terms;
    kr::RegisterMmaKernel kernel; // with the compiled facts the price read
};

std::optional<RmmaPriced> rmma_terms(const DeviceClass& d, const generic::Problem& problem,
                                     const Weight& weight, std::int32_t tokens, int schedule,
                                     double selector) {
    const ScopedDeviceClass scoped(d);
    const generic::Route r{generic::Family::RegisterMma, static_cast<std::uint8_t>(schedule)};
    const auto tile = generic::route_tile(r, problem.epilogue, tokens);
    if (!tile) { return std::nullopt; }
    const auto s = generic::rmma_schedule_info(schedule);
    kr::RegisterMmaKernel kernel;
    kernel.rows    = tile->out_rows;
    kernel.cols    = tile->tokens;
    kernel.cluster = tile->k_split;
    kernel.threads = 32.0 * s.warps_m * s.warps_k;
    kernel.warps_k = s.warps_k;
    kernel.m_tiles = s.m_tiles;
    kernel.k_step  = static_cast<double>(generic::rmma_step_values(problem.qtype, problem.k));
    kernel.heavy   = has_subblock_scales(problem.qtype) && rmma_terms_measured(d.major, d.minor);
    kernel.shared_bytes = kr::register_mma_static_shared_bytes(
        generic::rmma_geometry_decoder(problem.qtype), s.warps_m, s.warps_k, s.m_tiles, s.n_tiles,
        s.cluster, problem.epilogue == Epilogue::SwiGlu, d.major * 10 + d.minor);
    if (kernel.heavy) {
        try {
            if (const auto kept = generic::rmma_resources(schedule, weight, problem.epilogue,
                                                          tokens, problem.n)) {
                kernel.registers    = kept->registers;
                kernel.local_bytes  = static_cast<double>(kept->local_bytes);
                kernel.shared_bytes = static_cast<double>(kept->static_smem);
            }
        } catch (const std::exception&) {}
    }
    const kr::Call call{static_cast<double>(problem.n), static_cast<double>(problem.k),
                        static_cast<double>(tokens),
                        problem.epilogue == Epilogue::SwiGlu ? 2.0 : 1.0,
                        generic::bytes_per_weight(weight, problem)};
    const auto terms = kr::price_register_mma(kr::chip_facts_of(d),
                                              kr::calibration_of(d.major, d.minor), call, kernel);
    const double a16 = fitted_route_scales(d.major, d.minor).value_or(FittedRouteScales{}).a16_cost;
    if (std::abs(terms.total * a16 - selector) > selector * 1e-12) { return std::nullopt; }
    return RmmaPriced{terms, kernel};
}

// What explains a register-MMA route's rise on the board with more SMs (the header's predicates),
// on the heavy form only: None, or
//  - Instance: the launcher runs another compiled instance of the schedule on the larger board
//    (its registers or spill differ: the latency and wide variants are chosen by the board's DRAM
//    bytes per SM and cycle, generic_gemm_launch.cuh), so the two prices are of two kernels;
//  - SharedRead: the same instance, the shared activation read (under the fill) larger on the
//    board with more SMs, and the price without it no higher there.
enum class Explained { None, Instance, SharedRead };

Explained explain_rmma_rise(const DeviceClass& base, const DeviceClass& more,
                            const generic::Problem& problem, const Weight& weight,
                            std::int32_t tokens, int schedule, double stock, double scaled) {
    if (!has_subblock_scales(problem.qtype) || !rmma_terms_measured(base.major, base.minor)) {
        return Explained::None; // the plain form
    }
    const auto a = rmma_terms(base, problem, weight, tokens, schedule, stock);
    const auto b = rmma_terms(more, problem, weight, tokens, schedule, scaled);
    if (!a || !b || a->terms.legacy || b->terms.legacy || a->terms.total <= 0) {
        return Explained::None;
    }
    if (a->kernel.registers != b->kernel.registers ||
        a->kernel.local_bytes != b->kernel.local_bytes) {
        return Explained::Instance;
    }
    const double read_a = a->terms.l2_activation / a->terms.grid.fill;
    const double read_b = b->terms.l2_activation / b->terms.grid.fill;
    // Both totals carry the same A16 scale: compare the roofline's own totals.
    return read_b > read_a &&
                   b->terms.total - read_b <= (a->terms.total - read_a) * (1.0 + kResolution)
               ? Explained::SharedRead
               : Explained::None;
}

// The column width ggml's matrix kernel takes for the call on `d` where the generation's launch law
// picks it, as a8_candidates reads it (the slice's columns, the widest tile the device's shared
// memory holds, the board's reference clock and DRAM rate); 0 where no law picks it.
int mmq_law_width(const DeviceClass& d, const generic::Problem& p, std::int32_t tokens) {
    const bool format = mmq_columns::work_tiles_format(p.qtype);
    if (mmq_columns::launch_law(d.major, d.minor, format) == nullptr) return 0;
    const int columns = std::min(tokens, generic::kMmqPrefillSlice);
    const mmq_columns::LaunchFacts facts{p.n, p.k, d.sms, d.facts.reference_clock_hz(),
                                         d.facts.dram_bytes_per_s()};
    return mmq_columns::pick(columns, d.major, d.minor, true,
                             mmq_columns::widest_tile(p.qtype, d.major, d.minor), facts);
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
        // The header's first predicate: the legacy DRAM law bills idle SMs (#690).
        const bool legacy = kr::legacy_dram(kr::chip_facts_of(base));
        if (legacy) { legacy_rows.push_back(row.label); }
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
                                    if (prices == &sms && legacy) {
                                        ++sms_cases_legacy;
                                        if (it->second > seconds * (1.0 + kResolution)) {
                                            ++sms_cases_legacy_rose;
                                            legacy_worst =
                                                std::max(legacy_worst, it->second / seconds);
                                        }
                                        continue;
                                    }
                                    ++cases_checked;
                                    if (prices == &sms) {
                                        ++sms_cases_priced;
                                        ++sms_cases_by_family[family_name(route.first)];
                                        if (it->second < seconds * (1.0 - kResolution)) {
                                            ++sms_cases_fell;
                                        } else if (it->second <= seconds * (1.0 + kResolution)) {
                                            ++sms_cases_equal;
                                        }
                                    }
                                    if (it->second <= seconds * (1.0 + kResolution)) { continue; }
                                    const Explained why =
                                        prices == &sms && r.family == generic::Family::RegisterMma
                                            ? explain_rmma_rise(base, more_sms, problem, w.weight,
                                                                t, route.second, seconds,
                                                                it->second)
                                            : Explained::None;
                                    if (why != Explained::None) {
                                        const bool read = why == Explained::SharedRead;
                                        ++(read ? sms_cases_shared_read : sms_cases_instance);
                                        double& worst = read ? shared_read_worst : instance_worst;
                                        worst         = std::max(worst, it->second / seconds);
                                        continue;
                                    }
                                    if (r.family == generic::Family::GgmlMmq) {
                                        const DeviceClass& larger =
                                            prices == &sms ? more_sms : more_bw;
                                        const int at = mmq_law_width(base, problem, t);
                                        if (at != 0 && at != mmq_law_width(larger, problem, t)) {
                                            ++mmq_width_cases;
                                            mmq_width_worst =
                                                std::max(mmq_width_worst, it->second / seconds);
                                            continue;
                                        }
                                    }
                                    std::string key = family_name(route.first);
                                    if (prices == &sms) {
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
              << ", of them " << sms_cases_shared_read
              << " by the shared activation read alone (worst x" << shared_read_worst << ") and "
              << sms_cases_instance << " by another compiled instance on the larger board (worst x"
              << instance_worst << ")\n";
    std::cout << "    either scale: " << mmq_width_cases
              << " ggml_mmq rises where the larger board's launch law picks another width (worst x"
              << mmq_width_worst << "): counted, not failed\n";
    std::cout << "    SM scale not compared on " << legacy_rows.size()
              << " rows of the legacy DRAM law (no SM rate row, or one another card refutes; "
                 "#690): "
              << sms_cases_legacy << " cases, the price rose in " << sms_cases_legacy_rose
              << " (worst x" << legacy_worst << "): counted, not failed";
    for (const std::string& name : legacy_rows) { std::cout << "\n    legacy DRAM law: " << name; }
    std::cout << '\n';
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
// (the per-class gates select_route asks, in generic_select.cpp, are computed once per class key,
// and the key does not
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

} // namespace

int main() {
    try {
        const std::vector<Row> rows = rows_of_table();
        std::cout << rows.size() << " board rows of gpu_facts.json (" << gpu_facts_table().size()
                  << " in the table)\n";
        std::cout << "build sm_" << NINFERNO_BUILD_ARCH_LABEL << ": NVFP4 and cluster kernels "
#ifdef NINFERNO_WITHOUT_NVFP4
                  << "not compiled"
#else
                  << "compiled"
#endif
                  << ", block shared memory " << ops::kMaxSharedBytesPerBlock << " B\n";
        property_price_monotone(rows);
        property_non_fact_fields(rows);
        property_clock_invariance(rows);

        for (const auto& [id, g] : groups) {
            std::cout << "FAIL  " << id.first << " [" << id.second << "]: " << g.count
                      << " case(s) on " << g.rows.size() << " row(s)";
            if (g.worst > 0) { std::cout << ", worst x" << g.worst; }
            std::cout << "\n    e.g. " << g.example << '\n';
            for (const auto& [row, n] : g.rows) {
                std::cout << "    " << row << ": " << n << " case(s), worst x"
                          << g.row_worst.at(row) << '\n';
            }
        }
        std::cout << cases_checked << " cases checked, " << groups.size() << " failing group(s)\n";
        if (!groups.empty()) { return 1; }
        std::cout << "decision properties OK\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "decision properties: " << error.what() << '\n';
        return 1;
    }
}
