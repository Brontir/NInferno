// Property tests for the projection selector's decisions, set 1 (roadmap #620 step 6; the
// specification is Roadmap 0's report 08, Q3). Every property runs over the board rows of
// core/data/gpu_facts.json through the same substitution the selector goldens use
// (selector_facts.h board_device_class, ScopedDeviceClass), so it needs no GPU to run. Where a
// device is visible, the register-MMA price reads the registers and spill of the HOST GPU's
// compiled kernels (generic_cost.cpp rmma_kept -> rmma_resources, cudaFuncGetAttributes); without
// one it prices by threads alone. The same facts are read for every substituted board, so the
// properties hold on either, but the counts of P1 differ a little between the two.
//
//   P1  price monotone: for every priced route of a call of 17 tokens or more, a board with 1.25x
//       the SMs, or 1.25x the DRAM bandwidth, all else equal, prices it no slower (to 1e-9). The
//       price (price_routes), not the choice.
//   P3  fields the selector does not read: the board's labels (names, PCI address, UUID, ordinal),
//   the table's
//       sources and notes, TDP, nominal VRAM, PCIe generation, segment and the memory size the
//       driver reports move no route. Boards of the table that differ in their PCI id alone
//       (SKU twins) choose alike.
//   P6  clock invariance over every row: the clocks the driver reports (a factory overclock, a
//       vendor's lower boost, none at all) move no route of a board the table knows
//       (test_selector_golden.cpp run_clock_invariance does this for the golden boards only).
//       A row whose facts read the driver on purpose (a laptop without a published boost floor:
//       MachineFacts::driver_read) is counted and left out.
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
#include <set>
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

struct Known {
    const char* property;
    const char* key;
    const char* cause;
    int max_count; // the cases the waiver covers, at most (a host with a GPU reads other registers)
    double max_ratio; // the worst price ratio it covers
};

// The price of a route on a board with more SMs is higher by up to 1.5x for the families below
// (found by this test, roadmap #620 step 6). Cause, one for all four: price_routes
// (ops/generic/generic_cost.cpp, `cost` lambda and the I8 / staged / W4A8 tile loops) charges the
// busiest SM `q` CTAs of bytes at bw / sms, with q = ceil(CTAs / sms): bw / sms is that SM's share
// only when every SM pulls at once, so a grid whose last wave leaves SMs idle (50 -> 62 SMs for
// 100 CTAs: q stays 2, q x sms goes 100 -> 124) is overcharged when SMs are added. The DRAM floor
// n k bpw / bw does not move with the SMs. The warp-fill term moves with it too: the warps resident
// on the busy SM, min(q, occ) x threads / 32, divide the whole time (clamped below by
// kMinWarpFill), so a q that drops with more SMs lowers the fill and raises the time. Bandwidth
// x1.25 fails nowhere.
//
// Each failing case is sorted by what can explain it. "within the SM ratio": the price rose by no
// more than sms' / sms, the most q x sms (the bytes charged to the busiest SM) can grow. "beyond
// the SM ratio": the warp-fill term (or a chain term) adds to it. A waived group may not grow past
// max_count nor past max_ratio, so a change that makes the price rise with the SMs in more cases
// or by more (a bw / sqrt(sms) share, say) fails; a group that stops failing is an XPASS that
// fails the test, so the list is pruned as the model is fixed. The tolerance is 1e-9: the price
// is deterministic. (The test cannot see a route's CTA count, so it cannot tell a rise a wave
// count explains from one it does not, case by case.)
constexpr const char* kWaveCause =
    "price_routes charges the busiest SM q = ceil(CTAs/sms) CTAs of bytes at bw/sms and divides "
    "by a warp fill that depends on q (wave quantization of the memory term)";
constexpr Known kKnown[] = {
    {"P1 sms x1.25", "RegisterMma within the SM ratio", kWaveCause, 280000, 1.31},
    {"P1 sms x1.25", "RegisterMma beyond the SM ratio", kWaveCause, 750, 1.45},
    {"P1 sms x1.25", "StagedMma within the SM ratio", kWaveCause, 15200, 1.26},
    {"P1 sms x1.25", "W4a8Mma within the SM ratio", kWaveCause, 22900, 1.26},
    {"P1 sms x1.25", "W4a8Mma beyond the SM ratio", kWaveCause, 2400, 1.53},
    {"P1 sms x1.25", "I8Mma within the SM ratio", kWaveCause, 2050, 1.26},
};

// ---- Findings ----------------------------------------------------------------------------------

struct Group {
    int count    = 0;
    double worst = 0; // property-specific severity (a price ratio); 0 where none
    std::string example;
    std::set<std::string> rows;
};

std::map<std::pair<std::string, std::string>, Group> groups;
int cases_checked = 0;

void note(const std::string& property, const std::string& key, const std::string& row,
          const std::string& example, double severity = 0) {
    Group& g = groups[{property, key}];
    ++g.count;
    g.rows.insert(row);
    if (g.example.empty() || severity > g.worst) {
        g.example = example;
        g.worst   = std::max(g.worst, severity);
    }
}

// ---- The rows ----------------------------------------------------------------------------------

struct Row {
    std::string name;
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
        row.name   = r.name;
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

const char* family_name(int family) {
    constexpr const char* kNames[] = {"RegisterMma", "Scalar",    "W4a4Mma",       "TiledMma",
                                      "W4a8Mma",     "StagedMma", "Nvfp4TiledMma", "R8Mma",
                                      "GgmlMmq",     "GgmlVec",   "I8Mma"};
    return family >= 0 && family < static_cast<int>(std::size(kNames)) ? kNames[family] : "family?";
}

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
                                    if (it == prices->end()) { continue; }
                                    ++cases_checked;
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
                                    note(property, key, row.name, text.str(), it->second / seconds);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout << "P1: " << rows.size() - unpriced_rows << " priced rows, " << unpriced_rows
              << " without a roofline\n";
}

// ---- P3: fields the selector does not read move nothing
// ------------------------------------------------

// A board built as board_device_class builds it (synthetic_device_class from the row's numbers,
// the driver's values, the facts from machine_facts over `table`), with `edit` applied to the
// driver-side fields first: every derived field (mma_slow, the facts) follows from the relabelled
// inputs, none is edited on a built class.
DeviceClass rebuilt(const Row& row, const std::vector<GpuFactsRow>& table,
                    const std::function<void(DeviceClass&)>& edit) {
    const generic::FactsBoard& b = row.board;
    DeviceClass d                = synthetic_device_class(b.pci_device_id, b.major, b.minor, b.sms,
                                                          b.l2_mib * kMiB, b.memory_gib * 1024 * kMiB);
    d.sm_clock_khz               = b.sm_clock_mhz * 1000;
    d.memory_clock_khz           = b.memory_clock_mhz * 1000;
    d.memory_bus_bits            = b.bus_bits;
    edit(d);
    d.facts = machine_facts(driver_report_of(d), table);
    return d;
}

// The same board with fields the projection selector does not read changed. (Segment, TDP and the
// PCIe generation are facts a B2 decision may read, machine_facts.h; the selector reads none of
// them, and this is what the property pins.)
std::vector<std::pair<std::string, DeviceClass>> relabelled(const Row& row) {
    const std::span<const GpuFactsRow> embedded = gpu_facts_table();
    const std::vector<GpuFactsRow> stock(embedded.begin(), embedded.end());
    std::vector<std::pair<std::string, DeviceClass>> out;
    out.emplace_back("names", rebuilt(row, stock, [](DeviceClass& d) { d.name = "another name"; }));
    out.emplace_back("ordinal, PCI address, UUID", rebuilt(row, stock, [](DeviceClass& d) {
                         d.device              = 3;
                         d.identity.pci_domain = 1;
                         d.identity.pci_bus    = 0x41;
                         d.identity.uuid.fill(0xAB);
                     }));
    out.emplace_back("reported memory size", rebuilt(row, stock, [](DeviceClass& d) {
                         d.global_bytes = d.global_bytes * 2 + 4096;
                     }));
    // The table row's non-clock fields, through the loader's own derivation of the facts.
    std::vector<GpuFactsRow> table = stock;
    for (GpuFactsRow& r : table) {
        if (r.pci_device_id != row.source->pci_device_id) { continue; }
        r.name     = "renamed in the table";
        r.note     = "a different note";
        r.sources  = {"https://example.invalid/other"};
        r.tdp_w    = r.tdp_w * 2 + 1;
        r.vram_gib = r.vram_gib * 2 + 1;
        r.pcie_gen = r.pcie_gen == 5 ? 3 : 5;
        r.segment  = r.segment == Segment::Datacenter ? Segment::GeForce : Segment::Datacenter;
    }
    out.emplace_back("the table row's name, note, sources, tdp, vram, pcie, segment",
                     rebuilt(row, table, [](DeviceClass&) {}));
    return out;
}

void property_non_fact_fields(const std::vector<Row>& rows) {
    for (const Row& row : rows) {
        const DeviceClass d     = generic::board_device_class(row.board);
        const std::string stock = decisions_of(d);
        for (const auto& [what, variant] : relabelled(row)) {
            ++cases_checked;
            const std::string changed = decisions_of(variant);
            if (changed == stock) { continue; }
            note("P3 unread fields", what, row.name,
                 row.name + " (" + what + "): " + std::to_string(differing_lines(stock, changed)) +
                     " decisions move, first " + first_difference(stock, changed));
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
    std::cout << "P3: " << rows.size() << " rows relabelled, " << pairs << " SKU twin pairs\n";
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
                if (g.count > known->max_count || g.worst > known->max_ratio) {
                    std::cout << "    WAIVER EXCEEDED: at most " << known->max_count
                              << " cases and x" << known->max_ratio << '\n';
                    ++unexpected;
                }
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
