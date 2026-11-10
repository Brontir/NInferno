// Property tests for the projection selector's decisions, set 1 (roadmap #620 step 6; the
// specification is Roadmap 0's report 08, Q3). Every property runs over the board rows of
// core/data/gpu_facts.json through the same substitution the selector goldens use
// (selector_facts.h board_device_class, ScopedDeviceClass), so it needs no GPU: a host without a
// device prices the register MMA by threads alone (generic_cost.cpp rmma_kept), the same on every
// substituted board.
//
//   P1  price monotone: for every priced route of a call of 17 tokens or more, a board with 1.25x
//       the SMs, or 1.25x the DRAM bandwidth, all else equal, prices it no slower (within the
//       model's 1 % resolution). The price (price_routes), not the choice.
//   P3  non-fact fields: the board's labels (names, PCI address, UUID, ordinal), the table's
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
#include "ops/generic/selector_facts.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <optional>
#include <set>
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
};

// None is known until the stands have run the test; entries are added with their cause.
constexpr Known kKnown[] = {
    {"", "", ""},
};

// ---- Findings ----------------------------------------------------------------------------------

struct Group {
    int count = 0;
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
        placed.board = generic::FactsBoard{
            placed.name.c_str(),
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

constexpr QType kFormats[] = {QType::BF16,         QType::Q4_G64_FP16,  QType::Q8_G32_FP16,
                              QType::NVFP4,        QType::MXFP4,        QType::GGML_Q4_K,
                              QType::GGML_Q6_K,    QType::GGML_IQ2_S,   QType::GGML_Q4_0,
                              QType::GGML_IQ4_XS};
constexpr Shape kShapes[]  = {{1024, 4096}, {17408, 5120}};
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
                                const auto r = generic::route_query(d, q, policy, e, rows, s.k, t,
                                                                    prefill);
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

std::size_t differing_lines(const std::string& a, const std::string& b) {
    std::istringstream la(a), lb(b);
    std::string x, y;
    std::size_t n = 0;
    while (std::getline(la, x) && std::getline(lb, y)) { n += x != y ? 1 : 0; }
    return n;
}

// ---- P1: the price is monotone in SMs and bandwidth ---------------------------------------------

// The weight as the converter lays it out, its bytes never read (route_query.cpp describe_weight).
struct DescribedWeight {
    WeightParent parent;
    WeightView view;
    Weight weight;
    DescribedWeight()                                  = default;
    DescribedWeight(const DescribedWeight&)            = delete;
    DescribedWeight& operator=(const DescribedWeight&) = delete;
};

void describe(DescribedWeight& out, QType qtype, Epilogue epilogue, std::int32_t n, std::int32_t k) {
    const std::uint64_t own = epilogue == Epilogue::SwiGlu ? 2ULL * static_cast<std::uint64_t>(n)
                                                           : static_cast<std::uint64_t>(n);
    const bool nvfp4        = qtype == QType::NVFP4;
    out.parent.data         = reinterpret_cast<const std::byte*>(std::uintptr_t{1} << 32);
    out.parent.weight_scale_divisor = nvfp4 ? 1.0F : 0.0F;
    const std::uint64_t shape[2]    = {own, static_cast<std::uint64_t>(k)};
    out.parent.geometry = weight_geometry(qtype, default_weight_layout(qtype), shape);
    out.view.shape      = {own, static_cast<std::uint64_t>(k)};
    out.view.parts      = {WeightRegion{&out.parent, 0, own * static_cast<std::uint64_t>(k)}};
    out.weight          = native_weight(out.view, nvfp4 ? 1.0F : 0.0F);
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

constexpr double kResolution = 0.01; // the model's resolution: 1 % (report 08, Q3 P1)
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
                                    row.name + " " + generic::qtype_name(q) + " " +
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
                                    std::ostringstream text;
                                    text << where << ": " << seconds << " s -> " << it->second
                                         << " s";
                                    note(property,
                                         "family " + std::to_string(route.first),
                                         row.name, text.str(), it->second / seconds);
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

// ---- P3: labels and non-fact fields move nothing ------------------------------------------------

// The same board with every field that is not a fact the selector may read changed.
std::vector<std::pair<std::string, DeviceClass>> relabelled(const Row& row, const DeviceClass& d) {
    std::vector<std::pair<std::string, DeviceClass>> out;
    {
        DeviceClass v = d;
        v.name        = "another name";
        v.facts.sku   = "another sku";
        out.emplace_back("names", std::move(v));
    }
    {
        DeviceClass v         = d;
        v.device              = 3;
        v.identity.pci_domain = 1;
        v.identity.pci_bus    = 0x41;
        v.identity.uuid.fill(0xAB);
        out.emplace_back("ordinal, PCI address, UUID", std::move(v));
    }
    {
        DeviceClass v = d;
        v.facts.tdp_w = v.facts.tdp_w * 2 + 1;
        v.facts.vram_gib = v.facts.vram_gib * 2 + 1;
        v.facts.pcie_gen = v.facts.pcie_gen == 5 ? 3 : 5;
        v.facts.segment  = v.facts.segment == Segment::Datacenter ? Segment::GeForce
                                                                  : Segment::Datacenter;
        v.global_bytes = v.global_bytes * 2 + 4096;
        out.emplace_back("tdp, vram, pcie generation, segment, reported memory", std::move(v));
    }
    {
        // The table's text fields, through the loader's own derivation of the facts.
        std::vector<GpuFactsRow> table(gpu_facts_table().begin(), gpu_facts_table().end());
        for (GpuFactsRow& r : table) {
            if (r.pci_device_id != row.source->pci_device_id) { continue; }
            r.name    = "renamed in the table";
            r.note    = "a different note";
            r.sources = {"https://example.invalid/other"};
            r.tdp_w   = r.tdp_w * 2 + 1;
            r.vram_gib = r.vram_gib * 2 + 1;
            r.pcie_gen = r.pcie_gen == 5 ? 3 : 5;
            r.segment  = r.segment == Segment::Datacenter ? Segment::GeForce : Segment::Datacenter;
        }
        DeviceClass v = d;
        v.facts       = machine_facts(driver_report_of(d), table);
        out.emplace_back("the table row's name, note, sources, tdp, vram, pcie, segment",
                         std::move(v));
    }
    return out;
}

void property_non_fact_fields(const std::vector<Row>& rows) {
    for (const Row& row : rows) {
        const DeviceClass d     = generic::board_device_class(row.board);
        const std::string stock = decisions_of(d);
        for (const auto& [what, variant] : relabelled(row, d)) {
            ++cases_checked;
            const std::string changed = decisions_of(variant);
            if (changed == stock) { continue; }
            note("P3 non-fact fields", what, row.name,
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
            note("P3 non-fact fields", "SKU twins (PCI id alone differs)",
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
        for (const auto v : kVariants) {
            ++cases_checked;
            const DeviceClass moved = generic::board_device_class(row.board, v.sm, v.memory);
            const std::string changed = decisions_of(moved);
            if (changed == stock) { continue; }
            std::ostringstream key;
            key << "SM x" << v.sm << ", memory x" << v.memory;
            note("P6 clock invariance", key.str(), row.name,
                 row.name + " (" + key.str() + "): " +
                     std::to_string(differing_lines(stock, changed)) + " decisions move, first " +
                     first_difference(stock, changed));
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
            const bool known = std::any_of(std::begin(kKnown), std::end(kKnown), [&](const Known& k) {
                return id.first == k.property && id.second == k.key;
            });
            std::cout << (known ? "XFAIL " : "FAIL  ") << id.first << " [" << id.second << "]: "
                      << g.count << " case(s) on " << g.rows.size() << " row(s)";
            if (g.worst > 0) { std::cout << ", worst x" << g.worst; }
            std::cout << "\n    e.g. " << g.example << '\n';
            if (!known) { ++unexpected; }
        }
        for (const Known& k : kKnown) {
            if (k.property[0] == '\0') { continue; }
            if (groups.find({k.property, k.key}) == groups.end()) {
                std::cout << "XPASS " << k.property << " [" << k.key
                          << "] no longer fails: remove it from kKnown\n";
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
