// The ISA facts reader (core/isa_facts.h), no GPU: every row of src/core/data/isa_facts.json is
// read and equals the file field by field (the facts through this test's own list of names and
// members, so a fact the reader stops taking, or takes into another member, fails here); the list
// of facts the reader takes is this test's list; the tensor rate of every row is its MMA's flops
// over its cycles; an architecture without a row is unknown; a fact a row lacks is unknown, never
// a number; and a malformed table is refused.
#include "core/isa_facts.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <stdexcept>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#ifndef NINFERNO_SOURCE_DIR
#    error "NINFERNO_SOURCE_DIR must name the source tree"
#endif

namespace {

using namespace ninferno;
using Json = nlohmann::json;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

// The facts a row holds, by the table's name and unit, written out here apart from the reader.
struct Expected {
    std::string name;
    std::string unit;
    std::function<const std::optional<double>&(const IsaRow&)> member;
};

std::vector<Expected> expected_facts() {
    std::vector<Expected> v;
    const auto one = [&](const char* name, const char* unit, std::optional<double> IsaRow::* m) {
        v.push_back(
            {name, unit, [m](const IsaRow& r) -> const std::optional<double>& { return r.*m; }});
    };
    const auto warps = [&](const char* name, const char* unit, IsaByWarps IsaRow::* m) {
        v.push_back({std::string(name) + "_w1", unit,
                     [m](const IsaRow& r) -> const std::optional<double>& { return (r.*m).w1; }});
        v.push_back({std::string(name) + "_w2", unit,
                     [m](const IsaRow& r) -> const std::optional<double>& { return (r.*m).w2; }});
        v.push_back({std::string(name) + "_w4", unit,
                     [m](const IsaRow& r) -> const std::optional<double>& { return (r.*m).w4; }});
        v.push_back({std::string(name) + "_w8", unit,
                     [m](const IsaRow& r) -> const std::optional<double>& { return (r.*m).w8; }});
    };
    one("tensor_flops_per_sm_clock", "flop", &IsaRow::tensor_flops_per_sm_clock);
    one("mma_cycles_per_sm", "clk", &IsaRow::mma_cycles_per_sm);
    one("mma_cycles_per_partition", "clk", &IsaRow::mma_cycles_per_partition);
    one("mma_dependent_latency", "clk", &IsaRow::mma_dependent_latency);
    one("int32_cycles_per_warp_instruction", "clk", &IsaRow::int32_cycles_per_warp_instruction);
    one("fp32_cycles_per_warp_instruction", "clk", &IsaRow::fp32_cycles_per_warp_instruction);
    warps("l1_latency", "clk", &IsaRow::l1_latency);
    warps("l2_latency", "clk", &IsaRow::l2_latency);
    one("shared_latency", "clk", &IsaRow::shared_latency);
    warps("dram_latency", "clk", &IsaRow::dram_latency);
    warps("dram_tlbwarm_latency", "clk", &IsaRow::dram_tlbwarm_latency);
    warps("dram_tlbwarm_latency_ns", "ns", &IsaRow::dram_tlbwarm_latency_ns);
    return v;
}

Json table_file() {
    const std::string path = std::string(NINFERNO_SOURCE_DIR) + "/src/core/data/isa_facts.json";
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path);
    return Json::parse(in);
}

std::string text_of(const Json& j) { return j.dump(); }

// The reader takes exactly the facts of expected_facts(), in the table's spelling.
void names_are_the_tests() {
    std::set<std::string> want, got;
    for (const Expected& e : expected_facts()) want.insert(e.name);
    for (const std::string& n : isa_fact_names()) {
        expect(got.insert(n).second, "the reader takes " + n + " twice");
    }
    for (const std::string& n : want) expect(got.count(n) == 1, "the reader takes " + n);
    for (const std::string& n : got) {
        expect(want.count(n) == 1, "the reader takes " + n + ", which this test does not check");
    }
}

// Every row of the file is read, and equals it field by field.
void rows_equal_the_file(const Json& file) {
    const auto table   = isa_facts_table();
    const Json& arches = file.at("arches");
    expect(table.size() == arches.size(), "one row per architecture of the file (" +
                                              std::to_string(table.size()) + " of " +
                                              std::to_string(arches.size()) + ")");
    const std::vector<Expected> facts = expected_facts();
    std::set<std::string> seen_somewhere;
    for (const auto& [key, j] : arches.items()) {
        const std::string where = "sm_" + key;
        const int cc            = j.at("cc").get<int>();
        const auto row          = isa_row(cc / 10, cc % 10);
        expect(row.has_value(), where + ": isa_row(" + std::to_string(cc / 10) + ", " +
                                    std::to_string(cc % 10) + ") finds it");
        if (!row) continue;
        expect(row->arch == key, where + ": arch " + row->arch);
        expect(row->major * 10 + row->minor == cc, where + ": major.minor is its cc");
        expect(row->card == j.at("card").get<std::string>(), where + ": card");
        expect(row->probe == j.at("tool").at("probe").get<std::string>(), where + ": probe");
        expect(row->provenance == j.at("provenance").get<std::string>(), where + ": provenance");
        expect(row->mma_shape == j.at("mma_shape").get<std::string>(), where + ": mma_shape");
        expect(row->mma_flops == j.at("mma_flops").get<int>(), where + ": mma_flops");
        expect(row->sms == j.at("sms").get<int>(), where + ": sms");
        expect(j.contains("sass_per_mma") ? row->sass_per_mma == j.at("sass_per_mma").get<int>()
                                          : !row->sass_per_mma.has_value(),
               where + ": sass_per_mma, unknown when the row has none");
        const Json& jf = j.at("facts");
        for (const Expected& e : facts) {
            const std::optional<double>& got = e.member(*row);
            const auto it                    = jf.find(e.name);
            if (it == jf.end()) {
                expect(!got.has_value(), where + ": " + e.name + " is not in the row: unknown");
                continue;
            }
            seen_somewhere.insert(e.name);
            expect(it->at("unit").get<std::string>() == e.unit,
                   where + ": " + e.name + " in " + e.unit);
            const double want = it->at("value").get<double>();
            expect(got.has_value() && *got == want,
                   where + ": " + e.name + " = " + std::to_string(want) + ", read " +
                       (got ? std::to_string(*got) : std::string("nothing")));
        }
    }
    // A fact renamed in the table would read as unknown everywhere: each is in some row.
    for (const Expected& e : facts) {
        expect(seen_somewhere.count(e.name) == 1, e.name + " is in no row of the table");
    }
}

// The tensor rate is the row's MMA flops over its cycles per SM (as the probe rounds it), and the
// architectures of the register-MMA law (sm_86, sm_89, sm_120a) carry it.
void tensor_rate(const Json& file) {
    for (const IsaRow& r : isa_facts_table()) {
        const std::string where = "sm_" + r.arch;
        if (!r.tensor_flops_per_sm_clock || !r.mma_cycles_per_sm) continue;
        const double from_cycles = r.mma_flops / *r.mma_cycles_per_sm;
        expect(std::fabs(*r.tensor_flops_per_sm_clock - from_cycles) <= 5e-4 + 1e-9 * from_cycles,
               where + ": tensor_flops_per_sm_clock " +
                   std::to_string(*r.tensor_flops_per_sm_clock) + " is mma_flops / " +
                   "mma_cycles_per_sm = " + std::to_string(from_cycles));
    }
    for (const char* key : {"86", "89", "120a"}) {
        const Json& j     = file.at("arches").at(key);
        const int cc      = j.at("cc").get<int>();
        const auto row    = isa_row(cc / 10, cc % 10);
        const double want = j.at("facts").at("tensor_flops_per_sm_clock").at("value").get<double>();
        expect(row && row->tensor_flops_per_sm_clock == want,
               std::string("sm_") + key + ": tensor_flops_per_sm_clock " + std::to_string(want));
        if (row && row->tensor_flops_per_sm_clock) {
            std::cout << "sm_" << key << " (" << row->card
                      << "): tensor_flops_per_sm_clock = " << *row->tensor_flops_per_sm_clock
                      << '\n';
        }
    }
}

// An architecture without a row is unknown; a row speaks for its own architecture only.
void unknown_architectures() {
    std::set<int> known;
    for (const IsaRow& r : isa_facts_table()) known.insert(r.major * 10 + r.minor);
    for (const auto& [major, minor] : std::vector<std::pair<int, int>>{
             {8, 0}, {9, 0}, {10, 0}, {12, 1}, {6, 1}, {0, 0}, {-1, 7}}) {
        const int cc = major * 10 + minor;
        if (known.count(cc) != 0) continue;
        expect(!isa_row(major, minor).has_value(),
               "sm_" + std::to_string(major) + std::to_string(minor) + " has no row: unknown");
    }
}

// A crafted one-row table: what is missing is unknown, what is malformed is refused.
void parse_rules(const Json& file) {
    const Json base     = file.at("arches").at("86");
    const auto table_of = [](const Json& row, const std::string& key = "86") {
        Json t           = {{"schema", 1}, {"arches", Json::object()}};
        t["arches"][key] = row;
        return t;
    };
    const auto refused = [&](const Json& t, const std::string& what) {
        bool threw = false;
        try {
            (void)parse_isa_facts(text_of(t));
        } catch (const std::exception&) { threw = true; }
        expect(threw, "refused: " + what);
    };

    {
        Json row = base;
        row["facts"].erase("l1_latency_w4");
        row["facts"].erase("tensor_flops_per_sm_clock");
        const auto rows = parse_isa_facts(text_of(table_of(row)));
        expect(rows.size() == 1 && !rows[0].l1_latency.w4 && !rows[0].tensor_flops_per_sm_clock,
               "a fact the row lacks is unknown");
        expect(rows.size() == 1 && rows[0].l1_latency.w1 && rows[0].mma_cycles_per_sm,
               "the facts it has are read");
    }
    {
        Json row        = base;
        row["facts"]    = Json::object();
        const auto rows = parse_isa_facts(text_of(table_of(row)));
        bool none       = rows.size() == 1;
        for (const Expected& e : expected_facts()) none = none && !e.member(rows[0]).has_value();
        expect(none, "a row without facts has every fact unknown");
    }
    {
        Json row            = base;
        row["sass_per_mma"] = 4;
        const auto rows     = parse_isa_facts(text_of(table_of(row)));
        expect(rows.size() == 1 && rows[0].sass_per_mma == 4, "sass_per_mma is read when given");
    }
    {
        Json row                              = base;
        row["facts"]["l2_latency_w2"]["unit"] = "ns";
        refused(table_of(row), "a fact in another unit than its name's");
    }
    {
        Json row                                = base;
        row["facts"]["shared_latency"]["value"] = 0;
        refused(table_of(row), "a fact that is not positive");
    }
    {
        Json row                                = base;
        row["facts"]["shared_latency"]["value"] = "23";
        refused(table_of(row), "a fact that is not a number");
    }
    refused(table_of(base, "89"), "a key that is not the row's cc");
    {
        Json t             = table_of(base);
        t["arches"]["86a"] = base;
        refused(t, "a cc twice");
    }
    {
        Json t      = table_of(base);
        t["schema"] = 2;
        refused(t, "a schema other than 1");
    }
    {
        Json row = base;
        row.erase("card");
        refused(table_of(row), "a row without its card");
    }
}

} // namespace

int main() {
    std::size_t rows = 0;
    try {
        const Json file = table_file();
        names_are_the_tests();
        rows_equal_the_file(file);
        tensor_rate(file);
        unknown_architectures();
        parse_rules(file);
        rows = isa_facts_table().size();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        ++failures;
    }
    std::cout << (failures == 0 ? "PASS" : "FAILED") << " (" << rows << " rows, "
              << isa_fact_names().size() << " facts a row)\n";
    return failures == 0 ? 0 : 1;
}
