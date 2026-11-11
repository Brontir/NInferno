// The SM DRAM rate facts reader (core/sm_rate_facts.h), no GPU: every row of
// src/core/data/sm_rate_facts.json is read and equals the file field by field (the facts through
// this test's own list of names, units and members, so a fact the reader stops taking, or takes
// into another member, fails here); the list of facts the reader takes is this test's list; the
// ring2 rows are the file's; every architecture of the ISA facts table has a row (a missing one
// keeps the legacy DRAM law: a re-probe is owed); an architecture without a row is unknown; and a
// malformed table is refused.
#include "core/isa_facts.h"
#include "core/sm_rate_facts.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
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
    double SmRateRow::* member;
};

const std::vector<Expected>& expected_facts() {
    static const std::vector<Expected> v = {
        {"sm_dram_bytes_per_clock_free", "B/clk", &SmRateRow::bytes_per_clock_free},
        {"sm_dram_bytes_per_clock_placed", "B/clk", &SmRateRow::bytes_per_clock_placed},
        {"sm_dram_gbps_free", "GB/s", &SmRateRow::gbps_free},
        {"sm_dram_gbps_placed", "GB/s", &SmRateRow::gbps_placed},
        {"sm_mhz_free", "MHz", &SmRateRow::sm_mhz_free},
        {"sm_mhz_placed", "MHz", &SmRateRow::sm_mhz_placed},
        {"dram_stream_latency_ns", "ns", &SmRateRow::dram_stream_latency_ns},
        {"ring2_b_w_w1_free", "B", &SmRateRow::ring2_b_w_w1_free},
        {"ring2_b_w_w1_placed", "B", &SmRateRow::ring2_b_w_w1_placed},
        {"ring2_linear_to_w_free", "warps", &SmRateRow::ring2_linear_to_w_free},
        {"ring2_linear_to_w_placed", "warps", &SmRateRow::ring2_linear_to_w_placed},
        {"chain4_b_w_w1_free", "B", &SmRateRow::chain4_b_w_w1_free},
        {"ring4_b_w_w1_free", "B", &SmRateRow::ring4_b_w_w1_free},
    };
    return v;
}

Json table_file() {
    const std::string path = std::string(NINFERNO_SOURCE_DIR) + "/src/core/data/sm_rate_facts.json";
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path);
    return Json::parse(in);
}

void names_are_the_tests() {
    std::set<std::string> want, got;
    for (const Expected& e : expected_facts()) want.insert(e.name);
    for (const std::string& n : sm_rate_fact_names()) {
        expect(got.insert(n).second, "the reader takes " + n + " twice");
    }
    expect(got == want, "the reader takes exactly this test's facts");
}

void rows_equal_the_file(const Json& file) {
    const auto table   = sm_rate_facts_table();
    const Json& arches = file.at("arches");
    expect(table.size() == arches.size(), "one row per architecture of the file");
    expect(!table.empty(), "the table has a row");
    for (const auto& [key, j] : arches.items()) {
        const std::string where = "sm_" + key;
        const int cc            = j.at("cc").get<int>();
        const auto row          = sm_rate_row(cc / 10, cc % 10);
        expect(row.has_value(), where + ": sm_rate_row finds it");
        if (!row) continue;
        expect(row->arch == key, where + ": arch");
        expect(row->major * 10 + row->minor == cc, where + ": major.minor is its cc");
        expect(row->card == j.at("card").get<std::string>(), where + ": card");
        expect(row->probe == j.at("tool").at("probe").get<std::string>(), where + ": probe");
        expect(row->provenance == "Measured" && j.at("provenance") == "Measured",
               where + ": provenance Measured");
        expect(row->sms == j.at("sms").get<int>(), where + ": sms");
        for (const Expected& e : expected_facts()) {
            const Json& f = j.at("facts").at(e.name);
            expect(f.at("unit").get<std::string>() == e.unit, where + ": " + e.name + " unit");
            expect((*row).*e.member == f.at("value").get<double>(), where + ": " + e.name);
        }
        for (const auto& [mode, got] :
             {std::pair{"free", &row->ring2_free}, std::pair{"placed", &row->ring2_placed}}) {
            const Json& want = j.at("ring2_rows").at(mode);
            expect(got->size() == want.size(), where + ": ring2 " + mode + " rows");
            for (std::size_t i = 0; i < std::min(got->size(), want.size()); ++i) {
                expect((*got)[i].first == want[i][0].get<double>() &&
                           (*got)[i].second == want[i][1].get<double>(),
                       where + ": ring2 " + mode + " row " + std::to_string(i));
            }
        }
        std::cout << where << " (" << row->card << "): P " << row->bytes_per_clock_free << " / "
                  << row->bytes_per_clock_placed << " B/clk, L0 " << row->dram_stream_latency_ns
                  << " ns, ring2 b_w at W = 1 " << row->ring2_b_w_w1_free << " B\n";
    }
}

// Every architecture the ISA probe measured is measured here too: the price of #690 reads both.
void every_isa_architecture() {
    for (const IsaRow& r : isa_facts_table()) {
        expect(sm_rate_row(r.major, r.minor).has_value(),
               "sm_" + r.arch + " (" + r.card +
                   ") has an ISA row and no SM rate row: run tools/bench/warp_inflight/"
                   "warp_inflight.py run on it and merge");
    }
}

void unknown_architectures() {
    std::set<int> known;
    for (const SmRateRow& r : sm_rate_facts_table()) known.insert(r.major * 10 + r.minor);
    for (const auto& [major, minor] :
         std::vector<std::pair<int, int>>{{8, 0}, {9, 0}, {10, 0}, {12, 1}, {6, 1}, {0, 0}}) {
        if (known.count(major * 10 + minor) != 0) continue;
        expect(!sm_rate_row(major, minor).has_value(),
               "sm_" + std::to_string(major) + std::to_string(minor) + " has no row: unknown");
    }
}

void parse_rules(const Json& file) {
    const Json base     = file.at("arches").at("86");
    const auto table_of = [](const Json& row, const std::string& key = "86") {
        Json t           = {{"schema", 1}, {"arches", Json::object()}};
        t["arches"][key] = row;
        return t.dump();
    };
    const auto refused = [&](const std::string& text, const std::string& what) {
        bool threw = false;
        try {
            (void)parse_sm_rate_facts(text);
        } catch (const std::exception&) { threw = true; }
        expect(threw, "refused: " + what);
    };
    expect(parse_sm_rate_facts(table_of(base)).size() == 1, "the 86 row alone is read");
    {
        Json row = base;
        row["facts"].erase("dram_stream_latency_ns");
        refused(table_of(row), "a fact missing");
    }
    {
        Json row                                             = base;
        row["facts"]["sm_dram_bytes_per_clock_free"]["unit"] = "GB/s";
        refused(table_of(row), "a fact in another unit");
    }
    {
        Json row                                        = base;
        row["facts"]["dram_stream_latency_ns"]["value"] = 0;
        refused(table_of(row), "a fact that is not positive");
    }
    {
        Json row                                        = base;
        row["facts"]["dram_stream_latency_ns"]["value"] = "302";
        refused(table_of(row), "a fact that is not a number");
    }
    {
        Json row                        = base;
        row["ring2_rows"]["free"][1][0] = 1;
        refused(table_of(row), "ring rows whose W does not rise");
    }
    {
        Json row                    = base;
        row["ring2_rows"]["placed"] = Json::array();
        refused(table_of(row), "no ring rows");
    }
    refused(table_of(base, "89"), "a key that is not the row's cc");
    {
        Json t             = Json::parse(table_of(base));
        t["arches"]["86a"] = base;
        refused(t.dump(), "a cc twice");
    }
    {
        Json t      = Json::parse(table_of(base));
        t["schema"] = 2;
        refused(t.dump(), "a schema other than 1");
    }
}

} // namespace

int main() {
    try {
        const Json file = table_file();
        names_are_the_tests();
        rows_equal_the_file(file);
        every_isa_architecture();
        unknown_architectures();
        parse_rules(file);
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
