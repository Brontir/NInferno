// The SM DRAM rate facts reader (core/sm_rate_facts.h), no GPU: every row of
// src/core/data/sm_rate_facts.json is read and equals the file field by field (the facts through
// this test's own list of names, units and members, so a fact the reader stops taking, or takes
// into another member, fails here); the list of facts the reader takes is this test's list; the
// ring rows and the transfer checks are the file's, and a row is refuted exactly when one of its
// checks is outside the bounds; every architecture of the ISA facts table has a row (a missing
// one keeps the legacy DRAM law: a re-probe is owed); the architectures without a row are named
// and unknown; and a malformed table is refused.
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
        {"max_warp_bytes_in_flight", "B", &SmRateRow::max_warp_bytes_in_flight},
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

void rings_equal(const SmRateRings& got, const Json& j, const std::string& where) {
    const std::pair<const char*, const std::vector<SmRateRingRow>*> all[] = {
        {"ring2_rows free", &got.ring2_free},
        {"ring2_rows placed", &got.ring2_placed},
        {"ring4_rows free", &got.ring4_free},
        {"ring4_rows placed", &got.ring4_placed}};
    for (const auto& [name, rows] : all) {
        const std::string n(name);
        const Json& want = j.at(n.substr(0, n.find(' '))).at(n.substr(n.find(' ') + 1));
        expect(rows->size() == want.size(), where + ": " + n + " rows");
        for (std::size_t i = 0; i < std::min(rows->size(), want.size()); ++i) {
            expect((*rows)[i].w == want[i][0].get<double>() &&
                       (*rows)[i].gbps == want[i][1].get<double>() &&
                       (*rows)[i].sm_mhz == want[i][2].get<double>(),
                   where + ": " + n + " row " + std::to_string(i));
        }
    }
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
        rings_equal(row->rings, j, where);
        expect(row->warp_bytes_saturated == j.at("warp_bytes_saturated").get<bool>() &&
                   row->ring4_over_ring2_w1 == j.at("ring4_over_ring2_w1").get<double>(),
               where + ": warp saturation");
        // The rule (warp_inflight.py facts_row): where the row carries a deciding chain test
        // (`cap_test`, probe 5 on, with the card's own ISA latency), saturated is its verdict,
        // `cap` or `no cap`. Otherwise it is the ring rule (SATURATION_GAIN): the 4-load ring
        // keeps under x1.5 of the 2-load ring's b_w at W = 1, the midpoint between no gain and
        // its count's doubling.
        const auto chains = j.find("cap_test");
        if (chains != j.end() && chains->at("decides").get<bool>()) {
            const std::string verdict = chains->at("verdict").get<std::string>();
            expect(verdict == "cap" || verdict == "no cap", where + ": a deciding chain verdict");
            expect(row->warp_bytes_saturated == (verdict == "cap"),
                   where + ": saturated is the chain test's verdict");
        } else {
            expect(row->warp_bytes_saturated ==
                       (row->ring4_b_w_w1_free / row->ring2_b_w_w1_free < 1.5),
                   where + ": saturated is the ring rule over its readings");
        }
        const Json& transfer = j.at("transfer");
        expect(row->transfer_refuted == transfer.at("refuted").get<bool>(), where + ": refuted");
        const Json& runs = transfer.at("runs");
        expect(row->transfers.size() == runs.size(), where + ": transfer runs");
        bool outside = false;
        for (std::size_t i = 0; i < std::min(row->transfers.size(), runs.size()); ++i) {
            const SmRateTransfer& got = row->transfers[i];
            const Json& want          = runs[i];
            const std::string at      = where + " transfer " + got.card;
            expect(got.card == want.at("card").get<std::string>(), at + ": card");
            expect(got.source == want.at("source").get<std::string>(), at + ": source");
            expect(got.sms == want.at("sms").get<int>(), at + ": sms");
            expect(got.dram_stream_latency_ns == want.at("dram_stream_latency_ns").get<double>(),
                   at + ": L0");
            expect(got.within_bounds == want.at("within_bounds").get<bool>(), at + ": verdict");
            rings_equal(got.rings, want, at);
            outside = outside || !got.within_bounds;
        }
        expect(row->transfer_refuted == outside, where + ": refuted is the runs' verdict");
        std::cout << where << " (" << row->card << "): P " << row->bytes_per_clock_free << " / "
                  << row->bytes_per_clock_placed << " B/clk, L0 " << row->dram_stream_latency_ns
                  << " ns, b_w_max " << row->max_warp_bytes_in_flight << " B ("
                  << (row->warp_bytes_saturated ? "a cap" : "no cap") << ", ring4 / ring2 x"
                  << row->ring4_over_ring2_w1 << "), " << row->transfers.size()
                  << " transfer check(s)" << (row->transfer_refuted ? ", refuted" : "") << '\n';
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

// The architectures the probe has not measured are unknown: sm_80 (A100), sm_90 (H100), sm_100
// (B300), sm_121 (GB10), and anything else; a row is found by its own cc only. Written out, not
// filtered by the table: a row for one of these is a decision (and its boards leave the legacy
// DRAM law, test_kernel_roofline.cpp check 8 names them), so this list moves with it.
void unknown_architectures() {
    for (const auto& [major, minor] : std::vector<std::pair<int, int>>{
             {8, 0}, {9, 0}, {10, 0}, {10, 3}, {12, 1}, {8, 7}, {6, 1}, {0, 0}, {-1, 7}}) {
        expect(!sm_rate_row(major, minor).has_value(),
               "sm_" + std::to_string(major) + std::to_string(minor) + " has no row: unknown");
    }
    std::set<int> known;
    for (const SmRateRow& r : sm_rate_facts_table()) known.insert(r.major * 10 + r.minor);
    expect(known == std::set<int>{70, 75, 86, 89, 120},
           "the architectures with a row are sm_70, sm_75, sm_86, sm_89, sm_120");
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
        row["ring4_rows"]["placed"] = Json::array();
        refused(table_of(row), "no ring rows");
    }
    {
        Json row = base;
        row["ring2_rows"]["free"][0].erase(2);
        refused(table_of(row), "a ring row without its clock");
    }
    {
        // The verdict must be its runs': a refuted flag over runs within the bounds is refused,
        // and a run outside them must refute.
        Json row                   = file.at("arches").at("89");
        const auto rows89          = parse_sm_rate_facts(table_of(row, "89"));
        row["transfer"]["refuted"] = true;
        expect(rows89.size() == 1 && !rows89[0].transfer_refuted && !rows89[0].transfers.empty(),
               "the sm_89 row has transfer checks, none refuting");
        refused(table_of(row, "89"), "refuted over runs within the bounds");
        row["transfer"]["refuted"]                  = false;
        row["transfer"]["runs"][0]["within_bounds"] = false;
        refused(table_of(row, "89"), "a run outside the bounds that does not refute");
        row["transfer"]["refuted"] = true;
        const auto refuted         = parse_sm_rate_facts(table_of(row, "89"));
        expect(refuted.size() == 1 && refuted[0].transfer_refuted, "a refuting run is read");
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
