#include "core/sm_rate_facts.h"

#include "core/embedded_sm_rate_facts.h"

#include <nlohmann/json.hpp>

#include <climits>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace ninferno {

namespace {

using Json = nlohmann::json;

// A fact the reader takes: its name in the table, its unit there, and the member it fills.
struct Field {
    std::string name;
    std::string_view unit;
    double SmRateRow::* member;
};

const std::vector<Field>& fields() {
    static const std::vector<Field> all = {
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
    return all;
}

// An integer field: a JSON integer in int range, never a fraction or a string.
int positive_int(const Json& j, const char* what) {
    const Json& v = j.at(what);
    if (!v.is_number_integer()) throw std::runtime_error(std::string(what) + " is not an integer");
    const long long x = v.get<long long>();
    if (x <= 0 || x > INT_MAX) throw std::runtime_error(std::string(what) + " is not positive");
    return static_cast<int>(x);
}

std::string text(const Json& j, const char* what) {
    std::string s = j.at(what).get<std::string>();
    if (s.empty()) throw std::runtime_error(std::string(what) + " is empty");
    return s;
}

double positive(const Json& v, const std::string& where) {
    if (!v.is_number()) throw std::runtime_error(where + " is not a number");
    const double x = v.get<double>();
    if (!std::isfinite(x) || x <= 0) throw std::runtime_error(where + " is not positive");
    return x;
}

std::vector<std::pair<double, double>> ring_rows(const Json& j, const char* mode) {
    std::vector<std::pair<double, double>> out;
    const std::string where = std::string("ring2_rows ") + mode;
    for (const Json& r : j.at(mode)) {
        if (!r.is_array() || r.size() != 2) throw std::runtime_error(where + ": not a (W, GB/s)");
        const double w = positive(r.at(0), where + " W");
        if (!out.empty() && w <= out.back().first) {
            throw std::runtime_error(where + ": W does not rise");
        }
        out.emplace_back(w, positive(r.at(1), where + " GB/s"));
    }
    if (out.empty()) throw std::runtime_error(where + " is empty");
    return out;
}

SmRateRow parse_row(const std::string& key, const Json& j) {
    SmRateRow r;
    r.arch       = key;
    const int cc = positive_int(j, "cc");
    r.major      = cc / 10;
    r.minor      = cc % 10;
    // The key is the cc, or the cc and an arch-set suffix ("120a").
    const std::string digits = std::to_string(cc);
    if (key != digits && key != digits + "a" && key != digits + "f") {
        throw std::runtime_error("the key is not its cc " + digits);
    }
    r.card            = text(j, "card");
    r.probe           = text(j.at("tool"), "probe");
    r.provenance      = text(j, "provenance");
    r.sms             = positive_int(j, "sms");
    const Json& facts = j.at("facts");
    for (const Field& f : fields()) {
        const std::string where = "fact " + f.name;
        const auto it           = facts.find(f.name);
        if (it == facts.end()) throw std::runtime_error(where + " is missing");
        const std::string unit = it->at("unit").get<std::string>();
        if (unit != f.unit) {
            throw std::runtime_error(where + " is in " + unit + ", not " + std::string(f.unit));
        }
        r.*f.member = positive(it->at("value"), where);
    }
    const Json& rings = j.at("ring2_rows");
    r.ring2_free      = ring_rows(rings, "free");
    r.ring2_placed    = ring_rows(rings, "placed");
    return r;
}

} // namespace

std::vector<SmRateRow> parse_sm_rate_facts(std::string_view json_text) {
    const Json root    = Json::parse(json_text);
    const Json& schema = root.at("schema");
    if (!schema.is_number_integer() || schema.get<long long>() != 1) {
        throw std::runtime_error("sm rate facts: schema is not 1");
    }
    std::vector<SmRateRow> rows;
    std::unordered_set<int> seen;
    for (const auto& [key, j] : root.at("arches").items()) {
        try {
            rows.push_back(parse_row(key, j));
        } catch (const std::exception& e) {
            throw std::runtime_error("sm rate facts: sm_" + key + ": " + e.what());
        }
        if (!seen.insert(rows.back().major * 10 + rows.back().minor).second) {
            throw std::runtime_error("sm rate facts: sm_" + key + ": its cc appears twice");
        }
    }
    return rows;
}

std::span<const SmRateRow> sm_rate_facts_table() {
    static const std::vector<SmRateRow> table = parse_sm_rate_facts(embedded::kSmRateFactsJson);
    return table;
}

std::optional<SmRateRow> sm_rate_row(int major, int minor) {
    for (const SmRateRow& r : sm_rate_facts_table()) {
        if (r.major == major && r.minor == minor) return r;
    }
    return std::nullopt;
}

std::span<const std::string> sm_rate_fact_names() {
    static const std::vector<std::string> names = [] {
        std::vector<std::string> v;
        for (const Field& f : fields()) v.push_back(f.name);
        return v;
    }();
    return names;
}

} // namespace ninferno
