#include "core/isa_facts.h"

#include "core/embedded_isa_facts.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace ninferno {

namespace {

using Json = nlohmann::json;

// A fact the reader takes: its name in the table, its unit there, and the member it fills (a
// member of the row, or one warp count of a member that is measured by warps).
struct Field {
    std::string name;
    std::string_view unit;
    std::optional<double> IsaRow::* scalar    = nullptr;
    IsaByWarps IsaRow::* by_warps             = nullptr;
    std::optional<double> IsaByWarps::* warps = nullptr;

    std::optional<double>& slot(IsaRow& r) const {
        return scalar ? r.*scalar : (r.*by_warps).*warps;
    }
};

const std::vector<Field>& fields() {
    static const std::vector<Field> all = [] {
        std::vector<Field> v;
        const auto one = [&](std::string_view name, std::string_view unit,
                             std::optional<double> IsaRow::* member) {
            v.push_back(Field{std::string(name), unit, member});
        };
        const auto by_warps = [&](std::string_view name, std::string_view unit,
                                  IsaByWarps IsaRow::* member) {
            const std::pair<const char*, std::optional<double> IsaByWarps::*> warps[] = {
                {"_w1", &IsaByWarps::w1},
                {"_w2", &IsaByWarps::w2},
                {"_w4", &IsaByWarps::w4},
                {"_w8", &IsaByWarps::w8}};
            for (const auto& [suffix, w] : warps) {
                v.push_back(Field{std::string(name) + suffix, unit, nullptr, member, w});
            }
        };
        one("tensor_flops_per_sm_clock", "flop", &IsaRow::tensor_flops_per_sm_clock);
        one("mma_cycles_per_sm", "clk", &IsaRow::mma_cycles_per_sm);
        one("mma_cycles_per_partition", "clk", &IsaRow::mma_cycles_per_partition);
        one("mma_dependent_latency", "clk", &IsaRow::mma_dependent_latency);
        one("int32_cycles_per_warp_instruction", "clk", &IsaRow::int32_cycles_per_warp_instruction);
        one("fp32_cycles_per_warp_instruction", "clk", &IsaRow::fp32_cycles_per_warp_instruction);
        by_warps("l1_latency", "clk", &IsaRow::l1_latency);
        by_warps("l2_latency", "clk", &IsaRow::l2_latency);
        one("shared_latency", "clk", &IsaRow::shared_latency);
        by_warps("dram_latency", "clk", &IsaRow::dram_latency);
        by_warps("dram_tlbwarm_latency", "clk", &IsaRow::dram_tlbwarm_latency);
        by_warps("dram_tlbwarm_latency_ns", "ns", &IsaRow::dram_tlbwarm_latency_ns);
        return v;
    }();
    return all;
}

int positive_int(const Json& j, const char* what) {
    const int v = j.at(what).get<int>();
    if (v <= 0) throw std::runtime_error(std::string(what) + " is not positive");
    return v;
}

std::string text(const Json& j, const char* what) {
    std::string s = j.at(what).get<std::string>();
    if (s.empty()) throw std::runtime_error(std::string(what) + " is empty");
    return s;
}

IsaRow parse_row(const std::string& key, const Json& j) {
    IsaRow r;
    r.arch       = key;
    const int cc = positive_int(j, "cc");
    r.major      = cc / 10;
    r.minor      = cc % 10;
    // The key is the cc, or the cc and an arch-set suffix ("120a").
    const std::string digits = std::to_string(cc);
    if (key != digits && key != digits + "a" && key != digits + "f") {
        throw std::runtime_error("the key is not its cc " + digits);
    }
    r.card       = text(j, "card");
    r.probe      = text(j.at("tool"), "probe");
    r.provenance = text(j, "provenance");
    r.mma_shape  = text(j, "mma_shape");
    r.mma_flops  = positive_int(j, "mma_flops");
    r.sms        = positive_int(j, "sms");
    if (j.contains("sass_per_mma")) r.sass_per_mma = positive_int(j, "sass_per_mma");

    const Json& facts = j.at("facts");
    for (const Field& f : fields()) {
        const auto it = facts.find(f.name);
        if (it == facts.end()) continue; // unknown on this row
        const std::string where = "fact " + f.name;
        const std::string unit  = it->at("unit").get<std::string>();
        if (unit != f.unit) {
            throw std::runtime_error(where + " is in " + unit + ", not " + std::string(f.unit));
        }
        const Json& v = it->at("value");
        if (!v.is_number()) throw std::runtime_error(where + " is not a number");
        const double x = v.get<double>();
        if (!std::isfinite(x) || x <= 0) throw std::runtime_error(where + " is not positive");
        f.slot(r) = x;
    }
    return r;
}

} // namespace

std::vector<IsaRow> parse_isa_facts(std::string_view json_text) {
    const Json root = Json::parse(json_text);
    if (root.at("schema").get<int>() != 1) throw std::runtime_error("isa facts: schema is not 1");
    std::vector<IsaRow> rows;
    std::unordered_set<int> seen;
    for (const auto& [key, j] : root.at("arches").items()) {
        try {
            rows.push_back(parse_row(key, j));
        } catch (const std::exception& e) {
            throw std::runtime_error("isa facts: sm_" + key + ": " + e.what());
        }
        if (!seen.insert(rows.back().major * 10 + rows.back().minor).second) {
            throw std::runtime_error("isa facts: sm_" + key + ": its cc appears twice");
        }
    }
    return rows;
}

std::span<const IsaRow> isa_facts_table() {
    static const std::vector<IsaRow> table = parse_isa_facts(embedded::kIsaFactsJson);
    return table;
}

std::optional<IsaRow> isa_row(int major, int minor) {
    for (const IsaRow& r : isa_facts_table()) {
        if (r.major == major && r.minor == minor) return r;
    }
    return std::nullopt;
}

std::span<const std::string> isa_fact_names() {
    static const std::vector<std::string> names = [] {
        std::vector<std::string> v;
        for (const Field& f : fields()) v.push_back(f.name);
        return v;
    }();
    return names;
}

} // namespace ninferno
