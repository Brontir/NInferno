#include "core/isa_facts.h"

#include "core/embedded_isa_facts.h"

#include <nlohmann/json.hpp>

#include <climits>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <utility>

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

// A launch fact: its name in the entry (all in ns) and the member it fills.
struct LaunchField {
    std::string name;
    std::optional<double> IsaLaunch::* member;
};

const std::vector<LaunchField>& launch_fields() {
    static const std::vector<LaunchField> all = {
        {"node_ns", &IsaLaunch::node_ns},
        {"node_pdl_ns", &IsaLaunch::node_pdl_ns},
        {"stream_ns", &IsaLaunch::stream_ns},
        {"floor_ns_1", &IsaLaunch::floor_ns_1},
        {"floor_ns_sms", &IsaLaunch::floor_ns_sms},
        {"floor_ns_4sms", &IsaLaunch::floor_ns_4sms},
        {"floor_ns_1024", &IsaLaunch::floor_ns_1024},
        {"floor_ns_4096", &IsaLaunch::floor_ns_4096},
    };
    return all;
}

// One fact's value from a facts object: in `unit`, a positive finite number.
std::optional<double> fact_of(const Json& facts, const std::string& name, std::string_view unit) {
    const auto it = facts.find(name);
    if (it == facts.end()) return std::nullopt; // unknown on this row
    const std::string where = "fact " + name;
    const std::string got   = it->at("unit").get<std::string>();
    if (got != unit)
        throw std::runtime_error(where + " is in " + got + ", not " + std::string(unit));
    const Json& v = it->at("value");
    if (!v.is_number()) throw std::runtime_error(where + " is not a number");
    const double x = v.get<double>();
    if (!std::isfinite(x) || x <= 0) throw std::runtime_error(where + " is not positive");
    return x;
}

// An integer field: a JSON integer in int range, never a fraction or a string that get<int>()
// would truncate or convert.
int integer_of(const Json& j, const char* what) {
    const Json& v = j.at(what);
    if (!v.is_number_integer()) {
        throw std::runtime_error(std::string(what) + " is not an integer");
    }
    const long long x = v.get<long long>();
    if (x < INT_MIN || x > INT_MAX)
        throw std::runtime_error(std::string(what) + " is out of range");
    return static_cast<int>(x);
}

int positive_int(const Json& j, const char* what) {
    const int v = integer_of(j, what);
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
    for (const Field& f : fields()) f.slot(r) = fact_of(facts, f.name, f.unit);

    // The launch entry (launch_probe), where the architecture has one: of the same architecture,
    // its card and probe named, its facts in ns.
    if (const auto it = j.find("launch"); it != j.end()) {
        const Json& l = *it;
        if (positive_int(l, "cc") != cc)
            throw std::runtime_error("launch: not a run of cc " + digits);
        IsaLaunch launch;
        launch.card    = text(l, "card");
        launch.probe   = text(l.at("tool"), "probe");
        launch.sms     = positive_int(l, "sms");
        const Json& lf = l.at("facts");
        for (const LaunchField& f : launch_fields()) {
            try {
                launch.*f.member = fact_of(lf, f.name, "ns");
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("launch: ") + e.what());
            }
        }
        r.launch = std::move(launch);
    }
    return r;
}

} // namespace

std::vector<IsaRow> parse_isa_facts(std::string_view json_text) {
    const Json root = Json::parse(json_text);
    if (integer_of(root, "schema") != 1) throw std::runtime_error("isa facts: schema is not 1");
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

std::span<const std::string> isa_launch_fact_names() {
    static const std::vector<std::string> names = [] {
        std::vector<std::string> v;
        for (const LaunchField& f : launch_fields()) v.push_back(f.name);
        return v;
    }();
    return names;
}

} // namespace ninferno
