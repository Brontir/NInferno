#include "core/die_round_facts.h"
#include "core/embedded_die_round_facts.h"
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace ninferno {

namespace {

using Json = nlohmann::json;

RowsOpenCurve curve_of(const Json& j, const std::string& what) {
    if (!j.is_array()) throw std::runtime_error("die_round_facts: " + what + " is not a list");
    RowsOpenCurve c;
    for (const Json& p : j) {
        if (!p.is_array() || p.size() != 2 || !p[0].is_number() || !p[1].is_number()) {
            throw std::runtime_error("die_round_facts: " + what + ": a point is not [rows, share]");
        }
        c.emplace_back(p[0].get<double>(), p[1].get<double>());
    }
    if (c.empty()) throw std::runtime_error("die_round_facts: " + what + " has no point");
    return c;
}

std::map<std::string, RowsOpenCurve> curves_of(const Json& j, const std::string& what) {
    if (!j.is_object()) throw std::runtime_error("die_round_facts: " + what + " is not a map");
    std::map<std::string, RowsOpenCurve> m;
    for (const auto& [geo, pts] : j.items()) m[geo] = curve_of(pts, what + " " + geo);
    return m;
}

std::optional<double> number(const Json& j, const char* key) {
    if (!j.contains(key)) return std::nullopt;
    if (!j[key].is_number()) throw std::runtime_error(std::string("die_round_facts: ") + key + " is not a number");
    return j[key].get<double>();
}

} // namespace

DieRoundTable parse_die_round_facts(std::string_view json_text) {
    const Json root = Json::parse(json_text);
    DieRoundTable t;
    t.probe_row_bytes = root.at("probe_row_bytes").get<double>();
    for (const auto& [a, v] : root.at("arch").at("stream_share").items()) t.arch_stream_share[a] = v.get<double>();
    for (const auto& [a, v] : root.at("arch").at("row_flat").items()) t.arch_row_flat[a] = v.get<double>();
    for (const auto& [chip, j] : root.at("dies").items()) {
        DieRoundFacts d;
        d.chip           = chip;
        d.stream_share   = number(j, "stream_share");
        d.head_t1_ns     = number(j, "head_t1_ns");
        d.in_phase_waves = number(j, "in_phase_waves");
        d.in_phase_share = number(j, "in_phase_share");
        d.row_flat       = number(j, "row_flat");
        if (j.contains("rows_open")) {
            const Json& ro = j["rows_open"];
            if (ro.contains("request") && ro["request"].is_string()) d.request = ro["request"].get<std::string>();
            if (ro.contains("curve")) d.curve = curves_of(ro["curve"], chip + " curve");
            if (ro.contains("pitch")) {
                for (const auto& [pb, geo] : ro["pitch"].items()) {
                    d.pitch[std::stod(pb)] = curves_of(geo, chip + " pitch " + pb);
                }
            }
        }
        t.dies[chip] = std::move(d);
    }
    return t;
}

const DieRoundTable& die_round_facts() {
    static const DieRoundTable table = parse_die_round_facts(embedded::kDieRoundFactsJson);
    return table;
}

const DieRoundFacts* die_round_row(std::string_view chip) {
    const auto& dies = die_round_facts().dies;
    const auto it    = dies.find(std::string(chip));
    return it == dies.end() ? nullptr : &it->second;
}

} // namespace ninferno
