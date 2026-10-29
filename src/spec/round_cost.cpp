#include "spec/round_cost.h"

#include "core/generation.h"
#include "core/machine_facts.h"
#include "spec/profile.h"

#include <cmath>
#include <stdexcept>

namespace ninferno::spec {

namespace {

using Json = nlohmann::json;

Json pass_json(const PassFacts& p) {
    Json out{{"bytes", p.bytes}, {"params", p.params}};
    if (p.experts > 0) {
        out["expert_bytes"]  = p.expert_bytes;
        out["expert_params"] = p.expert_params;
        out["experts"]       = p.experts;
        out["experts_used"]  = p.experts_used;
    }
    if (!p.linears.empty()) {
        Json groups = Json::array();
        for (const auto& g : p.linears) {
            groups.push_back(Json{{"format", g.format}, {"n", g.n}, {"k", g.k}, {"count", g.count},
                                  {"head", g.head}, {"expert", g.expert}});
        }
        out["linears"] = std::move(groups);
    }
    return out;
}

PassFacts parse_pass(const Json& doc) {
    PassFacts p;
    p.bytes         = doc.at("bytes").get<double>();
    p.params        = doc.at("params").get<double>();
    p.expert_bytes  = doc.value("expert_bytes", 0.0);
    p.expert_params = doc.value("expert_params", 0.0);
    p.experts       = doc.value("experts", 0);
    p.experts_used  = doc.value("experts_used", 0);
    if (p.bytes <= 0 || p.params <= 0 || p.expert_bytes < 0 || p.expert_params < 0 ||
        p.experts < 0 || p.experts_used < 0 || p.experts_used > p.experts ||
        (p.experts > 0 && p.experts_used == 0) ||
        ((p.experts > 0) != (p.expert_bytes > 0))) {
        throw std::runtime_error("round facts: a pass's bytes, parameters or experts are malformed");
    }
    if (const auto groups = doc.find("linears"); groups != doc.end()) {
        for (const auto& g : *groups) {
            LinearGroup l;
            l.format = g.at("format").get<std::string>();
            l.n      = g.at("n").get<std::int64_t>();
            l.k      = g.at("k").get<std::int64_t>();
            l.count  = g.at("count").get<std::int32_t>();
            l.head   = g.value("head", false);
            l.expert = g.value("expert", false);
            if (l.format.empty() || l.n < 1 || l.k < 1 || l.count < 1) {
                throw std::runtime_error("round facts: a linear's format, rows, columns or count are malformed");
            }
            p.linears.push_back(std::move(l));
        }
    }
    return p;
}

double pass_cost(const PassFacts& p, std::int32_t rows, double kappa) {
    const double active = p.experts > 0 ? double(p.experts_used) / double(p.experts) : 1.0;
    const double bytes  = p.bytes + p.expert_bytes * moe_read_fraction(p, rows);
    const double ops    = 2.0 * double(rows) * (p.params + p.expert_params * active);
    return bytes + kappa * ops;
}

CardRates rates(int major, int minor, int sms, double clock_hz, double dram_bytes_per_s) {
    CardRates out;
    if (major <= 0 || sms <= 0 || clock_hz <= 0 || dram_bytes_per_s <= 0) { return out; }
    const GenerationRow& row = generation_row(major, minor);
    const double per_sm_clock = row.s8 > 0 ? row.s8 : kDp4aOpsPerSmClock;
    out.dram_bytes_per_s      = dram_bytes_per_s;
    out.int8_ops_per_s        = per_sm_clock * double(sms) * clock_hz;
    out.generation            = row.name;
    for (const auto& e : kSmallTEta) {
        if (e.generation == out.generation) {
            out.eta       = e.eta;
            out.eta_cards = e.cards;
        }
    }
    return out;
}

} // namespace

nlohmann::json round_facts_json(const RoundFacts& facts) {
    Json drafters = Json::object();
    for (const auto& [mode, d] : facts.drafters) {
        Json j         = pass_json(d.pass);
        j["rows"]      = d.rows;
        j["per_draft"] = d.per_draft;
        if (d.head) { j["head"] = pass_json(*d.head); }
        drafters[mode] = std::move(j);
    }
    return Json{{"target", pass_json(facts.target)}, {"drafters", std::move(drafters)}};
}

RoundFacts parse_round_facts(const nlohmann::json& doc) {
    RoundFacts out;
    try {
        out.target = parse_pass(doc.at("target"));
        for (const auto& [mode, d] : doc.at("drafters").items()) {
            if (mode != "mtp" && mode != "dflash" && mode != "dflash2") { continue; }
            DraftFacts f;
            f.pass      = parse_pass(d);
            f.rows      = d.at("rows").get<std::int32_t>();
            f.per_draft = d.at("per_draft").get<bool>();
            if (const auto h = d.find("head"); h != d.end()) { f.head = parse_pass(*h); }
            if (f.rows < 1) { throw std::runtime_error("round facts: a drafter's rows below 1"); }
            out.drafters[mode] = f;
        }
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(std::string("round facts: ") + error.what());
    }
    return out;
}

double moe_read_fraction(const PassFacts& pass, std::int32_t rows) noexcept {
    if (pass.experts <= 0) { return 1.0; }
    const double pick  = double(pass.experts_used) / double(pass.experts);
    const double apart = 1.0 - std::pow(1.0 - pick, double(rows));
    return pick + kMoeRoutedShare * (apart - pick);
}

CardRates card_rates(int compute_capability, int sms, int sm_clock_khz, double dram_gbps) {
    return rates(compute_capability / 10, compute_capability % 10, sms, sm_clock_khz * 1e3,
                 dram_gbps * 1e9);
}

CardRates card_rates(const GpuKey& gpu) {
    return card_rates(gpu.compute_capability, gpu.sms, gpu.sm_clock_khz, gpu.dram_gbps);
}

CardRates card_rates_of_board(std::uint32_t pci_device_id) {
    const GpuFactsRow* row = pci_device_id != 0 ? find_gpu_facts(pci_device_id) : nullptr;
    if (row == nullptr) { return {}; }
    return rates(row->major, row->minor, row->sms, row->boost_mhz * 1e6,
                 row->bus_bits / 8.0 * row->gbps * 1e9);
}

std::optional<double> round_cost(const RoundFacts& facts, const SpecConfig& config, double kappa) {
    if (!config.enabled() || config.k < 1) { return std::nullopt; }
    const auto d = facts.drafters.find(mode_name(config.mode));
    if (d == facts.drafters.end()) { return std::nullopt; }
    const double plain  = pass_cost(facts.target, 1, kappa);
    const double verify = pass_cost(facts.target, config.k + 1, kappa);
    const double passes = d->second.per_draft ? double(config.k) : 1.0;
    const double head   = d->second.head ? pass_cost(*d->second.head, config.k, kappa) : 0.0;
    const double draft  = passes * pass_cost(d->second.pass, d->second.rows, kappa) + head;
    return (verify + draft) / plain;
}

} // namespace ninferno::spec
