#include "ui/catalog.h"

#include "core/device_class.h"
#include "core/units.h"
#include "product/plan/plan_facts.h"
#include "spec/registry.h"
#include "ui/embedded_catalog.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ninferno::ui {

namespace {

// Bits per weight a quantization stores on average (scales included), for a size estimate
// before the listing arrives.
double effective_bits(const CatalogSource& source) {
    if (source.format == "nvfp4") { return 4.5; }
    if (source.quant.find("MXFP4") != std::string::npos) { return 4.25; }
    return source.bits + 0.85;
}

} // namespace

Catalog Catalog::parse(const nlohmann::json& document) {
    Catalog out;
    for (const auto& g : document.at("gpus")) {
        CatalogGpu gpu;
        gpu.key           = g.at("key").get<std::string>();
        gpu.match         = g.value("match", std::vector<std::string>{gpu.key});
        gpu.memory_gib    = g.value("memory_gib", 0.0);
        gpu.bandwidth_gbs = g.value("bandwidth_gbs", 0.0);
        gpu.fp4           = g.value("fp4", false);
        out.gpus_.push_back(std::move(gpu));
    }
    for (const auto& m : document.at("models")) {
        CatalogModel model;
        model.id       = m.at("id").get<std::string>();
        model.name     = m.value("name", model.id);
        model.kind     = m.value("kind", std::string());
        model.use      = m.value("use", std::string());
        model.spec     = m.value("spec", std::string());
        model.params_b = m.value("params_b", 0.0);
        model.active_b = m.value("active_b", model.params_b);
        for (const auto& s : m.at("sources")) {
            CatalogSource source;
            source.repo   = s.at("repo").get<std::string>();
            source.quant  = s.value("quant", std::string());
            source.format = s.value("format", std::string("gguf"));
            source.bits   = s.value("bits", 4);
            if (source.bits > 8) {
                throw std::runtime_error("catalog: " + model.id + " lists a source above 8 bits");
            }
            model.sources.push_back(std::move(source));
        }
        for (const auto& r : m.value("measured", nlohmann::json::array())) {
            CatalogMeasure measure;
            measure.gpu       = r.at("gpu").get<std::string>();
            measure.tps       = r.value("tps", 0.0);
            measure.llama_cpp = r.value("llama_cpp", 0.0);
            measure.context   = r.value("context", std::string());
            measure.quant     = r.value("quant", std::string());
            model.measured.push_back(std::move(measure));
        }
        model.load = m.value("load", nlohmann::json::array());
        if (!model.load.is_array()) {
            throw std::runtime_error("catalog: " + model.id + "'s load is not a list");
        }
        model.drafters   = m.value("drafters", nlohmann::json::array());
        model.spec_suite = m.value("spec_suite", nlohmann::json::array());
        if (!model.drafters.is_array() || !model.spec_suite.is_array()) {
            throw std::runtime_error("catalog: " + model.id +
                                     "'s drafters or spec_suite is not a list");
        }
        if (model.sources.empty()) {
            throw std::runtime_error("catalog: " + model.id + " has no source");
        }
        out.models_.push_back(std::move(model));
    }
    return out;
}

const Catalog& Catalog::builtin() {
    static const Catalog catalog = parse(nlohmann::json::parse(embedded::kCatalogJson));
    return catalog;
}

const CatalogGpu* Catalog::gpu_for(const std::string& nvml_name) const {
    for (const auto& gpu : gpus_) {
        for (const auto& pattern : gpu.match) {
            if (nvml_name.find(pattern) != std::string::npos) { return &gpu; }
        }
    }
    return nullptr;
}

const CatalogGpu* Catalog::gpu_by_key(const std::string& key) const {
    for (const auto& gpu : gpus_) {
        if (gpu.key == key) { return &gpu; }
    }
    return nullptr;
}

std::string size_key(const std::string& repo, const std::string& quant) {
    return repo + "|" + quant;
}

TargetGpu target_of(const GpuStat& stat) {
    TargetGpu out;
    out.name          = stat.name;
    out.memory_gib    = to_gib(stat.memory_total);
    out.bandwidth_gbs = stat.bandwidth_gbs();
    // Native NVFP4 tensor cores in this release: its sm_120a engine (compute capability 12.0).
    // The 10.x and 11.x engines are portable builds that decode NVFP4 in software, so a GGUF
    // source suits them better (a 12.1 GB10 is aarch64, which the x86_64 release does not run).
    out.fp4 = stat.cc_major == 12 && stat.cc_minor == 0;
    if (stat.cc_major > 0) {
        out.chip          = gpu_chip(stat.pci_device_id, stat.cc_major, stat.cc_minor, stat.sms);
        // The board's mma_slow as the device query takes it (board_mma_slow: the PCI-ID table's
        // row, else the chip's), so a GTX 16 board on TU106 borrows no tensor-core chip's profile.
        const DriverReport driver{
            .pci_device_id = stat.pci_device_id, .major = stat.cc_major, .minor = stat.cc_minor};
        const bool slow   = board_mma_slow(machine_facts(driver), stat.cc_major, stat.cc_minor,
                                           out.chip);
        out.profile_chain =
            gpu_similarity_chain(out.chip, stat.sms, stat.cc_major, stat.cc_minor, slow);
    }
    return out;
}

SpeculationPlan speculation_plan(const std::string& registry_id, const TargetGpu& gpu) {
    SpeculationPlan out;
    out.profile                   = "none";
    const spec::ModelEntry* entry = spec::Registry::builtin().by_id(registry_id);
    if (entry == nullptr) { return out; }
    for (const auto& drafter : entry->drafters) {
        if (std::find(out.drafters.begin(), out.drafters.end(), drafter.kind) ==
            out.drafters.end()) {
            out.drafters.push_back(drafter.kind);
        }
    }
    std::vector<std::string> classes;
    for (const auto& profile : entry->profiles) {
        classes.push_back(profile.value("gpu_class", std::string()));
    }
    auto chain = gpu.profile_chain;
    if (std::find(chain.begin(), chain.end(), "*") == chain.end()) { chain.push_back("*"); }
    for (const auto& key : chain) {
        if (std::find(classes.begin(), classes.end(), key) == classes.end()) { continue; }
        out.profile_class = key;
        if (key == "*") {
            out.profile = "generic";
        } else if (!gpu.chip.empty() && (key == gpu.chip || key.rfind(gpu.chip + ":", 0) == 0)) {
            out.profile = "own";
        } else {
            out.profile = "similar";
        }
        break;
    }
    return out;
}

void attach_speculation(nlohmann::json& model_json, const SpeculationPlan& plan) {
    model_json["speculation"] = {{"drafters", plan.drafters},
                                 {"profile", plan.profile},
                                 {"profile_class", plan.profile_class}};
}

CatalogModel with_fitting_quant(const CatalogModel& model, const TargetGpu& gpu,
                                const std::map<std::string, nlohmann::json>& listings,
                                const QuantFits& fits, const QuantPickOf& pick) {
    if (gpu.memory_gib <= 0) { return model; }
    const double budget = gpu.memory_gib * 0.95 - kReserveGib;
    // The plan's word where it has one, else the weights against the budget.
    const auto fit = [&](const std::string& repo, const nlohmann::json& q) {
        if (fits) {
            if (const std::optional<bool> planned = fits(repo, q)) { return *planned; }
        }
        return to_gib(q.value("bytes", 0.0)) <= budget;
    };
    for (const auto& source : model.sources) {
        const auto it = listings.find(source.repo);
        if (it == listings.end()) { continue; }
        for (const auto& q : it->second) {
            if (q.value("quant", std::string()) == source.quant && !q.value("drafter", false) &&
                fit(source.repo, q)) {
                return model; // a listed source fits already
            }
        }
    }
    const nlohmann::json* best = nullptr;
    std::string best_repo;
    // The rule's pick first (the same rule the detail page and ninferno-serve apply).
    if (pick) {
        for (const auto& source : model.sources) {
            if (source.format != "gguf") { continue; }
            const auto it = listings.find(source.repo);
            if (it == listings.end()) { continue; }
            if (const auto quant = pick(source.repo)) {
                for (const auto& q : it->second) {
                    if (q.value("quant", std::string()) == *quant && !q.value("drafter", false)) {
                        CatalogModel out = model;
                        out.sources.push_back({source.repo, *quant, "gguf", q.value("bits", 4)});
                        return out;
                    }
                }
            }
        }
    }
    for (const auto& source : model.sources) {
        if (source.format != "gguf") { continue; }
        const auto it = listings.find(source.repo);
        if (it == listings.end()) { continue; }
        for (const auto& q : it->second) {
            const int bits = q.value("bits", 0);
            if (q.value("drafter", false) || bits < 2 || bits > 4 || !fit(source.repo, q)) {
                continue;
            }
            if (best == nullptr || q.value("bytes", 0.0) > best->value("bytes", 0.0)) {
                best      = &q;
                best_repo = source.repo;
            }
        }
    }
    if (best == nullptr) { return model; }
    CatalogModel out = model;
    out.sources.push_back(
        {best_repo, best->value("quant", std::string()), "gguf", best->value("bits", 4)});
    return out;
}

nlohmann::json kv_summary(const product::PlanSourceFacts* facts) {
    if (facts == nullptr) {
        return {{"known", false}, {"architecture", ""}, {"kv_head_dim", 0}, {"native_context", 0}};
    }
    const auto& m = facts->model;
    return {{"known", m.full_layers + m.sliding_layers > 0},
            {"architecture", m.architecture},
            {"kv_head_dim", m.full_key_dim},
            {"native_context", m.native_context}};
}

ModelEstimate estimate(const Catalog& catalog, const CatalogModel& model, const TargetGpu& gpu,
                       const SizeTable& sizes) {
    ModelEstimate out;
    out.model               = &model;
    const CatalogGpu* known = catalog.gpu_for(gpu.name);
    const double bandwidth  = gpu.bandwidth_gbs > 0 ? gpu.bandwidth_gbs
                              : known != nullptr    ? known->bandwidth_gbs
                                                    : 0.0;

    // The measurement on this GPU model, else the one on the measured card nearest in bandwidth.
    const CatalogMeasure* here    = nullptr;
    const CatalogMeasure* nearest = nullptr;
    double nearest_bandwidth      = 0;
    for (const auto& measure : model.measured) {
        const CatalogGpu* card = catalog.gpu_by_key(measure.gpu);
        if (card == nullptr) { continue; }
        if (known != nullptr && card->key == known->key) { here = &measure; }
        if (bandwidth > 0 && (nearest == nullptr || std::abs(card->bandwidth_gbs - bandwidth) <
                                                        std::abs(nearest_bandwidth - bandwidth))) {
            nearest           = &measure;
            nearest_bandwidth = card->bandwidth_gbs;
        }
    }

    for (std::size_t i = 0; i < model.sources.size(); ++i) {
        const auto& source = model.sources[i];
        SourceEstimate e;
        e.source = i;
        if (const auto it = sizes.find(size_key(source.repo, source.quant)); it != sizes.end()) {
            e.size_gib   = to_gib(it->second);
            e.size_known = true;
        } else {
            e.size_gib = to_gib(model.params_b * 1e9 * effective_bits(source) / 8.0);
        }
        const double need     = e.size_gib + kReserveGib;
        e.fit                 = gpu.memory_gib <= 0             ? "no"
                                : need <= gpu.memory_gib * 0.95 ? "fits"
                                : need <= gpu.memory_gib        ? "tight"
                                                                : "no";
        const bool same_quant = [&](const CatalogMeasure* m) {
            return m != nullptr && (m->quant.empty() || m->quant == source.quant);
        }(here);
        if (here != nullptr && same_quant) {
            e.tps       = here->tps;
            e.llama_cpp = here->llama_cpp;
            e.tps_basis = "measured";
            e.context   = here->context.empty() ? "auto" : here->context;
        } else if (nearest != nullptr && nearest_bandwidth > 0) {
            // Decode follows bandwidth only so far: the measured cards gain at most ~1.5x from a
            // faster bus (Qwen3.8 27B: 1.55x for 1.78x from the 4090 to the 5090), small models
            // less (host and launch overheads); a slower bus loses in proportion.
            e.tps       = nearest->tps * std::min(bandwidth / nearest_bandwidth, 1.5);
            e.tps_basis = "scaled from " + nearest->gpu;
            e.context   = "auto";
        } else if (bandwidth > 0 && model.active_b > 0) {
            // Active weight bytes read per token, at the 55 % of peak bandwidth the measured cards
            // reach without speculation.
            const double bytes = model.active_b * 1e9 * effective_bits(source) / 8.0;
            e.tps              = 0.55 * bandwidth * 1e9 / bytes;
            e.tps_basis        = "bandwidth";
            e.context          = "auto";
        } else {
            e.context = "auto";
        }
        out.sources.push_back(std::move(e));
    }

    // Preferred: NVFP4 on an FP4 card, GGUF elsewhere; among those that fit, the most bits
    // (higher quality) first; else whatever fits.
    const auto rank = [&](const SourceEstimate& e) {
        const auto& source = model.sources[e.source];
        int score          = 0;
        if (e.fit == "fits") {
            score += 100;
        } else if (e.fit == "tight") {
            score += 50;
        }
        if (gpu.fp4 ? source.format == "nvfp4" : source.format == "gguf") { score += 10; }
        score += source.bits;
        return score;
    };
    for (std::size_t i = 0; i < out.sources.size(); ++i) {
        if (out.sources[i].fit == "no") { continue; }
        if (!out.preferred || rank(out.sources[i]) > rank(out.sources[*out.preferred])) {
            out.preferred = i;
        }
    }
    out.recommended = out.preferred.has_value() && out.sources[*out.preferred].fit == "fits";
    return out;
}

nlohmann::json to_json(const ModelEstimate& e) {
    const auto& model      = *e.model;
    nlohmann::json sources = nlohmann::json::array();
    for (const auto& s : e.sources) {
        const auto& source = model.sources[s.source];
        sources.push_back(
            {{"repo", source.repo},
             {"quant", source.quant},
             {"format", source.format},
             {"bits", source.bits},
             {"arg", source.quant.empty() ? source.repo : source.repo + ":" + source.quant},
             {"size_gib", std::round(s.size_gib * 10) / 10},
             {"size_known", s.size_known},
             {"fit", s.fit},
             {"tps", std::round(s.tps * 10) / 10},
             {"tps_basis", s.tps_basis},
             {"llama_cpp", s.llama_cpp},
             {"context", s.context}});
    }
    return {{"id", model.id},
            {"name", model.name},
            {"kind", model.kind},
            {"use", model.use},
            {"spec", model.spec},
            {"params_b", model.params_b},
            {"active_b", model.active_b},
            {"sources", sources},
            {"load", model.load},
            {"drafters", model.drafters},
            {"spec_suite", model.spec_suite},
            {"preferred", e.preferred ? nlohmann::json(*e.preferred) : nlohmann::json(nullptr)},
            {"recommended", e.recommended}};
}

} // namespace ninferno::ui
