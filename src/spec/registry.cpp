#include "spec/registry.h"

#include "core/ascii.h"
#include "core/file_io.h"
#include "core/sha256.h"

#include "spec/embedded_data.h"
#include "spec/json_fields.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace ninferno::spec {
namespace {

using Json = nlohmann::json;

std::vector<std::string> strings(const Json& doc, const char* key) {
    const auto it = doc.find(key);
    if (it == doc.end() || it->is_null()) { return {}; }
    return it->get<std::vector<std::string>>();
}

} // namespace

DrafterDecl parse_drafter(const Json& d) {
    DrafterDecl out;
    out.name          = json_text(d, "name");
    out.kind          = d.at("kind").get<std::string>();
    out.repo          = json_text(d, "repo");
    out.revision      = json_text(d, "revision");
    out.recipe        = json_text(d, "recipe");
    out.component     = json_text(d, "component");
    out.license       = json_text(d, "license");
    out.ranking       = json_text(d, "ranking");
    out.gguf          = json_text(d, "gguf");
    out.proposal_rows = d.value("proposal_rows", std::uint64_t{0});
    out.variants      = strings(d, "variants");
    out.quant         = json_text(d, "quant");
    if (out.name.empty()) { out.name = out.kind; }
    if (out.name.find('.') != std::string::npos && out.quant.empty()) {
        throw std::invalid_argument("drafter name '" + out.name +
                                    "': a '.' names a variant (<name>.<variant>)");
    }
    for (const auto& v : out.variants) {
        if (v != "q8" && v != "q4" && v != "nvfp4") {
            throw std::invalid_argument("drafter " + out.name + ": variant '" + v +
                                        "' is not q8, q4 or nvfp4");
        }
    }
    if (!out.variants.empty() && !out.component.empty()) {
        throw std::invalid_argument("drafter " + out.name +
                                    ": a companion component has no variants");
    }
    return out;
}

std::pair<std::string, std::string> split_variant(std::string_view name) {
    const auto dot = name.find('.');
    if (dot == std::string_view::npos) { return {std::string(name), std::string()}; }
    return {std::string(name.substr(0, dot)), std::string(name.substr(dot + 1))};
}

Json drafter_json(const DrafterDecl& d) {
    Json out{{"name", d.name}, {"kind", d.kind}};
    if (!d.repo.empty()) { out["repo"] = d.repo; }
    if (!d.revision.empty()) { out["revision"] = d.revision; }
    if (!d.recipe.empty()) { out["recipe"] = d.recipe; }
    if (!d.component.empty()) { out["component"] = d.component; }
    if (!d.license.empty()) { out["license"] = d.license; }
    if (!d.ranking.empty()) { out["ranking"] = d.ranking; }
    if (!d.gguf.empty()) { out["gguf"] = d.gguf; }
    if (d.proposal_rows) { out["proposal_rows"] = d.proposal_rows; }
    if (!d.variants.empty()) { out["variants"] = d.variants; }
    if (!d.quant.empty()) { out["quant"] = d.quant; }
    return out;
}

ModelEntry parse_model_entry(const Json& item) {
    ModelEntry entry;
    entry.id       = json_text(item, "id");
    entry.engine   = json_text(item, "engine");
    entry.evidence = json_text(item, "evidence");
    if (const auto match = item.find("match"); match != item.end()) {
        entry.repos             = strings(*match, "repos");
        entry.lineage           = strings(*match, "lineage");
        entry.vocab_fingerprint = json_text(*match, "vocab_fingerprint");
        if (const auto g = match->find("geometry"); g != match->end()) {
            entry.geometry.hidden_size       = g->value("hidden_size", std::uint64_t{0});
            entry.geometry.num_hidden_layers = g->value("num_hidden_layers", std::uint64_t{0});
            entry.geometry.vocab_size        = g->value("vocab_size", std::uint64_t{0});
        }
    }
    for (const auto& d : item.value("drafters", Json::array())) {
        entry.drafters.push_back(parse_drafter(d));
    }
    entry.profiles = item.value("profiles", Json::array());
    return entry;
}

Registry Registry::parse(const std::string& text) {
    const auto doc = Json::parse(text);
    if (doc.value("schema", 0) != 1) { throw std::runtime_error("drafter registry: unknown schema"); }
    Registry out;
    out.published_ = json_text(doc, "published");
    out.version_   = doc.value("version", std::uint64_t{0});
    // (A named copy: items() of a temporary would dangle inside the loop.)
    const auto kinds = doc.value("kinds", Json::object());
    for (const auto& [name, info] : kinds.items()) {
        KindInfo kind;
        kind.name          = name;
        kind.engines       = strings(info, "engines");
        kind.architectures = strings(info, "architectures");
        kind.recipe        = json_text(info, "recipe");
        out.kinds_.push_back(std::move(kind));
    }
    for (const auto& item : doc.at("models")) { out.models_.push_back(parse_model_entry(item)); }
    return out;
}

const Registry& Registry::builtin() {
    static const Registry registry = [] {
        const char* path       = std::getenv("NINFERNO_DRAFTER_REGISTRY");
        const std::string text = path != nullptr && *path
                                     ? read_file(path, "$NINFERNO_DRAFTER_REGISTRY")
                                     : std::string(embedded::kDrafterRegistryJson);
        Registry out           = parse(text);
        out.source_sha256_     = sha256_hex(text);
        return out;
    }();
    return registry;
}

const ModelEntry* Registry::by_repo(std::string_view repo) const {
    const auto want = ascii_lower(repo);
    for (const auto& model : models_) {
        for (const auto& name : model.repos) {
            if (ascii_lower(name) == want) { return &model; }
        }
    }
    return nullptr;
}

const ModelEntry* Registry::by_lineage(std::string_view base) const {
    const auto want = ascii_lower(base);
    for (const auto& model : models_) {
        for (const auto& name : model.lineage) {
            if (ascii_lower(name) == want) { return &model; }
        }
    }
    return nullptr;
}

std::string engine_for_architecture(std::string_view architecture) {
    if (architecture.starts_with("Gemma4") && architecture.find("Assistant") == std::string::npos) {
        return "gemma4";
    }
    if (architecture.starts_with("Qwen3_5")) { return "qwen3_5"; }
    if (architecture == "NemotronHForCausalLM") { return "nemotron_h"; }
    // The dense family: its converted artifacts (dense and MoE, any source), and the checkpoints
    // the registry names drafters for.
    if (architecture == "DenseDecoderForCausalLM" || architecture == "MoeDecoderForCausalLM" ||
        architecture == "MuseGlimmerForConditionalGeneration") {
        return "dense";
    }
    return {};
}

double shape_distance(const Geometry& a, const Geometry& b) {
    if (a.hidden_size == 0 || a.num_hidden_layers == 0 || b.hidden_size == 0 ||
        b.num_hidden_layers == 0) {
        throw std::invalid_argument("shape_distance: incomplete geometry");
    }
    const auto log_ratio = [](std::uint64_t x, std::uint64_t y) {
        return std::abs(std::log2(static_cast<double>(x) / static_cast<double>(y)));
    };
    return log_ratio(a.hidden_size, b.hidden_size) +
           log_ratio(a.num_hidden_layers, b.num_hidden_layers);
}

const ModelEntry* Registry::nearest(std::string_view engine, const Geometry& geometry,
                                    std::string_view fingerprint, bool with_mtp) const {
    const auto complete = [](const Geometry& g) {
        return g.hidden_size != 0 && g.num_hidden_layers != 0 && g.vocab_size != 0;
    };
    if (engine.empty() || !complete(geometry)) { return nullptr; }
    const ModelEntry* best = nullptr;
    double best_distance   = 0;
    for (const auto& model : models_) {
        if (model.engine != engine || !complete(model.geometry) ||
            model.geometry.vocab_size != geometry.vocab_size ||
            (!fingerprint.empty() && !model.vocab_fingerprint.empty() &&
             model.vocab_fingerprint != fingerprint)) {
            continue;
        }
        const double distance = shape_distance(model.geometry, geometry);
        if (distance > kMaxModelDistance) { continue; }
        const bool mtp = std::any_of(model.drafters.begin(), model.drafters.end(),
                                     [](const DrafterDecl& d) { return d.kind == "mtp"; });
        if (distance > 0 && !(with_mtp && mtp)) { continue; }
        if (best == nullptr || distance < best_distance) {
            best          = &model;
            best_distance = distance;
        }
    }
    return best;
}

const ModelEntry* Registry::by_id(std::string_view id) const {
    for (const auto& model : models_) {
        if (model.id == id) { return &model; }
    }
    return nullptr;
}

std::optional<std::string> Registry::kind_for_architecture(std::string_view arch) const {
    for (const auto& kind : kinds_) {
        if (std::find(kind.architectures.begin(), kind.architectures.end(), arch) !=
            kind.architectures.end()) {
            return kind.name;
        }
    }
    return std::nullopt;
}

bool Registry::supported(std::string_view kind, std::string_view engine) const {
    for (const auto& info : kinds_) {
        if (info.name == kind) {
            return std::find(info.engines.begin(), info.engines.end(), engine) != info.engines.end();
        }
    }
    return false;
}

} // namespace ninferno::spec
