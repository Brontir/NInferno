#include "runtime/engine/family_registry.h"

#include "artifact/reader.h"
#include "runtime/engine/contract_backend.h"
#include "runtime/engine/model_instance.h"
#include "runtime/engine/plan_dry_run.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ninferno::runtime {
namespace {

// The families of this build, by the model_type their artifacts declare. A family is added by one
// row here and its loader; nothing else in the Engine or the server changes.
// Their automatic storages are one table beside the plan's (plan_dry_run.h family_storage).
constexpr std::array kFamilies{
    FamilyEntry{"qwen3_5_text", &load_qwen3_5}, FamilyEntry{"qwen3_5_moe_text", &load_qwen3_5},
    FamilyEntry{"gemma4_text", &load_gemma4},   FamilyEntry{"dense_decoder", &load_dense},
    FamilyEntry{"moe_decoder", &load_dense},    FamilyEntry{"nemotron_h", &load_nemotron_h},
};

} // namespace

std::span<const FamilyEntry> families() noexcept { return kFamilies; }

std::string artifact_model_type(const std::filesystem::path& path) {
    const artifact::Reader reader(path);
    const auto& components = reader.directory().components;
    const auto text        = components.find("text");
    if (text == components.end()) {
        throw std::invalid_argument("artifact " + path.string() + " has no text component");
    }
    const auto& config = text->second.config;
    if (!config.is_object() || !config.contains("model_type") ||
        !config.at("model_type").is_string()) {
        throw std::invalid_argument("artifact " + path.string() +
                                    " text component declares no model_type");
    }
    return config.at("model_type").get<std::string>();
}

namespace {

// The drafter architectures an artifact may hold on its own (ninferno-convert writes a drafter
// alone as well as beside its target): they only propose tokens for a target.
constexpr std::array<std::string_view, 4> kDrafterTypes{"dflash", "gemma4_assistant",
                                                        "gemma4_dflash", "nemotron_h_mtp"};

const FamilyEntry& family_entry(std::string_view model_type) {
    for (const FamilyEntry& family : kFamilies) {
        if (family.model_type == model_type) { return family; }
    }
    // Serving a drafter's own artifact named only the families served, which read as a missing
    // family (RTX 5080 test matrix: gpt-oss-20b-dflash.ninferno, 26b_mtp_gguf.ninferno).
    if (std::find(kDrafterTypes.begin(), kDrafterTypes.end(), model_type) != kDrafterTypes.end()) {
        throw std::invalid_argument("artifact model_type '" + std::string(model_type) +
                                    "' is a drafter: serve its target model and attach this "
                                    "artifact with --drafter (and --spec, --draft-tokens)");
    }
    std::string served;
    for (const FamilyEntry& family : kFamilies) {
        if (!served.empty()) { served += ", "; }
        served += family.model_type;
    }
    throw std::invalid_argument("artifact model_type '" + std::string(model_type) +
                                "' has no model family in this build (served: " + served + ")");
}

} // namespace

FamilyLoader family_loader(std::string_view model_type) { return family_entry(model_type).load; }

KvCacheStorage automatic_kv_cache(std::string_view model_type) {
    (void)family_entry(model_type); // throws for a model_type no family declares
    return family_storage(model_type).value_or(PlanFamilyStorage{}).kv;
}

RecurrentStateStorage automatic_recurrent_state(std::string_view model_type) {
    (void)family_entry(model_type);
    return family_storage(model_type).value_or(PlanFamilyStorage{}).recurrent;
}

KvCacheStorage automatic_kv_cache(std::string_view model_type, int compute_capability) {
    return kv_storage_on(automatic_kv_cache(model_type), compute_capability);
}

std::unique_ptr<Backend> make_backend(const EngineOptions& options, DeviceContext& device) {
    validate_engine_options(options);
    ConstructedModel constructed =
        construct_model(options, device, family_loader(artifact_model_type(options.artifact_path)));
    // The Engine serves under the options as construction resolved them (an automatic
    // max_context among them).
    return std::make_unique<ContractBackend<ModelInstance>>(
        std::move(constructed.instance), std::move(constructed.load),
        std::move(constructed.context_cost), constructed.options, device);
}

} // namespace ninferno::runtime
