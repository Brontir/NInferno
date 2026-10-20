#include "runtime/engine/family_registry.h"

#include "artifact/reader.h"
#include "runtime/engine/contract_backend.h"
#include "runtime/engine/model_instance.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ninferno::runtime {
namespace {

// The families of this build, by the model_type their artifacts declare. A family is added by one
// row here and its loader; nothing else in the Engine or the server changes.
constexpr std::array kFamilies{
    FamilyEntry{"qwen3_5_text", &load_qwen3_5, KvCacheStorage::Int8Group64,
                RecurrentStateStorage::FP16},
    FamilyEntry{"qwen3_5_moe_text", &load_qwen3_5, KvCacheStorage::Int8Group64,
                RecurrentStateStorage::FP16},
    // Gemma 4 keeps its full layers' pool in FP8 rows (the sliding rings stay BF16): RTX 5090
    // long1 26B at 218K decodes 173 -> 221 t/s, 31B at 98K 55 -> 63.
    FamilyEntry{"gemma4_text", &load_gemma4, KvCacheStorage::Fp8E4M3Row256},
    // The dense decoders keep FP8 rows (E4M3 and a scale per row) unless --kv-dtype says otherwise:
    // half the K/V a long decode step reads (gpt-oss ~24.6 KB a context token in BF16).
    FamilyEntry{"dense_decoder", &load_dense, KvCacheStorage::Fp8E4M3Row256},
    FamilyEntry{"moe_decoder", &load_dense, KvCacheStorage::Fp8E4M3Row256},
    FamilyEntry{"nemotron_h", &load_nemotron_h},
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
    return family_entry(model_type).kv_cache;
}

RecurrentStateStorage automatic_recurrent_state(std::string_view model_type) {
    return family_entry(model_type).recurrent_state;
}

KvCacheStorage automatic_kv_cache(std::string_view model_type, int compute_capability) {
    const KvCacheStorage storage = automatic_kv_cache(model_type);
    return storage == KvCacheStorage::Fp8E4M3Row256 && compute_capability < 89
               ? KvCacheStorage::BFloat16
               : storage;
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
