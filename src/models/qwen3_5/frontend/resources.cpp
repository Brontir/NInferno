#include "models/qwen3_5/frontend/resources.h"

#include "artifact/schema.h"

#include <mutex>
#include <string>
#include <utility>

namespace ninferno::models::qwen3_5 {
namespace {

// The last tokenizer built, kept for the same three files (#839). A start reads the tokenizer
// twice: the plan's choice counts its public ids (runtime::plan_tokenizer_tokens, through
// public_token_count) and then the load builds it, about 1.3 s each on a 12.8 MB tokenizer.json.
// The Tokenizer owns everything it decoded (no view into these bytes), so the load can share the
// one the choice built. Keyed by the bytes themselves, compared whole, so another file with the
// same size is never mistaken for it; one entry, replaced by the next distinct tokenizer. Its
// cost: the three files' bytes (about 13 MB) and one tokenizer stay for the process, also after
// their model unloads (ninferno-ui), until another tokenizer replaces them.
std::shared_ptr<const frontend::Tokenizer>
shared_tokenizer(const frontend::TokenizerResources& files) {
    struct Kept {
        std::string json, config, generation;
        std::shared_ptr<const frontend::Tokenizer> tokenizer;
    };

    static std::mutex mutex;
    static Kept kept;
    {
        const std::lock_guard lock(mutex);
        if (kept.tokenizer && kept.json == files.tokenizer_json &&
            kept.config == files.tokenizer_config_json &&
            kept.generation == files.generation_config_json) {
            return kept.tokenizer;
        }
    }
    auto built = std::make_shared<const frontend::Tokenizer>(files);
    Kept next{std::string(files.tokenizer_json), std::string(files.tokenizer_config_json),
              std::string(files.generation_config_json), built};
    const std::lock_guard lock(mutex);
    kept = std::move(next);
    return built;
}

} // namespace

void attach_qwen3_5_tokenizer(FrontendResources& resources) {
    resources.tokenizer = shared_tokenizer(
        frontend::TokenizerResources{resources.tokenizer_json, resources.tokenizer_config_json,
                                     resources.generation_config_json});
    resources.public_token_count = static_cast<std::uint32_t>(resources.tokenizer->vocab_size());
}

void parse_resources(FrontendResources& resources, const Config& config) {
    attach_qwen3_5_tokenizer(resources);
    const auto count = resources.tokenizer->vocab_size();
    if (!count || count > config.text.vocab_size ||
        !resources.tokenizer->has_exact_token_domain(count)) {
        throw artifact::ArtifactError(
            "tokenizer must expose a contiguous public domain within embedding rows");
    }
    resources.public_token_count = static_cast<std::uint32_t>(count);
    for (const auto token : resources.tokenizer->default_stop_token_ids()) {
        if (!resources.tokenizer->is_valid_token(token)) {
            throw artifact::ArtifactError("stop token is outside the public token domain");
        }
    }
    if (config.vision) {
        for (const auto bytes :
             {resources.preprocessor_config_json, resources.video_preprocessor_config_json}) {
            const auto value = artifact::parse_json(bytes, "Vision processor");
            if (!value.is_object() || !value.contains("patch_size") ||
                !value.contains("temporal_patch_size") || !value.contains("merge_size")) {
                throw artifact::ArtifactError("Vision processor is missing patch geometry");
            }
            if (artifact::require_u64(value.at("patch_size"), "patch_size", true) !=
                    config.vision->patch_size ||
                artifact::require_u64(value.at("temporal_patch_size"), "temporal_patch_size",
                                      true) != config.vision->temporal_patch_size ||
                artifact::require_u64(value.at("merge_size"), "merge_size", true) !=
                    config.vision->spatial_merge_size) {
                throw artifact::ArtifactError(
                    "Vision processor geometry differs from component config");
            }
        }
    }
    if (config.draft && config.draft->dflash2 && config.draft->dflash2->selector_top_k > count) {
        throw artifact::ArtifactError("DFlash2 selector_top_k exceeds public token domain");
    }
}

} // namespace ninferno::models::qwen3_5
