#include "runtime/engine/hf_text_frontend.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "models/qwen3_5/frontend/resources.h"

#include <stdexcept>
#include <utility>

namespace ninferno::runtime {

void refuse_qwen_only_options(const EngineOptions& options, std::string_view family) {
    if (options.rope_yarn_factor != 1.0F) {
        throw std::invalid_argument("--rope-yarn-factor is for Qwen 3.5/3.6 models; " +
                                    std::string(family) + " keeps its checkpoint's RoPE");
    }
    // Only the Qwen Gated DeltaNet state has a narrower storage; Nemotron-H keeps its Mamba2
    // state in FP32 and the attention-only families have none.
    if (options.recurrent_state != RecurrentStateStorage::FP32) {
        throw std::invalid_argument("--state-dtype bf16|f16 is for the Qwen 3.5/3.6/3.8 Gated "
                                    "DeltaNet state; " +
                                    std::string(family) + " keeps no such state: drop it");
    }
}

HfTextFiles read_hf_text_files(const std::filesystem::path& artifact) {
    const artifact::Reader reader(artifact);
    artifact::Binder binder(reader);
    const auto resource = [&](std::string_view role) {
        const auto bytes = binder.host_object(binder.resource("text", role));
        return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    HfTextFiles out;
    out.tokenizer_json         = resource("tokenizer.json");
    out.tokenizer_config_json  = resource("tokenizer_config.json");
    out.chat_template_jinja    = resource("chat_template.jinja");
    out.generation_config_json = resource("generation_config.json");
    return out;
}

std::int32_t attach_hf_text_frontend(FamilyLoad& out, const HfTextFiles& files,
                                     const EngineOptions& options, std::size_t embedding_rows,
                                     std::string_view family, bool gemma4_vision) {
    models::qwen3_5::FrontendResources resources{
        .tokenizer_json         = files.tokenizer_json,
        .tokenizer_config_json  = files.tokenizer_config_json,
        .chat_template_jinja    = files.chat_template_jinja,
        .generation_config_json = files.generation_config_json,
    };
    models::qwen3_5::attach_hf_text_tokenizer(resources);
    const std::size_t vocabulary = resources.public_token_count;
    if (vocabulary == 0 || vocabulary > embedding_rows ||
        !resources.tokenizer->has_exact_token_domain(vocabulary)) {
        throw std::invalid_argument(std::string(family) +
                                    " tokenizer must expose a contiguous public domain within the "
                                    "embedding rows");
    }
    models::qwen3_5::FrontendOptions frontend_options{
        .chat_template_path = options.chat_template_path,
        .sampling           = models::qwen3_5::checkpoint_sampling(files.generation_config_json),
        .media_cache_bytes  = options.media_cache_bytes,
        .media_live_bytes   = options.media_live_bytes,
        .media_preprocess_threads = options.media_preprocess_threads};
    models::qwen3_5::apply_hf_text_options(frontend_options);
    if (gemma4_vision) {
        frontend_options.vision_enabled   = true;
        frontend_options.vision_component = true;
        frontend_options.vision_style     = models::qwen3_5::VisionStyle::Gemma4;
    }
    out.make_frontend = [resources = std::move(resources),
                         frontend_options](std::uint32_t max_context) {
        models::qwen3_5::FrontendOptions sized = frontend_options;
        sized.max_context                      = max_context;
        return models::qwen3_5::make_frontend(resources, std::move(sized));
    };
    return static_cast<std::int32_t>(vocabulary);
}

} // namespace ninferno::runtime
