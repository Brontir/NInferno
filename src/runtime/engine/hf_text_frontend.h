#pragma once

#include "runtime/engine/model_instance.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace ninferno::runtime {

// The tokenizer, chat-template and generation-config files an artifact carries as text
// resources. The family's loaded model keeps them: its frontend reads them through views.
struct HfTextFiles {
    std::string tokenizer_json;
    std::string tokenizer_config_json;
    std::string chat_template_jinja;
    std::string generation_config_json;
};

[[nodiscard]] HfTextFiles read_hf_text_files(const std::filesystem::path& artifact);

// Refuses the options only the Qwen 3.5 family implements, before the weights load, so a family
// that would ignore one says so instead: --rope-yarn-factor (the checkpoint's own RoPE stays) and
// --state-dtype bf16|f16 (it keeps no Gated DeltaNet state). `family` names the model in the
// error. --activations is every family's (resolve_activation_precision).
void refuse_qwen_only_options(const EngineOptions& options, std::string_view family);

// Builds the HF tokenizer over `files`, checks that its public vocabulary is a contiguous domain
// within the `embedding_rows`, and sets out.make_frontend. Returns the public vocabulary. `family`
// names the model in the error. `gemma4_vision`: the model takes images through Gemma 4's
// vision tower (its prompt layout), else media are refused.
[[nodiscard]] std::int32_t attach_hf_text_frontend(FamilyLoad& out, const HfTextFiles& files,
                                                   const EngineOptions& options,
                                                   std::size_t embedding_rows,
                                                   std::string_view family,
                                                   bool gemma4_vision = false);

} // namespace ninferno::runtime
