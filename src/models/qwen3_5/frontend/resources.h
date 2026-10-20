#pragma once

#include "models/qwen3_5/config.h"
#include "models/qwen3_5/frontend/tokenizer.h"

#include <memory>
#include <string_view>

namespace ninferno::models::qwen3_5 {

// Byte views borrow the model's Host backing. Tokenizer owns its decoded vocabulary and tables;
// loading and request preparation share this one immutable interpretation.
struct FrontendResources {
    std::string_view tokenizer_json;
    std::string_view tokenizer_config_json;
    std::string_view chat_template_jinja;
    std::string_view generation_config_json;
    std::string_view preprocessor_config_json;
    std::string_view video_preprocessor_config_json;
    std::shared_ptr<const text::TokenCodec> tokenizer;
    std::uint32_t public_token_count = 0;
};

// The Qwen 3.5 family's tokenizer over the resources' tokenizer files: sets resources.tokenizer
// and resources.public_token_count (its whole vocabulary). parse_resources calls it, then checks
// the vocabulary against the model; the chat-template parity test calls it alone.
void attach_qwen3_5_tokenizer(FrontendResources& resources);

void parse_resources(FrontendResources& resources, const Config& config);

} // namespace ninferno::models::qwen3_5
