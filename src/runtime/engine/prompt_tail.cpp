#include "runtime/engine/prompt_tail.h"

#include "artifact/reader.h"
#include "models/load_options.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/load.h"
#include "runtime/engine/family_registry.h"
#include "runtime/engine/hf_text_frontend.h"

#include <ninferno/types.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace ninferno::runtime {
namespace {

// The frontiers of a one-message prompt of `words` words, as distances from its end (0 left out:
// a frontier at the end splits nothing).
std::set<std::uint32_t> tail_frontiers(const models::qwen3_5::Frontend& frontend, int words) {
    std::string text;
    for (int i = 0; i < words; ++i) { text += i == 0 ? "word" : " word"; }
    PromptInput input;
    ChatMessage message;
    message.role = ChatRole::User;
    message.parts.push_back(MessagePart{.kind = MessagePartKind::Text, .text = std::move(text)});
    input.messages.push_back(std::move(message));
    const models::qwen3_5::PreparedPrompt prepared = frontend.prepare(std::move(input));
    const models::qwen3_5::PreparedPromptData& prompt =
        models::qwen3_5::PreparedPromptAccess::view(prepared);
    const auto tokens = static_cast<std::uint32_t>(prompt.token_ids.size());
    std::set<std::uint32_t> out;
    for (const std::uint32_t f : prompt.identity.rewrite_execution_frontiers) {
        if (f < tokens) { out.insert(tokens - f); }
    }
    return out;
}

std::uint32_t count_tail_calls(const models::qwen3_5::Frontend& frontend) {
    // Two message lengths: a frontier inside the generation prompt keeps its distance from the end.
    const std::set<std::uint32_t> a = tail_frontiers(frontend, 40);
    const std::set<std::uint32_t> b = tail_frontiers(frontend, 90);
    std::vector<std::uint32_t> both;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(both));
    return static_cast<std::uint32_t>(both.size());
}

} // namespace

std::uint32_t prompt_tail_calls(const std::filesystem::path& artifact) {
    const std::string type = artifact_model_type(artifact);
    if (type == "qwen3_5_text" || type == "qwen3_5_moe_text") {
        const artifact::Reader reader(artifact);
        const models::qwen3_5::LoadPlan plan = models::qwen3_5::plan_load(reader);
        models::qwen3_5::FrontendOptions options;
        options.architecture = plan.config().text.architecture;
        options.max_context  = 4096;
        return count_tail_calls(models::qwen3_5::make_frontend(plan.resources(), options));
    }
    const HfTextFiles files = read_hf_text_files(artifact);
    models::qwen3_5::FrontendResources resources{
        .tokenizer_json         = files.tokenizer_json,
        .tokenizer_config_json  = files.tokenizer_config_json,
        .chat_template_jinja    = files.chat_template_jinja,
        .generation_config_json = files.generation_config_json,
    };
    models::qwen3_5::attach_hf_text_tokenizer(resources);
    models::qwen3_5::FrontendOptions options;
    models::qwen3_5::apply_hf_text_options(options);
    options.max_context = 4096;
    return count_tail_calls(models::qwen3_5::make_frontend(resources, options));
}

std::optional<std::uint32_t> prompt_tail_calls_or_warn(const std::filesystem::path& artifact) {
    try {
        return prompt_tail_calls(artifact);
    } catch (const std::exception& error) {
        std::cerr << "ninferno: the prefill's tail calls of " << artifact.string()
                  << " are not counted (priced as none): " << error.what() << '\n';
        return std::nullopt;
    }
}

} // namespace ninferno::runtime
