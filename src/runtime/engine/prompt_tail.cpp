#include "runtime/engine/prompt_tail.h"

#include "artifact/reader.h"
#include "models/load_options.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/program/planning/neutral_frontier.h"
#include "runtime/engine/family_registry.h"
#include "runtime/engine/hf_text_frontend.h"

#include <ninferno/types.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace ninferno::runtime {
namespace {

// A one-message prompt of `words` words, as ninferno-serve prepares an OpenAI chat request's: its
// default automatic cache boundary after the message's last part (serve/openai_common.cpp).
struct Probe {
    // The rewrite execution frontiers as distances from the prompt's end (0 left out: a frontier
    // at the end splits nothing).
    std::set<std::uint32_t> tail;
    // A cache capture the program places off the prompt's static frontiers (on the 64-token grid
    // below the message's end, neutral_frontier.h).
    bool off_grid_capture = false;
};

Probe probe(const models::qwen3_5::Frontend& frontend, int words) {
    std::string text;
    for (int i = 0; i < words; ++i) { text += i == 0 ? "word" : " word"; }
    PromptInput input;
    ChatMessage message;
    message.role = ChatRole::User;
    message.parts.push_back(MessagePart{.kind = MessagePartKind::Text, .text = std::move(text)});
    input.messages.push_back(std::move(message));
    input.context_cache.allow_engine_automatic_shared_prefixes = false;
    input.context_cache.markers.push_back(PromptCacheMarker{
        .after_message_count      = 1,
        .kind                     = PromptCacheMarkerKind::SharedStablePrefix,
        .evidence                 = SharedCandidateEvidence::DefaultAutomatic,
        .location                 = PromptCacheMarkerLocation::MessagePartBoundary,
        .after_message_part_count = 1,
    });
    const models::qwen3_5::PreparedPrompt prepared = frontend.prepare(std::move(input));
    const models::qwen3_5::PreparedPromptData& prompt =
        models::qwen3_5::PreparedPromptAccess::view(prepared);
    const auto tokens = static_cast<std::uint32_t>(prompt.token_ids.size());
    const std::span<const std::uint32_t> frontiers(prompt.identity.rewrite_execution_frontiers);
    Probe out;
    for (const std::uint32_t f : frontiers) {
        if (f < tokens) { out.tail.insert(tokens - f); }
    }
    // Where the program places the captures (request_plan.cpp): a frontier that is neither a static
    // one nor the prompt's end splits the prefill.
    for (const auto& opportunity :
         models::qwen3_5::runtime_support::neutral_cache_opportunities(
             std::span<const models::qwen3_5::PreparedCacheOpportunity>(
                 prompt.context_cache.opportunities),
             tokens, frontiers)) {
        out.off_grid_capture =
            out.off_grid_capture ||
            (opportunity.frontier != tokens &&
             !std::binary_search(frontiers.begin(), frontiers.end(), opportunity.frontier));
    }
    return out;
}

std::uint32_t count_tail_calls(const models::qwen3_5::Frontend& frontend) {
    // Two message lengths: a frontier inside the generation prompt keeps its distance from the end.
    const Probe a = probe(frontend, 300);
    const Probe b = probe(frontend, 701);
    std::vector<std::uint32_t> both;
    std::set_intersection(a.tail.begin(), a.tail.end(), b.tail.begin(), b.tail.end(),
                          std::back_inserter(both));
    // The automatic shared prefix at the last message's end (OpenAI's default automatic boundary
    // in ninferno-serve) is captured at the 64-token
    // grid point below it, so the prompt's last chunk stops there and the remainder, 17 to 80
    // tokens, is one more call: a weight read like the tail's. At most one prompt in 64 ends on
    // the grid.
    const bool split = a.off_grid_capture || b.off_grid_capture;
    return static_cast<std::uint32_t>(both.size()) + (split ? 1U : 0U);
}

} // namespace

std::uint32_t prompt_tail_calls(const std::filesystem::path& artifact,
                                const std::filesystem::path& chat_template) {
    const std::string type = artifact_model_type(artifact);
    if (type == "qwen3_5_text" || type == "qwen3_5_moe_text") {
        const artifact::Reader reader(artifact);
        const models::qwen3_5::LoadPlan plan = models::qwen3_5::plan_load(reader);
        // The text frontend the engine makes without --vision (model_instance.cpp load_qwen3_5).
        models::qwen3_5::FrontendOptions options;
        options.chat_template_path = chat_template;
        options.architecture       = plan.config().text.architecture;
        options.vision_enabled     = false;
        options.vision_component   = reader.directory().components.contains("vision");
        options.max_context        = 4096;
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
    options.chat_template_path = chat_template;
    models::qwen3_5::apply_hf_text_options(options);
    options.max_context = 4096;
    return count_tail_calls(models::qwen3_5::make_frontend(resources, options));
}

std::optional<std::uint32_t> prompt_tail_calls_or_warn(const std::filesystem::path& artifact,
                                                       const std::filesystem::path& chat_template) {
    try {
        return prompt_tail_calls(artifact, chat_template);
    } catch (const std::exception& error) {
        std::cerr << "ninferno: the prefill's tail calls of " << artifact.string()
                  << " are not counted (priced as none): " << error.what() << '\n';
        return std::nullopt;
    }
}

} // namespace ninferno::runtime
