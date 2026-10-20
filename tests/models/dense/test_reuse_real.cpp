// Prefix reuse through the Engine on a dense artifact whose generation prompt ends on an execution
// boundary (SmolLM3: the assistant turn opens and a later turn renders the same opening).
// NINFERNO_TEST_DENSE_REUSE_ARTIFACT names the artifact; without it the test skips.
//
// A sequence seen on ninferno-serve (requests as the server prepares them: the same 391 / 388 /
// 391 tokens as the server): A (a tool request), B (A's call and its result back, restored from
// A's turn closure; its prefill captures a shared stable prefix exactly at A's prompt frontier),
// then A again. The third request restores that frontier with nothing left to prefill
// (zero suffix): the shared image holds the tail hidden its capturing prefill wrote, so the engine
// samples from it, and the answer is A's, token for token (cold == warm). Before the fix the
// Program refused it ("zero-suffix reuse has no target tail hidden") and the engine went
// unavailable. Also: A again straight after A (the private checkpoints), cold == warm.

#include "ninferno/engine.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <cstdint>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

constexpr const char* kTool =
    R"({"type":"function","function":{"name":"get_weather","description":"Current weather for )"
    R"(a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},)"
    R"("required":["city"]}}})";

ninferno::ChatMessage text_message(ninferno::ChatRole role, std::string text) {
    ninferno::ChatMessage message;
    message.role = role;
    message.parts.push_back({.kind = ninferno::MessagePartKind::Text, .text = std::move(text)});
    return message;
}

// As ninferno-serve prepares an OpenAI chat request: the automatic prompt-cache boundary after the
// last message's last part, and no Engine structural candidates of its own.
ninferno::PromptInput prompt(std::vector<ninferno::ChatMessage> messages) {
    ninferno::PromptInput input;
    input.context_cache.markers.push_back(ninferno::PromptCacheMarker{
        .after_message_count      = static_cast<std::uint32_t>(messages.size()),
        .kind                     = ninferno::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence                 = ninferno::SharedCandidateEvidence::DefaultAutomatic,
        .location                 = ninferno::PromptCacheMarkerLocation::MessagePartBoundary,
        .after_message_part_count = static_cast<std::uint32_t>(messages.back().parts.size()),
    });
    input.context_cache.allow_engine_automatic_shared_prefixes = false;
    input.messages                = std::move(messages);
    input.options.enable_thinking = true;
    input.options.tool_jsons      = {kTool};
    return input;
}

ninferno::PromptInput ask() {
    return prompt({text_message(ninferno::ChatRole::User, "What is the weather in Paris?")});
}

ninferno::PromptInput round_trip() {
    ninferno::ChatMessage call;
    call.role = ninferno::ChatRole::Assistant;
    call.tool_calls.push_back(
        {.id = "call_1", .name = "get_weather", .arguments_json = R"({"city":"Paris"})"});
    ninferno::ChatMessage result = text_message(
        ninferno::ChatRole::Tool, R"({"city": "Paris", "temperature_c": 18, "sky": "clear"})");
    result.tool_call_id = "call_1";
    return prompt({text_message(ninferno::ChatRole::User, "What is the weather in Paris?"),
                   std::move(call), std::move(result)});
}

ninferno::GenerationResult run(ninferno::Engine& engine, ninferno::PromptInput input,
                             std::uint32_t tokens) {
    ninferno::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = true;
    return engine.generate(engine.prepare(std::move(input)), options);
}

ninferno::EngineOptions engine_options(const char* artifact) {
    ninferno::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 2048;
    options.kv_capacity = ninferno::KvCapacityPolicy::explicit_capacity(2048);
    options.prefill_chunk                        = 1024;
    options.max_concurrency                      = 1;
    options.context_cache.host_state_slots       = 2;
    options.context_cache.host_kv_capacity_bytes = 256ULL << 20;
    return options;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFERNO_TEST_DENSE_REUSE_ARTIFACT");
    if (artifact == nullptr || *artifact == 0) {
        std::cout << "SKIP: set NINFERNO_TEST_DENSE_REUSE_ARTIFACT\n";
        return 77;
    }
    try {
        {
            ninferno::Engine engine(engine_options(artifact));
            const ninferno::GenerationResult cold = run(engine, ask(), 48);
            const ninferno::GenerationResult trip = run(engine, round_trip(), 16);
            check(trip.reused_prompt_tokens != 0, "the round trip restores A's prefix");
            const ninferno::GenerationResult warm = run(engine, ask(), 48);
            std::cout << "A " << cold.prompt.prompt_tokens << " tokens; B reused "
                      << trip.reused_prompt_tokens << "; A again reused "
                      << warm.reused_prompt_tokens << '\n';
            check(warm.reused_prompt_tokens == cold.prompt.prompt_tokens,
                  "A again restores its whole prompt (zero suffix), reused " +
                      std::to_string(warm.reused_prompt_tokens));
            check(warm.generated_token_ids == cold.generated_token_ids,
                  "A again after the round trip answers as A cold");
        }
        {
            ninferno::Engine engine(engine_options(artifact));
            const ninferno::GenerationResult cold = run(engine, ask(), 48);
            const ninferno::GenerationResult warm = run(engine, ask(), 48);
            check(warm.reused_prompt_tokens != 0, "A again restores a prefix of A");
            check(warm.generated_token_ids == cold.generated_token_ids,
                  "A again straight after A answers as A cold");
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "dense reuse real: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
