// prefix_probe: measurement tool for prefix-reuse identity (not a ctest).
//
//   prefix_probe <artifact> <spec> <cache 0|1> <steps> [concurrency]
//
// spec: none | mtp:K | dflash:K | dflash2:K (DFlash/DFlash2 use the optimized proposal head).
// steps: comma list run in order on one Engine:
//   <i>      prompt case i through prepare (template identity, cache markers);
//   p<i>     case i through prepare_tokens (raw ids, no template identity);
//   c<i>+<j> cases i and j submitted together (greedy), both reported;
//   |        before an item starts a fresh Engine.
// PROBE_CHUNK=<n> sets EngineOptions::prefill_chunk. Prints one JSON line per request: prompt ids,
// output ids, reuse path, reused tokens, speculative rounds and accepted per position. Two runs
// of one request compare byte for byte when their prefill decompositions agree.
// Cases:
//   0 plain   "Write two sentences about lighthouses." 64 greedy
//   1 think   "What is 17 * 23? Think briefly." 160 greedy
//   2 seeded  "Invent a name for a new color and describe it." 48 T=0.9 top_p=.95 seed 4242, no think
//   3 rivers  "Name three rivers." 64 greedy
//   4 archive system prompt of 110 lines + question, 48 greedy, no think
//   5 paris   "Describe Paris." 200 greedy, no think
//   6 tools   "Weather in Oslo?" with a weather tool, 256 greedy
//   7 doc     the archive inside the user message, 64 greedy, no think
//   8 doc     as 7 with one output token (a long prefill without a decode)
#include "ninferno/engine.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Case {
    std::string system;
    std::string user;
    std::uint32_t tokens = 64;
    std::optional<bool> thinking;
    bool sampled = false;
    std::vector<std::string> tools;
};

std::string archive() {
    std::string text = "You are archive assistant 3. Archive:\n";
    for (int line = 0; line < 110; ++line) {
        text += "Entry " + std::to_string(3000 + line) + " stores item " +
                std::to_string((3 * 37 + line * 11) % 997) + " in shelf " +
                std::to_string(line % 13) + ".\n";
    }
    text.pop_back();
    return text;
}

const std::string kWeather =
    R"({"type":"function","function":{"name":"get_weather","description":"Current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["c","f"]}},"required":["city"]}}})";

std::vector<Case> cases() {
    return {
        {"", "Write two sentences about lighthouses.", 64, std::nullopt, false, {}},
        {"", "What is 17 * 23? Think briefly.", 160, std::nullopt, false, {}},
        {"", "Invent a name for a new color and describe it.", 48, false, true, {}},
        {"", "Name three rivers.", 64, std::nullopt, false, {}},
        {archive(), "Which shelf holds entry 3005?", 48, false, false, {}},
        {"", "Describe Paris.", 200, false, false, {}},
        {"", "Weather in Oslo?", 256, std::nullopt, false, {kWeather}},
        {"", "Summarize this archive in three sentences.\n" + archive(), 64, false, false, {}},
        {"", "Summarize this archive in three sentences.\n" + archive(), 1, false, false, {}},
    };
}

ninferno::PromptInput input(const Case& c) {
    ninferno::PromptInput prompt;
    const auto add = [&](ninferno::ChatRole role, const std::string& text) {
        ninferno::ChatMessage m;
        m.role = role;
        m.parts.push_back(ninferno::MessagePart{
            .kind = ninferno::MessagePartKind::Text, .text = text, .media = {}});
        prompt.messages.push_back(std::move(m));
    };
    if (!c.system.empty()) { add(ninferno::ChatRole::System, c.system); }
    add(ninferno::ChatRole::User, c.user);
    prompt.options.enable_thinking = c.thinking;
    prompt.options.tool_jsons      = c.tools;
    return prompt;
}

ninferno::EngineOptions engine_options(const std::string& artifact, const std::string& spec,
                                     bool cache, std::uint32_t concurrency) {
    ninferno::EngineOptions o;
    o.artifact_path   = artifact;
    o.max_context     = 16384;
    o.kv_capacity     = ninferno::KvCapacityPolicy::explicit_capacity(16384);
    o.max_concurrency = concurrency;
    o.kv_cache        = ninferno::KvCacheStorage::Int8Group64;
    if (const char* chunk = std::getenv("PROBE_CHUNK")) {
        o.prefill_chunk = static_cast<std::uint32_t>(std::atoi(chunk));
    }
    if (spec != "none") {
        const auto colon = spec.find(':');
        const std::string kind = spec.substr(0, colon);
        o.speculative.draft_tokens =
            static_cast<std::uint32_t>(std::atoi(spec.substr(colon + 1).c_str()));
        o.speculative.backend = kind == "mtp"      ? ninferno::SpeculativeBackend::Mtp
                                : kind == "dflash" ? ninferno::SpeculativeBackend::DFlash
                                                   : ninferno::SpeculativeBackend::DFlash2;
        o.speculative.proposal_head = ninferno::ProposalHead::Optimized;
    }
    o.context_cache.enabled = cache;
    if (!cache) {
        o.context_cache.host_state_slots       = 0;
        o.context_cache.host_kv_capacity_bytes = 0;
    }
    return o;
}

const char* path_name(ninferno::PrefixReusePath p) {
    switch (p) {
    case ninferno::PrefixReusePath::Root: return "root";
    case ninferno::PrefixReusePath::PrivateEndpoint: return "endpoint";
    case ninferno::PrefixReusePath::PrivateTurnClosure: return "turn_closure";
    case ninferno::PrefixReusePath::PrivateResponseReplay: return "response_replay";
    case ninferno::PrefixReusePath::PrivateLongAnchor: return "long_anchor";
    case ninferno::PrefixReusePath::SharedStablePrefix: return "shared";
    }
    return "?";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: prefix_probe <artifact> <spec> <cache 0|1> <steps> [concurrency]\n";
        return 2;
    }
    const std::string artifact = argv[1];
    const std::string spec     = argv[2];
    const bool cache           = std::atoi(argv[3]) != 0;
    const std::string steps    = argv[4];
    const auto concurrency =
        argc > 5 ? static_cast<std::uint32_t>(std::atoi(argv[5])) : std::uint32_t{4};
    const std::vector<Case> all = cases();
    try {
        std::unique_ptr<ninferno::Engine> engine;
        int engine_index = -1;
        int step         = 0;
        std::stringstream list(steps);
        std::string item;
        while (std::getline(list, item, ',')) {
            if (item.empty()) { continue; }
            if (item[0] == '|') {
                engine.reset();
                item = item.substr(1);
                if (item.empty()) { continue; }
            }
            if (!engine) {
                engine = std::make_unique<ninferno::Engine>(
                    engine_options(artifact, spec, cache, concurrency));
                ++engine_index;
            }
            if (item[0] == 'c') {
                // c<i>+<j>: submit case i, then case j, concurrently; report both.
                const auto plus = item.find('+');
                const std::size_t pair[2] = {
                    static_cast<std::size_t>(std::atoi(item.substr(1, plus - 1).c_str())),
                    static_cast<std::size_t>(std::atoi(item.substr(plus + 1).c_str()))};
                std::vector<ninferno::GenerationHandle> handles;
                for (const std::size_t idx : pair) {
                    const Case& c = all.at(idx);
                    ninferno::RequestOptions options;
                    options.execution.requested_output_tokens = c.tokens;
                    options.execution.sampling.temperature    = 0.0F;
                    handles.push_back(engine->submit(engine->prepare(input(c)), options));
                }
                for (int h = 0; h < 2; ++h) {
                    ninferno::GenerationResult r = handles[h].wait();
                    nlohmann::ordered_json line = {
                        {"step", step++}, {"engine", engine_index}, {"case", pair[h]},
                        {"concurrent", item}, {"spec", spec}, {"cache", cache},
                        {"prompt_tokens", r.prompt.prompt_tokens},
                        {"path", path_name(r.prefix_reuse_path)}, {"hit", r.reused_prompt_tokens},
                        {"out", r.generated_token_ids}, {"rounds", r.speculative.rounds},
                        {"drafted", r.speculative.drafted_tokens},
                        {"accepted", r.speculative.accepted_tokens},
                        {"app", r.speculative.accepted_per_position}};
                    std::cout << line.dump() << std::endl;
                }
                continue;
            }
            const bool raw  = item[0] == 'p';
            const auto idx  = static_cast<std::size_t>(std::atoi(item.c_str() + (raw ? 1 : 0)));
            const Case& c   = all.at(idx);
            ninferno::RequestOptions options;
            options.execution.requested_output_tokens = c.tokens;
            if (c.sampled) {
                options.execution.sampling.temperature = 0.9F;
                options.execution.sampling.top_p       = 0.95F;
                options.execution.sampling.seed        = 4242;
            } else {
                options.execution.sampling.temperature = 0.0F;
            }
            std::string text;
            if (!c.system.empty()) { text += "<|im_start|>system\n" + c.system + "<|im_end|>\n"; }
            text += "<|im_start|>user\n" + c.user + "<|im_end|>\n<|im_start|>assistant\n";
            if (c.thinking && !*c.thinking) { text += "<think>\n\n</think>\n\n"; }
            std::vector<ninferno::TokenId> ids = engine->tokenize_text(text);
            ninferno::PreparedPrompt prepared =
                raw ? engine->prepare_tokens(ids) : engine->prepare(input(c));
            ninferno::GenerationResult r = engine->generate(std::move(prepared), options);
            nlohmann::ordered_json line = {
                {"step", step++},
                {"engine", engine_index},
                {"case", idx},
                {"raw", raw},
                {"spec", spec},
                {"cache", cache},
                {"prompt_tokens", r.prompt.prompt_tokens},
                {"path", path_name(r.prefix_reuse_path)},
                {"hit", r.reused_prompt_tokens},
                {"out", r.generated_token_ids},
                {"rounds", r.speculative.rounds},
                {"drafted", r.speculative.drafted_tokens},
                {"accepted", r.speculative.accepted_tokens},
                {"app", r.speculative.accepted_per_position},
                {"prompt", ids},
            };
            std::cout << line.dump() << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "prefix_probe: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
