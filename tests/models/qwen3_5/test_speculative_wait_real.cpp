// How a speculative engine waits for its rounds does not change what it emits. Speculative rounds
// are timed and paced (round_pacing.h): a stalling host may block on a round's stop event rather
// than spin. An engine that blocks on every round (--decode-wait block, DecodePacingOptions::wait)
// emits exactly the tokens, finish reasons, stop strings and speculative rounds of one that spins
// (--decode-wait spin), under --activations exact, for:
// - greedy prose and code;
// - an output limit that no round is likely to end on;
// - a stop token and a stop string, chosen from the spinning engine's output;
// - two requests in one batch with different output limits;
// - a sampled request (seeded);
// - a follow-up turn that reuses a stopped conversation's prefix;
// - a request cancelled part way: its tokens are a prefix of the uncancelled output.
// Every uncancelled request must speculate (fewer rounds than tokens).
// NINFERNO_TEST_SPEC_WAIT_ARTIFACT names the artifact; without it the test skips.
// NINFERNO_TEST_SPEC_WAIT_BACKEND: mtp, dflash (the default) or dflash2; NINFERNO_TEST_SPEC_WAIT_K:
// the draft depth (default 3); NINFERNO_TEST_SPEC_WAIT_DRAFTER: a drafter shipped apart from the
// target.

#include "ninferno/engine.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninferno;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

constexpr std::uint32_t kContext = 4096;
constexpr std::uint32_t kOutputs = 96;

constexpr const char* kProse =
    "The history of the lighthouse is a history of the coast itself. Before the first towers "
    "were raised, sailors read the shore by its hills, its river mouths and the smoke of its "
    "villages, and a ship that arrived after dark waited offshore for the morning. Explain, in a "
    "few paragraphs, why the keepers' logbooks became valuable to historians of climate.";

constexpr const char* kCode =
    "Write a C function that parses a comma-separated line of unsigned integers into an array, "
    "reporting malformed input and overflow, and explain each branch of it.";

constexpr const char* kFollowUp = "\n\nNow say the same in two sentences.";

struct Outcome {
    std::vector<TokenId> tokens;
    std::string content;
    FinishReason finish = FinishReason::None;
    std::optional<std::string> stop;
    std::uint32_t reused = 0;
    std::uint64_t rounds = 0;
};

Outcome outcome(const GenerationResult& result) {
    return {.tokens  = result.generated_token_ids,
            .content = result.content,
            .finish  = result.finish_reason,
            .stop    = result.matched_stop_string,
            .reused  = result.reused_prompt_tokens,
            .rounds  = result.speculative.rounds};
}

// What the spinning engine's outputs choose for both engines.
struct Plan {
    std::optional<TokenId> stop_token;
    std::optional<std::string> stop_string;
};

std::string env(const char* name, const std::string& fallback) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != 0 ? std::string(value) : fallback;
}

SpeculativeBackend backend_under_test() {
    const std::string backend = env("NINFERNO_TEST_SPEC_WAIT_BACKEND", "dflash");
    if (backend == "mtp") { return SpeculativeBackend::Mtp; }
    if (backend == "dflash2") { return SpeculativeBackend::DFlash2; }
    if (backend != "dflash") {
        throw std::invalid_argument(
            "NINFERNO_TEST_SPEC_WAIT_BACKEND must be mtp, dflash or dflash2");
    }
    return SpeculativeBackend::DFlash;
}

std::uint32_t depth_under_test() {
    return static_cast<std::uint32_t>(std::stoul(env("NINFERNO_TEST_SPEC_WAIT_K", "3")));
}

EngineOptions engine_options(const std::string& artifact) {
    EngineOptions options;
    options.artifact_path   = artifact;
    options.max_context     = kContext;
    options.kv_capacity     = KvCapacityPolicy::explicit_capacity(2 * kContext);
    options.prefill_chunk   = 512;
    options.max_concurrency = 2;
    options.activations     = ActivationMode::Bits16;
    // An explicit plan: the Device checkpoints are the prefix cache's.
    options.context_cache.device_state_slots     = 2;
    options.context_cache.host_state_slots       = 0;
    options.context_cache.host_kv_capacity_bytes = 0;
    // The speculation under test.
    options.speculative.backend      = backend_under_test();
    options.speculative.draft_tokens = depth_under_test();
    options.speculative.drafter      = env("NINFERNO_TEST_SPEC_WAIT_DRAFTER", "");
    return options;
}

RequestOptions request(std::uint32_t outputs, bool reuse = false) {
    RequestOptions out;
    out.execution.requested_output_tokens = outputs;
    out.execution.sampling.temperature    = 0.0F;
    out.execution.allow_prefix_reuse      = reuse;
    out.stop.include_model_defaults       = true;
    return out;
}

class CountingSink final : public OutputSink {
public:
    void start(GenerationStart) override {}
    void progress(PromptProgress) override {}
    void timing(GenerationTimingObservation) override {}
    void publish(OutputDelta) override { deltas.fetch_add(1, std::memory_order_relaxed); }
    std::atomic<int> deltas{0};
};

// Every case in one engine; `plan` is filled from the outputs where it is empty (the spinning
// engine) and read otherwise.
std::map<std::string, Outcome> run(const std::string& artifact, DecodeWait wait, Plan& plan) {
    EngineOptions options = engine_options(artifact);
    options.pacing.wait   = wait;
    Engine engine(options);
    std::map<std::string, Outcome> out;
    const std::vector<TokenId> prose = engine.tokenize_text(kProse);
    const std::vector<TokenId> code  = engine.tokenize_text(kCode);

    const auto generate = [&](const std::vector<TokenId>& prompt, RequestOptions options) {
        return outcome(engine.generate(engine.prepare_tokens(prompt), std::move(options)));
    };

    out["prose"] = generate(prose, request(kOutputs, true));
    out["code"]  = generate(code, request(kOutputs));
    out["limit"] = generate(prose, request(37));

    if (!plan.stop_token) {
        // A token past the first few that appears nowhere before it.
        const std::vector<TokenId>& tokens = out["prose"].tokens;
        for (std::size_t i = 12; i < tokens.size() && !plan.stop_token; ++i) {
            if (std::find(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(i),
                          tokens[i]) == tokens.begin() + static_cast<std::ptrdiff_t>(i)) {
                plan.stop_token = tokens[i];
            }
        }
    }
    if (plan.stop_token) {
        RequestOptions options = request(kOutputs);
        options.stop.token_ids = {*plan.stop_token};
        out["stop_token"]      = generate(prose, std::move(options));
    }
    if (!plan.stop_string) {
        // Five printable characters from the middle of the output, found nowhere before.
        const std::string& text = out["prose"].content;
        for (std::size_t at = text.size() / 2; at + 5 <= text.size() && !plan.stop_string; ++at) {
            const std::string candidate = text.substr(at, 5);
            const bool ascii            = std::all_of(candidate.begin(), candidate.end(),
                                                      [](char c) { return c > 0x1f && c < 0x7f; });
            if (ascii && text.find(candidate) == at) { plan.stop_string = candidate; }
        }
    }
    if (plan.stop_string) {
        RequestOptions options = request(kOutputs, true);
        options.stop.strings   = {StopString{.text = *plan.stop_string}};
        out["stop_string"]     = generate(prose, std::move(options));
    }

    {
        // Two rows in one batch; the shorter one finishes while the longer one decodes.
        auto first  = engine.submit(engine.prepare_tokens(prose), request(29));
        auto second = engine.submit(engine.prepare_tokens(code), request(kOutputs + 3));
        out["concurrent_prose"] = outcome(first.wait());
        out["concurrent_code"]  = outcome(second.wait());
    }

    {
        RequestOptions options                 = request(kOutputs);
        options.execution.sampling.temperature = 0.8F;
        options.execution.sampling.seed        = 7;
        out["sampled"]                         = generate(code, std::move(options));
    }

    if (out.count("stop_string")) {
        // The stopped conversation's follow-up reuses its prefix.
        std::vector<TokenId> follow_up   = prose;
        const std::vector<TokenId>& turn = out["stop_string"].tokens;
        follow_up.insert(follow_up.end(), turn.begin(), turn.end());
        const std::vector<TokenId> tail = engine.tokenize_text(kFollowUp);
        follow_up.insert(follow_up.end(), tail.begin(), tail.end());
        out["follow_up"] = generate(follow_up, request(kOutputs, true));
    }

    {
        CountingSink sink;
        const CancellationView cancel([&sink] { return sink.deltas.load() >= 8; });
        out["cancelled"] = outcome(
            engine.generate(engine.prepare_tokens(code), request(4 * kOutputs), &sink, cancel));
    }
    return out;
}

void compare(const std::string& model, const std::map<std::string, Outcome>& spin,
             const std::map<std::string, Outcome>& block) {
    for (const auto& [name, reference] : spin) {
        const std::string label = model + " " + name;
        const auto found        = block.find(name);
        if (found == block.end()) {
            check(false, label + ": the blocking engine did not run the case");
            continue;
        }
        const Outcome& other = found->second;
        std::cout << label << ": " << reference.tokens.size() << " tokens in " << reference.rounds
                  << " rounds spinning; " << other.tokens.size() << " in " << other.rounds
                  << " blocking\n";
        if (name == "cancelled") {
            // Cancellation lands wherever the host sees it: each output is a prefix of the
            // uncancelled one.
            const std::vector<TokenId>& full = spin.at("code").tokens;
            for (const Outcome* result : {&reference, &other}) {
                const auto common =
                    static_cast<std::ptrdiff_t>(std::min(result->tokens.size(), full.size()));
                check(result->finish == FinishReason::Cancelled &&
                          std::equal(full.begin(), full.begin() + common, result->tokens.begin()),
                      label + ": the cancelled output is not a prefix of the uncancelled one");
            }
            continue;
        }
        check(other.tokens == reference.tokens && other.finish == reference.finish &&
                  other.stop == reference.stop,
              label + ": the blocking engine's output differs from the spinning engine's");
        // A solo request's rounds follow from its drafts and acceptance alone.
        if (!name.starts_with("concurrent")) {
            check(other.rounds == reference.rounds,
                  label + ": the blocking engine ran another number of rounds");
        }
        check(reference.rounds > 0 && reference.rounds < reference.tokens.size(),
              label + ": the request did not speculate");
    }
}

} // namespace

int main() {
    const std::string artifact = env("NINFERNO_TEST_SPEC_WAIT_ARTIFACT", "");
    if (artifact.empty()) {
        std::cout << "SKIP: set NINFERNO_TEST_SPEC_WAIT_ARTIFACT\n";
        return 77;
    }
    std::cout << std::unitbuf; // each result as it comes (a timed-out job keeps what ran)
    try {
        const std::string model = artifact.substr(artifact.find_last_of('/') + 1);
        Plan plan;
        const auto spin  = run(artifact, DecodeWait::Spin, plan);
        const auto block = run(artifact, DecodeWait::Block, plan);
        check(plan.stop_token && plan.stop_string,
              model + ": no stop token or stop string was found in the output");
        check(spin.count("stop_token") && spin.at("stop_token").finish == FinishReason::StopToken,
              model + ": the stop token did not stop the output");
        check(spin.count("stop_string") &&
                  spin.at("stop_string").finish == FinishReason::StopString,
              model + ": the stop string did not stop the output");
        check(spin.count("follow_up") && spin.at("follow_up").reused != 0 &&
                  block.count("follow_up") && block.at("follow_up").reused != 0,
              model + ": the follow-up did not reuse the stopped conversation's prefix");
        compare(model, spin, block);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "speculative wait real: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
