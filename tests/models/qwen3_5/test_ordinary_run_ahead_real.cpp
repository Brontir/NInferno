// Ordinary decode run-ahead through the Engine (ninferno-serve's path) on real artifacts whose
// executed but uncommitted rows stay invisible (Gemma 4, dense decoders with sliding rings). An
// engine that runs eight steps per round (NINFERNO_DECODE_STEPS=8) emits exactly the tokens, finish
// reasons and stop strings of one that runs one step per round (NINFERNO_DECODE_STEPS=1), under
// --activations exact (whose kernels give the same result at every batch size), for:
// - a stop token in the middle of a round (the path an end-of-sequence token takes);
// - a stop string whose text spans two rounds;
// - output limits at a round's end and inside a round;
// - two requests in one batch with different output limits;
// - a sampled request (seeded);
// - a follow-up turn that forks from a conversation which stopped in the middle of a round, after
//   a prompt longer than the sliding window (the rows past the stop wrapped the rings);
// - a request cancelled in the middle of its rounds: its tokens are a prefix of the uncancelled
//   output, and the same prompt runs again as before.
// The eight-step engine must run fewer rounds than tokens. The stop token and the stop string are
// chosen from the one-step engine's output so that they fall where the rounds put them.
// NINFERNO_TEST_RUN_AHEAD_ARTIFACTS names the artifacts, separated by ':'; without it the test
// skips.

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
#include <sstream>
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

constexpr std::uint32_t kContext = 3072;
constexpr std::uint32_t kSteps   = 8;
constexpr std::uint32_t kOutputs = 64;

constexpr const char* kProse =
    "The history of the lighthouse is a history of the coast itself. Before the first towers "
    "were raised, sailors read the shore by its hills, its river mouths and the smoke of its "
    "villages, and a ship that arrived after dark waited offshore for the morning. The earliest "
    "lights were open fires kept on headlands by monks or by the harbour masters of trading "
    "towns, and their keepers were paid in part by a toll on every cargo that came in safely. ";

constexpr const char* kQuestion =
    "Explain, in a few paragraphs, why the keepers' logbooks became valuable to historians of "
    "climate, and what their limits are as a source.";

constexpr const char* kCode =
    "Write a C function that parses a comma-separated line of unsigned integers into an array, "
    "reporting malformed input and overflow, and explain each branch of it.";

constexpr const char* kFollowUp =
    "\n\nNow compare those logbooks with the records ships kept at sea over the same years.";

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
            .rounds  = result.engine_timing.decode_rounds};
}

// What the one-step engine's outputs choose for both engines.
struct Plan {
    std::optional<TokenId> stop_token;
    std::optional<std::string> stop_string;
    std::optional<TokenId> long_stop_token;
};

EngineOptions engine_options(const std::string& artifact) {
    EngineOptions options;
    options.artifact_path   = artifact;
    options.max_context     = kContext;
    options.kv_capacity     = KvCapacityPolicy::explicit_capacity(2 * kContext);
    options.prefill_chunk   = 512;
    options.max_concurrency = 2;
    options.activations     = ActivationMode::Exact;
    // An explicit plan: the Device checkpoints are the prefix cache's.
    options.context_cache.device_state_slots     = 2;
    options.context_cache.host_state_slots       = 0;
    options.context_cache.host_kv_capacity_bytes = 0;
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

// Whether output token i is not the last of its round: token 0 comes from the prefill, and round
// r (from 1) produces tokens kSteps * (r - 1) + 1 to kSteps * r.
bool mid_round(std::size_t i) { return i != 0 && i % kSteps != 0; }

// The first token at an index in [first, last) that ends no round and appears nowhere before it.
std::optional<TokenId> mid_round_token(const std::vector<TokenId>& tokens, std::size_t first,
                                       std::size_t last) {
    for (std::size_t i = first; i < std::min(last, tokens.size()); ++i) {
        if (!mid_round(i) || i % kSteps < 2) { continue; }
        if (std::find(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(i),
                      tokens[i]) == tokens.begin() + static_cast<std::ptrdiff_t>(i)) {
            return tokens[i];
        }
    }
    return std::nullopt;
}

class CountingSink final : public OutputSink {
public:
    void start(GenerationStart) override {}
    void progress(PromptProgress) override {}
    void timing(GenerationTimingObservation) override {}
    void publish(OutputDelta) override { deltas.fetch_add(1, std::memory_order_relaxed); }
    std::atomic<int> deltas{0};
};

// Every case in one engine; `plan` is filled from the outputs where it is empty (the one-step
// engine) and read otherwise.
std::map<std::string, Outcome> run(const std::string& artifact, const char* steps, Plan& plan) {
    ::setenv("NINFERNO_DECODE_STEPS", steps, 1);
    Engine engine(engine_options(artifact));
    std::map<std::string, Outcome> out;
    const std::vector<TokenId> prose = engine.tokenize_text(std::string(kProse) + kQuestion);
    const std::vector<TokenId> code  = engine.tokenize_text(kCode);
    std::string text;
    for (int i = 0; i < 24; ++i) { text += kProse; }
    const std::vector<TokenId> longer = engine.tokenize_text(text + kQuestion);

    const auto generate = [&](const std::vector<TokenId>& prompt, RequestOptions options) {
        return outcome(engine.generate(engine.prepare_tokens(prompt), std::move(options)));
    };

    out["prose"]              = generate(prose, request(kOutputs));
    out["limit_round_end"]    = generate(prose, request(1 + 3 * kSteps));
    out["limit_inside_round"] = generate(prose, request(3 * kSteps - 2));

    if (!plan.stop_token) { plan.stop_token = mid_round_token(out["prose"].tokens, 10, kOutputs); }
    if (plan.stop_token) {
        RequestOptions options = request(kOutputs);
        options.stop.token_ids = {*plan.stop_token};
        out["stop_token"]      = generate(prose, std::move(options));
    }

    if (!plan.stop_string) {
        // The text of the first 8k + 1 tokens ends a round; the next token starts the next one.
        for (std::uint32_t k = 1; k < kOutputs / kSteps - 1 && !plan.stop_string; ++k) {
            const std::string before = generate(prose, request(kSteps * k + 1)).content;
            const std::string after  = generate(prose, request(kSteps * k + 2)).content;
            if (before.size() < 3 || after.size() < before.size() + 2 ||
                after.compare(0, before.size(), before) != 0) {
                continue;
            }
            const std::string candidate = after.substr(before.size() - 3, 5);
            const bool ascii            = std::all_of(candidate.begin(), candidate.end(),
                                                      [](char c) { return c > 0x1f && c < 0x7f; });
            if (ascii && out["prose"].content.find(candidate) == before.size() - 3) {
                plan.stop_string = candidate;
            }
        }
    }
    if (plan.stop_string) {
        RequestOptions options = request(kOutputs);
        options.stop.strings   = {StopString{.text = *plan.stop_string}};
        out["stop_string"]     = generate(prose, std::move(options));
    }

    {
        // Two rows in one batch; the shorter limit ends inside a round of the longer one.
        auto first  = engine.submit(engine.prepare_tokens(prose), request(3 * kSteps + 4));
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

    // A conversation longer than the sliding window stops in the middle of a round; its follow-up
    // forks from the cached conversation.
    out["long"] = generate(longer, request(kOutputs, true));
    if (!plan.long_stop_token) {
        plan.long_stop_token = mid_round_token(out["long"].tokens, 10, kOutputs);
    }
    if (plan.long_stop_token) {
        RequestOptions options = request(kOutputs, true);
        options.stop.token_ids = {*plan.long_stop_token};
        out["long_stop"]       = generate(longer, std::move(options));

        std::vector<TokenId> follow_up   = longer;
        const std::vector<TokenId>& turn = out["long_stop"].tokens;
        follow_up.insert(follow_up.end(), turn.begin(), turn.end());
        const std::vector<TokenId> tail = engine.tokenize_text(kFollowUp);
        follow_up.insert(follow_up.end(), tail.begin(), tail.end());
        out["long_follow_up"] = generate(follow_up, request(kOutputs, true));
    }

    {
        // Cancelled after a few published deltas; then the same prompt again.
        CountingSink sink;
        const CancellationView cancel([&sink] { return sink.deltas.load() >= 12; });
        out["cancelled"] = outcome(engine.generate(engine.prepare_tokens(longer),
                                                   request(4 * kOutputs), &sink, cancel));
        out["after_cancel"] = generate(longer, request(kOutputs, true));
    }
    return out;
}

const char* finish_name(FinishReason reason) {
    switch (reason) {
    case FinishReason::None:
        return "none";
    case FinishReason::OutputLimit:
        return "length";
    case FinishReason::ContextCapacity:
        return "context";
    case FinishReason::StopToken:
        return "stop token";
    case FinishReason::StopString:
        return "stop string";
    case FinishReason::Cancelled:
        return "cancelled";
    }
    return "?";
}

void compare(const std::string& model, const std::map<std::string, Outcome>& one,
             const std::map<std::string, Outcome>& eight) {
    for (const auto& [name, reference] : one) {
        const std::string label = model + " " + name;
        const auto found        = eight.find(name);
        if (found == eight.end()) {
            check(false, label + ": the eight-step engine did not run the case");
            continue;
        }
        const Outcome& other = found->second;
        std::cout << label << ": " << reference.tokens.size() << " tokens ("
                  << finish_name(reference.finish) << ") in " << reference.rounds << " rounds; "
                  << other.tokens.size() << " (" << finish_name(other.finish) << ") in "
                  << other.rounds << " rounds at eight steps\n";
        if (name == "cancelled") {
            // Cancellation lands wherever the host sees it: each output is a prefix of the
            // uncancelled one.
            const std::vector<TokenId>& full = one.at("long").tokens;
            for (const Outcome* result : {&reference, &other}) {
                const auto common = static_cast<std::ptrdiff_t>(
                    std::min(result->tokens.size(), full.size()));
                check(result->finish == FinishReason::Cancelled &&
                          std::equal(full.begin(), full.begin() + common, result->tokens.begin()),
                      label + ": the cancelled output is not a prefix of the uncancelled one");
            }
            continue;
        }
        check(other.tokens == reference.tokens && other.finish == reference.finish &&
                  other.stop == reference.stop,
              label + ": the eight-step engine's output differs from the one-step engine's");
        if (other.finish == FinishReason::OutputLimit && other.tokens.size() > 2 * kSteps) {
            check(other.rounds * 2 < other.tokens.size(),
                  label + ": the eight-step engine did not run several steps per round");
        }
    }
}

std::vector<std::string> split(const std::string& text) {
    std::vector<std::string> out;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        if (!item.empty()) { out.push_back(item); }
    }
    return out;
}

} // namespace

int main() {
    const char* artifacts = std::getenv("NINFERNO_TEST_RUN_AHEAD_ARTIFACTS");
    if (artifacts == nullptr || *artifacts == 0) {
        std::cout << "SKIP: set NINFERNO_TEST_RUN_AHEAD_ARTIFACTS\n";
        return 77;
    }
    std::cout << std::unitbuf; // each result as it comes (a timed-out job keeps what ran)
    try {
        for (const std::string& artifact : split(artifacts)) {
            const std::string model = artifact.substr(artifact.find_last_of('/') + 1);
            Plan plan;
            const auto one   = run(artifact, "1", plan);
            const auto eight = run(artifact, "8", plan);
            check(plan.stop_token && plan.stop_string && plan.long_stop_token,
                  model + ": no stop token or stop string was found where the rounds put them");
            check(one.count("stop_token") &&
                      one.at("stop_token").finish == FinishReason::StopToken,
                  model + ": the stop token did not stop the output");
            check(one.count("stop_string") &&
                      one.at("stop_string").finish == FinishReason::StopString,
                  model + ": the stop string did not stop the output");
            check(one.count("long_follow_up") && one.at("long_follow_up").reused != 0 &&
                      eight.count("long_follow_up") && eight.at("long_follow_up").reused != 0,
                  model + ": the follow-up did not fork from the cached conversation");
            check(one.at("after_cancel").tokens == one.at("long").tokens,
                  model + ": after a cancellation the prompt's output changed");
            compare(model, one, eight);
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "ordinary run-ahead real: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
