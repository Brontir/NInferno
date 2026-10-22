// DFlash speculation through the Engine (ninferno-serve's path) on a real dense artifact and its
// published DFlash drafter. Under --activations exact a greedy speculative request emits exactly
// the tokens plain decode emits (a verify column's bytes are a decode step's), at every depth,
// with the model's stop tokens and prefix reuse on: after a prompt prefilled in chunks, after one
// whose chunk is wider than the drafter's sliding window, from a checkpoint of an earlier request
// (the target's and the drafter's rings forked; against a plain engine's fork), and with four
// requests of different depths and a plain one sharing launches, and with eight rows at K=7 and
// K=3 decoding through every frontier range at batch 8. Each greedy case keeps a floor on
// its tokens per round. A sampled request (temperature > 0) runs the sampling accept: its first
// token is plain decode's draw, and the same seed repeats its output. The fast engines' agreement
// is reported.
// NINFERNO_TEST_DENSE_DFLASH_ARTIFACT names the target, NINFERNO_TEST_DENSE_DFLASH_DRAFTER its
// drafter; without both the test skips. NINFERNO_TEST_DENSE_DFLASH_DEPTH is the deepest window (15
// when unset; 7 for gpt-oss's drafter, whose block holds 8 columns); the middle depth is at most 7.

#include "ninferno/engine.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
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
// The tokens per round a greedy case must keep (Muse Glimmer's drafter keeps 2.3 to 3.3 on these
// prompts): a drafter whose context or block pass is wrong accepts next to nothing.
constexpr double kGreedyFloor = 2.0;
// A sampled request accepts a draft with the target's probability of it.
constexpr double kSampledFloor = 1.3;
// A follow-up forked from a cached conversation (its own text: its acceptance is its own).
constexpr double kForkFloor   = 1.5;
// The fast engines' prose (their verify may take other activation routes than a decode step):
// 2.5 tokens per round at K=15 and 2.32 at K=3 on the GGUF target (RTX 4090), 1.92 and 1.81
// on the NVFP4 one (RTX 5090, whose exact prose keeps 2.14); a broken drafter falls far below.
constexpr double kFastFloor   = 1.7;
constexpr float kTemperature  = 0.7F;
constexpr std::uint64_t kSeed = 1234;
// The deepest window the drafter's block holds, and the middle one (the environment sets it).
std::uint32_t deep   = 15;
std::uint32_t middle = 7;

constexpr const char* kProse =
    "The history of the lighthouse is a history of the coast itself. Before the first towers "
    "were raised, sailors read the shore by its hills, its river mouths and the smoke of its "
    "villages, and a ship that arrived after dark waited offshore for the morning. The earliest "
    "lights were open fires kept on headlands by monks or by the harbour masters of trading "
    "towns, and their keepers were paid in part by a toll on every cargo that came in safely. "
    "Coal replaced wood, oil replaced coal, and in the nineteenth century the Fresnel lens "
    "gathered the light of a single lamp into a beam that could be seen from twenty miles away. "
    "Each tower was given its own rhythm of flashes so that a navigator could tell one from "
    "another, and the lists of those rhythms were printed in almanacs next to the tides. ";

constexpr const char* kQuestion =
    "Explain, in a few paragraphs, why the keepers' logbooks became valuable to historians of "
    "climate, and what their limits are as a source.";

constexpr const char* kFollowUp =
    "\n\nNow compare those logbooks with the records ships kept at sea over the same years.";

constexpr const char* kCode =
    "Write a C function that parses a comma-separated line of unsigned integers into an array, "
    "reporting malformed input and overflow, and explain each branch of it.";

EngineOptions engine_options(const char* artifact, ActivationMode activations,
                             std::uint32_t prefill_chunk, std::uint32_t concurrency,
                             std::uint32_t cached_states) {
    EngineOptions options;
    options.artifact_path = artifact;
    options.max_context   = kContext;
    // Pages for every row's context, at most 6144 tokens (the 24 GB card holds the target beside
    // its drafter).
    options.kv_capacity =
        KvCapacityPolicy::explicit_capacity(std::min(concurrency * kContext, 6144U));
    options.prefill_chunk   = prefill_chunk;
    options.max_concurrency = concurrency;
    options.activations     = activations;
    // An explicit plan (the cached states too): no automatic fit, so no kernel warm-up of every
    // prefill width up to the chunk. The Device checkpoints are the prefix cache's.
    options.context_cache.device_state_slots     = cached_states;
    options.context_cache.host_state_slots       = 0;
    options.context_cache.host_kv_capacity_bytes = 0;
    return options;
}

EngineOptions dflash_options(EngineOptions options, const char* drafter) {
    options.speculative.backend              = SpeculativeBackend::DFlash;
    options.speculative.drafter              = drafter;
    options.speculative.draft_tokens         = deep;
    options.speculative.request_draft_tokens = {3, middle};
    return options;
}

// The model's stop tokens stay on; `reuse` lets the request fork from a cached prefix.
RequestOptions request(std::optional<std::uint32_t> depth, bool reuse, bool sampled = false) {
    RequestOptions out;
    out.execution.requested_output_tokens = kOutputs;
    out.execution.sampling.temperature    = sampled ? kTemperature : 0.0F;
    if (sampled) { out.execution.sampling.seed = kSeed; }
    out.execution.allow_prefix_reuse = reuse;
    out.stop.include_model_defaults  = true;
    if (depth) { out.execution.speculation = RequestSpeculation{.draft_tokens = *depth}; }
    return out;
}

std::size_t common_prefix(const std::vector<TokenId>& a, const std::vector<TokenId>& b) {
    std::size_t same = 0;
    while (same < a.size() && same < b.size() && a[same] == b[same]) { ++same; }
    return same;
}

double tokens_per_round(const SpeculativeStats& stats) {
    return stats.rounds == 0 ? 0.0
                             : double(stats.rounds + stats.accepted_tokens) / double(stats.rounds);
}

// A speculative result against its plain counterpart: equal tokens and finish when `exact`,
// reported always; its tokens per round at least `floor`, and its counters whole.
void compare(const std::string& label, const GenerationResult& plain,
             const GenerationResult& speculative, bool exact, double floor = kGreedyFloor) {
    const SpeculativeStats& s = speculative.speculative;
    const std::size_t same =
        common_prefix(plain.generated_token_ids, speculative.generated_token_ids);
    std::cout << label << ": " << speculative.generated_token_ids.size() << " tokens, the first "
              << same << " equal to plain decode's; " << s.rounds << " rounds ("
              << s.wide_rounds << " wide), " << tokens_per_round(s) << " tokens per round; "
              << speculative.reused_prompt_tokens << " prompt tokens reused\n";
    check(s.backend == SpeculativeBackend::DFlash && s.rounds != 0,
          label + ": the request did not speculate");
    check(!s.watchdog_tripped && s.plain_steps == 0, label + ": the request left speculation");
    // A round drafts no more than the output has room for: only the last token may be a round
    // without drafts (a fallback step).
    check(s.fallback_steps <= 1, label + ": " + std::to_string(s.fallback_steps) +
                                     " rounds without drafts");
    check(tokens_per_round(s) >= floor, label + ": " + std::to_string(tokens_per_round(s)) +
                                            " tokens per round, below the floor of " +
                                            std::to_string(floor));
    // The anchor from the prefill, then each round's accepted drafts and its correction; a stop
    // token may end the output inside the last round.
    const std::uint64_t accounted = 1 + s.rounds + s.accepted_tokens + s.fallback_steps;
    const std::size_t produced    = speculative.generated_token_ids.size();
    check(produced != 0 && (speculative.finish_reason == FinishReason::OutputLimit
                                ? accounted == produced
                                : accounted >= produced),
          label + ": the rounds do not account for the output");
    if (exact) {
        check(!plain.generated_token_ids.empty() &&
                  speculative.generated_token_ids == plain.generated_token_ids &&
                  speculative.finish_reason == plain.finish_reason,
              label + ": the speculative tokens differ from plain greedy decode's");
    }
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFERNO_TEST_DENSE_DFLASH_ARTIFACT");
    const char* drafter  = std::getenv("NINFERNO_TEST_DENSE_DFLASH_DRAFTER");
    if (artifact == nullptr || *artifact == 0 || drafter == nullptr || *drafter == 0) {
        std::cout << "SKIP: set NINFERNO_TEST_DENSE_DFLASH_ARTIFACT and "
                     "NINFERNO_TEST_DENSE_DFLASH_DRAFTER\n";
        return 77;
    }
    if (const char* depth = std::getenv("NINFERNO_TEST_DENSE_DFLASH_DEPTH"); depth && *depth) {
        deep   = static_cast<std::uint32_t>(std::stoul(depth));
        middle = std::min(middle, deep);
    }
    std::cout << std::unitbuf; // each result as it comes (a timed-out job keeps what ran)
    const std::string kd = "K=" + std::to_string(deep);
    const std::string km = "K=" + std::to_string(middle);
    try {
        std::vector<TokenId> prose, code, longer, follow_up;
        GenerationResult plain_prose, plain_code, plain_longer, plain_follow_up, plain_sampled;
        {
            // Plain decode, prefilled in narrow chunks, no prefix reuse (exact: neither the cut
            // nor a fork changes a token).
            Engine engine(engine_options(artifact, ActivationMode::Bits16, 128, 1, 1));
            prose = engine.tokenize_text(std::string(kProse) + kQuestion);
            code  = engine.tokenize_text(kCode);
            std::string text;
            for (int i = 0; i < 18; ++i) { text += kProse; }
            longer = engine.tokenize_text(text + kQuestion);
            const auto plain = [&](const std::vector<TokenId>& prompt, bool sampled = false) {
                return engine.generate(engine.prepare_tokens(prompt, false),
                                       request({}, false, sampled));
            };
            plain_prose   = plain(prose);
            plain_code    = plain(code);
            plain_longer  = plain(longer);
            plain_sampled = plain(prose, true);
            // The prose request's turn, its answer and a question after it.
            follow_up = prose;
            follow_up.insert(follow_up.end(), plain_prose.generated_token_ids.begin(),
                             plain_prose.generated_token_ids.end());
            const std::vector<TokenId> tail = engine.tokenize_text(kFollowUp);
            follow_up.insert(follow_up.end(), tail.begin(), tail.end());
            plain_follow_up = plain(follow_up);
        }
        std::cout << "prompts: " << prose.size() << ", " << code.size() << ", " << longer.size()
                  << " and " << follow_up.size() << " tokens\n";
        check(longer.size() > 2560 && longer.size() + kOutputs < kContext,
              "the long prompt must pass the 2560-token chunk and fit the context");
        check(plain_prose.generated_token_ids.size() == kOutputs &&
                  plain_code.generated_token_ids.size() == kOutputs,
              "the reference requests must run to the output limit");
        // A follow-up turn forks from the conversation its first turn cached (the prose prompt and
        // its answer): the target's rings and pages, and the drafter's rings, come from the
        // checkpoint. The answer's keys and values were written by decode calls, not by a prefill,
        // so the reference is a plain engine's fork, not plain_follow_up. Each pair runs in an
        // engine of its own; nullopt is a plain engine's request.
        const auto fork = [&](std::optional<std::uint32_t> first, std::optional<std::uint32_t> then,
                              bool speculative_engine) {
            EngineOptions options = engine_options(artifact, ActivationMode::Bits16, 2560, 1, 1);
            if (speculative_engine) { options = dflash_options(std::move(options), drafter); }
            Engine engine(std::move(options));
            const GenerationResult turn =
                engine.generate(engine.prepare_tokens(prose), request(first, true));
            GenerationResult follow =
                engine.generate(engine.prepare_tokens(follow_up), request(then, true));
            check(turn.generated_token_ids == plain_prose.generated_token_ids,
                  "exact fork: the first turn differs from plain decode's");
            check(follow.reused_prompt_tokens != 0,
                  "exact fork: the follow-up did not fork from the cached conversation");
            return follow;
        };
        const GenerationResult plain_fork = fork(std::nullopt, std::nullopt, false);
        std::cout << "exact fork plain: " << plain_fork.reused_prompt_tokens
                  << " prompt tokens reused, the first "
                  << common_prefix(plain_follow_up.generated_token_ids,
                                   plain_fork.generated_token_ids)
                  << " equal to the unforked follow-up's\n";
        {
            const GenerationResult follow = fork(deep, 0U, true);
            std::cout << "exact fork " << kd << " -> plain: the first "
                      << common_prefix(plain_fork.generated_token_ids, follow.generated_token_ids)
                      << " equal to the plain engine's fork\n";
            check(follow.generated_token_ids == plain_fork.generated_token_ids,
                  "exact fork " + kd + " -> plain: the follow-up differs from the plain engine's fork");
        }
        // The drafter's rings at the checkpoint lack the last round's accepted columns (a round
        // appends the previous round's), so its first rounds draft on a slightly shorter context.
        compare("exact fork " + kd + " -> " + kd, plain_fork, fork(deep, deep, true), true,
                kForkFloor);
        {
            // Four rows. A chunk wider than the drafter's 2048-token window: the long prompt's
            // first chunk commits only its last window columns to the drafter's rings.
            Engine engine(dflash_options(
                engine_options(artifact, ActivationMode::Bits16, 2560, 4, 2), drafter));
            compare("exact prose " + kd, plain_prose,
                    engine.generate(engine.prepare_tokens(prose), request(deep, true)), true);
            compare("exact code " + km, plain_code,
                    engine.generate(engine.prepare_tokens(code), request(middle, true)), true);
            compare("exact long " + kd, plain_longer,
                    engine.generate(engine.prepare_tokens(longer), request(deep, false)),
                    true);
            // Four requests at once, a plain one among them: the shallower rows draft at their
            // launch's width.
            auto deepest = engine.submit(engine.prepare_tokens(prose), request(deep, false));
            auto shallow = engine.submit(engine.prepare_tokens(code), request(3, false));
            auto between = engine.submit(engine.prepare_tokens(longer), request(middle, true));
            auto none    = engine.submit(engine.prepare_tokens(prose), request(0, false));
            const GenerationResult deep_result    = deepest.wait();
            const GenerationResult shallow_result = shallow.wait();
            const GenerationResult middle_result  = between.wait();
            const GenerationResult none_result    = none.wait();
            compare("exact concurrent prose " + kd, plain_prose, deep_result, true);
            compare("exact concurrent code K=3", plain_code, shallow_result, true);
            compare("exact concurrent long " + km, plain_longer, middle_result, true);
            check(none_result.generated_token_ids == plain_prose.generated_token_ids &&
                      none_result.speculative.rounds == 0,
                  "exact concurrent plain: the request speculated or its tokens differ");
            check(shallow_result.speculative.wide_rounds != 0,
                  "exact concurrent code K=3 never shared a wider launch");
            // The sampling accept: the first token is the prefill's draw, as plain decode's; the
            // same seed repeats the output (the second from the cached prefix).
            const GenerationResult sampled =
                engine.generate(engine.prepare_tokens(prose), request(middle, false, true));
            const GenerationResult again =
                engine.generate(engine.prepare_tokens(prose), request(middle, true, true));
            compare("exact sampled prose " + km, plain_sampled, sampled, false, kSampledFloor);
            compare("exact sampled prose " + km + " again", plain_sampled, again, false,
                    kSampledFloor);
            check(!sampled.generated_token_ids.empty() &&
                      !plain_sampled.generated_token_ids.empty() &&
                      sampled.generated_token_ids[0] == plain_sampled.generated_token_ids[0],
                  "exact sampled: the first token is not plain decode's draw");
            check(sampled.generated_token_ids == again.generated_token_ids,
                  "exact sampled: the same seed did not repeat the output");
        }
        {
            // Eight rows (C=8, as ninferno-serve --concurrency 8 runs them), K=7: at 6+ rows the
            // drafter's sliding attention over its 16-column block stays direct through 128 keys,
            // so the (96, 128] frontier range captures another topology than the ranges around
            // it; the graph family must not update one onto the other (it did: the engine threw
            // cudaErrorGraphExecUpdateFailure at start, and the pending error then aborted the next
            // launch). The short code prompt decodes through that range at batch 8, the prose
            // prompt past it; two code rows at K=3 share the K=7 launches.
            Engine engine(dflash_options(
                engine_options(artifact, ActivationMode::Bits16, 512, 8, 1), drafter));
            for (const auto& [label, prompt, reference] :
                 {std::tuple{"code", &code, &plain_code},
                  std::tuple{"prose", &prose, &plain_prose}}) {
                std::vector<GenerationHandle> handles;
                std::vector<std::uint32_t> depths;
                for (std::uint32_t row = 0; row < 8; ++row) {
                    depths.push_back(row % 4 == 3 ? 3U : 7U);
                    handles.push_back(engine.submit(engine.prepare_tokens(*prompt, false),
                                                    request(depths.back(), false)));
                }
                for (std::size_t row = 0; row < handles.size(); ++row) {
                    compare(std::string("exact C=8 ") + label + " row " + std::to_string(row) +
                                " K=" + std::to_string(depths[row]),
                            *reference, handles[row].wait(), true);
                }
            }
        }
        {
            // The drafter's width (SpeculativeOptions::draft_width) changes the drafts, never the
            // tokens: the defaults above ran the whole block in one-row rounds and K + 1 columns
            // in wider ones; here one-row rounds draft K + 1 columns and four-row rounds the block.
            EngineOptions options = dflash_options(
                engine_options(artifact, ActivationMode::Bits16, 2560, 4, 1), drafter);
            options.speculative.draft_width = DFlashDraftWidth::Drafts;
            Engine narrow(options);
            const GenerationResult result =
                narrow.generate(narrow.prepare_tokens(prose), request(7, false));
            compare("exact prose K=7, drafts width", plain_prose, result, true);
            check(result.speculative.narrow_rounds == result.speculative.rounds,
                  "exact prose K=7, drafts width: a round drafted the whole block");
        }
        {
            EngineOptions options = dflash_options(
                engine_options(artifact, ActivationMode::Bits16, 2560, 4, 1), drafter);
            options.speculative.draft_width = DFlashDraftWidth::Block;
            Engine engine(options);
            auto prose_row = engine.submit(engine.prepare_tokens(prose), request(7, false));
            auto code_row  = engine.submit(engine.prepare_tokens(code), request(3, false));
            auto long_row  = engine.submit(engine.prepare_tokens(longer), request(deep, false));
            const GenerationResult prose_result = prose_row.wait();
            const GenerationResult code_result  = code_row.wait();
            const GenerationResult long_result  = long_row.wait();
            compare("exact concurrent prose K=7, block width", plain_prose, prose_result, true);
            compare("exact concurrent code K=3, block width", plain_code, code_result, true);
            compare("exact concurrent long " + kd + ", block width", plain_longer, long_result, true);
            check(prose_result.speculative.narrow_rounds == 0 &&
                      code_result.speculative.narrow_rounds == 0 &&
                      long_result.speculative.narrow_rounds == 0,
                  "exact concurrent, block width: a round drafted K + 1 columns only");
        }
        // Under the default widths a request's rounds switch width with its launch's rows: the long
        // row drafts K + 1 columns beside the brief one, the whole block alone once the brief one
        // finished, and K + 1 columns again beside the late one; the late one the block again once
        // the long one finished. Every output is plain decode's, with the rounds' graphs and
        // without them.
        for (const bool graphs : {true, false}) {
            EngineOptions options = dflash_options(
                engine_options(artifact, ActivationMode::Bits16, 2560, 4, 1), drafter);
            options.use_cuda_graph = graphs;
            Engine engine(options);
            const std::string tag                   = graphs ? " (graphs)" : " (no graphs)";
            constexpr std::uint32_t kBrief          = 16;
            RequestOptions brief                    = request(7, false);
            brief.execution.requested_output_tokens = kBrief;
            auto long_row  = engine.submit(engine.prepare_tokens(prose), request(7, false));
            auto brief_row = engine.submit(engine.prepare_tokens(code), brief);
            const GenerationResult brief_result = brief_row.wait();
            // A few rounds of the long row alone before the late row joins it.
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            auto late_row = engine.submit(engine.prepare_tokens(code), request(7, false));
            const GenerationResult long_result = long_row.wait();
            const GenerationResult late_result = late_row.wait();
            check(brief_result.generated_token_ids ==
                      std::vector<TokenId>(plain_code.generated_token_ids.begin(),
                                           plain_code.generated_token_ids.begin() + kBrief),
                  "width switch brief K=7" + tag + ": its tokens differ from plain decode's");
            compare("width switch long K=7" + tag, plain_prose, long_result, true);
            compare("width switch late K=7" + tag, plain_code, late_result, true);
            for (const auto& [label, result] :
                 {std::pair<std::string, const GenerationResult*>{"long", &long_result},
                  std::pair<std::string, const GenerationResult*>{"late", &late_result}}) {
                const SpeculativeStats& s = result->speculative;
                std::cout << "width switch " << label << tag << ": " << s.narrow_rounds << " of "
                          << s.rounds << " rounds at K + 1 columns\n";
                check(s.narrow_rounds > 0 && s.narrow_rounds < s.rounds,
                      "width switch " + label + tag + ": the request did not draft at both widths");
            }
        }
        GenerationResult fast_plain;
        {
            Engine engine(engine_options(artifact, ActivationMode::Auto, 2560, 1, 1));
            fast_plain = engine.generate(engine.prepare_tokens(prose, false), request({}, false));
        }
        {
            Engine engine(dflash_options(engine_options(artifact, ActivationMode::Auto, 2560, 1, 1),
                                         drafter));
            // Reported only: the fast engines' verify widths may take other activation routes than
            // a decode step.
            compare("fast prose " + kd, fast_plain,
                    engine.generate(engine.prepare_tokens(prose, false), request(deep, false)),
                    false, kFastFloor);
            compare("fast prose K=3", fast_plain,
                    engine.generate(engine.prepare_tokens(prose, false), request(3, false)), false,
                    kFastFloor);
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "dense DFlash real: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
