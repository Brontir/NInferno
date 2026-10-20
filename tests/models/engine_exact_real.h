// --activations exact through the Engine (ninferno-serve's path) against the CLI's own Session run
// as ninferno-gemma4 / ninferno-dense --activations exact run it (the BF16-activation profile, and
// the invariant route profile for the whole run: ops/prefill_routes.h), on a real artifact: the
// same prompt ids, context and prefill chunk give the same greedy tokens, byte for byte. Shared by
// the Gemma 4 and dense tests; the family supplies its Session.

#pragma once

#include "ninferno/engine.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace ninferno::test::engine_exact {

inline int failures = 0;

inline void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

inline constexpr std::uint32_t kContext = 4096;
// Narrow enough that the prompt prefills in several chunks at a width the fast mode runs with
// four-bit activations (T > 16), wide enough to be a real prefill.
inline constexpr std::uint32_t kChunk   = 128;
inline constexpr std::uint32_t kDecode  = 64;

// Ordinary prose, a few hundred tokens under any of the tested tokenizers.
inline constexpr const char* kText =
    "The history of the lighthouse is a history of the coast itself. Before the first towers "
    "were raised, sailors read the shore by its hills, its river mouths and the smoke of its "
    "villages, and a ship that arrived after dark waited offshore for the morning. The earliest "
    "lights were open fires kept on headlands by monks or by the harbour masters of trading "
    "towns, and their keepers were paid in part by a toll on every cargo that came in safely. "
    "Coal replaced wood, oil replaced coal, and in the nineteenth century the Fresnel lens "
    "gathered the light of a single lamp into a beam that could be seen from twenty miles away. "
    "Each tower was given its own rhythm of flashes so that a navigator could tell one from "
    "another, and the lists of those rhythms were printed in almanacs next to the tides. The "
    "keepers lived with their families at the foot of the tower, trimmed the wicks, wound the "
    "clockwork that turned the lens, and wrote the weather into a logbook every four hours. "
    "Explain, in a few paragraphs, why the keepers' logbooks became valuable to historians of "
    "climate, and what their limits are as a source.";

struct Run {
    std::vector<TokenId> prompt;
    std::vector<TokenId> tokens;
};

// Greedy tokens through the Engine at `activations` (keys and values in `kv`, the sliding rings in
// `rings`): the artifact's own tokenizer encodes kText.
inline Run engine_run(const char* artifact, ActivationMode activations,
                      KvCacheStorage kv = KvCacheStorage::BFloat16,
                      KvCacheStorage rings = KvCacheStorage::BFloat16) {
    EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = kContext;
    options.kv_capacity                          = KvCapacityPolicy::explicit_capacity(kContext);
    options.prefill_chunk                        = kChunk;
    options.max_concurrency                      = 1;
    options.activations                          = activations;
    options.kv_cache                             = kv;
    options.sliding_kv_cache                     = rings;
    options.context_cache.host_state_slots       = 0;
    options.context_cache.host_kv_capacity_bytes = 0;
    Engine engine(options);
    Run run;
    run.prompt = engine.tokenize_text(kText);
    RequestOptions request;
    request.execution.requested_output_tokens = kDecode;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    const GenerationResult result =
        engine.generate(engine.prepare_tokens(run.prompt, false), request);
    run.tokens.assign(result.generated_token_ids.begin(), result.generated_token_ids.end());
    return run;
}

// The number of leading tokens two runs share.
inline std::size_t common_prefix(const std::vector<TokenId>& a, const std::vector<std::int32_t>& b) {
    std::size_t same = 0;
    while (same < a.size() && same < b.size() && a[same] == static_cast<TokenId>(b[same])) {
        ++same;
    }
    return same;
}

// The checks: the exact Engine's tokens equal the Session's (up to the Engine's own stop), and
// the fast Engine's first divergence from them is reported (not asserted: a model whose recorded
// policies are all BF16 has nothing to change).
inline void compare(const std::string& label, const Run& exact, const Run& fast,
                    const std::vector<std::int32_t>& session) {
    const std::size_t same = common_prefix(exact.tokens, session);
    std::cout << label << ": prompt " << exact.prompt.size() << " tokens; exact Engine "
              << exact.tokens.size() << " tokens, the first " << same
              << " equal to the Session's " << session.size() << "; fast Engine equal for its first "
              << common_prefix(fast.tokens, session) << '\n';
    check(exact.prompt.size() > kChunk, label + ": the prompt fits one prefill chunk");
    check(exact.tokens.size() >= 8, label + ": the exact Engine stopped within 8 tokens");
    check(same == exact.tokens.size() && same <= session.size(),
          label + ": the exact Engine's tokens differ from the CLI Session's under its exact "
                  "profile");
}

} // namespace ninferno::test::engine_exact
