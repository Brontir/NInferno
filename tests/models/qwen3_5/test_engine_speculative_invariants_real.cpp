// Speculative round bookkeeping on a real artifact, for every backend the artifact carries.
//
// A speculative round licenses A accepted drafts plus one correction/bonus token. Over a request
// that runs to its output limit every round is fully committed, so the published counters must
// reconcile exactly with the generated length:
//
//   generated = 1 (prefill token) + rounds + accepted + fallback_steps + plain_steps
//
// and the per-position counters form a survival curve (non-increasing, summing to accepted).
// Output budgets around the draft window must never be overshot by a multi-token round, and a
// stop token that lands inside an accepted run must end the output at its first occurrence.
//
// The acceptance watchdog acts: a floor no round can reach trips it at its minimum round count,
// and the request finishes with plain decode (the plain graphs a speculative engine captures),
// alone and with a concurrent request.
//
// Per-request speculation: an engine that captures several depths runs each request at its own
// depth (0: plain decode) and floor, alone and batched with requests of other depths (the launch
// runs at the widest, and every row drafts at its width); the counters reconcile.
//
// Two contracts on the tokens. B, byte-identical: a request alone at depth d in an
// engine of several depths equals an engine whose only depth is d (the narrowed launch window runs
// the same kernels), and a plain request alone or batched (its own ordinary launch) equals a
// plain engine. D, near-tie tolerance: batched speculating rows and a watchdog switch run other
// kernel shapes than their reference, the single-depth engine's output at their depth (plain
// decode where no such engine ran), and greedy output agrees only up to near-ties; D outputs are
// compared up to the reference's end of turn (past it the continuation is out of distribution),
// and every divergence is scored at the end with a CausalScoring engine on the shared prefix, as
// the determinism test does: a log-probability gap above kBugGap between the two candidates
// fails, smaller gaps are reported.
#include "ninferno/engine.h"
#include "real_artifact.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double kNearTieGap = 0.25; // nats: an honest near-tie
constexpr double kBugGap     = 1.0;  // nats: a divergence this decisive is a defect

constexpr ninferno::TokenId kEndOfTurn = 248046; // the prompt's own turn delimiter

struct Divergence {
    std::string label;
    std::vector<ninferno::TokenId> reference, candidate;
    std::size_t at = 0;
};
std::vector<Divergence> g_divergences;
// Each backend's single-depth engine output (exercise_backend's reference), by backend and depth.
std::vector<std::pair<std::pair<ninferno::SpeculativeBackend, std::uint32_t>,
                      std::vector<ninferno::TokenId>>>
    g_single_depth;

// D contract: records the first difference from the reference within the reference's turn,
// scored at the end.
void compare_near(const std::vector<ninferno::TokenId>& reference,
                      const std::vector<ninferno::TokenId>& candidate, const std::string& label) {
    const std::size_t turn = static_cast<std::size_t>(
        std::find(reference.begin(), reference.end(), kEndOfTurn) - reference.begin());
    std::size_t at = 0;
    while (at < turn && at < reference.size() && at < candidate.size() &&
           reference[at] == candidate[at]) {
        ++at;
    }
    if (at >= turn || (at == reference.size() && at == candidate.size())) { return; }
    std::cout << label << ": diverges from its reference at token " << at << '\n';
    g_divergences.push_back({label, reference, candidate, at});
}

// B contract: token for token.
int require_identical(const std::vector<ninferno::TokenId>& reference,
                      const std::vector<ninferno::TokenId>& candidate, const std::string& label) {
    if (reference == candidate) { return 0; }
    std::size_t at = 0;
    while (at < reference.size() && at < candidate.size() && reference[at] == candidate[at]) {
        ++at;
    }
    std::cerr << label << ": differs from its byte-identical reference at token " << at << '\n';
    return 1;
}

const std::vector<ninferno::TokenId>* single_depth_output(ninferno::SpeculativeBackend backend,
                                                        std::uint32_t depth) {
    for (const auto& [key, tokens] : g_single_depth) {
        if (key.first == backend && key.second == depth) { return &tokens; }
    }
    return nullptr;
}

// A speculating row's D reference: the single-depth engine at its depth, else plain decode.
const std::vector<ninferno::TokenId>& near_reference(ninferno::SpeculativeBackend backend,
                                                   std::uint32_t depth,
                                                   const std::vector<ninferno::TokenId>& plain) {
    const std::vector<ninferno::TokenId>* single = single_depth_output(backend, depth);
    return single != nullptr ? *single : plain;
}

const std::vector<ninferno::TokenId> kPrompt{
    248045, 846,    198, 109266, 3709,  96220, 117443, 97913,
    1710,   248046, 198, 248045, 74455, 198,   248068, 198,
};

ninferno::EngineOptions engine_options(const char* artifact, ninferno::SpeculativeBackend backend,
                                     std::uint32_t draft_tokens) {
    ninferno::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 4096;
    options.kv_capacity = ninferno::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk                        = 1024;
    options.kv_cache                             = ninferno::KvCacheStorage::Int8Group64;
    options.speculative.backend                  = backend;
    options.speculative.draft_tokens             = draft_tokens;
    options.speculative.proposal_head =
        ninferno::test::proposal_head(artifact, ninferno::ProposalHead::Optimized);
    options.context_cache.enabled                = false;
    options.context_cache.host_state_slots       = 0;
    options.context_cache.host_kv_capacity_bytes = 0;
    return options;
}

ninferno::RequestOptions greedy_options(std::uint32_t outputs) {
    ninferno::RequestOptions options;
    options.execution.requested_output_tokens    = outputs;
    options.execution.allow_prefix_reuse         = false;
    options.execution.sampling.temperature       = 0.0F;
    options.execution.sampling.presence_penalty  = 0.0F;
    options.execution.sampling.frequency_penalty = 0.0F;
    options.stop.include_model_defaults          = false;
    return options;
}

// `widest`: the widest launch the request may have shared (a row drafts at its launch's width).
int check_counters(const ninferno::GenerationResult& result, const std::string& label,
                   ninferno::SpeculativeBackend backend, std::uint32_t draft_tokens,
                   bool require_rounds, std::uint32_t widest = 0) {
    widest = std::max(widest, draft_tokens);
    const ninferno::SpeculativeStats& stats = result.speculative;
    const std::uint64_t committed =
        stats.rounds + stats.accepted_tokens + stats.fallback_steps + stats.plain_steps;
    const std::uint64_t per_position =
        std::accumulate(stats.accepted_per_position.begin(), stats.accepted_per_position.end(),
                        std::uint64_t{0});
    const bool survival = std::is_sorted(stats.accepted_per_position.rbegin(),
                                         stats.accepted_per_position.rend());
    if (!stats.enabled || stats.backend != backend || stats.draft_window != draft_tokens ||
        stats.accepted_per_position.size() < draft_tokens ||
        stats.accepted_per_position.size() > widest ||
        (stats.wide_rounds == 0 && stats.accepted_per_position.size() != draft_tokens) ||
        (require_rounds && stats.rounds == 0) || stats.accepted_tokens > stats.drafted_tokens ||
        stats.drafted_tokens > stats.rounds * widest ||
        per_position != stats.accepted_tokens || !survival ||
        result.generated_token_ids.size() != committed + 1) {
        std::cerr << label << ": speculative counters do not reconcile: generated="
                  << result.generated_token_ids.size() << " rounds=" << stats.rounds
                  << " drafted=" << stats.drafted_tokens << " accepted=" << stats.accepted_tokens
                  << " fallback=" << stats.fallback_steps << " plain=" << stats.plain_steps
                  << " per_position_sum=" << per_position
                  << " survival=" << survival << '\n';
        return 1;
    }
    return 0;
}

int exercise_backend(const char* artifact, ninferno::SpeculativeBackend backend,
                     std::uint32_t draft_tokens, const char* name) {
    ninferno::Engine engine(engine_options(artifact, backend, draft_tokens));
    const std::string label = std::string(name) + " K=" + std::to_string(draft_tokens);

    const ninferno::GenerationResult reference =
        engine.generate(engine.prepare_tokens(kPrompt), greedy_options(96));
    if (reference.generated_token_ids.size() != 96 ||
        reference.finish_reason != ninferno::FinishReason::OutputLimit) {
        std::cerr << label << ": reference request did not reach its output limit\n";
        return 1;
    }
    if (check_counters(reference, label + " reference", backend, draft_tokens, true) != 0) {
        return 1;
    }
    g_single_depth.push_back({{backend, draft_tokens}, reference.generated_token_ids});

    // Budgets on both sides of the round width: the licensed extent must shrink with the budget.
    for (std::uint32_t budget = 1; budget <= 2 * draft_tokens + 3; ++budget) {
        const ninferno::GenerationResult bounded =
            engine.generate(engine.prepare_tokens(kPrompt), greedy_options(budget));
        if (bounded.generated_token_ids.size() != budget ||
            bounded.finish_reason != ninferno::FinishReason::OutputLimit) {
            std::cerr << label << ": budget " << budget << " produced "
                      << bounded.generated_token_ids.size() << " tokens\n";
            return 1;
        }
        if (check_counters(bounded, label + " budget " + std::to_string(budget), backend,
                           draft_tokens, false) != 0) {
            return 1;
        }
    }

    // A stop token inside an accepted run ends the output at its first occurrence.
    int speculative_stops = 0;
    for (const std::size_t at : {std::size_t{9}, std::size_t{23}, std::size_t{41}}) {
        const ninferno::TokenId stop     = reference.generated_token_ids[at];
        ninferno::RequestOptions options = greedy_options(96);
        options.stop.token_ids.push_back(stop);
        const ninferno::GenerationResult stopped =
            engine.generate(engine.prepare_tokens(kPrompt), std::move(options));
        const auto& ids = stopped.generated_token_ids;
        if (stopped.finish_reason != ninferno::FinishReason::StopToken || ids.empty() ||
            ids.back() != stop || std::find(ids.begin(), ids.end() - 1, stop) != ids.end() - 1) {
            std::cerr << label << ": stop token " << stop << " did not end the output at its "
                      << "first occurrence (generated " << ids.size() << ")\n";
            return 1;
        }
        speculative_stops += stopped.speculative.rounds != 0 ? 1 : 0;
    }
    if (speculative_stops == 0) {
        std::cerr << label << ": no stop fixture executed a speculative round\n";
        return 1;
    }
    std::cout << "ok " << label << " acceptance " << reference.speculative.accepted_tokens << '/'
              << reference.speculative.drafted_tokens << '\n';
    return 0;
}

// A watchdog floor above any round's tokens trips at watchdog_min_rounds; from then on the
// request decodes plainly. Its output must equal plain greedy decode, alone and beside a second
// request (the steps then mix a speculative and a plain launch while only one row has tripped).
// No shared trip floor: a row stops at its own floor even while the other still drafts.
int exercise_watchdog(const char* artifact, ninferno::SpeculativeBackend backend,
                      std::uint32_t draft_tokens, const char* name,
                      const std::vector<ninferno::TokenId>& plain_reference) {
    ninferno::EngineOptions options         = engine_options(artifact, backend, draft_tokens);
    options.speculative.acceptance_floor  = 1000.0;
    options.speculative.shared_trip_floor = 0.0;
    options.max_concurrency               = 2;
    ninferno::Engine engine(options);
    const std::string label = std::string(name) + " K=" + std::to_string(draft_tokens) +
                              " watchdog";
    const auto outputs = static_cast<std::uint32_t>(plain_reference.size());
    const auto check   = [&](const ninferno::GenerationResult& result, const std::string& what) {
        const ninferno::SpeculativeStats& stats = result.speculative;
        if (!stats.watchdog_tripped ||
            stats.watchdog_round != options.speculative.watchdog_min_rounds ||
            stats.rounds != stats.watchdog_round || stats.plain_steps == 0) {
            std::cerr << what << ": watchdog did not switch the request to plain decode (tripped="
                      << stats.watchdog_tripped << " round=" << stats.watchdog_round
                      << " rounds=" << stats.rounds << " plain=" << stats.plain_steps << ")\n";
            return 1;
        }
        if (check_counters(result, what, backend, draft_tokens, true) != 0) { return 1; }
        compare_near(near_reference(backend, draft_tokens, plain_reference),
                     result.generated_token_ids, what);
        return 0;
    };
    if (check(engine.generate(engine.prepare_tokens(kPrompt), greedy_options(outputs)), label) !=
        0) {
        return 1;
    }
    // Two concurrent requests: their steps batch both rows (the second is admitted after the
    // first's prefill, so the rows trip in different steps and some steps mix the launches).
    ninferno::GenerationHandle first =
        engine.submit(engine.prepare_tokens(kPrompt), greedy_options(outputs));
    ninferno::GenerationHandle second =
        engine.submit(engine.prepare_tokens(kPrompt), greedy_options(outputs));
    if (check(first.wait(), label + " concurrent 1") != 0 ||
        check(second.wait(), label + " concurrent 2") != 0) {
        return 1;
    }
    std::cout << "ok " << label << '\n';
    return 0;
}

ninferno::RequestOptions depth_options(std::uint32_t outputs, std::uint32_t depth, double floor) {
    ninferno::RequestOptions options = greedy_options(outputs);
    options.execution.speculation  = ninferno::RequestSpeculation{.draft_tokens     = depth,
                                                                .acceptance_floor = floor};
    return options;
}

int exercise_request_depths(const char* artifact, ninferno::SpeculativeBackend backend,
                            std::uint32_t default_depth, std::vector<std::uint32_t> depths,
                            const char* name, const std::vector<ninferno::TokenId>& plain_96,
                            const std::vector<ninferno::TokenId>& plain_reference) {
    ninferno::EngineOptions options             = engine_options(artifact, backend, default_depth);
    options.speculative.request_draft_tokens = depths;
    options.speculative.shared_trip_floor    = 0.0; // the tripped row rides its neighbour
    options.max_concurrency                  = 3;
    ninferno::Engine engine(options);
    const auto outputs = static_cast<std::uint32_t>(plain_reference.size());
    const auto near = [&](const ninferno::GenerationResult& result, std::uint32_t depth,
                          const std::string& what) {
        compare_near(near_reference(backend, depth, plain_reference), result.generated_token_ids,
                     what);
        return 0;
    };
    const std::uint32_t widest = *std::max_element(depths.begin(), depths.end());
    std::vector<std::uint32_t> all = depths;
    all.push_back(default_depth);
    all.push_back(0);
    // Alone (B): each depth against its single-depth engine, plain against the plain engine; a
    // depth no single-depth engine ran is compared with plain decode (D).
    for (const std::uint32_t depth : all) {
        const std::string what = std::string(name) + " request depth " + std::to_string(depth);
        const ninferno::GenerationResult result =
            engine.generate(engine.prepare_tokens(kPrompt), depth_options(96, depth, 0.0));
        const std::vector<ninferno::TokenId>* single = single_depth_output(backend, depth);
        if (depth == 0) {
            if (require_identical(plain_96, result.generated_token_ids, what) != 0) { return 1; }
        } else if (single != nullptr) {
            if (require_identical(*single, result.generated_token_ids, what) != 0) { return 1; }
        } else {
            compare_near(plain_96, result.generated_token_ids, what);
        }
        if (depth == 0) {
            const auto& s = result.speculative;
            if (s.enabled || s.rounds != 0 || s.plain_steps + 1 != 96) {
                std::cerr << what << ": a plain request speculated (rounds=" << s.rounds
                          << " plain=" << s.plain_steps << ")\n";
                return 1;
            }
        } else if (check_counters(result, what, backend, depth, true) != 0 ||
                   result.speculative.wide_rounds != 0) {
            std::cerr << what << ": wide rounds " << result.speculative.wide_rounds << '\n';
            return 1;
        }
    }
    // Batched: the widest, the shallowest and a plain request together; then a request with a
    // floor no round reaches beside one that keeps speculating (the tripped one stops drafting and
    // verifies without drafts in its neighbour's launch: fallback steps).
    const std::uint32_t shallowest = *std::min_element(depths.begin(), depths.end());
    ninferno::GenerationHandle wide =
        engine.submit(engine.prepare_tokens(kPrompt), depth_options(outputs, widest, 0.0));
    ninferno::GenerationHandle shallow =
        engine.submit(engine.prepare_tokens(kPrompt), depth_options(outputs, shallowest, 0.0));
    ninferno::GenerationHandle plain =
        engine.submit(engine.prepare_tokens(kPrompt), depth_options(outputs, 0, 0.0));
    const ninferno::GenerationResult wide_result    = wide.wait();
    const ninferno::GenerationResult shallow_result = shallow.wait();
    const ninferno::GenerationResult plain_result   = plain.wait();
    const std::string batched                     = std::string(name) + " batched depths";
    if (near(wide_result, widest, batched + " widest") != 0 ||
        near(shallow_result, shallowest, batched + " shallowest") != 0 ||
        require_identical(plain_reference, plain_result.generated_token_ids,
                          batched + " plain") != 0 ||
        check_counters(wide_result, batched + " widest", backend, widest, true) != 0 ||
        check_counters(shallow_result, batched + " shallowest", backend, shallowest, true,
                       widest) != 0 ||
        plain_result.speculative.rounds != 0) {
        return 1;
    }
    if (shallow_result.speculative.wide_rounds == 0) {
        std::cerr << batched << ": the shallow request never shared a wider launch\n";
        return 1;
    }
    ninferno::GenerationHandle tripping =
        engine.submit(engine.prepare_tokens(kPrompt), depth_options(outputs, widest, 1000.0));
    ninferno::GenerationHandle steady =
        engine.submit(engine.prepare_tokens(kPrompt), depth_options(outputs, widest, 0.0));
    const ninferno::GenerationResult tripped = tripping.wait();
    const ninferno::GenerationResult kept    = steady.wait();
    const std::string floors               = std::string(name) + " per-request floors";
    if (near(tripped, widest, floors + " tripped") != 0 ||
        near(kept, widest, floors + " kept") != 0 ||
        !tripped.speculative.watchdog_tripped || tripped.speculative.fallback_steps == 0 ||
        kept.speculative.watchdog_tripped || kept.speculative.plain_steps != 0 ||
        check_counters(tripped, floors + " tripped", backend, widest, true) != 0 ||
        check_counters(kept, floors + " kept", backend, widest, true) != 0) {
        std::cerr << floors << ": the floor did not act on its own request only\n";
        return 1;
    }
    // A depth beyond the window is refused.
    try {
        (void)engine.generate(engine.prepare_tokens(kPrompt), depth_options(8, widest + 1, 0.0));
        std::cerr << name << ": a depth beyond the window was accepted\n";
        return 1;
    } catch (const std::exception&) {}
    std::cout << "ok " << name << " request depths" << '\n';
    return 0;
}

// Load bounds (SpeculativeOptions::load) of {2 rows: the shallowest depth, 3 rows: plain}: two
// requests at the widest depth decoding together verify at the shallowest (capped rounds, lossless
// against plain decode); three together turn plain for the rest of their requests. The three are
// submitted before any is waited on and each generates the reference's 160 tokens, against a
// prefill of a few milliseconds, so they share steps; each must report load_plain, which fails the
// test if one ran to its end before the third joined. The engine plans no speculative launch of
// three rows (speculative_batch_limit 2): three rows that speculated would throw. Nor does it
// capture a window deeper than the shallowest at two rows. An MTP launch there throws (launch_mtp
// checks its window's batch mask, which the round workspace and graphs follow), and that fails the
// requests and so this test: MTP rounds never run without a graph, and their ungraphed_rounds is 0
// by construction. A DFlash row still appending the features of a round it ran alone runs its
// first round beside the other without a graph (ungraphed_rounds), with the same tokens.
int exercise_load_bounds(const char* artifact, ninferno::SpeculativeBackend backend,
                         std::uint32_t default_depth, std::vector<std::uint32_t> depths,
                         const char* name, const std::vector<ninferno::TokenId>& plain_reference) {
    const std::uint32_t widest     = *std::max_element(depths.begin(), depths.end());
    const std::uint32_t shallowest = *std::min_element(depths.begin(), depths.end());
    ninferno::EngineOptions options             = engine_options(artifact, backend, default_depth);
    options.speculative.request_draft_tokens = depths;
    options.speculative.load                 = {{.lanes = 2, .draft_tokens = shallowest},
                                                {.lanes = 3, .draft_tokens = 0}};
    options.max_concurrency                  = 3;
    ninferno::Engine engine(options);
    const auto outputs = static_cast<std::uint32_t>(plain_reference.size());
    const std::string label = std::string(name) + " load bounds";
    std::uint64_t ungraphed = 0;
    {
        ninferno::GenerationHandle first =
            engine.submit(engine.prepare_tokens(kPrompt), depth_options(outputs, widest, 0.0));
        ninferno::GenerationHandle second =
            engine.submit(engine.prepare_tokens(kPrompt), depth_options(outputs, widest, 0.0));
        const ninferno::GenerationResult a = first.wait();
        const ninferno::GenerationResult b = second.wait();
        compare_near(plain_reference, a.generated_token_ids, label + " two rows 1");
        compare_near(plain_reference, b.generated_token_ids, label + " two rows 2");
        for (const ninferno::GenerationResult* r : {&a, &b}) {
            if (check_counters(*r, label + " two rows", backend, widest, true) != 0 ||
                r->speculative.capped_rounds == 0 || r->speculative.load_plain) {
                std::cerr << label << ": two rows were not capped at depth " << shallowest
                          << " (capped rounds " << r->speculative.capped_rounds << ")\n";
                return 1;
            }
            ungraphed += r->speculative.ungraphed_rounds;
        }
    }
    std::vector<ninferno::GenerationHandle> handles;
    for (int i = 0; i < 3; ++i) {
        handles.push_back(
            engine.submit(engine.prepare_tokens(kPrompt), depth_options(outputs, widest, 0.0)));
    }
    for (auto& handle : handles) {
        const ninferno::GenerationResult r = handle.wait();
        compare_near(plain_reference, r.generated_token_ids, label + " three rows");
        const std::uint64_t committed = r.speculative.rounds + r.speculative.accepted_tokens +
                                        r.speculative.fallback_steps + r.speculative.plain_steps;
        if (!r.speculative.load_plain || r.speculative.plain_steps == 0 ||
            r.speculative.watchdog_tripped || r.generated_token_ids.size() != committed + 1) {
            std::cerr << label << ": three rows did not turn plain (plain steps "
                      << r.speculative.plain_steps << ")\n";
            return 1;
        }
    }
    // A request that joins a row a load bound turned plain, after the load fell below the bound's
    // lanes, decodes plainly beside it (no second, speculative launch): two short requests end
    // while the long one they latched with still has most of its tokens ahead.
    {
        constexpr std::uint32_t kShort = 32;
        const std::vector<ninferno::TokenId> short_reference(plain_reference.begin(),
                                                           plain_reference.begin() + kShort);
        ninferno::GenerationHandle longer =
            engine.submit(engine.prepare_tokens(kPrompt), depth_options(outputs, widest, 0.0));
        ninferno::GenerationHandle first =
            engine.submit(engine.prepare_tokens(kPrompt), depth_options(kShort, widest, 0.0));
        ninferno::GenerationHandle second =
            engine.submit(engine.prepare_tokens(kPrompt), depth_options(kShort, widest, 0.0));
        (void)first.wait();
        (void)second.wait();
        ninferno::GenerationHandle joining =
            engine.submit(engine.prepare_tokens(kPrompt), depth_options(kShort, widest, 0.0));
        const ninferno::GenerationResult joined = joining.wait();
        const ninferno::GenerationResult held   = longer.wait();
        compare_near(short_reference, joined.generated_token_ids, label + " joining");
        compare_near(plain_reference, held.generated_token_ids, label + " latched");
        if (!held.speculative.load_plain) {
            std::cerr << label << ": the long request did not turn plain beside two others\n";
            return 1;
        }
        if (joined.speculative.rounds != 0 || !joined.speculative.load_plain) {
            std::cerr << label << ": a request joining a latched row speculated ("
                      << joined.speculative.rounds << " rounds)\n";
            return 1;
        }
    }
    // The latch does not spread: L latches beside two short requests, N1 joins L (plain beside
    // it), L ends while N1 still has most of its tokens ahead, and N2 joins N1 alone: N2 speculates
    // (N1 decodes plainly because it joined L, not because a bound turned it plain).
    {
        constexpr std::uint32_t kShort = 32;
        constexpr std::uint32_t kLatch = 96;
        const std::vector<ninferno::TokenId> short_reference(plain_reference.begin(),
                                                           plain_reference.begin() + kShort);
        const std::vector<ninferno::TokenId> latch_reference(plain_reference.begin(),
                                                           plain_reference.begin() + kLatch);
        ninferno::GenerationHandle latched =
            engine.submit(engine.prepare_tokens(kPrompt), depth_options(kLatch, widest, 0.0));
        ninferno::GenerationHandle first =
            engine.submit(engine.prepare_tokens(kPrompt), depth_options(kShort, widest, 0.0));
        ninferno::GenerationHandle second =
            engine.submit(engine.prepare_tokens(kPrompt), depth_options(kShort, widest, 0.0));
        (void)first.wait();
        (void)second.wait();
        // When each joiner ended, from the Engine's own clock: its total_seconds runs from its
        // preparation to its completion on the worker, so the time before its preparation plus
        // that is its end, whenever the test thread gets to wait on it.
        using Clock = std::chrono::steady_clock;

        const auto ended = [](Clock::time_point before, const ninferno::GenerationResult& r) {
            return before + std::chrono::duration_cast<Clock::duration>(
                                std::chrono::duration<double>(r.timings.total_seconds));
        };

        const Clock::time_point n1_before = Clock::now();
        ninferno::GenerationHandle n1 =
            engine.submit(engine.prepare_tokens(kPrompt), depth_options(outputs, widest, 0.0));
        const ninferno::GenerationResult origin = latched.wait();
        const Clock::time_point n2_before     = Clock::now();
        ninferno::GenerationHandle n2 =
            engine.submit(engine.prepare_tokens(kPrompt), depth_options(kShort, widest, 0.0));
        const ninferno::GenerationResult second_joiner = n2.wait();
        const ninferno::GenerationResult first_joiner  = n1.wait();
        // N1 started first; ending after N2 means the two shared every step of N2.
        const bool beside = ended(n1_before, first_joiner) > ended(n2_before, second_joiner);
        compare_near(latch_reference, origin.generated_token_ids, label + " origin");
        compare_near(plain_reference, first_joiner.generated_token_ids, label + " first joiner");
        compare_near(short_reference, second_joiner.generated_token_ids, label + " second joiner");
        if (!origin.speculative.load_plain || !first_joiner.speculative.load_plain) {
            std::cerr << label << ": the latch did not reach the first joiner\n";
            return 1;
        }
        if (!beside) {
            std::cerr << label << ": the first joiner ended before the second; N2 ran alone and "
                                  "the latch was not tested\n";
            return 1;
        }
        if (second_joiner.speculative.rounds == 0 || second_joiner.speculative.load_plain) {
            std::cerr << label << ": the latch spread to a request joining a joiner\n";
            return 1;
        }
    }
    std::cout << "ok " << label << " (" << ungraphed << " rounds without a graph)\n";
    return 0;
}

// The shared trip floor (SpeculativeOptions::shared_trip_floor, default 1.4): a request whose floor
// no round reaches keeps drafting beside a request that keeps drafting, as long as it emits at
// least 1.4 tokens per round (the watchdog's verdicts are held: shared_trip_deferrals), and stops
// at its floor when it runs alone. Its output stays the target's greedy decode either way.
int exercise_shared_trips(const char* artifact, ninferno::SpeculativeBackend backend,
                          std::uint32_t draft_tokens, const char* name,
                          const std::vector<ninferno::TokenId>& plain_reference) {
    ninferno::EngineOptions options = engine_options(artifact, backend, draft_tokens);
    options.max_concurrency       = 2;
    ninferno::Engine engine(options);
    const auto outputs      = static_cast<std::uint32_t>(plain_reference.size());
    const std::string label = std::string(name) + " shared trips";
    const ninferno::RequestOptions tripping_options = depth_options(outputs, draft_tokens, 1000.0);
    const ninferno::GenerationResult alone =
        engine.generate(engine.prepare_tokens(kPrompt), tripping_options);
    compare_near(near_reference(backend, draft_tokens, plain_reference), alone.generated_token_ids,
                 label + " alone");
    if (!alone.speculative.watchdog_tripped || alone.speculative.shared_trip_deferrals != 0) {
        std::cerr << label << ": a request alone did not stop at its floor\n";
        return 1;
    }
    ninferno::GenerationHandle tripping =
        engine.submit(engine.prepare_tokens(kPrompt), tripping_options);
    ninferno::GenerationHandle steady =
        engine.submit(engine.prepare_tokens(kPrompt), depth_options(outputs, draft_tokens, 0.0));
    const ninferno::GenerationResult held = tripping.wait();
    const ninferno::GenerationResult kept = steady.wait();
    compare_near(near_reference(backend, draft_tokens, plain_reference), held.generated_token_ids,
                 label + " held");
    compare_near(near_reference(backend, draft_tokens, plain_reference), kept.generated_token_ids,
                 label + " kept");
    const double tokens_per_round =
        held.speculative.rounds == 0
            ? 0.0
            : 1.0 + double(held.speculative.accepted_tokens) / double(held.speculative.rounds);
    // Held while it emitted at least 1.4 tokens per round: some verdicts held, and it stopped
    // only if its rolling tokens per round fell below 1.4 (or the steady request finished first).
    if (held.speculative.shared_trip_deferrals == 0 && tokens_per_round >= 1.6) {
        std::cerr << label << ": no verdict was held beside a drafting request ("
                  << tokens_per_round << " tokens per round)\n";
        return 1;
    }
    if (kept.speculative.watchdog_tripped || kept.speculative.shared_trip_deferrals != 0) {
        std::cerr << label << ": the steady request's watchdog acted\n";
        return 1;
    }
    std::cout << "ok " << label << " (held " << held.speculative.shared_trip_deferrals
              << " verdicts at " << tokens_per_round << " tokens per round)\n";
    return 0;
}

// Scores every recorded divergence: the log-probabilities of both candidate tokens after the
// shared prefix, from a CausalScoring engine. Returns the number of decisive (bug) divergences.
int score_divergences(const char* artifact) {
    if (g_divergences.empty()) { return 0; }
    ninferno::EngineOptions options;
    options.artifact_path = artifact;
    options.purpose       = ninferno::EnginePurpose::CausalScoring;
    options.max_context   = 1024;
    options.kv_capacity   = ninferno::KvCapacityPolicy::explicit_capacity(1024);
    options.prefill_chunk = 512;
    options.context_cache.enabled                = false;
    options.context_cache.host_state_slots       = 0;
    options.context_cache.host_kv_capacity_bytes = 0;
    ninferno::Engine scorer(options);
    int bugs = 0;
    for (const Divergence& d : g_divergences) {
        std::vector<ninferno::TokenId> prefix = kPrompt;
        prefix.insert(prefix.end(), d.reference.begin(),
                      d.reference.begin() + static_cast<std::ptrdiff_t>(d.at));
        if (d.at >= d.reference.size() || d.at >= d.candidate.size()) {
            std::cerr << d.label << ": outputs differ in length at " << d.at << '\n';
            ++bugs;
            continue;
        }
        const auto score = [&](ninferno::TokenId token) {
            std::vector<ninferno::TokenId> sequence = prefix;
            sequence.push_back(token);
            return static_cast<double>(
                scorer.score_tokens(sequence, static_cast<std::uint32_t>(prefix.size())).back());
        };
        const double lp_reference = score(d.reference[d.at]);
        const double lp_candidate = score(d.candidate[d.at]);
        const double gap          = std::abs(lp_reference - lp_candidate);
        std::cout << "divergence " << d.label << " at " << d.at << ": reference "
                  << d.reference[d.at] << " (" << lp_reference << "), candidate "
                  << d.candidate[d.at] << " (" << lp_candidate << "), gap " << gap << ' '
                  << (gap <= kNearTieGap ? "near-tie" : gap <= kBugGap ? "WARN: not a near-tie"
                                                                       : "BUG: decisive")
                  << '\n';
        bugs += gap > kBugGap ? 1 : 0;
    }
    return bugs;
}

} // namespace

int main() {
    std::cout.setf(std::ios::unitbuf); // progress survives an abort
    const char* artifact = std::getenv("NINFERNO_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFERNO_TEST_ARTIFACT is not set\n";
        return 77;
    }
    std::vector<ninferno::TokenId> plain_reference;
    std::vector<ninferno::TokenId> plain_96;
    {
        ninferno::EngineOptions options =
            engine_options(artifact, ninferno::SpeculativeBackend::None, 0);
        options.speculative.proposal_head = ninferno::ProposalHead::Full;
        ninferno::Engine plain(options);
        plain_reference =
            plain.generate(plain.prepare_tokens(kPrompt), greedy_options(160)).generated_token_ids;
        plain_96 =
            plain.generate(plain.prepare_tokens(kPrompt), greedy_options(96)).generated_token_ids;
    }
    if (!ninferno::test::real_artifact(artifact).mtp) {
        ninferno::test::skip_case("MTP", "the artifact has no MTP component");
    } else {
        for (const std::uint32_t k : {1U, 3U, 5U}) {
            if (exercise_backend(artifact, ninferno::SpeculativeBackend::Mtp, k, "MTP") != 0) {
                return 1;
            }
        }
        if (exercise_watchdog(artifact, ninferno::SpeculativeBackend::Mtp, 3, "MTP",
                              plain_reference) != 0 ||
            exercise_request_depths(artifact, ninferno::SpeculativeBackend::Mtp, 3, {1, 5}, "MTP",
                                    plain_96, plain_reference) != 0 ||
            exercise_load_bounds(artifact, ninferno::SpeculativeBackend::Mtp, 3, {1, 5}, "MTP",
                                 plain_reference) != 0 ||
            exercise_shared_trips(artifact, ninferno::SpeculativeBackend::Mtp, 3, "MTP",
                                  plain_reference) != 0) {
            return 1;
        }
    }
    // Masked-draft backends are optional artifact components; absent ones are skipped.
    const std::pair<ninferno::SpeculativeBackend, const char*> masked[] = {
        {ninferno::SpeculativeBackend::DFlash, "DFlash"},
        {ninferno::SpeculativeBackend::DFlash2, "DFlash2"},
    };
    for (const auto& [backend, name] : masked) {
        if (!ninferno::test::real_artifact(artifact).has(backend)) {
            ninferno::test::skip_case(name, "the artifact has no such component");
            continue;
        }
        for (const std::uint32_t k : {3U, 7U}) {
            try {
                if (exercise_backend(artifact, backend, k, name) != 0) { return 1; }
                if (k == 7 &&
                    (exercise_watchdog(artifact, backend, k, name, plain_reference) != 0 ||
                     exercise_request_depths(artifact, backend, k, {3, 11}, name, plain_96,
                                             plain_reference) != 0 ||
                     exercise_load_bounds(artifact, backend, k, {3, 11}, name,
                                          plain_reference) != 0 ||
                     exercise_shared_trips(artifact, backend, k, name, plain_reference) != 0)) {
                    return 1;
                }
            } catch (const std::exception& error) {
                const std::string what = error.what();
                if (what.find("DFlash") == std::string::npos &&
                    what.find("dflash") == std::string::npos) {
                    throw;
                }
                std::cout << "skip " << name << ": " << what << '\n';
                break;
            }
        }
    }
    if (score_divergences(artifact) != 0) { return 1; }
    std::cout << "ok\n";
    return 0;
}
