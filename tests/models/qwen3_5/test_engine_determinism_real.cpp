// Determinism contracts on a real artifact.
//
// Contract classes: B = byte-identical tokens; D = tolerance, where a greedy
// trajectory may leave the reference only at a near-tie. At every D divergence the probe scores
// both candidates with a CausalScoring Engine on the shared prefix and reports the log-probability
// gap; a divergence with a gap above kBugGap is a bug.
//
// Every Engine, the scorer included, runs --activations exact (ActivationMode::Exact): the
// contracts below are those of the exact mode. Under fast (NINFERNO_DET_ACTIVATIONS=fast) a wide
// call encodes NVFP4 activations in four bits, the answer depends on the batch width, and the gap
// bound does not hold.
//
// Phases (MTP K=3, int8 KV, CUDA graphs, the "Matrix-min" configuration):
//   solo     B  one request repeated alone, greedy and seeded sampling.
//   params   B  sampling parameters are request data, not graph state: outputs of a
//               fresh Engine equal those after other parameters were captured and replayed; top_k=1
//               at T=1 equals greedy; different temperatures give different outputs.
//   batch    D  each prompt inside batches of 2..8 concurrent requests against its solo run
//               (all C, not only capture widths); seeded sampling likewise.
//   aba      D  A (tools prompt) -> B (A + 8K padding) -> A again with the prefix cache and MTP
//               (vllm#43559): A again must not leave A cold beyond a near-tie; the two warm
//               A runs are compared with each other as well.
//   structured B one json_schema request four times on a fresh Engine: the run that compiles the
//               grammar commits the same tokens as the later runs, with a clean constraint.
//   cachecold D  each prompt cold on an Engine with the prefix cache on vs off, and cold vs warm.
//   boundary D  template-rendered chat prompts (and one json_schema request) with prefix reuse on,
//               where the cold prefill may split at an automatic boundary, vs reuse off.
//   specoff  D  greedy MTP K=3 against plain greedy decode, solo.
//   twins    D  C identical requests (greedy and seeded) submitted together, row against row
//               (independent of neighbours; rows split only at near-ties).
//   restart  B  with NINFERNO_DET_COMPARE the solo references of an earlier process must match.
//   chunks   D  prompt lengths chunk*n + {-1..+2}, chunk 1024/4096/8192, MTP off and K=3, against
//               an unchunked prefill.
//   pollute  B/D after a 6000-token request: a one-token prompt, an endpoint hit with a short
//               remainder (B, against a fresh Engine), a batch of one-token rows (D).
//   fullpool    KV pool of max_context and one page less/more; a request runs to the context end
//               with MTP and without: refused or finished, never stalled; the next one is served.
//   hosttier D  64 requests over 8 shared system prompts on a small Device pool (evictions to
//               the Host tier and back), twice per Engine and on two Engines, against cache off;
//               a watchdog reports a lockup.
//   gapdiag     (explicit only) classifies one batch divergence by the token each execution path
//               chooses at its first difference; NINFERNO_DET_GAP_C, NINFERNO_DET_GAP_PROMPT.
//   cancelchunks (explicit) a 32K prompt cancelled after chunk 1, 5 and the last: reuse stays
//               within the computed prefix and the answer equals cold.
//   mixed       (explicit) verify rows, a one-token prefill tail, a prefix hit with a two-token
//               remainder and a plain decode in one batch, six orders, each == solo.
//   snapshot    (explicit) concurrent shared-prefix prefills, cuts at the served frontier +-1,
//               a turn after a decode across it, each == cold.
//   vrampeak    (explicit) memory growth after ready under the first heavy traffic stays
//               within the sizing's allowances.
//   acceptance  (explicit) AL per family x KV x K against NINFERNO_DET_ACCEPT_REF (or written to
//               NINFERNO_DET_ACCEPT_OUT), and windowed AL over a long generation.
//   modes       (explicit) exact against fast, each prompt alone: first difference, and both
//               tokens scored under each mode (a codec pair, reported, never a bug).
//   speed       (explicit) decode tok/s of C concurrent greedy requests (NINFERNO_DET_SPEED_C),
//               the price of an activation mode (NINFERNO_DET_ACTIVATIONS).
// Matrix env: NINFERNO_DET_KV bf16|int8|k8v4, NINFERNO_DET_SPEC mtp|dflash|dflash2, NINFERNO_DET_K,
// NINFERNO_DET_GRAPHS 0, NINFERNO_DET_HEAD full (default: the optimized proposal head),
// NINFERNO_DET_ACTIVATIONS exact|fast (default exact).
//
// Env: NINFERNO_TEST_ARTIFACT (required, else skip 77); NINFERNO_DET_REPEATS (solo repeats, 10);
// NINFERNO_DET_TOKENS (output tokens, 160); NINFERNO_DET_DUMP (write references);
// NINFERNO_DET_COMPARE (earlier dump to compare); NINFERNO_DET_PHASES (comma list, default all);
// NINFERNO_DET_PROMPTS (how many of the 8 prompts, 8; the batch phase needs at least 2).
// NINFERNO_DET_CONTEXT (each generation row's context, 12288); NINFERNO_DET_CHUNK_LENGTHS (the
// chunks phase's prompt lengths); NINFERNO_DET_DRAFTER (the drafter artifact of a family that ships
// it apart, e.g. Gemma 4's MTP assistant); NINFERNO_DET_POISON (a byte 1..255: each generation
// Engine runs with MaintainerOptions::debug_poison, --debug-poison, set to it).
#include "ninferno/engine.h"

#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using ninferno::TokenId;
using Tokens = std::vector<TokenId>;

constexpr double kNearTieGap = 0.25; // nats: an honest near-tie
constexpr double kBugGap     = 1.0;  // nats: a divergence this decisive is a defect

std::uint32_t env_u32(const char* name, std::uint32_t fallback) {
    const char* value = std::getenv(name);
    return value == nullptr || *value == '\0' ? fallback
                                              : static_cast<std::uint32_t>(std::stoul(value));
}

std::string env_string(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
}

std::string chat(const std::string& system, const std::string& user) {
    std::string text;
    if (!system.empty()) { text += "<|im_start|>system\n" + system + "<|im_end|>\n"; }
    text += "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
    text += "<think>\n\n</think>\n\n";
    return text;
}

const std::vector<std::string>& user_prompts() {
    static const std::vector<std::string> prompts = {
        "A train leaves at 09:40 and arrives at 13:05. How long is the trip in minutes? Show the "
        "arithmetic step by step.",
        "Write a Python function that returns the longest palindromic substring of a string, "
        "with a short docstring and two doctests.",
        "Describe the water cycle to a ten-year-old in one paragraph.",
        "Return a JSON object with keys name, year and tags describing the first Moon landing. "
        "Output only the JSON.",
        "Translate to French and to German: 'The library opens at nine and closes at five.'",
        "List five prime numbers greater than 100 and explain how you checked each one.",
        "Explain the difference between a mutex and a semaphore with a C++ example.",
        "Summarize the plot of Romeo and Juliet in exactly three sentences.",
    };
    return prompts;
}

struct Run {
    Tokens prompt;
    Tokens output;
    std::string content;
    ninferno::FinishReason finish = ninferno::FinishReason::None;
    std::uint32_t reused       = 0;
    std::uint64_t spec_rounds  = 0;
};

struct Divergence {
    std::string label;
    Tokens prompt;
    Tokens reference;
    Tokens candidate;
    std::size_t at = 0;
};

struct Report {
    int bugs = 0;
    std::vector<Divergence> pending; // D-class divergences, scored at the end
    nlohmann::ordered_json lines = nlohmann::ordered_json::array();

    void line(nlohmann::ordered_json value) {
        std::cout << value.dump() << std::endl;
        lines.push_back(std::move(value));
    }
};

ninferno::RequestOptions request(std::uint32_t tokens, bool reuse,
                               std::optional<ninferno::SamplingOverrides> sampling = std::nullopt) {
    ninferno::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.allow_prefix_reuse      = reuse;
    if (sampling) {
        options.execution.sampling = *sampling;
    } else {
        options.execution.sampling.temperature       = 0.0F;
        options.execution.sampling.presence_penalty  = 0.0F;
        options.execution.sampling.frequency_penalty = 0.0F;
    }
    return options;
}

ninferno::SamplingOverrides sampled(float temperature, std::int32_t top_k, float top_p,
                                  std::uint64_t seed) {
    ninferno::SamplingOverrides s;
    s.temperature       = temperature;
    s.top_k             = top_k;
    s.top_p             = top_p;
    s.min_p             = 0.0F;
    s.presence_penalty  = 0.0F;
    s.frequency_penalty = 0.0F;
    s.seed              = seed;
    return s;
}

Run generate(ninferno::Engine& engine, const Tokens& prompt,
             const ninferno::RequestOptions& options) {
    ninferno::GenerationResult result = engine.generate(engine.prepare_tokens(prompt), options);
    return Run{prompt, result.generated_token_ids, result.content, result.finish_reason,
               result.reused_prompt_tokens, result.speculative.rounds};
}

std::size_t first_difference(const Tokens& a, const Tokens& b) {
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) { return i; }
    }
    return a.size() == b.size() ? std::string::npos : n;
}

// B contract: a mismatch is a bug. D contract: a mismatch is scored later.
void compare(Report& report, const std::string& phase, const std::string& label, const Run& ref,
             const Run& got, bool byte_contract, bool sampled_run = false) {
    const std::size_t at = first_difference(ref.output, got.output);
    nlohmann::ordered_json line = {{"phase", phase},
                                   {"check", label},
                                   {"contract", byte_contract ? "B" : "D"},
                                   {"compared_tokens", std::min(ref.output.size(), got.output.size())},
                                   {"reused", got.reused}};
    if (at == std::string::npos) {
        line["result"] = "identical";
    } else if (byte_contract) {
        line["result"] = "BUG: byte contract broken";
        line["first_difference"] = at;
        ++report.bugs;
    } else if (sampled_run) {
        // A sampled trajectory flips wherever a draw lands near a CDF boundary, so the logit gap
        // at the first difference says nothing; independence from neighbours is the twins phase.
        line["result"]           = "diverged (sampled: D, not gap-scored)";
        line["first_difference"] = at;
    } else {
        line["result"]           = "diverged (scored below)";
        line["first_difference"] = at;
        if (at < ref.output.size() && at < got.output.size()) {
            report.pending.push_back(Divergence{phase + ":" + label, ref.prompt, ref.output,
                                                got.output, at});
        } else {
            line["result"] = "BUG: one output is a strict prefix of the other";
            ++report.bugs;
        }
    }
    report.line(std::move(line));
}

// NINFERNO_DET_ACTIVATIONS exact (default) | fast, for every Engine of the probe.
ninferno::ActivationMode probe_activations() {
    const std::string mode = env_string("NINFERNO_DET_ACTIVATIONS");
    if (mode.empty() || mode == "exact") { return ninferno::ActivationMode::Exact; }
    if (mode == "fast") { return ninferno::ActivationMode::Fast; }
    throw std::invalid_argument("NINFERNO_DET_ACTIVATIONS must be exact or fast");
}

ninferno::EngineOptions generation_options(const std::string& artifact, std::uint32_t concurrency,
                                         bool cache, bool speculate = true) {
    ninferno::EngineOptions options;
    options.artifact_path             = artifact;
    options.activations               = probe_activations();
    // NINFERNO_DET_CONTEXT: each row's context (12288); a 16 GB card fits C=8 with a smaller one.
    const std::uint32_t context       = env_u32("NINFERNO_DET_CONTEXT", 12288);
    options.max_context               = context;
    options.kv_capacity = ninferno::KvCapacityPolicy::explicit_capacity(context * concurrency);
    options.prefill_chunk             = 4096;
    options.kv_cache                  = ninferno::KvCacheStorage::Int8Group64;
    // Matrix overrides: NINFERNO_DET_KV bf16|int8|k8v4, NINFERNO_DET_SPEC
    // mtp|dflash|dflash2, NINFERNO_DET_K (draft tokens), NINFERNO_DET_GRAPHS 0 (eager decode),
    // NINFERNO_DET_SPEC off (no speculation anywhere: isolates the target from the drafter).
    if (env_string("NINFERNO_DET_SPEC") == "off") { speculate = false; }
    const std::string kv = env_string("NINFERNO_DET_KV");
    if (kv == "bf16") { options.kv_cache = ninferno::KvCacheStorage::BFloat16; }
    if (kv == "k8v4") { options.kv_cache = ninferno::KvCacheStorage::Fp8KeyNvfp4Value; }
    if (env_string("NINFERNO_DET_GRAPHS") == "0") { options.use_cuda_graph = false; }
    if (const std::string poison = env_string("NINFERNO_DET_POISON"); !poison.empty()) {
        const unsigned long byte = std::stoul(poison, nullptr, 0);
        if (byte == 0 || byte > 0xFF) {
            throw std::invalid_argument("NINFERNO_DET_POISON must be a byte in 1..255");
        }
        options.maintainer.debug_poison = static_cast<std::uint8_t>(byte);
    }
    if (speculate) {
        const std::string spec            = env_string("NINFERNO_DET_SPEC");
        options.speculative.backend       = spec == "dflash" ? ninferno::SpeculativeBackend::DFlash
                                            : spec == "dflash2" ? ninferno::SpeculativeBackend::DFlash2
                                                                : ninferno::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens =
            env_u32("NINFERNO_DET_K", spec.empty() || spec == "mtp" ? 3 : 7);
        // NINFERNO_DET_HEAD full: artifacts without the optimized (LM-head) proposal head.
        options.speculative.proposal_head = env_string("NINFERNO_DET_HEAD") == "full"
                                                ? ninferno::ProposalHead::Full
                                                : ninferno::ProposalHead::Optimized;
        // NINFERNO_DET_DRAFTER: a drafter shipped apart from the target (Gemma 4's MTP assistant).
        options.speculative.drafter = env_string("NINFERNO_DET_DRAFTER");
    }
    options.max_concurrency           = concurrency;
    options.max_pending_requests      = 16;
    // NINFERNO_DET_PENDING_MS: admission deadline. Exact prefills NVFP4 with BF16 activations, far
    // slower than the W4A4 prefill, so a queued hosttier request waits minutes, not seconds.
    options.pending_timeout_ms =
        env_u32("NINFERNO_DET_PENDING_MS", options.activations == ninferno::ActivationMode::Exact
                                               ? 1800000U
                                               : options.pending_timeout_ms);
    options.context_cache.enabled     = cache;
    if (!cache) {
        options.context_cache.host_state_slots       = 0;
        options.context_cache.host_kv_capacity_bytes = 0;
    }
    return options;
}

struct Probe {
    std::string artifact;
    std::uint32_t tokens  = 160;
    std::uint32_t repeats = 10;
    std::set<std::string> phases;
    std::vector<Tokens> prompts;
    std::map<std::string, Run> references; // solo references, dumped and compared across runs

    bool on(const std::string& phase) const { return phases.empty() || phases.contains(phase); }
};

void phase_solo(Probe& probe, ninferno::Engine& engine, Report& report) {
    for (std::size_t p = 0; p < probe.prompts.size(); ++p) {
        const std::string key = "p" + std::to_string(p);
        const Run greedy      = generate(engine, probe.prompts[p], request(probe.tokens, false));
        const Run seeded      = generate(engine, probe.prompts[p],
                                         request(probe.tokens, false, sampled(0.8F, 20, 0.9F, 1234)));
        probe.references[key + ".greedy"] = greedy;
        probe.references[key + ".seeded"] = seeded;
        int greedy_same = 0;
        int seeded_same = 0;
        for (std::uint32_t r = 1; r < probe.repeats; ++r) {
            const Run g = generate(engine, probe.prompts[p], request(probe.tokens, false));
            const Run s = generate(engine, probe.prompts[p],
                                   request(probe.tokens, false, sampled(0.8F, 20, 0.9F, 1234)));
            greedy_same += g.output == greedy.output ? 1 : 0;
            seeded_same += s.output == seeded.output ? 1 : 0;
            if (g.output != greedy.output) {
                compare(report, "solo", key + " greedy repeat " + std::to_string(r), greedy, g,
                        true);
            }
            if (s.output != seeded.output) {
                compare(report, "solo", key + " seeded repeat " + std::to_string(r), seeded, s,
                        true);
            }
        }
        report.line({{"phase", "solo"},
                     {"check", key},
                     {"contract", "B"},
                     {"repeats", probe.repeats},
                     {"greedy_identical", greedy_same + 1},
                     {"seeded_identical", seeded_same + 1},
                     {"tokens", greedy.output.size()},
                     {"spec_rounds", greedy.spec_rounds}});
    }
}

// Runs the sampling-parameter ladder in the given order on one Engine; returns outputs by name.
std::map<std::string, Run> params_ladder(Probe& probe, ninferno::Engine& engine,
                                         const std::vector<std::string>& order) {
    const Tokens& prompt = probe.prompts[2];
    const std::map<std::string, ninferno::RequestOptions> ladder = {
        {"greedy", request(probe.tokens, false)},
        {"t1_seed7", request(probe.tokens, false, sampled(1.0F, 0, 1.0F, 7))},
        {"t03_seed7", request(probe.tokens, false, sampled(0.3F, 0, 1.0F, 7))},
        {"t1_topk1_seed7", request(probe.tokens, false, sampled(1.0F, 1, 1.0F, 7))},
        {"t07_topp05_seed9", request(probe.tokens, false, sampled(0.7F, 0, 0.5F, 9))},
        {"t12_topk5_seed9", request(probe.tokens, false, sampled(1.2F, 5, 1.0F, 9))},
    };
    std::map<std::string, Run> out;
    for (const std::string& name : order) { out[name] = generate(engine, prompt, ladder.at(name)); }
    return out;
}

void phase_params(Probe& probe, Report& report) {
    const std::vector<std::string> forward = {"greedy",           "t1_seed7",       "t03_seed7",
                                              "t1_topk1_seed7",   "t07_topp05_seed9",
                                              "t12_topk5_seed9"};
    std::vector<std::string> backward(forward.rbegin(), forward.rend());
    std::map<std::string, Run> first;
    std::map<std::string, Run> second;
    {
        ninferno::Engine engine(generation_options(probe.artifact, 1, false));
        first = params_ladder(probe, engine, forward);
    }
    {
        ninferno::Engine engine(generation_options(probe.artifact, 1, false));
        second = params_ladder(probe, engine, backward);
    }
    for (const std::string& name : forward) {
        compare(report, "params", name + " fresh-order vs reversed-order Engine", first[name],
                second[name], true);
    }
    const auto expect = [&](bool ok, const std::string& what) {
        report.line({{"phase", "params"}, {"check", what}, {"result", ok ? "ok" : "BUG"}});
        if (!ok) { ++report.bugs; }
    };
    expect(first["t1_topk1_seed7"].output == first["greedy"].output,
           "top_k=1 at T=1 equals greedy");
    expect(first["t1_seed7"].output != first["greedy"].output, "T=1 differs from greedy");
    expect(first["t1_seed7"].output != first["t03_seed7"].output,
           "T=1 and T=0.3 with one seed differ");
    expect(first["t07_topp05_seed9"].output != first["t12_topk5_seed9"].output,
           "top_p=0.5 and top_k=5 differ");
}

void phase_batch(Probe& probe, Report& report) {
    ninferno::Engine engine(generation_options(probe.artifact, 8, false));
    std::vector<Run> solo;
    std::vector<Run> solo_seeded;
    for (const Tokens& prompt : probe.prompts) {
        solo.push_back(generate(engine, prompt, request(probe.tokens, false)));
        solo_seeded.push_back(
            generate(engine, prompt, request(probe.tokens, false, sampled(0.8F, 20, 0.9F, 99))));
    }
    for (std::uint32_t c = 2; c <= 8; ++c) {
        for (const bool seeded : {false, true}) {
            std::vector<ninferno::GenerationHandle> handles;
            std::vector<std::size_t> which;
            for (std::uint32_t i = 0; i < c; ++i) {
                const std::size_t p = (i + c) % probe.prompts.size();
                which.push_back(p);
                handles.push_back(engine.submit(
                    engine.prepare_tokens(probe.prompts[p]),
                    seeded ? request(probe.tokens, false, sampled(0.8F, 20, 0.9F, 99))
                           : request(probe.tokens, false)));
            }
            for (std::uint32_t i = 0; i < c; ++i) {
                ninferno::GenerationResult result = handles[i].wait();
                Run got{probe.prompts[which[i]], result.generated_token_ids, result.content,
                        result.finish_reason, result.reused_prompt_tokens,
                        result.speculative.rounds};
                // Dumped and compared across processes (NINFERNO_DET_DUMP / _COMPARE): the same
                // batches on the same binary are B, e.g. with and without NINFERNO_DET_POISON.
                probe.references["batch C=" + std::to_string(c) + (seeded ? " seeded" : " greedy") +
                                 " p" + std::to_string(which[i])] = got;
                compare(report, "batch",
                        "C=" + std::to_string(c) + (seeded ? " seeded" : " greedy") + " p" +
                            std::to_string(which[i]),
                        seeded ? solo_seeded[which[i]] : solo[which[i]], got, false, seeded);
            }
        }
    }
}

// As in vllm#43559: through the product path. The chat template renders 20 tools into the
// system turn, which the Engine publishes as an automatic shared prefix; B shares it and adds 8K
// tokens of user padding. A after B and A after A must reuse a prefix, and must not leave A cold
// beyond a near-tie. Outputs are compared token by token; divergences are listed unscored because
// the template-rendered prompt ids are not public.
void phase_aba(Probe& probe, Report& report) {
    ninferno::Engine engine(generation_options(probe.artifact, 1, true));
    std::vector<std::string> tools;
    for (int i = 0; i < 20; ++i) {
        tools.push_back(nlohmann::ordered_json{
            {"type", "function"},
            {"function",
             {{"name", "tool_" + std::to_string(i)},
              {"description", "Utility number " + std::to_string(i) +
                                  (i == 7 ? ": returns the current weather for a city." : ".")},
              {"parameters",
               {{"type", "object"},
                {"properties", {{"city", {{"type", "string"}}}, {"days", {{"type", "integer"}}}}},
                {"required", {"city"}}}}}}}
                            .dump());
    }
    const std::string question =
        "What is the weather in Oslo for the next 3 days? Use the right tool.";
    std::string padding;
    while (engine.tokenize_text(padding).size() < 8192) {
        padding += "Background note: the following log lines are unrelated to the question. "
                   "Line " +
                   std::to_string(padding.size()) + " reports nominal status.\n";
    }
    const auto input = [&](const std::string& user) {
        ninferno::ChatMessage message;
        message.role = ninferno::ChatRole::User;
        message.parts.push_back(ninferno::MessagePart{
            .kind = ninferno::MessagePartKind::Text, .text = user, .media = {}});
        ninferno::PromptInput prompt;
        prompt.messages.push_back(std::move(message));
        prompt.options.enable_thinking = false;
        prompt.options.tool_jsons      = tools;
        return prompt;
    };
    const auto run = [&](const std::string& user) {
        ninferno::GenerationResult result =
            engine.generate(engine.prepare(input(user)), request(probe.tokens, true));
        return Run{Tokens{}, result.generated_token_ids, result.content, result.finish_reason,
                   result.reused_prompt_tokens, result.speculative.rounds};
    };
    const std::string b_text = padding + "\n" + question;
    const Run a1 = run(question);
    const Run b1 = run(b_text);
    const Run a2 = run(question);
    const Run b2 = run(b_text);
    const Run a3 = run(question);
    const Run a4 = run(question);
    report.line({{"phase", "aba"},
                 {"check", "cache use"},
                 {"a1_reused", a1.reused},
                 {"b1_reused", b1.reused},
                 {"a2_reused", a2.reused},
                 {"b2_reused", b2.reused},
                 {"a3_reused", a3.reused},
                 {"a4_reused", a4.reused},
                 {"a1_tool_call", a1.content.find("<tool_call>") != std::string::npos},
                 {"a2_tool_call", a2.content.find("<tool_call>") != std::string::npos},
                 {"a1_spec_rounds", a1.spec_rounds},
                 {"a2_spec_rounds", a2.spec_rounds}});
    if (b1.reused == 0 || a2.reused == 0) {
        report.line({{"phase", "aba"},
                     {"check", "B after A and A after B reuse the shared tools prefix"},
                     {"result", "BUG: no prefix reuse"}});
        ++report.bugs;
    }
    compare(report, "aba", "A cold vs A after B (cache hit)", a1, a2, false);
    compare(report, "aba", "A after B vs A after B,A,B", a2, a3, false);
    compare(report, "aba", "A after A (warm) vs A after B", a4, a2, false);
    compare(report, "aba", "B after A vs B after A,B,A", b1, b2, false);
}

// Determinism under constrained decoding: one json_schema request repeated on a fresh Engine. The first
// run compiles the grammar; every run must commit the same tokens with a clean constraint.
void phase_structured(Probe& probe, Report& report) {
    ninferno::Engine engine(generation_options(probe.artifact, 1, false));
    ninferno::OutputConstraintSpec spec;
    spec.kind       = ninferno::OutputConstraintKind::JsonSchema;
    spec.definition = R"({"type":"object","properties":{"city":{"type":"string"},)"
                      R"("population":{"type":"integer"},"landmarks":{"type":"array",)"
                      R"("items":{"type":"string"}}},"required":["city","population","landmarks"],)"
                      R"("additionalProperties":false})";
    const Tokens prompt = engine.tokenize_text(chat("", "Describe Paris."));
    std::vector<Run> runs;
    for (int i = 0; i < 4; ++i) {
        ninferno::RequestOptions options = request(200, false);
        options.constraint.compiled    = engine.compile_output_constraint(spec);
        ninferno::GenerationResult result =
            engine.generate(engine.prepare_tokens(prompt), std::move(options));
        report.line({{"phase", "structured"},
                     {"check", "run " + std::to_string(i)},
                     {"tokens", result.generated_token_ids.size()},
                     {"finish", static_cast<int>(result.finish_reason)},
                     {"last_token", result.generated_token_ids.empty()
                                        ? -1
                                        : result.generated_token_ids.back()},
                     {"violated", result.constraint.violated},
                     {"violation", result.constraint.violation},
                     {"complete", result.constraint.complete},
                     {"masked_positions", result.constraint.masked_positions},
                     {"spec_rounds", result.speculative.rounds}});
        runs.push_back(Run{prompt, result.generated_token_ids, result.content,
                           result.finish_reason, result.reused_prompt_tokens,
                           result.speculative.rounds});
        if (result.constraint.violated) {
            report.line({{"phase", "structured"},
                         {"check", "run " + std::to_string(i) + " constraint clean"},
                         {"result", "BUG: constraint violated"}});
            ++report.bugs;
        }
    }
    for (std::size_t i = 1; i < runs.size(); ++i) {
        compare(report, "structured", "run 0 vs run " + std::to_string(i), runs[0], runs[i], true);
    }
}

// A cold request must not depend on whether the prefix cache is enabled: the Engine with the
// cache on runs each prompt first (nothing to reuse), and must match the Engine with the cache off.
// Scored as D, since the cache may change the prefill chunk plan.
void phase_cachecold(Probe& probe, Report& report) {
    std::vector<Run> off;
    {
        ninferno::Engine engine(generation_options(probe.artifact, 1, false));
        for (const Tokens& prompt : probe.prompts) {
            off.push_back(generate(engine, prompt, request(probe.tokens, true)));
        }
    }
    ninferno::Engine engine(generation_options(probe.artifact, 1, true));
    for (std::size_t p = 0; p < probe.prompts.size(); ++p) {
        const Run cold = generate(engine, probe.prompts[p], request(probe.tokens, true));
        const Run warm = generate(engine, probe.prompts[p], request(probe.tokens, true));
        compare(report, "cachecold", "p" + std::to_string(p) + " cache off vs cache on, cold",
                off[p], cold, false);
        compare(report, "cachecold", "p" + std::to_string(p) + " cache on: cold vs warm", cold,
                warm, false);
    }
}

// The serve path: chat prompts rendered by the template, where prefix reuse lets the Engine split
// the cold prefill at an automatic shared-prefix boundary to capture a checkpoint. The same
// requests with reuse on and with reuse off must agree up to near-ties, constrained or not.
void phase_boundary(Probe& probe, Report& report) {
    const auto input = [](const std::string& user) {
        ninferno::ChatMessage message;
        message.role = ninferno::ChatRole::User;
        message.parts.push_back(ninferno::MessagePart{
            .kind = ninferno::MessagePartKind::Text, .text = user, .media = {}});
        ninferno::PromptInput prompt;
        prompt.messages.push_back(std::move(message));
        prompt.options.enable_thinking = false;
        return prompt;
    };
    std::vector<std::string> users = user_prompts();
    users.push_back("Describe Paris.");
    ninferno::OutputConstraintSpec spec;
    spec.kind       = ninferno::OutputConstraintKind::JsonSchema;
    spec.definition = R"({"type":"object","properties":{"city":{"type":"string"},)"
                      R"("population":{"type":"integer"},"landmarks":{"type":"array",)"
                      R"("items":{"type":"string"}}},"required":["city","population","landmarks"],)"
                      R"("additionalProperties":false})";
    const auto run_all = [&](bool reuse) {
        ninferno::Engine engine(generation_options(probe.artifact, 1, true));
        std::vector<Run> runs;
        for (std::size_t i = 0; i < users.size(); ++i) {
            ninferno::RequestOptions options = request(probe.tokens, reuse);
            if (i + 1 == users.size()) {
                options.constraint.compiled = engine.compile_output_constraint(spec);
            }
            ninferno::PreparedPrompt prepared = engine.prepare(input(users[i]));
            const std::uint32_t prompt_tokens = prepared.summary().prompt_tokens;
            ninferno::GenerationResult result = engine.generate(std::move(prepared), options);
            Tokens ids = engine.tokenize_text(chat("", users[i]));
            if (ids.size() != prompt_tokens) { ids.clear(); } // scoring needs the exact prompt
            runs.push_back(Run{ids, result.generated_token_ids, result.content,
                               result.finish_reason, result.reused_prompt_tokens,
                               result.speculative.rounds});
        }
        return runs;
    };
    const std::vector<Run> with_reuse    = run_all(true);
    const std::vector<Run> without_reuse = run_all(false);
    for (std::size_t i = 0; i < users.size(); ++i) {
        const std::string label = (i + 1 == users.size() ? std::string("json_schema")
                                                          : "p" + std::to_string(i)) +
                                  " reuse on vs off";
        report.line({{"phase", "boundary"},
                     {"check", label + " sizes"},
                     {"reuse_tokens", with_reuse[i].output.size()},
                     {"no_reuse_tokens", without_reuse[i].output.size()},
                     {"reused_prompt_tokens", with_reuse[i].reused},
                     {"prompt_ids_known", !with_reuse[i].prompt.empty()}});
        compare(report, "boundary", label, without_reuse[i], with_reuse[i],
                /*byte_contract=*/false);
    }
}

// Greedy with MTP against plain greedy decode, solo. The contract allows a
// divergence only at a near-tie; the logit gap at the first difference decides.
void phase_specoff(Probe& probe, Report& report) {
    std::vector<Run> plain;
    {
        ninferno::Engine engine(generation_options(probe.artifact, 1, false, false));
        for (const Tokens& prompt : probe.prompts) {
            plain.push_back(generate(engine, prompt, request(probe.tokens, false)));
        }
    }
    ninferno::Engine engine(generation_options(probe.artifact, 1, false, true));
    for (std::size_t p = 0; p < probe.prompts.size(); ++p) {
        const Run spec = generate(engine, probe.prompts[p], request(probe.tokens, false));
        compare(report, "specoff", "p" + std::to_string(p) + " plain decode vs MTP K=3", plain[p],
                spec, false);
    }
}

// A seeded request is independent of its neighbours. C identical requests submitted
// together see identical inputs. A sampler sharing one random stream across rows would split them
// at the first sampled token; row-dependent numerics (the engine is not batch invariant)
// split them only at near-ties. Greedy rows are D with gap scoring, seeded rows D unscored.
void phase_twins(Probe& probe, Report& report) {
    ninferno::Engine engine(generation_options(probe.artifact, 8, false));
    for (const std::uint32_t c : {2U, 3U, 5U, 8U}) {
        for (const bool seeded : {false, true}) {
            for (const std::size_t p : {std::size_t{0}, std::size_t{2}}) {
                std::vector<ninferno::GenerationHandle> handles;
                for (std::uint32_t i = 0; i < c; ++i) {
                    handles.push_back(engine.submit(
                        engine.prepare_tokens(probe.prompts[p]),
                        seeded ? request(probe.tokens, false, sampled(0.8F, 20, 0.9F, 99))
                               : request(probe.tokens, false)));
                }
                std::vector<Run> rows;
                for (auto& handle : handles) {
                    ninferno::GenerationResult result = handle.wait();
                    rows.push_back(Run{probe.prompts[p], result.generated_token_ids,
                                       result.content, result.finish_reason,
                                       result.reused_prompt_tokens, result.speculative.rounds});
                }
                for (std::uint32_t i = 1; i < c; ++i) {
                    compare(report, "twins",
                            "C=" + std::to_string(c) + (seeded ? " seeded" : " greedy") + " p" +
                                std::to_string(p) + " row 0 vs row " + std::to_string(i),
                            rows[0], rows[i], false, seeded);
                }
            }
        }
    }
}

// Long token prompts cut to exact lengths (the chunk-boundary and full-pool phases).
Tokens long_prompt(ninferno::Engine& engine, std::size_t length) {
    static Tokens document;
    if (document.size() < length) {
        std::string text;
        for (int line = 0; engine.tokenize_text(text).size() < length + 64; ++line) {
            text += "Record " + std::to_string(line) + ": " +
                    user_prompts()[static_cast<std::size_t>(line) % user_prompts().size()] + "\n";
        }
        document = engine.tokenize_text(chat("", "Read the records and summarize them.\n" + text));
    }
    return Tokens(document.begin(), document.begin() + static_cast<std::ptrdiff_t>(length));
}

// Prompt lengths chunk*n + {-1, 0, +1, +2} for chunk in {1024, 4096, 8192}, MTP off and
// K=3. The first 64 greedy tokens must match an unchunked prefill (prefill_chunk 12288) up to
// near-ties (D); acceptance (speculative rounds) is reported next to it.
void phase_chunks(Probe& probe, Report& report) {
    constexpr std::uint32_t kTokens = 64;
    const std::vector<std::pair<std::uint32_t, std::vector<std::uint32_t>>> plan = {
        {1024, {1, 3}}, {4096, {1, 2}}, {8192, {1}}};
    std::vector<std::size_t> lengths;
    for (const auto& [chunk, multiples] : plan) {
        for (const std::uint32_t n : multiples) {
            for (const int delta : {-1, 0, 1, 2}) {
                const auto length = static_cast<std::size_t>(chunk * n + delta);
                if (std::find(lengths.begin(), lengths.end(), length) == lengths.end()) {
                    lengths.push_back(length);
                }
            }
        }
    }
    // NINFERNO_DET_CHUNK_LENGTHS (comma list): these prompt lengths instead, e.g. to map which
    // tail widths after a chunk boundary diverge.
    if (const std::string requested = env_string("NINFERNO_DET_CHUNK_LENGTHS");
        !requested.empty()) {
        lengths.clear();
        std::stringstream list(requested);
        for (std::string item; std::getline(list, item, ',');) {
            if (!item.empty()) { lengths.push_back(static_cast<std::size_t>(std::stoul(item))); }
        }
    }
    // The prompts are cut mid-text, so the model may stop at once; without model stop tokens
    // every run compares all kTokens.
    ninferno::RequestOptions greedy = request(kTokens, false);
    greedy.stop.include_model_defaults = false;
    for (const bool speculate : {false, true}) {
        const auto options = [&](std::uint32_t chunk) {
            ninferno::EngineOptions o = generation_options(probe.artifact, 1, false, speculate);
            o.prefill_chunk         = chunk;
            return o;
        };
        std::vector<Tokens> prompts;
        std::vector<Run> unchunked;
        {
            ninferno::Engine engine(options(12288));
            for (const std::size_t length : lengths) {
                prompts.push_back(long_prompt(engine, length));
                unchunked.push_back(generate(engine, prompts.back(), greedy));
            }
        }
        for (const auto& [chunk, multiples] : plan) {
            ninferno::Engine engine(options(chunk));
            for (std::size_t i = 0; i < lengths.size(); ++i) {
                if (lengths[i] + 1 < chunk) { continue; } // shorter prompts fit in one chunk
                const Run chunked = generate(engine, prompts[i], greedy);
                const std::string label = std::string(speculate ? "MTP K=3" : "MTP off") +
                                          " chunk " + std::to_string(chunk) + " length " +
                                          std::to_string(lengths[i]);
                report.line({{"phase", "chunks"},
                             {"check", label + " acceptance"},
                             {"unchunked_spec_rounds", unchunked[i].spec_rounds},
                             {"chunked_spec_rounds", chunked.spec_rounds}});
                compare(report, "chunks", label + " vs unchunked", unchunked[i], chunked, false);
            }
        }
    }
}

// A slot "polluted" by a long request must not leak into the next one. (a) a one-token
// prompt, (b) an endpoint hit with a one-token remainder, (d) a batch of one-token rows, each
// against the same sequence on a fresh Engine. (a) and (b) keep the plan, so they are B; (d)
// compares batch rows with solo runs, so it is D (the engine is not batch invariant).
void phase_pollute(Probe& probe, Report& report) {
    constexpr std::uint32_t kTokens = 48;
    const std::vector<TokenId> singles = {9707, 785, 3838, 40};
    const auto polluted_engine         = [&](std::uint32_t concurrency) {
        auto engine = std::make_unique<ninferno::Engine>(
            generation_options(probe.artifact, concurrency, true));
        (void)generate(*engine, long_prompt(*engine, 6000), request(64, true));
        return engine;
    };
    // (a)
    {
        const Tokens one = {singles[0]};
        Run fresh;
        {
            ninferno::Engine engine(generation_options(probe.artifact, 1, true));
            fresh = generate(engine, one, request(kTokens, true));
        }
        auto engine        = polluted_engine(1);
        const Run polluted = generate(*engine, one, request(kTokens, true));
        compare(report, "pollute", "(a) one-token prompt after a 6000-token request", fresh,
                polluted, true);
    }
    // (b) prompt P, then P + its output + one token: an endpoint hit with a remainder of one.
    {
        const Tokens p = probe.prompts[0];
        const auto sequence = [&](ninferno::Engine& engine) {
            const Run first = generate(engine, p, request(kTokens, true));
            Tokens next     = p;
            next.insert(next.end(), first.output.begin(), first.output.end());
            next.push_back(singles[1]);
            return generate(engine, next, request(kTokens, true));
        };
        Run fresh;
        {
            ninferno::Engine engine(generation_options(probe.artifact, 1, true));
            fresh = sequence(engine);
        }
        auto engine        = polluted_engine(1);
        const Run polluted = sequence(*engine);
        report.line({{"phase", "pollute"},
                     {"check", "(b) reuse"},
                     {"fresh_reused", fresh.reused},
                     {"polluted_reused", polluted.reused}});
        compare(report, "pollute", "(b) endpoint hit with remainder 1 after a 6000-token request",
                fresh, polluted, true);
    }
    // (d)
    {
        std::vector<Run> solo;
        {
            ninferno::Engine engine(generation_options(probe.artifact, 1, true));
            for (const TokenId token : singles) {
                solo.push_back(generate(engine, Tokens{token}, request(kTokens, false)));
            }
        }
        auto engine = polluted_engine(4);
        std::vector<ninferno::GenerationHandle> handles;
        for (const TokenId token : singles) {
            handles.push_back(
                engine->submit(engine->prepare_tokens(Tokens{token}), request(kTokens, false)));
        }
        for (std::size_t i = 0; i < handles.size(); ++i) {
            ninferno::GenerationResult result = handles[i].wait();
            const Run row{Tokens{singles[i]}, result.generated_token_ids, result.content,
                          result.finish_reason, result.reused_prompt_tokens,
                          result.speculative.rounds};
            compare(report, "pollute",
                    "(d) one-token batch row " + std::to_string(i) + " vs fresh solo", solo[i],
                    row, false);
        }
    }
}

// A KV pool of exactly max_context (and one 64-token page more or less), one request
// that runs to the end of the context with MTP and without. Either the Engine refuses at start or
// admission, or the request finishes at the context limit; it never stalls, and the next request
// is served. A watchdog turns a stall into a reported failure.
void phase_fullpool(Probe& probe, Report& report) {
    constexpr std::uint32_t kContext = 8192;
    for (const bool speculate : {false, true}) {
        for (const int pages : {-1, 0, 1}) {
            const std::uint32_t capacity = kContext + static_cast<std::uint32_t>(pages * 64);
            nlohmann::ordered_json line = {{"phase", "fullpool"},
                                           {"check", std::string(speculate ? "MTP K=3" : "MTP off") +
                                                         " capacity max_context" +
                                                         (pages < 0   ? "-1 page"
                                                          : pages > 0 ? "+1 page"
                                                                      : "")}};
            std::unique_ptr<ninferno::Engine> engine;
            try {
                ninferno::EngineOptions options =
                    generation_options(probe.artifact, 1, false, speculate);
                options.max_context = kContext;
                options.kv_capacity = ninferno::KvCapacityPolicy::explicit_capacity(capacity);
                engine              = std::make_unique<ninferno::Engine>(options);
            } catch (const std::exception& error) {
                line["result"] = std::string("refused at start: ") + error.what();
                report.line(std::move(line));
                continue;
            }
            const Tokens prompt = long_prompt(*engine, kContext - 200);
            const auto started  = std::chrono::steady_clock::now();
            std::atomic<bool> done{false};
            std::thread watchdog([&] {
                while (!done.load()) {
                    if (std::chrono::steady_clock::now() - started > std::chrono::seconds(300)) {
                        std::cout << nlohmann::json{{"phase", "fullpool"},
                                                    {"result", "BUG: stalled for 300 s"}}
                                         .dump()
                                  << std::endl;
                        std::_Exit(1);
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                }
            });
            try {
                // No model stop tokens: the request must run into the context limit.
                ninferno::RequestOptions to_the_end = request(400, false);
                to_the_end.stop.include_model_defaults = false;
                ninferno::GenerationResult full =
                    engine->generate(engine->prepare_tokens(prompt), to_the_end);
                line["prompt_tokens"]    = prompt.size();
                line["generated_tokens"] = full.generated_token_ids.size();
                line["finish"]           = static_cast<int>(full.finish_reason);
                line["total_tokens"] = prompt.size() + full.generated_token_ids.size();
                const bool at_limit = full.finish_reason == ninferno::FinishReason::ContextCapacity;
                line["result"] = at_limit ? "finished at the context limit"
                                          : "BUG: did not finish at the context limit";
                if (!at_limit) { ++report.bugs; }
            } catch (const std::exception& error) {
                line["result"] = std::string("refused at admission: ") + error.what();
            }
            try {
                const Run next = generate(*engine, probe.prompts[0], request(32, false));
                line["next_request_tokens"] = next.output.size();
            } catch (const std::exception& error) {
                line["result"] = std::string("BUG: next request failed: ") + error.what();
                ++report.bugs;
            }
            line["seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                            started)
                                  .count();
            done = true;
            watchdog.join();
            report.line(std::move(line));
        }
    }
}

// 64 requests over 8 shared ~1500-token system prompts on an Engine whose Device pool
// holds only a fraction of them (KV 12288 tokens, 2 extra Device StateImage slots), so prefixes
// are evicted to the Host tier and come back. Each request runs twice (cold pass, then a pass that
// hits the cache) and the whole scenario runs on two fresh Engines. Every output is compared
// with a cache-off Engine (D, scored) and the two Engines' second passes with each other (D:
// admission order and batch composition are timing-dependent). A watchdog turns a lockup into
// a reported failure.
void phase_hosttier(Probe& probe, Report& report) {
    constexpr std::uint32_t kTokens = 48;
    constexpr std::size_t kWave     = 16;
    std::vector<std::string> systems;
    for (int k = 0; k < 8; ++k) {
        std::string text = "You are archive assistant " + std::to_string(k) + ". Archive:\n";
        for (int line = 0; line < 110; ++line) {
            text += "Entry " + std::to_string(k * 1000 + line) + " stores item " +
                    std::to_string((k * 37 + line * 11) % 997) + " in shelf " +
                    std::to_string(line % 13) + ".\n";
        }
        text.pop_back(); // no trailing newline: the template trims message content
        systems.push_back(std::move(text));
    }
    const std::vector<std::string> questions = {
        "Which shelf holds entry {k}005?", "How many shelves are there?",
        "Name the item stored in entry {k}010.", "Summarize the archive in one sentence.",
        "Which entry comes first?", "Is shelf 7 used? Answer yes or no.",
        "List two entries on shelf 3.", "What is your role?"};
    struct Case {
        std::string system;
        std::string user;
    };
    std::vector<Case> cases;
    for (std::size_t q = 0; q < questions.size(); ++q) {
        for (std::size_t k = 0; k < systems.size(); ++k) {
            std::string user = questions[q];
            if (const auto at = user.find("{k}"); at != std::string::npos) {
                user.replace(at, 3, std::to_string(k));
            }
            cases.push_back(Case{systems[k], user});
        }
    }
    const auto input = [](const Case& c) {
        ninferno::PromptInput prompt;
        for (const auto& [role, text] : {std::pair{ninferno::ChatRole::System, c.system},
                                         std::pair{ninferno::ChatRole::User, c.user}}) {
            ninferno::ChatMessage m;
            m.role = role;
            m.parts.push_back(ninferno::MessagePart{
                .kind = ninferno::MessagePartKind::Text, .text = text, .media = {}});
            prompt.messages.push_back(std::move(m));
        }
        prompt.options.enable_thinking = false;
        return prompt;
    };
    const auto options = [&](bool cache) {
        ninferno::EngineOptions o = generation_options(probe.artifact, 8, cache);
        o.max_context           = 4096;
        o.kv_capacity           = ninferno::KvCapacityPolicy::explicit_capacity(12288);
        if (cache) { o.context_cache.device_state_slots = 2; }
        return o;
    };
    std::atomic<bool> done{false};
    const auto started = std::chrono::steady_clock::now();
    // NINFERNO_DET_WATCHDOG_S (default 900 s, 4800 s under exact, whose prefill is slower).
    const std::uint32_t limit =
        env_u32("NINFERNO_DET_WATCHDOG_S",
                probe_activations() == ninferno::ActivationMode::Exact ? 4800 : 900);
    std::thread watchdog([&] {
        while (!done.load()) {
            if (std::chrono::steady_clock::now() - started > std::chrono::seconds(limit)) {
                std::cout << nlohmann::json{{"phase", "hosttier"},
                                            {"result", "BUG: phase exceeded " +
                                                           std::to_string(limit) + " s (lockup?)"}}
                                 .dump()
                          << std::endl;
                std::_Exit(1);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });
    // Joined on every exit: an Engine error in a pass must reach main() as that error, not as
    // std::terminate on a joinable thread.
    struct WatchdogJoin {
        std::atomic<bool>& done;
        std::thread& thread;
        ~WatchdogJoin() {
            done = true;
            thread.join();
        }
    } join_watchdog{done, watchdog};
    const auto run_pass = [&](ninferno::Engine& engine, std::vector<Run>& out) {
        out.assign(cases.size(), Run{});
        for (std::size_t first = 0; first < cases.size(); first += kWave) {
            std::vector<ninferno::GenerationHandle> handles;
            for (std::size_t i = first; i < std::min(cases.size(), first + kWave); ++i) {
                handles.push_back(
                    engine.submit(engine.prepare(input(cases[i])), request(kTokens, true)));
            }
            for (std::size_t h = 0; h < handles.size(); ++h) {
                ninferno::GenerationResult result = handles[h].wait();
                const Case& c = cases[first + h];
                Tokens ids    = engine.tokenize_text(chat(c.system, c.user));
                if (ids.size() != result.prompt.prompt_tokens) { ids.clear(); }
                out[first + h] = Run{ids, result.generated_token_ids, result.content,
                                     result.finish_reason, result.reused_prompt_tokens,
                                     result.speculative.rounds};
            }
        }
    };
    std::vector<Run> cold;
    {
        ninferno::Engine engine(options(false));
        run_pass(engine, cold);
    }
    std::vector<std::vector<Run>> second_passes;
    for (int trial = 0; trial < 2; ++trial) {
        ninferno::Engine engine(options(true));
        std::vector<Run> first_pass;
        std::vector<Run> second_pass;
        run_pass(engine, first_pass);
        run_pass(engine, second_pass);
        std::uint64_t reused_first  = 0;
        std::uint64_t reused_second = 0;
        std::size_t hits            = 0;
        for (std::size_t i = 0; i < cases.size(); ++i) {
            reused_first += first_pass[i].reused;
            reused_second += second_pass[i].reused;
            hits += second_pass[i].reused > 0 ? 1U : 0U;
            const std::string tag = "trial " + std::to_string(trial) + " case " + std::to_string(i);
            compare(report, "hosttier", tag + " first pass vs cache off", cold[i], first_pass[i],
                    false);
            compare(report, "hosttier", tag + " second pass vs cache off", cold[i],
                    second_pass[i], false);
        }
        report.line({{"phase", "hosttier"},
                     {"check", "trial " + std::to_string(trial) + " reuse"},
                     {"reused_tokens_first_pass", reused_first},
                     {"reused_tokens_second_pass", reused_second},
                     {"second_pass_hits", hits},
                     {"cases", cases.size()}});
        second_passes.push_back(std::move(second_pass));
    }
    for (std::size_t i = 0; i < cases.size(); ++i) {
        compare(report, "hosttier", "case " + std::to_string(i) + " second pass: trial 0 vs 1",
                second_passes[0][i], second_passes[1][i], false);
    }
}

// A long synthetic document of varied sentences, so long prompts are not degenerate repetition.
std::string long_document(std::size_t sentences, const std::string& tag) {
    static const char* colours[] = {"red", "green", "blue", "amber", "violet", "grey", "white"};
    static const char* animals[] = {"fox",   "heron", "otter", "lynx", "badger", "crane",
                                    "stoat", "hare",  "ibis",  "mole", "wren"};
    static const char* cities[]  = {"Oslo", "Lima", "Kyiv",  "Quito", "Perth",
                                    "Riga", "Nice", "Turku", "Bern"};
    std::string text = "Case " + tag + ". Field notes follow.\n";
    for (std::size_t i = 0; i < sentences; ++i) {
        text += "Record " + std::to_string(i) + ": the " + colours[i % 7] + " " + animals[i % 11] +
                " walked " + std::to_string(i % 13 + 2) + " miles to " + cities[i % 9] +
                " on day " + std::to_string(i * 7 % 365) + ".\n";
    }
    return text;
}

// As in trtllm#10111 and sglang#35255: a ~32K-token prompt, chunk 2048, is cancelled after its
// 1st, 5th and last chunk; then the same prompt runs to the end. Nothing past what was computed may
// be reused (reused tokens <= prefill tokens computed before the cancel) and the answer equals a
// cold run on a cache-off Engine (D, gap-scored). Each round has its own document, so one round's
// catalogue cannot serve another.
void phase_cancelchunks(Probe& probe, Report& report) {
    constexpr std::uint32_t kChunk = 2048;
    const auto options             = [&](bool cache) {
        ninferno::EngineOptions o = generation_options(probe.artifact, 1, cache);
        o.max_context           = 40960;
        o.kv_capacity           = ninferno::KvCapacityPolicy::explicit_capacity(2 * 40960);
        o.prefill_chunk         = kChunk;
        return o;
    };
    ninferno::Engine cold(options(false));
    ninferno::Engine engine(options(true));
    for (const std::uint32_t k : {1U, 5U, 0U}) { // 0: after the last full chunk
        const std::string tag = "k" + std::to_string(k);
        Tokens prompt         = engine.tokenize_text(
            chat("", long_document(1750, tag) + "\nWhich city did record 7 reach, and on which day?"));
        const std::uint32_t chunks = static_cast<std::uint32_t>(prompt.size() / kChunk);
        const std::uint32_t after  = k == 0 ? chunks : k;
        const Run reference        = generate(cold, prompt, request(48, false));
        const std::uint64_t base   = engine.runtime_stats().computed_prefill_tokens;
        const ninferno::CancellationView cancel([&] {
            return engine.runtime_stats().computed_prefill_tokens - base >=
                   static_cast<std::uint64_t>(after) * kChunk;
        });
        ninferno::GenerationResult cancelled =
            engine.generate(engine.prepare_tokens(prompt), request(48, true), nullptr, cancel);
        const std::uint64_t computed = engine.runtime_stats().computed_prefill_tokens - base;
        const Run again              = generate(engine, prompt, request(48, true));
        const bool bounded           = again.reused <= computed;
        report.line({{"phase", "cancelchunks"},
                     {"check", "cancel after chunk " + std::to_string(after) + " of " +
                                   std::to_string((prompt.size() + kChunk - 1) / kChunk)},
                     {"prompt_tokens", prompt.size()},
                     {"cancelled_finish", static_cast<int>(cancelled.finish_reason)},
                     {"cancelled_output_tokens", cancelled.generated_token_ids.size()},
                     {"computed_before_cancel", computed},
                     {"reused_after", again.reused},
                     {"result", bounded ? "reuse within computed prefix"
                                        : "BUG: reused more than was computed"}});
        if (!bounded) { ++report.bugs; }
        compare(report, "cancelchunks", "after cancel at chunk " + std::to_string(after) + " vs cold",
                reference, again, false);
    }
}

// As in vllm#47381, vllm#51812 and sglang#19415: one batch mixing three requests in verify rounds, a
// prompt whose last prefill chunk is one token (chunk + 1), a prefix hit with a two-token
// remainder, and a plain decode; six arrival orders. Every row equals its solo run on the same
// Engine (D, gap-scored). Only the hit request may reuse a prefix.
void phase_mixed(Probe& probe, Report& report) {
    ninferno::Engine engine(generation_options(probe.artifact, 6, true));
    const Tokens hit_base = engine.tokenize_text(chat(
        "You are a meticulous archivist. " + long_document(40, "sys"), "Summarize record 3."));
    (void)generate(engine, hit_base, request(16, true)); // publishes the endpoint
    Tokens hit        = hit_base;
    const Tokens tail = engine.tokenize_text(" Also, please be brief about it.");
    hit.insert(hit.end(), tail.begin(), tail.begin() + 2); // a hit with a two-token remainder
    Tokens one_token_tail = engine.tokenize_text(chat("", long_document(260, "tail") + "\nCount the foxes."));
    one_token_tail.resize(4096 + 1); // chunk 4096: the last prefill chunk is one token
    struct Row {
        std::string name;
        Tokens prompt;
        std::uint32_t tokens;
        bool reuse;
    };
    const std::vector<Row> rows = {{"verify0", probe.prompts[0], 160, false},
                                   {"verify1", probe.prompts[1], 160, false},
                                   {"verify2", probe.prompts[2], 160, false},
                                   {"tail1", one_token_tail, 64, false},
                                   {"hit2", hit, 64, true},
                                   {"decode", probe.prompts[3], 160, false}};
    std::vector<Run> solo;
    for (const Row& row : rows) { solo.push_back(generate(engine, row.prompt, request(row.tokens, row.reuse))); }
    std::vector<std::vector<std::size_t>> orders;
    for (std::size_t r = 0; r < rows.size(); ++r) {
        std::vector<std::size_t> order;
        for (std::size_t i = 0; i < rows.size(); ++i) { order.push_back((i + r) % rows.size()); }
        orders.push_back(order);
    }
    orders.push_back({5, 4, 3, 2, 1, 0});
    for (std::size_t o = 0; o < orders.size(); ++o) {
        std::vector<ninferno::GenerationHandle> handles;
        for (const std::size_t i : orders[o]) {
            handles.push_back(engine.submit(engine.prepare_tokens(rows[i].prompt),
                                            request(rows[i].tokens, rows[i].reuse)));
        }
        for (std::size_t j = 0; j < handles.size(); ++j) {
            const std::size_t i            = orders[o][j];
            ninferno::GenerationResult result = handles[j].wait();
            const Run got{rows[i].prompt, result.generated_token_ids, result.content,
                          result.finish_reason, result.reused_prompt_tokens,
                          result.speculative.rounds};
            if (rows[i].reuse && o == 0) {
                report.line({{"phase", "mixed"},
                             {"check", "hit row reuse"},
                             {"reused", got.reused},
                             {"prompt_tokens", rows[i].prompt.size()}});
            }
            compare(report, "mixed", "order " + std::to_string(o) + " " + rows[i].name + " vs solo",
                    solo[i], got, false);
        }
    }
}

// As in vllm#51113 and sglang#35821: four concurrent prefills that share a long system prefix on
// an empty cache (chunk 512, MTP K=3), then the same prefix cut one token before, at and one token
// after the frontier the cache serves, and a request whose decode crosses that frontier followed
// by a hit on it. Every output equals a cache-off Engine (D, gap-scored); a hit never covers more
// than the prompt.
void phase_snapshot(Probe& probe, Report& report) {
    const auto options = [&](bool cache) {
        ninferno::EngineOptions o = generation_options(probe.artifact, 4, cache);
        o.prefill_chunk         = 512;
        return o;
    };
    ninferno::Engine cold(options(false));
    ninferno::Engine engine(options(true));
    const std::string system = "You are a field biologist. " + long_document(150, "snap");
    const std::vector<std::string> users = {"Which animal walked the farthest?",
                                            "List the cities in order.",
                                            "What colour was record 12?",
                                            "How many records mention Oslo?"};
    std::vector<Tokens> prompts;
    for (const std::string& user : users) { prompts.push_back(engine.tokenize_text(chat(system, user))); }
    std::vector<ninferno::GenerationHandle> handles;
    for (const Tokens& prompt : prompts) {
        handles.push_back(engine.submit(engine.prepare_tokens(prompt), request(64, true)));
    }
    std::uint32_t frontier = 0;
    for (std::size_t i = 0; i < handles.size(); ++i) {
        ninferno::GenerationResult result = handles[i].wait();
        const Run got{prompts[i], result.generated_token_ids, result.content, result.finish_reason,
                      result.reused_prompt_tokens, result.speculative.rounds};
        compare(report, "snapshot", "concurrent shared-prefix prefill " + std::to_string(i) + " vs cold",
                generate(cold, prompts[i], request(64, false)), got, false);
    }
    // The frontier a later request is served from.
    {
        const Run probe_hit = generate(engine, prompts[0], request(8, true));
        frontier            = probe_hit.reused;
        report.line({{"phase", "snapshot"}, {"check", "served frontier"}, {"reused", frontier},
                     {"prompt_tokens", prompts[0].size()}});
    }
    if (frontier > 2) {
        for (const int delta : {-1, 0, 1}) {
            Tokens cut(prompts[1].begin(),
                       prompts[1].begin() + static_cast<std::ptrdiff_t>(frontier + delta));
            const Run got = generate(engine, cut, request(48, true));
            report.line({{"phase", "snapshot"},
                         {"check", "prefix cut at frontier" + std::string(delta < 0 ? "-1" : delta > 0 ? "+1" : "")},
                         {"prompt_tokens", cut.size()},
                         {"reused", got.reused},
                         {"result", got.reused <= cut.size() ? "hit within the prompt"
                                                             : "BUG: hit longer than the prompt"}});
            if (got.reused > cut.size()) { ++report.bugs; }
            compare(report, "snapshot", "prefix cut at frontier" + std::to_string(delta) + " vs cold",
                    generate(cold, cut, request(48, false)), got, false);
        }
    }
    // Decode across the frontier, then a hit on what that decode published.
    Tokens first = prompts[2];
    const Run crossing = generate(engine, first, request(96, true));
    Tokens follow      = first;
    follow.insert(follow.end(), crossing.output.begin(), crossing.output.end());
    const Tokens more = engine.tokenize_text("<|im_end|>\n<|im_start|>user\nAnd record 13?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    follow.insert(follow.end(), more.begin(), more.end());
    const Run hit = generate(engine, follow, request(48, true));
    report.line({{"phase", "snapshot"}, {"check", "hit after a decode across the frontier"},
                 {"reused", hit.reused}, {"prompt_tokens", follow.size()}});
    compare(report, "snapshot", "follow-up turn vs cold", generate(cold, follow, request(48, false)),
            hit, false);
}

// As in vllm#36599: automatic KV sizing, MTP, eight concurrent requests with long prompts as
// the first traffic after ready. The device memory in use may grow after ready only within the
// allowances the sizing set aside (CUDA graphs and planned slack), and the workspace's logical
// peak stays inside its capacity. A 1 ms sampler records the lowest free memory.
void phase_vrampeak(Probe& probe, Report& report) {
    ninferno::EngineOptions options = generation_options(probe.artifact, 8, true);
    options.max_context           = 16384;
    options.kv_capacity           = ninferno::KvCapacityPolicy::automatic();
    ninferno::Engine engine(options);
    const ninferno::MemorySummary before = engine.memory_summary();
    std::size_t free_ready = 0, total = 0;
    cudaMemGetInfo(&free_ready, &total);
    std::atomic<bool> done{false};
    std::atomic<std::size_t> lowest_seen{free_ready};
    std::thread sampler([&] {
        while (!done.load()) {
            std::size_t f = 0, t = 0;
            if (cudaMemGetInfo(&f, &t) == cudaSuccess && f < lowest_seen.load()) {
                lowest_seen.store(f);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    // Joined on every exit: an Engine error below must reach the report, not std::terminate on a
    // joinable thread (first 5080 run: "terminate called without an active exception").
    struct JoinOnExit {
        std::atomic<bool>& done;
        std::thread& thread;
        ~JoinOnExit() {
            done = true;
            if (thread.joinable()) { thread.join(); }
        }
    } join_on_exit{done, sampler};
    std::vector<ninferno::GenerationHandle> handles;
    for (int i = 0; i < 8; ++i) {
        const Tokens prompt = engine.tokenize_text(
            chat("", long_document(i == 0 ? 780 : 200 + 60 * i, "vram" + std::to_string(i)) +
                         "\nSummarize the notes."));
        handles.push_back(engine.submit(engine.prepare_tokens(prompt), request(256, false)));
    }
    for (auto& handle : handles) { (void)handle.wait(); }
    done = true;
    sampler.join();
    const std::size_t lowest = lowest_seen.load();
    const ninferno::MemorySummary after = engine.memory_summary();
    const std::size_t growth    = free_ready > lowest ? free_ready - lowest : 0;
    const std::size_t allowance = before.cuda_graph_allowance_bytes + before.planned_slack_bytes;
    const bool within           = growth <= allowance &&
                        after.workspace.peak_used_bytes <= after.workspace.capacity_bytes;
    report.line({{"phase", "vrampeak"},
                 {"check", "first heavy traffic after ready"},
                 {"kv_capacity_tokens", before.kv_capacity},
                 {"free_at_ready_mib", free_ready >> 20},
                 {"lowest_free_mib", lowest >> 20},
                 {"growth_mib", growth >> 20},
                 {"allowance_mib", allowance >> 20},
                 {"cuda_graph_allowance_mib", before.cuda_graph_allowance_bytes >> 20},
                 {"planned_slack_mib", before.planned_slack_bytes >> 20},
                 {"workspace_peak_mib", after.workspace.peak_used_bytes >> 20},
                 {"workspace_capacity_mib", after.workspace.capacity_bytes >> 20},
                 {"result", within ? "within the sizing" : "BUG: memory grew past the sizing"}});
    if (!within) { ++report.bugs; }
}

// Acceptance per task family (vllm#14464, sglang#27737), KV codec and draft depth, and
// windowed acceptance over a long generation. Each cell reports tokens per round (AL) and requires
// drafts; with NINFERNO_DET_ACCEPT_REF naming a JSON reference, every cell must keep
// AL >= 0.95 x reference, and without one the table is written to NINFERNO_DET_ACCEPT_OUT as the
// next reference. Windowed: a 4096-token story in sixteen 256-token continuations (each resumes
// from the cached prefix); no later window's AL may fall more than 30 % below the first four's.
// modes (explicit only): each prompt alone, greedy, under exact and under fast (the probe's other
// Engine settings). The two modes are different numerics (a codec pair, D), so at
// their first difference both tokens are scored under an exact and a fast scorer: how far apart
// the modes are, not a bug count.
void phase_modes(Probe& probe, Report& report) {
    std::map<ninferno::ActivationMode, std::vector<Run>> runs;
    for (const auto mode : {ninferno::ActivationMode::Exact, ninferno::ActivationMode::Fast}) {
        ninferno::EngineOptions options = generation_options(probe.artifact, 1, false);
        options.activations           = mode;
        ninferno::Engine engine(options);
        for (const Tokens& prompt : probe.prompts) {
            runs[mode].push_back(generate(engine, prompt, request(probe.tokens, false)));
        }
    }
    struct Split {
        std::size_t prompt, at;
        TokenId exact, fast;
    };
    std::vector<Split> splits;
    for (std::size_t p = 0; p < probe.prompts.size(); ++p) {
        const Run& e = runs[ninferno::ActivationMode::Exact][p];
        const Run& f = runs[ninferno::ActivationMode::Fast][p];
        const std::size_t at = first_difference(e.output, f.output);
        report.line({{"phase", "modes"},
                     {"check", "p" + std::to_string(p) + " exact vs fast"},
                     {"compared_tokens", std::min(e.output.size(), f.output.size())},
                     {"first_difference", at == std::string::npos ? -1L : static_cast<long>(at)}});
        if (at != std::string::npos && at < e.output.size() && at < f.output.size()) {
            splits.push_back({p, at, e.output[at], f.output[at]});
        }
    }
    for (const auto mode : {ninferno::ActivationMode::Exact, ninferno::ActivationMode::Fast}) {
        ninferno::EngineOptions options;
        options.artifact_path = probe.artifact;
        options.purpose       = ninferno::EnginePurpose::CausalScoring;
        options.activations   = mode;
        options.max_context   = 12288;
        options.kv_capacity   = ninferno::KvCapacityPolicy::explicit_capacity(12288);
        options.prefill_chunk = 4096;
        options.kv_cache      = generation_options(probe.artifact, 1, false).kv_cache;
        ninferno::Engine scorer(options);
        for (const Split& s : splits) {
            Tokens prefix = probe.prompts[s.prompt];
            const Run& e  = runs[ninferno::ActivationMode::Exact][s.prompt];
            prefix.insert(prefix.end(), e.output.begin(),
                          e.output.begin() + static_cast<std::ptrdiff_t>(s.at));
            const auto score = [&](TokenId token) {
                Tokens sequence = prefix;
                sequence.push_back(token);
                return static_cast<double>(
                    scorer.score_tokens(sequence, static_cast<std::uint32_t>(prefix.size()))
                        .back());
            };
            const double lp_exact = score(s.exact);
            const double lp_fast  = score(s.fast);
            report.line(
                {{"phase", "modes"},
                 {"check", "p" + std::to_string(s.prompt) + " scorer " +
                               (mode == ninferno::ActivationMode::Exact ? "exact" : "fast")},
                 {"position", s.at},
                 {"logp_exact_token", lp_exact},
                 {"logp_fast_token", lp_fast},
                 {"gap", std::abs(lp_exact - lp_fast)}});
        }
    }
}

// speed (explicit only): decode throughput of C concurrent greedy requests (NINFERNO_DET_SPEED_C,
// default 1,2,4,8; NINFERNO_DET_SPEED_REPEATS, default 3; NINFERNO_DET_TOKENS per request), cache
// off, prompts as in the batch phase. Wall time from submission to the last result, one warm batch
// first. It prices an activation mode (NINFERNO_DET_ACTIVATIONS) on one binary.
void phase_speed(Probe& probe, Report& report) {
    std::vector<std::uint32_t> widths;
    const std::string requested = env_string("NINFERNO_DET_SPEED_C");
    std::stringstream list(requested.empty() ? std::string("1,2,4,8") : requested);
    for (std::string item; std::getline(list, item, ',');) {
        if (!item.empty()) { widths.push_back(static_cast<std::uint32_t>(std::stoul(item))); }
    }
    const std::uint32_t repeats = env_u32("NINFERNO_DET_SPEED_REPEATS", 3);
    for (const std::uint32_t c : widths) {
        ninferno::Engine engine(generation_options(probe.artifact, c, false));
        const auto run_batch = [&] {
            std::vector<ninferno::GenerationHandle> handles;
            const auto start = std::chrono::steady_clock::now();
            for (std::uint32_t i = 0; i < c; ++i) {
                const std::size_t p = (i + c) % probe.prompts.size();
                handles.push_back(engine.submit(engine.prepare_tokens(probe.prompts[p]),
                                                request(probe.tokens, false)));
            }
            std::size_t tokens = 0;
            std::uint64_t rounds = 0;
            for (auto& handle : handles) {
                const ninferno::GenerationResult result = handle.wait();
                tokens += result.generated_token_ids.size();
                rounds += result.speculative.rounds;
            }
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            return std::make_tuple(tokens, seconds, rounds);
        };
        (void)run_batch();
        for (std::uint32_t r = 0; r < repeats; ++r) {
            const auto [tokens, seconds, rounds] = run_batch();
            report.line({{"phase", "speed"},
                         {"concurrency", c},
                         {"repeat", r},
                         {"tokens", tokens},
                         {"seconds", seconds},
                         {"tok_s", static_cast<double>(tokens) / seconds},
                         {"spec_rounds", rounds}});
        }
    }
}

void phase_acceptance(Probe& probe, Report& report) {
    const std::vector<std::pair<std::string, std::vector<std::size_t>>> families = {
        {"math", {0, 5}}, {"code", {1, 6}}, {"prose", {2, 4, 7}}, {"json", {3}}};
    nlohmann::ordered_json table = nlohmann::ordered_json::object();
    for (const char* kv : {"bf16", "int8"}) {
        for (const std::uint32_t k : {1U, 3U}) {
            ninferno::EngineOptions options = generation_options(probe.artifact, 1, false);
            options.kv_cache = std::string(kv) == "bf16" ? ninferno::KvCacheStorage::BFloat16
                                                         : ninferno::KvCacheStorage::Int8Group64;
            options.speculative.draft_tokens = k;
            ninferno::Engine engine(options);
            for (const auto& [family, indices] : families) {
                std::uint64_t rounds = 0, accepted = 0, drafted = 0;
                for (const std::size_t p : indices) {
                    if (p >= probe.prompts.size()) { continue; }
                    const ninferno::GenerationResult r =
                        engine.generate(engine.prepare_tokens(probe.prompts[p]), request(256, false));
                    rounds += r.speculative.rounds;
                    accepted += r.speculative.accepted_tokens;
                    drafted += r.speculative.drafted_tokens;
                }
                const std::string cell = std::string("kv=") + kv + " k=" + std::to_string(k) +
                                         " " + family;
                const double al = rounds ? double(accepted + rounds) / double(rounds) : 0.0;
                table[cell]     = al;
                report.line({{"phase", "acceptance"}, {"check", cell}, {"rounds", rounds},
                             {"drafted", drafted}, {"accepted", accepted}, {"tokens_per_round", al},
                             {"acceptance_rate", drafted ? double(accepted) / double(drafted) : 0.0}});
                if (drafted == 0) {
                    report.line({{"phase", "acceptance"}, {"check", cell},
                                 {"result", "BUG: no drafts"}});
                    ++report.bugs;
                }
            }
        }
    }
    // Windowed acceptance, int8 KV, K=3, prefix reuse on.
    {
        ninferno::EngineOptions options = generation_options(probe.artifact, 1, true);
        options.max_context           = 12288;
        ninferno::Engine engine(options);
        Tokens context = engine.tokenize_text(chat(
            "", "Write a long, detailed adventure story about a lighthouse keeper and a storm. "
                "Keep going for many pages; do not stop early."));
        std::vector<double> windows;
        for (int w = 0; w < 16; ++w) {
            const ninferno::GenerationResult r =
                engine.generate(engine.prepare_tokens(context), request(256, true));
            if (r.speculative.rounds == 0 || r.generated_token_ids.empty()) { break; }
            windows.push_back(double(r.speculative.accepted_tokens + r.speculative.rounds) /
                              double(r.speculative.rounds));
            context.insert(context.end(), r.generated_token_ids.begin(), r.generated_token_ids.end());
            if (r.finish_reason != ninferno::FinishReason::OutputLimit) { break; }
        }
        // A collapse is a later window falling below the start of the generation: baseline = the
        // first four windows' mean; no later window may be more than 30 % below it. (The mean of
        // all windows was the first criterion; a story that settles into repetition raises the
        // late windows to AL = k + 1 and made the early, normal ones look like a collapse.)
        double mean = 0;
        for (const double v : windows) { mean += v; }
        mean = windows.empty() ? 0 : mean / double(windows.size());
        const std::size_t head = std::min<std::size_t>(4, windows.size());
        double baseline        = 0;
        for (std::size_t w = 0; w < head; ++w) { baseline += windows[w]; }
        baseline = head ? baseline / double(head) : 0;
        const double lowest_later =
            windows.size() > head ? *std::min_element(windows.begin() + head, windows.end())
                                  : baseline;
        const bool held = windows.size() > head && lowest_later >= 0.7 * baseline;
        report.line({{"phase", "acceptance"}, {"check", "windowed AL over 256-token windows"},
                     {"windows", windows}, {"mean", mean}, {"baseline_first4", baseline},
                     {"lowest_later", lowest_later},
                     {"result", held ? "no later window 30 % below the first four"
                                     : "BUG: windowed acceptance collapsed"}});
        if (!held) { ++report.bugs; }
        table["windowed mean"] = mean;
    }
    const std::string ref_path = env_string("NINFERNO_DET_ACCEPT_REF");
    std::ifstream ref_stream(ref_path);
    if (!ref_path.empty() && ref_stream) {
        const nlohmann::json ref = nlohmann::json::parse(ref_stream);
        for (auto it = table.begin(); it != table.end(); ++it) {
            if (!ref.contains(it.key())) { continue; }
            const double want = ref.at(it.key()).get<double>();
            const double got  = it.value().get<double>();
            const bool ok     = got >= 0.95 * want;
            report.line({{"phase", "acceptance"}, {"check", "gate " + it.key()},
                         {"reference", want}, {"measured", got},
                         {"result", ok ? "AL >= 0.95 x reference" : "BUG: acceptance regressed"}});
            if (!ok) { ++report.bugs; }
        }
    } else if (const std::string out = env_string("NINFERNO_DET_ACCEPT_OUT"); !out.empty()) {
        std::ofstream(out) << table.dump(1) << '\n';
        report.line({{"phase", "acceptance"}, {"check", "reference written"}, {"path", out}});
    }
}

// gapdiag (explicit only): classifies one batch divergence. It replays the batch phase's C-wide
// greedy batch (NINFERNO_DET_GAP_C, default 8), takes prompt NINFERNO_DET_GAP_PROMPT (default 7)
// and its first difference from the solo run, then asks which token each execution path chooses at
// that position: whole solo runs and prefix-forced single steps (the shared prefix prefilled, one
// greedy token) under int8 and BF16 KV with MTP K=3 and without speculation, and the causal
// scorer with its default KV and with the generation KV. A defect shows as one path disagreeing
// with all others; a near-tie shows as paths splitting between the two tokens.
void phase_gapdiag(Probe& probe, Report& report) {
    const std::uint32_t width = env_u32("NINFERNO_DET_GAP_C", 8);
    const std::size_t target  = env_u32("NINFERNO_DET_GAP_PROMPT", 7) % probe.prompts.size();
    Run solo;
    Run batched;
    {
        ninferno::Engine engine(generation_options(probe.artifact, 8, false));
        std::vector<Run> solos;
        for (const Tokens& prompt : probe.prompts) {
            solos.push_back(generate(engine, prompt, request(probe.tokens, false)));
        }
        solo = solos[target];
        std::vector<ninferno::GenerationHandle> handles;
        std::vector<std::size_t> which;
        for (std::uint32_t i = 0; i < width; ++i) {
            const std::size_t p = (i + width) % probe.prompts.size();
            which.push_back(p);
            handles.push_back(engine.submit(engine.prepare_tokens(probe.prompts[p]),
                                            request(probe.tokens, false)));
        }
        for (std::uint32_t i = 0; i < width; ++i) {
            ninferno::GenerationResult result = handles[i].wait();
            if (which[i] == target) {
                batched = Run{probe.prompts[target], result.generated_token_ids, result.content,
                              result.finish_reason, 0, result.speculative.rounds};
            }
        }
        const Run again = generate(engine, probe.prompts[target], request(probe.tokens, false));
        report.line({{"phase", "gapdiag"},
                     {"check", "solo repeat after the batch"},
                     {"identical", again.output == solo.output}});
    }
    const std::size_t at = first_difference(solo.output, batched.output);
    report.line({{"phase", "gapdiag"},
                 {"check", "C=" + std::to_string(width) + " greedy p" + std::to_string(target)},
                 {"first_difference", at == std::string::npos ? -1L : static_cast<long>(at)},
                 {"solo_tokens", solo.output.size()},
                 {"batched_tokens", batched.output.size()}});
    if (at == std::string::npos || at >= solo.output.size() || at >= batched.output.size()) {
        report.line({{"phase", "gapdiag"}, {"result", "no scorable divergence this run"}});
        return;
    }
    const TokenId ref  = solo.output[at];
    const TokenId cand = batched.output[at];
    Tokens prefix      = probe.prompts[target];
    prefix.insert(prefix.end(), solo.output.begin(),
                  solo.output.begin() + static_cast<std::ptrdiff_t>(at));
    const auto name = [&](TokenId t) {
        return t == ref ? std::string("solo") : t == cand ? std::string("batched") : "other";
    };
    struct Arm {
        const char* label;
        ninferno::KvCacheStorage kv;
        bool speculate;
    };
    for (const Arm arm : {Arm{"int8+mtp3", ninferno::KvCacheStorage::Int8Group64, true},
                          Arm{"int8+nospec", ninferno::KvCacheStorage::Int8Group64, false},
                          Arm{"bf16+mtp3", ninferno::KvCacheStorage::BFloat16, true},
                          Arm{"bf16+nospec", ninferno::KvCacheStorage::BFloat16, false}}) {
        ninferno::EngineOptions options =
            generation_options(probe.artifact, 1, false, arm.speculate);
        options.kv_cache              = arm.kv;
        ninferno::Engine engine(options);
        const Run whole  = generate(engine, probe.prompts[target], request(probe.tokens, false));
        const Run forced = generate(engine, prefix, request(1, false));
        const std::size_t whole_at = first_difference(solo.output, whole.output);
        report.line(
            {{"phase", "gapdiag"},
             {"check", std::string("path ") + arm.label},
             {"whole_run_first_difference_from_solo",
              whole_at == std::string::npos ? -1L : static_cast<long>(whole_at)},
             {"whole_run_token_at", at < whole.output.size() ? whole.output[at] : -1},
             {"whole_run_choice",
              at < whole.output.size() ? name(whole.output[at]) : std::string("none")},
             {"prefix_forced_token", forced.output.empty() ? -1 : forced.output[0]},
             {"prefix_forced_choice",
              forced.output.empty() ? std::string("none") : name(forced.output[0])}});
    }
    for (const bool generation_kv : {false, true}) {
        ninferno::EngineOptions options;
        options.artifact_path = probe.artifact;
        options.purpose       = ninferno::EnginePurpose::CausalScoring;
        options.activations   = probe_activations();
        options.max_context   = 12288;
        options.kv_capacity   = ninferno::KvCapacityPolicy::explicit_capacity(12288);
        options.prefill_chunk = 4096;
        if (generation_kv) { options.kv_cache = ninferno::KvCacheStorage::Int8Group64; }
        ninferno::Engine scorer(options);
        const auto score = [&](TokenId token) {
            Tokens sequence = prefix;
            sequence.push_back(token);
            return static_cast<double>(
                scorer.score_tokens(sequence, static_cast<std::uint32_t>(prefix.size())).back());
        };
        const double lp_ref  = score(ref);
        const double lp_cand = score(cand);
        report.line({{"phase", "gapdiag"},
                     {"check", std::string("scorer kv=") + (generation_kv ? "int8" : "default")},
                     {"position", at},
                     {"solo_token", ref},
                     {"batched_token", cand},
                     {"logp_solo", lp_ref},
                     {"logp_batched", lp_cand},
                     {"gap", std::abs(lp_ref - lp_cand)}});
    }
}

void score_divergences(const Probe& probe, Report& report) {
    if (report.pending.empty()) { return; }
    ninferno::EngineOptions options;
    options.artifact_path = probe.artifact;
    options.purpose       = ninferno::EnginePurpose::CausalScoring;
    options.max_context   = 12288;
    options.kv_capacity   = ninferno::KvCapacityPolicy::explicit_capacity(12288);
    options.prefill_chunk = 4096;
    // The gap is judged on the numerics that generated the tokens: the generation's KV codec. A
    // BF16 scorer against int8-KV outputs mistook a 0.375-nat int8 near-tie for a 2.0-nat BUG
    // (a batch of eight).
    options.kv_cache = generation_options(probe.artifact, 1, false).kv_cache;
    // And on its activation mode: a W4A4 prefill scorer judged A16-decoded tokens 1 nat apart from
    // the path that chose them.
    options.activations = probe_activations();
    ninferno::Engine scorer(options);
    report.line({{"phase", "score"},
                 {"check", "scorer KV codec"},
                 {"kv_cache", static_cast<int>(options.kv_cache)}});
    for (const Divergence& d : report.pending) {
        if (d.prompt.empty()) {
            report.line({{"phase", "score"}, {"check", d.label}, {"position", d.at},
                         {"result", "unscored: prompt ids unknown"}});
            continue;
        }
        Tokens prefix = d.prompt;
        prefix.insert(prefix.end(), d.reference.begin(),
                      d.reference.begin() + static_cast<std::ptrdiff_t>(d.at));
        const auto score = [&](TokenId token) {
            Tokens sequence = prefix;
            sequence.push_back(token);
            const std::vector<float> lp =
                scorer.score_tokens(sequence, static_cast<std::uint32_t>(prefix.size()));
            return static_cast<double>(lp.back());
        };
        const double lp_ref  = score(d.reference[d.at]);
        const double lp_cand = score(d.candidate[d.at]);
        const double gap     = std::abs(lp_ref - lp_cand);
        const char* verdict  = gap <= kNearTieGap ? "near-tie"
                               : gap <= kBugGap   ? "WARN: not a near-tie"
                                                  : "BUG: decisive divergence";
        if (gap > kBugGap) { ++report.bugs; }
        report.line({{"phase", "score"},
                     {"check", d.label},
                     {"position", d.at},
                     {"reference_token", d.reference[d.at]},
                     {"candidate_token", d.candidate[d.at]},
                     {"logp_reference", lp_ref},
                     {"logp_candidate", lp_cand},
                     {"gap", gap},
                     {"result", verdict}});
    }
}

nlohmann::json dump_references(const Probe& probe) {
    nlohmann::json out = nlohmann::json::object();
    for (const auto& [key, run] : probe.references) { out[key] = run.output; }
    return out;
}

void compare_restart(const Probe& probe, const std::string& path, Report& report) {
    std::ifstream stream(path);
    if (!stream) {
        report.line({{"phase", "restart"}, {"check", "earlier dump"}, {"result", "missing"}});
        return;
    }
    const nlohmann::json earlier = nlohmann::json::parse(stream);
    int same  = 0;
    int total = 0;
    for (const auto& [key, run] : probe.references) {
        if (!earlier.contains(key)) { continue; }
        ++total;
        const Tokens before = earlier.at(key).get<Tokens>();
        if (before == run.output) {
            ++same;
            continue;
        }
        Run old = run;
        old.output = before;
        compare(report, "restart", key + " earlier process vs this process", old, run, true);
    }
    report.line({{"phase", "restart"},
                 {"check", "solo references across processes"},
                 {"contract", "B"},
                 {"identical", same},
                 {"compared", total}});
}

} // namespace

int main() {
    const std::string artifact = env_string("NINFERNO_TEST_ARTIFACT");
    if (artifact.empty()) {
        std::cout << "skip: NINFERNO_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        Probe probe;
        probe.artifact = artifact;
        probe.tokens   = env_u32("NINFERNO_DET_TOKENS", 160);
        probe.repeats  = env_u32("NINFERNO_DET_REPEATS", 10);
        std::stringstream phases(env_string("NINFERNO_DET_PHASES"));
        for (std::string phase; std::getline(phases, phase, ',');) {
            if (!phase.empty()) { probe.phases.insert(phase); }
        }
        Report report;
        {
            ninferno::Engine engine(generation_options(artifact, 1, false));
            const auto available = static_cast<std::uint32_t>(user_prompts().size());
            const std::uint32_t prompt_count =
                std::min<std::uint32_t>(env_u32("NINFERNO_DET_PROMPTS", available), available);
            for (std::uint32_t p = 0; p < prompt_count; ++p) {
                probe.prompts.push_back(engine.tokenize_text(chat("", user_prompts()[p])));
            }
            if (probe.on("solo") || probe.on("restart")) { phase_solo(probe, engine, report); }
        }
        if (probe.on("params")) { phase_params(probe, report); }
        if (probe.on("batch")) { phase_batch(probe, report); }
        if (probe.on("aba")) { phase_aba(probe, report); }
        if (probe.on("structured")) { phase_structured(probe, report); }
        if (probe.on("cachecold")) { phase_cachecold(probe, report); }
        if (probe.on("boundary")) { phase_boundary(probe, report); }
        if (probe.on("specoff")) { phase_specoff(probe, report); }
        if (probe.on("twins")) { phase_twins(probe, report); }
        if (probe.on("chunks")) { phase_chunks(probe, report); }
        if (probe.on("pollute")) { phase_pollute(probe, report); }
        if (probe.on("fullpool")) { phase_fullpool(probe, report); }
        if (probe.on("hosttier")) { phase_hosttier(probe, report); }
        if (probe.phases.contains("gapdiag")) { phase_gapdiag(probe, report); }
        if (probe.phases.contains("cancelchunks")) { phase_cancelchunks(probe, report); }
        if (probe.phases.contains("mixed")) { phase_mixed(probe, report); }
        if (probe.phases.contains("snapshot")) { phase_snapshot(probe, report); }
        if (probe.phases.contains("vrampeak")) { phase_vrampeak(probe, report); }
        if (probe.phases.contains("acceptance")) { phase_acceptance(probe, report); }
        if (probe.phases.contains("speed")) { phase_speed(probe, report); }
        if (probe.phases.contains("modes")) { phase_modes(probe, report); }
        score_divergences(probe, report);
        if (const std::string dump = env_string("NINFERNO_DET_DUMP"); !dump.empty()) {
            std::ofstream(dump) << dump_references(probe).dump() << '\n';
        }
        if (const std::string earlier = env_string("NINFERNO_DET_COMPARE"); !earlier.empty()) {
            compare_restart(probe, earlier, report);
        }
        std::cout << "determinism: bugs=" << report.bugs
                  << " divergences_scored=" << report.pending.size() << '\n';
        return report.bugs == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "determinism probe error: " << error.what() << '\n';
        return 1;
    }
}
