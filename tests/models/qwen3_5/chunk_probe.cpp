// chunk_probe: measurement tool for Qwen prefill chunk invariance (not a ctest).
//
//   chunk_probe <artifact> <tokens.i32> <out-prefix> <prefill-chunk> <max-new> [kv]
//
// Writes <out-prefix>.lp (FP32 log p(token[i] | token[0..i)) for i in [1,T), from the causal
// scoring path) and <out-prefix>.gen (the greedy continuation of the whole prompt, one id per
// line, from the generation path). Both engines prefill with <prefill-chunk>; extra off-grid cuts
// come from NINFERNO_PROBE_PREFILL_CUTS=p1,p2,... (both engines' debug_prefill_cuts, the
// --debug-prefill-cuts option). max-new 0 skips generation, a negative value or
// NINFERNO_PROBE_NO_SCORE skips scoring. Causal scoring always prefills in 1024-token chunks.
// NINFERNO_PROBE_ACTIVATIONS=exact|fast selects the activation mode of both engines (default: the
// product default).
#include "ninferno/engine.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "usage: chunk_probe <artifact> <tokens.i32> <out> <chunk> <max-new> [kv]\n";
        return 2;
    }
    const std::string artifact = argv[1];
    const std::string out      = argv[3];
    const auto chunk           = static_cast<std::uint32_t>(std::atoi(argv[4]));
    const int max_new          = std::atoi(argv[5]);
    const std::string kv       = argc > 6 ? argv[6] : "bf16";
    std::vector<ninferno::TokenId> tokens;
    {
        std::ifstream in(argv[2], std::ios::binary);
        std::int32_t v = 0;
        while (in.read(reinterpret_cast<char*>(&v), sizeof(v))) { tokens.push_back(v); }
    }
    if (tokens.size() < 2) {
        std::cerr << "too few tokens\n";
        return 2;
    }
    const ninferno::KvCacheStorage storage =
        kv == "fp8" ? ninferno::KvCacheStorage::Fp8E4M3Row256 : ninferno::KvCacheStorage::BFloat16;
    const auto T = static_cast<std::uint32_t>(tokens.size());
    ninferno::ActivationMode activations = ninferno::kDefaultActivationMode;
    if (const char* mode = std::getenv("NINFERNO_PROBE_ACTIVATIONS")) {
        const std::string m = mode;
        if (m != "exact" && m != "fast") {
            std::cerr << "NINFERNO_PROBE_ACTIVATIONS must be exact or fast\n";
            return 2;
        }
        activations =
            m == "exact" ? ninferno::ActivationMode::Exact : ninferno::ActivationMode::Fast;
    }
    std::vector<std::uint64_t> cuts;
    if (const char* text = std::getenv("NINFERNO_PROBE_PREFILL_CUTS")) {
        std::istringstream list(text);
        for (std::string item; std::getline(list, item, ',');) {
            if (!item.empty()) { cuts.push_back(std::stoull(item)); }
        }
    }
    try {
        if (max_new >= 0 && std::getenv("NINFERNO_PROBE_NO_SCORE") == nullptr) {
            ninferno::EngineOptions options;
            options.artifact_path = artifact;
            options.purpose       = ninferno::EnginePurpose::CausalScoring;
            options.max_context   = (T + 127U) / 128U * 128U;
            options.kv_cache      = storage;
            options.prefill_chunk = chunk;
            options.activations   = activations;
            options.maintainer.debug_prefill_cuts = cuts;
            ninferno::Engine engine(options);
            if (engine.options().prefill_chunk != chunk) {
                std::cerr << "score engine chunk normalized to " << engine.options().prefill_chunk
                          << "\n";
            }
            const std::vector<float> lp = engine.score_tokens(tokens, 1);
            std::ofstream f(out + ".lp", std::ios::binary);
            f.write(reinterpret_cast<const char*>(lp.data()),
                    static_cast<std::streamsize>(lp.size() * sizeof(float)));
            std::cout << "scored " << lp.size() << " chunk " << engine.options().prefill_chunk
                      << "\n";
        }
        if (max_new > 0) {
            ninferno::EngineOptions options;
            options.artifact_path = artifact;
            options.purpose       = ninferno::EnginePurpose::Generation;
            options.max_context   = (T + static_cast<std::uint32_t>(max_new) + 255U) / 128U * 128U;
            options.kv_capacity =
                ninferno::KvCapacityPolicy::explicit_capacity(options.max_context);
            options.kv_cache                = storage;
            options.prefill_chunk           = chunk;
            options.context_cache.enabled   = false;
            options.activations             = activations;
            options.maintainer.debug_prefill_cuts = cuts;
            ninferno::Engine engine(options);
            ninferno::RequestOptions request;
            request.execution.sampling.temperature  = 0.0F;
            request.execution.requested_output_tokens = static_cast<std::uint32_t>(max_new);
            request.execution.allow_prefix_reuse    = false;
            request.stop.include_model_defaults     = false;
            ninferno::GenerationResult result =
                engine.generate(engine.prepare_tokens(tokens, false), request);
            std::ofstream f(out + ".gen");
            for (const ninferno::TokenId id : result.generated_token_ids) { f << id << "\n"; }
            std::cout << "generated " << result.generated_token_ids.size() << " chunk "
                      << engine.options().prefill_chunk << "\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "chunk_probe: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
