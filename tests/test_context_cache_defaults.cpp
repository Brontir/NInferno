#include "runtime/engine/family_registry.h"
#include "runtime/engine/model_instance.h"

#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    using ninferno::EngineOptions;
    using ninferno::kMaximumPreparedPromptCacheCandidatesPerRequest;
    using ninferno::runtime::normalize_engine_options;

    int failures = 0;

    // A single request can produce up to kMaximumPreparedPromptCacheCandidatesPerRequest distinct
    // shared-prefix candidates (frontend.cpp's opportunities.reserve(7U): four explicit markers
    // plus the engine's tool/leading-instruction/full-prompt automatic candidates). The default
    // Engine-wide shared catalog must be able to hold at least one request's own candidates even
    // at the smallest concurrency, or ordinary DefaultAutomatic-evidence traffic starts losing
    // cache hits to its own prior turns as soon as the catalog fills.
    for (const std::uint32_t concurrency : {1U, 2U, 8U}) {
        EngineOptions options;
        options.max_concurrency                     = concurrency;
        const EngineOptions normalized              = normalize_engine_options(options);
        const std::uint32_t default_shared_prefixes = *normalized.context_cache.max_shared_prefixes;
        failures +=
            check(default_shared_prefixes >=
                      static_cast<std::uint32_t>(kMaximumPreparedPromptCacheCandidatesPerRequest),
                  "default shared-prefix catalog capacity is smaller than one request's own "
                  "candidate ceiling");
        failures +=
            check(default_shared_prefixes >= concurrency,
                  "default shared-prefix catalog capacity did not cover active concurrency");
    }

    // An explicit override is still respected verbatim, including a deliberately small value.
    {
        EngineOptions options;
        options.max_concurrency                   = 1;
        options.context_cache.max_shared_prefixes = 1;
        const EngineOptions normalized            = normalize_engine_options(options);
        failures += check(*normalized.context_cache.max_shared_prefixes == 1,
                          "explicit max_shared_prefixes override was not preserved");
    }

    // A disabled context cache still normalizes to a root-only zero capacity.
    {
        EngineOptions options;
        options.max_concurrency        = 1;
        options.context_cache.enabled  = false;
        const EngineOptions normalized = normalize_engine_options(options);
        failures +=
            check(*normalized.context_cache.max_shared_prefixes == 0,
                  "disabled context cache did not normalize shared-prefix capacity to zero");
    }

    // An automatic KV storage is the family's: INT8 where the family runs it (the Qwen 3.5 line),
    // FP8 rows in the dense decoders and Gemma 4, BF16 for the families that keep BF16 keys and
    // values.
    {
        using ninferno::KvCacheStorage;
        using ninferno::runtime::automatic_kv_cache;
        failures += check(automatic_kv_cache("qwen3_5_text") == KvCacheStorage::Int8Group64 &&
                              automatic_kv_cache("qwen3_5_moe_text") == KvCacheStorage::Int8Group64,
                          "the Qwen 3.5 families do not take INT8 keys and values automatically");
        failures +=
            check(automatic_kv_cache("dense_decoder") == KvCacheStorage::Fp8E4M3Row256 &&
                      automatic_kv_cache("moe_decoder") == KvCacheStorage::Fp8E4M3Row256 &&
                      automatic_kv_cache("gemma4_text") == KvCacheStorage::Fp8E4M3Row256,
                  "the dense decoders and Gemma 4 do not take FP8 keys and values automatically");
        for (const std::string_view type : {"nemotron_h"}) {
            failures += check(automatic_kv_cache(type) == KvCacheStorage::BFloat16,
                              "a BF16-only family takes a non-BF16 automatic KV storage");
        }
        // Below sm_89 (no E4M3 conversion in hardware) FP8 rows fall back to BF16; INT8 does not.
        failures += check(automatic_kv_cache("dense_decoder", 86) == KvCacheStorage::BFloat16 &&
                              automatic_kv_cache("moe_decoder", 75) == KvCacheStorage::BFloat16 &&
                              automatic_kv_cache("moe_decoder", 70) == KvCacheStorage::BFloat16,
                          "FP8 rows are automatic on a GPU without FP8 conversion");
        failures +=
            check(automatic_kv_cache("dense_decoder", 89) == KvCacheStorage::Fp8E4M3Row256 &&
                      automatic_kv_cache("moe_decoder", 100) == KvCacheStorage::Fp8E4M3Row256 &&
                      automatic_kv_cache("dense_decoder", 120) == KvCacheStorage::Fp8E4M3Row256,
                  "FP8 rows are not automatic on a GPU with FP8 conversion");
        failures += check(automatic_kv_cache("qwen3_5_text", 75) == KvCacheStorage::Int8Group64,
                          "INT8 keys and values depend on the GPU's FP8 conversion");
        // Explicit storage is kept, and without the flag the artifact is never read.
        EngineOptions options;
        options.kv_cache = KvCacheStorage::Fp8E4M3Row256;
        failures +=
            check(normalize_engine_options(options).kv_cache == KvCacheStorage::Fp8E4M3Row256,
                  "an explicit KV storage was not preserved");
    }

    {
        // --activations exact promises bytes that do not depend on call boundaries, where a BF16
        // recurrent state rounds; the fast profile keeps the opt-in.
        EngineOptions options;
        options.artifact_path   = "model.ninferno";
        options.max_context     = 4096;
        options.recurrent_state = ninferno::RecurrentStateStorage::BF16;
        const auto accepted     = [&](ninferno::ActivationMode activations) {
            options.activations = activations;
            try {
                ninferno::runtime::validate_engine_options(options);
                return true;
            } catch (const std::invalid_argument&) { return false; }
        };
        failures += check(!accepted(ninferno::ActivationMode::Bits16),
                          "--activations exact accepted a BF16 recurrent state");
        failures += check(accepted(ninferno::ActivationMode::Auto),
                          "the fast profile refused a BF16 recurrent state");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
