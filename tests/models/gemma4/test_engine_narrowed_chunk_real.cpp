// A separate-drafter family (Gemma 4 with its MTP assistant) whose automatic prefill chunk the fit
// narrows starts and speculates. A startup check once planned the loaded model again without its
// drafter to say whether the drafter caused the narrowing; Gemma 4 refuses that plan while its
// assistant is loaded (validate_options throws invalid_argument) and the engine did not start
// whenever the chunk narrowed (Gemma 4 31B + MTP on a 16 GB card). The check is gone; this test
// keeps the path that narrows the chunk of a speculating separate-drafter engine running.
//
// The narrowing is made deterministic by the free memory the Engine sees: cudaMemGetInfo is wrapped
// (--wrap) to report `g_ballast` bytes less. Two engines calibrate it: one with the automatic chunk
// (1024 when the device fits it) and one at chunk 128, each giving the bytes its runtime took
// after the weights. The engine under test sees free memory between the two (counted from the
// later measurement, the narrow engine's free memory after its weights), so chunk 1024 does not
// fit and a narrower one does; the explicit KV capacity keeps the pool from shrinking instead.
// NINFERNO_TEST_NARROW_ARTIFACT names the Gemma 4 target, NINFERNO_TEST_NARROW_DRAFTER its MTP
// assistant; without both the test skips.

#include "ninferno/engine.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::atomic<std::size_t> g_ballast{0};

} // namespace

extern "C" {
cudaError_t CUDARTAPI __real_cudaMemGetInfo(std::size_t*, std::size_t*);

cudaError_t CUDARTAPI __wrap_cudaMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes) {
    const cudaError_t status = __real_cudaMemGetInfo(free_bytes, total_bytes);
    if (status == cudaSuccess && free_bytes != nullptr) {
        const std::size_t ballast = g_ballast.load();
        *free_bytes               = *free_bytes > ballast ? *free_bytes - ballast : 0;
    }
    return status;
}
}

namespace {

constexpr std::uint32_t kContext     = 8192;
constexpr std::uint32_t kConcurrency = 2;
constexpr std::uint32_t kWidest      = 1024;
constexpr std::uint32_t kNarrowest   = 128;

// A prose request in the Gemma 4 chat template, thinking off.
const std::vector<ninferno::TokenId> kPrompt = {
    2,      105,    2364,   107,    6974,   496,    236743, 236812, 236771, 236771, 236772, 3017,
    2822,   3925,   1003,   496,    63937,  64222,  1015,   15249,  496,    3618,   528,    496,
    11988,  236761, 106,    107,    105,    4368,   107,    100,    45518,  107,    101};

ninferno::EngineOptions options_for(const char* artifact, const char* drafter,
                                  std::optional<std::uint32_t> chunk) {
    ninferno::EngineOptions options;
    options.artifact_path   = artifact;
    options.max_context     = kContext;
    options.max_concurrency = kConcurrency;
    // Explicit: every slot's full context, so the fit narrows the chunk rather than the pool.
    options.kv_capacity = ninferno::KvCapacityPolicy::explicit_capacity(kContext * kConcurrency);

    options.prefill_chunk           = chunk.value_or(kWidest);
    options.prefill_chunk_automatic = !chunk.has_value();

    options.speculative.backend      = ninferno::SpeculativeBackend::Mtp;
    options.speculative.drafter      = drafter;
    options.speculative.draft_tokens = 4;

    options.context_cache.enabled                = false;
    options.context_cache.host_state_slots       = 0;
    options.context_cache.host_kv_capacity_bytes = 0;
    return options;
}

// The bytes the runtime took after the weights, and the chunk it chose.
struct Sized {
    std::size_t runtime_bytes      = 0;
    std::uint32_t chunk            = 0;
    std::size_t free_after_weights = 0;
};

Sized sized(const ninferno::EngineOptions& options) {
    const ninferno::Engine engine(options);
    const ninferno::MemorySummary memory = engine.memory_summary();
    return {.runtime_bytes      = memory.available_after_weights_bytes -
                                  memory.available_after_startup_bytes,
            .chunk              = engine.options().prefill_chunk,
            .free_after_weights = memory.available_after_weights_bytes};
}

ninferno::RequestOptions greedy(std::uint32_t outputs) {
    ninferno::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.allow_prefix_reuse      = false;
    options.execution.sampling.temperature    = 0.0F;
    options.stop.include_model_defaults       = false;
    return options;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFERNO_TEST_NARROW_ARTIFACT");
    const char* drafter  = std::getenv("NINFERNO_TEST_NARROW_DRAFTER");
    if (artifact == nullptr || *artifact == 0 || drafter == nullptr || *drafter == 0) {
        std::cout << "SKIP: set NINFERNO_TEST_NARROW_ARTIFACT and NINFERNO_TEST_NARROW_DRAFTER\n";
        return 77;
    }
    try {
        const Sized wide   = sized(options_for(artifact, drafter, std::nullopt));
        const Sized narrow = sized(options_for(artifact, drafter, kNarrowest));
        std::cout << "calibration: chunk " << wide.chunk << " takes " << wide.runtime_bytes
                  << " B, chunk " << kNarrowest << " takes " << narrow.runtime_bytes
                  << " B, free after weights " << narrow.free_after_weights << " B\n";
        if (wide.chunk != kWidest) {
            std::cout << "SKIP: the device does not fit chunk " << kWidest
                      << " unconstrained (chose " << wide.chunk << ")\n";
            return 77;
        }
        if (wide.runtime_bytes < narrow.runtime_bytes + (64ULL << 20)) {
            std::cout << "SKIP: chunk " << kWidest << " takes less than 64 MiB more than "
                      << kNarrowest << "; no margin to narrow it deterministically\n";
            return 77;
        }
        // The free memory seen after the weights, between the two: tried at the midpoint, then
        // nearer each end, until the fit narrows the chunk. A refusal (CapacityError) moves on;
        // anything else is the failure this test is for.
        for (const double at : {0.5, 0.3, 0.7, 0.15, 0.85}) {
            const auto seen = static_cast<std::size_t>(
                static_cast<double>(narrow.runtime_bytes) +
                at * static_cast<double>(wide.runtime_bytes - narrow.runtime_bytes));
            g_ballast = narrow.free_after_weights - seen;
            try {
                ninferno::Engine engine(options_for(artifact, drafter, std::nullopt));
                const std::uint32_t chunk = engine.options().prefill_chunk;
                std::cout << "seen " << seen << " B free after weights: chunk " << chunk << '\n';
                if (chunk >= kWidest) { continue; }
                const ninferno::GenerationResult result =
                    engine.generate(engine.prepare_tokens(kPrompt), greedy(32));
                if (result.generated_token_ids.size() != 32 || result.speculative.rounds == 0) {
                    std::cerr << "FAIL: the narrowed engine did not speculate 32 tokens ("
                              << result.generated_token_ids.size() << " tokens, "
                              << result.speculative.rounds << " rounds)\n";
                    return 1;
                }
                std::cout << "ok: chunk narrowed to " << chunk << ", "
                          << result.speculative.rounds << " rounds\n";
                return 0;
            } catch (const ninferno::CapacityError& refused) {
                std::cout << "seen " << seen << " B: refused (" << refused.what() << ")\n";
            }
        }
        std::cerr << "FAIL: no free memory between the two plans narrowed the chunk\n";
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
