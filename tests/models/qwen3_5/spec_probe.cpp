// spec_probe: concurrent greedy generation, for speed and output comparisons (not a ctest).
//
//   spec_probe <artifact> <tokens.i32> <out-prefix> <concurrency> <none|mtp|dflash> <K>
//              <prompt-tokens> <max-new> <reps>
//
// Prompt i is tokens [i * prompt, (i + 1) * prompt) of the file. Each repetition submits all C
// requests at once and waits for them; it prints one line with the batch wall time, the output
// tokens, their aggregate rate and the speculative counters. The first repetition's generated ids
// go to <out-prefix>.gen<i>, one id per line. A warm-up batch runs first and is not reported.
#include "ninferno/engine.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 10) {
        std::cerr << "usage: spec_probe <artifact> <tokens.i32> <out> <C> <none|mtp|dflash> <K> "
                     "<prompt> <max-new> <reps>\n";
        return 2;
    }
    const std::string artifact = argv[1];
    const std::string out      = argv[3];
    const auto concurrency     = static_cast<std::uint32_t>(std::atoi(argv[4]));
    const std::string backend  = argv[5];
    const auto draft           = static_cast<std::uint32_t>(std::atoi(argv[6]));
    const auto prompt          = static_cast<std::uint32_t>(std::atoi(argv[7]));
    const auto max_new         = static_cast<std::uint32_t>(std::atoi(argv[8]));
    const int reps             = std::atoi(argv[9]);
    std::vector<ninferno::TokenId> corpus;
    {
        std::ifstream in(argv[2], std::ios::binary);
        std::int32_t v = 0;
        while (in.read(reinterpret_cast<char*>(&v), sizeof(v))) { corpus.push_back(v); }
    }
    if (corpus.size() < static_cast<std::size_t>(concurrency) * prompt) {
        std::cerr << "token file too short\n";
        return 2;
    }
    try {
        ninferno::EngineOptions options;
        options.artifact_path   = artifact;
        options.purpose         = ninferno::EnginePurpose::Generation;
        options.max_concurrency = concurrency;
        options.max_context     = (prompt + max_new + 16 + 255U) / 256U * 256U;
        options.kv_capacity =
            ninferno::KvCapacityPolicy::explicit_capacity(options.max_context * concurrency);
        options.kv_cache              = ninferno::KvCacheStorage::BFloat16;
        options.prefill_chunk         = 1024;
        options.context_cache.enabled = false;
        if (backend == "mtp") {
            options.speculative.backend = ninferno::SpeculativeBackend::Mtp;
        } else if (backend == "dflash") {
            options.speculative.backend = ninferno::SpeculativeBackend::DFlash;
        }
        options.speculative.draft_tokens = backend == "none" ? 0 : draft;
        ninferno::Engine engine(options);

        ninferno::RequestOptions request;
        request.execution.sampling.temperature    = 0.0F;
        request.execution.requested_output_tokens = max_new;
        request.execution.allow_prefix_reuse      = false;
        request.stop.include_model_defaults       = false;

        const auto batch = [&](int rep) {
            std::vector<ninferno::GenerationHandle> handles;
            const auto begin = std::chrono::steady_clock::now();
            for (std::uint32_t i = 0; i < concurrency; ++i) {
                std::vector<ninferno::TokenId> ids(
                    corpus.begin() + static_cast<std::ptrdiff_t>(i) * prompt,
                    corpus.begin() + static_cast<std::ptrdiff_t>(i + 1) * prompt);
                handles.push_back(
                    engine.submit(engine.prepare_tokens(std::move(ids), false), request));
            }
            std::uint64_t tokens = 0, rounds = 0, drafted = 0, accepted = 0;
            double decode = 0.0;
            for (std::uint32_t i = 0; i < concurrency; ++i) {
                ninferno::GenerationResult result = handles[i].wait();
                tokens += result.generated_token_ids.size();
                rounds += result.speculative.rounds;
                drafted += result.speculative.drafted_tokens;
                accepted += result.speculative.accepted_tokens;
                decode += result.timings.decode_seconds;
                if (rep == 0) {
                    std::ofstream f(out + ".gen" + std::to_string(i));
                    for (const ninferno::TokenId id : result.generated_token_ids) {
                        f << id << "\n";
                    }
                }
            }
            const double wall =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            if (rep >= 0) {
                std::cout << "rep " << rep << " wall_s " << wall << " tokens " << tokens
                          << " tok_s " << static_cast<double>(tokens) / wall << " rounds " << rounds
                          << " drafted " << drafted << " accepted " << accepted << " mean_decode_s "
                          << decode / concurrency << std::endl;
            }
        };
        batch(-1);
        for (int rep = 0; rep < reps; ++rep) { batch(rep); }
    } catch (const std::exception& error) {
        std::cerr << "spec_probe: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
