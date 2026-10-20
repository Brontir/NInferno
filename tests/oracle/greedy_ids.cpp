// Greedy continuation of raw token ids through the public Engine, printed as ids: the quick
// cross-check of a family against a reference's greedy tokens (a hybrid's recurrent state, its
// prefill chunking and its batched decode all show in the ids).
//
//   ninferno_greedy_ids ARTIFACT IDS NEW [chunk=N] [concurrency=C] [context=N]
//
// IDS is a comma-separated list. With concurrency C, C identical requests run at once (a batched
// decode); every one must print the same ids.

#include "ninferno/engine.h"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc < 4) {
            std::cerr << "usage: ninferno_greedy_ids ARTIFACT IDS NEW [chunk=N] [concurrency=C] "
                         "[context=N]\n";
            return 2;
        }
        std::map<std::string, std::string> kv;
        for (int i = 4; i < argc; ++i) {
            const std::string a = argv[i];
            const auto eq       = a.find('=');
            if (eq != std::string::npos) { kv[a.substr(0, eq)] = a.substr(eq + 1); }
        }
        const auto get = [&](const char* key, std::uint32_t fallback) {
            const auto it = kv.find(key);
            return it == kv.end() ? fallback : static_cast<std::uint32_t>(std::stoul(it->second));
        };
        std::vector<ninferno::TokenId> ids;
        std::stringstream list(argv[2]);
        for (std::string item; std::getline(list, item, ',');) { ids.push_back(std::stoi(item)); }
        const auto fresh = static_cast<std::uint32_t>(std::stoul(argv[3]));
        ninferno::EngineOptions options;
        options.artifact_path   = argv[1];
        options.max_context     = get("context", 512);
        options.prefill_chunk   = get("chunk", 256);
        options.max_concurrency = get("concurrency", 1);
        options.kv_capacity     = ninferno::KvCapacityPolicy::explicit_capacity(
            options.max_context * options.max_concurrency);
        ninferno::Engine engine(options);
        ninferno::RequestOptions request;
        request.execution.sampling.temperature = 0.0F;
        request.execution.requested_output_tokens = fresh;
        request.execution.allow_prefix_reuse      = false;
        request.stop.include_model_defaults       = false;
        std::vector<ninferno::GenerationHandle> handles;
        for (std::uint32_t i = 0; i < options.max_concurrency; ++i) {
            handles.push_back(engine.submit(engine.prepare_tokens(ids, false), request));
        }
        for (auto& handle : handles) {
            const auto result = handle.wait();
            std::cout << "ids";
            for (const auto id : result.generated_token_ids) { std::cout << ' ' << id; }
            std::cout << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ninferno_greedy_ids: " << error.what() << '\n';
        return 1;
    }
}
