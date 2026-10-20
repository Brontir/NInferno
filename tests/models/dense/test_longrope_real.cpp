// LongRoPE through the Engine (ninferno-serve's path: the dense Program) against ninferno-dense's
// Session, on a model whose rope has two modes (Phi-4-mini, Phi-3.5-mini: RopeGroup::long_mode).
// NINFERNO_TEST_DENSE_LONGROPE_ARTIFACT names the artifact, NINFERNO_TEST_DENSE_LONGROPE_TOKENS a
// token file (little-endian int32 ids of real text in that model's vocabulary, at least long_after
// + 1024 of them); without them the test skips.
//
// The session's context fixes the mode (a context past long_after runs the long one). For each
// mode, the same prompt and the same context through both paths give the same greedy tokens:
//   - long:  context 2 * long_after, a prompt of long_after + 1000 tokens (past long_after);
//   - short: context long_after, a prompt of long_after - 1000 tokens.
// Both decode eagerly, each step under the attention envelope of its own frontier (the Engine
// without CUDA graphs, the Session with eager_decode: ninferno-serve --no-cuda-graph against
// ninferno-dense --eager-decode), so the tokens are compared exactly. And the two modes differ on
// the short prompt (the Session's log-probabilities under both contexts), so a path that ran the
// wrong mode would not go unseen.

#include "core/device.h"
#include "models/dense/model.h"
#include "models/dense/session.h"
#include "ninferno/engine.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninferno::models::dense;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

std::vector<std::int32_t> read_tokens(const char* path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<std::int32_t> out;
    std::int32_t v = 0;
    while (file.read(reinterpret_cast<char*>(&v), 4)) { out.push_back(v); }
    return out;
}

constexpr std::int32_t kDecode = 48;
constexpr std::int32_t kChunk  = 2048;

struct Case {
    const char* mode;
    std::int32_t context;
    std::vector<std::int32_t> prompt;
    std::vector<std::int32_t> session_tokens;
};

std::vector<std::int32_t> engine_tokens(const char* artifact, const Case& c) {
    ninferno::EngineOptions options;
    options.artifact_path   = artifact;
    options.max_context     = static_cast<std::uint32_t>(c.context);
    options.kv_capacity     = ninferno::KvCapacityPolicy::explicit_capacity(options.max_context);
    options.prefill_chunk   = kChunk;
    options.max_concurrency = 1;
    options.use_cuda_graph  = false;
    options.context_cache.host_state_slots       = 1;
    options.context_cache.host_kv_capacity_bytes = 64ULL << 20;
    ninferno::Engine engine(options);
    const std::vector<ninferno::TokenId> ids(c.prompt.begin(), c.prompt.end());
    ninferno::RequestOptions request;
    request.execution.requested_output_tokens = kDecode;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    const ninferno::GenerationResult result =
        engine.generate(engine.prepare_tokens(ids, false), request);
    return {result.generated_token_ids.begin(), result.generated_token_ids.end()};
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFERNO_TEST_DENSE_LONGROPE_ARTIFACT");
    const char* tokens   = std::getenv("NINFERNO_TEST_DENSE_LONGROPE_TOKENS");
    if (artifact == nullptr || *artifact == 0 || tokens == nullptr || *tokens == 0) {
        std::cout << "SKIP: set NINFERNO_TEST_DENSE_LONGROPE_ARTIFACT and "
                     "NINFERNO_TEST_DENSE_LONGROPE_TOKENS\n";
        return 77;
    }
    try {
        const std::vector<std::int32_t> all = read_tokens(tokens);
        std::vector<Case> cases;
        {
            ninferno::DeviceContext device(0);
            auto model = Model::load(artifact, device, ActivationProfile::Artifact);
            std::int32_t long_after = 0;
            for (const RopeGroup& group : model->config().rope_groups) {
                if (!group.long_inv_freq.empty()) { long_after = group.long_after; }
            }
            if (long_after <= 1024) {
                std::cerr << "FAIL: the artifact has no LongRoPE group past 1024 positions\n";
                return 1;
            }
            if (all.size() < static_cast<std::size_t>(long_after) + 1024) {
                std::cerr << "FAIL: the token file holds " << all.size() << " ids, fewer than "
                          << long_after + 1024 << '\n';
                return 1;
            }
            const auto prefix = [&](std::int32_t n) {
                return std::vector<std::int32_t>(all.begin(), all.begin() + n);
            };
            cases.push_back({"long", 2 * long_after, prefix(long_after + 1000), {}});
            cases.push_back({"short", long_after, prefix(long_after - 1000), {}});
            for (Case& c : cases) {
                SessionOptions options;
                options.max_context   = c.context;
                options.prefill_chunk = kChunk;
                options.eager_decode  = true;
                Session session(*model, device, options);
                session.prefill(c.prompt);
                c.session_tokens = session.decode(kDecode);
            }
            // The modes differ: the short prompt's log-probabilities under a long context.
            std::vector<float> scores[2];
            const std::vector<std::int32_t> scored = prefix(1024);
            for (int i = 0; i < 2; ++i) {
                SessionOptions options;
                options.max_context   = cases[i].context;
                options.prefill_chunk = kChunk;
                Session session(*model, device, options);
                scores[i] = session.score(scored);
            }
            check(scores[0].size() == scores[1].size() &&
                      std::memcmp(scores[0].data(), scores[1].data(), scores[0].size() * 4) != 0,
                  "the long and the short mode score the same log-probabilities");
        }
        for (const Case& c : cases) {
            const std::vector<std::int32_t> served = engine_tokens(artifact, c);
            const std::size_t n = std::min(served.size(), c.session_tokens.size());
            std::size_t same    = 0;
            while (same < n && served[same] == c.session_tokens[same]) { ++same; }
            std::cout << c.mode << " mode: context " << c.context << ", prompt " << c.prompt.size()
                      << " tokens; Engine " << served.size() << " tokens, the first " << same
                      << " equal to the Session's\n";
            check(served.size() >= 8, std::string(c.mode) + ": the Engine stopped within 8 tokens");
            check(same == n,
                  std::string(c.mode) + ": the Engine's tokens differ from the Session's");
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "dense LongRoPE real: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
