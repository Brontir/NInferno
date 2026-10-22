// --activations exact through the Engine (ninferno-serve) equals ninferno-gemma4 --activations
// exact (the Session under ActivationProfile::A16Full and the invariant route profile) byte for
// byte: the same prompt ids, context and prefill chunk give the same greedy tokens
// (engine_exact_real.h). With FP8 keys and values on every layer (--kv-dtype fp8 --sliding-kv-dtype
// fp8: the full pool and the sliding rings of the StateImage) the Engine equals the Session's
// --kv-dtype fp8, which keeps every layer in FP8, byte for byte as well.
// NINFERNO_TEST_GEMMA4_EXACT_ARTIFACT names the artifact; without it the test skips.

#include "../engine_exact_real.h"

#include "core/device.h"
#include "ninferno/ops/prefill_routes.h"
#include "models/gemma4/model.h"
#include "models/gemma4/session.h"

#include <cstdlib>
#include <exception>
#include <iostream>

int main() {
    using namespace ninferno::test::engine_exact;
    const char* artifact = std::getenv("NINFERNO_TEST_GEMMA4_EXACT_ARTIFACT");
    if (artifact == nullptr || *artifact == 0) {
        std::cout << "SKIP: set NINFERNO_TEST_GEMMA4_EXACT_ARTIFACT\n";
        return 77;
    }
    try {
        const Run exact = engine_run(artifact, ninferno::ActivationMode::Bits16);
        const Run fast  = engine_run(artifact, ninferno::ActivationMode::Auto);
        std::vector<std::int32_t> session_tokens;
        {
            ninferno::DeviceContext device(0);
            const ninferno::ops::ScopedInvariantRoutes invariant_routes(true); // as the CLI's exact
            using namespace ninferno::models::gemma4;
            auto model = Model::load(artifact, device, ActivationProfile::A16Full);
            SessionOptions options;
            options.max_context   = static_cast<std::int32_t>(kContext);
            options.prefill_chunk = static_cast<std::int32_t>(kChunk);
            Session session(*model, device, options);
            session.prefill(exact.prompt);
            session_tokens = session.decode(static_cast<std::int32_t>(kDecode));
        }
        compare("gemma4", exact, fast, session_tokens);
        const Run fp8 = engine_run(artifact, ninferno::ActivationMode::Bits16,
                                   ninferno::KvCacheStorage::Fp8E4M3Row256,
                                   ninferno::KvCacheStorage::Fp8E4M3Row256);
        std::vector<std::int32_t> session_fp8;
        {
            ninferno::DeviceContext device(0);
            const ninferno::ops::ScopedInvariantRoutes invariant_routes(true);
            using namespace ninferno::models::gemma4;
            auto model = Model::load(artifact, device, ActivationProfile::A16Full);
            SessionOptions options;
            options.max_context   = static_cast<std::int32_t>(kContext);
            options.prefill_chunk = static_cast<std::int32_t>(kChunk);
            options.kv_fp8        = true;
            Session session(*model, device, options);
            session.prefill(fp8.prompt);
            session_fp8 = session.decode(static_cast<std::int32_t>(kDecode));
        }
        const std::size_t same_fp8 = common_prefix(fp8.tokens, session_fp8);
        std::cout << "gemma4 fp8 keys and values (rings too): exact Engine " << fp8.tokens.size()
                  << " tokens, the first " << same_fp8 << " equal to the Session's; BF16 Engine "
                  << "equal for its first " << common_prefix(exact.tokens, session_fp8) << '\n';
        check(same_fp8 == fp8.tokens.size() && fp8.tokens.size() >= 8,
              "gemma4: the FP8 exact Engine's tokens differ from the FP8 Session's");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "gemma4 engine exact real: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
