// --activations exact through the Engine (ninferno-serve) equals ninferno-dense --activations exact
// (the Session under ActivationProfile::A16 and the invariant route profile) byte for byte: the
// same prompt ids, context and prefill chunk give the same greedy tokens (engine_exact_real.h).
// NINFERNO_TEST_DENSE_EXACT_ARTIFACT names the artifact (a quantized one, so exact changes routes);
// without it the test skips.

#include "../engine_exact_real.h"

#include "core/device.h"
#include "ninferno/ops/prefill_routes.h"
#include "models/dense/model.h"
#include "models/dense/session.h"

#include <cstdlib>
#include <exception>
#include <iostream>

int main() {
    using namespace ninferno::test::engine_exact;
    const char* artifact = std::getenv("NINFERNO_TEST_DENSE_EXACT_ARTIFACT");
    if (artifact == nullptr || *artifact == 0) {
        std::cout << "SKIP: set NINFERNO_TEST_DENSE_EXACT_ARTIFACT\n";
        return 77;
    }
    try {
        const Run exact = engine_run(artifact, ninferno::ActivationMode::Bits16);
        const Run fast  = engine_run(artifact, ninferno::ActivationMode::Auto);
        std::vector<std::int32_t> session_tokens;
        {
            ninferno::DeviceContext device(0);
            const ninferno::ops::ScopedInvariantRoutes invariant_routes(true); // as the CLI's exact
            using namespace ninferno::models::dense;
            auto model = Model::load(artifact, device, ActivationProfile::A16);
            SessionOptions options;
            options.max_context   = static_cast<std::int32_t>(kContext);
            options.prefill_chunk = static_cast<std::int32_t>(kChunk);
            Session session(*model, device, options);
            session.prefill(exact.prompt);
            session_tokens = session.decode(static_cast<std::int32_t>(kDecode));
        }
        compare("dense", exact, fast, session_tokens);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "dense engine exact real: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
