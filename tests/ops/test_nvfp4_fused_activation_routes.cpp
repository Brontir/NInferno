// Activation routes of the fused NVFP4 plans on the Qwen3.5-27B shapes: the attention and GDN
// input projections (with the GDN conv snapshot schedules), LinearAdd and LinearSwiGLU.
//
// - The A16Only policy (ActivationMode::Bits16, ninferno/types.h) is served at every width, prefill
//   chunks included, by the token-invariant A16 routes (nvfp4_a16_exact.cuh): never a W4A4 route
//   and never a refusal. SwiGLU applies SiLU in that route's epilogue (no workspace); the GDN conv
//   takes the projection's BF16 plane at every width, then convolves.
// - A route probe (ops/common/route_probe.h) forcing one activation route reaches these plans too;
//   forcing A16 keeps the tuned A16 routes (fused through T=16, the Linear A16 projection above).
// Each plan sizes its workspace from the route it takes, so the capacity shows the route without a
// launch: W4A4 takes the quantized-activation planes, the fused A16 routes take none, and a
// materialized A16 projection takes its BF16 plane.
#include "ninferno/ops/attn_input_proj.h"
#include "ninferno/ops/gdn_input_proj.h"
#include "ninferno/ops/linear_add.h"
#include "ninferno/ops/linear_swiglu.h"
#include "ops/common/route_probe.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>

namespace {

using namespace ninferno;
using ninferno::ops::LinearPolicy;
using ninferno::ops::detail::RouteProbe;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "fused activation routes: " << what << '\n';
        ++failures;
    }
}

struct Sizes {
    std::size_t attn, gdn, add, swiglu, gdn_conv;
};

Sizes sizes(LinearPolicy policy, std::int32_t tokens) {
    return {ops::attn_input_proj_workspace_capacity_bytes(QType::NVFP4, 14336, 5120, policy, tokens,
                                                          tokens),
            ops::gdn_input_proj_workspace_capacity_bytes(QType::NVFP4, 16384, 5120, policy, tokens,
                                                         tokens),
            ops::linear_add_workspace_capacity_bytes(QType::NVFP4, 5120, 6144, policy, tokens,
                                                     tokens),
            ops::linear_swiglu_workspace_capacity_bytes(QType::NVFP4, 34816, 5120, policy, tokens,
                                                        tokens),
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(QType::NVFP4, 16384, 5120,
                                                                       policy, 1, tokens, tokens)};
}

// No quantized-activation plane anywhere. A16Only: no SwiGLU workspace, the GDN conv's projected
// BF16 plane at every width. A probe forcing A16: both take their projected planes past the fused
// A16 routes (T > 16).
void expect_a16(const Sizes& s, std::int32_t tokens, bool exact, const std::string& what) {
    const std::string at = what + " T=" + std::to_string(tokens);
    expect(s.attn == 0 && s.gdn == 0 && s.add == 0, at + ": a projection takes W4A4 workspace");
    expect((s.swiglu > 0) == (!exact && tokens > 16), at + ": SwiGLU is not the expected route");
    expect((s.gdn_conv > 0) == (exact || tokens > 16), at + ": GDN conv is not the expected route");
    if (exact) {
        // The projected plane alone: [10240, T] BF16, 256-byte aligned.
        const std::size_t bytes = std::size_t{10240} * static_cast<std::size_t>(tokens) * 2;
        const std::size_t plane = (bytes + 255) / 256 * 256;
        expect(s.gdn_conv <= plane + 256, at + ": GDN conv takes W4A4 planes");
    }
}

} // namespace

int main() {
    try {
        // A16Only at every width: decode, verify and prefill chunks.
        for (const std::int32_t tokens : {1, 4, 12, 16, 17, 32, 128, 1024, 4096}) {
            expect_a16(sizes(LinearPolicy::A16Only, tokens), tokens, true, "A16Only");
        }

        // The materialized GDN conv projects under the call's policy: A16Only takes the projected
        // plane alone, never the W4A4 projection's planes on top of it.
        for (const std::int32_t tokens : {17, 32, 128, 1024}) {
            expect(sizes(LinearPolicy::A16Only, tokens).gdn_conv <
                       sizes(LinearPolicy::AllowA4, tokens).gdn_conv,
                   "A16Only T=" + std::to_string(tokens) + ": GDN conv projects in W4A4");
        }

        // AllowA4 (the Fast mode's recorded policy): A16 below each crossover, W4A4 above it; the
        // GDN projection is always A4.
        const Sizes t1 = sizes(LinearPolicy::AllowA4, 1);
        expect(t1.attn == 0 && t1.add == 0 && t1.swiglu == 0, "AllowA4 T=1 is A16");
        expect(t1.gdn > 0, "AllowA4 T=1 GDN projection is W4A4");
        // LinearAdd [5120,6144] keeps A16 through T=16; every plan is W4A4 from T=17.
        const Sizes t12 = sizes(LinearPolicy::AllowA4, 12);
        expect(t12.attn > 0 && t12.gdn > 0 && t12.add == 0 && t12.swiglu > 0,
               "AllowA4 T=12 is W4A4 but for LinearAdd");
        const Sizes t17 = sizes(LinearPolicy::AllowA4, 17);
        expect(t17.attn > 0 && t17.gdn > 0 && t17.add > 0 && t17.swiglu > 0,
               "AllowA4 T=17 is W4A4");

        {
            const RouteProbe a16(0, 0);
            for (const std::int32_t tokens : {1, 4, 12, 16, 32, 128}) {
                expect_a16(sizes(LinearPolicy::AllowA4, tokens), tokens, false, "probe A16");
            }
        }
        {
            const RouteProbe a4(0, 1);
            for (const std::int32_t tokens : {1, 2, 4, 12}) {
                const Sizes s = sizes(LinearPolicy::AllowA4, tokens);
                expect(s.attn > 0 && s.gdn > 0 && s.add > 0 && s.swiglu > 0 && s.gdn_conv > 0,
                       "probe A4 T=" + std::to_string(tokens) + " is not W4A4 everywhere");
            }
        }
        const Sizes back = sizes(LinearPolicy::AllowA4, 1);
        expect(back.attn == 0 && back.gdn == t1.gdn && back.swiglu == 0,
               "the tables after the probe");
    } catch (const std::exception& error) {
        expect(false, std::string("refused: ") + error.what());
    }

    if (failures != 0) { return 1; }
    std::cout << "fused activation routes: ok\n";
    return 0;
}
