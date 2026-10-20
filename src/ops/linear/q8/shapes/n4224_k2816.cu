#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"

// Gemma 4 [4224,2816]. Bands measured on the RTX 5090 with interleaved cold-weight timing over
// T in {1..4096}; each band takes the winner at its upper edge.
//
// Above T=16 (the widest decode or verify call) every band is an MMA route, which all give a token
// the same bytes. The K-split routes sum K in another order: they once also took T=17..64, so a
// prefill chunk of that size gave its tokens other bytes than a wider chunk would, and a prompt
// decoded differently depending on where it was cut.
namespace ninferno::ops::detail {
namespace {
using Geometry = Q8N4224K2816;
using Access   = Q8ScaleAccess;
using Stage    = Q8ActivationStage;
} // namespace

Q8Launch select_q8_n4224_k2816(std::int32_t tokens) {
    if (tokens <= 4)
        return launch_q8_a16_sliced<
            Geometry, 4,
            Q8A16SlicedKMmaSchedule<8, 4, 1, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>>;
    if (tokens <= 5)
        return launch_q8_a16_sliced<
            Geometry, 8,
            Q8A16SlicedKMmaSchedule<8, 4, 1, 4, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 8)
        return launch_q8_a16_sliced<
            Geometry, 8,
            Q8A16SlicedKMmaSchedule<8, 4, 1, 2, Access::Direct, Cache::ca, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 16)
        return launch_q8_a16_sliced<
            Geometry, 16,
            Q8A16SlicedKMmaSchedule<16, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 160) return launch_q8_a16_mma_r32_t64;
    if (tokens <= 320) return launch_q8_a16_mma_r48_t64;
    if (tokens <= 384) return launch_q8_a16_mma_r64_t96;
    if (tokens <= 512) return launch_q8_a16_mma_r64_t128;
    if (tokens <= 768) return launch_q8_a16_mma_r128_t80;
    if (tokens <= 1024) return launch_q8_a16_mma_r96_t96;
    if (tokens <= 1536) return launch_q8_a16_mma_r64_t128;
    if (tokens <= 2048) return launch_q8_a16_mma_r96_t96;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace ninferno::ops::detail
