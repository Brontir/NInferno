#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"

// Gemma 4 [16384,5376]. Bands measured on the RTX 5090 with interleaved cold-weight timing over
// T in {1..4096}; each band takes the winner at its upper edge.
namespace ninferno::ops::detail {
namespace {
using Geometry = Q8N16384K5376;
using Access   = Q8ScaleAccess;
using Stage    = Q8ActivationStage;
} // namespace

Q8Launch select_q8_n16384_k5376(std::int32_t tokens) {
    if (tokens <= 4)
        return launch_q8_a16_sliced<
            Geometry, 4,
            Q8A16SlicedKMmaSchedule<8, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::PaddedZero>>;
    if (tokens <= 8)
        return launch_q8_a16_sliced<
            Geometry, 8,
            Q8A16SlicedKMmaSchedule<8, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::PaddedZero>>;
    if (tokens <= 16)
        return launch_q8_a16_sliced<
            Geometry, 16,
            Q8A16SlicedKMmaSchedule<16, 4, 1, 4, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 24)
        return launch_q8_a16_sliced<
            Geometry, 24,
            Q8A16SlicedKMmaSchedule<24, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 32)
        return launch_q8_a16_sliced<
            Geometry, 32,
            Q8A16SlicedKMmaSchedule<32, 4, 1, 4, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 48)
        return launch_q8_a16_sliced<
            Geometry, 48,
            Q8A16SlicedKMmaSchedule<48, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::PaddedZero>>;
    if (tokens <= 64) return launch_q8_a16_mma_r32_t64;
    if (tokens <= 96) return launch_q8_a16_mma_r64_t96;
    if (tokens <= 128) return launch_q8_a16_mma_r64_t128;
    if (tokens <= 160) return launch_q8_a16_mma_r128_t80;
    if (tokens <= 192) return launch_q8_a16_mma_r64_t96;
    if (tokens <= 256) return launch_q8_a16_mma_r64_t128;
    return launch_q8_a16_mma_r64x32_t128;
}

} // namespace ninferno::ops::detail
