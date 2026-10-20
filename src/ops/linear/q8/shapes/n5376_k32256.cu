#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"

// z-lab DFlash drafter for Gemma 4, [5376,32256]. Measured on the RTX 5090 with interleaved
// cold-weight timing and a null arm. Through T=64 each capacity bucket takes the launcher with
// the smallest worst ratio to the per-T best over its measured T; above, each band takes the
// winner at its upper edge, merged with the band below while that stays within 6%.
namespace ninferno::ops::detail {
namespace {
using Geometry = Q8N5376K32256;
using Access   = Q8ScaleAccess;
using Stage    = Q8ActivationStage;
} // namespace

Q8Launch select_q8_n5376_k32256(std::int32_t tokens) {
    if (tokens <= 1) return launch_q8_a16_gemv_shape<Geometry, 1, Q8A16GemvTiling<1, 4, 8, 4, 4>>;
    if (tokens <= 2)
        return launch_q8_a16_gemv_shape<Geometry, 2, Q8A16GemvTiling<1, 8, 16, 2, 2>>;
    if (tokens <= 4)
        return launch_q8_a16_sliced<
            Geometry, 4,
            Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Shared, Cache::cg, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 8)
        return launch_q8_a16_sliced<
            Geometry, 8,
            Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Shared, Cache::cg, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 16)
        return launch_q8_a16_sliced<
            Geometry, 16,
            Q8A16SlicedKMmaSchedule<16, 8, 1, 2, Access::Shared, Cache::cg, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 24)
        return launch_q8_a16_sliced<
            Geometry, 24,
            Q8A16SlicedKMmaSchedule<24, 8, 1, 2, Access::Shared, Cache::cg, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 32)
        return launch_q8_a16_sliced<
            Geometry, 32,
            Q8A16SlicedKMmaSchedule<32, 8, 1, 2, Access::Shared, Cache::cg, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 48)
        return launch_q8_a16_sliced<
            Geometry, 48,
            Q8A16SlicedKMmaSchedule<48, 4, 1, 2, Access::Shared, Cache::cg, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 64)
        return launch_q8_a16_sliced<
            Geometry, 64,
            Q8A16SlicedKMmaSchedule<64, 4, 1, 2, Access::Shared, Cache::cg, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 96) return launch_q8_a16_mma_r32_t96;
    if (tokens <= 128) return launch_q8_a16_mma_r32_t128;
    if (tokens <= 192) return launch_q8_a16_mma_r48_t64;
    if (tokens <= 256) return launch_q8_a16_mma_r64_t128;
    if (tokens <= 384) return launch_q8_a16_mma_r64_t96;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace ninferno::ops::detail
