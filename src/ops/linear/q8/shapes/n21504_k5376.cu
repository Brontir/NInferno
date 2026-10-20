#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"

// z-lab DFlash drafter for Gemma 4, [21504,5376]. Measured on the RTX 5090 with interleaved
// cold-weight timing and a null arm. Through T=64 each capacity bucket takes the launcher with
// the smallest worst ratio to the per-T best over its measured T; above, each band takes the
// winner at its upper edge, merged with the band below while that stays within 6%.
namespace ninferno::ops::detail {
namespace {
using Geometry = Q8N21504K5376;
using Access   = Q8ScaleAccess;
using Stage    = Q8ActivationStage;
} // namespace

Q8Launch select_q8_n21504_k5376(std::int32_t tokens) {
    if (tokens <= 4)
        return launch_q8_a16_sliced<
            Geometry, 4,
            Q8A16SlicedKMmaSchedule<8, 4, 1, 2, Access::Shared, Cache::cg, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 8)
        return launch_q8_a16_sliced<
            Geometry, 8,
            Q8A16SlicedKMmaSchedule<8, 4, 1, 2, Access::Shared, Cache::cg, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 16)
        return launch_q8_a16_sliced<
            Geometry, 16,
            Q8A16SlicedKMmaSchedule<16, 4, 1, 2, Access::Shared, Cache::cg, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 24)
        return launch_q8_a16_sliced<
            Geometry, 24,
            Q8A16SlicedKMmaSchedule<24, 4, 1, 2, Access::Shared, Cache::cg, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 48) return launch_q8_a16_mma_r32_t64;
    if (tokens <= 64) return launch_q8_a16_mma_r48_t64;
    if (tokens <= 96) return launch_q8_a16_mma_r64_t96;
    if (tokens <= 128) return launch_q8_a16_mma_r64_t128;
    if (tokens <= 192) return launch_q8_a16_mma_r96_t96;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace ninferno::ops::detail
