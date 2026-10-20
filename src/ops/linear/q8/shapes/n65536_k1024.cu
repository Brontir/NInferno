#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"

// Gemma 4 MTP drafter [65536,1024]. Decode and the verify batch (T <= 16) take the launcher with
// the smallest worst-case ratio to the per-T best over each capacity bucket, from interleaved
// cold-weight timing with a null arm; T > 16 is not used by the drafter and takes the tiled MMA.
namespace ninferno::ops::detail {
namespace {
using Geometry = Q8N65536K1024;
using Access   = Q8ScaleAccess;
using Stage    = Q8ActivationStage;
} // namespace

Q8Launch select_q8_n65536_k1024(std::int32_t tokens) {
    if (tokens <= 1)
        return launch_q8_a16_gemv_shape<Geometry, 1, Q8A16GemvTiling<1, 4, 16, 4, 4>>;
    if (tokens <= 2)
        return launch_q8_a16_sliced<
            Geometry, 4,
            Q8A16SlicedKMmaSchedule<8, 4, 1, 2, Access::Shared, Cache::cg, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 4)
        return launch_q8_a16_sliced<
            Geometry, 4,
            Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Shared, Cache::cg, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 6)
        return launch_q8_a16_sliced<
            Geometry, 8,
            Q8A16SlicedKMmaSchedule<8, 4, 1, 2, Access::Shared, Cache::cg, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 8)
        return launch_q8_a16_sliced<
            Geometry, 8,
            Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Shared, Cache::cg, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 16)
        return launch_q8_a16_sliced<
            Geometry, 16,
            Q8A16SlicedKMmaSchedule<16, 8, 1, 2, Access::Shared, Cache::cg, Cache::cg, Stage::RuntimeActive>>;
    return launch_q8_a16_mma_r32_t64;
}

} // namespace ninferno::ops::detail
