#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"

#include "core/device_class.h"
#include "ops/common/sm_wave_rules.h"

// Gemma 4 [5376,16384]. Bands measured on the RTX 5090 with interleaved cold-weight timing over
// T in {1..4096}; each band takes the winner at its upper edge.
namespace ninferno::ops::detail {
namespace {
using Geometry = Q8N5376K16384;
using Access   = Q8ScaleAccess;
using Stage    = Q8ActivationStage;

int sm_count() { return device_class().sms; }
} // namespace

Q8Launch select_q8_n5376_k16384(std::int32_t tokens) {
    if (tokens <= 4)
        return launch_q8_a16_sliced<
            Geometry, 4,
            Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 8)
        return launch_q8_a16_sliced<
            Geometry, 8,
            Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Direct, Cache::ca, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 16)
        return launch_q8_a16_sliced<
            Geometry, 16,
            Q8A16SlicedKMmaSchedule<16, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::PaddedZero>>;
    if (tokens <= 24)
        return launch_q8_a16_sliced<
            Geometry, 24,
            Q8A16SlicedKMmaSchedule<24, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 32)
        return launch_q8_a16_sliced<
            Geometry, 32,
            Q8A16SlicedKMmaSchedule<32, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 48)
        return launch_q8_a16_sliced<
            Geometry, 48,
            Q8A16SlicedKMmaSchedule<48, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 64)
        return launch_q8_a16_sliced<
            Geometry, 64,
            Q8A16SlicedKMmaSchedule<64, 4, 1, 2, Access::Direct, Cache::ca, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 96) return launch_q8_a16_mma_r32_t96;
    if (tokens <= 128) return launch_q8_a16_mma_r32_t128;
    if (tokens <= 192) return launch_q8_a16_mma_r48_t64;
    if (tokens <= 256) return launch_q8_a16_mma_r64_t128;
    if (tokens <= 384) return launch_q8_a16_mma_r64_t96;
    // The 256-token tile (128 rows, one CTA an SM: MmaR128x64T256, 92 KiB of shared memory, launch
    // bound one an SM) where its 42 row tiles by T/256 columns fill whole waves of the device's SMs
    // and its waves cost no more than the 128-token tile's (64 rows, two CTAs an SM):
    // sm_rules::fills_whole_waves and wide_tile_pays. On the RTX 5090's 170 SMs: T a multiple of
    // 1024; never on the 4090's 128 or the 5070 Ti's 70 (measured slower there).
    const int sms = sm_count();
    if (sm_rules::fills_whole_waves(tokens, sms, 5376 / 128, 256, 1) &&
        sm_rules::wide_tile_pays(tokens, sms, 5376, 128, 256, 64, 128))
        return launch_q8_a16_mma_r128x64_t256;
    return launch_q8_a16_mma_r64x32_t128;
}

} // namespace ninferno::ops::detail
